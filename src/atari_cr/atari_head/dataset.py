import os
import random
from typing import List
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import polars as pl

from atari_cr.models import EpisodeRecord
from atari_cr.module_overrides import tqdm
from atari_cr.atari_head.utils import (
    SCREEN_SIZE, VISUAL_DEGREES_PER_PIXEL, preprocess, save_video)

# Screen size from https://github.com/corgiTrax/Gaze-Data-Processor/blob/master/data_visualizer.py
DATASET_SCREEN_SIZE = [160, 210]


class GazeDataset(Dataset):
    """
    Dataset object for the Atari-HEAD dataset

    :param Tensor[N,W,H] frames: Greyscale game images
    :param Tensor[N,W,H] saliency: Saliency maps belonging to the frames
    :param Tensor[N] train_indcies: Indices of frame stacks used for training
    :param Tensor[N] val_indcies: Inidcies of frame stacks used for validation
    :param Tensor[N] durations: How long each frame was looked at
    """

    def __init__(
        self, frames: Tensor, saliency: Tensor, train_indices: Tensor,
        val_indices: Tensor, durations: Tensor, gazes: list[Tensor]
    ):
        self.frames = frames
        self.saliency = saliency
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.durations = durations
        self.gazes = gazes

    def split(self, batch_size=512):
        """
        Splits the dataset into one dataset containing train data and one
        dataset containing test data and returns them as loaders.

        :return: Train and validation loader
        """
        # Shuffle happens now, after the indices have been split
        # This is because subsequent images are highly correlated
        train_loader = DataLoader(
            self, batch_size, sampler=SubsetRandomSampler(self.train_indices)
        )
        val_loader = DataLoader(
            self, batch_size, sampler=SubsetRandomSampler(self.val_indices)
        )

        return train_loader, val_loader

    def to_loader(self, batch_size=512):
        """ Get the entire dataset as a single dataloader """
        return DataLoader(self, batch_size, sampler=SubsetRandomSampler(
                torch.cat([self.train_indices, self.val_indices])))

    def __len__(self):
        return len(self.train_indices) + len(self.val_indices)

    def __getitem__(self, idx: Tensor):
        """
        Loads the images from paths specified in self.data and creates a
        saliency map from the gaze_positions.

        :returns Tuple[Array, Array]: frame_stack and saliency_map
        """
        # Input
        item = (self.frames[idx - 3:idx + 1], )

        # Output
        item = (*item,
            self.saliency[idx])

        return item

    @staticmethod
    def from_atari_head_files(
        game_dir: str,
        single_trial=False,
        test_split=0.2,
        load_saliency=False,
        # remove_blinks=False, TODO
    ):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata
        and image paths.

        :param bool load_single_run: Whether to load only a single game trial for
            cases where the entire dataset is not needed
        """
        dfs, frames, saliency, gazes = [], [], [], []

        # Count the files that still need to be loaded
        csv_files = list(
            filter(
                lambda filename: filename.endswith(".csv"),
                os.listdir(game_dir),
            )
        )
        if single_trial:
            csv_files = [random.choice(csv_files)]

        print("Loading images and saliency maps into memory")
        for filename in tqdm(csv_files, total=len(csv_files)):
            csv_path = os.path.join(game_dir, filename)
            subdir_name = os.path.splitext(filename)[0]

            # Load images and gaze positions
            trial_data = (
                pl.scan_csv(csv_path, null_values=["null"])
                .select([
                    pl.col("frame_id"),
                    pl.col("gaze_positions").alias("gazes"),
                    pl.col("duration(ms)")
                ])
                .with_row_index("trial_frame_id")
                .collect()
            )

            # Move the frames out of the df
            trial_frames = torch.from_numpy(np.array([
                preprocess(cv2.imread(os.path.join(game_dir, subdir_name, id + ".png")))
                 for id in trial_data["frame_id"]]))
            assert len(trial_frames) == len(trial_data)
            trial_data = trial_data.drop(pl.col("frame_id"))

            # Move the gazes out of the df
            trial_gazes = [Tensor(eval(x)) for x in trial_data["gazes"]]
            trial_data = trial_data.drop("gazes")

            # Mark train and test data
            split_index = int(len(trial_data) * (1 - test_split))
            trial_data = trial_data.with_columns(
                pl.Series(
                    name="train",
                    values=[True] * split_index
                    + [False] * (len(trial_data) - split_index),
                )
            )

            # Scale the coords from the original resolution down to the screen size
            scaling = ((Tensor(SCREEN_SIZE) - Tensor([1, 1])) \
                       / Tensor(DATASET_SCREEN_SIZE))
            trial_gazes = [gazes * scaling if gazes.numel() > 0 else gazes
                           for gazes in trial_gazes]

            # Load or create saliency maps
            saliency_path = os.path.join(game_dir, "saliency")
            save_path = os.path.join(saliency_path, filename)[:-4] + ".pt"
            if os.path.exists(save_path) and load_saliency:
                trial_saliency = torch.load(save_path, weights_only=False)
            else:
                print(f"Creating saliency maps for {filename}")
                os.makedirs(saliency_path, exist_ok=True)
                trial_saliency = [GazeDataset.create_saliency_map(gazes.cuda()).cpu()
                                  for gazes in trial_gazes]
                torch.save(trial_saliency, save_path)
                print(f"Saliency maps saved under {save_path}")

            dfs.append(trial_data)
            frames.extend(trial_frames)
            gazes.extend(trial_gazes)
            saliency.extend(trial_saliency)

        # Combine the trial data
        df: pl.DataFrame = pl.concat(dfs, how="vertical")
        frames = torch.stack(frames)
        saliency = torch.stack(saliency)
        assert len(df) == len(frames) and len(saliency) == len(df)

        # Create a global id column
        df = df.with_columns(pl.arange(len(df)).alias("id"))

        # Train and val indices
        train_indices = Tensor(list(df.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & pl.col("train")).select(
            pl.col("id")).collect().to_series())).to(torch.int32)
        val_indices = Tensor(list(df.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & ~pl.col("train")).select(
            pl.col("id")).collect().to_series())).to(torch.int32)

        # Cast durations to torch tensor
        durations = Tensor([(duration if duration else 0.)
                                  for duration in list(df["duration(ms)"])])

        return GazeDataset(frames, saliency, train_indices, val_indices,
                           durations, gazes)

    @staticmethod
    def from_game_data(records: List[EpisodeRecord], test_split = 0.2):
        """ Create a dataset from an agent's recorded game. """
        frames, saliency, train_indices, val_indices = [], [], [], []
        durations = [0.]

        for i, record in enumerate(records):
            # Get the saliency maps and gaze durations
            gaze_lists, gazes = [], [] # List of gazes for all rows, list for one frame
            for x, y, pauses, duration in record.annotations[
                        "sensory_action_x", "sensory_action_y", "pauses", "duration"
                    ].iter_rows():
                if x is not None and y is not None:
                    gazes.append([x, y])
                # Add to the gaze duration
                durations[-1] += duration
                if pauses == 0:
                    # Add all accumulated gazes to the frame that no longer pauses
                    gaze_lists.append(gazes)
                    gazes = []
                    # Make a new duration for the next frame
                    durations.append(0.)

            # Merge saliency for all the records
            saliency.extend([GazeDataset.create_saliency_map(Tensor(gazes).clone())
                 for gazes in gaze_lists])

            # Exclude the last episode frame and drop frames with a pause
            mask = np.concat([(record.annotations["pauses"] == 0), [False]])
            records[i].frames = records[i].frames[mask]

            # Get train and val indices
            episode_train_indices, episode_val_indices = \
                GazeDataset._split_episode(len(record.frames), len(frames),
                                           test_split)
            train_indices.extend(episode_train_indices)
            val_indices.extend(episode_val_indices)

            # Append preprocessed frames to the list of frames
            frames.extend([preprocess(frame) for frame in record.frames])

        train_indices = Tensor(train_indices).to(torch.int32)
        val_indices = Tensor(val_indices).to(torch.int32)
        frames = Tensor(np.stack(frames))
        saliency = torch.stack(saliency)
        # Remove the trailing empty duration and cast to milliseconds
        durations = Tensor(durations[:-1]) * 1000
        assert len(train_indices) + len(val_indices) + 3 * len(records) \
            == len(frames) == len(durations)

        return GazeDataset(frames, saliency, train_indices, val_indices, durations,
                           gaze_lists)

    @staticmethod
    def create_saliency_map(gaze_positions: Tensor):
        """
        Takes gaze positions on a 84 by 84 pixels screen to turn them into a saliency
        map

        :param Tensor[N x 2] gaze_positions: A Tensor containing all gaze positions
            associated with one frame
        :return Tensor[W x H]: Greyscale saliency map
        """
        device = gaze_positions.device

        # Double precision; helpful for AUC score calculation later
        dtype = torch.float64
        gaze_positions = gaze_positions.to(dtype)

        # Return zeros when there is no gaze
        if gaze_positions.shape[-1] == 0:
            return torch.zeros(SCREEN_SIZE).to(device).to(dtype)

        # Generate x and y indices
        x = torch.arange(0, SCREEN_SIZE[0], 1).to(device).to(dtype)
        y = torch.arange(0, SCREEN_SIZE[1], 1).to(device).to(dtype)
        x, y = torch.meshgrid(x, y, indexing="xy")

        # Adjust sigma to correspond to one visual degree
        sigmas = Tensor(1 / VISUAL_DEGREES_PER_PIXEL).to(dtype).to(device)
        # Update the sigma to move percentiles for auc calculation
        # into the range of float64
        sigmas *= 4

        # Expand the original tensors for broadcasting
        n_positions = gaze_positions.shape[0]
        gaze_positions = gaze_positions.view(n_positions, 2, 1, 1).expand(
            -1, -1, *SCREEN_SIZE
        )
        x = x.view(1, *SCREEN_SIZE).expand(n_positions, *SCREEN_SIZE)
        y = y.view(1, *SCREEN_SIZE).expand(n_positions, *SCREEN_SIZE)
        mesh = torch.stack([x, y], dim=1)
        sigmas = sigmas.view(1, 2, 1, 1).expand(n_positions, -1, *SCREEN_SIZE)
        # gaze_positions is now Nx2x84x84 with 84x84 identical copies
        # x and y are now both Nx84x84 with N identical copies
        # mesh is Nx2x84x84 with N copies of every possible
        # combination of x and y coordinates
        saliency_map = torch.mean(
            torch.exp(
                -torch.sum(((mesh - gaze_positions) ** 2) / (2 * sigmas**2), dim=1)
            ),
            dim=0,
        )

        # Normalization
        if saliency_map.sum().item() == 0:
            saliency_map = torch.ones(saliency_map.shape)
        saliency_map = saliency_map / saliency_map.sum()
        return saliency_map

    @staticmethod
    def _split_episode(frame_count: int, start_idx: int,
                       test_split = 0.2) -> tuple[list[int], list[int]]:
        """
        Return train and val indices for one game episode

        :param int frame_count:
        :param int start_idx: Number of frames that have already been indexed
        """
        indices = np.arange(3, frame_count) + start_idx
        split_idx = int(len(indices) * (1 - test_split))
        return indices[:split_idx].tolist(), indices[split_idx:].tolist()

    def to_video(self, out_path: str):
        """ Saves a short video of the first frames in the dataset """
        # Get the first 1000 frames as a numpy array
        frames = self.frames[:1000].numpy()
        gazes = [x.numpy().astype(int) for x in self.gazes[:1000]]
        # gazes = [x[:,[1,0]] if x.size > 0 else x for x in gazes]
        # Broadcast to RGB
        frames = np.tile(frames[...,None], (1,1,1,3))
        frames = (frames * 255).astype(np.uint8)

        # Draw gazes onto the video
        for i in range(len(frames)):
            for j in range(len(gazes[i])):
                cv2.drawMarker(frames[i], gazes[i][j], (0,255,0), 1, 5)

        save_video(frames, out_path)

if __name__ == "__main__":
    dataset = GazeDataset.from_atari_head_files("data/Atari-HEAD/asterix", True)
    dataset.to_video("debug.mp4")
