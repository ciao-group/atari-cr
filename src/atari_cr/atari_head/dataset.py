import os
from typing import List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import polars as pl

from atari_cr.models import EpisodeRecord
from atari_cr.module_overrides import tqdm
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE, preprocess

SCREEN_SIZE = [84, 84]
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
    :param bool class_output: Whether to output the gaze positions as 1D
        discrete class labels
    """

    def __init__(
        self, frames: torch.Tensor, saliency: torch.Tensor,
        train_indices: torch.Tensor, val_indices: torch.Tensor, durations: torch.Tensor,
        class_output=False,
    ):
        self.frames = frames
        self.saliency = saliency
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.durations = durations
        self.class_output = class_output

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

    def __getitem__(self, idx: torch.Tensor):
        """
        Loads the images from paths specified in self.data and creates a
        saliency map from the gaze_positions.

        :returns Tuple[Array, Array]: frame_stack and saliency_map
        """
        # Input
        item = (self.frames[idx - 3:idx + 1], )

        # Output
        if self.class_output:
            raise NotImplementedError
        else:
            item = (*item,
                self.saliency[idx])

        return item

    @staticmethod
    def from_atari_head_files(
        root_dir: str,
        load_single_run="",
        test_split=0.2,
        load_saliency=False,
        class_output=False,
    ):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata
        and image paths.

        :param str load_single_run: Name of a single trial to be used. If
            empty, all trials are loaded
        """
        dfs, frames, saliency = [], [], []

        # Count the files that still need to be loaded
        csv_files = list(
            filter(
                lambda filename: filename.endswith(".csv")
                and load_single_run in filename,
                os.listdir(root_dir),
            )
        )

        print("Loading images and saliency maps into memory")
        for filename in tqdm(csv_files, total=len(csv_files)):
            csv_path = os.path.join(root_dir, filename)
            subdir_name = os.path.splitext(filename)[0]

            # Load images and gaze positions
            trial_data = (
                pl.scan_csv(csv_path, null_values=["null"])
                .select([
                    pl.col("frame_id"),
                    # Parse gaze positions string to a list of float arrays
                    pl.when(pl.col("gaze_positions") == "[]")
                        .then(None)
                        .otherwise(pl.col("gaze_positions"))
                        .alias("gazes")
                        .str.strip_chars("[])")
                        .str.replace_all(r"['\(,n\\]", "")
                        .str.split(") ")
                        .list.eval(
                            pl.element().str.split(" ").cast(pl.List(pl.Float32))
                        ),
                    pl.col("duration(ms)")
                ])
                .with_row_index("trial_frame_id")
                .collect()
            )

            # Move the frames out of the df
            trial_frames = torch.from_numpy(np.array([
                preprocess(cv2.imread(os.path.join(root_dir, subdir_name, id + ".png")))
                 for id in trial_data["frame_id"]]))
            assert len(trial_frames) == len(trial_data)
            trial_data = trial_data.drop(pl.col("frame_id"))

            # Move the gazes out of the df
            trial_gazes = [torch.from_numpy(np.array([[]]) if series is None else
                np.stack(series.to_numpy())) for series in trial_data["gazes"]]
            trial_data = trial_data.drop(pl.col("gazes"))

            # Mark train and test data
            split_index = int(len(trial_data) * (1 - test_split))
            trial_data = trial_data.with_columns(
                pl.Series(
                    name="train",
                    values=[True] * split_index
                    + [False] * (len(trial_data) - split_index),
                )
            )

            if class_output:
                # A small fraction of frames has up to 16.000 gaze positions
                # The number of gazes for these is trimmed
                # to the 0.999 percentile to facilitate learning
                gaze_counts = [len(a) for a in trial_gazes]
                # Number of gazes at 0.999 quantile
                max_gazes = int(np.quantile(gaze_counts, 0.999))
                trial_gazes = [gazes[:max_gazes] for gazes in trial_gazes]
                # Trim the gaze positions to be within the screen
                min_coords = torch.Tensor([0, 0])
                max_coords = torch.Tensor(DATASET_SCREEN_SIZE)

                def trim_coords(t: torch.Tensor):
                    if t.shape == torch.Size([0]):
                        return t
                    n = t.size(0)
                    t = torch.max(t, min_coords.view([1, 2]).expand([n, 2]))
                    t = torch.min(t, max_coords.view([1, 2]).expand([n, 2]))
                    return t

                trial_gazes = [trim_coords(gazes) for gazes in trial_gazes]

            # Scale the coords from the original resolution down to the screen size
            scaling = ((torch.Tensor(SCREEN_SIZE) - torch.Tensor([1, 1])) \
                       / torch.Tensor(DATASET_SCREEN_SIZE))
            trial_gazes = [gazes * scaling if gazes.size(1) == 2 else gazes
                           for gazes in trial_gazes]

            # Load or create saliency maps
            saliency_path = os.path.join(root_dir, "saliency")
            save_path = os.path.join(saliency_path, filename)[:-4] + ".pt"
            if os.path.exists(save_path) and load_saliency:
                trial_saliency = torch.load(save_path, weights_only=False)
            else:
                print(f"Creating saliency maps for {filename}")
                os.makedirs(saliency_path, exist_ok=True)
                trial_saliency = [GazeDataset.create_saliency_map(gazes).cpu()
                                  for gazes in trial_gazes]
                torch.save(trial_saliency, save_path)
                print(f"Saliency maps saved under {save_path}")

            dfs.append(trial_data)
            frames.extend(trial_frames)
            saliency.extend(trial_saliency)

        # Combine the trial data
        df: pl.DataFrame = pl.concat(dfs, how="vertical")
        frames = torch.stack(frames)
        saliency = torch.stack(saliency)
        assert len(df) == len(frames) and len(saliency) == len(df)

        # Create a global id column
        df = df.with_columns(pl.arange(len(df)).alias("id"))

        if class_output:
            # Turn the coordinates from a (x,y) pair to a 1D vector of size 7056 for
            # transformer usage
            def to_gaze_class(t: torch.Tensor):
                """:param Tensor[N,2] t:"""
                return (t.to(torch.int16) * torch.Tensor([1, 84])).sum(axis=1) + 2

            df = df.with_columns(
                pl.col("gazes").map_elements(to_gaze_class).alias("gaze_classes")
            )

        # Train and val indices
        train_indices = torch.Tensor(list(df.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & pl.col("train")).select(
            pl.col("id")).collect().to_series())).to(torch.int32)
        val_indices = torch.Tensor(list(df.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & ~pl.col("train")).select(
            pl.col("id")).collect().to_series())).to(torch.int32)

        # Cast durations to torch tensor
        durations = torch.Tensor([(duration if duration else 0.)
                                  for duration in list(df["duration(ms)"])])

        return GazeDataset(frames, saliency, train_indices, val_indices,
                           durations, class_output=class_output)

    @staticmethod
    def from_game_data(records: List[EpisodeRecord], test_split = 0.2):
        """ Create a dataset from an agent's recorded game """
        frames, saliency, train_indices, val_indices = [], [], [], []
        durations = [0.]

        for record in records:
            # Get the saliency maps and gaze durations
            gaze_lists, gazes = [], [] # List of gazes for all rows, list for one frame
            pause_indices = []
            for i, (x, y, pauses, emma_time) in enumerate(record.annotations[
                "sensory_action_x", "sensory_action_y", "pauses", "emma_time"
                    ].iter_rows()):
                # OPTIONAL: Why is the last sensory action not None in the pauseable env
                # There should be one more fov loc than sensory action
                if x is not None and y is not None:
                    gazes.append([x, y])
                # Add to the gaze duration
                durations[-1] += emma_time
                if pauses == 0:
                    # Add all accumulated gazes to the frame that no longer pauses
                    gaze_lists.append(gazes)
                    gazes = []
                    # Make a new duration for the next frame
                    durations.append(0.)
                else:
                    # Keep accumulating gazes and mark this env step for removal
                    pause_indices.append(i)

            # Merge saliency for all the records
            saliency.extend([GazeDataset.create_saliency_map(torch.Tensor(gazes))
                 for gazes in gaze_lists])

            # Drop frames with a pause
            mask = np.ones(record.frames.shape[0], dtype=bool)
            mask[pause_indices] = False
            record.frames = record.frames[mask]

            # Get train and val indices
            episode_train_indices, episode_val_indices = \
                GazeDataset._split_episode(len(record.frames), len(frames), test_split)
            train_indices.extend(episode_train_indices)
            val_indices.extend(episode_val_indices)

            # Append preprocessed frames to the list of frames
            frames.extend([preprocess(frame) for frame in record.frames])

        train_indices = torch.Tensor(train_indices).to(torch.int32)
        val_indices = torch.Tensor(val_indices).to(torch.int32)
        frames = torch.Tensor(np.stack(frames))
        saliency = torch.stack(saliency)
        # Remove the trailing empty duration and cast to milliseconds
        durations = torch.Tensor(durations[:-1]) * 1000
        assert len(train_indices) + len(val_indices) + 3 * len(records) \
            == len(frames) == len(durations)

        return GazeDataset(frames, saliency, train_indices, val_indices, durations)

    @staticmethod
    def create_saliency_map(gaze_positions: torch.Tensor):
        """
        Takes gaze positions on a 84 by 84 pixels screen to turn them into a saliency
        map

        :param Tensor[N x 2] gaze_positions: A Tensor containing all gaze positions
            associated with one frame
        :return Tensor[W x H]: Greyscale saliency map
        """
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Double precision; helpful for AUC score calculation later
        dtype = torch.float64
        gaze_positions = gaze_positions.to(DEVICE).to(dtype)

        # Return zeros when there is no gaze
        if gaze_positions.shape[-1] == 0:
            return torch.zeros(SCREEN_SIZE).to(DEVICE).to(dtype)

        # Generate x and y indices
        x = torch.arange(0, SCREEN_SIZE[0], 1).to(DEVICE).to(dtype)
        y = torch.arange(0, SCREEN_SIZE[1], 1).to(DEVICE).to(dtype)
        x, y = torch.meshgrid(x, y, indexing="xy")

        # Adjust sigma to correspond to one visual degree
        # Screen Size: 44,6 x 28,5 visual degrees
        # Visual Degrees per Pixel: 0,5310 x 0,3393
        sigmas = (
            torch.Tensor(SCREEN_SIZE).to(dtype)
            / torch.Tensor(VISUAL_DEGREE_SCREEN_SIZE).to(dtype)
        ).to(DEVICE)
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
        saliency_map, _ = torch.max(
            torch.exp(
                -torch.sum(((mesh - gaze_positions) ** 2) / (2 * sigmas**2), dim=1)
            ),
            dim=0,
        )

        # Make the tensor sum to 1 for KL Divergence
        softmax = False
        if softmax:
            # Softmaxing instead of normalizing breaks model training
            # Probably because of extremely high entropy after softmaxing
            saliency_map = torch.sparse.softmax(saliency_map.view(-1), dim=0).view(
                SCREEN_SIZE
            )
        else:
            # Normalization
            if saliency_map.sum() == 0:
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

if __name__ == "__main__":
    episode_dirs = [
        "tests/assets/ray-10-16/" + dir for dir in os.listdir("tests/assets/ray-10-16")]
    dataset = GazeDataset.from_game_data(
            [EpisodeRecord.load(dir) for dir in episode_dirs])
    loader = dataset.to_loader()
    print(dataset)
