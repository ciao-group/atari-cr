import os
import pickle
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from atari_cr.common.tqdm import tqdm

from atari_cr.atari_head.og_heatmap import DatasetWithHeatmap
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE, preprocess

# from atari_cr.atari_head.og_heatmap import DatasetWithHeatmap

class GazeDataset(Dataset):
    def __init__(self, frames: List[torch.Tensor], gaze_lists: List[List[torch.Tensor]], 
                 saliency_maps: List[torch.Tensor], train: Optional[List[bool]] = None, 
                 output_gazes: bool = False):
        if train is None:
            train = [True] * len(frames)
        
        self.data = pd.DataFrame({
            "frame": frames, 
            "gazes": gaze_lists,
            "saliency_map": saliency_maps,
            "train": train
        }).reset_index()
        self.output_gazes = output_gazes

    @staticmethod
    def from_atari_head_files(root_dir: str, load_single_run="", test_split=0.2, load_saliency=False, use_og_saliency=False):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata and image paths.
        """
        tqdm.pandas()
        dfs = []

        # Count the files that still need to be loaded
        i = 0
        csv_files = list(filter(lambda filename: filename.endswith('.csv'), os.listdir(root_dir)))
        total_files = len(csv_files)

        print("Loading images into memory")
        for filename in tqdm(csv_files, total=total_files):
            if not load_single_run in filename: continue
            i += 1

            csv_path = os.path.join(root_dir, filename)
            subdir_name = os.path.splitext(filename)[0]

            trial_data = pd.read_csv(csv_path)

            # Load the images
            trial_data["image_path"] = trial_data["frame_id"].apply(lambda id: os.path.join(root_dir, subdir_name, id + ".png"))
            trial_data["image_tensor"] = trial_data["image_path"] \
                .apply(lambda path: cv2.imread(path)) \
                .apply(preprocess)
                # .apply(lambda path: transforms.Resize((84, 84))(read_image(path, ImageReadMode.GRAY)).view([84, 84]))
            
            # Create saliency maps
            trial_data["gaze_positions"] = trial_data["gaze_positions"].apply(GazeDataset._parse_gaze_string)

            # Mark train and test data
            split_index = int(len(trial_data) * (1 - test_split))
            trial_data.loc[:split_index, "train"] = True
            trial_data.loc[split_index:, "train"] = False

            trimmed_trial_data = pd.DataFrame({
                "frame": trial_data["image_tensor"],
                "gazes": trial_data["gaze_positions"],
                "train": trial_data["train"].astype(bool)
            })

            if use_og_saliency: 
                gazes = list(t.reshape([-1]).tolist() for t in trial_data["gaze_positions"])
                with open("52_gazes.pkl", "wb") as f: pickle.dump(gazes, f)
                saliency_maps = DatasetWithHeatmap().createGazeHeatmap(gazes, 84)
                trimmed_trial_data["saliency"] = pd.Series([array for array in saliency_maps.reshape(saliency_maps.shape[:-1])])
            else: 
                # Load or create saliency maps
                saliency_path = os.path.join(root_dir, "saliency")
                save_path = os.path.join(saliency_path, filename)[:-4] + ".np"
                if os.path.exists(save_path) and load_saliency:
                    print(f"Loading existing saliency maps for {filename}")
                    with open(save_path, "rb") as f: saliency_maps = np.load(f, allow_pickle=True)
                    trimmed_trial_data["saliency"] = pd.Series([array for array in saliency_maps])
                else:
                    print(f"Creating saliency maps for {filename}")
                    os.makedirs(saliency_path, exist_ok=True)
                    trimmed_trial_data["saliency"] = trimmed_trial_data["gazes"].progress_apply(lambda gazes: GazeDataset.create_saliency_map(gazes).numpy())
                    with open(save_path, "wb") as f: np.save(f, trimmed_trial_data["saliency"].to_numpy())
                    print(f"Saliency maps saved under {save_path}")

            dfs.append(trimmed_trial_data)

            if load_single_run:
                break

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
            
        return GazeDataset(
            combined_df["frame"],
            combined_df["gazes"],
            combined_df["saliency"],
            combined_df["train"]
        )
    
    def split(self):
        """
        Splits the dataset into one dataset containing train data and one dataset containing test data
        """
        train_df = self.data[self.data["train"]]
        test_df = self.data[~self.data["train"]]

        train_dataset = GazeDataset(train_df["frame"], train_df["gazes"], train_df["saliency_map"], train_df['train'])
        test_dataset = GazeDataset(test_df["frame"], test_df["gazes"], test_df["saliency_map"], test_df['train'])

        return train_dataset, test_dataset

    @staticmethod
    def _parse_gaze_string(gaze_string: str) -> torch.tensor:
        """
        Parses the string with gaze information into a torch tensor.
        """
        return torch.tensor([
            [float(number) for number in s.strip(", []\\n").split(",")] \
                for s in gaze_string.replace("(", "").replace("'", "").split(")")[:-1]
        ])

    # TODO: Make the frame stacks end at every trial end; currently frame stack contain images of different trials
    def __len__(self): return len(self.data) - 3

    def __getitem__(self, idx):
        """
        Loads the images from paths specified in self.data and creates a saliency map from the gaze_positions.
        
        :returns Tuple[Array, Array, Array]: frame_stack, saliency_map and optionally gazes
        """
        item = (
            torch.stack(list(self.data.loc[idx:idx + 3, "frame"])),
            self.data.loc[idx + 3, "saliency_map"],
        )
        item = (*item, self.data.loc[idx + 3, "gazes"] if self.output_gazes else np.nan)

        assert item[0].shape == torch.Size([4, 84, 84])
        return item
    
    @staticmethod
    def create_saliency_map(gaze_positions: torch.Tensor):
        """ 
        Takes gaze positions on a 84 by 84 pixels screen to turn them into a saliency map 
        
        :param Tensor[Nx2]: A Tensor containing all gaze positions associated with one frame
        """
        SCREEN_SIZE = [84, 84]

        # OPTIONAL: Implement this for a batch of saliency maps
        if gaze_positions.shape[0] == 0:
            return torch.zeros(SCREEN_SIZE)

        # Generate x and y indices
        x = torch.arange(0, SCREEN_SIZE[0], 1)
        y = torch.arange(0, SCREEN_SIZE[1], 1)
        x, y = torch.meshgrid(x, y, indexing="xy")

        # Adjust sigma to correspond to one visual degree
        # Screen Size: 44,6 x 28,5 visual degrees; Visual Degrees per Pixel: 0,5310 x 0,3393
        sigmas = 1 / (torch.Tensor(VISUAL_DEGREE_SCREEN_SIZE) / torch.Tensor(SCREEN_SIZE))
        # Update the sigma to more closely match the outputs of the original gaze predictor
        # sigmas *= 3
        # Scale the coords from the original resolution down to the screen size
        gaze_positions *= (torch.Tensor(SCREEN_SIZE) / torch.Tensor([160, 210]))

        # Expand the original tensors for broadcasting
        n_positions = gaze_positions.shape[0]
        gaze_positions = gaze_positions.view(n_positions, 2, 1, 1).expand(-1, -1, *SCREEN_SIZE)
        x = x.view(1, *SCREEN_SIZE).expand(n_positions, *SCREEN_SIZE)
        y = y.view(1, *SCREEN_SIZE).expand(n_positions, *SCREEN_SIZE)
        mesh = torch.stack([x, y], dim=1)
        sigmas = sigmas.view(1, 2, 1, 1).expand(n_positions, -1, *SCREEN_SIZE)
        # gaze_positions is now Nx2x84x84 with 84x84 identical copies
        # x and y are now both Nx84x84 with N identical copies
        # mesh is Nx2x84x84 with N copies of every possible combination of x and y coordinates
        saliency_map, _ = torch.max(torch.exp(-torch.sum(((mesh - gaze_positions)**2) / (2 * sigmas**2), dim=1)), dim=0)

        # Make the tensor sum to 1 for KL Divergence
        if saliency_map.sum() == 0: saliency_map = torch.ones(saliency_map.shape)
        saliency_map = saliency_map / saliency_map.sum()
        return saliency_map
