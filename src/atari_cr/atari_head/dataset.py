import os
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from atari_cr.atari_head.utils import create_saliency_map, preprocess

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
    def from_atari_head_files(root_dir: str, load_single_run="", test_split=0.2):
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

            # Load or create saliency maps
            saliency_path = os.path.join(root_dir, "saliency")
            save_path = os.path.join(saliency_path, filename)[:-4] + ".np"
            if os.path.exists(save_path):
                print(f"Loading existing saliency maps for {filename}")
                with open(save_path, "rb") as f: saliency_maps = np.load(f, allow_pickle=True)
                trimmed_trial_data["saliency"] = pd.Series([array for array in saliency_maps])
            else:
                print(f"Creating saliency maps for {filename}")
                os.makedirs(saliency_path, exist_ok=True)
                trimmed_trial_data["saliency"] = trimmed_trial_data["gazes"].progress_apply(lambda gazes: create_saliency_map(gazes).numpy())
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

    # TODO: Make the frame stacks end at every trial end; current;y frame stack contain images of different trials
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
