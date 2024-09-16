import os
from typing import List
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from atari_cr.atari_head.utils import create_saliency_map, preprocess

class GazeDataset(Dataset):
    def __init__(self, frames: List[torch.Tensor], gaze_lists: List[List[torch.Tensor]], 
                 saliency_maps: List[torch.Tensor], output_gazes: bool = False):
        self.data = pd.DataFrame({
            "frame": frames, 
            "gazes": gaze_lists,
            "saliency_map": saliency_maps
        })
        self.output_gazes = output_gazes

    @staticmethod
    def from_atari_head_files(root_dir: str, load_single_run=""):
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

            df = pd.read_csv(csv_path)

            # Load the images
            # print(f"Loading images ({i}/{1 if load_single_run else total_files})")
            df["image_path"] = df["frame_id"].apply(lambda id: os.path.join(root_dir, subdir_name, id + ".png"))
            df["image_tensor"] = df["image_path"] \
                .apply(lambda path: cv2.imread(path)) \
                .apply(preprocess)
                # .apply(lambda path: transforms.Resize((84, 84))(read_image(path, ImageReadMode.GRAY)).view([84, 84]))
            # df = df.set_index("frame_id")
            
            # Create saliency maps
            df["gaze_positions"] = df["gaze_positions"].apply(GazeDataset._parse_gaze_string)

            data = pd.DataFrame({
                "frame": df["image_tensor"],
                "gazes": df["gaze_positions"]
            })

            # Load or create saliency maps
            saliency_path = os.path.join(root_dir, "saliency")
            save_path = os.path.join(saliency_path, filename)[:-4] + ".np"
            if os.path.exists(save_path):
                print(f"Loading existing saliency maps for {filename}")
                with open(save_path, "rb") as f: saliency_maps = np.load(f, allow_pickle=True)
                data["saliency"] = pd.Series([array for array in saliency_maps])
            else:
                print(f"Creating saliency maps for {filename}")
                os.makedirs(saliency_path, exist_ok=True)
                data["saliency"] = data["gazes"].progress_apply(lambda gazes: create_saliency_map(gazes).numpy())
                with open(save_path, "wb") as f: np.save(f, data["saliency"].to_numpy())
                print(f"Saliency maps saved under {save_path}")

            dfs.append(data)

            if load_single_run:
                break

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
            
        return GazeDataset(
            combined_df["frame"],
            combined_df["gazes"],
            combined_df["saliency"]
        )

    @staticmethod
    def _parse_gaze_string(gaze_string: str) -> torch.tensor:
        """
        Parses the string with gaze information into a torch tensor.
        """
        return torch.tensor([
            [float(number) for number in s.strip(", []\\n").split(",")] \
                for s in gaze_string.replace("(", "").replace("'", "").split(")")[:-1]
        ])

    def __len__(self):
        # return len(self.data)
        return len(self.data) - 3

    def __getitem__(self, idx):
        """
        Loads the images from paths specified in self.data and creates a saliency map from the gaze_positions.
        """
        item = (
            torch.stack(list(self.data.loc[idx:idx + 3, "frame"])),
            self.data.loc[idx + 3, "saliency_map"],
        )
        item = (*item, self.data.loc[idx + 3, "gazes"] if self.output_gazes else np.nan)

        assert item[0].shape == torch.Size([4, 84, 84])
        return item