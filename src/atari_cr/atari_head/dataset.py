from enum import Enum
import os
import pickle
from typing import List, Literal, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from atari_cr.common.module_overrides import tqdm

from atari_cr.atari_head.og_heatmap import DatasetWithHeatmap
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE, preprocess

MAX_GAZES = 3000 # Maximum allowed number of gaze positions, rows with more are cut from the data
SCREEN_SIZE = [84, 84]
DATASET_SCREEN_SIZE = [160, 210] # Taken from https://github.com/corgiTrax/Gaze-Data-Processor/blob/master/data_visualizer.py

class Mode(Enum):
    SALIENCY = 0
    GAZE_CLASSES = 1

class GazeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, train: Optional[List[bool]] = None, 
                 max_gazes: Optional[int] = None, mode: Mode = Mode.SALIENCY):
        if not "train" in data.columns:
            if train is not None: data["train"] = train
            else: data["train"] = True

        self.data = data.reset_index(drop=True)
        self.max_gazes = max_gazes 
        self.mode = mode

    @staticmethod
    def from_atari_head_files(root_dir: str, load_single_run="", test_split=0.2, 
            load_saliency=False, use_og_saliency=False, mode: Mode=Mode.SALIENCY):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata and image paths.

        :param str load_single_run: Name of a single trial to be used. If empty, all trials are loaded
        """
        tqdm.pandas()
        dfs = []

        # Count the files that still need to be loaded
        csv_files = list(filter(
            lambda filename: filename.endswith('.csv') and load_single_run in filename, 
            os.listdir(root_dir)
        ))

        print("Loading images into memory")
        for filename in tqdm(csv_files, total=len(csv_files)):
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

            # A small fraction of frames has up to 16.000 gaze positions
            # The number of gazes for these is trimmed to the 0.999 percentile to facilitate learning
            trimmed_trial_data["gaze_counts"] = trimmed_trial_data["gazes"].apply(len)
            max_gazes = int(trimmed_trial_data["gaze_counts"].quantile(0.999)) # Number of gazes at 0.999 quantile
            trimmed_trial_data["gazes"] = trimmed_trial_data["gazes"].apply(lambda t: t[:max_gazes])
            trimmed_trial_data["gaze_counts"] = trimmed_trial_data["gazes"].apply(len)

            # Trim the gaze positions to be within the screen
            # Original max coords: [287.45, 419.12], min coords: [-112.39,-40.02]
            if mode == Mode.GAZE_CLASSES:
                min_coords = torch.Tensor([0,0])
                max_coords = torch.Tensor(DATASET_SCREEN_SIZE)
                def trim_coords(t: torch.Tensor):
                    if t.shape == torch.Size([0]): return t
                    n = t.size(0)
                    t = torch.max(t, min_coords.view([1,2]).expand([n,2]))
                    t = torch.min(t, max_coords.view([1,2]).expand([n,2]))
                    return t
                trimmed_trial_data["gazes"] = trimmed_trial_data["gazes"].apply(trim_coords)

            # Scale the coords from the original resolution down to the screen size
            # Desired max coords: [83,83], min coords: [0,0]
            scaling = (torch.Tensor(SCREEN_SIZE) - torch.Tensor([1,1])) / torch.Tensor(DATASET_SCREEN_SIZE)
            def scale_coords(t: torch.Tensor):
                if t.shape == torch.Size([0]): return t
                return t * scaling
            trimmed_trial_data["gazes"] = trimmed_trial_data["gazes"].apply(scale_coords)

            a = torch.stack([t.max(dim=0)[0] for t in trimmed_trial_data["gazes"] if t.shape != torch.Size([0])])
            print(a.max(dim=0))

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

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Pad the gazes to be retrievable using data loaders
        def pad_gazes(t: torch.Tensor):
            """ :param Tensor[N,2] t: """
            max_n = combined_df["gaze_counts"].max()
            padded_t = torch.full([max_n, 2], -1) # Pad empty coords with [-1,-1]
            if t.shape == torch.Size([0]): return padded_t
            padded_t[:t.size(0), :] = t
            return padded_t
        combined_df["gazes"] = combined_df["gazes"].apply(pad_gazes)

        if mode == Mode.GAZE_CLASSES:
            # Turn the coordinates from a (x,y) pair to a 1D vector of size 7056 for transformer usage
            def to_gaze_class(t: torch.Tensor):
                """ :param Tensor[N,2] t: """
                return (t.to(torch.int16) * torch.Tensor([1,84])).sum(axis=1) + 2
            combined_df["gaze_classes"] = combined_df["gazes"].apply(to_gaze_class)
            
        return GazeDataset(
            combined_df,
            max_gazes = combined_df["gaze_counts"].max(),
            mode=mode
        )
    
    def split(self, batch_size=512):
        """ 
        Splits the dataset into one dataset containing train data and one dataset containing test data
        and returns them as loaders.

        :return: Train and validation loader 
        """
        # TODO: Make frame stacks not include more than one trial
        np.random.seed(seed=42)

        train_df = self.data[self.data["train"]]
        val_df = self.data[~self.data["train"]]

        train_dataset = GazeDataset(train_df, max_gazes=self.max_gazes, mode=self.mode)
        val_dataset = GazeDataset(val_df, max_gazes=self.max_gazes, mode=self.mode)

        # Shuffle after the split because subsequent images are highly correlated
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=True)

        return train_loader, val_loader

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
        
        :returns Tuple[Array, Array, Array]: frame_stack, saliency_map and gazes
        """
        # Input
        item = (torch.stack(list(self.data.loc[idx:idx + 3, "frame"])),)

        # Output
        match(self.mode):
            case Mode.SALIENCY: item = (*item,
                self.data.loc[idx + 3, "saliency"],
                self.data.loc[idx + 3, "gazes"],
            )
            case Mode.GAZE_CLASSES: item = (*item,
                self.data.loc[idx + 3, "gaze_classes"],
            )

        return item
    
    @staticmethod
    def create_saliency_map(gaze_positions: torch.Tensor):
        """ 
        Takes gaze positions on a 84 by 84 pixels screen to turn them into a saliency map 
        
        :param Tensor[N x 2] gaze_positions: A Tensor containing all gaze positions associated with one frame
        :return Tensor[W x H]: Greyscale saliency map
        """
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
        softmax = False
        if softmax:
            # For some reason, softmaxing instead of normalizing breaks model training
            saliency_map = nn.Softmax(dim=0)(saliency_map.view(-1)).view(SCREEN_SIZE)
        else:
            if saliency_map.sum() == 0: saliency_map = torch.ones(saliency_map.shape)
            saliency_map = saliency_map / saliency_map.sum()
        return saliency_map
