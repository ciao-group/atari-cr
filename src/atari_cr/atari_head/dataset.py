import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import polars as pl

from atari_cr.common.module_overrides import tqdm
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE, preprocess

SCREEN_SIZE = [84, 84]
# Screen size from https://github.com/corgiTrax/Gaze-Data-Processor/blob/master/data_visualizer.py
DATASET_SCREEN_SIZE = [160, 210]


class GazeDataset(Dataset):
    """
    Dataset object for the Atari-HEAD dataset

    :param bool class_output: Whether to output the gaze positions as 1D
        discrete class labels
    """

    def __init__(
        self, data: pl.DataFrame, class_output=False
    ):
        self.data = data
        self.class_output = class_output

    def split(self, batch_size=256):
        """
        Splits the dataset into one dataset containing train data and one
        dataset containing test data and returns them as loaders.

        :return: Train and validation loader
        """
        # Find train and val indices
        train_indices = list(self.data.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & pl.col("train")).select(
            pl.col("id")).collect().to_series())
        val_indices = list(self.data.lazy().filter(
            (pl.col("trial_frame_id") >= 3) & ~pl.col("train")).select(
            pl.col("id")).collect().to_series())

        # Shuffle after the split because
        # subsequent images are highly correlated
        train_loader = DataLoader(
            self, batch_size, sampler=SubsetRandomSampler(train_indices)
        )
        val_loader = DataLoader(
            self, batch_size, sampler=SubsetRandomSampler(val_indices)
        )

        return train_loader, val_loader

    @staticmethod
    def _parse_gaze_string(gaze_string: str) -> np.ndarray:
        """
        Parses the string with gaze information into a numpy array.
        """
        return np.array(
            [
                [float(number) for number in s.strip(", []\\n").split(",")]
                for s in gaze_string.replace("(", "").replace("'", "").split(")")[:-1]
            ]
        )

    def __len__(self):
        return (self.data["stack_id"] >= 0).sum()

    def __getitem__(self, idx: torch.Tensor):
        """
        Loads the images from paths specified in self.data and creates a
        saliency map from the gaze_positions.

        :returns Tuple[Array, Array, Array]: frame_stack, saliency_map and
            gazes
        """
        # Input
        item = self.data.lazy().filter(pl.col("id").is_between(idx - 3, idx)).select(
            pl.col("frame")).collect().to_series()
        item = (torch.stack(list(item)), )

        # Output
        if self.class_output:
            raise NotImplementedError
        else:
            item = (*item,
                self.data.lazy().filter(
                    pl.col("id") == idx).select(pl.col("saliency")).collect().item())

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
        dfs = []

        # Count the files that still need to be loaded
        csv_files = list(
            filter(
                lambda filename: filename.endswith(".csv")
                and load_single_run in filename,
                os.listdir(root_dir),
            )
        )

        print("Loading images into memory")
        for filename in tqdm(csv_files, total=len(csv_files)):
            csv_path = os.path.join(root_dir, filename)
            subdir_name = os.path.splitext(filename)[0]

            # Load images and gaze positions
            trial_data = (
                pl.scan_csv(csv_path, null_values=["null"])
                .select([
                    pl.col("frame_id")
                    .map_elements(
                        lambda id: preprocess(
                            cv2.imread(os.path.join(root_dir, subdir_name, id + ".png"))
                        ),
                        pl.Object,
                    )
                    .alias("frame"),
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
                    ).map_elements(torch.Tensor, pl.Object)
                ])
                .with_row_index("trial_frame_id")
                .collect()
            )

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
                gaze_counts = trial_data["gazes"].list.len()
                # Number of gazes at 0.999 quantile
                max_gazes = int(np.quantile(gaze_counts, 0.999))
                trial_data = trial_data.with_columns([pl.col("gazes").head(max_gazes)])

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

                trial_data = trial_data.with_columns(
                    pl.col("gazes").map_elements(trim_coords)
                )

            # Scale the coords from the original resolution down to the screen size
            scaling = ((torch.Tensor(SCREEN_SIZE) - torch.Tensor([1, 1])) \
                       / torch.Tensor(DATASET_SCREEN_SIZE))

            trial_data = trial_data.with_columns(
                pl.col("gazes").map_elements(lambda t: t * scaling))

            # Load or create saliency maps
            saliency_path = os.path.join(root_dir, "saliency")
            save_path = os.path.join(saliency_path, filename)[:-4] + ".np"
            if os.path.exists(save_path) and load_saliency:
                print(f"Loading existing saliency maps for {filename}")
                with open(save_path, "rb") as f:
                    saliency_maps = np.load(f, allow_pickle=True)
                trial_data = trial_data.with_columns(
                    pl.Series(name="saliency", values=list(saliency_maps),
                              dtype=pl.Object)
                )
            else:
                print(f"Creating saliency maps for {filename}")
                os.makedirs(saliency_path, exist_ok=True)
                trial_data = trial_data.with_columns(
                    pl.col("gazes")
                    .map_elements(
                        lambda gazes: GazeDataset
                            .create_saliency_map(gazes).cpu(),
                        pl.Object
                    ).alias("saliency")
                )
                np.save(save_path, trial_data["saliency"].to_numpy())
                print(f"Saliency maps saved under {save_path}")

            dfs.append(trial_data)

        # Combine all dataframes
        df: pl.DataFrame = pl.concat(dfs, how="vertical")

        # Create a global id column
        df = df.with_columns(pl.arange(len(df)).alias("id"))

        # Fill null values for saliency with white tensors
        zeros = pl.lit(torch.zeros(SCREEN_SIZE, dtype=torch.float64), allow_object=True)
        df = df.with_columns(pl.col("saliency").fill_null(zeros))

        # # Create an index for retrieving stacks of 4 frames
        # stack_cond = pl.col("trial_frame_id") >= 3
        # n_stacks = df.lazy().filter(stack_cond).select(pl.len()).collect().item()
        # df = df.with_columns(
        #     pl.when(stack_cond)
        #     .then(0).otherwise(1)
        #     # .then(pl.arange(0, n_stacks))
        #     # .otherwise(pl.arange(-1, -(len(df) - n_stacks) - 1, -1))
        #     .alias("stack_id")
        # )

        if class_output:
            # Turn the coordinates from a (x,y) pair to a 1D vector of size 7056 for
            # transformer usage
            def to_gaze_class(t: torch.Tensor):
                """:param Tensor[N,2] t:"""
                return (t.to(torch.int16) * torch.Tensor([1, 84])).sum(axis=1) + 2

            df = df.with_columns(
                pl.col("gazes").map(to_gaze_class).alias("gaze_classes")
            )

        return GazeDataset(df, class_output=class_output)

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
        if gaze_positions.shape[0] == 0:
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
