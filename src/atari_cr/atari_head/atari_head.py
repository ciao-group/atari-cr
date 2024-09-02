import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchmetrics.classification import BinaryAccuracy
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import h5py

from atari_cr.common.utils import gradfilter_ema, grid_image
from atari_cr.common.models import RecordBuffer

# Screen Size in visual degrees: 44,6 x 28,5
# Visual Degrees per Pixel with 84 x 84 pixels: 0,5310 x 0,3393
VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5)

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
    def from_gaze_data(frames: List[torch.Tensor], gaze_lists: List[List[torch.Tensor]]):
        """
        Create the dataset from gameplay images and associated gaze positions

        :param List[Tensor[Nx84x84]] frames: List of gameplay frames 
        :param List[List[Tensor[2]]] gaze_lists: List of List of gaze positions associated with one frame
        """
        saliency_maps = [None] * len(gaze_lists)
        for i, gazes in enumerate(gaze_lists):
            if i % 1000 == 0:
                print(f"Creating saliency maps ({i+1}/{len(gaze_lists)+1})")
            saliency_maps[i] = create_saliency_map(gazes)

        return GazeDataset(
            frames, 
            gaze_lists,
            saliency_maps,
        )

    @staticmethod
    def from_atari_head_files(root_dir: str, load_single_run=False):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata and image paths.
        """
        dfs = []

        # Count the files that still need to be loaded
        i, total_files = 0, 0
        for _, _, filenames in os.walk(root_dir):
            total_files += len(list(filter(lambda filename: filename.endswith('.csv'), filenames)))

        # Walk through the directory
        for filename in filter(lambda filename: filename.endswith('.csv'), os.listdir(root_dir)):
            i += 1

            csv_path = os.path.join(root_dir, filename)
            subdir_name = os.path.splitext(filename)[0]

            df = pd.read_csv(csv_path)

            # Load the images
            print(f"Loading images ({i}/{1 if load_single_run else total_files})")
            df["image_path"] = df["frame_id"].apply(lambda id: os.path.join(root_dir, subdir_name, id + ".png"))
            df["image_tensor"] = df["image_path"] \
                .apply(lambda path: cv2.imread(path)) \
                .apply(preprocess)
                # .apply(lambda path: transforms.Resize((84, 84))(read_image(path, ImageReadMode.GRAY)).view([84, 84]))
            df = df.set_index("frame_id")
            
            # Create saliency maps
            df["gaze_positions"] = df["gaze_positions"].apply(GazeDataset._parse_gaze_string)

            data = pd.DataFrame({
                "frame": df["image_tensor"],
                "gazes": df["gaze_positions"]
            })

            dfs.append(data)

            if load_single_run:
                break

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
            
        return GazeDataset.from_gaze_data(
            combined_df["frame"],
            combined_df["gazes"]
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

    @staticmethod
    def _show_tensor(t: torch.Tensor, save_path = "debug.png"):
        """
        Saves a grayscale tensor as a .png image for debugging.
        """
        if t.dtype == torch.float32:
            t = t - t.min()  # Shift to positive range
            t = t / t.max()  # Normalize to [0, 1]
            t = (t * 255).byte()  # Scale to [0, 255] and convert to uint8

        image = Image.fromarray(t.numpy(), "L")
        image.save(save_path)


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
        if self.output_gazes:
            item = (*item, self.data.loc[idx + 3, "gazes"])

        assert item[0].shape == torch.Size([4, 84, 84])
        return item

class GazePredictionNetwork(nn.Module):
    """
    Neural network predicting a saliency map for a given stack of 4 greyscale atari game images.
    """
    def __init__(self):
        super(GazePredictionNetwork, self).__init__()
        
        # Convolutional layers
        self.conv2d_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.batch_normalization_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batch_normalization_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batch_normalization_3 = nn.BatchNorm2d(64)
        
        # Deconvolutional (transpose convolution) layers
        self.conv2d_transpose_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.batch_normalization_4 = nn.BatchNorm2d(64)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.batch_normalization_5 = nn.BatchNorm2d(32)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4)
        
        # Softmax layer; Uses log softmax to conform to the KLDiv expected input
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Convolutional layers
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.batch_normalization_1(x)
        x = self.dropout(x)
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = self.batch_normalization_2(x)
        x = self.dropout(x)
        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.batch_normalization_3(x)
        x = self.dropout(x)
        
        # Deconvolutional layers
        x = self.conv2d_transpose_1(x)
        x = F.relu(x)
        x = self.batch_normalization_4(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_2(x)
        x = F.relu(x)
        x = self.batch_normalization_5(x)
        x = self.dropout(x)
        x = self.conv2d_transpose_3(x)
        
        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.log_softmax(x)
        x = x.view(x.size(0), 84, 84)
        
        return x
    
    @staticmethod
    def from_h5(save_path: str):
        f = h5py.File(save_path, 'r')

        model = GazePredictionNetwork()
        state_dict = model.state_dict()

        h5_weights = {}
        for key in f["model_weights"]:
            if len(f["model_weights"][key]) > 0:
                h5_weights[key] = f["model_weights"][key][key]

        for layer in h5_weights:
            for key in h5_weights[layer]:
                value = h5_weights[layer][key]
                key: str = key[:-2]
                key = key.replace("gamma", "weight") \
                    .replace("beta", "bias") \
                    .replace("moving", "running") \
                    .replace("variance", "var") \
                    .replace("kernel", "weight")

                value = torch.Tensor(value)
                value = value.permute(list(reversed(range(len(value.shape)))))
                key = key.replace("kernel", "weight")

                state_dict[f"{layer}.{key}"] = torch.Tensor(value)

        # OPTIONAL: Optimizer weights

        model.load_state_dict(state_dict)

        return model
    
class GazePredictor():
    """
    Wrapper around GazePredictionNetwork to handle training etc.
    """
    def __init__(
            self, 
            prediction_network: GazePredictionNetwork,
            dataset: GazeDataset,
            output_dir: str
        ):
        self.prediction_network = prediction_network
        self.train_loader, self.val_loader = self._init_data_loaders(dataset)
        self.output_dir = output_dir

        # Loss function, optimizer, compute device and tesorboard writer
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adadelta(self.prediction_network.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=0.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_network.to(self.device)
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))

        # Init Grokfast
        self.grads = None

        # Count the number of trained epochs
        self.epoch = 0

    def train(self, n_epochs: int):
        for self.epoch in range(self.epoch + n_epochs):
            self.prediction_network.train()
            running_loss = 0.0

            for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.prediction_network(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.grads = gradfilter_ema(self.prediction_network, self.grads)
                self.optimizer.step()

                running_loss += loss.item()

                # Log every 100 mini-batches
                if batch_idx % 100 == 99:  
                    global_batch_count = self.epoch * len(self.train_loader) / self.train_loader.batch_size + batch_idx
                    self.writer.add_scalar("Train Loss", running_loss / 100, global_batch_count)
                    print(f'[Epoch {self.epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            self.eval()

            if self.epoch % 10 == 9:
                self.save()

        print('Training finished')
        
        # Save the trained model
        self.save()
    
    def save(self):
        torch.save(
            self.prediction_network.state_dict(), 
            os.path.join(self.output_dir, "models", f"{self.epoch + 1}.pth")
        )

    def _init_data_loaders(self, dataset: GazeDataset):
        """
        :returns `Tuple[DataLoader, DataLoader]` train_loader, val_loader: `torch.DataLoader` objects for training and validation
        """
        # Creating data indices for training and validation splits
        validation_split = 0.2
        indices = list(range(len(dataset)))
        split = int(np.floor(validation_split * len(dataset)))

        # Shuffle
        np.random.seed(seed=0)
        np.random.shuffle(indices)

        # Creating data samplers and loaders
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

        return train_loader, val_loader
    
    @staticmethod
    def from_save_file(save_path: str, dataset: GazeDataset):
        model = GazePredictionNetwork().load_state_dict(torch.load(save_path))

        predictor = GazePredictor(model, dataset, save_path)
        predictor.epoch = int(save_path.split("/")[-1][:-4])

        return predictor

def transform_to_proper_csv(game_dir: str):
    """
    Transforms the pseudo csv format used by Atari-HEAD to proper csv

    :param str game_dir: The directory containing files for one game. Obtained by unzipping \<game\>.zip
    """
    csv_files = list(filter(lambda file_name: ".txt" in file_name, os.listdir(game_dir)))
    for file_name in csv_files:

        # Read the original file
        file_path = f"{game_dir}/{file_name}"
        with open(file_path, "r") as f:
            lines = f.readlines()

        data = []
        for line in lines[1:]:
            # Put the gaze positions into a list of tuples instead of a flat list
            tupled_gaze_positions = []
            gaze_positions = line.split(",")[6:]
            for i in range(len(gaze_positions) // 2):
                x_coord = gaze_positions[2 * i]
                y_coord = gaze_positions[2 * i + 1]
                tupled_gaze_positions.append((x_coord, y_coord))

            # Append a new row to the data
            data.append([
                *line.split(",")[:6],
                tupled_gaze_positions
            ])

        # Export the data to csv and delete the original files
        df = pd.DataFrame(data, columns=[
            "frame_id", 
            "episode_id", 
            "score", 
            "duration(ms)", 
            "unclipped_reward", 
            "action", 
            "gaze_positions"
        ])
        df.to_csv(".".join(file_path.split(".")[:-1]) + ".csv", index=False)
        os.remove(file_path)

def create_saliency_map(gaze_positions: torch.Tensor):
    """ 
    Takes gaze positions on a 84 by 84 pixels screen to turn them into a saliency map 
    
    :param Tensor[Nx2]: A Tensor containing all gaze positions associated with one frame
    """
    screen_size = [84, 84]

    # OPTIONAL: Implement this for a batch of saliency maps
    if gaze_positions.shape[0] == 0:
        return torch.zeros(screen_size, dtype=torch.uint8)

    # Generate x and y indices
    x = torch.arange(0, screen_size[0], 1)
    y = torch.arange(0, screen_size[1], 1)
    x, y = torch.meshgrid(x, y, indexing="xy")

    # Adjust sigma to correspond to one visual degree
    # Screen Size: 44,6 x 28,5 visual degrees; Visual Degrees per Pixel: 0,5310 x 0,3393
    sigmas = 1 / (torch.Tensor(VISUAL_DEGREE_SCREEN_SIZE) / torch.Tensor(screen_size))
    sigma = 2
    # Scale the coords from the original resolution down to the screen size
    gaze_positions *= (torch.Tensor(screen_size) / torch.Tensor([160, 210]))

    # Expand the original tensors for broadcasting
    n_positions = gaze_positions.shape[0]
    gaze_positions = gaze_positions.view(n_positions, 2, 1, 1).expand(-1, -1, *screen_size)
    x = x.view(1, *screen_size).expand(n_positions, *screen_size)
    y = y.view(1, *screen_size).expand(n_positions, *screen_size)
    mesh = torch.stack([x, y], dim=1)
    sigmas = sigmas.view(1, 2, 1, 1).expand(n_positions, -1, *screen_size)
    # gaze_positions is now Nx2x84x84 with 84x84 identical copies
    # x and y are now both Nx84x84 with N identical copies
    # mesh is Nx2x84x84 with N copies of every possible combination of x and y coordinates
    saliency_map, _ = torch.max(torch.exp(-torch.sum(((mesh - gaze_positions)**2) / (2 * sigmas**2), dim=1)), dim=0)

    # Make the tensor sum to 1 for KL Divergence
    saliency_map = torch.nn.Softmax()(saliency_map.flatten()).view(screen_size)
    return saliency_map

def open_mp4_as_frame_list(path: str):
    video = cv2.VideoCapture(path)

    frames = []
    while True:
        success, frame = video.read()
        
        if success:
            frames.append(frame)
        else:
            break

    video.release()
    return frames

def evaluate_agent(atari_head_gaze_predictor: nn.Module, recordings_path: str, game: str):
    """
    Evaluate an agent given a path to their gameplay record buffer data.

    :param str recordings_path: Path to the agent's eval data, containing images and associated gaze positions 
    :returns Tuple[float, float]: KL-Divergence and AUC of the agent's saliency maps compared to Atari-HEAD 
    """
    kl_divs, aucs = [], []
    for file in filter(lambda x: x.endswith(".pt"), os.listdir(recordings_path)):
        data = RecordBuffer.from_file(file)
        dataset = data.to_atari_head(game)

        # Load the data and compare the agents saliency to the gaze predictor's saliency
        loader = DataLoader(dataset, batch_size=16, drop_last=True)
        for frame_stacks, _, agent_saliency_maps in loader:
            ground_truth_saliency_maps = atari_head_gaze_predictor(frame_stacks)

            kl_divergence = nn.KLDivLoss()(agent_saliency_maps, ground_truth_saliency_maps)
            auc = roc_auc_score(agent_saliency_maps.flatten(), ground_truth_saliency_maps.flatten()) 

            aucs.append(auc)
            kl_divs.append(kl_divergence)

        # # Get saliency maps for all 4-stacks of frames        
        # data = torch.load(os.path.join(recordings_path, file))
        # frames = open_mp4_as_frame_list(data["rgb"])
        # greyscale_frames = [Image.fromarray(frame).convert("L") for frame in frames]
        # scaled_tensors = [transform(frame) for frame in greyscale_frames]
        # ground_truth_saliency_maps = []
        # for i in range(len(scaled_tensors) - 3):
        #     frame_stack = torch.vstack(scaled_tensors[i:i + 4])
        #     # saliency_map = <model>(frame_stack)
        #     # ground_truth_saliency_maps.append(saliency_map)
        # # ground_truth_saliency_maps = torch.stack(ground_truth_saliency_maps)

        # # Get saliency maps made from agents gazes
        # gazes = data["fov_loc"]
        # agent_saliency_maps = []
        # for gaze in gazes[3:]:
        #     agent_saliency_maps.append(create_saliency_map(torch.Tensor(gaze).unsqueeze(0)))
        # agent_saliency_maps = torch.stack(agent_saliency_maps)

        # # Compare them using KL Divergence and AUC
        # assert len(ground_truth_saliency_maps) == len(agent_saliency_maps)
        # kl_divergence = nn.KLDivLoss()(agent_saliency_maps, ground_truth_saliency_maps)
        # auc = roc_auc_score(agent_saliency_maps.flatten(), ground_truth_saliency_maps.flatten()) 

    return np.mean(kl_divs), np.mean(aucs)

def debug_recording(recordings_path: str):
    """
    :param str recordings_path: Path to the agent's eval data, containing images and associated gaze positions 
    """
    # Get the recording data of the first recording as a dict
    file = list(filter(lambda x: x.endswith(".pt"), os.listdir(recordings_path)))[0]
    data: RecordBuffer = torch.load(os.path.join(recordings_path, file))

    # Extract a list of frames and a list of gazes
    frames = open_mp4_as_frame_list(data["rgb"])
    actions = data["action"]
    assert len(frames) == len(actions)

    for frame, action in zip(frames, actions):

        boxing_pause_action = 18
        if action == boxing_pause_action:
            # Write "pause" on the frame
            text = "pause"
            position = (10, 20)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.3
            color = (255, 0, 0)
            thickness = 1
            frame = cv2.putText(frame, text, position, font, font_scale, color, thickness)

    # Display images in a grid
    grid = np.array(frames[:16])
    grid = grid.reshape([4, 4, *grid.shape[1:]])
    grid = grid_image(grid)
    Image.fromarray(grid).save("debug.png")

def preprocess(image: np.ndarray):
    """
    Image preprocessing function from IL-CGL.
    Warp frames to 84x84 as done in the Nature paper and later work.

    :param np.ndarray image: uint8 greyscale image loaded using `cv2.imread`
    """
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return torch.Tensor(frame / 255.0)

def saliency_auc(saliency_map1: torch.Tensor, saliency_map2: torch.Tensor):
    """
    Predicts the AUC between two greyscale saliency maps by comparing their most salient pixels each

    :param Tensor[WxH] saliency_map1: First saliency map
    :param Tensor[WxH] saliency_map2: Second saliency map
    """
    # Flatten both maps
    saliency_map1 = saliency_map1.flatten()
    saliency_map2 = saliency_map2.flatten()
    assert saliency_map1.shape == saliency_map2.shape, "Saliency maps need to have the same size"

    # Get saliency maps containing only the most salient pixels for every 5% percentile
    percentiles = torch.arange(0.05, 1., 0.05)
    saliency_map1_thresholds = torch.quantile(saliency_map1, percentiles).view(-1, 1).expand(-1, len(saliency_map1))
    saliency_map2_thresholds = torch.quantile(saliency_map2, percentiles).view(-1, 1).expand(-1, len(saliency_map2))
    thresholded_saliency_maps1 = saliency_map1.expand([len(percentiles), -1]) >= saliency_map1_thresholds
    thresholded_saliency_maps2 = saliency_map2.expand([len(percentiles), -1]) >= saliency_map2_thresholds

    return BinaryAccuracy()(thresholded_saliency_maps1, thresholded_saliency_maps2)


if __name__ == "__main__":    
    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")

    # Create dataset and data loader
    env_name = "ms_pacman"
    transform = transforms.Resize((84, 84))
    dataset = GazeDataset.from_atari_head_files(root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=True)
    loader = DataLoader(dataset, 64)

    output_dir = f"output/atari_head/{env_name}"
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # # Load the Atari HEAD model
    # model_files = os.listdir(model_dir)
    # if len(model_files) > 0:
    #     print("Loading existing gaze predictor")
    #     latest_epoch = sorted([int(file[:-4]) for file in model_files])[-1]
    #     save_path = os.path.join(model_dir, f"{latest_epoch}.pth")
    #     gaze_predictor = GazePredictor.from_save_file(save_path, dataset)
    # else:
    #     gaze_predictor = GazePredictor(GazePredictionNetwork(), dataset, output_dir)
    # gaze_predictor.train(n_epochs=1)

    # Compare a run made by the agent with a run from atari head
    atari_head_gaze_predictor = GazePredictionNetwork.from_h5(f"data/h5_gaze_predictors/{env_name}.hdf5")

    # # TODO: Test if the model produces reasonable output on the dataset
    # frame_stacks, saliency_maps = next(iter(loader))
    # assert frame_stacks.max() < 1.0 and frame_stacks.min() >= 0, "Greyscale values should be between 0 and 1"
    # preds = atari_head_gaze_predictor(frame_stacks)
    # kl_div = nn.functional.kl_div(preds, saliency_maps)
    # auc = saliency_auc(preds.exp(), saliency_maps)

    # TODO: Read ms_pacman_52_RZ_2394668_Aug-10-14-52-42_preds.np and compare it to ground truth

    evaluate_agent(
        atari_head_gaze_predictor, 
        "/home/niko/Repos/atari-cr/output/runs/pauseable128_1m_fov50/boxing/recordings", 
        "boxing"
    )
    debug_recording("/home/niko/Repos/atari-cr/output/runs/pauseable128_1m_fov50/boxing/recordings")