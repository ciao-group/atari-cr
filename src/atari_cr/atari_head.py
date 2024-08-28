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
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

from atari_cr.common.utils import gradfilter_ema, grid_image
from atari_cr.common.models import RecordBuffer

# Screen Size in visual degrees: 44,6 x 28,5
# Visual Degrees per Pixel with 84 x 84 pixels: 0,5310 x 0,3393
VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5)

class GazePredictionNetwork(nn.Module):
    """
    Neural network predicting a saliency map for a given stack of 4 greyscale atari game images.
    """
    def __init__(self):
        super(GazePredictionNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv1_norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2_norm = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv3_norm = nn.BatchNorm2d(64)
        
        # Deconvolutional (transpose convolution) layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.deconv1_norm = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.deconv2_norm = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4)
        
        # Softmax layer; Uses log softmax to conform to the KLDiv expected input
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_norm(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_norm(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_norm(x)
        x = self.dropout(x)
        
        # Deconvolutional layers
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv1_norm(x)
        x = self.dropout(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv2_norm(x)
        x = self.dropout(x)
        x = self.deconv3(x)
        
        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        x = x.view(x.size(0), 84, 84)
        
        return x

class GazeDataset(Dataset):
    def __init__(self, frames: List[torch.Tensor], gaze_lists: List[List[torch.Tensor]], 
                 saliency_maps: List[torch.Tensor]):
        self.data = pd.DataFrame({
            "frame": frames, 
            "gazes": gaze_lists,
            "saliency_map": saliency_maps
        })
    # def __init__(self, root_dir, transform=None):
    #     self.transform = transform
    #     # self.data, self.image_data = self._load_data(root_dir)
    #     self.data = self._load_data(root_dir)

    @staticmethod
    def from_gaze_data(frames: List[torch.Tensor], gaze_lists: List[List[torch.Tensor]]):
        """
        Create the dataset from gameplay images and associated gaze positions

        :param List[Tensor[Nx84x84]] frames: List of gameplay frames 
        :param List[List[Tensor[2]]] gaze_lists: List of List of gaze positions associated with one frame
        """
        return GazeDataset(
            frames, 
            gaze_lists,
            [create_saliency_map(gazes) for gazes in gaze_lists],
        )

    @staticmethod
    def from_atari_head_files(root_dir: str, transform=None):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata and image paths.
        """
        dfs = []

        # Count the files that still need to be loaded
        i, total_files = 0, 0
        for _, _, filenames in os.walk(root_dir):
            total_files += len(list(filter(lambda filename: filename.endswith('.csv'), filenames)))

        # Walk through the directory
        for (dirpath, dirnames, filenames) in os.walk(root_dir):
            for filename in filter(lambda filename: filename.endswith('.csv'), filenames):
                i += 1

                csv_path = os.path.join(dirpath, filename)
                subdir_name = os.path.splitext(filename)[0]

                df = pd.read_csv(csv_path)
                # df = df.set_index("frame_id")

                # Load the images
                print(f"Loading images ({i}/{total_files})")
                # df["frame_id"] = pd.Series(df.index)
                df["image_path"] = df["frame_id"].apply(lambda id: os.path.join(dirpath, subdir_name, id + ".png"))
                df["image_tensor"] = df["image_path"] \
                    .apply(lambda path: transform(read_image(path, ImageReadMode.GRAY)))
                df = df.set_index("frame_id")
                
                # Create saliency maps
                print(f"Creating saliency maps ({i}/{total_files})")
                df["gaze_positions"] = df["gaze_positions"].apply(GazeDataset._parse_gaze_string)

                data = pd.DataFrame({
                    "frame": df["image_tensor"],
                    "gazes": df["gaze_positions"]
                })
            
                # # Function to create the image_ids list
                # image_ids = deque(maxlen=4)
                # def get_image_ids(frame_id):
                #     image_ids.append(frame_id)
                #     return list(image_ids)
                
                # # Apply the function to create the image_paths column
                # df['stack_images'] = list(pd.Series(df.index).apply(get_image_ids))
                
                # # Remove rows where we don't have 4 images yet
                # df = df[df['stack_images'].apply(len) == 4]

                dfs.append(data)

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
        # # Load images
        # images = []
        # for id in self.data.iloc[idx]["stack_images"]:
        #     images.append(self.image_data.loc[id]["image_tensor"])
        # images = torch.vstack(images)

        # # Load saliency map
        # saliency_map = self.data.iloc[idx]["saliency_map"]

        # # Convert to float to work with neural network
        # images = (images * 255).to(torch.float32)
        # saliency_map = (saliency_map * 255).to(torch.float32)

        # return images, saliency_map, self.data.iloc[idx]["gaze_positions"]

        return (
            torch.Tensor(self.data.loc[idx:idx + 4, "frame"]),
            self.data.loc[idx + 3, "saliency_map"],
            self.data.loc[idx + 3, "gazes"],
        )
    
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
    if gaze_positions.shape[0] == 0:
        return torch.zeros((84, 84), dtype=torch.uint8)

    # Generate x and y indices
    x = torch.arange(0, 84, 1)
    y = torch.arange(0, 84, 1)
    x, y = torch.meshgrid(x, y, indexing="xy")

    # Adjust sigma to correspond to one visual degree
    # Screen Size: 44,6 x 28,5 visual degrees; Visual Degrees per Pixel: 0,5310 x 0,3393
    sigmas = 1 / (torch.Tensor(VISUAL_DEGREE_SCREEN_SIZE) / 84)
    sigma = 2
    # Scale the coords from the original resolution down to 84 x 84
    gaze_positions *= torch.Tensor([84/160, 84/210])

    # Expand the original tensors for broadcasting
    n_positions = gaze_positions.shape[0]
    gaze_positions = gaze_positions.view(n_positions, 2, 1, 1).expand(-1, -1, 84, 84)
    x = x.view(1, 84, 84).expand(n_positions, 84, 84)
    y = y.view(1, 84, 84).expand(n_positions, 84, 84)
    mesh = torch.stack([x, y], dim=1)
    sigmas = sigmas.view(1, 2, 1, 1).expand(n_positions, -1, 84, 84)
    # gaze_positions is now Nx2x84x84 with 84x84 identical copies
    # x and y are now both Nx84x84 with N identical copies
    # mesh is Nx2x84x84 with N copies of every possible combination of x and y coordinates
    saliency_map, _ = torch.max(torch.exp(-torch.sum(((mesh - gaze_positions)**2) / (2 * sigmas**2), dim=1)), dim=0)

    # GazeDataset._show_tensor((saliency_map * 255).to(torch.uint8))
    return  (saliency_map * 255).to(torch.uint8)

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
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

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


if __name__ == "__main__":    
    # Create dataset and data loader
    transform = transforms.Resize((84, 84))
    dataset = GazeDataset.from_atari_head_files(root_dir='Atari-HEAD/freeway', transform=transform)

    env_name = "freeway"
    output_dir = f"output/atari_head/{env_name}"
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, env_name + ".pth")

    # Load the Atari HEAD model
    model_files = os.listdir(model_dir)
    if len(model_files) > 0:
        print("Loading existing gaze predictor")
        latest_epoch = sorted([int(file[:-4]) for file in model_files])[-1]
        save_path = os.path.join(model_dir, f"{latest_epoch}.pth")
        gaze_predictor = GazePredictor.from_save_file(save_path, dataset)
    else:
        gaze_predictor = GazePredictor(GazePredictionNetwork(), dataset, output_dir)
    gaze_predictor.train(n_epochs=1)

    # Compare a run made by the agent with a run from atari head
    # TODO: Fill in the model
    # raise NotImplementedError
    # atari_head_gaze_predictor = None
    # evaluate_agent(
    #     atari_head_gaze_predictor, 
    #     "/home/niko/Repos/atari-cr/output/runs/pauseable128_1m_fov50/boxing/recordings", 
    #     "boxing"
    # )
    # debug_recording("/home/niko/Repos/atari-cr/output/runs/pauseable128_1m_fov50/boxing/recordings")