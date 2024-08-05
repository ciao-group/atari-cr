import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
import pandas as pd
from PIL import Image
import tarfile
import io
import numpy as np
from collections import deque

from atari_cr.common.grokking import gradfilter_ema

class GazePredictionNetwork(nn.Module):
    """
    Neural network predicting a saliency map for a given stack of 4 greyscale atari game images.
    """
    def __init__(self):
        super(GazePredictionNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, dtype=torch.float32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dtype=torch.float32)
        
        # Deconvolutional (transpose convolution) layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, dtype=torch.float32)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, dtype=torch.float32)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, dtype=torch.float32)
        
        # Softmax layer; Uses log softmax to conform to the KLDiv expected input
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Deconvolutional layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        
        # Reshape and apply softmax
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        x = x.view(x.size(0), 84, 84)
        
        return x

class GazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads the data in the Atari-HEAD format into a dataframe with metadata and image paths.
        """
        dfs, image_dfs = [], []

        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filter(lambda filename: filename.endswith('.csv'), filenames):

                csv_path = os.path.join(dirpath, filename)
                subdir_name = os.path.splitext(filename)[0]

                df = pd.read_csv(csv_path)
                df = df.set_index("frame_id")

                # Load the images
                image_df = pd.DataFrame(columns=["frame_id", "image_path", "image_tensor"])
                image_df["frame_id"] = pd.Series(df.index)
                image_df["image_path"] = image_df["frame_id"].apply(lambda id: os.path.join(dirpath, subdir_name, id + ".png"))
                image_df["image_tensor"] = image_df["image_path"] \
                    .apply(lambda path: self.transform(read_image(path, ImageReadMode.GRAY)))
                image_df = image_df.set_index("frame_id")

                # TODO: Create saliency maps in df
                
                # Function to create the image_ids list
                image_ids = deque(maxlen=4)
                def get_image_ids(frame_id):
                    image_ids.append(frame_id)
                    return list(image_ids)
                
                # Apply the function to create the image_paths column
                df['stack_images'] = list(pd.Series(df.index).apply(get_image_ids))
                
                # Remove rows where we don't have 4 images yet
                df = df[df['stack_images'].apply(len) == 4]

                dfs.append(df)
                image_dfs.append(image_df)

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        self.image_df = pd.concat(image_dfs)
            
        return combined_df

    @staticmethod
    def _parse_gaze_string(gaze_string: str) -> torch.tensor:
        """
        Parses the string with gaze information into a torch tensor.
        """
        return torch.tensor([
            [float(number) for number in s.strip(", []\\n").split(",")] \
                for s in gaze_string.replace("(", "").replace("'", "").split(")")[:-1]
        ])

    def _create_saliency_map(self, gaze_positions):
        saliency_map = torch.zeros((84, 84), dtype=torch.uint8)

        # Generate x and y indices
        x = torch.arange(0, 84, 1)
        y = torch.arange(0, 84, 1)
        x, y = torch.meshgrid(x, y)

        # Adjust sigma to correspond to one visual degree
        # Screen Size: 44,6 x 28,5 visual degrees; Visual Degrees per Pixel: 0,5310 x 0,3393
        sigma = 2
        for x0, y0 in gaze_positions:
            # Scale the coords from the original resolution down to 84 x 84
            x0 = (x0 * 84) / 160
            y0 = (y0 * 84) / 210
            gaussian = (torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * 255).to(torch.uint8)
            saliency_map = torch.max(saliency_map, gaussian)

        return saliency_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads the images from paths specified in self.data and creates a saliency map from the gaze_positions.
        """
        # Load images
        images = []
        for id in self.data.iloc[idx]["stack_images"]:
            images.append(self.image_df.loc[id]["image_tensor"])
        images = torch.vstack(images)

        # Create saliency map
        gaze_string = self.data.iloc[idx]['gaze_positions']
        gaze_positions = self._parse_gaze_string(gaze_string)
        saliency_map = self._create_saliency_map(gaze_positions)

        # Convert to float to work with neural network
        images = (images * 255).to(torch.float32)
        saliency_map = (saliency_map * 255).to(torch.float32)

        return images, saliency_map
    
class GazePredictor():
    """
    Wrapper around GazePredictionNetwork to handle training etc.
    """
    def __init__(
            self, 
            prediction_network: GazePredictionNetwork,
            dataset: GazeDataset,
            save_path: str
        ):
        self.prediction_network = prediction_network
        self.train_loader, self.val_loader = self._init_data_loaders(dataset)
        self.save_path = save_path

        # Loss function, optimizer and compute device
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(self.prediction_network.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_network.to(self.device)

        # Init Grokfast
        self.grads = None

        # Count the number of trained epochs
        self.epoch = 0

    def train(self, n_epochs: int):
        for epoch in range(self.epoch + n_epochs):
            self.prediction_network.train()
            running_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.prediction_network(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.grads = gradfilter_ema(self.prediction_network, self.grads)
                self.optimizer.step()

                running_loss += loss.item()

                # Print every 100 mini-batches
                if batch_idx % 100 == 99:  
                    print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            self.eval()

        print('Training finished')
        
        # Save the trained model
        self.epoch = epoch
        self.save()

    def eval(self):
        self.prediction_network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.prediction_network(inputs)
                val_loss += self.loss_function(outputs, targets).item()

        print(f'Epoch {self.epoch + 1} completed. Validation Loss: {val_loss / len(self.val_loader):.3f}')
    
    def save(self):
        torch.save({
                "prediction_network": self.prediction_network.state_dict(),
                "epoch": self.epoch
            }, 
            self.save_path
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
        checkpoint = torch.load(save_path)
        model = checkpoint["prediction_network"]
        epoch = checkpoint["epoch"]

        predictor = GazePredictor(model, dataset, save_path)
        predictor.epoch = epoch

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


if __name__ == "__main__":
    # Create dataset and data loader
    transform = transforms.Resize((84, 84))
    dataset = GazeDataset(root_dir='Atari-HEAD/freeway', transform=transform)

    # Initialize the model
    env_name = "freeway"
    output_dir = "output/atari_head_saliency_networks"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, env_name + ".pth")

    if os.path.exists(save_path):
        print("Loading existing gaze predictor")
        gaze_predictor = GazePredictor.from_save_file(save_path, dataset)
    else:
        gaze_predictor = GazePredictor(GazePredictionNetwork(), dataset, save_path)
    gaze_predictor.train(n_epochs=100)

    # Compare a run made by the agent with a run from atari head