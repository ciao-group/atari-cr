import os
import time
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import h5py
from tap import Tap

from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.common.utils import gradfilter_ema, grid_image, show_tensor, grid_image2

class ArgParser(Tap):
    debug: bool = False # Debug mode for less data loading

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

                if not isinstance(value[:], np.ndarray): breakpoint()
                value = torch.Tensor(value[:])
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
        for self.epoch in range(self.epoch, self.epoch + n_epochs):
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
                    print(f'[Epoch {self.epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0

            self.prediction_network.eval()

            if self.epoch % 10 == 9:
                self.save()

        print('Training finished')
        
        # Save the trained model
        self.save()
    
    def save(self):
        save_path = f"{self.epoch + 1}.pth"
        torch.save(
            self.prediction_network.state_dict(), 
            os.path.join(self.output_dir, "models", save_path)
        )
        print(f"Saved model to {save_path}")

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
    def from_save_file(save_path: str, dataset: GazeDataset, output_dir: str):
        model = GazePredictionNetwork()
        model.load_state_dict(torch.load(save_path))

        predictor = GazePredictor(model, dataset, output_dir)
        predictor.epoch = int(save_path.split("/")[-1][:-4])

        return predictor


if __name__ == "__main__": 
    args = ArgParser().parse_args()

    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")

    # Create dataset and data loader
    env_name = "ms_pacman"
    single_run = "52_RZ_2394668_Aug-10-14-52-42" if args.debug else ""
    dataset = GazeDataset.from_atari_head_files(root_dir=f'data/Atari-HEAD/{env_name}', load_single_run=single_run)
    loader = DataLoader(dataset, 128)

    # Create the dir for saving the trained model
    output_dir = f"output/atari_head/{env_name}"
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load an existing Atari HEAD model
    model_files = os.listdir(model_dir)
    if len(model_files) > 0:
        print("Loading existing gaze predictor")
        latest_epoch = sorted([int(file[:-4]) for file in model_files])[-1]
        save_path = os.path.join(model_dir, f"{latest_epoch}.pth")
        gaze_predictor = GazePredictor.from_save_file(save_path, dataset, output_dir)
    else:
        print("Creating new gaze model from hdfs5 weights")
        model = GazePredictionNetwork.from_h5(f"data/h5_gaze_predictors/{env_name}.hdf5")
        gaze_predictor = GazePredictor(model, dataset, output_dir)

    # Train the model
    gaze_predictor.train(n_epochs=10)

    # # Test if the model produces reasonable output on the dataset
    # frame_stacks, saliency_maps = next(iter(loader))
    # assert frame_stacks.max() < 1.0 and frame_stacks.min() >= 0, "Greyscale values should be between 0 and 1"
    # preds = model(frame_stacks)
    # kl_div = nn.functional.kl_div(preds, saliency_maps)
    # auc = saliency_auc(preds.exp(), saliency_maps)
    # preds = preds.exp().detach()

    # # Read ms_pacman_52_RZ_2394668_Aug-10-14-52-42_preds.np and compare it to ground truth
    # with open("ms_pacman_52_RZ_2394668_Aug-10-14-52-42_preds.np", "rb") as f: 
    #     predicted_saliency_maps = np.load(f)
    # saliency_maps = dataset.data["saliency_map"]
    # saliency_maps = torch.stack(list(saliency_maps)).numpy()
