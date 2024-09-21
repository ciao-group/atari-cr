import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from PIL import Image

from atari_cr.common.models import RecordBuffer
from atari_cr.common.utils import grid_image

# Screen Size in visual degrees: 44,6 x 28,5
# Visual Degrees per Pixel with 84 x 84 pixels: 0,5310 x 0,3393
VISUAL_DEGREE_SCREEN_SIZE = (44.6, 28.5)

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
    sigmas *= 2.5
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
    data: RecordBuffer = torch.load(os.path.join(recordings_path, file), weights_only=False)

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

def saliency_auc(saliency_map1: torch.Tensor, saliency_map2: torch.Tensor, device: torch.device):
    """
    Predicts the AUC between two greyscale saliency maps by comparing their most salient pixels each

    :param Tensor[WxH] saliency_map1: First saliency map
    :param Tensor[WxH] saliency_map2: Second saliency map
    """
    # Flatten both maps
    saliency_map1 = saliency_map1.flatten().to(device)
    saliency_map2 = saliency_map2.flatten().to(device)
    assert saliency_map1.shape == saliency_map2.shape, "Saliency maps need to have the same size"

    # Get saliency maps containing only the most salient pixels for every 5% percentile
    percentiles = torch.arange(0.05, 1., 0.05).to(device)
    saliency_map1_thresholds = torch.quantile(saliency_map1, percentiles).view(-1, 1).expand(-1, len(saliency_map1))
    saliency_map2_thresholds = torch.quantile(saliency_map2, percentiles).view(-1, 1).expand(-1, len(saliency_map2))
    thresholded_saliency_maps1 = saliency_map1.expand([len(percentiles), -1]) >= saliency_map1_thresholds
    thresholded_saliency_maps2 = saliency_map2.expand([len(percentiles), -1]) >= saliency_map2_thresholds

    return BinaryAccuracy().to(device)(thresholded_saliency_maps1, thresholded_saliency_maps2)