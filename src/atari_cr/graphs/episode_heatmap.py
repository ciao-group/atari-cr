import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torch import Tensor

from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.graphs.common import best_auc
from atari_cr.models import EpisodeRecord

def save_heatmap(heatmap: Tensor, background: np.ndarray, label: str):
    heatmap = cv2.resize(heatmap.numpy(), (255, 255))
    plt.clf()
    plt.imshow(background)
    plt.imshow(heatmap, cmap="jet", alpha=0.75)
    plt.savefig(f"{output_dir}/{label}.png")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    output_dir = "output/graphs/episode_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # Best Agent heatmap
    best_episode = EpisodeRecord.load(best_auc["eval"])
    background = best_episode.frames[0]
    heatmap = GazeDataset.create_saliency_map(
        best_episode.annotations["fov_x", "fov_y"].to_torch())
    save_heatmap(heatmap, background, "agent")

    # Atari HEAD heatmap
    heatmap = GazeDataset.from_atari_head_files(
        f"data/Atari-HEAD/{best_auc['env']}", single_trial=True, load_saliency=True
    ).saliency.mean(dim=0)
    save_heatmap(heatmap, background, "human")

