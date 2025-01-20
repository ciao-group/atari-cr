import os
from matplotlib import pyplot as plt
import torch
import polars as pl
from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.atari_head.gaze_predictor import GazePredictor

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)

    envs = ["asterix", "seaquest", "hero"]
    for env in envs:
        # Plot the eval performance
        checkpoint_dir = f"output/atari_head/{env}/300"
        out_dir = "output/graphs/gaze_pred_showcase/eval"
        os.makedirs(out_dir, exist_ok=True)
        eval = (pl.read_csv(f"{checkpoint_dir}/eval.csv")
            .filter(~(pl.col("model") == "Gaze Predictor (Train)")))
        plt.clf()
        plt.tight_layout()
        plt.scatter(eval["model"], eval["auc"])
        plt.savefig(f"{out_dir}/{env}.png")

    for env in envs:
        # Load Model
        checkpoint_dir = f"output/atari_head/{env}/300"
        predictor = GazePredictor.load(f"{checkpoint_dir}/checkpoint.pth")
        predictor.model.eval()

        # Load Data
        dataset = GazeDataset.from_atari_head_files(
            "data/Atari-HEAD/" + env, single_trial=True)
        _, val_loader = dataset.split(3)
        frame_stacks, saliency_maps = next(iter(val_loader))
        with torch.no_grad():
            preds = predictor.model(
                frame_stacks.to(next(predictor.model.parameters()).device)).exp().cpu()

        # Plot Graph
        fig, axs = plt.subplots(3,3, figsize=(3,3))
        for i in range(3):
            for j in range(3):
                axs[j,i].axis("off")
                axs[j,i].imshow(frame_stacks[i,-1])
            axs[1,i].imshow(saliency_maps[i], cmap="jet", alpha=0.8)
            axs[2,i].imshow(preds[i], cmap="jet", alpha=0.8)
        fig.tight_layout()

        # Save it
        out_dir = "output/graphs/gaze_pred_showcase"
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{env}.png")
