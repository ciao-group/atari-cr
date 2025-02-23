import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from atari_cr.graphs.common import Run, CMAP

def plot(hist: np.ndarray, out_path: str, color: str):
    plt.clf()
    plt.yscale("log")
    plt.ylim(1e-6, 2.)
    plt.bar(np.arange(0, 1025, 50), hist / hist.sum(), width=50, color=color)
    plt.savefig(out_path)

if __name__ == "__main__":
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)
    run_dir = "output/good_ray_runs/exp_2_3m_2025-01-30_15-24-36"

    results = (Run(run_dir).results_df()
        .select("env", "seed", "duration_error", "gaze_duration")
        .sort("env", "seed")
    )
    envs = ["asterix", "seaquest", "hero"]
    pause_probs = pl.DataFrame({
        "Player": ["Agent", "Human"],
        **{ e: [-1.,-1.] for e in envs}
    })
    for i, env in enumerate(envs):
        # Agent data
        durations = np.concat(results.filter(pl.col("env") == env)["gaze_duration"])
        hist, _ = np.histogram(durations, BINS)
        pause_probs[0, env] = (1 - (hist / hist.sum())[1].round(3)) * 100
        plot(hist, f"{output_dir}/{env}.png", color=CMAP[i])

        # Human data
        hist = get_histogram(env)
        pause_probs[1, env] = (1 - hist[1].numpy().round(3)) * 100
        plot(hist.numpy(), f"{output_dir}/human_{env}.png", color=CMAP[i])

# How often do people and humans pause
print(pause_probs)
