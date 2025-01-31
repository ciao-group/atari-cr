import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from atari_cr.graphs.common import results_df

def plot(hist: np.ndarray, out_path: str):
    plt.clf()
    plt.yscale("log")
    plt.ylim(1e-6, 2.)
    plt.bar(np.arange(0, 1025, 50), hist / hist.sum(), width=50)
    plt.savefig(out_path)

if __name__ == "__main__":
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)
    run_dir = "output/good_ray_runs/exp_2_3m_2025-01-30_15-24-36"

    results = (results_df(run_dir)
        .select("env", "seed", "duration_error", "gaze_duration")
        .sort("env", "seed")
    )
    for env in ["asterix", "seaquest", "hero"]:
        # Agent data
        durations = np.concat(
            [eval(s) for s in results.filter(pl.col("env") == env)["gaze_duration"]])
        hist, _ = np.histogram(durations, BINS)
        plot(hist, f"{output_dir}/{env}.png")

        # Human data
        hist = get_histogram(env)
        plot(hist.numpy(), f"{output_dir}/human_{env}.png")
