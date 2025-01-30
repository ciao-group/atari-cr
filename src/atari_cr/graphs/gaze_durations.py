import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from atari_cr.graphs.colors import COLORBLIND

from atari_cr.graphs.common import progress_df, results_df

def plot(hist, agent = False, label = "", log=False):
    plt.clf()
    hist: np.ndarray = hist / hist.sum()

    # Only text annotation for the highest value
    x = np.arange(0, 1025, 50)
    if log:
        plt.yscale("log")
    else:
        highest = hist.max()
        highest_idx = hist.argmax()
        hist[highest_idx] = 0
        second_highest = hist.max() or highest
        plt.text(x[highest_idx], second_highest * 1.025, f"{highest:.3f}", ha='center',
                va='bottom')
        plt.ylim(0, second_highest * 1.1)

    if not log:
        # Put a single bar for the highest value
        y = np.zeros(21)
        y[highest_idx] = highest
        plt.bar(x, y, width=50, color=COLORBLIND["orange"])

    # Normal plot
    plt.bar(x, hist, width=50, color=COLORBLIND["blue"])

    # Styling
    # plt.rcParams.update({'font.size': 4})
    plt.xticks(np.arange(0, 1050, 200))
    # plt.title(("Agent" if agent else "Human") + " frame durations for " + game_name)
    plt.xlabel("Duration[ms]")

    plt.savefig(f"{output_dir}/{game_name}_{label}_agent.png" if agent \
        else f"{output_dir}/{game_name}.png")


if __name__ == "__main__":
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)
    WITH_AGENT = False
    run_dir = "output/good_ray_runs/fov_pvm_2025-01-19_20-33-29"
    # run_dir = "output/good_ray_runs/rewards_5M_2025-01-28_15-02-17"

    # results = (results_df(run_dir)
    #     # .sort("fov", "pvm")
    #     .select("fov", "pvm", "duration_error", "gaze_duration")
    # )
    progress = (progress_df(run_dir)
        .select("fov", "pvm", "timestep", "duration_error")
        .group_by("fov", "pvm", "timestep").mean()
        .sort("fov", "pvm")
    )
    pass