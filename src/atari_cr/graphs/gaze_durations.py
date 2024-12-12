import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from atari_cr.graphs.colors import COLORBLIND

if __name__ == "__main__":
    agent = True
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)

    trial_path = ("output/ray_results/lambda_2024-12-12_18-53-56/"
              "lambda_ms_pacman_0fb0b_00000_0_seed=0,timed_env=True_2024-12-12_18-53-57")
    agent_durations = eval(
        pl.scan_csv(os.path.join(trial_path, "progress.csv"))
        .tail(1)
        .select(pl.col("gaze_duration"))
        .collect()
        .item()
        .replace(" ", ",")
    )

    x = np.arange(0, 1025, 50)
    hist = np.histogram(agent_durations, BINS)[0] if agent \
        else get_histogram(game_name).numpy()
    hist: np.ndarray = hist / hist.sum()

    # Only text annotation for the highest value
    highest = hist.max()
    highest_idx = hist.argmax()
    hist[highest_idx] = 0
    second_highest = hist.max()
    plt.text(x[highest_idx], second_highest * 1.025, f"{highest:.3f}", ha='center',
            va='bottom')
    plt.ylim(0, second_highest * 1.1)

    # Put a single bar for the highest value
    y = np.zeros(21)
    y[highest_idx] = highest
    plt.bar(x, y, width=50, color=COLORBLIND["orange"])

    # Normal plot
    plt.bar(x, hist, width=50, color=COLORBLIND["blue"])

    # Styling
    plt.xticks(np.arange(0, 1050, 100))
    plt.title(("Agent" if agent else "Human") + " frame durations per episode")
    plt.xlabel("Duration[ms]")

    plt.savefig(f"{output_dir}/{game_name}_agent.png" if agent \
        else f"{output_dir}/{game_name}.png")
