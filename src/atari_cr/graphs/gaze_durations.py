import os
from atari_cr.atari_head.durations import BINS, get_histogram
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from atari_cr.graphs.colors import COLORBLIND

def plot(hist, agent = False, label = ""):
    plt.clf()
    hist: np.ndarray = hist / hist.sum()

    # Only text annotation for the highest value
    x = np.arange(0, 1025, 50)
    highest = hist.max()
    highest_idx = hist.argmax()
    hist[highest_idx] = 0
    second_highest = hist.max() or highest
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

    plt.savefig(f"{output_dir}/{game_name}_{label}_agent.png" if agent \
        else f"{output_dir}/{game_name}.png")


if __name__ == "__main__":
    game_name = "ms_pacman"
    output_dir = "output/graphs/histograms"
    os.makedirs(output_dir, exist_ok=True)

    agent_trials = {
        "pause": "output/good_ray_runs/timed_env_2024-12-12_22-20-31/"
            "lambda_ms_pacman_ebebf_00002_2_seed=0,timed_env=False_2024-12-12_22-20-33",
        "timed": "output/good_ray_runs/timed_env_2024-12-12_22-20-31/"
            "lambda_ms_pacman_ebebf_00001_1_seed=1,timed_env=True_2024-12-12_22-20-33"
    }
    for label, trial_path in agent_trials.items():
        agent_durations = eval(
            pl.scan_csv(os.path.join(trial_path, "progress.csv"))
            .tail(1)
            .select(pl.col("gaze_duration"))
            .collect()
            .item()
            .replace(".", ",")
        )
        hist = np.histogram(agent_durations, BINS)[0]
        plot(hist, agent=True, label=label)

    hist = get_histogram(game_name).numpy()
    plot(hist)
