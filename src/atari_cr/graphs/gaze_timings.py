""" Comparison of gaze timings within the episode between an agent and Atari-HEAD """
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from atari_cr.graphs.common import CMAP, Run

def time_bins(pauses: list[bool], n_bins = 20):
    """ Get a histogram of average pauses per time step.

    :param list[bool] pauses: List containing 1 for a pause and 0 for no pause.
    """
    if np.sum(pauses) == 0:
        return np.zeros(n_bins)

    bin_size = len(pauses) // n_bins
    residual = len(pauses) % n_bins
    bin_start = 0
    bins = np.empty(n_bins)
    for i in range(n_bins):
        bin_end = bin_start + bin_size + (1 if i < residual else 0)
        bins[i] = np.sum(pauses[bin_start:bin_end])
        bin_start = bin_end
    # Count pause probability in every bin, instead of total pauses
    return n_bins * bins / len(pauses)

def plot(hist: np.ndarray, out_path: str, color: str):
    width = 1
    plt.clf()
    plt.bar(np.arange(20) + 0.5 * width, hist, width, label='Human', color=color)
    plt.xticks([0, 20], ["start", "end"])
    plt.savefig(out_path)

# Get agent histogram
run = Run("output/good_ray_runs/exp_2_3m_2025-01-30_15-24-36")
output_dir = "output/graphs/gaze_timings"
os.makedirs(output_dir, exist_ok=True)

env_labels = [t.args["env"] for t in run.trials]
histograms = [time_bins(t.record().annotations["pauses"].to_numpy())
    for t in run.trials]

env_histograms = []
human_histograms = []
envs = ["asterix", "seaquest", "hero"]
for i, env in enumerate(envs):
    # Agent plots
    indices = np.where(np.array(env_labels) == env)[0]
    # Sum up the histograms for one env
    env_histogram = histograms[indices[0]]
    for j in indices[1:]:
        env_histogram += histograms[j]
    # Append the summed histogram to a list of summed histograms
    env_histograms.append(env_histogram)
    plot(env_histogram, f"{output_dir}/{env}.png", CMAP[i])

    # Human plots
    csv_files = [e.path for e in os.scandir(f"data/Atari-HEAD/{env}")
        if e.path.endswith(".csv")]
    pause_times = np.concat([
        pl.read_csv(f, null_values="null")
            .select(pl.col("duration(ms)"))
            .to_numpy()[:-1]
        for f in csv_files # Every file is one episode
    ])
    # Count durations >= 55 ms as a pause
    pauses = np.maximum(0, pause_times.flatten() - 55).astype(bool)
    # Get 20 bins of roughly equal size
    human_hist = time_bins(pauses)
    human_histograms.append(human_hist)
    plot(human_hist, f"{output_dir}/human_{env}.png", CMAP[i])

print("Average number of pauses per time step:")
print(pl.DataFrame({
    "Env": envs,
    "Agent": [np.round(h.mean(), 3) for h in env_histograms],
    "Human": [np.round(h.mean(), 3) for h in human_histograms],
}))
