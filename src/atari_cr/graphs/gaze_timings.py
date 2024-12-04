""" Comparison of gaze timings within the episode between an agent and Atari-HEAD """
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import polars as pl

from atari_cr.models import EpisodeRecord

def records_path(run_name: str, env_name: str = "ms_pacman"):
    """ Get the path of the records for a ray run given by its name """
    run_path = subprocess.run(["find", "/tmp/ray", "-name", run_name],
                          capture_output=True, text=True).stdout[:-1] \
        + "/working_dirs"
    trial_records = [f"{e.path}/output/runs/tuning/{env_name}/recordings"
                     for e in os.scandir(run_path)]
    return trial_records

def time_bins(pauses: list[float], n_bins = 20):
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
    return bins / np.sum(bins)

env_name = "ms_pacman"

# Get agent histogram
quant_search_records = records_path("lambda_2024-12-03_13-00-04")
q8_record0 = [r for r in quant_search_records if "quantization=8" in r][0]
q8_record0 = next(os.scandir(q8_record0)).path # First eval round
q8_record0 = EpisodeRecord.load(q8_record0)
agent_hist = time_bins(q8_record0.annotations["pauses"].to_numpy())

# Get Atari HEAD histogram
csv_files = [e.path for e in os.scandir(f"data/Atari-HEAD/{env_name}")
             if e.path.endswith(".csv")]
pause_times = np.concat([pl.read_csv(f, null_values="null")
                         .select(pl.col("duration(ms)"))
                         .to_numpy()[:-1] for f in csv_files])
# Only count additional times per frame, 50ms is what every frame takes at least
pause_times = np.maximum(0, pause_times - 50)

# Get 20 bins of roughly equal size
human_hist = time_bins(pause_times)

# Plot
x = np.arange(20)
width = 0.4
plt.bar(x + 0.5 * width, human_hist, width, label='Human', color='blue')
plt.bar(x + 1.5 * width, agent_hist, width, label='Agent', color='orange')
plt.xticks([])
plt.yticks([])
plt.xlabel("Time since episode start")
plt.ylabel("Pause time")
plt.title("Pause timings within one episode")
plt.legend()
plt.savefig("debug.png")
