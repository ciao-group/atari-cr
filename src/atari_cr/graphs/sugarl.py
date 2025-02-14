from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import os

from atari_cr.graphs.common import CMAP, Run

out_dir = "output/graphs/sugarl_ablation"
os.makedirs(out_dir, exist_ok=True)
run = Run("output/good_ray_runs/sugarl_2025-02-07_15-58-26")
# results = run.results_df()
progress = run.progress_df()
# Average over both seeds
progress = (progress
    .select("ignore_sugarl", "env", "raw_reward", "human_likeness", "timestep")
    .sort("timestep")
)

for metric in ["raw_reward", "human_likeness"]:
    for env in progress["env"].unique():
        for i, sugarl in enumerate([True, False]):
            score = progress.filter(pl.col("env") == env)
            plt.clf()
            plt.ylim(0,1 if metric == "human_likeness" else 1.05 * score[metric].max())
            score = score.filter(pl.col("ignore_sugarl") != sugarl)[metric]

            mean = score.rolling_mean(100)
            x = np.arange(len(score)) * 1e6 / len(score)
            plt.plot(x, mean, color=CMAP[1])
            plt.scatter(x, score, marker=".", alpha=0.25)
            plt.savefig(f"{out_dir}/{metric}_{env}_{'' if sugarl else 'no_'}sugarl.png")

results = run.results_df()
print(
    results
        .select("ignore_sugarl","env","raw_reward")
        .group_by("env","ignore_sugarl").median()
        .sort("env","ignore_sugarl","raw_reward")
)
