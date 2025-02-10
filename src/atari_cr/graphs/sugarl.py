from matplotlib import pyplot as plt
import polars as pl
import os

from atari_cr.graphs.common import Run

out_dir = "output/graphs/sugarl_ablation"
os.makedirs(out_dir, exist_ok=True)
run = Run("output/good_ray_runs/sugarl_2025-02-07_15-58-26")
results = run.results_df()
# Average over both seeds
results = results.select("ignore_sugarl", "env", "raw_reward", "human_likeness"
    # .group_by("ignore_sugarl").median()
)

for metric in ["raw_reward", "human_likeness"]:
    for env in results["env"].unique():
        rewards = results.filter(pl.col("env") == env)
        rewards = [
            results
                .filter(pl.col("ignore_sugarl") != sugarl)
                .filter(pl.col("env") == env)
                [metric]
            for sugarl in [True, False]
        ]
        plt.clf()
        plt.boxplot(rewards, tick_labels=["sugarl", "no_sugarl"])
        plt.savefig(f"{out_dir}/{metric}_{env}.png")
