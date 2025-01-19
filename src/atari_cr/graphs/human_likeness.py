import os
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from atari_cr.graphs.common import results_df


if __name__ == "__main__":
    run_dir = "output/good_ray_runs/human_2025-01-15_14-07-46"
    out_dir = "output/graphs/human_likeness"
    os.makedirs(out_dir, exist_ok=True)
    results = results_df(run_dir)
    cmap = plt.get_cmap("viridis", 3)

    for metric in ["pause_cost", "saccade_cost_scale"]:
        plt.clf()
        plt.xscale("log")
        handles = []
        envs = ["asterix", "seaquest", "hero"]
        for i, env in enumerate(envs):
            result = results.filter(pl.col("env") == env)
            plt.scatter(
                result[metric], result["human_likeness"],
                c=[cmap(i)]*9
            )
        mean = results.group_by(metric).mean().sort(metric)
        plt.plot(
            mean[metric], mean["human_likeness"],
            c="C3"
        )
        plt.legend([*envs, "mean"])
        plt.savefig(f"{out_dir}/{metric}.png")
