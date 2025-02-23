import os
import polars as pl
from itertools import product
from matplotlib import pyplot as plt

from atari_cr.graphs.common import CMAP, Run

if __name__ == "__main__":
    run = Run("output/good_ray_runs/rewards_5M_2025-01-28_15-02-17")
    results = run.results_df(ignore_durations=True)
    # Average over both seeds
    results = (results
        .group_by("env", "total_timesteps", "use_pause_env").mean()
        .sort("use_pause_env", "env", "total_timesteps")
        .select("use_pause_env", "env", "total_timesteps", "raw_reward")
    )
    print(results)

    progress = run.progress_df()
    progress = (progress
        .group_by("env", "total_timesteps", "timestep").mean()
        .sort("env", "total_timesteps", "timestep")
        .select("env", "total_timesteps", "timestep", "raw_reward")
    )

    out_dir = "output/graphs/rewards_5m"
    os.makedirs(out_dir, exist_ok=True)
    for env, total_timesteps in product(
        ["asterix", "seaquest", "hero"],
        [1_000_000, 5_000_000]
    ):
        trial = (progress
            .filter(
                (pl.col("env") == env) &
                (pl.col("total_timesteps") == total_timesteps)
            ).with_columns(pl.col("raw_reward").rolling_mean(100).alias("windowed_reward"))
        )
        plt.clf()
        plt.scatter(*trial["timestep", "raw_reward"], marker=".", alpha=0.25)
        plt.plot(*trial["timestep", "windowed_reward"], color=CMAP[1])
        plt.savefig(f"{out_dir}/{env}_{int(total_timesteps / 1000000)}m.png")
