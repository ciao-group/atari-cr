import json
import os
import polars as pl
from matplotlib import colormaps, pyplot as plt
from matplotlib.category import UnitData


CMAP = colormaps["viridis"]

best_auc_run = ("output/good_ray_runs/human_2025-01-15_14-07-46/"
    "b7cec_00002_2_env=hero,pause_cost=0.0000,saccade_cost_scale=0.0000_2025-01-15_14-07-48")
best_auc = {
    "run": best_auc_run,
    "eval": f"{best_auc_run}/seed0_step3000005_eval00",
    "env": "hero",
}

def results_df(run_dir: str):
    """ Get a DataFrame containing the results of a run. """
    results = []
    for dir in [e for e in os.scandir(run_dir) if e.is_dir()]:
        with open(f"{dir.path}/params.json", "r") as f:
            params = json.load(f)
        params.update(params.pop("searchable"))
        result = pl.read_csv(f"{dir.path}/progress.csv").row(-1, named=True)
        result.update(params)
        results.append(result)
    return pl.DataFrame(results).with_columns([
        pl.col("env").cast(pl.Enum(["asterix", "seaquest", "hero"])),
        pl.col("fov").cast(pl.Enum(["window", "window_periph", "exponential"]))
    ])

def scatter_with_mean(results_df: pl.DataFrame, metrics: list[str], out_dir: str,
                      target_metric = "human_likeness", log_x=False):
    """ Create and save a scatter plot with a mean drawn into it """
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    results_df = results_df.select(["env", *metrics, target_metric])
    for metric in metrics:
        plt.clf()
        plt.xscale("log" if log_x else "linear")
        envs = ["asterix", "seaquest", "hero"]
        for i, env in enumerate(envs):
            env_result = results_df.filter(pl.col("env") == env)
            plt.scatter(
                env_result[metric], env_result[target_metric],
                color=cmap[i]
            )
        median = results_df.group_by(metric).median().sort(metric)
        plt.plot(metric, target_metric, data=median, color=cmap[i+1],
            xunits=UnitData(median[metric].to_list())
            if median[metric].dtype == pl.Enum else None)
        plt.ylim(results_df[target_metric].min() * 0.95,
                 results_df[target_metric].max() * 1.05)
        plt.xticks(median[metric].to_list())
        plt.legend([*envs, "median"])
        plt.savefig(f"{out_dir}/{metric}.png")
