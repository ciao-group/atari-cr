import json
import os
import polars as pl
from matplotlib import colormaps


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
    return pl.DataFrame(results).with_columns(pl.col("env").cast(
        pl.Enum(["asterix", "seaquest", "hero"])))
