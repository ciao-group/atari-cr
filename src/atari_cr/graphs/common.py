import json
import os
import polars as pl
from matplotlib import colormaps


CMAP = colormaps["viridis"]
HEATMAP_COLOR = colormaps["jet"]

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
