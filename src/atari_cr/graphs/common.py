import json
import os
import subprocess
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.category import UnitData

from atari_cr.models import EpisodeRecord

# Styling
plt.rcParams.update({'font.size': 14}) # For two subfigure
plt.rcParams.update({'font.size': 21}) # For three subfigures

CMAP = plt.rcParams['axes.prop_cycle'].by_key()['color']

best_auc_trial = ("output/good_ray_runs/human_2025-01-15_14-07-46/"
    "b7cec_00002_2_env=hero,pause_cost=0.0000,saccade_cost_scale=0.0000_2025-01-15_14-07-48")
best_auc = {
    "run": best_auc_trial,
    "eval": f"{best_auc_trial}/seed0_step3000005_eval00",
    "env": "hero",
}

class Trial:
    def __init__(self, name: str, record_path: str):
        self.name = name
        self.record_path = record_path
        self.args = dict(x.split("=") for x in name.split("_")[-3].split(","))

    def record(self):
        """ Get the record of the first evaluation. """
        return EpisodeRecord.load(next(os.scandir(self.record_path)).path)

class Run:
    """
    :attr str path: Path to the dir containing the run.
        Example: `output/good_ray_runs/exp_2_3m_2025-01-30_15-24-36`
    """
    def __init__(self, path: str, with_trials=False):
        self.path = path
        if with_trials:
            self.trials = self._trials()

    def results_df(self, ignore_durations = False):
        """ Get a DataFrame containing the results of a run. """
        results = []
        for dir in [e for e in os.scandir(self.path) if e.is_dir()]:
            with open(f"{dir.path}/params.json", "r") as f:
                params = json.load(f)
            params.update(params.pop("searchable"))
            # Average over the last 30 episodes (like mnih2015), including eval episodes
            result_df = pl.read_csv(f"{dir.path}/progress.csv").tail(30)
            result = result_df.mean().row(0, named=True)
            result.update(params)
            if not ignore_durations:
                durations, emma_times = [], []
                for d in result_df["gaze_duration"].to_list():
                    durations.extend(eval(d))
                result.update({"gaze_duration": durations})
                for t in result_df["emma_times"].to_list():
                    emma_times.extend(eval(t))
                result.update({"emma_times": emma_times})
            results.append(result)
        return pl.DataFrame(results).with_columns([
            pl.col("env").cast(pl.Enum(["asterix", "seaquest", "hero"])),
            pl.col("fov").cast(pl.Enum(["window", "window_periph", "exponential"]))
        ])

    def progress_df(self) -> pl.DataFrame:
        """ Get a DataFrame containing every timestep of the trials of a run. """
        results = []
        for dir in [e for e in os.scandir(self.path) if e.is_dir()]:
            with open(f"{dir.path}/params.json", "r") as f:
                params = json.load(f)
            params.update(params.pop("searchable"))
            result = (pl.read_csv(f"{dir.path}/progress.csv")
                # Add the params as columns
                .with_columns([pl.lit(v).alias(k) for k,v in params.items()])
                # Cast string columns to enum type
                .with_columns([
                    pl.col("env").cast(pl.Enum(["asterix", "seaquest", "hero"])),
                    pl.col("fov").cast(pl.Enum(["window", "window_periph", "exponential"]))
                ]))
            results.append(result)
        return pl.concat(results, how="vertical")

    def _trials(self):
        """ Get a path for the records of every trial in the run. """
        run_date = "_".join(self.path.split("_")[-2:])
        run_path = subprocess.run(
            ["find", "/tmp/ray", "-name", f"*{run_date}*"],
            capture_output=True, text=True
        ).stdout.split("\n")[1] + "/working_dirs"
        env = os.listdir(next(os.scandir(run_path)).path + "/output/runs/tuning")[0]

        trials = []
        for e in os.scandir(run_path):
            env = os.listdir(e.path + "/output/runs/tuning")[0]
            trials.append(Trial(
                e.name,
                f"{e.path}/output/runs/tuning/{env}/recordings"
            ))
        return trials

def scatter_with_median(results_df: pl.DataFrame, metrics: list[str], out_dir: str,
                      target_metric = "human_likeness", log_x=False, xlabels=None):
    """ Create and save a scatter plot with a mean drawn into it """
    results_df = results_df.select(["env", *metrics, target_metric])
    for j, metric in enumerate(metrics):
        plt.clf()
        plt.xscale("log" if log_x else "linear")
        envs = ["asterix", "seaquest", "hero"]
        for i, env in enumerate(envs):
            env_result = results_df.filter(pl.col("env") == env)
            plt.scatter(
                env_result[metric], env_result[target_metric],
                color=CMAP[i], alpha=0.5
            )
        median = results_df.group_by(metric).median().sort(metric)
        plt.plot(metric, target_metric, data=median, color=CMAP[i+1],
            xunits=UnitData(median[metric].to_list()) if median[metric].dtype == pl.Enum
                else None, alpha=0.5)
        plt.ylim(results_df[target_metric].min() * 0.95,
                 results_df[target_metric].max() * 1.05)
        plt.xticks(median[metric].to_list())
        plt.legend([*envs, "median"], loc='best', bbox_to_anchor=(0.55, 0.5, 0.45, 0.5))
        plt.xlabel(xlabels[j] if xlabels
            else " ".join([w.capitalize() for w in metric.split("_")]))
        plt.ylabel(" ".join([w.capitalize() for w in target_metric.split("_")]))
        plt.savefig(f"{out_dir}/{metric}.png", bbox_inches='tight', pad_inches=0.1)
        print(f"Medians for {metric}:")
        print(median.select([metric, target_metric]))
