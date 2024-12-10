import os
import polars as pl
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":
    search_output_path = \
        "/home/niko/Repos/atari-cr/output/ray_results/lambda_2024-12-04_13-51-31"

    # Collect dicts with the last reported evaluation to ray and the params of the trial
    eval_results = []
    for trial_name in [e.path for e in os.scandir(search_output_path)
                       if (e.is_dir() and e.name.startswith("lambda"))]:
        eval_result = (
            pl.scan_csv(
                os.path.join(trial_name, "progress.csv"),
                schema_overrides={
                    "raw_reward": pl.Float32,
                    "reward": pl.Float32,
                    "truncated": pl.Float32,
                    "pauses": pl.Float32,
                    "prevented_pauses": pl.Float32,
                    "no_action_pauses": pl.Float32,
                })
            .tail(1)
            .collect()
            .row(0, named=True))
        with open(os.path.join(trial_name, "params.json"), "r") as f:
            params = json.load(f)
        eval_result.update(params)
        eval_results.append(eval_result)
    eval_results = pl.DataFrame(eval_results)

    # Plot a line for each fovea for score(pvm) and auc(pvm)
    fig, ax = plt.subplots(1, 1)
    ax.plot(*eval_results.filter(pl.col("fov") == "windowed")["pvm_stack", "reward"])
    ax.plot(*eval_results.filter(pl.col("fov") == "exponential")["pvm_stack", "reward"])
    fig.savefig("debug.png")
