import os
import polars as pl
import json
import matplotlib.pyplot as plt


if __name__ == "__main__":
    search_output_path = \
        "/home/niko/Repos/atari-cr/output/good_ray_runs/lambda_2024-11-29_17-45-30"

    # Collect dicts with the last reported evaluation to ray and the params of the trial
    eval_results = []
    for trial_name in [e.path for e in os.scandir(search_output_path) if e.is_dir()]:
        eval_result = (pl.scan_csv(os.path.join(trial_name, "progress.csv"))
            .tail(1)
            .collect()
            .row(0, named=True))
        params = json.load(os.path.join(trial_name, "params.json"))
        eval_result.update(params)
        eval_results.append(eval_result)
    eval_results = pl.DataFrame(eval_results)

    # Plot a line for each fovea for score(pvm) and auc(pvm)
    fig, ax = plt.subplots(1, 1)
    ax.plot(*eval_results.filter(pl.col("fov") == "windowed")["pvm_stack", "reward"])
    ax.plot(*eval_results.filter(pl.col("fov") == "exponential")["pvm_stack", "reward"])
    fig.show()
