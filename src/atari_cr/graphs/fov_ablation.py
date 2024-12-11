import os
import polars as pl
import json
import matplotlib.pyplot as plt
from atari_cr.graphs.colors import COLORBLIND


if __name__ == "__main__":
    search_output_path = \
        "/home/niko/Repos/atari-cr/output/good_ray_runs/fov_pvm_1m_2024-12-05_10-54-02"

    # Collect dicts with the last reported evaluation to ray and the params of the trial
    eval_results = []
    for trial_name in [e.path for e in os.scandir(search_output_path)
                       if (e.is_dir() and e.name.startswith("lambda"))]:

        # Take the last episode reward as for every agent
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
            .row(0, named=True)
        )

        # Add the agent's hyperparams to the results
        with open(os.path.join(trial_name, "params.json"), "r") as f:
            params = json.load(f)
        eval_result.update(params)

        eval_results.append(eval_result)
    eval_results = pl.DataFrame(eval_results)
    eval_results = eval_results.sort("pvm_stack")

    # Plot the agent's results (reward, auc) against its hyperparams (fovea and pvm)
    # One line in the dataframe is one agent
    plt.style.use("tableau-colorblind10")
    def plot(metric):
        plt.clf() # Reset the plot
        plt.plot(*eval_results.filter(pl.col("fov") == "window")["pvm_stack", metric],
                color=COLORBLIND["orange"], label="Window")
        plt.plot(*eval_results
                    .filter(pl.col("fov") == "exponential")["pvm_stack", metric],
                color=COLORBLIND["blue"], label="Exponential")
        plt.legend()
        plt.xlabel("Memorized timesteps")
        match metric:
            case "reward": plt.ylabel("Episode Reward")
            case "auc": plt.ylabel("Human-likeness AUC")
            case "duration_error": plt.ylabel("Gaze distribution differenece to humans")
        plt.savefig(f"output/graphs/fov_ablation/{metric}.png")

    plot("reward")
    plot("auc")
    plot("duration_error")
