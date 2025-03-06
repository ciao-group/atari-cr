import os
import polars as pl

from atari_cr.graphs.common import Run, scatter_with_median

pl.Config.set_float_precision(4)


if __name__ == "__main__":
    out_dir = "output/graphs/fov_ablation"
    os.makedirs(out_dir, exist_ok=True)
    # results = Run("output/good_ray_runs/fov_pvm_2025-01-19_20-33-29",
    results = (
        Run("output/good_ray_runs/fov_pvm_v2_2025-02-10_17-05-35", with_trials=False)
            .results_df()
            .with_columns(
                pl.when(pl.col("fov") == "exponential")
                    .then(pl.lit("exp"))
                    .otherwise(pl.col("fov").cast(str))
                    .cast(pl.Enum(["window", "window_periph", "exp"]))
                    .alias("fov")
            )
    )

    scatter_with_median(results, ["fov", "pvm"], out_dir,
                        xlabels=["Fovea Type", "PVM Size"])

    for metric in ["human_likeness", "duration_error", "auc", "raw_reward"]:
        print(f"{metric}:")
        print(
            results
                .select("fov", "pvm", pl.col(metric).round(4))
                .sort("fov","pvm")
                .pivot("fov", index="pvm", values=metric, aggregate_function="mean")
        )

    metric = "duration_error"
    print("norm duration_error:")
    print(
        results
            .select("fov", "pvm", (1 - (1 + pl.col(metric)).log(10) / 3).round(4).alias(metric))
            .sort("fov","pvm")
            .pivot("fov", index="pvm", values=metric, aggregate_function="mean")
    )
