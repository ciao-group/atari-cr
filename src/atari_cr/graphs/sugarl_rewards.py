import os
import polars as pl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use("tableau-colorblind10")

    progress_files = [
        ("output/good_ray_runs/reproduction_1M_2024-12-31_13-40-46/lambda_asterix_7613c_00000_0_env=asterix_2024-12-31_13-40-48/progress.csv"),
        ("output/good_ray_runs/reproduction_1M_2024-12-31_13-40-46/lambda_hero_7613c_00002_2_env=hero_2024-12-31_13-40-48/progress.csv"),
        ("output/good_ray_runs/reproduction_1M_2024-12-31_13-40-46/lambda_seaquest_7613c_00001_1_env=seaquest_2024-12-31_13-40-48/progress.csv"),
        ("output/good_ray_runs/reproduction_4td_2025-01-03_18-20-47/lambda_asterix_12eff_00000_0_env=asterix_2025-01-03_18-20-48/progress.csv"),
        ("output/good_ray_runs/reproduction_4td_2025-01-03_18-20-47/lambda_hero_12eff_00002_2_env=hero_2025-01-03_18-20-48/progress.csv"),
        ("output/good_ray_runs/reproduction_4td_2025-01-03_18-20-47/lambda_seaquest_12eff_00001_1_env=seaquest_2025-01-03_18-20-48/progress.csv"),
    ]

    out_dir = "output/graphs/rewards"
    os.makedirs(out_dir, exist_ok=True)

    rewards = []
    for file in progress_files:
        rewards.append(
            pl.scan_csv(file, ignore_errors=True, null_values=["null"],
                schema_overrides={ "raw_reward": float })
            # .filter("eval_env")
            .tail(50)
            .select("raw_reward")
            .mean()
            .collect()
            .item()
        )
    r = (
        pl.DataFrame({
            "env": ["asterix", "hero", "seaquest", "asterix", "hero", "seaquest"],
            "td": [1,1,1,4,4,4],
            "rewards": rewards,
        })
        .pivot(index="env", columns="td", values="rewards")
        .with_columns((pl.col("4") - pl.col("1")).alias("4(%)") / pl.col("1"))
    )
    print(r)
    print("Mean increase when going to TD=4", r["4(%)"].mean())

    for ymax, label, file in zip(
        [650, 6100, 210, 650, 6100, 210],
        ["asterix" , "hero", "seaquest", "asterix_4td" , "hero_4td", "seaquest_4td"],
        progress_files):

        rewards = (
            pl.scan_csv(file, ignore_errors=True, null_values=["null"],
                schema_overrides={ "raw_reward": float })
            .select("raw_reward")
            .with_columns(pl.col("raw_reward").rolling_mean(100).alias("smoothed"))
            .collect()
        )

        plt.rcParams.update({'font.size': 28})
        plt.figure(figsize=(10, 10))
        plt.ylim(top=ymax)
        plt.subplots_adjust(left=0.125, right=0.985, top=0.95, bottom=0.015)
        plt.xticks([])
        plt.title(label)
        plt.plot(rewards["raw_reward"], color="orange", alpha=0.3)
        plt.plot(rewards["smoothed"], color="orange")
        plt.savefig(os.path.join(out_dir, f"{label}.png"))
        plt.clf()
