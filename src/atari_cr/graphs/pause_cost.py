import os

from atari_cr.graphs.common import Run, scatter_with_median


if __name__ == "__main__":
    out_dir = "output/graphs/pause_costs"
    os.makedirs(out_dir, exist_ok=True)
    results = (Run("output/good_ray_runs/human_2025-01-15_14-07-46", with_trials=False)
        .results_df(ignore_durations=True))
    scatter_with_median(results, ["pause_cost", "saccade_cost_scale"], out_dir,
                      log_x=True)
