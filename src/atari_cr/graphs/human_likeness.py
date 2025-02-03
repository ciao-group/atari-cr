import os

from atari_cr.graphs.common import Run, scatter_with_mean


if __name__ == "__main__":
    out_dir = "output/graphs/human_likeness"
    os.makedirs(out_dir, exist_ok=True)
    results = Run("output/good_ray_runs/human_2025-01-15_14-07-46").results_df()
    scatter_with_mean(results, ["pause_cost", "saccade_cost_scale"], out_dir,
                      log_x=True)
