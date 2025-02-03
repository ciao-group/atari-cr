import os

from atari_cr.graphs.common import Run, scatter_with_mean


if __name__ == "__main__":
    out_dir = "output/graphs/fov_ablation"
    os.makedirs(out_dir, exist_ok=True)
    results = Run("output/good_ray_runs/fov_pvm_2025-01-19_20-33-29").results_df()
    scatter_with_mean(results, ["fov", "pvm"], out_dir)
