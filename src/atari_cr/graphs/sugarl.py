from atari_cr.graphs.common import Run

run = Run("output/good_ray_runs/sugarl_2025-02-07_15-58-26")
results = run.results_df()
# Average over both seeds
results = (results
    .select("ignore_sugarl", "raw_reward")
    .group_by("ignore_sugarl").median()
)
print(results)
