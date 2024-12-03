from ray import train, tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

def tuning(config: dict):
    # Copy quantization value into the two corresponding values
    if "sensory_action_space_quantization" in config:
        quantization = config.pop("sensory_action_space_quantization")
        config["sensory_action_x_size"] = quantization
        config["sensory_action_y_size"] = quantization

    # Run the experiment
    args = ArgParser().from_dict(config)
    eval_returns, out_paths = main(args)

if __name__ == "__main__":
    GAZE_TARGET = False
    DEBUG = False
    GRID_SEARCH = True
    concurrent_runs = 3 if DEBUG else 4
    num_samples = 2 * concurrent_runs if DEBUG else 20
    time_steps = 500_000 if DEBUG else 5_000_000

    trainable = tune.with_resources(
        lambda config: tuning(config, DEBUG),
        {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

    param_space = {
        # Fixed
        "clip_reward": False,
        "capture_video": True,
        "total_timesteps": time_steps,
        "no_pvm_visualization": True,
        "use_pause_env": True,
        "env": "ms_pacman",
        "exp_name": "tuning",
        "learning_start": 5_000, # Instead of 80k to prevent masked actions faster
        "debug": DEBUG,
        "evaluator":
            "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.3/999/checkpoint.pth",
        # Already searched
        "action_repeat": 5,
        "fov_size": 20,
        "sensory_action_space_quantization": 4, # from 9-16
        "pvm_stack": 16, # from 9-16
        "saccade_cost_scale": 0.0002, # lowered from 0.0015 (9-16) for more pauses
        "pause_cost": 0.2, # from 9-16
        # Fixed overrides
        "pause_cost": 0., # make only saccade costs matter
        "pvm_stack": 3, # from sugarl code
        "saccade_cost_scale": 0.,
        "use_pause_env": False,
        # Searchable
        "seed": tune.grid_search([0, 1]),
        "sensory_action_space_quantization": tune.grid_search([4, 8, 16, 32]),
        # "pause_cost": tune.grid_search([-1e-3, 0, 1e-3]),
        # "fov": tune.choice([fov for fov in get_args(FovType) if fov != "gaussian"]),
    }

    metric, mode = ("human_error", "min") if GAZE_TARGET else ("raw_reward", "max")
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=None if GRID_SEARCH else num_samples,
            scheduler=None if GRID_SEARCH else ASHAScheduler(
                stop_last_trials=False
            ),
            search_alg=None if GRID_SEARCH else OptunaSearch(),
            metric=metric,
            mode=mode,
        ),
        run_config=train.RunConfig(
            storage_path="/home/niko/Repos/atari-cr/output/ray_results",
            stop=None if GRID_SEARCH else TrialPlateauStopper(
                metric, mode=mode, num_results=8, grace_period=1000),
        )
    )
    results = tuner.fit()
    print("Best result:\n", results.get_best_result().config)
