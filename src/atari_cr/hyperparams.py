from ray import train, tune

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.experiment.trial import Trial

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

def trainable(config: dict):
    # Copy quantization value into the two corresponding values
    if "sensory_action_space_quantization" in config:
        quantization = config.pop("sensory_action_space_quantization")
        config["sensory_action_x_size"] = quantization
        config["sensory_action_y_size"] = quantization

    # Log the trial name
    searchable = config.pop("searchable")
    print("Trial: " +
          ",".join([f"{k}={v}" for k,v in searchable.items()]))
    config.update(searchable)

    # Start training
    args = ArgParser().from_dict(config)
    eval_returns, out_paths = main(args)

if __name__ == "__main__":
    GAZE_TARGET = True
    DEBUG = False
    GRID_SEARCH = True
    concurrent_runs = 3
    num_samples = 2 * concurrent_runs if DEBUG else 90
    time_steps = 500_000 if DEBUG else 3_000_000

    trainable = tune.with_resources(
        trainable,
        {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

    metric, mode = ("human_likeness", "max") if GAZE_TARGET else ("raw_reward", "max")
    tuner = tune.Tuner(
        trainable,
        param_space={
            # Fixed
            "clip_reward": False,
            "capture_video": True,
            "total_timesteps": time_steps,
            "no_pvm_visualization": True,
            "use_pause_env": True,
            "env": "seaquest", # Other: breakout ms_pacman seaquest asterix hero
            "exp_name": "tuning",
            "learning_start": 80_000,
            "debug": DEBUG,
            "evaluator":
                "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.3/999/checkpoint.pth",
            "pvm_stack": 3, # from sugarl code
            "fov": "exponential",
            "periph": False,
            "timed_env": True,
            "gamma": 0.99,
            # Already searched
            "action_repeat": 5,
            "sensory_action_space_quantization": 8, # from 12-03
            "saccade_cost_scale": 0.0002, # lowered from 0.0015 (9-16) for more pauses
            "pause_cost": 0.2, # from 9-16
            "s_action_feat": False,
            "pause_feat": False,
            "td_steps": 4, # from 12-26
            # Fixed overrides
            "searchable": {
                "pause_cost": tune.grid_search([1e-5, 1e-3, 1e-1]),
                "saccade_cost_scale": tune.grid_search([1e-5, 1e-3, 1e-1]),
                # "fov": tune.grid_search(["window_periph", "window", "exponential"]),
                "env": tune.grid_search(["asterix", "seaquest", "hero"]),
            }
        },
        tune_config=tune.TuneConfig(
            # num_samples=num_samples,
            # scheduler=ASHAScheduler(
            #     stop_last_trials=False
            # ),
            # search_alg=OptunaSearch(),
            metric=metric,
            mode=mode,
            trial_name_creator=lambda t: t.trial_id,
        ),
        run_config=train.RunConfig(
            storage_path="/home/niko/Repos/atari-cr/output/ray_results",
            # stop=None if GRID_SEARCH else TrialPlateauStopper(
            #     metric, mode=mode, num_results=8, grace_period=1000),
        )
    )
    results = tuner.fit()
    print("Best result:\n", results.get_best_result().config)
