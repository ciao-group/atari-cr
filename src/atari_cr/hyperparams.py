from ray import train, tune
from typing import TypedDict, get_args

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser
from atari_cr.models import FovType

class ConfigParams(TypedDict):
    pause_cost: float
    pvm_stack: int
    sensory_action_space_quantization: int
    saccade_cost_scale: float

def tuning(config: ConfigParams, time_steps: int, debug = False):
    # Copy quantization value into the two corresponding values
    if "sensory_action_space_quantization" in config:
        quantization = config.pop("sensory_action_space_quantization")
        config["sensory_action_x_space"] = quantization
        config["sensory_action_y_space"] = quantization

    # Add basic config
    args_dict = {}
    args_dict.update({
        "clip_reward": True,
        "capture_video": True,
        "total_timesteps": time_steps,
        "no_pvm_visualization": True,
        # "no_model_output": True,
        # "use_pause_env": True,
        "env": "ms_pacman"
    })

    # Other args
    args_dict.update({
        "debug": debug,
        "evaluator":
            "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.3/999/checkpoint.pth"})

    # Add already found hyper params
    args_dict.update({
        "action_repeat": 5,
        "fov_size": 20,
        "sensory_action_space_quantization": 4, # from 9-16
        "pvm_stack": 16, # from 9-16
        "saccade_cost_scale": 0.0015, # from 9-16
        "pause_cost": 0.2, # from 9-16
    })

    # Set fixed params
    args_dict.update({
        "pause_cost": 0., # make only saccade costs matter
        "pvm_stack": 3 # from sugarl code
    })

    # Add hyperparameter config
    args_dict.update(config)
    args_dict["exp_name"] = "tuning"

    # Run the experiment
    args = ArgParser().from_dict(args_dict)
    eval_returns, out_paths = main(args)


if __name__ == "__main__":
    GAZE_TARGET = False
    DEBUG = False
    concurrent_runs = 3 if DEBUG else 3
    num_samples = 1 * concurrent_runs if DEBUG else 20
    time_steps = 1_000_000

    trainable = tune.with_resources(
        lambda config: tuning(config, time_steps, DEBUG),
        {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

    param_space: ConfigParams = {
        # "pause_cost": tune.quniform(0.00, 0.03, 0.002),
        # "pvm_stack": tune.randint(1, 20),
        # "sensory_action_space_quantization": tune.randint(1, 21), # from 10-21
        # "saccade_cost_scale": tune.quniform(0.0000, 0.0100, 0.0005),
        # "fov": tune.choice([fov for fov in get_args(FovType) if fov != "gaussian"]),
        "seed": tune.randint(0,420),
        "use_pause_env": tune.choice([True, False])
    }

    metric, mode = ("windowed_auc", "max") if GAZE_TARGET else ("raw_reward", "max")
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            # scheduler=None if DEBUG else ASHAScheduler(
            #     stop_last_trials=False
            # ),
            search_alg=OptunaSearch(),
            metric=metric,
            mode=mode,
        ),
        run_config=train.RunConfig(
            storage_path="/home/niko/Repos/atari-cr/output/ray_results",
            stop=TrialPlateauStopper(metric, mode=mode, grace_period=1000),
        )
    )
    results = tuner.fit()
    print("Best result:\n", results.get_best_result().config)
