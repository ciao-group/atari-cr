from typing import Dict
from ray import train, tune
from typing import TypedDict

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

class ConfigParams(TypedDict):
    action_repeat: int
    sticky_action_probability: float

def tuning(config: ConfigParams, time_steps: int):
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
        "no_model_output": True,
        "use_pause_env": True,
        "env": "ms_pacman"
    })

    # Add already found hyper params
    args_dict.update({
        # "action_repeat": 5,
        "fov_size": 50,
        "sensory_action_space_quantization": 4,
    })

    # Add vaguely found hyperparameters
    args_dict.update({
        "no_action_pause_cost": 1.4,
        "pause_cost": 0.3,
        "pvm_stack": 13, 
        "saccade_cost_scale": 0.0015,
    })

    # Add hyperparameter config
    args_dict.update(config)
    args_dict["exp_name"] = "tuning"

    args = ArgParser().from_dict(args_dict)
    eval_returns = main(args)

    # Send the current training result back to Tune
    result = {"episode_reward": sum(eval_returns) / len(eval_returns)}
    print("eee", eval_returns)

    # TODO: Test against Atari-HEAD
    # kl_div = 0
    # result =  {"loss": kl_div} 

    return result

if __name__ == "__main__":
    DEBUG = False
    concurrent_runs = 3 if DEBUG else 4
    num_samples = 1 * concurrent_runs if DEBUG else 50
    time_steps = int(1e6) if DEBUG else int(1e6)

    trainable = lambda config: tuning(config, time_steps=time_steps)
    trainable = tune.with_resources(trainable, {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

    # param_space: ConfigParams = {
    #     "pause_cost": tune.quniform(0.10, 0.40, 0.05),
    #     "no_action_pause_cost": tune.quniform(1.0, 2.0, 0.2),
    #     "pvm_stack": tune.randint(2, 16),
    #     "saccade_cost_scale": tune.quniform(0.0001, 0.0020, 0.0001)
    # }
    param_space: ConfigParams = {
        "action_repeat": tune.choice([1, 2, 3, 4]),
        "sticky_action_probability": tune.quniform(0, 1, 0.1),
    }

    metric, mode = "episode_reward", "max"
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            # scheduler=ASHAScheduler(
            #     stop_last_trials=False
            # ),
            search_alg=OptunaSearch(),
            metric=metric,
            mode=mode
        ),
        run_config=train.RunConfig(
            storage_path="/home/niko/Repos/atari-cr/output/ray_results",
            # stop={"training_iteration": 5000}
        )
    )
    results = tuner.fit()
    print("Best result:\n", results.get_best_result().config)