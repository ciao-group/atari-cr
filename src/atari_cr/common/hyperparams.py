import tempfile
from typing import Dict
from ray import train, tune
from sys import argv
from typing import TypedDict

import ray
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

class ConfigParams(TypedDict):
    pause_cost: float
    no_action_pause_cost: float
    pvm_stack: int
    fov_size: int
    sensory_action_space_granularity: int
    grokfast: bool

def tuning(config: ConfigParams):
    # Copy granularity value into the two corresponding values
    granularity = config.pop("sensory_action_space_granularity")
    config["sensory_action_x_space"] = granularity
    config["sensory_action_y_space"] = granularity

    # Add basic config
    args_dict = {}
    args_dict.update({
        "clip_reward": True,
        "capture_video": True,
        "total_timesteps": 1000000,
        "no_pvm_visualization": True,
        "no_model_output": True,
        "use_pause_env": True,
        "env": "ms_pacman"
    })

    # Add hyperparameter config
    args_dict.update(config)
    args_dict["exp_name"] = "tuning"

    args = ArgParser().from_dict(args_dict)
    eval_returns = main(args)

    # Send the current training result back to Tune
    result = {"score": sum(eval_returns)}

    # TODO: Test against Atari-HEAD
    # kl_div = 0
    # result =  {"loss": kl_div}    

    return result

if __name__ == "__main__":
    concurrent_runs = 4
    tuning = tune.with_resources(tuning, {"cpu": 8//concurrent_runs, "gpu": 1/concurrent_runs})

    param_space = {
        "pause_cost": tune.quniform(0.01, 0.10, 0.01),
        "no_action_pause_cost": tune.quniform(0.1, 2.0, 0.1),
        "pvm_stack": tune.randint(1, 12),
        "fov_size": tune.qrandint(10, 80, 10),
        "sensory_action_space_granularity": tune.randint(1, 16),
        "grokfast": tune.choice([True, False])
    }

    metric, mode = "score", "max"
    tuner = tune.Tuner(
        tuning,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=3,
            scheduler=ASHAScheduler(),
            search_alg=HyperOptSearch(metric=metric, mode=mode),
            metric=metric,
            mode=mode
        ),
    )
    results = tuner.fit()
    print("Best result:\n", results.get_best_result().config)