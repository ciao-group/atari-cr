from typing import Dict
from ray import train, tune
from sys import argv

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

def tuning(config: Dict):
    # Copy granularity value into the two corresponding values
    granularity = config.pop("sensory_action_space_granularity")
    config["sensory_action_x_space"] = granularity
    config["sensory_action_y_space"] = granularity

    # Add basic config
    argv.extend([
        "--clip_reward",
        "--capture_video",
        "--exp_name", "tuning",
        "--total_timesteps", "1000000",
        # 
        "--no_pvm_visualization",
        "--no_model_output",
        "--disable_tensorboard",
        # 
    ])
    # Add config config
    for key, value in config.items():
        argv.extend([f"--{key}", str(value)])

    args = ArgParser().parse_args()
    eval_returns = main(args)

    # TODO: Test against Atari-HEAD
    kl_div = 0

    return {"loss": kl_div}

if __name__ == "__main__":
    param_space = {
        "pause_cost": tune.quniform(0.01, 0.10, 0.01),
        "no_action_pause_cost": tune.quniform(0.1, 3.0, 0.1),
        "pvm_stack": tune.randint(1, 12),
        "fov_size": tune.qrandint(10, 100, 10),
        "sensory_action_space_granularity": tune.randint(1, 16),
        "grokfast": tune.choice([True, False])
    }

    tuner = tune.Tuner(
        tuning,
        param_space=param_space
    )
    tuner.fit()