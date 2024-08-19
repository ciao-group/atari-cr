import os
from typing import List, Literal, Union

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from tap import Tap

from active_gym.atari_env import AtariEnv, AtariEnvArgs, AtariFixedFovealEnv

from atari_cr.common.utils import seed_everything, get_sugarl_reward_scale_atari, get_env_attributes
from atari_cr.common.pauseable_env import PauseableFixedFovealEnv
from atari_cr.common.models import SensoryActionMode
from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN

# OPTIONAL: Remove normal pause cost in favor of a bigger penalty for 30 pauses in a row
# OPTIONAL: Test realtive actions better

class ArgParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py") # The name of this experiment
    seed: int = 0 # The seed of the experiment
    disable_cuda: bool = False # Whether to force the use of CPU 
    capture_video: bool = False # Whether to capture videos of the agent performances (check out `videos` folder)

    # Env settings
    env: str = "boxing" # The ID of the environment
    env_num: int = 1 # The number of envs to train in parallel
    frame_stack: int = 4 # The number of frames making up one observation
    action_repeat: int = 4 # The number of times an action is repeated
    clip_reward: bool = False # Whether to clip rewards

    # Fovea settings
    fov_size: int = 20 # The size of the fovea
    fov_init_loc: int = 0 # Where to initialize the fovea
    sensory_action_mode: Literal["absolute", "relative"] = "absolute" # Whether to interprate sensory actions absolutely or relatively
    sensory_action_space: int = 10 # Maximum size of pixels to move the fovea in one relative sensory step. Ignored for absolute sensory action mode
    resize_to_full: bool = False # No idea what that is
    sensory_action_x_size: int = 4 # How many smallest sensory steps fit in x direction
    sensory_action_y_size: int = 4 # How many smallest sensory steps fit in y direction
    pvm_stack: int = 3 # How many normal observation to aggregate in the PVM buffer

    # Algorithm specific arguments
    total_timesteps: int = 3000000 # The number of timesteps
    learning_rate: float = 1e-4 # The learning rate
    buffer_size: int = 100000 # The size of the replay buffer
    gamma: float = 0.99 # The discount factor gamma
    target_network_frequency: int = 1000 # How many timesteps to wait before updating the target Q network
    batch_size: int = 32 # The batch size during training
    start_e: float = 1.0 # The starting value for the exploration probability epsilon
    end_e: float = 0.01 # The final value for the exploration probability epsilon
    exploration_fraction: float = 0.10 # The fraction of `total-timesteps` it takes from until the final value of the exploration probability epsilon
    learning_start: int = 80000 # How many timesteps to wait before training starts on the replay buffer
    train_frequency: int = 4 # How many steps to take in the env before a new training iteration

    # Eval args
    eval_frequency: int = -1 # How many steps to take in the env before a new evaluation
    eval_num: int = 10 # How envs are created for evaluation

    # Pause args
    use_pause_env: bool = False # Whether to use an env that lets the agent pause for only making a sensory action
    pause_cost: float = 0.1 # The cost for the env to only take a sensory step
    successive_pause_limit: int = 20 # The maximum number of successive pauses before pauses are forbidden. This prevents the agent from halting
    no_action_pause_cost: float = 0.1 # The additional cost of pausing without performing a sensory action
    saccade_cost_scale: float = 0.001 # How much the agent is punished for bigger eye movements

    # Misc
    ignore_sugarl: bool = False # Whether to ignore the sugarl term for Q network learning
    grokfast: bool = False # Whether to use grokfast
    disable_tensorboard: bool = False # Whether to disable tensorboard
    no_model_output: bool = False # Whether to disable saving the finished model
    no_pvm_visualization: bool = False # Whether to disable output of visualizations of the content of the PVM buffer

def main(args: ArgParser):
    sensory_action_mode = SensoryActionMode.from_string(args.sensory_action_mode)
    seed_everything(args.seed)

    def make_env(seed: int, **kwargs):
        def thunk():
            env_args = AtariEnvArgs(
                game=args.env, 
                seed=seed, 
                obs_size=(84, 84), 
                frame_stack=args.frame_stack, 
                action_repeat=args.action_repeat,
                fov_size=(args.fov_size, args.fov_size), 
                fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                sensory_action_mode=sensory_action_mode,
                sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
                resize_to_full=args.resize_to_full,
                clip_reward=args.clip_reward,
                mask_out=True,
                **kwargs
            )
            if args.use_pause_env:
                env = AtariEnv(env_args)    
                env = PauseableFixedFovealEnv(
                    env, env_args, args.pause_cost, args.successive_pause_limit, 
                    args.no_action_pause_cost, args.saccade_cost_scale
                )
            else:
                env_args.sensory_action_mode = str(sensory_action_mode)
                env = AtariFixedFovealEnv(env_args)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    def make_train_env():
        envs = [make_env(args.seed + i) for i in range(args.env_num)]
        return gym.vector.SyncVectorEnv(envs)

    def make_eval_env(seed):
        envs = [make_env(args.seed + seed, training=False, record=args.capture_video)]
        return gym.vector.SyncVectorEnv(envs)

    # Create one env for each pause cost
    env = make_train_env()

    # Create a tensorboard writer and log the env state
    if not args.disable_tensorboard:
        run_identifier = os.path.join(args.exp_name, args.env)
        run_dir = os.path.join("output/runs", run_identifier)
        tb_dir = os.path.join(run_dir, "tensorboard")
        writer = SummaryWriter(os.path.join(tb_dir, f"seed{args.seed}"))
        hyper_params_table = "\n".join([f"|{key}|{value}|" for key, value in get_env_attributes(env.envs[0])])
        writer.add_text(
            "Env Hyperparameters", 
            f"|param|value|\n|-|-|\n{hyper_params_table}",
        )

    agent = CRDQN(
        env=env,
        sugarl_r_scale=get_sugarl_reward_scale_atari(args.env),
        eval_env_generator=make_eval_env,
        fov_size=args.fov_size,
        seed=args.seed,
        cuda=(not args.disable_cuda),
        learning_rate=args.learning_rate,
        replay_buffer_size=args.buffer_size,
        pvm_stack=args.pvm_stack, 
        frame_stack=args.frame_stack,
        batch_size=args.batch_size,
        train_frequency=args.train_frequency,
        learning_start=args.learning_start,
        gamma=args.gamma,
        target_network_frequency=args.target_network_frequency,
        eval_frequency=args.eval_frequency,
        n_evals=args.eval_num,
        sensory_action_space_granularity=(args.sensory_action_x_size, args.sensory_action_y_size),
        epsilon_interval=(args.start_e, args.end_e),
        exploration_fraction=args.exploration_fraction,
        ignore_sugarl=args.ignore_sugarl,
        grokfast=args.grokfast,
        sensory_action_mode=args.sensory_action_mode,
        disable_tensorboard=args.disable_tensorboard,
        no_model_output=args.no_model_output,
        no_pvm_visualization=args.no_pvm_visualization,
        capture_video=args.capture_video,
    )
    eval_returns = agent.learn(
        n=args.total_timesteps,
        env_name=args.env, 
        experiment_name=args.exp_name
    )

    return eval_returns

if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(args)
