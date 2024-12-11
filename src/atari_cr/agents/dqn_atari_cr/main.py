import os

import gymnasium as gym
import torch
from tap import Tap

from active_gym.atari_env import AtariEnv, AtariEnvArgs, RecordWrapper, FixedFovealEnv
from atari_cr.atari_head.gaze_predictor import GazePredictor
from atari_cr.utils import (seed_everything, get_sugarl_reward_scale_atari)
from atari_cr.pauseable_env import PauseableFixedFovealEnv, SlowedFixedFovealEnv
from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN
from atari_cr.models import FovType

# OPTIONAL: Remove normal pause cost in favor of a bigger penalty for 30 pauses in a row
# OPTIONAL: Test realtive actions better

class ArgParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py") # Name of this experiment
    seed: int = 0 # The seed of the experiment
    disable_cuda: bool = False # Whether to force the use of CPU
    capture_video: bool = False # Whether to capture videos of the agent performances

    # Env settings
    env: str = "boxing" # The ID of the environment
    env_num: int = 1 # The number of envs to train in parallel
    frame_stack: int = 4 # The number of frames making up one observation
    action_repeat: int = 4 # The number of times an action is repeated
    clip_reward: bool = False # Whether to clip rewards
    sticky_action_prob: float = 0.0 # Probability an action is repeated next timestep

    # Fovea settings
    fov_size: int = 20 # The size of the fovea
    fov_init_loc: int = 0 # Where to initialize the fovea
    relative_sensory_actions: bool = False # Relative or absolute sensory actions
    sensory_action_space: int = 10 # Maximum distance in one sensory step
    resize_to_full: bool = False # No idea what that is
    sensory_action_x_size: int = 4 # How many smallest sensory steps fit in x direction
    sensory_action_y_size: int = 4 # How many smallest sensory steps fit in y direction
    pvm_stack: int = 3 # How many normal observation to aggregate in the PVM buffer

    # Algorithm specific arguments
    total_timesteps: int = 3000000 # The number of timesteps
    learning_rate: float = 1e-4 # The learning rate
    buffer_size: int = 100000 # The size of the replay buffer
    gamma: float = 0.99 # The discount factor gamma
    target_network_frequency: int = 1000 # Timesteps between Q network updates
    batch_size: int = 32 # The batch size during training
    start_e: float = 1.0 # The starting value for the exploration probability epsilon
    end_e: float = 0.01 # The final value for the exploration probability epsilon
    exploration_fraction: float = 0.10 # Fraction of timesteps to reach the max epsilon
    learning_start: int = 80000 # Timesteps before training starts on the replay buffer
    train_frequency: int = 4 # Steps to take in the env before a new training iteration

    # Eval args
    eval_frequency: int = -1 # How many steps to take in the env before a new evaluation
    eval_num: int = 10 # How envs are created for evaluation

    # Pause args
    use_pause_env: bool = False # Whether to allow pauses for more observations per step
    pause_cost: float = 0.1 # The cost for the env to only take a sensory step
    consecutive_pause_limit: int = 50 # Maximum allowed number of consecutive pauses
    saccade_cost_scale: float = 0.000 # How much to penalize bigger eye movements
    # EMMA reference: doi.org/10.1016/S1389-0417(00)00015-2

    # Misc
    ignore_sugarl: bool = False # Whether to ignore the sugarl term for Q net learning
    no_model_output: bool = False # Whether to disable saving the finished model
    no_pvm_visualization: bool = False # Whether to disable output of PVM visualizations
    debug: bool = False # Debug mode for more output
    evaluator: str = "" # Path to gaze predictor weights for evaluation
    fov: FovType = "window" # Type of fovea
    og_env: bool = False # Whether to use normal sugarl env
    slowed_env: bool = False # Whether to use a time sensitve env for pausing

def make_env(seed: int, args: ArgParser, training = False):
    def thunk():
        # Env args
        env_args = AtariEnvArgs(
            game=args.env,
            seed=seed,
            obs_size=(84, 84),
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            fov_size=(args.fov_size, args.fov_size),
            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
            sensory_action_mode="relative" if args.relative_sensory_actions \
                else "absolute",
            sensory_action_space=(
                -args.sensory_action_space, args.sensory_action_space),
            resize_to_full=args.resize_to_full,
            clip_reward=args.clip_reward,
            mask_out=True,
            training=training,
            record=args.capture_video,
        )

        # Pauseable or not pauseable env creation
        env = AtariEnv(env_args)
        if args.og_env:
            env = RecordWrapper(env, env_args)
            env = FixedFovealEnv(env, env_args)
        elif args.slowed_env:
            env = SlowedFixedFovealEnv(
                env, env_args, args.pause_cost, args.saccade_cost_scale,
                args.fov, not args.use_pause_env)
        else:
            env = PauseableFixedFovealEnv(
                env, env_args, args.pause_cost, args.saccade_cost_scale,
                args.fov, not args.use_pause_env)

        # Env configuration
        env.unwrapped.ale.setFloat(
            'repeat_action_probability', args.sticky_action_prob)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk

def make_train_env(args: ArgParser):
    envs = [make_env(args.seed + i, args) for i in range(args.env_num)]
    return gym.vector.SyncVectorEnv(envs)

def make_eval_env(seed, args: ArgParser):
    """ Return VecEnv with a single environment """
    envs = [make_env(args.seed + seed, args, training=False)]
    return gym.vector.SyncVectorEnv(envs)

def main(args: ArgParser):
    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")

    seed_everything(args.seed)

    # Create one env for each pause cost
    env = make_train_env(args)

    # Gaze predictor for evaluation of the agent's human-plausibility
    evaluator = GazePredictor.from_save_file(args.evaluator) if args.evaluator else None

    agent = CRDQN(
        env=env,
        sugarl_r_scale=get_sugarl_reward_scale_atari(args.env),
        env_name=args.env,
        eval_env_generator=lambda seed: make_eval_env(seed, args),
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
        sensory_action_space_quantization=(
            args.sensory_action_x_size, args.sensory_action_y_size),
        epsilon_interval=(args.start_e, args.end_e),
        exploration_fraction=args.exploration_fraction,
        ignore_sugarl=args.ignore_sugarl,
        no_model_output=args.no_model_output,
        no_pvm_visualization=args.no_pvm_visualization,
        capture_video=args.capture_video,
        debug=args.debug,
        evaluator=evaluator,
    )
    eval_returns, out_paths = agent.learn(
        n=args.total_timesteps,
        experiment_name=args.exp_name
    )

    return eval_returns, out_paths

if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(args)
