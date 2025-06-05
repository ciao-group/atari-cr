import os

from gymnasium.vector import SyncVectorEnv
import torch
from tap import Tap

from active_gym.atari_env import AtariEnv, AtariEnvArgs, RecordWrapper, FixedFovealEnv
from atari_cr.atari_head.gaze_predictor import GazePredictor
from atari_cr.utils import (seed_everything, get_sugarl_reward_scale_atari)
from atari_cr.pauseable_env import PauseableFixedFovealEnv
from atari_cr.agents.dqn_atari_cr.crdqn import CRDQN
from atari_cr.foveation import FovType


class ArgParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py") # Name of this experiment
    seed: int = 0 # Seed of the experiment
    disable_cuda: bool = False # Whether to force the use of CPU
    capture_video: bool = False # Whether to capture gameplay videos

    # Env settings
    env: str = "asterix" # ID of the environment
    env_num: int = 1 # Number of envs to train in parallel
    frame_stack: int = 4 # Number of frames making up one observation
    action_repeat: int = 5 # Number of times an action is repeated
    clip_reward: bool = False # Whether to clip rewards
    sticky_action_prob: float = 0.0 # Probability an action is repeated next timestep

    # Fovea settings
    fov_size: int = 20 # UNUSED # Size of the fovea
    fov_init_loc: int = 0 # Where to initialize the fovea
    relative_sensory_actions: bool = False # Relative or absolute sensory actions
    sensory_action_space: int = 10 # Maximum distance in one sensory step
    resize_to_full: bool = False # No idea what that is
    sensory_action_x_size: int = 8 # How many smallest sensory steps fit in x direction
    sensory_action_y_size: int = 8 # How many smallest sensory steps fit in y direction
    pvm_stack: int = 3 # How many normal observation to aggregate in the PVM buffer

    # Algorithm specific arguments
    total_timesteps: int = 3000000 # Number of timesteps
    learning_rate: float = 1e-4 # Learning rate
    buffer_size: int = 10_000 # Size of the replay buffer
    gamma: float = 0.99 # Discount factor gamma
    target_network_frequency: int = 10000 # Timesteps between Q network updates
    batch_size: int = 32 # Bathc size during training
    start_e: float = 1.0 # Starting value for the exploration probability epsilon
    end_e: float = 0.01 # Final value for the exploration probability epsilon
    exploration_fraction: float = 0.10 # Fraction of timesteps to reach the max epsilon
    learning_start: int = 80000 # Timesteps before training starts on the replay buffer
    train_frequency: int = 4 # Steps to take in the env before a new training iteration

    # Eval args
    eval_frequency: int = -1 # How many steps to take in the env before a new evaluation
    n_evals: int = 5 # How many envs are created for evaluation
    checkpoint: str = "" # Checkpoint to resume training

    # Pause args
    use_pause_env: bool = False # Whether to allow pauses for more observations per step
    pause_cost: float = 0.1 # Cost for the env to only take a sensory step
    consecutive_pause_limit: int = 20 # Maximum allowed number of consecutive pauses
    saccade_cost_scale: float = 0.000 # How much to penalize bigger eye movements
    # EMMA reference: doi.org/10.1016/S1389-0417(00)00015-2

    # Dirs
    atari_head_dir: str = "data/Atari-HEAD" # Path to unzipped Atari-HEAD files

    # Misc
    ignore_sugarl: bool = False # Whether to ignore the sugarl term for Q net learning
    no_model_output: bool = False # Whether to disable saving the finished model
    no_pvm_visualization: bool = False # Whether to disable output of PVM visualizations
    debug: bool = False # Debug mode for more output
    evaluator: bool = False # Whether to use a model to evaluate human-likeness
    fov: FovType = "window" # Type of fovea
    og_env: bool = False # Whether to use normal sugarl env
    timed_env: bool = False # Whether to use a time sensitve env for pausing
    pause_feat: bool = False # Whether to tell the policy how many pauses have been made
    s_action_feat: bool = False # Whether to give the prev sensory action to the policy
    td_steps: int = 4 # Number of steps for n-step TD learning
    mean_pvm: bool = False # Whether to combine obs in pvm using mean instead of max

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
        else:
            env = PauseableFixedFovealEnv(
                env, env_args, args.pause_cost, args.saccade_cost_scale,
                args.fov, not args.use_pause_env, args.consecutive_pause_limit,
                timer=args.timed_env, fov_weighting=args.mean_pvm)

        # Env configuration
        env.unwrapped.ale.setFloat(
            'repeat_action_probability', args.sticky_action_prob)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk

def make_train_env(args: ArgParser):
    envs = [make_env(args.seed + i, args) for i in range(args.env_num)]
    return SyncVectorEnv(envs)

def make_eval_env(seed, args: ArgParser):
    """ Return VecEnv with a single environment """
    envs = [make_env(args.seed + seed, args, training=False)]
    return SyncVectorEnv(envs)

def main(args: ArgParser):
    # Use bfloat16 to speed up matrix computation
    torch.set_float32_matmul_precision("medium")

    seed_everything(args.seed)

    # Create one env for each pause cost
    env = make_train_env(args)

    # Gaze predictor for evaluation of the agent's human-plausibility
    if not os.path.exists(args.atari_head_dir):
        raise FileNotFoundError("Atari-HEAD data not found. Download and unzip the dataset and"
            "point the param --atari_head_dir to it.")
    evaluator = GazePredictor.init(
        f"{args.atari_head_dir}/{args.env}",
        f"data/Atari-HEAD/{args.env}"
    ) if args.evaluator else None

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
        n_evals=args.n_evals,
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
        pause_feat=args.pause_feat,
        s_action_feat=args.s_action_feat,
        td_steps=args.td_steps,
        checkpoint=args.checkpoint,
        mean_pvm=args.mean_pvm,
    )
    eval_returns, out_paths = agent.learn(
        n=args.total_timesteps,
        experiment_name=args.exp_name
    )

    return eval_returns, out_paths

if __name__ == "__main__":
    args = ArgParser().parse_args()
    # Align windowed fov size with exponential fov size
    match args.fov:
        case "window": args.fov_size = 26
        case "window_periph": args.fov_size = 20
    main(args)
