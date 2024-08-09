import argparse
import os
from typing import Literal
from distutils.util import strtobool

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

# TODO: Remove normal pause cost in favor of a bigger penalty for 30 pauses in a row
# TODO: Do 20M steps (propably not veeery helpful)
# TODO: Test realtive actions better
# TODO: Go back to absolute actions because they are not really worse from a cr perspective
# TODO: Test other games
# TODO: Add saccade costs for foveal distance traveled

# TODO: Weitermachen mit Road Runner, Ms. Pac-Man, und Breakout; Boxing ist nicht in Atari-HEAD 

# TODO: (wenn nichts mehr klappt) Fovea zuerst über ganzen Bildschirm machen und dann immer kleiner werden lassen; gucken ab wann es schwierig wird
# TODO: (wenn nichts mehr klappt) Die Anzahl der Pausen pro Bild zb auf 4 festlegen
# TODO: (wenn nichts mehr klappt) 

# TODO: Check v1.28 results
# TODO: Typed argument parser

class ArgParser(Tap):
    exp_name: str = os.path.basename(__file__).rstrip(".py") # The name of this experiment
    seed: int = 0 # The seed of the experiment
    # disable_cuda: bool # Whether to force using CPU 
    # TODO: Merge that with no_video
    capture_video: bool # Whether to capture videos of the agent performances (check out `videos` folder)

    # Env settings
    env: str = "boxing" # The ID of the environment
    env_num: int = 1 # The number of envs to train in parallel
    fram_stack: int = 4 # The number of frames making up one observation
    action_repeat: int = 4 # The number of times an action is repeated
    clip_reward: bool # Whether to clip rewards

    # Fovea settings
    fov_size: int = 20 # The size of the fovea
    fov_init_loc: int = 0 # Where to initialize the fovea
    sensory_action_mode: Literal["absolute", "relative"] = "absolute" # Whether to interprate sensory actions absolutely or relatively
    sensory_action_space: int = 10 # Maximum size of pixels to move the fovea in one relative sensory step. Ignored for absolute sensory action mode
    # TODO: What is this?
    resize_to_full: bool = False
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
    use_pause_env: bool # Whether to use an env that lets the agent pause for only making a sensory action
    pause_cost: float = 0.1 # The cost for the env to only take a sensory step
    successive_pause_limit: int = 20 # The maximum number of successive pauses before pauses are forbidden. This prevents the agent from halting
    no_action_pause_cost: float = 0.1 # The additional cost of pausing without performing a sensory action

    # Misc
    ignore_sugarl: bool # Whether to ignore the sugarl term for Q network learning
    grokfast: bool # Whether to use grokfast
    disable_tensorboard: bool # Whether to disable tensorboard
    no_model_output: bool # Whether to disable saving the finished model
    no_video_output: bool # Whether to disble video output of the final agent acting in the env
    no_pvm_visualization: bool # Whether to disable output of visualizations of the content of the PVM buffer

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # env setting
    parser.add_argument("--env", type=str, default="boxing",
        help="the id of the environment")
    parser.add_argument("--env-num", type=int, default=1, 
        help="# envs in parallel")
    parser.add_argument("--frame-stack", type=int, default=4,
        help="frame stack #")
    parser.add_argument("--action-repeat", type=int, default=4,
        help="action repeat #")
    parser.add_argument("--clip-reward", action="store_true")

    # fov setting
    parser.add_argument("--fov-size", type=int, default=20)
    parser.add_argument("--fov-init-loc", type=int, default=0)
    parser.add_argument("--sensory-action-mode", type=str, default="absolute",
        help="How the sensory action is interpreted by the env. Either 'absolute' or 'relative'")
    parser.add_argument("--sensory-action-space", type=int, default=10,
        help="Maximum size of pixels to move the fovea in one relative sensory step. Ignored for absolute sensory action mode") 
    parser.add_argument("--resize-to-full", default=False, action="store_true")
    # for discrete observ action
    parser.add_argument("--sensory-action-x-size", type=int, default=4)
    parser.add_argument("--sensory-action-y-size", type=int, default=4)
    # pvm setting
    parser.add_argument("--pvm-stack", type=int, default=3)

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=3000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the self.optimizer")
    parser.add_argument("--buffer-size", type=int, default=100000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-start", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")

    # eval args
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="eval frequency. default -1 is eval at the end.")
    parser.add_argument("--eval-num", type=int, default=10,
        help="eval frequency. default -1 is eval at the end.")
    
    # Pause args
    parser.add_argument("--pause-cost", type=float, default=0.1,
        help="Cost for looking without taking an env action. Prevents the agent from abusing too many pauses")
    parser.add_argument("--successive-pause-limit", type=int, default=20,
        help="Limit to the amount of successive pauses the agent can make before a random action is selected instead. \
            This prevents the agent from halting")
    parser.add_argument("--ignore-sugarl", action="store_true",
        help="Whether to ignore the sugarl term in the loss calculation")
    parser.add_argument("--no-action-pause-cost", type=float, default=0.1,
        help="Penalty for performing a useless pause without a sensory action. This is meant to speed up training")
    parser.add_argument("--grokfast", action="store_true")
    parser.add_argument("--use-pause-env", action="store_true",
        help="Whether to use the normal sugarl setting without a pausable env.")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--no-model-output", action="store_true")
    parser.add_argument("--no-video-output", action="store_true")
    parser.add_argument("--no-pvm-visualization", action="store_true")
    
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    sensory_action_mode = SensoryActionMode.from_string(args.sensory_action_mode)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    assert device.type == "cuda"

    def make_env(seed, **kwargs):
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
                env = PauseableFixedFovealEnv(env, env_args, 
                    args.pause_cost, args.successive_pause_limit, args.no_action_pause_cost)
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
        cuda=args.cuda,
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
        no_video_output=args.no_video_output

    )
    eval_returns = agent.learn(
        n=args.total_timesteps,
        env_name=args.env, 
        experiment_name=args.exp_name
    )  

    return eval_returns

if __name__ == "__main__":
    args = parse_args()
    main(args)
