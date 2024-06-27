import argparse
import os, sys
import os.path as osp
import random
import time
from itertools import product
from distutils.util import strtobool

sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize
from torch.distributions.categorical import Categorical
from active_gym import AtariFixedFovealEnv, AtariEnvArgs
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import CnnPolicy, SACPolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from torch.utils.tensorboard import SummaryWriter

from common.buffer import DoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from common.pauseable_env import PauseableAtariFixedFovealEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
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
    parser.add_argument("--clip-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)


    # fov setting
    parser.add_argument("--fov-size", type=int, default=50)
    parser.add_argument("--fov-init-loc", type=int, default=0)
    parser.add_argument("--sensory-action-mode", type=str, default="absolute")
    parser.add_argument("--sensory-action-space", type=int, default=10) # ignored when sensory_action_mode="relative"
    parser.add_argument("--resize-to-full", default=False, action="store_true")
    # for discrete observ action
    parser.add_argument("--sensory-action-x-size", type=int, default=4)
    parser.add_argument("--sensory-action-y-size", type=int, default=4)
    # pvm setting
    parser.add_argument("--pvm-stack", type=int, default=3)

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.02,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--sensory-alpha-autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--motor-target-entropy-scale", type=float, default=0.2,
        help="coefficient for scaling the autotune entropy target")
    parser.add_argument("--sensory-target-entropy-scale", type=float, default=0.2,
        help="coefficient for scaling the autotune entropy target")

    # eval args
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="eval frequency. default -1 is eval at the end.")
    parser.add_argument("--eval-num", type=int, default=10,
        help="eval frequency. default -1 is eval at the end.")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    env_args = AtariEnvArgs(game=args.env, seed=args.seed, obs_size=(84, 84), **args)
    env = PauseableAtariFixedFovealEnv(env_args)
    policy = CnnPolicy()

    agent = SAC(policy, env)