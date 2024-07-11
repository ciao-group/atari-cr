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
from typing import Callable, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
from gymnasium.vector import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize

from common.buffer import DoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari, linear_schedule
from common.pauseable_env import PauseableAtariFixedFovealEnv
from torch.utils.tensorboard import SummaryWriter

from active_gym.atari_env import AtariEnvArgs


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # env setting
    parser.add_argument("--env", type=str, default="breakout",
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
    parser.add_argument("--total-timesteps", type=int, default=3000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the self.optimizer")
    parser.add_argument("--buffer-size", type=int, default=500000,
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
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(seed, **kwargs):
    def thunk():
        env_args = AtariEnvArgs(
            game=args.env, 
            seed=args.seed + seed, 
            obs_size=(84, 84), 
            frame_stack=args.frame_stack, 
            action_repeat=args.action_repeat,
            fov_size=(args.fov_size, args.fov_size), 
            fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
            sensory_action_mode=args.sensory_action_mode,
            sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
            resize_to_full=args.resize_to_full,
            clip_reward=args.clip_reward,
            mask_out=True,
            **kwargs
        )
        env = PauseableAtariFixedFovealEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_train_env():
    envs = [make_env(i) for i in range(args.env_num)]
    return gym.vector.SyncVectorEnv(envs)

def make_eval_env(seed):
    envs = [make_env(seed, training=False, record=args.capture_video)]
    return gym.vector.SyncVectorEnv(envs)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()

        # Get the size of the different network heads
        assert isinstance(env.single_action_space, Dict)
        assert sensory_action_set
        self.motor_action_space_size = env.single_action_space["motor"].n
        self.sensory_action_space_size = len(sensory_action_set)

        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, self.motor_action_space_size)
        self.sensory_action_head = nn.Linear(512, self.sensory_action_space_size)

    def forward(self, x):
        x = self.backbone(x)
        motor_action = self.motor_action_head(x)
        sensory_action = None
        if self.sensory_action_head:
            sensory_action = self.sensory_action_head(x)
        return motor_action, sensory_action

    def chose_action(self, env: gym.vector.VectorEnv, pvm_obs: np.ndarray, epsilon: float):
        """
        Epsilon greedy action selection

        Parameters
        ----------
        epsilon : float
            Probability of selecting a random action
        """
        # Execute random motor and sensory action with probability epsilon
        if random.random() < epsilon:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            motor_actions = np.array([actions[0]["motor"]])
            sensory_actions = np.array([random.randint(0, self.sensory_action_space_size-1)])
        else:
            motor_actions, sensory_actions = self.chose_eval_action(pvm_obs)

        return motor_actions, sensory_actions
    
    def chose_eval_action(self, pvm_obs: np.ndarray):
        resize = Resize(pvm_obs.shape[2:])
        motor_q_values, sensory_q_values = self(resize(torch.from_numpy(pvm_obs)).to(device))
        motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
        sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

        return motor_actions, sensory_actions


class SelfPredictionNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory"].n
        
        self.backbone = nn.Sequential(
            nn.Conv2d(8, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, motor_action_space_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class CRDQN:
    """
    Algorithm for DQN with Computational Rationality
    """
    def __init__(
            self, 
            env: gym.Env, 
            eval_env_generator: Callable[[int], gym.Env],
            writer: SummaryWriter, 
            sugarl_r_scale: float,
            fov_size = 50,
            sensory_action_space_granularity: Tuple[int] = (4, 4),
            learning_rate = 0.0001,
            replay_buffer_size = 100000,
            channel_stack_size = 4,
            pvm_stack_size = 3,
            epsilon_interval: Tuple[float] = (1., 0.01),
            exploration_fraction = 0.10,
            batch_size = 32,
            learning_start = 80000,
            train_frequency = 4,
            target_network_frequency = 1000,
            eval_frequency = -1,
            gamma = 0.99,
            cuda = True,
            n_evals = 10
        ):
        """
        Parameters
        ----------
        env : `gymnasium.Env`
        eval_env_generator : Callable
            Function, outputting an eval env given a seed
        self.writer : `tensorboard.SummaryWriter`
        sugarl_r_scale : float
        fov_size : int
        sensory_action_space_granularity : tuple of int
            The number of smallest sensory steps it takes 
            from left to right and from top to bottom
        learning_rate : float
            The learning rate used for the Q Network and Self Predicition Network
        replay_buffer_size : int
        channel_stack_size : int
            # TODO ???
        pvm_stack_size : int
            The number of recent observations to be used for action selection
        epsilon_interval : tuple of int
            Interval in which the propability for a random action 
            epsilon moves from one end to the other during training
        exploration_fraction : float
            The fraction of the total learning time steps it takes for epsilon
            to reach its end value
        batch_size
        learning_start : int
            The timestep at which to start training the Q Network
        train_frequency : int
            Number of timesteps between training sessions
        target_network_frequency : int
            Number of timesteps between target network updates
        eval_frequency : int
            Number of timesteps between evaluations; -1 for eval at the end
        gamma : float
            The discount factor gamma
        cuda : bool
            Whether to use cuda or not
        n_evals : int
            Number of eval episodes to be played
        """
        self.env = env
        self.writer = writer
        self.sugarl_r_scale = sugarl_r_scale
        self.fov_size = fov_size
        self.epsilon_interval = epsilon_interval
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.learning_start = learning_start
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency
        self.eval_frequency = eval_frequency
        self.n_evals = n_evals
        self.eval_env_generator = eval_env_generator
        self.pvm_stack_size = pvm_stack_size
        self.channel_stack_size = channel_stack_size

        self.n_envs = len(self.env.envs) if isinstance(self.env, VectorEnv) else 1
        self.current_timestep = 0

        # Get the observation size
        self.single_env = env.envs[0] if isinstance(env, VectorEnv) else env
        self.obs_size = env.observation_space.shape[2:]
        assert len(self.obs_size) == 2, "The CRDQN agent only supports 2D Environments"

        # Get the sensory action set as a list of discrete actions
        # How far can the fovea move from left to right and from top to bottom 
        max_sensory_action_step = np.array(self.obs_size) - np.array([self.fov_size, self.fov_size])
        sensory_action_step_size = max_sensory_action_step // sensory_action_space_granularity
        sensory_action_x_set = list(range(0, max_sensory_action_step[0], sensory_action_step_size[0]))[:sensory_action_space_granularity[0]]
        sensory_action_y_set = list(range(0, max_sensory_action_step[1], sensory_action_step_size[1]))[:sensory_action_space_granularity[1]]
        # Discrete action set as cross product of possible x and y steps
        self.sensory_action_set = [np.array(a) for a in list(product(sensory_action_x_set, sensory_action_y_set))]

        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        assert self.device.type == "cuda"

        # Q networks
        self.q_network = QNetwork(self.env, self.sensory_action_set).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = QNetwork(self.env, self.sensory_action_set).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Self Prediction Networks; used to judge the quality of sensory actions
        self.sfn = SelfPredictionNetwork(self.env, self.sensory_action_set).to(device)
        self.sfn_optimizer = optim.Adam(self.sfn.parameters(), lr=learning_rate)

        # Replay Buffer aka. Long Term Memory
        self.rb = DoubleActionReplayBuffer(
            replay_buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space["motor"],
            Discrete(len(self.sensory_action_set)),
            self.device,
            n_envs=self.env.num_envs if isinstance(self.env, VectorEnv) else 1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        # PVM Buffer aka. Short Term Memory, combining multiple observations
        # TODO: Investigation into pvm_buffer and its 4 channels
        # TODO: Better Belief Map than PVM maybe
        self.pvm_buffer = PVMBuffer(pvm_stack_size, (self.n_envs, channel_stack_size, *self.obs_size))

    def learn(self, n: int, env_name: str, experiment_name: str):
        """
        Acts in the environment and trains the agent for n timesteps
        """
        self.start_time = time.time()
        obs, infos = self.env.reset()

        while self.current_timestep < n:
            self.pvm_buffer.append(obs)
            # self.pvm_buffer.display()

            pvm_obs = self.pvm_buffer.get_obs(mode="stack_max")
            epsilon = self._epsilon_schedule(n)

            motor_actions, sensory_actions = self.q_network.chose_action(self.env, pvm_obs, epsilon)

            # Execute the game and log data
            next_obs, rewards, dones, _, infos = self.env.step({
                "motor": motor_actions, 
                "sensory": [self.sensory_action_set[a] for a in  sensory_actions] 
            })

            # Record episode returns
            if "final_info" in infos and True in dones:
                # Get the index of the env that finished 
                idx = np.argmax(dones)
                print((
                    f"[T: {time.time()-self.start_time:.2f}]"
                    f"[N: {self.current_timestep:07,d}] "
                    f"[R: {infos['final_info'][idx]['reward']:.2f}] "
                    f"[Pauses: {infos['final_info'][idx]['n_pauses']} x ({-infos['final_info'][idx]['pause_cost']})]"
                ))    
                self.writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], self.current_timestep)
                self.writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], self.current_timestep)
                self.writer.add_scalar("charts/epsilon", epsilon, self.current_timestep)

            # Save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs
            if True in dones:
                idx = np.argmax(dones)
                real_next_obs[idx] = infos["final_observation"][idx]
            pvm_buffer_copy = self.pvm_buffer.copy()
            pvm_buffer_copy.append(real_next_obs)
            real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
            self.rb.add(pvm_obs, real_next_pvm_obs, motor_actions, sensory_actions, rewards, dones, {})
            obs = next_obs

            self.current_timestep += self.n_envs

            obs_backup = obs

            # Continue gathering observations 
            # if no batches can can be formed from the buffer yet
            if self.current_timestep < self.batch_size:
                continue

            # Training
            if self.current_timestep % self.train_frequency == 0:
                self.train()
            if self.current_timestep % 100 == 0:
                self.writer.add_scalar("charts/SPS", int(self.current_timestep / (time.time() - self.start_time)), self.current_timestep)

            # Evaluation
            if (self.current_timestep % self.eval_frequency == 0 and self.eval_frequency > 0) or \
            (self.current_timestep >= n):
                self.evaluate(env_name, experiment_name)
            
            # Restore obs if eval occurs
            obs = obs_backup 

        self.env.close()
        # eval_env.close()
        self.writer.close()

    def train(self):
        """
        Performs one training iteration from the replay buffer
        """
        # Counter-balance the true global transitions used for training
        data = self.rb.sample(self.batch_size // self.n_envs) # counter-balance the true global transitions used for training

        # SFN learning
        # Concat at dimension T
        concat_observation = torch.concat([data.next_observations, data.observations], dim=1) 
        pred_motor_actions = self.sfn(Resize(self.obs_size)(concat_observation))
        sfn_loss = self.sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())
        self.sfn_optimizer.zero_grad()
        sfn_loss.backward()
        self.sfn_optimizer.step()
        observ_r = F.softmax(pred_motor_actions).gather(1, data.motor_actions).squeeze().detach()

        # Q network learning
        if self.current_timestep > self.learning_start:
            with torch.no_grad():
                motor_target, sensory_target = self.target_network(Resize(self.obs_size)(data.next_observations))
                motor_target_max, _ = motor_target.max(dim=1)
                sensory_target_max, _ = sensory_target.max(dim=1)
                # Scale step-wise reward with observ_r
                observ_r_adjusted = observ_r.clone()
                observ_r_adjusted[data.rewards.flatten() > 0] = 1 - observ_r_adjusted[data.rewards.flatten() > 0]
                td_target = data.rewards.flatten() - (1 - observ_r) * self.sugarl_r_scale + self.gamma * (motor_target_max+sensory_target_max) * (1 - data.dones.flatten())
                original_td_target = data.rewards.flatten() + self.gamma * (motor_target_max+sensory_target_max) * (1 - data.dones.flatten())

            old_motor_q_val, old_sensory_q_val = self.q_network(Resize(self.obs_size)(data.observations))
            old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
            old_sensory_val = old_sensory_q_val.gather(1, data.sensory_actions).squeeze()
            old_val = old_motor_val + old_sensory_val
            
            loss = F.mse_loss(td_target, old_val)

            if self.current_timestep % 100 == 0:
                self.writer.add_scalar("losses/td_loss", loss, self.current_timestep)
                self.writer.add_scalar("losses/q_values", old_val.mean().item(), self.current_timestep)
                self.writer.add_scalar("losses/motor_q_values", old_motor_val.mean().item(), self.current_timestep)
                self.writer.add_scalar("losses/sensor_q_values", old_sensory_val.mean().item(), self.current_timestep)
                self.writer.add_scalar("losses/sfn_loss", sfn_loss.item(), self.current_timestep)
                self.writer.add_scalar("losses/observ_r", observ_r.mean().item(), self.current_timestep)
                self.writer.add_scalar("losses/original_td_target", original_td_target.mean().item(), self.current_timestep)
                self.writer.add_scalar("losses/sugarl_r_scaled_td_target", td_target.mean().item(), self.current_timestep)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update the target network
        if (self.current_timestep // self.n_envs) % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def evaluate(self, env_name: str, experiment_name: str):
        # Set networks to eval mode
        self.q_network.eval()
        self.sfn.eval()
        
        eval_episodic_returns, eval_episodic_lengths, eval_ns_pauses = [], [], []
        eval_prevented_pause_actions = []

        for eval_ep in range(self.n_evals):
            eval_env = self.eval_env_generator(eval_ep)
            single_eval_env = eval_env.envs[0] if isinstance(eval_env, VectorEnv) else eval_env
            n_eval_envs = eval_env.num_envs if isinstance(eval_env, VectorEnv) else 1

            obs_eval, _ = eval_env.reset()
            done = False

            pvm_buffer_eval = PVMBuffer(
                self.pvm_stack_size, 
                (n_eval_envs, self.channel_stack_size, *self.obs_size)
            )

            # One episode in the environment
            successive_pause_actions, prevented_pause_actions = 0, 0
            while not done:
                pvm_buffer_eval.append(obs_eval)
                pvm_obs_eval = pvm_buffer_eval.get_obs(mode="stack_max")

                motor_actions, sensory_actions = self.q_network.chose_eval_action(pvm_obs_eval)

                # Keep track of successive pause actions to prevent the system from halting
                successive_pause_actions += single_eval_env.pause_action
                if prevented_pause_actions > 50 or successive_pause_actions > 20:
                    # print("Warning: Using random action selection to prevent successive pause actions")
                    motor_actions, sensory_actions = self.q_network.chose_action(eval_env, pvm_obs_eval, epsilon=1.)
                    successive_pause_actions = 0
                    prevented_pause_actions += 1

                next_obs_eval, rewards, dones, _, infos = eval_env.step({
                    "motor": motor_actions, 
                    "sensory": [self.sensory_action_set[a] for a in sensory_actions]
                })
                obs_eval = next_obs_eval
                done = dones[0]

            eval_episodic_returns.append(infos['final_info'][0]["reward"])
            eval_episodic_lengths.append(infos['final_info'][0]["ep_len"])
            eval_ns_pauses.append(infos['final_info'][0]["n_pauses"])
            eval_prevented_pause_actions.append(prevented_pause_actions)

            # Save results as video and pytorch object
            # Only save 1/4th of the evals as videos
            if single_eval_env.env.record and eval_ep % 4 == 0:
                record_file_dir = os.path.join("recordings", experiment_name, os.path.basename(__file__).rstrip(".py"), env_name)
                os.makedirs(record_file_dir, exist_ok=True)
                record_file_fn = f"{env_name}_seed{args.seed}_step{self.current_timestep:07d}_eval{eval_ep:02d}_record.pt"
                single_eval_env.save_record_to_file(os.path.join(record_file_dir, record_file_fn))
                
                if eval_ep == 0:
                    # Safe the model file
                    model_file_dir = os.path.join("trained_models", experiment_name, os.path.basename(__file__).rstrip(".py"), env_name)
                    os.makedirs(model_file_dir, exist_ok=True)
                    model_fn = f"{env_name}_seed{args.seed}_step{self.current_timestep:07d}_model.pt"
                    torch.save({"sfn": self.sfn.state_dict(), "q": self.q_network.state_dict()}, os.path.join(model_file_dir, model_fn))
                        

        self.writer.add_scalar("charts/eval_episodic_return", np.mean(eval_episodic_returns), self.current_timestep)
        self.writer.add_scalar("charts/eval_episodic_return_std", np.std(eval_episodic_returns), self.current_timestep)
        print((
            f"[N: {self.current_timestep:07,d}]"
            f"[Eval R: {np.mean(eval_episodic_returns):.2f}+/-{np.std(eval_episodic_returns):.2f}]"
            f"[R list: {','.join([f'{r:.2f}' for r in eval_episodic_returns])}]"
            f"[Pauses: {','.join([str(n) for n in eval_ns_pauses])}]"
        ))
        if not all(n == 0 for n in eval_prevented_pause_actions):
            print(f"WARNING: [Prevented Pauses]: {','.join(map(str, eval_prevented_pause_actions))}")

        # Set the networks back to training mode
        self.q_network.train()
        self.sfn.train()

    def _epsilon_schedule(self, total_timesteps: int):
        """
        Maps the current number of timesteps to a value of epsilon.
        """
        return linear_schedule(*self.epsilon_interval, self.exploration_fraction * total_timesteps, self.current_timestep)

if __name__ == "__main__":
    args = parse_args()
    args.env = args.env.lower()
    run_name = f"{args.env}__{os.path.basename(__file__)}__{args.seed}__{get_timestr()}"
    run_dir = os.path.join("runs", args.exp_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    assert device.type == "cuda"

    sugarl_r_scale = get_sugarl_reward_scale_atari(args.env)

    env = make_train_env()
    agent = CRDQN(
        env, 
        make_eval_env, 
        writer, 
        sugarl_r_scale,
        fov_size=args.fov_size,
        replay_buffer_size=args.buffer_size,
        learning_start=args.learning_start
    )

    agent.learn(args.total_timesteps, args.env, args.exp_name)