from typing import Optional, Union, Callable, Tuple, List, Dict
from itertools import product
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.transforms import Resize

import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Discrete
from ray.train import report

from active_gym import FixedFovealEnv
from atari_cr.common.pauseable_env import PauseableFixedFovealEnv
from atari_cr.common.models import SensoryActionMode
from atari_cr.common.buffers import DoubleActionReplayBuffer
from atari_cr.common.pvm_buffer import PVMBuffer
from atari_cr.common.utils import linear_schedule
from atari_cr.agents.dqn_atari_cr.networks import QNetwork, SelfPredictionNetwork
from atari_cr.common.utils import gradfilter_ema

class CRDQN:
    """
    Algorithm for DQN with Computational Rationality
    """
    def __init__(
            self, 
            env: Union[PauseableFixedFovealEnv, FixedFovealEnv], 
            eval_env_generator: Callable[[int], gym.Env],
            sugarl_r_scale: float,
            seed = 0,
            fov_size = 50,
            sensory_action_space_granularity: Tuple[int] = (4, 4),
            learning_rate = 0.0001,
            replay_buffer_size = 100000,
            frame_stack = 4,
            pvm_stack = 3,
            epsilon_interval: Tuple[float] = (1., 0.01),
            exploration_fraction = 0.10,
            batch_size = 32,
            learning_start = 80000,
            train_frequency = 4,
            target_network_frequency = 1000,
            eval_frequency = -1,
            gamma = 0.99,
            cuda = True,
            n_evals = 10,
            ignore_sugarl = True,
            grokfast = False, 
            sensory_action_mode = SensoryActionMode.ABSOLUTE,
            writer: Optional[SummaryWriter] = None,
            disable_tensorboard = False,
            no_model_output = False,
            no_pvm_visualization = False,
            capture_video = True,
            agent_id = 0,
        ):
        """
        :param env `gymnasium.Env`:
        :param Callable eval_env_generator: Function, outputting an eval env given a seed
        :param float sugarl_r_scale:
        :param int seed:
        :param int fov_size:
        :param Tuple[int] sensory_action_space_granularity: The number of smallest sensory steps 
            it takes from left to right and from top to bottom
        :param float learning_rate: The learning rate used for the Q Network and Self Predicition Network
        "param int replay_buffer_size:
        :param int frame_stack: The number of frames being stacked as on observation by the atari environment
        :param int pvm_stack: The number of recent observations to be used for action selection
        :param Tuple[int] epsilon_interval: Interval in which the propability for a random action 
            epsilon moves from one end to the other during training
        :param float exploration_fraction: The fraction of the total learning time steps 
            it takes for epsilon to reach its end value
        :param int batch_size:
        :param int learning_start: The timestep at which to start training the Q Network
        :param int train_frequency: The number of timesteps between training sessions
        :param int target_network_frequency: The number of timesteps between target network updates
        :param int eval_frequency: The number of timesteps between evaluations; -1 for eval at the end
        :param float gamma: The discount factor gamma
        :param bool cuda: Whether to use cuda or not
        :param int n_evals: Number of eval episodes to be played
        :param bool ignore_sugarl: Whether to ignore the sugarl term in the loss calculation
        :param bool grokfast: Whether to use grokfast (https://doi.org/10.48550/arXiv.2405.20233)
        :param Optional[SummaryWriter] writer: Tensorboard writer. Creates a new one if None is passed
        :param int agent_id: Identifier for an agent when used together with other agents
        """
        self.env = env
        self.sugarl_r_scale = sugarl_r_scale
        self.seed = seed
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
        self.pvm_stack = pvm_stack
        self.frame_stack = frame_stack
        self.ignore_sugarl = ignore_sugarl
        self.sensory_action_mode = sensory_action_mode
        self.grokfast = grokfast
        self.writer = writer
        self.disable_tensorboard = disable_tensorboard
        self.no_model_output = no_model_output
        self.no_pvm_visualization = no_pvm_visualization
        self.capture_video = capture_video
        self.agent_id = agent_id

        self.n_envs = len(self.env.envs) if isinstance(self.env, VectorEnv) else 1
        self.current_timestep = 0

        # Get the observation size
        self.envs = env.envs if isinstance(env, VectorEnv) else [env]
        for env in self.envs:
            assert isinstance(env, (PauseableFixedFovealEnv, FixedFovealEnv)), \
                "The environment is expected to be wrapped in a PauseableFixedFovealEnv"
        self.obs_size = self.env.observation_space.shape[2:]
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
        assert self.device.type == "cuda", f"Set up cuda to run. Current device: {self.device.type}"

        # Q networks
        self.q_network = QNetwork(self.env, self.sensory_action_set).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = QNetwork(self.env, self.sensory_action_set).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Self Prediction Networks; used to judge the quality of sensory actions
        self.sfn = SelfPredictionNetwork(self.env).to(self.device)
        self.sfn_optimizer = Adam(self.sfn.parameters(), lr=learning_rate)

        # Grokking
        self.sfn_grads = None
        self.q_network_grads = None

        # Replay Buffer aka. Long Term Memory
        self.rb = DoubleActionReplayBuffer(
            replay_buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space["motor_action"],
            Discrete(len(self.sensory_action_set)),
            self.device,
            n_envs=self.env.num_envs if isinstance(self.env, VectorEnv) else 1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        # PVM Buffer aka. Short Term Memory, combining multiple observations
        self.pvm_buffer = PVMBuffer(pvm_stack, (self.n_envs, frame_stack, *self.obs_size))

    def learn(self, n: int, env_name: str, experiment_name: str):
        """
        Acts in the environment and trains the agent for n timesteps
        """
        # Define output paths
        run_identifier = os.path.join(experiment_name, env_name)
        self.run_dir = os.path.join("output/runs", run_identifier)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")
        self.video_dir = os.path.join(self.run_dir, "recordings")
        self.pvm_dir = os.path.join(self.run_dir, "pvms")
        self.model_dir = os.path.join(self.run_dir, "trained_models")
        if isinstance(self.envs[0], FixedFovealEnv):
            self.model_dir = os.path.join(self.model_dir, "no_pause")

        # Init text logging logging
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"seed{self.seed}.txt")

        # Init tensorboard logging
        if not self.disable_tensorboard:
            os.makedirs(self.tb_dir, exist_ok=True)
            if not self.writer:
                self.writer = SummaryWriter(os.path.join(self.tb_dir, f"seed{self.seed}"))
            hyper_params_table = "\n".join([f"|{key}|{value}|" for key, value in self.__dict__.items()])
            self.writer.add_text(
                "Agent Hyperparameters", 
                f"|param|value|\n|-|-|\n{hyper_params_table}",
            )

        # Log pause cost
        if isinstance(self.env, PauseableFixedFovealEnv):
            self._log(f"---\nTraining start with pause costs {[env.pause_cost for env in self.envs]}")

        # Load existing run if there is one
        if os.path.exists(self.model_dir):
            seeded_models = list(filter(lambda s: f"seed{self.seed}" in s, os.listdir(self.model_dir)))
            if len(seeded_models) > 0:
                self._log("Loading existing checkpoint")
                timesteps = [int(model.split("_")[1][4:]) for model in seeded_models]
                latest_model = seeded_models[np.argmax(timesteps)]
                self.load_checkpoint(f"{self.model_dir}/{latest_model}")
                n += self.current_timestep

        # Start acting in the environment
        self.start_time = time.time()
        obs, infos = self.env.reset()
        self.pvm_buffer.append(obs)
        pvm_obs = self.pvm_buffer.get_obs(mode="stack_max")

        # Init return value
        eval_returns = []

        while self.current_timestep < n:
            # Chose action from q network
            self.epsilon = self._epsilon_schedule(n)
            motor_actions, sensory_action_indices = self.q_network.chose_action(self.env, pvm_obs, self.epsilon, self.device)

            # Transform the action to an absolute fovea position
            sensory_actions = np.array([self.sensory_action_set[i] for i in sensory_action_indices])

            # Perform the action in the environment
            next_pvm_obs, rewards, dones, _ = self._step(
                self.env, 
                self.pvm_buffer,
                motor_actions, 
                sensory_actions,
            )

            # Add new pvm ovbervation to the buffer   
            self.rb.add(pvm_obs, next_pvm_obs, motor_actions, sensory_action_indices, rewards, dones, {})
            pvm_obs = next_pvm_obs

            # Only train if a full batch is available
            self.current_timestep += self.n_envs
            if self.current_timestep > self.batch_size:

                # Save the model every 1M timesteps
                if (not self.no_model_output) and self.current_timestep % 1000000 == 0:
                    self._save_output(self.model_dir, "pt", self.save_checkpoint)

                # Training
                if self.current_timestep % self.train_frequency == 0:
                    self.train()

                # Evaluation
                if (self.current_timestep % self.eval_frequency == 0 and self.eval_frequency > 0) or \
                (self.current_timestep >= n):
                    eval_returns = self.evaluate()

        self.env.close()
        if not self.disable_tensorboard:
            self.writer.close()

        return eval_returns

    def train(self):
        """
        Performs one training iteration from the replay buffer
        """
        # Replay buffer sampling
        # Counter-balance the true global transitions used for training
        data = self.rb.sample(self.batch_size // self.n_envs)

        # SFN training
        observation_quality = self._train_sfn(data)

        # DQN training
        if self.current_timestep > self.learning_start:
            self._train_dqn(data, observation_quality)

    def evaluate(self):
        # Set networks to eval mode
        self.q_network.eval()
        self.sfn.eval()
        
        episode_infos = []
        for eval_ep in range(self.n_evals):
            # Create env
            eval_env = self.eval_env_generator(eval_ep)
            unwrapped_eval_env = eval_env.envs[0] if isinstance(eval_env, VectorEnv) else eval_env
            n_eval_envs = eval_env.num_envs if isinstance(eval_env, VectorEnv) else 1

            # Init env
            obs, _ = eval_env.reset()
            done = False
            eval_pvm_buffer = PVMBuffer(
                self.pvm_stack, 
                (n_eval_envs, self.frame_stack, *self.obs_size)
            )
            eval_pvm_buffer.append(obs)
            pvm_obs = eval_pvm_buffer.get_obs(mode="stack_max")

            # One episode in the environment
            while not done:
                # Chose an action from the Q network
                motor_actions, sensory_action_indices \
                    = self.q_network.chose_eval_action(pvm_obs, self.device)
                
                # Translate the action to an absolute fovea position
                sensory_actions = np.array([self.sensory_action_set[i] for i in sensory_action_indices])

                # Perform the action in the environment
                next_pvm_obs, rewards, dones, infos = self._step(
                    eval_env, 
                    eval_pvm_buffer,
                    motor_actions,
                    sensory_actions,
                    eval=True
                )
                done = dones[0]
                pvm_obs = next_pvm_obs

                # Save a visualization of the pvm buffer in the middle of the episode
                if (not self.no_pvm_visualization) and infos["ep_len"] == 50:
                    self._save_output(self.pvm_dir, "png", eval_pvm_buffer.to_png, eval_ep)

            episode_infos.append(infos['final_info'][0])

            # Save results as video and pytorch object
            # Only save 1/4th of the evals as videos
            if (self.capture_video) and unwrapped_eval_env.record and eval_ep % 4 == 0:
                self._save_output(self.video_dir, "pt", unwrapped_eval_env.save_record_to_file, eval_ep)
                
            # Safe the model file in the first eval run
            if (not self.no_model_output) and eval_ep == 0:
                self._save_output(self.model_dir, "pt", self.save_checkpoint, eval_ep)

            eval_env.close()

        # Log results
        self._log_eval_episodes(episode_infos)

        # Set the networks back to training mode
        self.q_network.train()
        self.sfn.train()

        eval_returns: List[float] = [episode_info["reward"] for episode_info in episode_infos]
        return eval_returns

    def save_checkpoint(self, file_path: str):
        torch.save(
            {
                "sfn": self.sfn.state_dict(), 
                "q": self.q_network.state_dict(), 
                "training_steps": self.current_timestep
            }, 
            file_path
        )

    def load_checkpoint(self, file_path: str):
        checkpoint = torch.load(file_path, weights_only=True)
        self.sfn.load_state_dict(checkpoint["sfn"])
        self.q_network.load_state_dict(checkpoint["q"])
        self.current_timestep = checkpoint["training_steps"]

    def _log(self, s: str):
        """
        Own print function. logging module does not work with the current gymnasium installation for some reason.
        """
        assert self.log_file, "self._log needs self.log_file to bet set"
        with open(self.log_file, "a") as f:
            f.write(f"\n{s}")
    
    def _save_output(self, output_dir: str, file_prefix: str, save_fn: Callable[[str], None], eval_ep: int = 0):
        """
        Saves different types of eval output to the file system in the context of the current episode
        """
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"seed{self.seed}_step{self.current_timestep:07d}_eval{eval_ep:02d}.{file_prefix}"
        if isinstance(self.env, FixedFovealEnv):
            file_name = f"seed{self.seed}_step{self.current_timestep:07d}_eval{eval_ep:02d}_no_pause.{file_prefix}"
        save_fn(os.path.join(output_dir, file_name))

    def _step(self, env: gym.Env, pvm_buffer: PVMBuffer, motor_actions, sensory_actions, eval = False):
        """
        Given an action, the agent does one step in the environment, 
        returning the next observation
        """
        # Take an action in the environment
        next_obs, rewards, dones, _, infos = env.step({
            "motor_action": motor_actions, 
            "sensory_action": sensory_actions
        })

        # Log episode returns and handle `terminal_observation`
        if not eval and "final_info" in infos and True in dones:
            finished_env_index = np.argmax(dones)
            self._log_episode(finished_env_index, infos)
            next_obs[finished_env_index] = infos["final_observation"][finished_env_index]

        # Get the next pvm observation
        pvm_buffer.append(next_obs)
        next_pvm_obs = pvm_buffer.get_obs(mode="stack_max")

        return next_pvm_obs, rewards, dones, infos

    def _log_episode(self, finished_env_index: int, infos):
        # Prepare the episode infos for the different supported envs
        episode_info = infos['final_info'][finished_env_index]
        if isinstance(self.envs[0], FixedFovealEnv):
            episode_info["n_pauses"], episode_info['pause_cost'] = 0, 0
            episode_info["no_action_pauses"], episode_info['prevented_pauses'] = 0, 0
            prevented_pause_counts = [0] * len(self.envs)
        elif isinstance(self.envs[0], PauseableFixedFovealEnv):
            prevented_pause_counts = [env.prevented_pauses for env in self.envs]
        else:
            raise ValueError(f"Environment '{self.envs[0]}' not supported")

        # Reward without pause costs
        raw_reward = episode_info['reward'] + episode_info['n_pauses'] * episode_info['pause_cost']
        prevented_pauses_warning = f"\nWARNING: [Prevented Pauses: {episode_info['prevented_pauses']}]" if episode_info['prevented_pauses'] else "" 

        self._log((
            f"[T: {time.time()-self.start_time:.2f}] "
            f"[N: {self.current_timestep:07,d}] "
            f"[R, Raw R: {episode_info['reward']:.2f}, {raw_reward:.2f}] "
            f"[Pauses: {episode_info['n_pauses']}] "
            f"{prevented_pauses_warning}"
        ))    
        # Log the amount of prevented pauses over the entire learning period
        if not all(prevented_pause_counts) == 0:
            self._log(f"WARNING: [Prevented Pauses: {','.join(map(str, prevented_pause_counts))}]")

        # Tensorboard
        if not self.disable_tensorboard:
            self.writer.add_scalar("charts/episodic_return", episode_info["reward"], self.current_timestep)
            self.writer.add_scalar("charts/episode_length", episode_info["ep_len"], self.current_timestep)
            self.writer.add_scalar("charts/epsilon", self.epsilon, self.current_timestep)
            self.writer.add_scalar("charts/raw_episodic_return", raw_reward, self.current_timestep)
            self.writer.add_scalar("charts/pauses", episode_info['n_pauses'], self.current_timestep)
            self.writer.add_scalar("charts/prevented_pauses", episode_info['prevented_pauses'], self.current_timestep)
            self.writer.add_scalar("charts/no_action_pauses", episode_info["no_action_pauses"], self.current_timestep)

        # Ray
        report({"episode_reward": episode_info["reward"]})

    def _log_eval_episodes(self, episode_infos: List[Dict]):
        # Unpack episode_infos
        episodic_returns, episode_lengths = [], []
        pause_counts, prevented_pauses = [], []
        no_action_pauses = []
        for episode_info in episode_infos:
            
            # Prepare the episode infos for the different supported envs
            if isinstance(self.envs[0], FixedFovealEnv):
                episode_info["n_pauses"], episode_info['pause_cost'] = 0, 0
                episode_info["no_action_pauses"], episode_info['prevented_pauses'] = 0, 0

            episodic_returns.append(episode_info["reward"])
            episode_lengths.append(episode_info["ep_len"])
            pause_counts.append(episode_info["n_pauses"])
            prevented_pauses.append(episode_info["prevented_pauses"])
            no_action_pauses.append(episode_info["no_action_pauses"])

        pause_cost = episode_info["pause_cost"]
        raw_episodic_returns = [episodic_return + pause_cost * pauses for episodic_return, pauses in zip(episodic_returns, pause_counts)]
        
        # Log everything
        prevented_pauses_warning = "" if all(n == 0 for n in prevented_pauses) else \
            f"\nWARNING: [Prevented Pauses]: {','.join(map(str, prevented_pauses))}"
        self._log((
            f"[N: {self.current_timestep:07,d}]"
            f" [Eval Return, Raw Eval Return: {np.mean(episodic_returns):.2f}+/-{np.std(episodic_returns):.2f}"
                f", {np.mean(raw_episodic_returns):.2f}+/-{np.std(raw_episodic_returns):.2f}]"
            f"\n[Returns: {','.join([f'{r:.2f}' for r in episodic_returns])}]"
            f"\n[Episode Lengths: {','.join([f'{r:.2f}' for r in episode_lengths])}]"
            f"\n[Pauses: {','.join([str(n) for n in pause_counts])} with cost {pause_cost}]{prevented_pauses_warning}"
        ))

        # Tensorboard
        if not self.disable_tensorboard:
            self.writer.add_histogram("eval/pause_counts", np.array(pause_counts), self.agent_id)
            self.writer.add_histogram("eval/episodic_return", np.array(episodic_returns), self.agent_id)
            self.writer.add_histogram("eval/raw_episodic_return", np.array(raw_episodic_returns), self.agent_id)
            self.writer.add_histogram("eval/episode_lengths", np.array(episode_lengths), self.agent_id)

            hparams = {
                "fov_size": self.fov_size,
                "pvm_stack": self.pvm_stack,
                "frame_stack": self.frame_stack,
                "sensory_action_mode": self.sensory_action_mode,
                "grokfast": self.grokfast,
            }
            if isinstance(self.envs[0], PauseableFixedFovealEnv):
                hparams.update({
                    "pause_cost": self.envs[0].pause_cost,
                    "no_action_pause_cost": self.envs[0].no_action_pause_cost,
                })
            metrics = {
                "hp/episodic_return": np.mean(episodic_returns), 
                "hp/raw_episodic_returns": np.mean(raw_episodic_returns),
                "hp/episode_lengths": np.mean(episode_lengths),
                "hp/pause_counts": np.mean(pause_counts),
                "hp/prevented_pauses": np.mean(prevented_pauses),
                "hp/no_action_pauses": np.mean(no_action_pauses)
            }
            self.writer.add_hparams(hparams, metrics) 

    def _train_sfn(self, data):
        # Prediction
        concat_observation = torch.concat([data.next_observations, data.observations], dim=1) 
        pred_motor_actions = self.sfn(Resize(self.obs_size)(concat_observation))
        self.sfn_loss = self.sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())

        # Back propagation
        self.sfn_optimizer.zero_grad()
        self.sfn_loss.backward()
        if self.grokfast:
            self.sfn_grads = gradfilter_ema(self.sfn, grads=self.sfn_grads)
        self.sfn_optimizer.step()

        # Return the probabilites the sfn would have also selected the truely selected action, given the limited observation
        # Higher probabilities suggest better information was provided from the visual input
        observation_quality = F.softmax(pred_motor_actions).gather(1, data.motor_actions).squeeze().detach()

        # Tensorboard
        if not self.disable_tensorboard:
            if self.current_timestep % 100 == 0:
                sfn_accuray = (pred_motor_actions.argmax(axis=1) == data.motor_actions.flatten()).sum() / pred_motor_actions.shape[0]
                self.writer.add_scalar("losses/sfn_loss", self.sfn_loss.item(), self.current_timestep)
                self.writer.add_scalar("losses/sfn_accuray", sfn_accuray, self.current_timestep)
                self.writer.add_scalar("losses/observation_quality", observation_quality.mean().item(), self.current_timestep)
        
        return observation_quality

    def _train_dqn(self, data, observation_quality):
        """
        Trains the behavior q network and copies it to the target q network with self.target_network_frequency.

        :param NDArray data: A sample from the replay buffer
        :param NDArray[Shape[self.batch_size], Float] observation_quality: A batch of probabilities of the SFN predicting the action that the agent selected  
        """
        # TODO: Investigate how pausing interacts with sugarl reward
        # Target network prediction
        with torch.no_grad():
            # Assign a value to every possible action in the next state for one batch 
            # motor_target.shape: [32, 19]
            motor_target, sensory_target = self.target_network(Resize(self.obs_size)(data.next_observations))
            # Get the maximum action value for one batch
            # motor_target_max.shape: [32]
            motor_target_max, _ = motor_target.max(dim=1)
            sensory_target_max, _ = sensory_target.max(dim=1)
            # Scale step-wise reward with observation_quality
            observation_quality_adjusted = observation_quality.clone()
            observation_quality_adjusted[data.rewards.flatten() > 0] = 1 - observation_quality_adjusted[data.rewards.flatten() > 0]
            td_target = data.rewards.flatten() - (1 - observation_quality) * self.sugarl_r_scale + self.gamma * (motor_target_max + sensory_target_max) * (1 - data.dones.flatten())
            original_td_target = data.rewards.flatten() + self.gamma * (motor_target_max + sensory_target_max) * (1 - data.dones.flatten())

        # Q network prediction
        old_motor_q_val, old_sensory_q_val = self.q_network(Resize(self.obs_size)(data.observations))
        old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
        old_sensory_val = old_sensory_q_val.gather(1, data.sensory_actions).squeeze()
        old_val = old_motor_val + old_sensory_val

        # Back propagation
        loss_without_sugarl = F.mse_loss(original_td_target, old_val)
        loss = F.mse_loss(td_target, old_val)
        backprop_loss = loss_without_sugarl if self.ignore_sugarl else loss
        self.optimizer.zero_grad()
        backprop_loss.backward()
        if self.grokfast:
            self.q_network_grads = gradfilter_ema(self.q_network, grads=self.q_network_grads)
        self.optimizer.step()

        # Tensorboard logging
        if (not self.disable_tensorboard) and self.current_timestep % 100 == 0:
            self.writer.add_scalar("losses/loss", loss, self.current_timestep)
            self.writer.add_scalar("losses/loss_without_sugarl", loss, self.current_timestep)
            self.writer.add_scalar("losses/sugarl_loss", loss - loss_without_sugarl, self.current_timestep)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), self.current_timestep)
            self.writer.add_scalar("losses/motor_q_values", old_motor_val.mean().item(), self.current_timestep)
            self.writer.add_scalar("losses/sensor_q_values", old_sensory_val.mean().item(), self.current_timestep)
            # self.writer.add_scalar("losses/original_td_target", original_td_target.mean().item(), self.current_timestep)
            self.writer.add_scalar("losses/sugarl_r_scaled_td_target", td_target.mean().item(), self.current_timestep)
        
        # Update the target network with self.target_network_frequency
        if (self.current_timestep // self.n_envs) % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _epsilon_schedule(self, total_timesteps: int):
        """
        Maps the current number of timesteps to a value of epsilon.
        """
        return linear_schedule(*self.epsilon_interval, self.exploration_fraction * total_timesteps, self.current_timestep)