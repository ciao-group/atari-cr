from collections import deque
from typing import Optional, Union, Callable, Tuple, List
from itertools import product
import os
import time
import polars as pl

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.transforms import Resize

from ray import train
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Discrete

from active_gym import FixedFovealEnv
from atari_cr.atari_head.dataset import GazeDataset
from atari_cr.atari_head.gaze_predictor import GazePredictor
from atari_cr.pauseable_env import PauseableFixedFovealEnv
from atari_cr.models import DurationInfo, EpisodeInfo, EpisodeRecord, TdUpdateInfo
from atari_cr.buffers import DoubleActionReplayBuffer, DoubleActionReplayBufferSamples
from atari_cr.pvm_buffer import PVMBuffer
from atari_cr.utils import linear_schedule
from atari_cr.agents.dqn_atari_cr.networks import QNetwork, SelfPredictionNetwork

class CRDQN:
    """
    Algorithm for DQN with Computational Rationality
    """
    def __init__(
            self,
            env: Union[PauseableFixedFovealEnv, FixedFovealEnv],
            eval_env_generator: Callable[[int], Union[
                VectorEnv, FixedFovealEnv, PauseableFixedFovealEnv]],
            sugarl_r_scale: float,
            env_name: str,
            seed = 0,
            fov_size = 50,
            sensory_action_space_quantization: Tuple[int] = (4, 4),
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
            no_model_output = False,
            no_pvm_visualization = False,
            capture_video = True,
            agent_id = 0,
            debug = False,
            evaluator: Optional[GazePredictor] = None,
            pause_feat = False,
            s_action_feat = False,
            td_steps = 1,
            checkpoint=""
        ):
        """
        :param env `gymnasium.Env`:
        :param Callable eval_env_generator: Function, outputting an eval env given a
            seed
        :param float sugarl_r_scale:
        :param int seed:
        :param int fov_size:
        :param Tuple[int] sensory_action_space_quantization: The number of smallest
            sensory steps it takes from left to right and from top to bottom
        :param float learning_rate: The learning rate used for the Q Network and Self
            Predicition Network
        "param int replay_buffer_size:
        :param int frame_stack: The number of frames being stacked as on observation by
            the atari environment
        :param int pvm_stack: The number of recent observations to be used for action
            selection
        :param Tuple[int] epsilon_interval: Interval in which the propability for a
            random action epsilon moves from one end to the other during training
        :param float exploration_fraction: The fraction of the total learning time steps
            it takes for epsilon to reach its end value
        :param int batch_size:
        :param int learning_start: The timestep at which to start training the Q Network
        :param int train_frequency: The number of timesteps between training sessions
        :param int target_network_frequency: The number of timesteps between target
            network updates
        :param int eval_frequency: The number of timesteps between evaluations; -1 for
            eval at the end
        :param float gamma: The discount factor gamma
        :param bool cuda: Whether to use cuda or not
        :param int n_evals: Number of eval episodes to be played
        :param bool ignore_sugarl: Whether to ignore the sugarl term in the loss
            calculation
        :param int agent_id: Identifier for an agent when used together with other
            agents
        :param GazePredictor evaluator: Supervised learning model trained on human data.
            Reference for how human plausible the models' gazes are
        """
        self.env = env
        self.sugarl_r_scale = sugarl_r_scale
        self.env_name = env_name
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
        self.no_model_output = no_model_output
        self.no_pvm_visualization = no_pvm_visualization
        self.capture_video = capture_video
        self.agent_id = agent_id
        self.debug = debug
        self.evaluator = evaluator
        self.td_steps = td_steps
        self.checkpoint = checkpoint

        self.n_envs = len(self.env.envs) if isinstance(self.env, VectorEnv) else 1
        self.timestep = 0

        # Get the observation size
        self.envs = env.envs if isinstance(env, VectorEnv) else [env]
        self.obs_size = self.env.observation_space.shape[2:]
        assert len(self.obs_size) == 2, "The CRDQN agent only supports 2D Environments"

        # Get the sensory action set as a list of discrete actions
        # How far can the fovea move from left to right and from top to bottom
        max_sensory_action_step = np.array(self.obs_size) - np.array(
            [self.fov_size, self.fov_size])
        discrete_coords = [np.linspace(0, max_sensory_action_step[i],
            sensory_action_space_quantization[i], endpoint=False).astype(int)
            for i in [0,1]]
        # Discrete action set as cross product of possible x and y steps
        self.sensory_action_set = np.stack(
            np.meshgrid(*discrete_coords)).T.reshape((-1,2))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cuda else "cpu")
        assert self.device.type == "cuda", \
            f"Set up cuda to run. Current device: {self.device.type}"

        # Q networks
        self.q_network = QNetwork(
            self.env, len(self.sensory_action_set), pause_feat, s_action_feat,
            self.env.envs[0].pause_action
        ).to(self.device)
        self.optimizer = Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = QNetwork(
            self.env, len(self.sensory_action_set), pause_feat, s_action_feat,
            self.env.envs[0].pause_action
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Self Prediction Networks; used to judge the quality of sensory actions
        self.sfn = SelfPredictionNetwork(self.env).to(self.device)
        self.sfn_optimizer = Adam(self.sfn.parameters(), lr=learning_rate)

        init_sensory_action = np.where(
            (self.sensory_action_set == self.env.envs[0].fov_init_loc)
            .all(axis=1))[0]
        # Replay Buffer aka. Long Term Memory
        self.rb = DoubleActionReplayBuffer(
            replay_buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space["motor_action"],
            Discrete(len(self.sensory_action_set)),
            init_sensory_action,
            self.device,
            n_envs=self.env.num_envs if isinstance(self.env, VectorEnv) else 1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            td_steps=td_steps,
        )

        # PVM Buffer aka. Short Term Memory, combining multiple observations
        self.pvm_buffer = PVMBuffer(
            pvm_stack, (self.n_envs, frame_stack, *self.obs_size))

        self.auc = 0.7
        self.auc_window = deque(maxlen=5)
        self.windowed_auc = self.auc

    def learn(self, n: int, experiment_name: str):
        """
        Acts in the environment and trains the agent for n timesteps
        """
        # Define output paths
        run_identifier = os.path.join(experiment_name, self.env_name)
        self.run_dir = os.path.join("output/runs", run_identifier)
        self.video_dir = os.path.join(self.run_dir, "recordings")
        self.pvm_dir = os.path.join(self.run_dir, "pvms")
        self.model_dir = os.path.join(self.run_dir, "trained_models")

        # Load existing run if there is one
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
            n += self.timestep
            self.learning_start += self.timestep
        elif not self.debug and os.path.exists(self.model_dir):
            seeded_models = list(filter(
                lambda s: f"seed{self.seed}" in s, os.listdir(self.model_dir)))
            if len(seeded_models) > 0:
                timesteps = [int(model.split("_")[1][4:]) for model in seeded_models]
                latest_model = seeded_models[np.argmax(timesteps)]
                self.load_checkpoint(f"{self.model_dir}/{latest_model}")
                n += self.timestep

        # Start acting in the environment
        self.start_time = time.time()
        obs, infos = self.env.reset()
        self.pvm_buffer.append(obs)
        pvm_obs = self.pvm_buffer.get_obs(mode="stack_max") # -> [1,4,84,84]

        # Init return value
        eval_returns = []
        td_update = None

        while self.timestep < n:
            # Cast sensory action ids to coordinates and remove the env axis
            prev_sensory_actions, consecutive_pauses = self._preprocess_features()

            # Chose action from q network
            self.epsilon = self._epsilon_schedule(n)
            motor_actions, sensory_action_ids = self.q_network.chose_action(
                pvm_obs, self.device,
                self._sensory_action_ids(np.array([e.fov_loc for e in self.env.envs])),
                consecutive_pauses, prev_sensory_actions, self.epsilon)

            # Transform the action to an absolute fovea position
            sensory_actions = self.sensory_action_set[sensory_action_ids]

            # Consecutive pauses as part of the observation
            consecutive_pauses = np.array([e.consecutive_pauses for e in self.env.envs])
            # Perform the action in the environment
            next_pvm_obs, rewards, dones, truncateds, infos = self._step(
                self.env,
                self.pvm_buffer,
                motor_actions,
                sensory_actions,
                td_update,
            )

            # Add new pvm ovbervation to the buffer
            self.rb.add(pvm_obs, next_pvm_obs, motor_actions, sensory_action_ids,
                        rewards, dones, consecutive_pauses)
            pvm_obs = next_pvm_obs

            # Add the sum of all substeps across all envs to the timestep
            self.timestep += sum([len(infos["final_info"][i]["raw_reward"] \
                if "final_info" in infos else infos["raw_reward"][i]) \
                    for i in range(self.n_envs)])
            # Only train if a full batch is available
            if self.timestep < self.batch_size:
                continue

            # Save the model every 1M timesteps
            if (not self.no_model_output) and self.timestep % 1000000 == 0:
                self._save_output(self.model_dir, "pt", self.save_checkpoint)

            # Training from the replay buffer
            if self.timestep % self.train_frequency == 0 \
                    and self.timestep > self.learning_start:
                # Replay buffer sampling
                # Counter-balance the true global transitions used for training
                data = self.rb.sample(self.batch_size // self.n_envs)
                # SFN and DQN training
                observation_quality = self._train_sfn(data)
                td_update = self._train_dqn(data, observation_quality)

            # Evaluation
            if (self.timestep % self.eval_frequency == 0 and
                self.eval_frequency > 0) or (self.timestep >= n):
                eval_returns, out_paths = self.evaluate(td_update)

            # Test against Atari-HEAD gaze predictor
            if self.timestep % 100_000 == 0:
                eval_returns, out_paths = self.evaluate(td_update, file_output=False)

        self.env.close()

        return eval_returns, out_paths

    def evaluate(self, td_update: TdUpdateInfo, file_output = True):
        # Set networks to eval mode
        self.q_network.eval()
        self.sfn.eval()

        episode_infos, out_paths, aucs = [], [], []
        for eval_ep in range(self.n_evals):
            # Create env
            eval_env = self.eval_env_generator(eval_ep) # VecEnv with a single env in it
            single_eval_env: Union[FixedFovealEnv, PauseableFixedFovealEnv] = \
                eval_env.envs[0] if isinstance(eval_env, VectorEnv) else eval_env
            n_eval_envs = eval_env.num_envs if isinstance(eval_env, VectorEnv) else 1

            # Init env
            obs, _ = eval_env.reset()
            done, truncated = False, False
            eval_pvm_buffer = PVMBuffer(
                self.pvm_stack,
                (n_eval_envs, self.frame_stack, *self.obs_size)
            )
            eval_pvm_buffer.append(obs)
            pvm_obs = eval_pvm_buffer.get_obs(mode="stack_max") # -> [1,4,84,84]

            # Save the pvm obs for logging
            if self.capture_video:
                pvm_observations = [pvm_obs[0,-1,...]] # -> [1,84,84]

            # One episode in the environment
            while not (done or truncated):
                prev_sensory_actions, consecutive_pauses = self._preprocess_features()

                # Chose an action from the Q network
                motor_actions, sensory_action_ids \
                    = self.q_network.chose_action(
                        pvm_obs, self.device,
                        self._sensory_action_ids(
                            np.array([e.fov_loc for e in eval_env.envs])),
                        consecutive_pauses, prev_sensory_actions
                    )

                # Forcefully do a pause some of the time in debug mode
                if isinstance(single_eval_env, PauseableFixedFovealEnv) and self.debug \
                    and np.random.choice([False, True], p=[0.5, 0.5]):
                    motor_actions = np.full(
                        motor_actions.shape, single_eval_env.pause_action)
                    # Also change the sensory_action
                    sensory_action_ids = np.full(
                        sensory_action_ids.shape,
                        np.random.randint(len(self.sensory_action_set)))

                # Translate the action to an absolute fovea position
                sensory_actions = self.sensory_action_set[sensory_action_ids]

                # Perform the action in the environment
                pvm_obs, rewards, dones, truncateds, infos = self._step(
                    eval_env,
                    eval_pvm_buffer,
                    motor_actions,
                    sensory_actions,
                    eval=True
                )
                done, truncated = dones[0], truncateds[0]

                if self.capture_video:
                    # Add the last observation again for every time the agent has
                    # skipped a frame
                    skipped_frames = len(infos["final_info"][0]["reward"]) - 1 \
                        if "final_info" in infos else len(infos["reward"][0]) - 1
                    for _ in range(skipped_frames):
                        pvm_observations.append(pvm_observations[-1])
                    pvm_observations.append(pvm_obs[0,-1,...]) # -> [N,84,84]

            info = infos["final_info"][0]
            if isinstance(eval_env.envs[0], PauseableFixedFovealEnv):
                info = info["episode_info"][-1]
            episode_infos.append(info)

            # At the end of every eval episode:
            if file_output:
                # Save a visualization of the pvm buffer at the end of the episode
                if (not self.no_pvm_visualization):
                    self._save_output(
                        self.pvm_dir, "png", eval_pvm_buffer.to_png, eval_ep)

                # Save results as video and csv file
                # Only save 1/4th of the evals as videos
                if (self.capture_video) and single_eval_env.record and eval_ep % 4 == 0:
                    if isinstance(single_eval_env, PauseableFixedFovealEnv):
                        single_eval_env.prev_episode.obs = np.stack(pvm_observations)
                        save_fn = lambda s: single_eval_env.prev_episode.save( # noqa: E731
                            s, with_obs=True)
                        extension = ""
                    else:
                        save_fn = single_eval_env.save_record_to_file
                        extension = "pt"
                    out_paths.append(self._save_output(
                        self.video_dir, extension, save_fn, eval_ep))

                # Safe the model file in the first eval run
                if (not self.no_model_output) and eval_ep == 0:
                    self._save_output(
                        self.model_dir, "pt", self.save_checkpoint, eval_ep)

            # AUC calculation
            duration_info = None
            episode_record = single_eval_env.prev_episode \
                if isinstance(single_eval_env, PauseableFixedFovealEnv) \
                else EpisodeRecord.from_record_buffer(
                    single_eval_env.env.prev_record_buffer)
            # No evaluation if the episode consists of only pauses
            if episode_record.annotations["pauses"].sum() < \
                    len(episode_record.annotations):
                if self.evaluator:
                    dataset = GazeDataset.from_game_data([episode_record])
                    loader = dataset.to_loader()
                    aucs.append(self.evaluator.eval(loader)["auc"])
                # Duration error calculation
                duration_info = DurationInfo.from_episodes(
                    [episode_record], self.env_name)

            eval_env.close()

        # Log mean result of eval episodes
        mean_episode_info: EpisodeInfo = \
            pl.DataFrame(episode_infos).mean().row(0, named=True)
        self._log_episode(mean_episode_info, td_update, duration_info, eval_env=True)

        # AUC and windowed AUC
        if self.evaluator:
            if aucs: self.auc = sum(aucs) / len(aucs)
            self.auc_window.append(self.auc)
            self.windowed_auc = sum(self.auc_window) / len(self.auc_window)

        # Set the networks back to training mode
        self.q_network.train()
        self.sfn.train()

        eval_returns: List[float] = [
            episode_info["raw_reward"] for episode_info in episode_infos]
        return eval_returns, out_paths

    def save_checkpoint(self, file_path: str):
        torch.save(
            {
                "sfn": self.sfn.state_dict(),
                "q": self.q_network.state_dict(),
                "training_steps": self.timestep
            },
            file_path
        )

    def load_checkpoint(self, file_path: str):
        checkpoint = torch.load(file_path, weights_only=True)
        self.sfn.load_state_dict(checkpoint["sfn"])
        self.q_network.load_state_dict(checkpoint["q"])
        self.timestep = checkpoint["training_steps"]

    def _save_output(self, output_dir: str, file_prefix: str,
                     save_fn: Callable[[str], None], eval_ep: int = 0):
        """
        Saves different types of eval output to the file system in the context of the
        current episode
        """
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"seed{self.seed}_step{self.timestep:07d}_eval{eval_ep:02d}"
        if file_prefix: file_name += f".{file_prefix}"
        out_path = os.path.join(output_dir, file_name)
        if not file_prefix: os.makedirs(out_path, exist_ok=True)
        save_fn(out_path)
        return out_path

    def _step(self, env: VectorEnv, pvm_buffer: PVMBuffer, motor_actions: np.ndarray,
              sensory_actions: np.ndarray, td_update: Optional[TdUpdateInfo] = None,
              eval = False):
        """
        Given an action, the agent does one step in the environment,
        returning the next observation

        :param Array[n_envs] motor_actions: Numpy array containing motor action for all
            parallel training envs.
        """
        # Take an action in the environment
        next_obs, rewards, dones, truncateds, infos = env.step({
            "motor_action": motor_actions,
            "sensory_action": sensory_actions
        })
        dones = dones | truncateds

        # Log episode returns and handle `terminal_observation`
        if not eval and "final_info" in infos and True in dones:
            finished_env_idx = np.argmax(dones)
            episode_info = infos['final_info'][finished_env_idx]
            next_obs[finished_env_idx] = infos["final_observation"][finished_env_idx]
            if isinstance(env.envs[0], PauseableFixedFovealEnv):
                episode_info = episode_info["episode_info"][-1]
            else: episode_info["raw_reward"] = episode_info["reward"]

            # Calculate gaze duration distribution and deviation from Atari-HEAD
            duration_info = None
            if isinstance(env.envs[0], PauseableFixedFovealEnv):
                duration_info = DurationInfo.from_episodes(
                    [e.prev_episode for (e, done) in zip(env.envs, dones) if done],
                    self.env_name
                )

            self._log_episode(episode_info, td_update, duration_info, eval_env=False)

        # Update the latest observation in the pvm buffer
        assert len(env.envs) == 1, \
            "Vector env with more than one env not supported for the following code"
        if isinstance(env.envs[0], PauseableFixedFovealEnv) \
            and motor_actions[0] == env.envs[0].pause_action:
            pvm_buffer.buffer[-1] = np.expand_dims(np.max(np.vstack(
                [pvm_buffer.buffer[-1], next_obs]), axis=0), axis=0)
        else:
            pvm_buffer.append(next_obs)

        # Get the next pvm observation
        next_pvm_obs = pvm_buffer.get_obs(mode="stack_max")

        return next_pvm_obs, rewards, dones, truncateds, infos

    def _log_episode(self, episode_info: EpisodeInfo,
            td_update: Optional[TdUpdateInfo], duration_info: Optional[DurationInfo],
            eval_env: bool):
        # Prepare the episode infos for the different supported envs
        if isinstance(self.envs[0], FixedFovealEnv):
            new_info = episode_info
            episode_info = EpisodeInfo.new()
            episode_info.update(new_info)

        # Ray logging
        ray_info = {
            "raw_reward": episode_info["raw_reward"],
            "sfn_loss": self.sfn_loss.item() if hasattr(self, "sfn_loss") else None,
            "timestep": self.timestep,
            "auc": self.auc,
            "windowed_auc": self.windowed_auc,
            "truncated": episode_info["truncated"],
            "eval_env": eval_env,
        }
        if td_update:
            ray_info.update({f"td/{k}": v for k,v in td_update._asdict().items()})
        if isinstance(self.envs[0], PauseableFixedFovealEnv):
            ray_info.update({
                "pauses": episode_info["pauses"],
                "prevented_pauses": episode_info["prevented_pauses"],
                "no_action_pauses": episode_info["no_action_pauses"],
                "saccade_cost": episode_info["saccade_cost"],
                "reward": episode_info["reward"],
            })
            if duration_info:
                ray_info.update({
                    "duration_error": duration_info.error,
                    "human_error": (1 - self.auc) + 5 * duration_info.error,
                    "gaze_duration": duration_info.durations
                })
        train.report(ray_info)

    def _train_sfn(self, data: DoubleActionReplayBufferSamples):
        # Prediction
        concat_observation = torch.concat(
            [data.next_observations, data.observations], dim=1)
        pred_motor_actions = self.sfn(Resize(self.obs_size)(concat_observation))
        self.sfn_loss = self.sfn.get_loss(
            pred_motor_actions, data.motor_actions.flatten())

        # Back propagation
        self.sfn_optimizer.zero_grad()
        self.sfn_loss.backward()
        self.sfn_optimizer.step()

        # Return the probabilites the sfn would have also selected the actually selected
        # action, given the limited observation. Higher probabilities suggest
        # better information was provided from the visual input
        observation_quality = F.softmax(pred_motor_actions, dim=0).gather(
            1, data.motor_actions).squeeze().detach()

        return observation_quality

    def _train_dqn(self, data: DoubleActionReplayBufferSamples,
                   observation_quality: torch.Tensor):
        """
        Trains the behavior q network and copies it to the target q network with
        self.target_network_frequency.

        :param NDArray data: A sample from the replay buffer
        :param NDArray[Shape[self.batch_size], Float] observation_quality: A batch of
            probabilities of the SFN predicting the action that the agent selected
        """
        # Cast sensory action ids to coordinates and remove the env axis
        prev_sensory_actions, consecutive_pauses = \
            self._preprocess_features(data)

        # Target network prediction
        with torch.no_grad():
            # Assign a value to every possible action in the next state for one batch
            motor_target, sensory_target = self.target_network(
                Resize(self.obs_size)(data.next_observations),
                consecutive_pauses, prev_sensory_actions)
                # -> [32, 19], [32, 19]
            # Get the next state value as the sum of max motor and sensory q values
            motor_target_max = motor_target.max(dim=1)[0]
            sensory_target_max = sensory_target.max(dim=1)[0]
            next_state_value = motor_target_max + sensory_target_max

            # Calculate the discount factors gamma
            gamma = torch.Tensor([self.gamma]).to(self.device)
            gamma = torch.pow(gamma, torch.arange(self.td_steps + 1).to(self.device))
            # Rewards after termination are set to 0
            gamma = torch.expand_copy(gamma, [data.dones.shape[0], *gamma.shape])
            for i, dones in enumerate(data.dones[...,0]):
                for j, done in enumerate(dones):
                    if done:
                        gamma[i, j+1:] = 0.
                        break

            # Reward function
            sugarl_penalty = torch.zeros([32]).to(self.device) if self.ignore_sugarl \
                else (1 - observation_quality) * self.sugarl_r_scale # -> [32]
            td_target = (data.rewards[...,0] * gamma[:,:-1]).sum(dim=1) \
                 + next_state_value * gamma[:,-1] - sugarl_penalty # -> [32]

        # Q network prediction
        old_motor_q_val, old_sensory_q_val = self.q_network(
            Resize(self.obs_size)(data.observations), data.consecutive_pauses,
            prev_sensory_actions) # -> [32], [32]
        old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
        old_sensory_val = old_sensory_q_val.gather(1, data.sensory_actions).squeeze()
        old_val = old_motor_val + old_sensory_val # -> [32]

        # Back propagation
        loss = F.mse_loss(td_target, old_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network with self.target_network_frequency
        if (self.timestep // self.n_envs) % self.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        r = TdUpdateInfo(old_val.mean().item(), td_target.mean().item(),
                data.rewards.flatten().mean().item(), sugarl_penalty.mean().item(),
                next_state_value.mean().item(), loss.mean().item(),
                motor_target_max.mean().item(), sensory_target_max.mean().item())

        return r

    def _epsilon_schedule(self, total_timesteps: int):
        """
        Maps the current number of timesteps to a value of epsilon.
        """
        # End exploration halfway through the trial
        total_timesteps //= 2
        return max(0, linear_schedule(*self.epsilon_interval,
            self.exploration_fraction * total_timesteps, self.timestep))

    def _preprocess_features(self,
                             data: Optional[DoubleActionReplayBufferSamples] = None):
        if data:
            prev_sensory_actions = torch.Tensor(self.sensory_action_set).to(
                self.device)[data.prev_sensory_actions][:,0,:]
            consecutive_pauses = torch.Tensor(data.consecutive_pauses).to(self.device)
        else:
            # Cast sensory action ids to coordinates and remove the env axis
            prev_sensory_actions = \
                torch.Tensor(np.array(self.env.envs[0].fov_loc)).to(self.device)
            # Same for the pauses
            consecutive_pauses = \
                torch.Tensor(np.array([self.env.envs[0].consecutive_pauses])).to(self.device)

        # Normalize the sensory actions between 0 and 1
        prev_sensory_actions /= \
            torch.Tensor(self.sensory_action_set.max(axis=0)).to(self.device)
        consecutive_pauses /= self.env.envs[0].consecutive_pause_limit

        return prev_sensory_actions, consecutive_pauses

    def _sensory_action_ids(self, coords: np.ndarray):
        """
        Return the sensory action ids for a given batch of coordinates

        :param Array[B,2] coords:
        :return Array[B]:
        """
        return np.array([np.where((self.sensory_action_set == c).all(axis=1))[0].item()
                         for c in coords])
