from active_gym.atari_env import AtariEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
from typing import Optional, Tuple
from torchvision.transforms import Resize
from active_gym import AtariEnvArgs

from atari_cr.foveation import Fovea, FovType
from atari_cr.models import EpisodeInfo, EpisodeRecord, StepInfo
from atari_cr.utils import EMMA_fixation_time
from atari_cr.atari_head.utils import VISUAL_DEGREES_PER_PIXEL


class PauseableFixedFovealEnv(gym.Wrapper):
    """
    Environemt wrapper making it possible to be paused to only take
    a sensory action without progressing the game.

    :param float pause_cost: Negative reward for the agent whenever they chose to
        not take an action in the environment to only look; prevents abuse of
        pausing
    :param int consecutive_pause_limit: Limit to the amount of consecutive pauses
        the agent can make before a random action is selected instead. This prevents
        the agent from halting
    :param float saccade_cost_scale: How much the agent is punished for bigger eye
        movements
    :param bool no_pause: Whether to disable the pause action
    :param bool timer: Whether to count the time it takes to look around, progressing
        the env once the game has generated a new frame at 20Hz.
    """
    def __init__(self, env: AtariEnv, args: AtariEnvArgs, pause_cost = 0.01,
            saccade_cost_scale = 0.001, fov: FovType = "window", no_pauses = False,
            consecutive_pause_limit = 50, timer = False, periph = False,
            fov_weighting=False):
        super().__init__(env)
        self.fov = Fovea(fov, args.fov_size, periph, weighting=fov_weighting)
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov.size) <
                np.array(env.obs_size)).all()
        self.no_pauses = no_pauses
        self.timer = timer

        # Get sensory action space for the sensory action mode
        self.relative_sensory_actions = args.sensory_action_mode == "relative"
        if self.relative_sensory_actions:
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.fov.type == "exponential":
            self.sensory_action_space = np.array(env.obs_size)
        elif self.fov.type == "window":
            self.sensory_action_space = \
                np.array(env.obs_size) - np.array(self.fov.size)
        else:
            raise ValueError("Invalid fov type")

        self.resize: Resize = Resize(env.obs_size) \
            if args.resize_to_full else None

        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a
            # sensory action without the game progressing
            "motor_action": Discrete(
                env.action_space.n if no_pauses else env.action_space.n + 1),
            "sensory_action": Box(low=self.sensory_action_space[0],
                                 high=self.sensory_action_space[1], dtype=int),
        })
        self.pause_action = None if no_pauses else \
            self.action_space["motor_action"].n - 1

        # How much the agent is punished for large eye movements
        self.saccade_cost_scale = saccade_cost_scale

        # Count and log the number of pauses made and their cost
        self.pause_cost = pause_cost
        self.consecutive_pause_limit = consecutive_pause_limit

        # Attributes for recording episodes
        self.record = args.record
        self.prev_episode: Optional[EpisodeRecord] = None
        self.env: AtariEnv

    def reset(self):
        self.state, info = self.env.reset() # -> [4,84,84;f64]
        assert self.state.shape == (4,84,84)
        self.state[:3] = np.full(self.state.shape[1:], 0.5)
        self.frames = [self.unwrapped.render()]

        self.timestep = -1
        self.step_infos = []
        self.episode_info = EpisodeInfo.new()
        self.consecutive_pauses = 0

        self.fov_loc = np.array(self.fov_init_loc, copy=True).astype(np.int32)
        fov_obs, visual_info = self.fov.apply(self.state.copy(), [self.fov_loc])

        # Time that has already passed since the last new frame
        # This is time that was still needed for the previous action
        self.time_passed = 0

        return fov_obs, info

    def step(self, action):
        # Calculate the time this step takes and the number of time to repeat the motor
        # action because of it
        prev_pause = self.time_passed > 50 # Whether the previous action was a pause
        step_time = self._step_time(action["sensory_action"])
        self.time_passed += step_time
        saccade_cost = self.saccade_cost_scale * step_time
        reward = -saccade_cost

        if action["motor_action"] == self.pause_action: # Pause step
            step_info = self._pre_step_logging(action)

            self.consecutive_pauses += 1
            done, truncated = False, False
            info = {"raw_reward": 0}
            # Durations for pause steps are instead counted for the next non-pause
            duration = 0

            # Mask out unwanted pause behavior
            if self.consecutive_pauses > self.consecutive_pause_limit:
                raise NotImplementedError()
            elif np.all(self.fov_loc == action["sensory_action"]):
                raise NotImplementedError()
            else:
                # Normal pause
                reward -= self.pause_cost

            step_info = self._post_step_logging(step_info, info["raw_reward"], reward,
                                                done, truncated, duration, saccade_cost)
            step_infos = [step_info]

        else: # Normal step
            self.consecutive_pauses = 0

            if not self.timer:
                # Just make one action if the timer is disabled
                action_repetitions = 1
                duration = step_time
            elif prev_pause:
                # Make one action that had as duration the time of the previous pauses
                # plus the time of the current action, rounded up to 50ms steps
                action_repetitions = 1
                duration = int((self.time_passed % 50 + 1) * 50)
            else:
                # Round the time up to 50ms steps and repeat the action for every 50ms
                # step
                action_repetitions = int(self.time_passed // 50 + 1)
                duration = 50
            self.time_passed = self.time_passed % 50

            # Repeat action as long as the saccade takes to complete
            step_infos = []
            for _ in range(action_repetitions):
                step_info = self._pre_step_logging(action)

                # Only the last state is kept
                self.state, partial_reward, done, truncated, info = \
                    self.env.step(action=action["motor_action"])
                # Sum up rewards
                reward += partial_reward

                # Log the step
                step_info = self._post_step_logging(step_info, info["raw_reward"],
                    partial_reward, done, truncated, duration, saccade_cost)
                step_infos.append(step_info)

                # Break early if done or truncated
                if done or truncated: break

        # Cast the step infos to have their values as lists in one dict
        info = { k: [] for k, v in step_infos[0].items() }
        for step_info in step_infos:
            for key in step_info.keys():
                info[key].append(step_info[key])

        # Sensory step
        fov_state = self._fov_step(
            full_state=self.state, action=action["sensory_action"])

        return fov_state, reward, done, truncated, info

    def _step_time(self, sensory_action: np.ndarray):
        """
        Calculate duration of one step as the time the sensory action takes to complete.

        :param Array[2; i32] sensory_action:
        :returns int: Time in ms
        """
        pixel_distance = sensory_action if self.relative_sensory_actions \
            else sensory_action - self.fov_loc
        eccentricity = np.sqrt(np.sum(np.square(
            pixel_distance * VISUAL_DEGREES_PER_PIXEL )))
        _, total_emma_time, fov_moved = EMMA_fixation_time(eccentricity)
        return int(total_emma_time * 1000)

    def _pre_step_logging(self, action: dict):
        """ Logs the state before the action is taken. """
        self.timestep += 1
        step_info = StepInfo.new()
        step_info["timestep"] = self.timestep
        step_info["fov_loc"] = self.fov_loc.tolist()
        step_info["motor_action"] = action["motor_action"]
        step_info["sensory_action"] = action["sensory_action"]
        step_info["pauses"] = int(action["motor_action"] == self.pause_action)
        self.frames.append(self.unwrapped.render())
        return step_info

    def _post_step_logging(self, step_info: StepInfo, raw_reward: float, reward: float,
            done: bool, truncated: bool, duration: float, saccade_cost: float):
        step_info["raw_reward"] = raw_reward
        step_info["reward"] = reward
        step_info["done"] = done
        step_info["truncated"] = int(truncated)
        step_info["duration"] = duration
        step_info["saccade_cost"] = saccade_cost
        # Episode info stores cumulative sums of the step info keys
        self.episode_info: EpisodeInfo
        for key in self.episode_info.keys():
            self.episode_info[key] += step_info[key]
        step_info["episode_info"] = self.episode_info.copy()
        # Exclude these pauses from cumulation
        step_info["consecutive_pauses"] = self.consecutive_pauses
        self.step_infos.append(step_info)

        # Log the infos of all steps in one record for the
        # entire episode
        if done or truncated:
            self.prev_episode = EpisodeRecord(
                np.stack(self.frames),
                EpisodeRecord.annotations_from_step_infos(self.step_infos),
                { "fov_size": self.fov.size.tolist(), "fov": self.fov.type }
            )
        return step_info

    def _clip_to_valid_fov(self, loc: np.ndarray):
        """ :param Array[W,H] loc: """
        return np.clip(loc, [0,0], self.sensory_action_space).astype(int)

    def _fov_step(self, full_state, action: np.ndarray):
        """
        Changes self.fov_loc by the given action and returns a version of the full
        state that is cropped to where the new self.fov_loc is

        :param Array[2] action:
        :returns Array[4,84,84]:
        """
        # Move the fovea
        if self.relative_sensory_actions:
            action = self._clip_to_valid_fov(action)
            action = self.fov_loc + action
        self.fov_loc = self._clip_to_valid_fov(action)

        fov_state, visual_info = self.fov.apply(full_state, [self.fov_loc])

        return fov_state

    def no_action_pause(self):
        """ Returns the action that would not unnecessarily pause the game without
        moving the fovea """
        return self.pause_action, self.fov_loc
