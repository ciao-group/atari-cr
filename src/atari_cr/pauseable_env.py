from active_gym.atari_env import AtariEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
from typing import Optional, Tuple
from torchvision.transforms import Resize
from active_gym import AtariEnvArgs

from atari_cr.models import EpisodeInfo, EpisodeRecord, StepInfo, FovType
from atari_cr.utils import EMMA_fixation_time
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE
from atari_cr.atari_head.dataset import SCREEN_SIZE
from atari_cr.graphs.eccentricity import A, B, C

MASKED_ACTION_PENTALTY = -1e9 # Negative reward for masking out actions


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
            consecutive_pause_limit = 50, timer = False):
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov: FovType = fov
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) <
                np.array(env.obs_size)).all()
        self.no_pauses = no_pauses

        # Get sensory action space for the sensory action mode
        self.relative_sensory_actions = args.sensory_action_mode == "relative"
        if self.relative_sensory_actions:
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.fov in ["gaussian", "exponential"]:
            self.sensory_action_space = np.array(env.obs_size)
        elif self.fov == "window":
            self.sensory_action_space = \
                np.array(env.obs_size) - np.array(self.fov_size)
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
        self.visual_degrees_per_pixel = np.array(VISUAL_DEGREE_SCREEN_SIZE) / \
            np.array(self.get_wrapper_attr("obs_size"))

        # Count and log the number of pauses made and their cost
        self.pause_cost = pause_cost
        self.consecutive_pause_limit = consecutive_pause_limit

        # Attributes for recording episodes
        self.record = args.record
        self.prev_episode: Optional[EpisodeRecord] = None
        self.env: AtariEnv

        # Timer is None if disabled or the game time otherwise
        self.timer = 0 if timer else None

    def reset(self):
        self.state, info = self.env.reset() # -> [4,84,84;f64]
        self.frames = [self.unwrapped.render()]

        self.timestep = -1
        self.step_infos = []
        self.episode_info = EpisodeInfo.new()
        self.consecutive_pauses = 0

        self.fov_loc = np.array(self.fov_init_loc, copy=True).astype(np.int32)
        fov_obs = self._crop_observation(self.state)

        return fov_obs, info

    def step(self, action):
        step_info = self._pre_step_logging(action)

        # Actual pause step
        if step_info["pauses"]:
            self.consecutive_pauses += 1
            done, truncated = False, False
            info = {"raw_reward": 0}

            # Mask out unwanted pause behavior
            if self.consecutive_pauses > self.consecutive_pause_limit:
                # Too many pauses in a row
                reward = MASKED_ACTION_PENTALTY
                step_info["prevented_pauses"] = 1
                truncated = True
            elif np.all(self.fov_loc == action["sensory_action"]):
                # No action pause
                reward = MASKED_ACTION_PENTALTY
                step_info["no_action_pauses"] = 1
            else:
                # Normal pause
                reward = -self.pause_cost

        else:
            self.consecutive_pauses = 0
            # Normal step, the state is saved for the next pause step
            self.state, reward, done, truncated, info = \
                self.env.step(action=action["motor_action"])

        # Calculate saccade duration and add a cost for it
        pixel_distance = action["sensory_action"] if self.relative_sensory_actions \
            else action["sensory_action"] - self.fov_loc
        eccentricity = np.sqrt(np.sum(np.square(
            pixel_distance * self.visual_degrees_per_pixel )))
        _, total_emma_time, fov_moved = EMMA_fixation_time(eccentricity)
        step_info["emma_time"] = total_emma_time
        step_info["saccade_cost"] = self.saccade_cost_scale * total_emma_time
        reward -= step_info["saccade_cost"]
        if self.timer is not None:
            # Increment the time by emma time in ms
            self.timer = total_emma_time * 1000
            # The first env step takes 50ms until the next frame appears
            # Then the motor action is repeated until total emma time is over
            if not step_info["pauses"]: step_info["emma_time"] = 0.050

        step_info = self._post_step_logging(step_info, info["raw_reward"], reward, done,
                                            truncated)
        step_infos = { k: [v] for k, v in step_info.copy().items() }

        if (self.timer is not None) and not step_info["pauses"]:
            # Repeat action as long as the saccade takes to complete
            while self.timer >= 50:
                step_info = self._pre_step_logging(action)
                step_info["emma_time"] = 0.050 # Set the duration to 50ms

                # Only the last state is kept
                self.state, partial_reward, done, truncated, info = \
                    self.env.step(action=action["motor_action"])

                # Sum up rewards
                reward += partial_reward

                # Decrement the timer and log the step
                self.timer -= 50
                step_info = self._post_step_logging(step_info,
                    info["raw_reward"], partial_reward, done, truncated)
                for key in step_info.keys():
                    step_infos[key].append(step_info[key])

                # Break early if done or truncated
                if done or truncated: break

        # Sensory step
        fov_state = self._fov_step(
            full_state=self.state, action=action["sensory_action"])

        return fov_state, reward, done, truncated, step_infos

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
            done: bool, truncated: bool):
        step_info["raw_reward"] = raw_reward
        step_info["reward"] = reward
        step_info["done"] = done
        step_info["truncated"] = int(truncated)
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
                { "fov_size": self.fov_size, "fov": self.fov }
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

        fov_state = self._crop_observation(full_state)

        return fov_state

    def _crop_observation(self, full_state: np.ndarray):
        """
        Get a version of the full_state that is cropped to the fovea around fov_loc if
        fov is set to 'window'. Otherwise get a mask over the output

        :param Array[4,84,84] full_state: Stack of four greyscale images of type float64
        """
        masked_state = np.zeros_like(full_state)
        if self.fov == "gaussian":
            distances_from_fov = self._pixel_eccentricities(
                full_state.shape[-2:], self.fov_loc)
            sigma = self.fov_size[0] / 2
            gaussian = np.exp(-np.sum(
                np.square(distances_from_fov) / (2 * np.square(sigma)),
                axis=0)) # -> [84,84]
            gaussian /= gaussian.max()
            masked_state = full_state * gaussian
        elif self.fov == "window":
            crop = full_state[:,
                self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1],
                self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
            ]
            # Fill the area outside the crop with zeros
            masked_state[:,
                self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1],
                self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
            ] = crop
        elif self.fov == "exponential":
            pixel_eccentricities = self._pixel_eccentricities(
                full_state.shape[-2:], self.fov_loc)
            # Convert from pixels to visual degrees
            eccentricities = (pixel_eccentricities.transpose([1,2,0]) *\
                (np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array(SCREEN_SIZE))
                ).transpose(2,0,1)
            # Absolute 1D distances
            abs_distances = np.sqrt(np.square(eccentricities).sum(axis=0))

            mask = A * np.exp(B * abs_distances) + C
            mask /= mask.max()
            masked_state = full_state * mask

        return masked_state

    @staticmethod
    def _pixel_eccentricities(screen_size: tuple[int, int], fov_loc: tuple[int, int]):
        """
        Returns a tensor containing x and y distances from the fov_loc for every
        possible x and y coord. Used for broadcasted calculations.

        :returns Array[2,W,H]:
        """
        x = np.arange(screen_size[0])
        y = np.arange(screen_size[1])
        mesh = np.stack(np.meshgrid(x, y, indexing="xy")) # -> [2,84,84]
        fov_loc = np.broadcast_to(np.array(
            fov_loc)[..., np.newaxis, np.newaxis], [2,*screen_size])
        return mesh - fov_loc # -> [2,84,84]

class SlowedFixedFovealEnv(PauseableFixedFovealEnv):
    """ Pauseable like in Atari-HEAD with the option to make the game run at a constant
    20Hz by pressing down a key """
    def __init__(self, env: AtariEnv, args: AtariEnvArgs, pause_cost = 0.01,
            saccade_cost_scale = 0.001, fov: FovType = "window", no_pauses = False,
            consecutive_pause_limit = 20):
        super().__init__(env, args, pause_cost, saccade_cost_scale, fov, no_pauses,
                         consecutive_pause_limit)

        # Two additional actions: PlayKeyDown, PlayKeyUp for pressing down or lifting
        # the key press again. The pressed key continues the env at a speed of 20Hz.
        # With no key pressed, the game waits for an action by the agent before
        # resuming
        self.action_space = Dict({
            "motor_action": Discrete(
                env.action_space.n if no_pauses else env.action_space.n + 2),
            "sensory_action": Box(low=self.sensory_action_space[0],
                                 high=self.sensory_action_space[1], dtype=int),
        })
        self.key_down_action = self.action_space["motor_action"].n - 1
        self.key_up_action = self.action_space["motor_action"].n - 2

        # Key is initially not pressed
        self.key_is_pressed = False

    def reset(self):
        self.timer = 0 # Time spent by the agent since last frame
        return super().reset()

    def step(self, action: dict):
        # Move or pause in the env
        if action["motor_action"] in [self.key_down_action, self.key_up_action]:
            # Press or unpress key
            self.key_is_pressed = (action["motor_action"] == self.key_down_action)
            # Set default step returns for a pause action
            reward, done, truncated = 0, False, False
            info = { "raw_reward": 0, "pauses": 1 }
            # Set the action to be repeated later to NOOP
            action["motor_action"] = np.int64(0) # NOOP
        else:
            # Normal env step
            self.state, reward, done, truncated, info = \
                    self.env.step(action=action["motor_action"])
            info["pauses"] = 0
        assert list(info.keys()) == ["raw_reward", "pauses"]

        # Add costs for the time it takes the agent to move its fovea
        prev_fov_loc = self.fov_loc.copy()
        eccentricity = np.sqrt(np.sum(np.square(
            (self.fov_loc - prev_fov_loc) * self.visual_degrees_per_pixel )))
        _, total_emma_time, fov_moved = EMMA_fixation_time(eccentricity)
        total_emma_time = total_emma_time.item()
        reward -= self.saccade_cost_scale * total_emma_time
        # Increment the time by emma time in ms
        self.timer += total_emma_time * 1000
        info["duration(ms)"] = self.timer

        if self.key_down_action:
            # Repeat action as long as the saccade takes to complete
            while self.timer >= 50:
                # Only the last state is kept
                # Rewards are summed up
                # Raw rewards in info are summed up
                # Break early if done or truncated
                self.state, partial_reward, done, truncated, partial_info = \
                    self.env.step(action=action["motor_action"])
                reward += partial_reward
                info["raw_reward"] += partial_info["raw_reward"]
                self.timer -= 50
                info["pauses"] = 0
                if done or truncated: break

        # Sensory action
        fov_state = self._fov_step(
            full_state=self.state, action=action["sensory_action"])

        return fov_state, reward, done, truncated, info

