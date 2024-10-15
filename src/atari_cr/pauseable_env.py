from active_gym.atari_env import AtariEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import cv2
import torch
import numpy as np
import polars as pl
from typing import Tuple
from torchvision.transforms import Resize

from atari_cr.models import EpisodeInfo, SensoryActionMode, StepInfo
from atari_cr.utils import EMMA_fixation_time
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE


class PauseableFixedFovealEnv(gym.Wrapper):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    """
    def __init__(self, env: AtariEnv, args, pause_cost = 0.01,
            consecutive_pause_limit = 20, no_action_pause_cost = 0.1,
            saccade_cost_scale = 0.001, use_emma = False):
        """
        :param float pause_cost: Negative reward for the agent whenever they chose to
            not take an action in the environment to only look; prevents abuse of
            pausing
        :param int consecutive_pause_limit: Limit to the amount of consecutive pauses
            the agent can make before a random action is selected instead. This prevents
            the agent from halting
        :param float saccade_cost_scale: How much the agent is punished for bigger eye
            movements
        :param bool use_emma: Whether to use the EMMA model
            (doi.org/10.1016/S1389-0417(00)00015-2) for saccade cost calculation. If
            not, the pixel length of the saccade is used.
        """
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) <
                np.array(self.get_wrapper_attr("obs_size"))).all()

        # Get sensory action space for the sensory action mode
        self.sensory_action_mode: SensoryActionMode = args.sensory_action_mode
        if self.sensory_action_mode == SensoryActionMode.RELATIVE:
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.sensory_action_mode == SensoryActionMode.ABSOLUTE:
            self.sensory_action_space = \
                np.array(self.get_wrapper_attr("obs_size")) - np.array(self.fov_size)
        else:
            raise ValueError(
                "sensory_action_mode needs to be either 'absolute' or 'relative'")

        self.resize: Resize = Resize(self.get_wrapper_attr("obs_size")) \
            if args.resize_to_full else None

        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a
            # sensory action without the game progressing
            "motor_action": Discrete(self.get_wrapper_attr("action_space").n + 1),
            "sensory_action": Box(low=self.sensory_action_space[0],
                                 high=self.sensory_action_space[1], dtype=int),
        })

        # How much the agent is punished for large eye movements
        self.saccade_cost_scale = saccade_cost_scale

        # Whether the current action is a pause action
        self.prev_pause_action = 0

        # Count and log the number of pauses made and their cost
        self.pause_cost = pause_cost
        self.consecutive_pause_limit = consecutive_pause_limit

        # Count the number of pauses without a sensory action for every episode
        self.no_action_pause_cost = no_action_pause_cost

        # Attributes from RecordWrapper class
        self.args = args
        self.record_buffer = None
        # The record buffer from the previously finished episode is used when saving
        # episode episode data to the file system. This prevents unfinished episodes
        # from being saved
        # self.prev_record_buffer = None
        self.record = args.record

        # EMMA
        self.use_emma = use_emma

        self.episode_info: EpisodeInfo
        self.env: AtariEnv

    def reset(self):
        self.state, info = self.env.reset()
        assert info == {'raw_reward': 0}
        assert self.state.shape == (4, 84, 84)
        assert self.state.dtype == np.float64

        self.frames = []
        self.step_infos = []
        self.episode_info = EpisodeInfo.new()
        self.timestep = -1
        self.consecutive_pauses = 0
        self.fov_loc = np.array(self.fov_init_loc, copy=True).astype(np.int32)

        fov_obs = self._crop_observation(self.state)

        # if self.record:
        #     # Save the record buffer of the previous episode and create a new one
        #     # On the first episode start, there will be no previous record buffer
        #     self.record_buffer = pl.DataFrame(
        #         schema=list(StepInfo.__annotations__.keys()))

        return fov_obs, info

    def step(self, action):
        # Log the state before the action is taken
        step_info = StepInfo.new()
        self.timestep += 1
        step_info["timestep"] = self.timestep
        step_info["fov_loc"] = self.fov_loc.tolist()
        step_info["motor_action"] = action["motor_action"]
        step_info["sensory_action"] = action["sensory_action"]
        step_info["pauses"] = int(self.is_pause(action["motor_action"]))
        self.frames.append(self.unwrapped.render())

        # Save the previous fov_loc to calculate the saccade cost
        prev_fov_loc = self.fov_loc.copy()

        # Disallow a pause on the first episode step because there is no
        # observation to look at yet and do a random action instead
        if not hasattr(self, "state") and step_info["pauses"]:
            action["motor_action"] = self._sample_no_pause()
            step_info["pauses"] = self.is_pause(action["motor_action"])
            assert not step_info["pauses"], \
                "Pause action should not happen on the first episode step"

        # Prevent the agent from being stuck on only using pauses
        # Perform a random motor action instead if too many pauses
        # have happened in a row
        if (self.episode_info["prevented_pauses"] > 50 or
            self.consecutive_pauses > self.consecutive_pause_limit) and \
                step_info["pauses"]:
            action["motor_action"] = self._sample_no_pause()
            step_info["pauses"] = self.is_pause(action["motor_action"])
            step_info["prevented_pauses"] = 1
            assert not step_info["pauses"], \
                "Pause action should not happen after many consecutive pauses"

        if step_info["pauses"]:
            if np.all(self.fov_loc == action["sensory_action"]):
                # No action pause
                step_info["reward"] = -self.no_action_pause_cost
                step_info["no_action_pauses"] = 1
            else:
                # Normal pause
                step_info["reward"] = -self.pause_cost

            raw_reward = 0
            done, truncated = False, False
            info = {"raw_reward": raw_reward}

            # Log another pause
            if self.prev_pause_action:
                self.consecutive_pauses += 1
            else:
                self.consecutive_pauses = 0
            # Set prev pause action for next step
            self.prev_pause_action = step_info["pauses"]


        else:
            self.consecutive_pauses = 0
            # Normal step, the state is saved for the next pause step
            self.state, step_info["reward"], done, truncated, info = \
                self.env.step(action=action["motor_action"])
            raw_reward = step_info["reward"]

        # Sensory step
        fov_state = self._fov_step(
            full_state=self.state, action=action["sensory_action"])

        # Add costs for the time it took the agent to move its fovea
        if self.use_emma:
            visual_degrees_per_pixel = np.array(VISUAL_DEGREE_SCREEN_SIZE) / \
                np.array(self.get_wrapper_attr("obs_size"))
            visual_degree_distance = np.sqrt(np.sum(
                np.square((self.fov_loc - prev_fov_loc) * visual_degrees_per_pixel) ))
            _, total_emma_time, fovea_did_move = EMMA_fixation_time(
                visual_degree_distance)
            step_info["saccade_cost"] = self.saccade_cost_scale * total_emma_time
        # Add cost for the distance traveled by the fovea
        else:
            pixel_distance = np.sqrt(np.sum( np.square(self.fov_loc - prev_fov_loc) ))
            step_info["saccade_cost"] = self.saccade_cost_scale * pixel_distance
        step_info["reward"] -= step_info["saccade_cost"]

        # Log the results of taking an action
        step_info["reward_wo_sugarl"] = info["raw_reward"]
        step_info["raw_reward"] = raw_reward
        step_info["done"] = done
        step_info["truncated"] = truncated
        for key in self.episode_info.keys():
            self.episode_info[key] += step_info[key]
        step_info["episode_info"] = self.episode_info.copy()
        step_info["consecutive_pauses"] = self.consecutive_pauses

        # Log the infos of all steps in one record buffer for the
        # entire episode
        self.step_infos.append(step_info)
        if done or truncated:
            self.trial_frames = np.stack(self.frames)
            self.record_buffer = pl.DataFrame(self.step_infos).with_columns(
                pl.col("episode_info").struct.rename_fields(
                    [f"cum_{col}" for col in self.episode_info.keys()])
                ).unnest("episode_info").with_columns([
                    # Split fov_loc and sensory_action into smaller columns
                    # To make the df exportable via csv later
                    pl.col("fov_loc").map_elements(
                        lambda a: a[0], pl.Int32).alias("fov_x"),
                    pl.col("fov_loc").map_elements(
                        lambda a: a[1], pl.Int32).alias("fov_y"),
                    pl.col("sensory_action").map_elements(
                        lambda a: a[0], pl.Int32).alias("sensory_action_x"),
                    pl.col("sensory_action").map_elements(
                        lambda a: a[1], pl.Int32).alias("sensory_action_y"),
                ]).drop(["fov_loc", "sensory_action"])

        return fov_state, step_info["reward"], done, truncated, step_info

    def save_record_to_file(self, file_path: str, draw_focus = False,
                            draw_pauses = False):
        """ Saves the record_buffer to an mp4 file and a metadata file. """
        video_path = file_path.replace(".csv", ".mp4")
        size = self.trial_frames[0].shape[:2][::-1]
        fps = 30
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

        # Style settings
        color = (255, 0, 0)
        thickness = 1

        if draw_focus:
            # The fov_loc is set in a 84x84 grid; the video output is 256x256
            # This scales it down
            COORD_SCALING = 256 / 84
            fov_locs = np.stack(self.record_buffer["fov_x"].to_numpy(),
                                self.record_buffer["fov_y"].to_numpy())
            fov_locs *= COORD_SCALING
            fov_size = np.array(self.fov_size) * COORD_SCALING

        for i, frame in enumerate(self.trial_frames):

            if draw_focus:
                top_left = fov_locs[i]
                bottom_right = top_left + fov_size
                frame = cv2.rectangle(
                    frame, top_left, bottom_right, color, thickness)

            if draw_pauses:
                text = f"Pauses: {self.record_buffer[i, 'episode_info']['pauses']}"
                position = (10, 20)
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 0.3
                frame = cv2.putText(
                    frame, text, position, font, font_scale, color, thickness)

            video_writer.write(frame)

        self.record_buffer.write_csv(file_path)
        video_writer.release()

    def is_pause(self, motor_action: int):
        """
        Checks if a given motor action is the pause action
        """
        return motor_action == len(self.get_wrapper_attr("actions"))

    def _sample_no_pause(self):
        """ Samples an action that is not a pause. """
        return np.random.randint(len(self.get_wrapper_attr("actions")))

    # def _log_step(self, reward, action, done, truncated, info, raw_reward, saccade_cost):
    #     self.ep_len += 1
    #     self.cumulative_reward += reward
    #     self.cumulative_raw_reward += raw_reward
    #     self.cumulative_saccade_cost += saccade_cost

    #     if self.record:
    #         rgb = self.env.render()
    #         self._save_transition(
    #             self.state,
    #             action,
    #             self.cumulative_reward,
    #             done,
    #             truncated,
    #             info,
    #             rgb=rgb,
    #             return_reward=reward,
    #             episode_pauses=self.n_pauses,
    #             fov_loc=self.fov_loc,
    #             raw_reward=raw_reward
    #         )

    #     if done or truncated:
    #         pass # TODO: Save the last env render without an accociated action

    def _skip_step(self, reward):
        """
        Make a step without actually making a step.

        Returns
        -------
        state, the given reward, done, truncated, info
        """
        done, truncated = False, False
        info = { "raw_reward": reward }
        return self.state, reward, done, truncated, info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(loc, 0, np.array(self.get_wrapper_attr("obs_size")) - np.array(self.fov_size))).astype(int)

    def _clip_to_valid_sensory_action_space(self, action):
        return np.rint(np.clip(action, *self.sensory_action_space)).astype(int)

    def _fov_step(self, full_state, action):
        """
        Changes self.fov_loc by the given action and returns a version of the full
        state that is cropped to where the new self.fov_loc is
        """
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        # Move the fovea
        if self.sensory_action_mode == SensoryActionMode.RELATIVE:
            action = self._clip_to_valid_sensory_action_space(action)
            action = self.fov_loc + action
        self.fov_loc = self._clip_to_valid_fov(action)

        fov_state = self._crop_observation(full_state)

        return fov_state

    def _crop_observation(self, full_state: np.ndarray):
        """
        Get a version of the full_state that is cropped the foveas around fov_loc

        :param Array[4,84,84] full_state: Stack of four greyscale images of type float64
        """
        # Make this work with a list of fov_locs
        crop = full_state[...,
            self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
            self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1]
        ]

        # Fill the area outside the crop with zeros
        mask = np.zeros_like(full_state)
        mask[...,
            self.fov_loc[0]:self.fov_loc[0] + self.fov_size[0],
            self.fov_loc[1]:self.fov_loc[1] + self.fov_size[1]
        ] = crop

        return mask

    # def _reset_record_buffer(self):
    #     """ Saves self.record_buffer in self.prev_record_buffer and creates a new one"""
    #     self.prev_record_buffer = copy.deepcopy(self.record_buffer)
    #     self.record_buffer = RecordBuffer.new()

    # def _save_transition(self):
    #     """
    #     Logs the step in self.record_buffer

    #     :param List[Array[2]] fov_loc: List of gazes made on one frame
    #     """
    #     if (done is not None) and (not done):
    #         self.record_buffer["rgb"].append(rgb)

    #     if done is not None and len(self.record_buffer["state"]) > 1:
    #         self.record_buffer["done"].append(done)
    #     if info is not None and len(self.record_buffer["state"]) > 1:
    #         self.record_buffer["info"].append(info)

    #     if action is not None:
    #         self.record_buffer["action"].append(action)
    #     if reward is not None:
    #         self.record_buffer["reward"].append(reward)
    #     if raw_reward is not None:
    #         self.record_buffer["raw_reward"].append(raw_reward)
    #     if truncated is not None:
    #         self.record_buffer["truncated"].append(truncated)
    #     if return_reward is not None:
    #         self.record_buffer["return_reward"].append(return_reward)
    #     if episode_pauses is not None:
    #         self.record_buffer["episode_pauses"].append(episode_pauses)
    #     if fov_loc is not None:
    #         self.record_buffer["fov_loc"].append(fov_loc)


# # OPTIONAL:
# class SlowableFixedFovealEnv(PauseableFixedFovealEnv):
#     """
#     Environemt making it possible to be paused to only take
#     a sensory action without progressing the game.
#     Additionally, the game can be run at 1/3 of the original
#     speed as it was done with the Atari-HEAD dataset
#     (http://arxiv.org/abs/1903.06754)
#     """
#     def __init__(self, env: gym.Env, args, pause_cost = 0.01, consecutive_pause_limit = 20,
#                  no_action_pause_cost = 0.1, saccade_cost_scale = 0.001):
#         super().__init__(env, args, pause_cost, consecutive_pause_limit, no_action_pause_cost, saccade_cost_scale)
#         self.ms_since_motor_step = 0

#     def step(self, action):
#         pause_action = action["motor_action"] == len(self.get_wrapper_attribute("actions"))
#         if pause_action and (self.state is not None):
#             # If it has been more than 50ms (results in 20 Hz) since the last motor action
#             # NOOP will be chosen as a motor action instead of continuing the pause
#             SLOWED_FRAME_RATE = 20
#             if self.ms_since_motor_step >= 1000/SLOWED_FRAME_RATE:
#                 action["motor_action"] = np.int64(0)
#                 self.ms_since_motor_step = 0
#                 return self.step(action)
#         else:
#             self.ms_since_motor_step = 0

#         fov_state, reward, done, truncated, info = super().step(action)

#         # Progress the time because a vision step has happened
#         # depending on how big the sensory action was
#         # Time for a saccade is fixed to 20 ms for now
#         self.ms_since_motor_step += 20

#         return fov_state, reward, done, truncated, info
