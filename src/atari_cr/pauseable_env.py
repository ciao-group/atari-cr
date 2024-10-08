from active_gym.atari_env import AtariEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import cv2
import torch
import numpy as np
from typing import TypedDict, List, Tuple
import copy
from torchvision.transforms import Resize

from atari_cr.models import SensoryActionMode, RecordBuffer
from atari_cr.utils import EMMA_fixation_time
from atari_cr.atari_head.utils import VISUAL_DEGREE_SCREEN_SIZE
        

class PauseableFixedFovealEnv(gym.Wrapper):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    """
    def __init__(self, env: gym.Env, args, pause_cost = 0.01, successive_pause_limit = 20, 
                 no_action_pause_cost = 0.1, saccade_cost_scale = 0.001, use_emma = False):
        """
        :param float pause_cost: Negative reward for the agent whenever they chose to not take
            an action in the environment to only look; prevents abuse of pausing
        :param int successive_pause_limit: Limit to the amount of successive pauses the agent can make before
            a random action is selected instead. This prevents the agent from halting

        :param float saccade_cost_scale: How much the agent is punished for bigger eye movements
        :param bool use_emma: Whether to use the EMMA model (doi.org/10.1016/S1389-0417(00)00015-2) for saccade cost calculation. If not, the pixel length of the saccade is used.
        """
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) < np.array(self.obs_size)).all()
        self.saccade_cost_scale = saccade_cost_scale

        # Get sensory action space for the sensory action mode
        self.sensory_action_mode: SensoryActionMode = args.sensory_action_mode 
        if self.sensory_action_mode == SensoryActionMode.RELATIVE:
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.sensory_action_mode == SensoryActionMode.ABSOLUTE:
            self.sensory_action_space = np.array(self.obs_size) - np.array(self.fov_size)
        else:
            raise ValueError("sensory_action_mode needs to be either 'absolute' or 'relative'")

        self.resize: Resize = Resize(self.get_wrapper_attr("obs_size")) if args.resize_to_full else None
        self.mask_out: bool = args.mask_out

        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a 
            # sensory action without the game progressing  
            "motor_action": Discrete(self.get_wrapper_attr("action_space").n + 1),
            "sensory_action": Box(low=self.sensory_action_space[0], 
                                 high=self.sensory_action_space[1], dtype=int),
        })
        # Whether to pause the game at the current step
        self.pause_action = False

        # Count and log the number of pauses made and their cost
        self.pause_cost = pause_cost
        self.successive_pause_limit = successive_pause_limit
        self.n_pauses = 0

        # Count the number of pauses without a sensory action for every episode
        self.no_action_pause_cost = no_action_pause_cost
        self.no_action_pauses = 0

        # Count successive pause actions to prevent the system
        # from halting
        self.successive_pauses = 0
        self.prevented_pauses = 0

        # Attributes from RecordWrapper class
        self.args = args
        self.cumulative_reward = 0
        self.cumulative_raw_reward = 0
        self.cumulative_saccade_cost = 0
        self.ep_len = 0
        self.record_buffer: RecordBuffer = None
        self.prev_record_buffer: RecordBuffer = None
        self.record = args.record
        if self.record:
            self._reset_record_buffer()

        # EMMA
        self.use_emma = use_emma

        self.env: AtariEnv

    def step(self, action):
        # The pause action is the last action in the action set
        prev_pause_action = self.pause_action
        self.pause_action = self.is_pause(action["motor_action"])

        # Save the previous fov_loc to calculate the saccade cost
        prev_fov_loc = self.fov_loc.copy()

        if self.pause_action:
            # Disallow a pause on the first episode step because there is no
            # observation to look at yet and do a random action instead
            if not hasattr(self, "state"):
                action["motor_action"] = np.random.randint(1, len(self.get_wrapper_attr("actions")))
                return self.step(action)

            # Prevent the agent from being stuck on only using pauses
            # Perform a random motor action instead if too many pauses
            # have happened in a row
            if self.prevented_pauses > 50 or self.successive_pauses > self.successive_pause_limit:
                while self.pause_action:
                    action["motor_action"] = self.action_space["motor_action"].sample()
                    self.pause_action = self.is_pause(action["motor_action"])
                self.pause_action = False
                self.prevented_pauses += 1
                return self.step(action)

            if np.all(self.fov_loc == action["sensory_action"]):  
                # Penalize a pause action without moving the fovea because it is useless
                state, reward, done, truncated, info = self._skip_step(-self.no_action_pause_cost)
                self.no_action_pauses += 1
            else:
                # Only make a sensory step with a small cost
                _, reward, done, truncated, info = self._skip_step(-self.pause_cost)
                state = self._fov_step(full_state=self.state, action=action["sensory_action"])

            raw_reward = 0

            # Log another pause
            self.n_pauses += 1
            if prev_pause_action:
                self.successive_pauses += 1
            else:
                self.successive_pauses = 0

        else:
            self.successive_pauses = 0
            # Normal step
            state, reward, done, truncated, info = self.env.step(action=action["motor_action"])
            raw_reward = reward
            # Safe the state for the next sensory step
            self.state = state
            # Sensory step
            state = self._fov_step(full_state=self.state, action=action["sensory_action"])  
            
        # Add costs for the time it took the agent to move its fovea
        if self.use_emma:
            visual_degrees_per_pixel = np.array(VISUAL_DEGREE_SCREEN_SIZE) / np.array(self.get_wrapper_attr("obs_size"))
            visual_degree_distance = np.sqrt(np.sum( np.square((self.fov_loc - prev_fov_loc) * visual_degrees_per_pixel) ))
            _, total_emma_time, fovea_did_move = EMMA_fixation_time(visual_degree_distance)
            saccade_cost = self.saccade_cost_scale * total_emma_time
        # Add cost for the distance traveled by the fovea
        else: 
            pixel_distance = np.sqrt(np.sum( np.square(self.fov_loc - prev_fov_loc) ))
            saccade_cost = self.saccade_cost_scale * pixel_distance
        reward -= saccade_cost 

        self._log_step(reward, action, done, truncated, info, raw_reward, saccade_cost)
        info = self._update_info(info)

        # Reset counters at the end of an episode
        if done:
            self.n_pauses = 0
            self.prevented_pauses = 0
            self.no_action_pauses = 0

        return state, reward, done, truncated, info
    
    def reset(self):
        full_state, info = self.env.reset()

        self.cumulative_reward = 0
        self.cumulative_raw_reward = 0
        self.cumulative_saccade_cost = 0
        self.ep_len = 0
        self.fov_loc = np.rint(np.array(self.fov_init_loc, copy=True)).astype(np.int32)
        fov_state = self._get_fov_state(full_state)

        info = self._update_info(info)

        if self.record:
            # Reset record_buffer
            self.record_buffer["fov_size"] = self.fov_size
            # Record Wrapper Stuff
            rgb = self.env.render()
            self._reset_record_buffer()
            self._save_transition(fov_state, done=False, info=info, rgb=rgb)

        return fov_state, info

    def save_record_to_file(self, file_path: str, draw_focus = False, draw_pauses = True):
        if self.record:
            video_path = file_path.replace(".pt", ".mp4")
            self.prev_record_buffer: RecordBuffer
            size = self.prev_record_buffer["rgb"][0].shape[:2][::-1]
            fps = 30
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for i, frame in enumerate(self.prev_record_buffer["rgb"]):
                    
                # Set text and rectangle style
                color = (255, 0, 0)
                thickness = 1

                if draw_focus:
                    y_loc, x_loc = self.prev_record_buffer["fov_loc"][i]
                    fov_size = self.prev_record_buffer["fov_size"]

                    # The fov_loc is set within a 84x84 grid while the video output is 256x256
                    # To scale them accordingly we multiply with the following
                    COORD_SCALING = 256 / 84
                    x_loc = int(x_loc * COORD_SCALING)
                    y_loc = int(y_loc * COORD_SCALING)
                    fov_size = (int(fov_size[0] * COORD_SCALING), int(fov_size[1] * COORD_SCALING))

                    top_left = (x_loc, y_loc)
                    bottom_right = (x_loc + fov_size[0], y_loc + fov_size[1])
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)

                if draw_pauses:
                    text = f"Pauses: {self.prev_record_buffer['episode_pauses'][i]}"
                    position = (10, 20)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 0.3
                    frame = cv2.putText(frame, text, position, font, font_scale, color, thickness)

                video_writer.write(frame)

            self.prev_record_buffer["rgb"] = video_path
            self.prev_record_buffer["state"] = [0] * len(self.prev_record_buffer["reward"])
            torch.save(self.prev_record_buffer, file_path)
            video_writer.release()

    def _log_step(self, reward, action, done, truncated, info, raw_reward, saccade_cost):
        self.ep_len += 1
        self.cumulative_reward += reward
        self.cumulative_raw_reward += raw_reward
        self.cumulative_saccade_cost += saccade_cost

        if self.record:
            rgb = self.env.render()
            self._save_transition(self.state, 
                action, self.cumulative_reward, 
                done, truncated, info, rgb=rgb, 
                return_reward=reward, 
                episode_pauses=self.n_pauses,
                fov_loc=self.fov_loc,
                raw_reward=raw_reward
            )

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

    def is_pause(self, motor_action: int):
        """
        Checks if a given motor action is the pause action
        """
        return motor_action == len(self.get_wrapper_attr("actions"))

    def _update_info(self, info: dict):
        info["reward"] = self.cumulative_reward
        info["raw_reward"] = self.cumulative_raw_reward
        info["ep_len"] = self.ep_len
        info["pause_cost"] = self.pause_cost
        info["n_pauses"] = self.n_pauses
        info["prevented_pauses"] = self.prevented_pauses
        info["fov_loc"] = self.fov_loc.copy()
        info["no_action_pauses"] = self.no_action_pauses
        info["saccade_cost"] = self.cumulative_saccade_cost
        return info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(loc, 0, np.array(self.get_wrapper_attr("obs_size")) - np.array(self.fov_size))).astype(int)

    def _clip_to_valid_sensory_action_space(self, action):
        return np.rint(np.clip(action, *self.sensory_action_space)).astype(int)

    def _fov_step(self, full_state, action):
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        # Move the fovea
        if self.sensory_action_mode == SensoryActionMode.RELATIVE:
            action = self._clip_to_valid_sensory_action_space(action)
            action = self.fov_loc + action
        self.fov_loc = self._clip_to_valid_fov(action)

        fov_state = self._get_fov_state(full_state)
        
        return fov_state

    def _get_fov_state(self, full_state):
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]]

        if self.mask_out:
            mask = np.zeros_like(full_state)
            mask[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]] = fov_state
            fov_state = mask
        elif self.resize:
            fov_state = self.resize(torch.from_numpy(fov_state))
            fov_state = fov_state.numpy()

        return fov_state

    def _reset_record_buffer(self):
        self.prev_record_buffer = copy.deepcopy(self.record_buffer)
        self.record_buffer = RecordBuffer.new()

    def _save_transition(self, state, action=None, reward=None, done=None, 
            truncated=None, info=None, rgb=None, return_reward=None, 
            episode_pauses=None, fov_loc=None, raw_reward=None):
        """
        Logs the step in self.record_buffer
        """
        if (done is not None) and (not done):
            self.record_buffer["state"].append(state)
            self.record_buffer["rgb"].append(rgb)

        if done is not None and len(self.record_buffer["state"]) > 1:
            self.record_buffer["done"].append(done) 
        if info is not None and len(self.record_buffer["state"]) > 1:
            self.record_buffer["info"].append(info)

        if action is not None:
            self.record_buffer["action"].append(action)
        if reward is not None:
            self.record_buffer["reward"].append(reward)
        if raw_reward is not None:
            self.record_buffer["raw_reward"].append(raw_reward)
        if truncated is not None:
            self.record_buffer["truncated"].append(truncated)
        if return_reward is not None:
            self.record_buffer["return_reward"].append(return_reward)
        if episode_pauses is not None:
            self.record_buffer["episode_pauses"].append(episode_pauses)
        if fov_loc is not None:
            self.record_buffer["fov_loc"].append(fov_loc)


# OPTIONAL:
class SlowableFixedFovealEnv(PauseableFixedFovealEnv):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    Additionally, the game can be run at 1/3 of the original
    speed as it was done with the Atari-HEAD dataset
    (http://arxiv.org/abs/1903.06754)
    """
    def __init__(self, env: gym.Env, args, pause_cost = 0.01, successive_pause_limit = 20, 
                 no_action_pause_cost = 0.1, saccade_cost_scale = 0.001):
        super().__init__(env, args, pause_cost, successive_pause_limit, no_action_pause_cost, saccade_cost_scale)
        self.ms_since_motor_step = 0

    def step(self, action):
        pause_action = action["motor_action"] == len(self.get_wrapper_attribute("actions"))
        if pause_action and (self.state is not None):
            # If it has been more than 50ms (results in 20 Hz) since the last motor action 
            # NOOP will be chosen as a motor action instead of continuing the pause
            SLOWED_FRAME_RATE = 20
            if self.ms_since_motor_step >= 1000/SLOWED_FRAME_RATE:
                action["motor_action"] = np.int64(0)
                self.ms_since_motor_step = 0
                return self.step(action)
        else:
            self.ms_since_motor_step = 0

        fov_state, reward, done, truncated, info = super().step(action)

        # Progress the time because a vision step has happened
        # depending on how big the sensory action was
        # Time for a saccade is fixed to 20 ms for now
        self.ms_since_motor_step += 20

        return fov_state, reward, done, truncated, info