from active_gym import FixedFovealEnv, AtariEnvArgs, AtariBaseEnv, RecordWrapper
from active_gym.atari_env import AtariEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import cv2
import torch
import numpy as np

# Small cost for a vision step without actual env step 
# to prevent the model from abusing only vision steps
PAUSE_COST = 0.005

class PauseableFixedFovealEnv(FixedFovealEnv):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    """
    def __init__(self, env: gym.Env, args):
        super().__init__(env, args)
        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a 
            # sensory action without the game progressing  
            "motor": Discrete(self.env.action_space.n + 1),
            "sensory": Box(low=self.sensory_action_space[0], 
                                 high=self.sensory_action_space[1], dtype=int),
        })
        # Whether to pause the game at the current step
        self.pause_action = False

        # Count and log the number of pauses made and their cost
        self.n_pauses = 0
        self.pause_cost = PAUSE_COST

    def step(self, action):
        # The pause action is the last action in the action set
        self.pause_action = action["motor"] == len(self.env.actions)
        if self.pause_action:
            # Disallow a pause on the first episode step because there is no
            # observation to look at yet
            if not hasattr(self, "state"):
                action["motor"] = np.random.randint(1, len(self.env.actions))
                return self.step(action)

            # Only make a sensory step with a small cost
            reward, done, truncated = -PAUSE_COST, False, False
            info = { "raw_reward": reward }

            # Log another pause
            self.n_pauses += 1
            
            # Sensory step
            fov_state = self._fov_step(full_state=self.state, action=action["sensory"])
            # Manually execute the RecordWrapper.step code
            assert isinstance(self.env, RecordWrapper), "This code assumes that the parent is a RecordWrapper"
            self.env.ep_len += 1
            self.env.cumulative_reward += reward
            info["reward"] = self.env.cumulative_reward
            info["ep_len"] = self.env.ep_len
            if self.env.record:
                rgb = self.env.env.render()
                self.env._save_transition(self.state, action, self.env.cumulative_reward, done, truncated, info, rgb=rgb, return_reward=reward)
        else:
            # Normal step
            state, reward, done, truncated, info = self.env.step(action=action["motor"])
            # Safe the state for the next sensory step
            self.state = state
            # Sensory step
            fov_state = self._fov_step(full_state=self.state, action=action["sensory"])   

        info["pause_cost"] = PAUSE_COST
        info["n_pauses"] = self.n_pauses
        info["fov_loc"] = self.fov_loc.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])   

        # Reset the pause number for the next episode
        if done:
            self.n_pauses = 0

        return fov_state, reward, done, truncated, info
    
    def save_record_to_file(self, file_path: str, draw_focus = True):
        if self.record:
            video_path = file_path.replace(".pt", ".mp4")
            size = self.prev_record_buffer["rgb"][0].shape[:2][::-1]
            fps = 30
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for i, frame in enumerate(self.prev_record_buffer["rgb"]):
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
                    color = (255, 0, 0)
                    thickness = 1
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                video_writer.write(frame)
            self.prev_record_buffer["rgb"] = video_path
            self.prev_record_buffer["state"] = [0] * len(self.prev_record_buffer["reward"])
            torch.save(self.prev_record_buffer, file_path)
            video_writer.release()

class SlowableFixedFovealEnv(PauseableFixedFovealEnv):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    Additionally, the game can be run at 1/3 of the original
    speed as it was done with the Atari-HEAD dataset
    (http://arxiv.org/abs/1903.06754)
    """
    def __init__(self, env: gym.Env, args):
        super().__init__(env, args)
        self.ms_since_motor_step = 0

    def step(self, action):
        pause_action = action["motor"] == len(self.env.actions)
        if pause_action and (self.state is not None):
            # If it has been more than 50ms (results in 20 Hz) since the last motor action 
            # NOOP will be chosen as a motor action instead of continuing the pause
            SLOWED_FRAME_RATE = 20
            if self.ms_since_motor_step >= 1000/SLOWED_FRAME_RATE:
                action["motor"] = np.int64(0)
                return self.step(action)
            
        fov_state, reward, done, truncated, info = super().step(action)

        # Progress the time because a vision step has happened
        # TODO: depending on how big the sensory action was
        # Time for a saccade is fixed to 20 ms for now
        self.ms_since_motor_step += 20

        return fov_state, reward, done, truncated, info
    
class FovealRecordWrapper(RecordWrapper):
    """
    RecordWrapper that draws the fovea onto the saved video files.
    """
    def reset(self, **kwargs):
        """
        The RecordWrapper.reset function is overwritten because it poorly handles kwargs
        """
        state, info = self.env.reset(**kwargs)

        self.cumulative_reward = 0
        self.ep_len = 0
        info = self._add_info(info)
        if self.record:
            rgb = self.env.render()
            # print ("_reset: reset record buffer")
            self._reset_record_buffer()
            self._save_transition(state, done=False, info=info, rgb=rgb)
        return state, info
    
    def save_record_to_file(self, file_path: str, draw_focus = True):
        if self.record:
            video_path = file_path.replace(".pt", ".mp4")
            size = self.prev_record_buffer["rgb"][0].shape[:2][::-1]
            fps = 30
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for i, frame in enumerate(self.prev_record_buffer["rgb"]):
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
                    color = (255, 0, 0)
                    thickness = 1
                    frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                video_writer.write(frame)
            self.prev_record_buffer["rgb"] = video_path
            self.prev_record_buffer["state"] = [0] * len(self.prev_record_buffer["reward"])
            torch.save(self.prev_record_buffer, file_path)
            video_writer.release()


def PauseableAtariFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    env = AtariEnv(args)
    env = FovealRecordWrapper(env, args)
    env = PauseableFixedFovealEnv(env, args)
    return env
    
def AtariHeadFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = SlowableFixedFovealEnv(base_env, args)
    return wrapped_env