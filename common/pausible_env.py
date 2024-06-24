from active_gym import FixedFovealEnv, AtariEnvArgs, AtariBaseEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import cv2
import torch
import numpy as np

class PausibleFixedFovealEnv(FixedFovealEnv):
    """
    Environemt making it possible to be paused to only take
    a sensory action without progressing the game.
    """
    def __init__(self, env: gym.Env, args):
        super().__init__(env, args)
        self.action_space = Dict({
            # One additional action lets the agent stop the game to perform a 
            # sensory action without the game progressing  
            "motor": Discrete(len(self.env.actions) + 1),
            "sensory": Box(low=self.sensory_action_space[0], 
                                 high=self.sensory_action_space[1], dtype=int),
        })

    def step(self, action):
        print(action)
        pause_action = action["motor"] == len(self.env.actions)
        if pause_action and (self.state is not None):
            # Only make a sensory step
            reward, done, truncated = 0, False, False
            info = {"raw_reward": reward if reward else 0}
            fov_state = self._fov_step(full_state=self.state, action=action["sensory"])
        else:
            # Normal step
            state, reward, done, truncated, info = self.env.step(action=action["motor"])
            # Safe the state for the next sensory step
            self.state = state
            fov_state = self._fov_step(full_state=state, action=action["sensory"])

        info["fov_loc"] = self.fov_loc.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
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

class SlowableFixedFovealEnv(PausibleFixedFovealEnv):
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

def PausibleAtariFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = PausibleFixedFovealEnv(base_env, args)
    return wrapped_env
    
def AtariHeadFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = SlowableFixedFovealEnv(base_env, args)
    return wrapped_env