import unittest

import numpy as np

from atari_cr.buffers import DoubleActionReplayBuffer
from gymnasium.spaces import Box, Discrete

class TestDoubleActionReplayBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffer_size = 100_000
        cls.rb = DoubleActionReplayBuffer(
            cls.buffer_size,
            observation_space=Box(-1, 1, [4,84,84]),
            motor_action_space=Discrete(10),
            sensory_action_space=Discrete(64),
            init_sensory_action=np.array([0]),
            device="cuda",
            n_envs=1,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            td_steps=4,
        )

    def test_last_index(self):
        self.rb.pos = self.buffer_size - 1 # Max index
        self.rb.full = True
        self.rb.sample(32)

    def test_first_index(self):
        self.rb.pos = 0
        self.rb.full = True
        self.rb.sample(32)

if __name__ == "__main__":
    unittest.main()
