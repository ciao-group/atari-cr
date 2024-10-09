import unittest
from sys import argv

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

class TestMain(unittest.TestCase):
    def test_main(self):
        """ Deterministic test for OG config """

        argv.extend([
            "--clip_reward",
            "--capture_video",
            "--exp_name", "test_og",
            "--total_timesteps", "5000",
            "--learning_start", "100",
            "--eval_num", "3",
            #
            "--no_pvm_visualization",
            "--no_model_output",
            "--disable_tensorboard"
        ])

        args = ArgParser().parse_args(known_only=True)
        main(args)

if __name__ == '__main__':
    unittest.main()
