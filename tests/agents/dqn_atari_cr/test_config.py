import unittest
from sys import argv

from atari_cr.agents.dqn_atari_cr.main import main, ArgParser

class TestMain(unittest.TestCase):
    def test_main(self):
        argv.extend([
            "--clip_reward",
                "--capture_video",
                "--env", "ms_pacman",
                "--exp_name", "dqn_cr_debug",
                "--total_timesteps", "10000",
                "--learning_start", "1000",
                "--debug",
                "--pause_cost", "0.05",
                "--use_pause_env",
                "--action_repeat", "5",
                "--evaluator",
                "/home/niko/Repos/atari-cr/output/atari_head/ms_pacman/drout0.3/100/checkpoint.pth"
        ])

        args = ArgParser().parse_args(known_only=True)
        main(args)

if __name__ == '__main__':
    unittest.main()
