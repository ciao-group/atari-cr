from ray.rllib.algorithms.dqn import DQNConfig

from atari_cr.agents.dqn_atari_cr.main import make_train_env, ArgParser

config = (
    DQNConfig()
    .environment(make_train_env(ArgParser().parse_args()))
)
algo = config.build()
algo.train()
algo.evaluate()
algo.stop()
