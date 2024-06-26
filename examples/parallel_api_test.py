import argparse

from pettingzoo.test import parallel_api_test

from pystk_gym import RaceEnv
from pystk_gym.common.graphics import GraphicConfig, GraphicQuality
from pystk_gym.common.race import RaceConfig
from pystk_gym.common.reward import get_reward_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pystk RaceEnv")
    parser.add_argument(
        "-m", "--mode", choices=["agent", "human", "rgb_array"], default="rgb_array"
    )
    args = parser.parse_args()

    reward_fn = get_reward_fn()
    race_config = RaceConfig.default_config()
    race_config.reverse = True
    race_config.num_karts = 1
    race_config.num_karts_controlled = 0
    env = RaceEnv(
        GraphicConfig(800, 600, GraphicQuality.HD),
        race_config,
        reward_fn,
        render_mode=args.mode,
    )
    parallel_api_test(env, 1000)
    env.close()
