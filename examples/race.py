from pettingzoo.test import parallel_api_test

from pystk_gym import RaceEnv
from pystk_gym.common.graphics import GraphicConfig, GraphicQuality
from pystk_gym.common.race import RaceConfig
from pystk_gym.common.reward import get_reward_fn

if __name__ == "__main__":
    reward_fn = get_reward_fn()
    env = RaceEnv(
        GraphicConfig(800, 600, GraphicQuality.HD),
        RaceConfig.default_config(),
        reward_fn,
    )
    parallel_api_test(env, 1000)
