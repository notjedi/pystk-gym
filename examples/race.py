import os

# os.environ["SDL_VIDEODRIVER"] = "x11"
# os.environ["SDL_OPENGL_ACCELERATED_VISUAL"] = "0"

import pystk
from pettingzoo.test import parallel_api_test

from pystk_gym import RaceEnv
from pystk_gym.common.graphics import EnvViewer, GraphicConfig, GraphicQuality
from pystk_gym.common.race import RaceConfig
from pystk_gym.common.reward import get_reward_fn

if __name__ == "__main__":
    # import pdb

    # pdb.set_trace()

    pystk.init(GraphicConfig(800, 600, GraphicQuality.HD).get_pystk_config())

    # print("init env_viewer")
    env_viewer = EnvViewer(
        GraphicConfig(800, 600, GraphicQuality.HD),
        human_controlled=True,
        id=1,
    )
    # print("after init")
    # breakpoint()
    # print(f"{env_viewer=}")
    pystk.init(GraphicConfig(800, 600, GraphicQuality.HD).get_pystk_config())

    import time

    time.sleep(5)

    # reward_fn = get_reward_fn()
    # env = RaceEnv(
    #     GraphicConfig(800, 600, GraphicQuality.HD),
    #     RaceConfig.default_config(),
    #     reward_fn,
    #     render_mode="human",
    # )
    # parallel_api_test(env, 1000)
