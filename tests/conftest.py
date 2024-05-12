import pytest

from pystk_gym.common.graphics import GraphicConfig
from pystk_gym.common.race import RaceConfig
from pystk_gym.common.reward import get_reward_fn
from pystk_gym.envs.race_env import RaceEnv


@pytest.fixture(name="track_kart_env")
def track_kart_race_env(track, kart):
    race_config = RaceConfig(track=track, kart=kart)
    env = RaceEnv(
        GraphicConfig.default_config(),
        race_config,
        get_reward_fn(),
    )
    yield env
    env.close()


@pytest.fixture
def race_env(graphic_conf: GraphicConfig, race_conf: RaceConfig):
    env = RaceEnv(
        graphic_conf,
        race_conf,
        get_reward_fn(),
    )
    yield env
    env.close()
