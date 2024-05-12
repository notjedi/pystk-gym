import pytest
from pettingzoo.test import parallel_api_test

from pystk_gym.common.graphics import GraphicConfig
from pystk_gym.common.race import RaceConfig


@pytest.mark.parametrize("track", RaceConfig.TRACKS)
@pytest.mark.parametrize("kart", RaceConfig.KARTS)
def test_track_kart_compatiblity(track_kart_env):
    assert True


@pytest.mark.parametrize(
    "graphic_conf, race_conf",
    [(GraphicConfig.default_config(), RaceConfig.default_config())],
)
def test_api(race_env):
    parallel_api_test(race_env, 1000)
