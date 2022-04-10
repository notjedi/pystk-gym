from typing import Callable, List, Tuple, Union

import gym
import numpy as np
import pystk
from gym import spaces

from ..common.actions import ActionType
from ..common.graphics import GraphicConfig
from ..common.kart import Kart
from ..common.race import Race, RaceConfig


class AbstractEnv(gym.Env):
    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        action_type: ActionType,
        reward_func: Callable,
        max_step_cnt: int,
    ):
        self.reward_func = reward_func
        self.action_type = action_type
        self.max_step_cnt = max_step_cnt

        # configure and init pystk
        self.configure(graphic_config, race_config)
        self.define_spaces()
        pystk.init(self.graphics)

        # init karts
        is_reverse, path_width, path_lines, path_distance = (
            self.race.get_race_info()["reverse"],
            self.race.get_path_width(),
            self.race.get_path_lines(),
            self.race.get_path_distance(),
        )
        self.controlled_karts = [
            Kart(
                kart,
                is_reverse,
                path_width,
                path_lines,
                path_distance,
            )
            for kart in self.race.get_controlled_karts()
        ]
        self._init_vars()

    def _init_vars(self):
        self.done = False
        self.steps = 0

    def _obs_space_from_graphics(self) -> spaces.Space:
        return spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.float32),
            high=np.full(self.observation_shape, 255, dtype=np.float32),
        )

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def configure(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
    ) -> None:
        self.graphics = graphic_config.get_pystk_config()
        self.race = Race(race_config.get_pystk_config())
        self.observation_shape = (
            self.graphics.screen_height,
            self.graphics.screen_width,
            3,
        )

    def define_spaces(self) -> None:
        self.observation_space = self._obs_space_from_graphics()
        self.action_space = self.action_type.space()

    def step(
        self, actions: Union[np.ndarray, list, dict]
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:

        # TODO: it would be nice if I could uee `observer()` from the Kart object itself without
        # passing in reference of race to the Kart obj

        self.steps += 1
        actions = self._action(actions)

        self.race.step(actions)
        obs = self.race.observe()
        infos = [kart.step() for kart in self.controlled_karts]
        rewards = self._reward(actions, infos)
        terminals = self._terminal(infos)

        self.done = any(terminals)
        return obs, rewards, terminal, infos

    def _terminal(self, infos: List[dict]) -> List[bool]:
        raise NotImplementedError

    def _action(self, actions) -> List[pystk.Action]:
        return [self.action_type.get_actions(action) for action in actions]

    def _reward(self, actions, infos) -> List[float]:
        return [self.reward_func(action, info) for action, info in zip(actions, infos)]

    def _is_done(self) -> List[bool]:
        return [self.steps > self.max_step_cnt] * len(self.controlled_karts) or [
            kart.is_done() for kart in self.controlled_karts
        ]

    def _reset(self) -> None:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        self.done = False
        self._reset()
        for kart in self.controlled_karts:
            kart.reset()
        # BUG: using reset here would not restart the race
        obs = self.race.reset()
        return obs

    def close(self):
        self.done = True
        self.race.close()
        pystk.clean()
