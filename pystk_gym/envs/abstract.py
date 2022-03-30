from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import pystk

from ..common.actions import ActionType
from ..common.kart import Kart
from ..common.race import Race, RaceConfig


class AbstractEnv(gym.Env):
    def __init__(
        self,
        race_config: RaceConfig,
        action_type: ActionType,
        reward_func: RewardType,
        observation_type: ObservationType,
    ):
        # TODO: accept config instead of actual objects and use self.configure()
        # TODO: init with default config
        self.reward_func = reward_func
        self.action_type = action_type
        self.observation_type = observation_type
        self.configure(race_config)

        self.done = False
        self.steps = 0
        is_reverse, path_width, path_lines, path_distance = (
            self.race.get_race_info()['reverse'],
            self.race.get_path_width(),
            self.race.get_path_lines(),
            self.race.get_path_distance(),
        )
        self.controlled_karts = [
            Kart(kart, self.observation_type, is_reverse, path_width, path_lines, path_distance)
            for kart in self.race.get_controlled_karts()
        ]
        self.define_spaces()

    def seed(self, seed: int):
        raise NotImplementedError

    def configure(
        self,
        race_config: RaceConfig,
    ):
        self.race = Race(race_config.get_race_config())

    def define_spaces(self):
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def step(
        self, actions: Union[np.ndarray, list, dict]
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:

        self.race.step(actions)
        obs = np.array([kart.observe() for kart in self.controlled_karts])
        infos = [kart.step() for kart in self.controlled_karts]
        rewards = self._reward(actions, infos)
        terminal = self._terminal(infos)

        # TODO: how should i update self.done
        dones = self._is_done()
        self.done = any(dones)

        return obs, rewards, terminal, infos

    def _terminal(self, infos: List[dict]) -> List[bool]:
        raise NotImplementedError

    def _reward(self, actions, infos) -> List[float]:
        return [self.reward_func.get_rewards(action, info) for action, info in zip(actions, infos)]

    def _is_done(self) -> List[bool]:
        return [kart.is_done() for kart in self.controlled_karts]

    def reset(self) -> np.ndarray:
        self.done = False
        self._reset()
        for kart in self.controlled_karts:
            kart.reset()
        obs = self.race.reset()
        return obs

    def _reset(self) -> None:
        raise NotImplementedError

    def close(self):
        self.done = True
        self.race.close()
        pystk.clean()
