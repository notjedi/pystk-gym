from typing import Union, Tuple, List, Optional

import gym
import numpy as np
import pystk

from ..common.actions import ActionType
from ..common.race import Race
from ..common.kart import Kart


class AbstractEnv(gym.Env):
    def __init__(
        self,
        race: Race,
        action_type: ActionType,
        reward_func: RewardType,
        observation_type: ObservationType,
    ):
        # TODO: accept config instead of actual objects
        # TODO: init with default config
        self.race = race
        self.reward_func = reward_func
        self.action_type = action_type
        self.observation_type = observation_type

        self.done = False
        self.steps = 0
        self.controlled_karts = [
            Kart(kart, self.observation_type) for kart in self.get_controlled_karts()
        ]
        self.define_spaces()

    def seed(self, seed: int):
        raise NotImplementedError

    def configure(self, race: Race,)

    def define_spaces(self):
        self.observation_space = self.observation_type.space()
        self.action = self.action_type.space()

    def get_controlled_karts(self) -> list:
        controlled_karts = []
        for kart in self.race.get_all_karts():
            if kart.controller == pystk.PlayerConfig.Controller.PLAYER_CONTROL:
                controlled_karts.append(kart)
        return controlled_karts

    def step(
        self, actions: Union[np.ndarray, list, dict]
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:

        # TODO
        obs = np.array([kart.observe() for kart in self.controlled_karts])
        infos = self._info()
        rewards = self._reward(actions, infos)
        dones = self._is_done()

        self.done = any(dones)

        return obs, rewards, dones, infos

    def _reward(self, actions, infos) -> List[float]:
        return self.reward_func.get_rewards(actions, infos)

    def _info(self) -> List[dict]:
        return [kart.get_info() for kart in self.controlled_karts]

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


class RaceEnv(AbstractEnv):
    def __init__(self, race: Race, action_type: ActionType):
        self.race = race
        self._node_idx = 0
        self.action_type = action_type
        self.reverse = self.race.get_config().reverse

    def step(self, actions: Optional[np.ndarray] = None):
        # TODO: convert actions to pystk.Action
        # TODO: vectorize get_actions
        # self.action_type.get_actions(actions)
        self.race.step(actions)
        for kart in self.controlled_karts:
            kart._update_node_idx()

    def reset(self):
        for kart in self.controlled_karts:
            kart.reset()

    def done(self):
        raise NotImplementedError
