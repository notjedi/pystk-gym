from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np

from ..common.actions import ActionType, MultiDiscreteAction
from ..common.graphics import GraphicConfig
from ..common.kart import Kart
from ..common.race import RaceConfig
from .abstract import AbstractEnv


class RaceEnv(AbstractEnv):
    action_aliases: Dict[str, Type[ActionType]] = {
        "MultiDiscrete": MultiDiscreteAction,
    }

    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        action_type: Union[Type[ActionType], str],
        reward_func: Callable,
        max_step_cnt: int = 1000,
    ):

        super().__init__(graphic_config, race_config, action_type, reward_func, max_step_cnt)
        # TODO: should i add an kwargs argument for instantiating this class?
        self.action_type = self.action_type_class()
        self._node_idx = 0
        self.reverse = self.race.get_config().reverse

    def get_controlled_karts(self) -> List[Kart]:
        return self.controlled_karts

    def _terminal(self, infos):
        step_limit_reached = self.steps > self.max_step_cnt
        return [
            info["is_inside_track"]
            or info["backward"]
            or info["no_movement"]
            or kart.is_done()
            or step_limit_reached
            for info, kart in zip(infos, self.get_controlled_karts())
        ]

    def _make_karts(self) -> None:
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

    def _step(
        self, obs, rewards, terminals, infos
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:
        nitro_locs = self.race.get_nitro_locs()
        all_kart_positions = self.race.get_all_kart_positions()

        for info, kart in zip(infos, self.get_controlled_karts()):
            info["position"] = all_kart_positions[kart.id]
            kart_loc = np.array(info['location'])
            # TODO: make this more functional programming like
            info["nitro"] = any(
                np.sqrt(np.sum(kart_loc - nitro_loc)) <= 1 for nitro_loc in nitro_locs
            )

        return obs, rewards, terminals, infos

    def _reset(self) -> None:
        self._make_karts()

    def is_done(self):
        return any(self._is_done())
