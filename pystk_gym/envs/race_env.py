from typing import Optional

import numpy as np

from ..common.actions import ActionType
from ..common.race import Race
from .abstract import AbstractEnv


class RaceEnv(AbstractEnv):
    def __init__(self, race: Race, action_type: ActionType):
        self.race = race
        self._node_idx = 0
        self.action_type = action_type
        self.reverse = self.race.get_config().reverse

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def _terminal(self, infos):
        step_limit_reached = self.steps > self.max_step_cnt
        return [
            info['is_inside_track']
            or info['backward']
            or info['no_movement']
            or step_limit_reached
            for info in infos
        ]

    def _reset(self) -> None:
        raise NotImplementedError

    def done(self):
        return self.done
