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

    def _terminal(self, infos):

        is_terminal = []
        for info in infos:
            is_terminal.append(info["is_inside_track"])
            is_terminal.append(info["backward"])
            is_terminal.append(info["no_movement"])
            is_terminal.append(info["is_inside_track"])

        return any(is_terminal)

    def done(self):
        raise NotImplementedError
