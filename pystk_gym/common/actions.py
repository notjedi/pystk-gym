from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pystk
from gym import spaces


class ActionType(object):
    """
    A type of action specifies its definition space,
    and how actions are executed in the environment.
    """

    POSSIBLE_ACTIONS = ["acceleration", "brake", "steer", "fire", "drift", "nitro", "rescue"]

    def __init__(self) -> None:
        self.current_action = pystk.Action()

    def space(self) -> spaces.Space:
        raise NotImplementedError

    def get_actions(self, actions: Optional[Union[np.ndarray, dict]] = None) -> pystk.Action:
        raise NotImplementedError


class MultiDiscreteAction(ActionType):
    """
    -----------------------------------------------------------------
    |         ACTIONS               |       POSSIBLE VALUES         |
    -----------------------------------------------------------------
    |       Acceleration            |           (0, 1)              |
    |       Brake                   |           (0, 1)              |
    |       Steer                   |         (-1, 0, 1)            |
    |       Fire                    |           (0, 1)              |
    |       Drift                   |           (0, 1)              |
    |       Nitro                   |           (0, 1)              |
    |       Rescue                  |           (0, 1)              |
    -----------------------------------------------------------------
    """

    def __init__(
        self,
        action_space: spaces.Space = spaces.MultiDiscrete([2, 2, 3, 2, 2, 2]),
        action_list: Iterable[str] = ActionType.POSSIBLE_ACTIONS,
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.action_list = action_list

    def space(self) -> spaces.Space:
        return self.action_space

    def _get_actions_from_dict(self, actions: dict) -> pystk.Action:
        assert self.action_space.contains(actions.values())
        assert set(actions.keys()).issubset(self.action_list)
        self.current_action = pystk.Action()

        for key, value in actions.items():
            if key == "steer":
                setattr(self.current_action, key, value - 1)
            setattr(self.current_action, key, value)

        return self.current_action

    def _get_actions_from_list(self, actions: np.ndarray) -> pystk.Action:
        assert self.action_space.contains(actions)
        self.current_action = pystk.Action()

        for key, value in zip(self.action_list, actions):
            if key == "steer":
                setattr(self.current_action, key, value - 1)
            setattr(self.current_action, key, value)

        return self.current_action

    def get_actions(self, actions: Union[np.ndarray, dict]) -> pystk.Action:
        if isinstance(actions, dict):
            return self._get_actions_from_dict(actions)
        elif isinstance(actions, np.ndarray):
            return self._get_actions_from_list(actions)
