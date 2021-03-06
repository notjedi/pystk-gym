from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
import pystk
from gym import spaces


class ActionType:
    """
    A type of action specifies its definition space,
    and how actions are executed in the environment.
    """

    POSSIBLE_ACTIONS = [
        "acceleration",
        "brake",
        "steer",
        "fire",
        "drift",
        "nitro",
        "rescue",
    ]

    def __init__(self, action_space, action_list) -> None:
        assert set(action_list).issubset(self.POSSIBLE_ACTIONS)
        self.action_space = action_space
        self.action_list = action_list

    def space(self) -> spaces.Space:
        """The action space."""
        return self.action_space

    def get_actions(self, actions: Union[np.ndarray, dict]) -> pystk.Action:
        """
        Returns a pystk.Action object after updating it with the given parameters.

        :param actions: list of actions to be updated
        """
        raise NotImplementedError()


class MultiDiscreteAction(ActionType):
    """
    A dynamic user-defined MultiDiscrete action space.

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
        action_space: spaces.Space = spaces.MultiDiscrete([2, 2, 3, 2, 2, 2, 2]),
        action_list: Iterable[str] = ActionType.POSSIBLE_ACTIONS,
    ) -> None:
        """
        :param action_space: a gym.spaces.Space object defining the action space
        :param action_list: the names for each action in the action_space (names must be a subset of
        ActionType.POSSIBLE_ACTIONS)
        """
        super().__init__(action_space, action_list)

    def space(self) -> spaces.Space:
        return self.action_space

    def _get_action_obj(
        self, action_names: Iterable[str], actions: Iterable[Union[int, float]]
    ) -> pystk.Action:
        """
        Returns a pystk.Action object after updating the keys with the corresponding values.

        :param keys: a list of action_names
        :param values: a list of action_values
        """
        current_action = pystk.Action()

        for name, action in zip(action_names, actions):
            if name == "steer":
                setattr(current_action, name, action - 1)
            setattr(current_action, name, action)

        return current_action

    def _get_actions_from_dict(self, actions: dict) -> pystk.Action:
        """
        Process a dict object mapping action_name to action_value and returns a pystk.Action object.

        :param actions: dict object mapping object_names to object_values
        """
        assert self.action_space.contains(actions.values())
        assert set(actions.keys()).issubset(self.action_list)
        return self._get_action_obj(actions.keys(), actions.values())

    def _get_actions_from_list(self, actions: Union[list, np.ndarray]) -> pystk.Action:
        """
        Process a list of action values and returns a pystk.Action object.

        :param actions: action values
        """
        assert self.action_space.contains(actions)
        return self._get_action_obj(self.action_list, actions)

    def get_actions(self, actions: Union[np.ndarray, List, dict]) -> pystk.Action:
        if isinstance(actions, dict):
            return self._get_actions_from_dict(actions)
        if isinstance(actions, (list, np.ndarray)):
            return self._get_actions_from_list(actions)
        if isinstance(actions, pystk.Action):
            return actions
        raise NotImplementedError
