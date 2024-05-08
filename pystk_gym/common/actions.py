from typing import Dict, Iterable, List, Union

import numpy as np
import numpy.typing as npt
import pystk
from gymnasium import spaces

ActionType = Union[
    pystk.Action,
    List[Union[int, float]],
    Dict[str, Union[int, float]],
    npt.NDArray[Union[np.float64, np.int64]],
]


def get_stk_action_obj(
    action_names: Iterable[str], actions_values: Iterable[Union[int, float]]
) -> pystk.Action:
    """
    Returns a pystk.Action object after updating the keys with the corresponding values.

    :param keys: a list of action_names
    :param values: a list of action_values
    """
    #  0             1      2      3     4      5      6        # index
    # [2,            2,     3,     2,    2,     2,     2]       # action_space
    # {acceleration, brake, steer, fire, drift, nitro, rescue}  # fields
    current_action = pystk.Action()

    for name, action in zip(action_names, actions_values):
        if name == "steer":
            setattr(current_action, name, action - 1)
        setattr(current_action, name, action)

    return current_action


class MultiDiscreteAction:
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

    ACTIONS = ["acceleration", "brake", "steer", "fire", "drift", "nitro", "rescue"]

    def __init__(self):
        self.action_space = spaces.MultiDiscrete([2, 2, 3, 2, 2, 2, 2])

    def _get_actions_from_dict(self, actions: dict) -> pystk.Action:
        """
        Process a dict object mapping action_name to action_value and returns a pystk.Action object.

        :param actions: dict object mapping object_names to object_values
        """
        return get_stk_action_obj(actions.keys(), actions.values())

    def _get_actions_from_list(
        self,
        actions: Union[
            List[Union[int, float]], npt.NDArray[Union[np.float64, np.int64]]
        ],
    ) -> pystk.Action:
        """
        Process a list of action values and returns a pystk.Action object.

        :param actions: action values
        """
        assert self.action_space.contains(actions)
        return get_stk_action_obj(MultiDiscreteAction.ACTIONS, actions)

    def get_pystk_action(self, actions: ActionType) -> pystk.Action:
        if isinstance(actions, dict):
            return self._get_actions_from_dict(actions)
        if isinstance(actions, (list, np.ndarray)):
            return self._get_actions_from_list(actions)
        if isinstance(actions, pystk.Action):
            return actions
        raise NotImplementedError

    def space(self) -> spaces.MultiDiscrete:
        """The action space."""
        return self.action_space
