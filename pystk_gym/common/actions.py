from enum import Enum

import numpy as np
import pystk


class Action(Enum):
    """
    Enum class to represent all actions in the game.

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

    # ACTION = (action_value, action_name)
    ACCELERATION = (1, "acceleration")
    BRAKE = (0, "brake")
    STEER = (0, "steer")
    FIRE = (0, "fire")
    DRIFT = (0, "drift")
    NITRO = (0, "nitro")
    RESCUE = (0, "rescue")

    def __init__(self, action_value: int, action_name: str) -> None:
        """
        :param action_value: value of the action.
        :param action_name: name of the action, should be the same as the
        attribute name in the `pystk.Action` class.
        """
        self.action_value = action_value
        self.action_name = action_name

    @classmethod
    def check_value_range(cls, action: "Action") -> bool:
        if action.value == cls.STEER:
            return -1 < action.action_value <= 1
        return 0 <= action.action_value <= 1

    @classmethod
    def get_actions(cls, actions: np.ndarray | list) -> pystk.Action:
        """
        :param actions: list of actions
        Returns a `pystk.Action` object after updating it with `actions`.
        """
        current_action = pystk.Action()
        for action in actions:
            if action.value == cls.STEER:
                setattr(current_action, action.action_name, action.action_value - 1)
            setattr(current_action, action.action_name, action.action_value)
        return current_action
