from typing import Any, Callable, Dict

import numpy as np
import pystk

from .info import Info


def get_reward_fn() -> Callable:
    FINISH = 1
    COLLECT_POWERUP = 0.2
    USE_POWERUP = 0.2
    DRIFT = 0.2
    NITRO = 0.2
    POSITION = 0.5
    NO_MOVEMENT = -0.2
    OUT_OF_TRACK = -0.3
    BACKWARDS = -0.7
    JUMP = -0.3

    no_movement_threshold = 5

    def reward_fn(action: pystk.Action, info: Dict[Info, Any]) -> float:
        reward = -0.02
        if action.nitro and info[Info.Nitro]:
            reward += NITRO

        if action.drift and info[Info.Velocity] > 10:
            reward += DRIFT
        elif action.drift and info[Info.Velocity] < 5:
            reward -= DRIFT

        if action.fire and info[Info.Powerup].value:
            reward += USE_POWERUP

        if info[Info.Done]:
            reward += FINISH

        reward += max(0, np.log(info[Info.Velocity] + 1e-9))

        reward += -info[Info.Rank] * POSITION

        if not info[Info.IsInsideTrack]:
            reward += OUT_OF_TRACK

        if info[Info.Backward]:
            reward += BACKWARDS
        if info[Info.NoMovement]:
            reward += NO_MOVEMENT

        delta_dist = info[Info.DeltaDist]
        if delta_dist > 5:
            reward += np.clip(delta_dist, 0, 5)

        if info[Info.NoMovementCount] >= no_movement_threshold:
            reward += NO_MOVEMENT

        if info[Info.Powerup].value:
            reward += COLLECT_POWERUP

        if info[Info.Jumping]:
            reward += JUMP

        return np.clip(reward, -10, 10)

    return reward_fn
