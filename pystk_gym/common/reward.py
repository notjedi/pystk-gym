from typing import Any, Callable, Dict

import numpy as np
import pystk

from .info import Info


def get_reward_fn() -> Callable:
    class Reward:
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
        if action.nitro and info[Info.NITRO]:
            reward += Reward.NITRO

        if action.drift and info[Info.VELOCITY] > 10:
            reward += Reward.DRIFT
        elif action.drift and info[Info.VELOCITY] < 5:
            reward -= Reward.DRIFT

        if action.fire and info[Info.POWERUP].value:
            reward += Reward.USE_POWERUP

        if info[Info.DONE]:
            reward += Reward.FINISH

        reward += max(0, np.log(info[Info.VELOCITY] + 1e-9))

        reward += -info[Info.RANK] * Reward.POSITION

        if not info[Info.IS_INSIDE_TRACK]:
            reward += Reward.OUT_OF_TRACK

        if info[Info.BACKWARD]:
            reward += Reward.BACKWARDS
        if info[Info.NO_MOVEMENT]:
            reward += Reward.NO_MOVEMENT

        delta_dist = info[Info.DELTA_DIST]
        if delta_dist > 5:
            reward += np.clip(delta_dist, 0, 5)

        if (
            Info.NO_MOVEMENT_COUNT in info
            and info[Info.NO_MOVEMENT_COUNT] >= no_movement_threshold
        ):
            reward += Reward.NO_MOVEMENT

        if info[Info.POWERUP].value:
            reward += Reward.COLLECT_POWERUP

        if info[Info.JUMPING]:
            reward += Reward.JUMP

        return np.clip(reward, -10, 10)

    return reward_fn
