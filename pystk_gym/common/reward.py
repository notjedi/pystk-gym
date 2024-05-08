from typing import Callable

import numpy as np
import pystk


def get_reward_fn() -> Callable:
    FINISH = 1
    COLLECT_POWERUP = 0.2
    USE_POWERUP = 0.2
    DRIFT = 0.2
    NITRO = 0.2
    EARLY_END = -1
    NO_MOVEMENT = -0.2
    OUT_OF_TRACK = -0.3
    BACKWARDS = -0.7
    JUMP = -0.3

    no_movement_threshold = 5

    def reward_fn(action: pystk.Action, info):
        reward = -0.02
        # if action.nitro and info["nitro"]:
        #     reward += NITRO

        if action.drift and info["velocity"] > 10:
            reward += DRIFT
        elif action.drift and info["velocity"] < 5:
            reward -= DRIFT

        if action.fire and info["powerup"].value:
            reward += USE_POWERUP

        if info["done"]:
            reward += FINISH

        reward += max(0, np.log(info["velocity"] + 1e-9))

        # TODO: add position stuff to info
        # POSITION = 0.5
        # if info["position"] < prevInfo["position"]:
        #     reward += POSITION
        # elif info["position"] > prevInfo["position"]:
        #     reward -= POSITION

        if not info["is_inside_track"]:
            reward += OUT_OF_TRACK

        if info["backward"]:
            reward += BACKWARDS
        if info["no_movement"]:
            reward += NO_MOVEMENT

        delta_dist = info["delta_dist"]
        if delta_dist > 5:
            reward += np.clip(delta_dist, 0, 5)

        if info["no_movement_count"] >= no_movement_threshold:
            reward += NO_MOVEMENT

        if info["powerup"].value:
            reward += COLLECT_POWERUP

        if info["jumping"]:
            reward += JUMP

        if info.get("early_end", False):
            reward += EARLY_END

        return np.clip(reward, -10, 10)

    return reward_fn
