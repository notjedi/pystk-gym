from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import pystk
from sympy import Point3D, Line3D

from .info import Info


class Kart:
    def __init__(
        self,
        kart: pystk.Kart,
        is_reverse: bool,
        path_width: npt.NDArray[np.float32],
        path_lines: List[Line3D],
        path_distance: npt.NDArray[np.float32],
    ):
        self.kart = kart
        self.id = kart.id
        self.is_reverse = is_reverse
        self.path_width = path_width
        self.path_lines = path_lines
        self.path_distance = path_distance
        self._node_idx = 0

    def _init_vars(self):
        self.jump_count = 0
        self._prev_info = None
        self.backward_count = 0
        self.no_movement_count = 0
        self.out_of_track_count = 0

    def _update_node_idx(self):
        # TODO: sanity check this logic when self.is_reverse == True
        dist_down_track = (
            0
            if self.is_reverse and self.kart.overall_distance <= 0
            else self.kart.distance_down_track
        )
        path_dist = self.path_distance[self._node_idx]
        while not path_dist[0] <= dist_down_track <= path_dist[1]:
            if dist_down_track < path_dist[0]:
                self._node_idx -= 1
            elif dist_down_track > path_dist[1]:
                self._node_idx += 1
            path_dist = self.path_distance[self._node_idx]

    def _get_jumping(self) -> bool:
        return self.kart.jumping

    def _get_powerup(self) -> pystk.Powerup.Type:
        return self.kart.powerup.type

    def _get_attachment(self) -> pystk.Attachment.Type:
        return self.kart.attachment.type

    def _get_finish_time(self) -> int:
        return int(self.kart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, self.kart.overall_distance)

    def _get_location(self) -> List[int]:
        return self.kart.location

    def _get_kart_dist_from_center(self) -> float:
        # compute the dist b/w the kart and the center of the track
        location = self.kart.location
        path_node = self.path_lines[self._node_idx]
        return float(path_node.distance(Point3D(location)).evalf())  # type: ignore

    def _get_is_inside_track(self) -> bool:
        # TODO: is 1 a sensitive tolerance to add? or should i change the value?
        # TODO: add user defined tolerance
        curr_path_width = self.path_width[self._node_idx][0]
        kart_dist = self._get_kart_dist_from_center()
        return abs(kart_dist) <= ((curr_path_width / 2) + 1)

    def _get_velocity(self) -> float:
        # returns the magnitude of velocity
        return np.sqrt(np.sum(np.array(self.kart.velocity) ** 2))

    def is_done(self) -> bool:
        return self.kart.finish_time > 0

    def get_info(self) -> Dict[Info, Any]:
        info = {}

        # basic info
        info[Info.Done] = self.is_done()
        info[Info.Jumping] = self._get_jumping()
        info[Info.Powerup] = self._get_powerup()
        info[Info.Location] = self._get_location()
        info[Info.Velocity] = self._get_velocity()
        info[Info.Attachment] = self._get_attachment()
        info[Info.FinishTime] = self._get_finish_time()
        info[Info.IsInsideTrack] = self._get_is_inside_track()
        info[Info.OverallDistance] = self._get_overall_distance()

        # count info
        info[Info.JumpCount] = self.jump_count
        info[Info.BackwardCount] = self.backward_count
        info[Info.NoMovementCount] = self.no_movement_count
        info[Info.OutOfTrackCount] = self.out_of_track_count

        # info based on _prev_info
        if self._prev_info:
            delta_dist = (
                info[Info.OverallDistance] - self._prev_info[Info.OverallDistance]
            )
            info[Info.DeltaDist] = delta_dist
            if delta_dist < 0:
                info[Info.Backward] = True
                info[Info.NoMovement] = False
            elif delta_dist == 0:
                info[Info.Backward] = False
                info[Info.NoMovement] = True
            else:
                info[Info.Backward] = False
                info[Info.NoMovement] = False
        else:
            info[Info.DeltaDist] = 0
            info[Info.Backward] = False
            info[Info.NoMovement] = False

        return info

    def step(self) -> Dict[Info, Any]:
        self._update_node_idx()
        info = self.get_info()

        if not info[Info.IsInsideTrack]:
            self.out_of_track_count += 1

        delta_dist = info[Info.DeltaDist]
        if delta_dist < 0:
            # TODO: check if vals(backward and no_movement) are assigned correctly
            self.backward_count += 1
        elif delta_dist == 0:
            self.no_movement_count += 1

        if info[Info.Jumping] and (
            self._prev_info is not None and not self._prev_info[Info.Jumping]
        ):
            self.jump_count += 1

        self.prev_info = info
        return info

    def reset(self):
        self._init_vars()
