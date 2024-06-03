from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import pystk
from sympy import Line3D, Point3D

from .info import Info


class Kart:
    def __init__(
        self,
        kart: pystk.Kart,
        is_reverse: bool,
        path_width: npt.NDArray[np.float32],
        path_lines: List[Line3D],
        path_distance: npt.NDArray[np.float32],
        return_info: bool = True,
    ):
        self.kart = kart
        self.id = kart.id
        self.is_reverse = is_reverse
        self.path_width = path_width
        self.path_lines = path_lines
        self.path_distance = path_distance
        self.return_info = return_info

        self.jump_count = 0
        self._prev_info = None
        self.backward_count = 0
        self.no_movement_count = 0
        self.out_of_track_count = 0
        self._node_idx = 0

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
        curr_path_width = self.path_width[self._node_idx][0]
        kart_dist = self._get_kart_dist_from_center()
        return abs(kart_dist) <= (curr_path_width / 2)

    def _get_velocity(self) -> float:
        return np.sqrt(np.sum(np.array(self.kart.velocity) ** 2))

    def is_done(self) -> bool:
        return self.kart.finish_time > 0

    def get_info(self) -> Dict[Info, Any]:
        info = {}

        # basic info
        info[Info.DONE] = self.is_done()
        info[Info.JUMPING] = self._get_jumping()
        info[Info.POWERUP] = self._get_powerup()
        info[Info.LOCATION] = self._get_location()
        info[Info.VELOCITY] = self._get_velocity()
        info[Info.ATTACHMENT] = self._get_attachment()
        info[Info.FINISH_TIME] = self._get_finish_time()
        info[Info.IS_INSIDE_TRACK] = self._get_is_inside_track()
        info[Info.OVERALL_DISTANCE] = self._get_overall_distance()

        # info based on _prev_info
        if self._prev_info:
            delta_dist = (
                info[Info.OVERALL_DISTANCE] - self._prev_info[Info.OVERALL_DISTANCE]
            )
            info[Info.DELTA_DIST] = delta_dist
            if delta_dist < 0:
                info[Info.BACKWARD] = True
                info[Info.NO_MOVEMENT] = False
            elif delta_dist == 0:
                info[Info.BACKWARD] = False
                info[Info.NO_MOVEMENT] = True
            else:
                info[Info.BACKWARD] = False
                info[Info.NO_MOVEMENT] = False
        else:
            info[Info.DELTA_DIST] = 0
            info[Info.BACKWARD] = False
            info[Info.NO_MOVEMENT] = False

        return info

    def step(self) -> Dict[Info, Any]:
        self._update_node_idx()
        if self.return_info:
            info = self.get_info()
            self.out_of_track_count += not info[Info.IS_INSIDE_TRACK]
            self.backward_count += info[Info.BACKWARD]
            self.no_movement_count += info[Info.NO_MOVEMENT]
            if info[Info.JUMPING] and (
                self._prev_info is not None and not self._prev_info[Info.JUMPING]
            ):
                self.jump_count += 1
            info[Info.OUT_OF_TRACK_COUNT] = self.out_of_track_count
            info[Info.BACKWARD_COUNT] = self.backward_count
            info[Info.NO_MOVEMENT_COUNT] = self.no_movement_count
            info[Info.JUMP_COUNT] = self.jump_count
            self._prev_info = info
        else:
            info = {}
        return info

    def reset(self):
        self.jump_count = 0
        self._prev_info = None
        self.backward_count = 0
        self.no_movement_count = 0
        self.out_of_track_count = 0
