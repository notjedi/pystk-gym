import numpy as np
import pystk
from sympy import Point3D


class Kart:
    def __init__(
        self,
        kart: pystk.Kart,
        is_reverse: bool,
        path_width: np.ndarray,
        path_lines: np.ndarray,
        path_distance: np.ndarray,
    ) -> None:
        self.kart = kart
        self.is_reverse = is_reverse
        self.path_width = path_width
        self.path_lines = path_lines
        self.path_distance = path_distance

    def _init_vars(self):
        self._node_idx = 0
        self.jump_count = 0
        self.backward_count = 0
        self.no_movement_count = 0
        self.out_of_track_count = 0
        self._prev_info = {}

    def _update_node_idx(self) -> None:
        dist_down_track = (
            0
            if self.is_reverse and self.kart.overall_distance <= 0
            else self.kart.distance_down_track
        )
        path_dist = self.path_distance[self._node_idx]
        while not (path_dist[0] <= dist_down_track <= path_dist[1]):
            if dist_down_track < path_dist[0]:
                self._node_idx -= 1
            elif dist_down_track > path_dist[1]:
                self._node_idx += 1
            path_dist = self.path_distance[self._node_idx]

    def _get_jumping(self) -> bool:
        return self.kart.jumping

    def _get_powerup(self):
        return self.kart.powerup.type

    def _get_attachment(self):
        return self.kart.attachment.type

    def _get_finish_time(self) -> int:
        return int(self.kart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, self.kart.overall_distance)

    def _get_kart_dist_from_center(self):
        # compute the dist b/w the kart and the center of the track
        location = self.kart.location
        path_node = self.path_lines[self._node_idx]
        return path_node.distance(Point3D(location)).evalf()

    def _get_is_inside_track(self):
        # TODO: is 1 a sensitive tolerance to add? or should i change the value?
        # TODO: add user defined tolerance
        curr_path_width = self.path_width[self._node_idx][0]
        kart_dist = self._get_kart_dist_from_center()
        return kart_dist <= ((curr_path_width / 2) + 1)

    def _get_velocity(self):
        # returns the magnitude of velocity
        return np.sqrt(np.sum(np.array(self.kart.velocity) ** 2))

    def is_done(self) -> bool:
        return self.kart.finish_time > 0

    def get_info(self) -> dict:
        info = {}

        # basic info
        info["done"] = self.is_done()
        info["jumping"] = self._get_jumping()
        info["powerup"] = self._get_powerup()
        info["velocity"] = self._get_velocity()
        info["attachment"] = self._get_attachment()
        info["finish_time"] = self._get_finish_time()
        info["is_inside_track"] = self._get_is_inside_track()
        info["overall_distance"] = self._get_overall_distance()

        # count info
        info["jump_count"] = self.jump_count
        info["backward_count"] = self.backward_count
        info["no_movement_count"] = self.no_movement_count
        info["out_of_track_count"] = self.out_of_track_count

        # info based on _prev_info
        if len(self._prev_info) == 0:
            self._prev_info = info

        delta_dist = info["overall_distance"] - self._prev_info["overall_distance"]
        info["delta_dist"] = delta_dist
        if delta_dist < 0:
            info["backward"] = True
            info["no_movement"] = False
        elif delta_dist == 0:
            info["backward"] = False
            info["no_movement"] = True
        else:
            info["backward"] = False
            info["no_movement"] = False

        return info

    def get_prev_info(self) -> dict:
        return self._prev_info

    def step(self) -> dict:
        self._update_node_idx()
        info = self.get_info()

        if not info["is_inside_track"]:
            self.out_of_track_count += 1

        delta_dist = info["delta_dist"]
        if delta_dist < 0:
            self.backward_count += 1
        elif delta_dist == 0:
            self.no_movement_count += 1

        if info["jumping"] and not self._prev_info["jumping"]:
            self.jump_count += 1

        self.prev_info = info
        return info

    def reset(self):
        self._init_vars()
