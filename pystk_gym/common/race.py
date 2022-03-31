from __future__ import annotations

from typing import Iterable, Optional, Union, List

import numpy as np
import pystk
from numpy.random import choice
from sympy import Line3D


class RaceConfig:

    TRACKS = [
        "abyss",
        "black_forest",
        "candela_city",
        "cocoa_temple",
        "cornfield_crossing",
        "fortmagma",
        "gran_paradiso_island",
        "hacienda",
        "lighthouse",
        "minigolf",
        "olivermath",
        "ravenbridge_mansion",
        "sandtrack",
        "scotland",
        "snowmountain",
        "snowtuxpeak",
        "stk_enterprise",
        "volcano_island",
        "xr591",
        "zengarden",
    ]

    KARTS = [
        "adiumy",
        "amanda",
        "beastie",
        "emule",
        "gavroche",
        "gnu",
        "hexley",
        "kiki",
        "konqi",
        "nolok",
        "pidgin",
        "puffy",
        "sara_the_racer",
        "sara_the_wizard",
        "suzanne",
        "tux",
        "wilber",
        "xue",
    ]

    def __init__(
        self,
        track: str | None = None,
        kart: str | None = None,
        num_karts: int = 5,
        laps: int = 1,
        reverse: bool | None = None,
        difficulty: int = 1,
        step_size: float = 0.045,
        self_control: bool = False,
    ) -> None:
        self.track = track
        self.kart = kart
        self.num_karts = num_karts
        self.laps = laps
        self.reverse = reverse
        self.difficulty = difficulty
        self.step_size = step_size
        self.self_control = self_control

    def get_pystk_config(self) -> pystk.RaceConfig:
        return self.get_race_config(
            self.track,
            self.kart,
            self.num_karts,
            self.laps,
            self.reverse,
            self.difficulty,
            self.step_size,
            self.self_control,
        )

    @staticmethod
    def default_config() -> RaceConfig:
        return RaceConfig(
            track="hacienda",
            kart="tux",
            num_karts=5,
            laps=1,
            reverse=False,
            difficulty=1,
        )

    @staticmethod
    def get_race_config(
        track: str | None = None,
        kart: str | None = None,
        num_karts: int = 5,
        laps: int = 1,
        reverse: bool | None = None,
        difficulty: int = 1,
        step_size: float = 0.045,
        self_control: bool = False,
    ) -> pystk.RaceConfig:

        track = choice(RaceConfig.TRACKS) if track is None else track
        kart = choice(RaceConfig.KARTS) if kart is None else kart
        reverse = choice([True, False]) if reverse is None else reverse

        # add a matrix/grid check test to check all combinations of TRACKS and KARTS
        # TODO: assert all tracks work
        # TODO: assert difficulty
        # TODO: add fps kinda thing in hertz like highway_env
        # range of difficulty is 1-3 # TODO: check
        assert track in RaceConfig.TRACKS
        assert kart in RaceConfig.KARTS

        config = pystk.RaceConfig()
        config.track = track
        config.num_kart = num_karts
        config.laps = laps
        config.reverse = reverse
        config.difficulty = difficulty
        config.step_size = step_size

        # TODO: try to parameterize team after testing
        # TODO: get image frame of all team players to speed up training
        config.players[0].team = 0
        config.players[0].kart = kart
        config.players[0].controller = (
            pystk.PlayerConfig.Controller.AI_CONTROL
            if self_control
            else pystk.PlayerConfig.Controller.PLAYER_CONTROL
        )
        return config


class Race:
    def __init__(self, config: pystk.RaceConfig) -> None:
        self.config = config
        self.done = False
        self.race = pystk.Race(self.config)
        self.track = pystk.Track()
        self.state = pystk.WorldState()

        self._node_idx = 0
        self.reverse = self.config.reverse
        self.controlled_karts_idxs = None

    def get_race_info(self) -> dict:
        # TODO: should i do return self.config.__dict__?
        info = {}
        info["laps"] = self.config.laps
        info["track"] = self.config.track
        info["reverse"] = self.config.reverse
        info["num_kart"] = self.config.num_kart
        info["step_size"] = self.config.step_size
        info["difficulty"] = self.config.difficulty
        return info

    def get_config(self) -> pystk.RaceConfig:
        return self.config

    def get_state(self) -> pystk.WorldState:
        return self.state

    def get_path_lines(self) -> np.ndarray:
        return np.array([Line3D(*node) for node in self.track.path_nodes])

    def get_path_width(self) -> np.ndarray:
        return np.array(self.track.path_width)

    def get_path_distance(self) -> np.ndarray:
        return np.array(
            sorted(self.track.path_distance[::-1], key=lambda x: x[0])
            if self.reverse
            else self.track.path_distance
        )

    def get_all_karts(self) -> List:
        return self.state.karts

    def get_controlled_karts(self) -> List:
        return list(np.array(self.get_all_karts())[self.get_controlled_kart_idxs()])

    def get_controlled_kart_idxs(self) -> list:
        if self.controlled_karts_idxs is None:
            self.controlled_karts_idxs = [
                kart.controller == pystk.PlayerConfig.Controller.PLAYER_CONTROL
                for kart in self.get_all_karts()
            ]
        return self.controlled_karts_idxs

    def step(self, actions: Optional[Union[pystk.Action, Iterable[pystk.Action]]]) -> np.ndarray:
        if actions is not None:
            self.race.step(actions)
        else:
            self.race.step()

        self.state.update()
        self.track.update()
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.array(self.race.render_data)[self.get_controlled_kart_idxs()]

    def reset(self) -> np.ndarray:
        self.race.start()
        self.race.step()
        self.state.update()
        self.track.update()
        self.done = False
        return self.observe()

    def close(self) -> None:
        self.race.stop()
        self.done = True
