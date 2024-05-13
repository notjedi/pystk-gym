from __future__ import annotations

import functools
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pystk
from sympy import Line3D

ObsType = np.ndarray[np.ndarray, np.dtype[np.uint8]]


class RaceConfig:
    TRACKS = [
        "abyss", "black_forest", "candela_city", "cocoa_temple", "cornfield_crossing", "fortmagma",
        "gran_paradiso_island", "hacienda", "lighthouse", "minigolf", "olivermath",
        "ravenbridge_mansion", "sandtrack", "scotland", "snowmountain", "snowtuxpeak",
        "stk_enterprise", "volcano_island", "xr591", "zengarden"
    ]  # fmt: skip
    KARTS = [
        "adiumy", "amanda", "beastie", "emule", "gavroche", "gnu", "hexley", "kiki", "konqi",
        "nolok", "pidgin", "puffy", "sara_the_racer", "sara_the_wizard", "suzanne", "tux", "wilber",
        "xue"
    ]  # fmt: skip
    NITRO_TYPE = [pystk.Item.Type.NITRO_SMALL, pystk.Item.Type.NITRO_BIG]

    def __init__(
        self,
        track: Optional[str] = None,
        kart: Optional[Union[str, List[str]]] = None,
        num_karts: int = 5,
        laps: int = 1,
        reverse: Optional[bool] = None,
        seed: int = 1337,
        difficulty: int = 1,
        step_size: float = 0.045,
        num_karts_controlled: int = 3,
    ):
        self.track = track
        self.kart = kart
        self.num_karts = num_karts
        self.laps = laps
        self.reverse = reverse
        self.seed = seed
        self.difficulty = difficulty
        self.step_size = step_size
        self.num_karts_controlled = num_karts_controlled

    def build(self) -> pystk.RaceConfig:
        return RaceConfig.get_race_config(
            self.track,
            self.kart,
            self.num_karts,
            self.laps,
            self.reverse,
            self.seed,
            self.difficulty,
            self.step_size,
            self.num_karts_controlled,
        )

    @staticmethod
    def default_config() -> RaceConfig:
        return RaceConfig(
            track="hacienda",
            kart="tux",
            num_karts=5,
            laps=1,
            reverse=False,
            seed=1337,
            difficulty=1,
            num_karts_controlled=1,
        )

    @staticmethod
    def get_race_config(
        track: Optional[str] = None,
        karts: Optional[Union[str, List[str]]] = None,
        num_karts: int = 5,
        laps: int = 1,
        reverse: Optional[bool] = None,
        seed: int = 1337,
        difficulty: int = 1,
        step_size: float = 0.045,
        num_karts_controlled: int = 4,
    ) -> pystk.RaceConfig:
        track = np.random.choice(RaceConfig.TRACKS) if track is None else track
        karts = np.random.choice(RaceConfig.KARTS) if karts is None else karts
        reverse = np.random.choice([True, False]) if reverse is None else reverse
        assert num_karts >= num_karts_controlled

        # TODO: add fps kinda thing in hertz like highway_env - is this what step_size does?
        if isinstance(karts, list):
            assert set(karts).issubset(
                RaceConfig.KARTS
            ), f"{karts} contains 1 or more invalid karts"
            assert len(karts) == num_karts_controlled
        elif isinstance(karts, str):
            assert karts in RaceConfig.KARTS, f"{karts} is not a valid kart."
            karts = [karts] * num_karts_controlled
        elif karts is None:
            karts = list(np.random.choice(RaceConfig.KARTS, size=num_karts_controlled))
        else:
            raise ValueError(f"does not support type {type(karts)} for list of karts.")

        assert track in RaceConfig.TRACKS, f"{track} is not a valid track."
        assert (
            1 <= difficulty <= 3
        ), f"Difficulty({difficulty}) should be between 1 and 3 (inclusive)"

        config = pystk.RaceConfig()
        config.track = track
        config.num_kart = num_karts
        config.laps = laps
        config.reverse = reverse
        config.seed = seed
        config.difficulty = difficulty
        config.step_size = step_size

        config.players[0].team = 0
        config.players[0].kart = karts[0]
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        for kart in karts[1:]:
            config.players.append(
                pystk.PlayerConfig(
                    kart, pystk.PlayerConfig.Controller.PLAYER_CONTROL, 0
                )
            )

        # game controlled karts
        for _ in range(num_karts - num_karts_controlled):
            config.players.append(
                pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1)
            )

        return config


class Race:
    def __init__(self, config: pystk.RaceConfig):
        self.config = config
        self.race = pystk.Race(self.config)
        self.track = pystk.Track()
        self.state = pystk.WorldState()

        self.race.start()
        self.race.step()
        self.state.update()
        self.track.update()
        self.reset()

    def get_race_info(self) -> Dict[str, Any]:
        info = {}
        info["laps"] = self.config.laps
        info["track"] = self.config.track
        info["reverse"] = self.config.reverse
        info["num_kart"] = self.config.num_kart
        info["step_size"] = self.config.step_size
        info["difficulty"] = self.config.difficulty
        return info

    def get_path_lines(self) -> List[Line3D]:
        return [Line3D(*node) for node in self.track.path_nodes]

    def get_path_width(self) -> npt.NDArray[np.float32]:
        return np.array(self.track.path_width)

    def get_path_distance(self) -> npt.NDArray[np.float32]:
        return np.array(
            sorted(self.track.path_distance[::-1], key=lambda x: x[0])
            if self.config.reverse
            else self.track.path_distance
        )

    @functools.lru_cache(maxsize=None)
    def get_controlled_kart_mask(self) -> List[bool]:
        # there are better ways to do this but i think this is the best way to be sure that we are
        # getting the correct player karts
        controlled_karts_idxs = []
        for i, (kart, player) in enumerate(
            zip(self.get_all_karts(), self.state.players)
        ):
            assert kart.id == player.kart.id
            controlled_karts_idxs.append(
                self.config.players[i].controller
                == pystk.PlayerConfig.Controller.PLAYER_CONTROL
            )
        return controlled_karts_idxs

    def get_all_karts(self) -> List[pystk.Kart]:
        return self.state.karts

    def get_controlled_karts(self) -> List[pystk.Kart]:
        return list(np.array(self.get_all_karts())[self.get_controlled_kart_mask()])

    def get_nitro_locs(self) -> List[List[int]]:
        return [
            item.location for item in self.state.items if item in RaceConfig.NITRO_TYPE
        ]

    def get_all_kart_rankings(self) -> Dict[int, int]:
        overall_dists = {
            kart.id: kart.overall_distance for kart in self.get_all_karts()
        }
        return {
            id: self.config.num_kart - i
            for i, (id, _) in enumerate(
                sorted(overall_dists.items(), key=lambda x: x[1])
            )
        }

    def observe(self) -> ObsType:
        return np.array(
            list(map(lambda x: x.image, self.race.render_data)), dtype=np.uint8
        )[self.get_controlled_kart_mask()]

    def step(
        self, actions: Optional[Union[pystk.Action, Iterable[pystk.Action]]]
    ) -> ObsType:
        if actions is not None:
            self.race.step(actions)
        else:
            self.race.step()

        self.state.update()
        self.track.update()
        return self.observe()

    def reset(self) -> ObsType:
        return self.observe()

    def close(self):
        self.race.stop()
        del self.race
