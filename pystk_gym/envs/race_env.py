import functools
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar
from copy import copy

import numpy as np
import pystk
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..common.actions import ActionType, MultiDiscreteAction
from ..common.graphics import EnvViewer, GraphicConfig
from ..common.kart import Kart
from ..common.race import ObsType, Race, RaceConfig

# https://github.com/python/typing/issues/59
C = TypeVar("C", bound="Comparable")
AgentID = TypeVar("AgentID", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, other: C) -> bool:
        pass


class RaceEnv(ParallelEnv):
    # TODO: TESTS check env or add tests with `from stable_baselines3.common.env_checker`
    # TODO: add seed method
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        reward_func: Callable,
        max_step_cnt: int = 1000,
    ):
        self.graphic_config = graphic_config
        self.reward_func = reward_func
        self.max_step_cnt = max_step_cnt
        self._init_vars()

        # graphics init
        self.graphics = graphic_config.get_pystk_config()
        pystk.init(self.graphics)
        # race init
        self.race = Race(race_config.build())
        self.observation_shape = (
            self.graphics.screen_height,
            self.graphics.screen_width,
            3,
        )
        # action init
        self.action_class = MultiDiscreteAction()
        self._make_karts()

        self.observation_spaces = {
            kart.id: spaces.Box(
                low=np.zeros(self.observation_shape, dtype=np.uint8),
                high=np.full(self.observation_shape, 255, dtype=np.uint8),
                dtype=np.uint8,
            )
            for kart in self.get_controlled_karts()
        }
        self.action_spaces = {
            kart.id: self.action_class.space() for kart in self.get_controlled_karts()
        }
        self.possible_agents = [kart.id for kart in self.get_controlled_karts()]
        self.agents = copy(self.possible_agents)

        self._node_idx = 0
        self.reverse = self.race.get_config().reverse

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Box:
        return spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.uint8),
            high=np.full(self.observation_shape, 255, dtype=np.uint8),
            dtype=np.uint8,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> spaces.MultiDiscrete:
        return self.action_class.space()

    @functools.lru_cache(maxsize=None)
    def get_controlled_karts(self) -> List[Kart]:
        return self.controlled_karts

    def _to_stk_action(
        self, actions: Dict[AgentID, ActionType]
    ) -> Dict[AgentID, pystk.Action]:
        return {
            agent_id: self.action_class.get_pystk_action(action)
            for agent_id, action in actions.items()
        }

    def _init_vars(self):
        self.viewers = None
        self.done = False
        self.steps = 0

    def _get_reward(
        self, actions: Dict[AgentID, ActionType], infos: Dict[AgentID, dict]
    ) -> Dict[AgentID, float]:
        return {
            agent_id: self.reward_func(
                self.action_class.get_pystk_action(actions[agent_id]), infos[agent_id]
            )
            for agent_id in actions.keys()
        }

    def _terminal(self, infos: Dict[AgentID, dict]) -> Dict[AgentID, bool]:
        step_limit_reached = self.steps > self.max_step_cnt
        return {
            agent_id: not info["is_inside_track"]
            or info["backward"]
            or info["no_movement"]
            or info["done"]
            or step_limit_reached
            for agent_id, info in infos.items()
        }

    def _make_karts(self):
        is_reverse, path_width, path_lines, path_distance = (
            self.race.get_race_info()["reverse"],
            self.race.get_path_width(),
            self.race.get_path_lines(),
            self.race.get_path_distance(),
        )
        self.controlled_karts = [
            Kart(
                kart,
                is_reverse,
                path_width,
                path_lines,
                path_distance,
            )
            for kart in self.race.get_controlled_karts()
        ]

    def get_actions_from_env_viewer(self) -> Optional[List[pystk.Action]]:
        if self.viewers is not None:
            return [self._to_stk_action(viewer.get_action()) for viewer in self.viewers]
        return None

    def step(
        self, actions: Dict[AgentID, ActionType]
    ) -> Tuple[
        Dict[AgentID, ObsType],  # observation dictionary
        Dict[AgentID, float],  # reward dictionary
        Dict[AgentID, bool],  # terminated dictionary
        Dict[AgentID, bool],  # truncated dictionary
        Dict[AgentID, dict],  # info dictionary
    ]:
        self.steps += 1
        actions = self._to_stk_action(actions)
        actions = {k: v for k, v in sorted(actions.items(), key=lambda x: x[0])}
        # TODO: should i assert len(actions) == num_controlled_karts

        # TODO: take multiple steps? if so, i have to render intermediate steps
        obs = {
            agent_id: obs
            for agent_id, obs in zip(
                actions.keys(), self.race.step(list(actions.values()))
            )
        }
        infos = {kart.id: kart.step() for kart in self.get_controlled_karts()}
        rewards = self._get_reward(actions, infos)
        terminals = self._terminal(infos)
        truncated = {kart.id: False for kart in self.get_controlled_karts()}
        self.agents = [
            kart.id
            for kart in self.get_controlled_karts()
            if not (terminals[kart.id] or truncated[kart.id])
        ]

        # nitro_locs = self.race.get_nitro_locs()
        # all_kart_positions = self.race.get_all_kart_positions()
        # for info, kart in zip(infos, self.get_controlled_karts()):
        #     info["position"] = all_kart_positions[kart.id]
        #     kart_loc = np.array(info["location"])
        #     # TODO: make this more functional programming like
        #     info["nitro"] = any(
        #         np.sqrt(np.sum(kart_loc - nitro_loc)) <= 1 for nitro_loc in nitro_locs
        #     )

        # FIXME: this is always true
        self.done = any(terminals)
        return obs, rewards, terminals, truncated, infos

    def _reset(self):
        self._make_karts()

    def is_done(self) -> List[bool]:
        return [
            self.steps > self.max_step_cnt or kart.is_done()
            for kart in self.get_controlled_karts()
        ]

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        if self.viewers is None:
            self.viewers = [
                EnvViewer(
                    self.graphic_config,
                    human_controlled=mode == "human",
                    id=kart.id,
                )
                for kart in self.get_controlled_karts()
            ]

        obs = self.race.observe()
        for image, viewer in zip(obs, self.viewers):
            viewer.display(image)
        if mode == "rgb_array":
            return obs
        # TODO: return
        return obs

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[AgentID, np.ndarray], Dict[AgentID, dict]]:
        # BUG: using reset here would not restart the race
        self.done = False
        self._init_vars()

        obs = self.race.reset()
        self._reset()
        for kart in self.get_controlled_karts():
            kart.reset()
        self.possible_agents = [kart.id for kart in self.get_controlled_karts()]
        self.agents = copy(self.possible_agents)
        obs = {
            kart.id: obs
            for kart, obs in zip(self.get_controlled_karts(), self.race.observe())
        }
        info = {kart.id: {} for kart in self.get_controlled_karts()}
        return obs, info

    def close(self):
        self.done = True
        self.race.close()
        if self.viewers is not None:
            for viewer in self.viewers:
                viewer.close()
        pystk.clean()
