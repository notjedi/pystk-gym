import functools
from abc import abstractmethod
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import numpy as np
import pystk
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..common.actions import ActionType, MultiDiscreteAction
from ..common.graphics import EnvViewer, GraphicConfig
from ..common.info import Info
from ..common.kart import Kart
from ..common.race import ObsType, Race, RaceConfig

# https://github.com/python/typing/issues/59
C = TypeVar("C", bound="Comparable")
AgentId = TypeVar("AgentId", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, other: C) -> bool:
        pass


class RaceEnv(ParallelEnv):
    metadata = {
        "render.modes": ["agent", "human", "rgb_array"],
    }

    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        reward_func: Callable,
        max_step_cnt: int = 1000,
        return_info: bool = True,
        render_mode: Literal["agent", "human", "rgb_array"] = "rgb_array",
    ):
        self.action_class = MultiDiscreteAction()
        self.graphic_config = graphic_config
        self.max_step_cnt = max_step_cnt
        self.reward_func = reward_func
        self.render_mode = render_mode
        self.steps = 0

        self.graphics = graphic_config.get_pystk_config()
        pystk.init(self.graphics)
        self.race = Race(race_config.build())
        self.observation_shape = (
            self.graphics.screen_height,
            self.graphics.screen_width,
            3,
        )
        self._make_karts(return_info)
        self.nitro_locs = self.race.get_nitro_locs()

        self.env_viewer: Optional[EnvViewer] = None
        if render_mode in ("human", "agent"):
            assert race_config.num_karts_controlled == 1 or race_config.num_karts == 1
            self.env_viewer = EnvViewer(
                self.graphic_config, human_controlled=render_mode == "human"
            )

        self.possible_agents = [kart.id for kart in self.get_controlled_karts()]
        self.agents = copy(self.possible_agents)

    def _make_karts(self, return_info: bool):
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
                return_info=return_info,
            )
            for kart in self.race.get_controlled_karts()
        ]

    def _to_stk_action(
        self, actions: Dict[AgentId, ActionType]
    ) -> Dict[AgentId, pystk.Action]:
        return {
            agent_id: self.action_class.get_pystk_action(action)
            for agent_id, action in actions.items()
        }

    def _update_info_dict_with_race_info(self, infos: Dict[AgentId, Dict[Info, Any]]):
        all_kart_rankings = self.race.get_all_kart_rankings()
        for info, kart in zip(infos.values(), self.get_controlled_karts()):
            info[Info.RANK] = all_kart_rankings[kart.id]
            kart_loc = np.array(info[Info.LOCATION])
            info[Info.NITRO] = any(
                np.sqrt(np.sum(np.square(kart_loc - nitro_loc))) <= 2
                for nitro_loc in self.nitro_locs
            )

    def _get_reward(
        self, actions: Dict[AgentId, ActionType], infos: Dict[AgentId, dict]
    ) -> Dict[AgentId, float]:
        return {
            agent_id: self.reward_func(
                self.action_class.get_pystk_action(actions[agent_id]), infos[agent_id]
            )
            for agent_id in actions.keys()
        }

    def _terminal(self, infos: Dict[AgentId, dict]) -> Dict[AgentId, bool]:
        step_limit_reached = self.steps > self.max_step_cnt
        return {
            agent_id: info[Info.OUT_OF_TRACK_COUNT] > 50
            or info[Info.BACKWARD_COUNT] > 50
            or info[Info.NO_MOVEMENT_COUNT] > 50
            or info[Info.DONE]
            or step_limit_reached
            for agent_id, info in infos.items()
        }

    @functools.lru_cache(maxsize=None)
    def get_controlled_karts(self) -> List[Kart]:
        return self.controlled_karts

    def is_done(self) -> List[bool]:
        return [
            self.steps > self.max_step_cnt or kart.is_done()
            for kart in self.get_controlled_karts()
        ]

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

    def step(self, actions: Dict[AgentId, ActionType]) -> Tuple[
        Dict[AgentId, ObsType],  # observation dictionary
        Dict[AgentId, float],  # reward dictionary
        Dict[AgentId, bool],  # terminated dictionary
        Dict[AgentId, bool],  # truncated dictionary
        Dict[AgentId, Dict[Info, Any]],  # info dictionary
    ]:
        self.steps += 1
        # TODO: take multiple steps? if so, i have to render intermediate steps
        if self.render_mode == "human":
            actions = {
                self.get_controlled_karts()[0].id: self.env_viewer.current_action
            }
        else:
            actions = self._to_stk_action(actions)
            actions = dict(sorted(actions.items(), key=lambda x: x[0]))

        obs = dict(zip(actions.keys(), self.race.step(list(actions.values()))))
        infos = {kart.id: kart.step() for kart in self.get_controlled_karts()}
        self._update_info_dict_with_race_info(infos)
        rewards = self._get_reward(actions, infos)
        terminals = self._terminal(infos)
        truncated = {kart.id: False for kart in self.get_controlled_karts()}
        self.agents = [
            kart.id
            for kart in self.get_controlled_karts()
            if not (terminals[kart.id] or truncated[kart.id])
        ]
        return obs, rewards, terminals, truncated, infos

    def render(
        self, mode: Literal["agent", "human", "rgb_array"] = "rgb_array"
    ) -> Optional[ObsType]:
        if mode == "rgb_array":
            return self.race.observe()
        if mode == "human":
            obs = self.race.observe()
            self.env_viewer.display(obs[0])
        elif mode == "agent":
            obs = self.race.observe_all()
            self.env_viewer.display(obs[0])
        return None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[AgentId, ObsType], Dict[AgentId, Dict[Info, Any]]]:
        self.steps = 0
        reset_obs = self.race.reset()
        for kart in self.get_controlled_karts():
            kart.reset()
        obs = {
            kart.id: obs for kart, obs in zip(self.get_controlled_karts(), reset_obs)
        }
        info = {kart.id: {} for kart in self.get_controlled_karts()}
        self.agents = copy(self.possible_agents)
        return obs, info

    def close(self):
        self.race.close()
        if self.env_viewer is not None:
            self.env_viewer.close()
        pystk.clean()
