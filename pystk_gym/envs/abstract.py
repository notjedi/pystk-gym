from abc import ABCMeta, abstractmethod
import functools
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import numpy as np
import pystk
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..common.actions import ActionType, MultiDiscreteAction
from ..common.graphics import EnvViewer, GraphicConfig
from ..common.kart import Kart
from ..common.race import Race, RaceConfig, ObsType


C = TypeVar("C", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, other: C) -> bool:
        pass


AgentID = TypeVar("AgentID", bound=Comparable)


class AbstractEnv(ParallelEnv):

    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        reward_func: Callable,
        max_step_cnt: int,
    ):

    def _terminal(self, infos: Dict[AgentID, dict]) -> Dict[AgentID, bool]:
        raise NotImplementedError

    def _reset(self) -> None:
        raise NotImplementedError

    @functools.lru_cache(maxsize=None)
    def get_controlled_karts(self) -> List[Kart]:
        raise NotImplementedError


    def observation_space(self, agent: AgentID) -> spaces.Space:
        raise NotImplementedError

    def action_space(self, agent: AgentID) -> spaces.Space:
        raise NotImplementedError


