from typing import Callable, Dict, List, Tuple, Type, Union, Optional

import gym
import numpy as np
import pystk
from gym import spaces

from ..common.actions import ActionType
from ..common.graphics import GraphicConfig, EnvViewer
from ..common.kart import Kart
from ..common.race import Race, RaceConfig


class AbstractEnv(gym.Env):

    # TODO: TESTS check env or add tests with `from stable_baselines3.common.env_checker import
    # check_env`
    action_aliases: Dict[str, Type[ActionType]] = {}
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        action_config: Dict,
        reward_func: Callable,
        max_step_cnt: int,
    ):
        self.graphic_config = graphic_config
        self.reward_func = reward_func
        self.max_step_cnt = max_step_cnt

        self._init_vars()
        self.viewer = None
        self.configure(graphic_config, race_config, action_config)
        self.define_spaces()

    def _init_vars(self):
        self.done = False
        self.steps = 0

    def _obs_space_from_graphics(self) -> spaces.Space:
        return spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.uint8),
            high=np.full(self.observation_shape, 255, dtype=np.uint8),
            dtype=np.uint8,
        )

    def _action(self, actions: Union[np.ndarray, List, dict]) -> List[pystk.Action]:
        return [self.action_type.get_actions(action) for action in actions]

    def _reward(self, actions, infos) -> List[float]:
        return [self.reward_func(action, info) for action, info in zip(actions, infos)]

    def _is_done(self) -> List[bool]:
        return [
            self.steps > self.max_step_cnt or kart.is_done() for kart in self.get_controlled_karts()
        ]

    def _get_action_from_name(self, action_name: str) -> Type[ActionType]:
        if action_name in self.action_aliases:
            return self.action_aliases[action_name]
        raise ValueError(f"Action {action_name} unknown")

    def _terminal(self, infos: List[dict]) -> List[bool]:
        raise NotImplementedError

    def _reset(self) -> None:
        raise NotImplementedError

    def configure(
        self,
        graphic_config: GraphicConfig,
        race_config: RaceConfig,
        action_config: Dict,
    ) -> None:

        # graphics init
        self.graphics = graphic_config.get_pystk_config()
        pystk.init(self.graphics)

        # race init
        self.race = Race(race_config.get_pystk_config())
        self.observation_shape = (
            self.graphics.screen_height,
            self.graphics.screen_width,
            3,
        )

        # action init
        if isinstance(action_config['class'], str):
            action_type_class = self._get_action_from_name(action_config['class'])
        else:
            action_type_class: Type[ActionType] = action_config['class']
        self.action_type: ActionType = action_type_class(
            action_config['space'], action_config['names']
        )

    def define_spaces(self) -> None:
        self.observation_space = self._obs_space_from_graphics()
        self.action_space = self.action_type.space()

    def get_controlled_karts(self) -> List[Kart]:
        raise NotImplementedError

    def _step(
        self, obs, rewards, terminals, infos
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:
        return obs, rewards, terminals, infos

    def step(
        self, action: Union[np.ndarray, List, Dict]
    ) -> Tuple[np.ndarray, List[float], List[bool], List[dict]]:

        self.steps += 1
        actions = self._action(action)

        obs = self.race.step(actions)
        infos = [kart.step() for kart in self.get_controlled_karts()]
        rewards = self._reward(actions, infos)
        terminals = self._terminal(infos)

        self.done = any(terminals)
        obs, rewards, terminals, infos = self._step(obs, rewards, terminals, infos)
        return obs, rewards, terminals, infos

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        if self.viewer is None:
            self.viewer = EnvViewer(self.graphic_config, human_controlled=mode == 'human')

        # TODO: vectorize this
        obs = self.race.observe()
        self.viewer.display(obs)
        if mode == 'rgb_array':
            return obs

    # TODO: make args compliant with gym.env.reset()
    def reset(self) -> np.ndarray:
        # BUG: using reset here would not restart the race
        self.done = False
        self.define_spaces()
        self._init_vars()

        obs = self.race.reset()
        self._reset()
        for kart in self.get_controlled_karts():
            kart.reset()
        return obs

    def close(self):
        self.done = True
        self.race.close()
        pystk.clean()
