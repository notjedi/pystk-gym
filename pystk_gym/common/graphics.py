from __future__ import annotations

from enum import Enum
from typing import Set

import os
import numpy as np
import pystk
import pygame


class GraphicQuality(Enum):
    """
    Enum class for all possible graphic qualities.
    """

    HD = (1, pystk.GraphicsConfig.hd)
    SD = (2, pystk.GraphicsConfig.sd)
    LD = (3, pystk.GraphicsConfig.ld)
    NONE = (4, pystk.GraphicsConfig.none)

    def __new__(cls, value, obj_ref):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._obj_ref = obj_ref
        return obj

    def get_obj(self) -> pystk.GraphicsConfig:
        return self._obj_ref()


class GraphicConfig:
    def __init__(self, width: int, height: int, graphic_quality: GraphicQuality) -> None:
        """
        :param width: screen width
        :param height: screen height
        :param graphic_quality: graphic quality
        """
        self.width = width
        self.height = height
        self.graphic_quality = graphic_quality

    def get_pystk_config(self) -> pystk.GraphicConfig:
        """Internal method to get a pystk.GraphicConfig object."""
        return self.get_graphic_config(self.width, self.height, self.graphic_quality)

    @staticmethod
    def default_config() -> GraphicConfig:
        """Default graphic config."""
        return GraphicConfig(600, 400, GraphicQuality.HD)

    @staticmethod
    def get_graphic_config(
        width: int, height: int, graphic_quality: GraphicQuality
    ) -> pystk.GraphicConfig:
        """Get pystk.GraphicConfig object using the parameters."""
        config = graphic_quality.get_obj()
        config.screen_width = width
        config.screen_height = height
        return config


# class EnvViewer:
#     def __init__(self, human_controlled: bool = False, id: str = ''):
#         self.human_controlled = human_controlled
#         self.action = pystk.Action()
#         self.visible = True
#         self.id = id
#
#         self.fig = plt.figure(num=id)
#         self.axes = self.fig.add_subplot(1, 1, 1)
#         self.axes.axis('off')
#         self.fig.tight_layout(pad=0)
#
#         if human_controlled:
#             self._key_state = set()
#             self.fig.canvas.mpl_connect(
#                 'figure_enter_event', lambda *a, **ka: self._key_state.clear()
#             )
#             self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
#             self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
#             self.fig.canvas.mpl_connect('close_event', self._close)
#             # disable the default keys
#             self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
#
#     def _update_action(self, key_state: Set[str]):
#         self.action.acceleration = int('w' in key_state or 'up' in key_state)
#         self.action.brake = 's' in key_state or 'down' in key_state
#         self.action.steer = int('d' in key_state or 'right' in key_state) - int(
#             'a' in key_state or 'left' in key_state
#         )
#         self.action.fire = ' ' in key_state
#         self.action.drift = 'm' in key_state
#         self.action.nitro = 'n' in key_state
#         self.action.rescue = 'r' in key_state
#
#     def _on_key_press(self, e):
#         self._key_state.add(e.key)
#         self._update_action(self._key_state)
#         return True
#
#     def _on_key_release(self, e):
#         if e.key == 'escape':
#             self.visible = False
#         else:
#             if e.key in self._key_state:
#                 self._key_state.remove(e.key)
#             self._update_action(self._key_state)
#         return True
#
#     def _close(self, e):
#         self.visible = False
#
#     def display(self, render_data: np.ndarray):
#         if hasattr(self.axes, '_im'):
#             self.axes._im.set_data(render_data)
#         else:
#             # TODO: check if this is needed
#             self.axes.imshow(render_data, interpolation='nearest')
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#
#     def get_action(self) -> pystk.Action:
#         return self.action
#
#     def close(self):
#         # TODO:check if pyplot.close()
#         plt.close()
#         self.visible = False


class EnvViewer:
    def __init__(self, graphic_config, human_controlled=False, id=1):
        self.screen_width = graphic_config.width
        self.screen_height = graphic_config.height
        self.human_controlled = human_controlled
        self.display_hertz = 60
        self.id = id

        pygame.init()
        pygame.display.set_caption("TuxKart")

        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        self.clock = pygame.time.Clock()
        self.action = {}

        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

    def display(self, render_data):
        if self.human_controlled:
            self.handle_events()
        self.screen.blit(render_data, (0, self.screen_width))
        self.clock.tick(self.display_hertz)
        print(f"id:= {self.id}, FPS:= {self.clock.get_fps()}")

    def get_action(self):
        return self.action

    def handle_events(self):
        self.action = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.action['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.action['up'] = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.action['down'] = True
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.action['right'] = True
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.action['left'] = True

                elif event.key == pygame.K_SPACE:
                    self.action['fire'] = True
                elif event.key == pygame.K_m:
                    self.action['drift'] = True
                elif event.key == pygame.K_n:
                    self.action['nitro'] = True
                elif event.key == pygame.K_r:
                    self.action['rescue'] = True
        return self.action
