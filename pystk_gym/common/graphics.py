from __future__ import annotations
from enum import Enum

import pygame
import pystk


class GraphicQuality(Enum):
    """
    Enum class for all possible graphic qualities.
    """

    HD = (1, pystk.GraphicsConfig.hd)
    SD = (2, pystk.GraphicsConfig.sd)
    LD = (3, pystk.GraphicsConfig.ld)
    NONE = (4, pystk.GraphicsConfig.none)

    def get_obj(self) -> pystk.GraphicsConfig:
        return self.value[1]()


class GraphicConfig:
    def __init__(self, width: int, height: int, graphic_quality: GraphicQuality):
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
        width: int,
        height: int,
        graphic_quality: GraphicQuality,
    ) -> pystk.GraphicConfig:
        """Get pystk.GraphicConfig object using the parameters."""
        config = graphic_quality.get_obj()
        config.screen_width = width
        config.screen_height = height
        return config


class EnvViewer:
    def __init__(self, graphic_config, human_controlled=False, id=1):
        self.screen_width = graphic_config.width
        self.screen_height = graphic_config.height
        self.human_controlled = human_controlled
        self.display_hertz = 60
        self.id = id

        pygame.init()
        pygame.display.set_caption("TuxKart")

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.OPENGL
        )
        self.clock = pygame.time.Clock()
        self.action = {}

    def display(self, render_data):
        if self.human_controlled:
            self.handle_events()
        self.screen.blit(render_data, (0, self.screen_width))
        self.clock.tick(self.display_hertz)
        print(f"id:= {self.id}, FPS:= {self.clock.get_fps()}")

    def get_action(self) -> pystk.Action:
        return self.action

    def handle_events(self):
        self.action = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.action["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.action["up"] = True
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.action["down"] = True
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    self.action["right"] = True
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    self.action["left"] = True

                elif event.key == pygame.K_SPACE:
                    self.action["fire"] = True
                elif event.key == pygame.K_m:
                    self.action["drift"] = True
                elif event.key == pygame.K_n:
                    self.action["nitro"] = True
                elif event.key == pygame.K_r:
                    self.action["rescue"] = True
        return self.action

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
