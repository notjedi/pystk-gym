from __future__ import annotations

import queue
import threading
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
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


class PyGameWrapper:
    def __init__(self, graphic_config: GraphicConfig):
        self.screen_width = graphic_config.width
        self.screen_height = graphic_config.height
        self.current_action = pystk.Action()
        self.display_hertz = 60

        pygame.init()
        pygame.display.set_caption("TuxKart")

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()

    def handle_events(self, human_controlled: bool):
        if not human_controlled:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                self.close()

            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                is_key_down = float(event.type == pygame.KEYDOWN)
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.current_action.acceleration = is_key_down
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.current_action.brake = is_key_down
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.current_action.steer = 1.0 if is_key_down else 0.0
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.current_action.steer = -1.0 if is_key_down else 0.0
                elif event.key == pygame.K_SPACE:
                    self.current_action.fire = is_key_down
                elif event.key == pygame.K_m:
                    self.current_action.drift = is_key_down
                elif event.key == pygame.K_n:
                    self.current_action.nitro = is_key_down
                elif event.key == pygame.K_r:
                    self.current_action.rescue = is_key_down

    def display(
        self, render_data: npt.NDArray[np.uint8], human_controlled: bool
    ) -> pystk.Action:
        self.handle_events(human_controlled)
        pygame.surfarray.blit_array(self.screen, render_data.swapaxes(0, 1))
        pygame.display.flip()
        self.clock.tick(self.display_hertz)
        # print(f"FPS:= {self.clock.get_fps()}")
        return self.current_action

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


def worker_thread(
    graphic_config: GraphicConfig,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    terminate_event: threading.Event,
    human_controlled: bool,
):
    pygame_wrapper = PyGameWrapper(graphic_config)
    while not terminate_event.is_set():
        try:
            render_data = input_queue.get(timeout=1)
            if pygame_wrapper.screen is not None:
                current_action = pygame_wrapper.display(render_data, human_controlled)
                output_queue.put(current_action)
            else:
                output_queue.put(None)
                pygame_wrapper.close()
                return
        except queue.Empty:
            pass
    pygame_wrapper.close()


class EnvViewer:
    def __init__(self, graphic_config: GraphicConfig, human_controlled=False):
        self.human_controlled = human_controlled
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.current_action = pystk.Action()
        self.terminate_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=worker_thread,
            args=(
                graphic_config,
                self.input_queue,
                self.output_queue,
                self.terminate_event,
                human_controlled,
            ),
        )
        self.worker_thread.start()

    def display(self, render_data: npt.NDArray[np.uint8]) -> Optional[pystk.Action]:
        self.input_queue.put(render_data)
        self.current_action = self.output_queue.get()
        if self.human_controlled and self.current_action is not None:
            return self.current_action
        return None

    def close(self):
        self.terminate_event.set()
        self.worker_thread.join()
