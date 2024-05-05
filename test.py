import time
import logging
import threading

import pygame
import numpy as np

# figure out why 2 gl context can't be in the same thread
# the correct flow for the program to work. if pystk cleans up before pygame, then the program crashes
# * pystk init -> pygame init -> pygame cleanup -> pystk cleanup

class PygameWrapper:
    def __init__(self, width, height):
        self.screen = None

        # with self.lock:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pygame Wrapper")

    def display_array(self, array):
        if not self.screen:
            return

        pygame.surfarray.blit_array(self.screen, array)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.screen = None
                pygame.quit()
                return

    def close(self):
        if not self.screen:
            return

        self.screen = None
        pygame.quit()


def pystk_init():
    import pystk

    logging.info("pystk init")
    graphic_config = pystk.GraphicsConfig.hd()
    graphic_config.screen_width = 800
    graphic_config.screen_height = 600
    pystk.init(graphic_config)
    time.sleep(5)


def pygame_init():
    time.sleep(0.5)
    logging.info("pygame init")
    WIDTH, HEIGHT = 640, 480

    pygame_wrapper = PygameWrapper(WIDTH, HEIGHT)

    while True:
        random_array = np.random.randint(0, 255, (WIDTH, HEIGHT, 3)).astype(np.uint8)
        pygame_wrapper.display_array(random_array)
        if not pygame_wrapper.screen:
            break

    pygame_wrapper.close()


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    pystk_thread = threading.Thread(target=pystk_init)
    pygame_thread = threading.Thread(target=pygame_init)

    pystk_thread.start()
    pygame_thread.start()

    # pystk_thread.join()
    pygame_thread.join()
