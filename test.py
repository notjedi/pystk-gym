import logging
import threading


def pystk_init():
    import pystk

    graphic_config = pystk.GraphicsConfig.hd()
    graphic_config.screen_width = 800
    graphic_config.screen_height = 600
    pystk.init(graphic_config)
    logging.info("pystk init successful")


def pygame_init():
    import pygame

    pygame.init()
    pygame.display.set_caption("TuxKart")
    pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.OPENGL)
    logging.info("pygame init successful")


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    pystk_thread = threading.Thread(target=pystk_init)
    pygame_thread = threading.Thread(target=pygame_init)

    pystk_thread.start()
    pygame_thread.start()

    pystk_thread.join()
    pygame_thread.join()
