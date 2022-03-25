from __future__ import annotations

from enum import Enum

import pystk


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

    def _get_graphic_config(self) -> pystk.GraphicConfig:
        """Internal method to get a pystk.GraphicConfig object."""
        config = self.graphic_quality.get_obj()
        config.screen_width = self.width
        config.screen_height = self.height
        return config

    @staticmethod
    def default_config() -> pystk.GraphicConfig:
        """Default graphic config."""
        return GraphicConfig(600, 400, GraphicQuality.HD)._get_graphic_config()

    @classmethod
    def get_graphic_config(
        cls, width: int, height: int, graphic_type: GraphicQuality
    ) -> pystk.GraphicConfig:
        """Get pystk.GraphicConfig object using the parameters."""
        config = cls(width, height, graphic_type)
        return config._get_graphic_config()
