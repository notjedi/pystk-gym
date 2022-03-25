from __future__ import annotations

from enum import Enum

import pystk


class GraphicType(Enum):

    HD = (1, pystk.GraphicsConfig.hd)
    SD = (2, pystk.GraphicsConfig.sd)
    LD = (3, pystk.GraphicsConfig.ld)
    NONE = (4, pystk.GraphicsConfig.none)

    def __new__(cls, value, obj_ref):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._obj_ref = obj_ref
        return obj

    def get_obj(self):
        return self._obj_ref()


class GraphicConfig:
    def __init__(self, width: int, height: int, graphic_type: GraphicType) -> None:
        self.width = width
        self.height = height
        self.graphic_type = graphic_type

    def _get_graphic_config(self) -> pystk.GraphicConfig:
        config = self.graphic_type.get_obj()
        config.screen_width = self.width
        config.screen_height = self.height
        return config

    @staticmethod
    def default_config() -> pystk.GraphicConfig:
        return GraphicConfig(600, 400, GraphicType.HD)._get_graphic_config()

    @classmethod
    def get_graphic_config(
        cls, width: int, height: int, graphic_type: GraphicType
    ) -> pystk.GraphicConfig:
        config = cls(width, height, graphic_type)
        return config._get_graphic_config()
