from enum import Flag, auto
from time import time
from typing import Dict, Set

import matplotlib
import numpy as np
import pystk
from matplotlib import pyplot as plt


class VT(Flag):
    """Visualization type"""

    IMAGE = auto()
    DEPTH = auto()
    INSTANCE = auto()
    SEMANTIC = auto()

    @classmethod
    def all(cls):
        return cls.IMAGE | cls.DEPTH | cls.INSTANCE | cls.SEMANTIC

    @classmethod
    def default(cls):
        return cls.IMAGE


CLASS_COLOR = (
    np.array(
        [
            0xFFFFFF,  # None
            0x4E9A06,  # Kart
            0x2E3436,  # Track
            0xEEEEEC,  # Background
            0x204A87,  # Pickup
            0xA40000,  # Bomb
            0xCE5C00,  # Object
            0x5C3566,  # Projectile
        ],
        dtype=">u4",
    )
    .view(np.uint8)
    .reshape((-1, 4))[:, 1:]
)

INSTANCE_COLOR = np.array(
    [
        0xFCE94F,
        0xEDD400,
        0xC4A000,
        0xFCAF3E,
        0xF57900,
        0xCE5C00,
        0xE9B96E,
        0xC17D11,
        0x8F5902,
        0x8AE234,
        0x73D216,
        0x4E9A06,
        0x729FCF,
        0x3465A4,
        0x204A87,
        0xAD7FA8,
        0x75507B,
        0x5C3566,
        0xEF2929,
        0xCC0000,
        0xA40000,
        0xEEEEEC,
        0xD3D7CF,
        0xBABDB6,
        0x888A85,
        0x555753,
        0x2E3436,
    ],
    dtype=">u4",
)
np.random.shuffle(INSTANCE_COLOR)
INSTANCE_COLOR = INSTANCE_COLOR.view(np.uint8).reshape((-1, 4))[:, 1:]

RENDER_KWARGS = {
    VT.SEMANTIC: dict(
        cmap=matplotlib.colors.ListedColormap(CLASS_COLOR / 255.0, N=32),
        vmin=0,
        vmax=31,
    ),
    VT.INSTANCE: dict(
        cmap=matplotlib.colors.ListedColormap(INSTANCE_COLOR / 255.0, N=1 << 16),
        vmin=0,
        vmax=(1 << 16) - 1,
    ),
}


def _c(i, m):
    return m[i % len(m)]


def semantic_seg(instance, colorize: bool = True):
    L = (np.array(instance) >> 24) & 0xFF
    if colorize:
        return _c(L, CLASS_COLOR)
    return L


def instance_seg(instance, colorize: bool = True):
    L = np.array(instance) & 0xFFFFFF
    if colorize:
        return _c(L, INSTANCE_COLOR)
    return L


class BaseUI:
    visualization_type: VT
    current_action: pystk.Action
    visible: bool
    pause: bool

    def __init__(self, visualization_type: VT):
        self.visualization_type = visualization_type
        self.current_action = pystk.Action()
        self.visible = False
        self.pause = False

    @staticmethod
    def _format_data(
        render_data: pystk.RenderData, colorize: bool = True
    ) -> Dict[VT, np.array]:
        r = dict()
        r[VT.IMAGE] = render_data.image
        r[VT.DEPTH] = render_data.depth
        r[VT.SEMANTIC] = semantic_seg(render_data.instance, colorize=colorize)
        r[VT.INSTANCE] = instance_seg(render_data.instance, colorize=colorize)
        return r

    def _update_action(self, key_state: Set[str]):
        self.current_action.acceleration = int("w" in self._ks or "up" in self._ks)
        self.current_action.brake = "s" in self._ks or "down" in self._ks
        self.current_action.steer = int("d" in self._ks or "right" in self._ks) - int(
            "a" in self._ks or "left" in self._ks
        )
        self.current_action.fire = " " in self._ks
        self.current_action.drift = "m" in self._ks
        self.current_action.nitro = "n" in self._ks
        self.current_action.rescue = "r" in self._ks
        if "p" in self._ks:
            self.pause = not self.pause
        # TODO: Complete

    def show(self, render_data: pystk.RenderData):
        raise NotImplementedError

    def sleep(self, s: float):
        raise NotImplementedError


class NoUI(BaseUI):
    def show(self, render_data: pystk.RenderData):
        pass

    def sleep(self, s: float):
        pass


class MplUI(BaseUI):
    _ax: Dict[VT, plt.Axes]
    _fig: plt.Figure
    _ks: Set[str]

    def __init__(self, visualization_type: VT = VT.default(), hide_menu=True):
        super().__init__(visualization_type)

        n_vis = sum([t in visualization_type for t in VT])

        self._fig = plt.figure()
        self._ax = {}
        k = 1
        nx = int(np.ceil(np.sqrt(n_vis)))
        ny = int(np.ceil(n_vis / nx))
        for t in VT:
            if t in visualization_type:
                self._ax[t] = self._fig.add_subplot(nx, ny, k)
                self._ax[t].axis("off")
                k += 1
        self._fig.tight_layout(pad=0)
        self._fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # if hide_menu:
        #     self._fig.canvas.toolbar.pack_forget()

        # Add the keyboard handling
        self._ks = set()
        self._fig.canvas.mpl_connect("key_press_event", self._key_press)
        self._fig.canvas.mpl_connect("key_release_event", self._key_release)
        self._fig.canvas.mpl_connect(
            "figure_enter_event", lambda *a, **ka: self._ks.clear()
        )
        self._fig.canvas.mpl_connect(
            "figure_enter_event", lambda *a, **ka: self._ks.clear()
        )
        self._fig.canvas.mpl_connect("close_event", self._close)
        # disable the default keys
        try:
            self._fig.canvas.mpl_disconnect(
                self._fig.canvas.manager.key_press_handler_id
            )
        except Exception as e:
            print(f"Exception {e} while trying to disable default keys")
            pass
        self.visible = True

    def _set_action(self):
        super()._update_action(self._ks)

    def _key_press(self, e):
        self._ks.add(e.key)
        self._set_action()
        return True

    def _key_release(self, e):
        if e.key in self._ks:
            self._ks.remove(e.key)
        self._set_action()
        return True

    def show(self, render_data: pystk.RenderData):
        data = self._format_data(render_data, colorize=False)
        for t, a in self._ax.items():
            d = data[t]
            if hasattr(a, "_im"):
                a._im.set_data(d)
            elif t in RENDER_KWARGS:
                a._im = a.imshow(d, interpolation="nearest", **RENDER_KWARGS[t])
            else:
                a._im = a.imshow(d, interpolation="nearest")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        plt.pause(1e-8)

    def _close(self, e):
        self.visible = False

    def sleep(self, s: float):
        plt.pause(s)


def action_dict(action):
    return {
        k: getattr(action, k)
        for k in ["acceleration", "brake", "steer", "fire", "drift"]
    }


if __name__ == "__main__":
    gui_env = ["Qt5Agg", "MacOSX", "Qt5Agg", "TKAgg", "GTKAgg", "Qt4Agg", "WXAgg"]
    found_backend = False
    for gui in gui_env:
        try:
            matplotlib.use(gui, force=True)
            import matplotlib.pyplot as plt

            plt.figure()
            plt.close()
            found_backend = True

            break
        except (ImportError, ValueError):
            continue

    UI = MplUI
    if not found_backend:
        print("no backend available")
        exit(1)

    matplotlib.rcParams["toolbar"] = "None"

    soccer_tracks = {"soccer_field", "icy_soccer_field"}
    arena_tracks = {"battleisland", "stadium"}

    team = 0  # 0 or 1
    kart = ""
    num_player = 1
    visualization = ["IMAGE", "DEPTH", "INSTANCE", "SEMANTIC"]

    save_dir = None
    verbose = False

    # graphic config
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 800
    config.screen_height = 600
    pystk.init(config)

    # race config
    config = pystk.RaceConfig()
    config.num_kart = 2

    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    # config.players[0].kart = args.kart
    config.players[0].team = team

    for i in range(1, num_player):
        config.players.append(
            pystk.PlayerConfig(
                kart, pystk.PlayerConfig.Controller.AI_CONTROL, (team + 1) % 2
            )
        )

    race = pystk.Race(config)
    race.start()
    race.step()

    uis = [UI([VT[x] for x in visualization]) for _ in range(num_player)]

    state = pystk.WorldState()
    state.update()
    t0 = time()
    n = 0

    while all(ui.visible for ui in uis):
        if not all(ui.pause for ui in uis):
            race.step(uis[0].current_action)
            state.update()
            if verbose and config.mode == config.RaceMode.SOCCER:
                print("Score ", state.soccer.score)
                print("      ", state.soccer.ball.location)
            elif verbose:
                print(
                    "Dist  ",
                    state.players[0].kart.overall_distance,
                    state.players[0].kart.distance_down_track,
                    state.players[0].kart.finished_laps,
                )

        for ui, d in zip(uis, race.render_data):
            ui.show(d)

        # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time() - t0)
        if delta_d > 0:
            ui.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
