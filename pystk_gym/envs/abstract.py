import gym


class AbstractEnv(gym.Env):
    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def default_config(cls) -> dict:
        return {}
