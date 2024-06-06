import random
import argparse

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from pettingzoo.test import parallel_api_test

from pystk_gym import RaceEnv
from pystk_gym.common.graphics import GraphicConfig, GraphicQuality
from pystk_gym.common.info import Info
from pystk_gym.common.race import RaceConfig
from pystk_gym.common.reward import get_reward_fn


def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict[AgentID, ObsType],
    agent: AgentID,
) -> ActionType:
    return env.action_space(agent).sample()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pystk RaceEnv")
    parser.add_argument(
        "-m", "--mode", choices=["agent", "human", "rgb_array"], default="rgb_array"
    )
    args = parser.parse_args()

    reward_fn = get_reward_fn()
    track = random.choice(RaceConfig.TRACKS)
    print(f"Track: {track}")
    race_config = RaceConfig(
        track=track, reverse=True, num_karts=1, num_karts_controlled=1, step_size=0.03
    )
    env = RaceEnv(
        GraphicConfig(960, 540, GraphicQuality.HD),
        race_config,
        reward_fn,
        render_mode=args.mode,
    )

    env.reset(seed=0, options={"options": 1})
    obs, infos = env.reset()

    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    while True:
        actions = {
            agent: sample_action(env, obs, agent)
            for agent in env.agents
            if (
                (agent in terminated and not terminated[agent])
                or (agent in truncated and not truncated[agent])
            )
        }
        observations, rewards, terminals, truncated, infos = env.step(actions)
        env.render(args.mode)
        if len(env.agents) == 0:
            break

    env.close()
