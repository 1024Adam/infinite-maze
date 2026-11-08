"""Tests for ``training.env.MazeEnv`` (skipped without the ``ml-env`` group)."""

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

from infinite_maze.core.simulation import Simulation  # noqa: E402
from infinite_maze.utils.config import config as game_config  # noqa: E402
from training.env import MazeEnv  # noqa: E402
from training.evaluate import evaluate  # noqa: E402
from training.observation import observation_size  # noqa: E402
from training.policies import Policy  # noqa: E402
from training.reward import RewardConfig  # noqa: E402

MAX_STEPS = 120


def make_env(seeds=(3,), **kwargs):
    kwargs.setdefault("max_steps", MAX_STEPS)
    return MazeEnv(seeds, **kwargs)


class ScriptedPolicy(Policy):
    """Plays a fixed action sequence (cycled), for comparing env and harness."""

    name = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def reset(self, seed):
        self.index = 0

    def act(self, obs):
        action = self.actions[self.index % len(self.actions)]
        self.index += 1
        return action


@pytest.mark.unit
class TestContract:
    def test_passes_gymnasium_check_env(self):
        check_env(make_env(), skip_render_check=True)

    def test_spaces(self):
        env = make_env()

        assert env.action_space.n == len(game_config.MOVEMENT_CONSTANTS)
        assert env.observation_space.shape == (observation_size(),)

    def test_info_reports_the_maze_seed(self):
        _, info = make_env(seeds=[5]).reset(seed=0)

        assert info["maze_seed"] == 5

    def test_options_force_a_maze_seed(self):
        env = make_env(seeds=[5, 6])

        _, info = env.reset(seed=0, options={"maze_seed": 6})

        assert info["maze_seed"] == 6

    def test_invalid_reward_config_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            make_env(reward_config=RewardConfig(death_penalty=0.01))

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError):
            make_env().step(1)


@pytest.mark.unit
class TestDeterminism:
    def test_same_seeds_and_actions_give_identical_trajectories(self):
        actions = [1, 1, 3, 1, 4, 1, 2, 0] * 4

        def rollout():
            env = make_env(seeds=[1, 2, 3])
            obs, info = env.reset(seed=7)
            trace = [(obs.copy(), 0.0, info["maze_seed"])]
            for action in actions:
                obs, reward, terminated, truncated, _ = env.step(action)
                trace.append((obs.copy(), reward, terminated))
                if terminated or truncated:
                    break
            return trace

        first, second = rollout(), rollout()

        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert np.array_equal(a[0], b[0])
            assert a[1:] == b[1:]


@pytest.mark.unit
class TestFrameSkip:
    def test_frame_skip_matches_manual_ticks(self):
        env = make_env(seeds=[3], frame_skip=4)
        env.reset(seed=0)
        env.step(1)
        env.step(3)

        sim = Simulation(seed=3, max_steps=MAX_STEPS)
        for intent in [(1, 0)] * 4 + [(0, -1)] * 4:
            sim.step(intent)

        assert env.simulation.getState() == sim.getState()

    def test_episode_ends_mid_repeat_at_max_steps(self):
        env = make_env(seeds=[3], frame_skip=7, max_steps=10)
        env.reset(seed=0)

        env.step(0)
        _, _, terminated, truncated, info = env.step(0)

        assert truncated and not terminated
        assert info["ticks"] == 10


@pytest.mark.unit
class TestRewardMatchesHarness:
    def test_episode_reward_sum_equals_evaluate_return(self):
        actions = [1, 1, 3, 1, 4, 2, 1, 0, 1, 1]
        frame_skip = 3
        env = make_env(seeds=[4], frame_skip=frame_skip)
        env.reset(seed=0)
        total = 0.0
        index = 0
        while True:
            _, reward, terminated, truncated, _ = env.step(
                actions[index % len(actions)]
            )
            index += 1
            total += reward
            if terminated or truncated:
                break

        report = evaluate(
            ScriptedPolicy(actions),
            [4],
            max_steps=MAX_STEPS,
            frame_skip=frame_skip,
        )

        assert report.episodes[0].total_return == pytest.approx(total)


@pytest.mark.unit
class TestRandomAgent:
    def test_random_agent_always_ends_and_never_raises(self):
        env = make_env(seeds=list(range(20)), max_steps=60)
        env.action_space.seed(0)
        for episode in range(20):
            env.reset(seed=episode)
            for _ in range(60):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            else:
                pytest.fail("episode did not end within max_steps")
