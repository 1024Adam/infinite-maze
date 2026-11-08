"""Unit tests for ``training.overlay`` (no window, no real maze generation)."""

from unittest.mock import patch

import pygame
import pytest

from infinite_maze.core.simulation import StepResult
from infinite_maze.entities.maze import Line
from tests.fixtures.golden_harness import KeyState
from tests.fixtures.training_helpers import RIGHT, horizontal, make_sim, vertical
from training.overlay import Overlay, ReplayPlayer, make_tracked_simulation
from training.reward import RewardConfig, RewardFunction


def step_result(moved_x=1, terminated=False):
    """A StepResult with the given horizontal displacement."""
    return StepResult(0, (moved_x, 0), terminated, False)


class TestOverlayState:
    def test_record_step_accumulates_the_return(self):
        overlay = Overlay(RewardFunction(RewardConfig(step_penalty=0.1)))

        overlay.record_step(RIGHT, step_result(1))
        overlay.record_step(RIGHT, step_result(2))

        assert overlay.last_intent == RIGHT
        assert overlay.last_breakdown.progress == 2.0
        assert overlay.cumulative == pytest.approx((1 - 0.1) + (2 - 0.1))

    def test_reset_episode_clears_history(self):
        overlay = Overlay()
        overlay.record_step(RIGHT, step_result())
        overlay.reset_episode()

        assert overlay.cumulative == 0.0
        assert overlay.last_intent is None and overlay.last_breakdown is None

    def test_panel_shows_seed_tick_pace_intent_reward_and_return(self):
        sim = make_sim(seed=42, start_pace=2)
        overlay = Overlay(RewardFunction(RewardConfig(step_penalty=0.01)))
        overlay.attach(sim)
        overlay.record_step(RIGHT, sim.step(RIGHT))

        text = "\n".join(overlay.panel_lines())

        assert "seed 42" in text
        assert "tick 1" in text and "pace 2" in text
        assert "intent (1, 0)" in text
        assert "reward +0.99" in text
        assert "progress +1.00" in text and "step -0.01" in text
        assert "return +0.99" in text

    def test_panel_before_any_step(self):
        overlay = Overlay()
        overlay.attach(make_sim(seed=1))

        text = "\n".join(overlay.panel_lines())

        assert "intent -" in text and "reward -" in text

    def test_attach_installs_the_game_hook(self):
        sim = make_sim()
        overlay = Overlay()

        overlay.attach(sim)

        assert sim.game.overlay is overlay


class TestToggle:
    def press(self, overlay, down):
        keys = KeyState([pygame.K_F3] if down else [])
        with patch("pygame.key.get_pressed", return_value=keys):
            overlay._poll_toggle()

    def test_f3_toggles_on_the_key_down_edge_only(self):
        overlay = Overlay()

        self.press(overlay, True)
        assert not overlay.enabled
        self.press(overlay, True)  # still held: no second toggle
        assert not overlay.enabled
        self.press(overlay, False)
        self.press(overlay, True)
        assert overlay.enabled

    def test_unsupported_key_state_is_treated_as_not_pressed(self):
        overlay = Overlay()

        with patch("pygame.key.get_pressed", return_value={}):
            overlay._poll_toggle()

        assert overlay.enabled


class TestDrawing:
    def real_surface(self):
        return pygame.Surface((640, 480))

    def test_draws_window_rays_and_panel_on_a_real_surface(self):
        sim = make_sim([horizontal(163, 160, 192), vertical(230, 150, 250)])
        overlay = Overlay(show_hitbox=True)
        overlay.attach(sim)
        surface = self.real_surface()
        surface.fill((255, 255, 255))

        with patch("pygame.key.get_pressed", return_value=KeyState()):
            overlay(surface)

        assert pygame.image.tostring(surface, "RGB") != bytes([255]) * (640 * 480 * 3)

    def test_disabled_overlay_draws_nothing(self):
        sim = make_sim([horizontal(163, 160, 192)])
        overlay = Overlay(enabled=False)
        overlay.attach(sim)
        surface = self.real_surface()
        surface.fill((255, 255, 255))
        before = pygame.image.tostring(surface, "RGB")

        with patch("pygame.key.get_pressed", return_value=KeyState()):
            overlay(surface)

        assert pygame.image.tostring(surface, "RGB") == before

    def test_unattached_overlay_does_nothing(self):
        surface = self.real_surface()
        before = pygame.image.tostring(surface, "RGB")

        with patch("pygame.key.get_pressed", return_value=KeyState()):
            Overlay()(surface)

        assert pygame.image.tostring(surface, "RGB") == before

    def test_drawing_does_not_change_the_simulation(self):
        sim = make_sim([horizontal(163, 160, 192)], seed=3)
        overlay = Overlay()
        overlay.attach(sim)
        before = (sim.getState(), [(l.getStart(), l.getEnd()) for l in sim.lines])

        with patch("pygame.key.get_pressed", return_value=KeyState()):
            overlay(self.real_surface())

        after = (sim.getState(), [(l.getStart(), l.getEnd()) for l in sim.lines])
        assert after == before


class TestTrackedSimulation:
    def test_reports_steps_and_resets_to_the_overlay(self):
        overlay = Overlay()
        tracked = make_tracked_simulation(overlay)

        with patch.object(Line, "generateMaze", return_value=[]):
            sim = tracked(seed=1, max_steps=None)
            assert overlay.sim is sim
            assert sim.game.overlay is overlay

            sim.step(RIGHT)
            assert overlay.last_intent == RIGHT
            assert overlay.cumulative != 0.0

            sim.reset()
            assert overlay.cumulative == 0.0


class TestReplayPlayer:
    REPLAY = {
        "version": 1,
        "seed": 1,
        "start_pace": 0,
        "maze": {"rows": 15, "cols": 20, "cell": 22},
        "intents": [[1, 0, 12], [0, 1, 6], [1, 0, 4]],
    }

    def player(self):
        with patch.object(Line, "generateMaze", return_value=[vertical(500, 0, 400)]):
            return ReplayPlayer(self.REPLAY, Overlay())

    def test_steps_forward_until_the_replay_ends(self):
        replay_player = self.player()

        steps = 0
        while replay_player.step_forward():
            steps += 1

        assert steps == 22
        assert replay_player.position == 22
        assert not replay_player.step_forward()

    def test_seek_back_matches_a_fresh_run_to_the_same_step(self):
        replay_player = self.player()
        replay_player.seek(20)
        replay_player.seek(9)

        reference = self.player()
        reference.seek(9)

        assert replay_player.position == 9
        assert replay_player.sim.getState() == reference.sim.getState()
        assert replay_player.overlay.cumulative == pytest.approx(
            reference.overlay.cumulative
        )
        assert replay_player.overlay.last_intent == reference.overlay.last_intent

    def test_seek_is_clamped_to_the_replay(self):
        replay_player = self.player()

        replay_player.seek(10_000)
        assert replay_player.position == 22
        replay_player.seek(-5)
        assert replay_player.position == 0

    def test_rewind_restores_the_original_maze_and_the_generator(self):
        replay_player = self.player()
        generator = Line.__dict__["generateMaze"]
        replay_player.seek(5)

        replay_player.seek(2)

        assert Line.__dict__["generateMaze"] is generator
        assert [(l.getStart(), l.getEnd()) for l in replay_player.sim.lines] == [
            ((500, 0), (500, 400))
        ]

    def test_invalid_replay_is_rejected(self):
        bad = dict(self.REPLAY, version=99)

        with pytest.raises(Exception, match="version"):
            ReplayPlayer(bad, Overlay())
