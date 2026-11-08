"""Integration tests for the ``Game.setOverlay`` hook and the training overlay."""

from unittest.mock import patch

import pygame
import pytest

from infinite_maze.core import engine
from infinite_maze.core.game import Game
from infinite_maze.core.replay import loadReplay
from infinite_maze.core.simulation import Simulation
from infinite_maze.entities.maze import Line
from infinite_maze.entities.player import Player
from tests.fixtures.golden_harness import KeyState
from tests.fixtures.pygame_mocks import MockPygameEvent, PygameMockContext
from tests.fixtures.replay_recipes import REPLAY_DIR
from tests.fixtures.training_helpers import vertical
from training import overlay as overlay_module
from training.overlay import Overlay, run_human_overlay, run_replay_overlay


def rendered_game(mocks):
    """A non-headless Game plus the mocked screen surface."""
    game = Game(headless=False)
    return game, mocks["display"]["surface"]


class TestGameOverlayHook:
    def test_callback_runs_once_per_update_screen_with_the_screen(self):
        with PygameMockContext() as mocks:
            game, surface = rendered_game(mocks)
            calls = []
            game.setOverlay(calls.append)
            player = Player(100, 100)

            for _ in range(3):
                game.updateScreen(player, [])

            assert calls == [surface] * 3

    def test_callback_is_drawn_before_the_frame_is_presented(self):
        with PygameMockContext() as mocks:
            game, _ = rendered_game(mocks)
            order = []
            game.setOverlay(lambda surface: order.append("overlay"))
            mocks["display"]["flip"].side_effect = lambda: order.append("flip")

            game.updateScreen(Player(100, 100), [])

            assert order == ["overlay", "flip"]

    def test_no_overlay_by_default_and_none_clears_it(self):
        with PygameMockContext() as mocks:
            game, _ = rendered_game(mocks)
            calls = []

            game.updateScreen(Player(100, 100), [])
            game.setOverlay(calls.append)
            game.setOverlay(None)
            game.updateScreen(Player(100, 100), [])

            assert game.overlay is None
            assert calls == []

    def test_headless_game_never_calls_the_overlay(self):
        game = Game(headless=True)
        calls = []
        game.setOverlay(calls.append)

        game.updateScreen(Player(100, 100, headless=True), [])

        assert calls == []


class TestOverlayOnTheGameScreen:
    """Draw with no keys held (the generic mock key state would toggle F3)."""

    @pytest.fixture(autouse=True)
    def no_keys_pressed(self):
        with patch("pygame.key.get_pressed", return_value=KeyState()):
            yield

    def test_overlay_draws_through_update_screen_without_raising(self):
        with PygameMockContext(mock_event=False) as mocks:
            game, surface = rendered_game(mocks)
            maze = [vertical(230, 150, 250)]
            with patch.object(Line, "generateMaze", return_value=maze):
                sim = Simulation(seed=1, game=game, player=Player(200, 200))
            overlay = Overlay(show_hitbox=True)
            overlay.attach(sim)
            blits_before = len(surface.blits)

            game.updateScreen(sim.player, sim.getLines())

            assert len(surface.blits) > blits_before
            assert mocks["draw"]["rect"].called and mocks["draw"]["line"].called

    def test_disabled_overlay_adds_nothing_to_the_screen(self):
        with PygameMockContext(mock_event=False) as mocks:
            game, surface = rendered_game(mocks)
            with patch.object(Line, "generateMaze", return_value=[]):
                sim = Simulation(seed=1, game=game, player=Player(200, 200))
            game.updateScreen(sim.player, sim.getLines())
            plain_blits = len(surface.blits)
            rect_calls = mocks["draw"]["rect"].call_count

            Overlay(enabled=False).attach(sim)
            game.updateScreen(sim.player, sim.getLines())

            assert len(surface.blits) - plain_blits == plain_blits
            assert mocks["draw"]["rect"].call_count == rect_calls

    def test_attaching_an_overlay_does_not_change_replay_results(self):
        replay = loadReplay(str(REPLAY_DIR / "fuzz_pace2.json"))
        with PygameMockContext(mock_event=False) as mocks:
            game, _ = rendered_game(mocks)
            sim = Simulation(
                seed=replay["seed"],
                start_pace=replay["start_pace"],
                max_steps=None,
                game=game,
                player=Player(80, 223),
            )
            overlay = Overlay(show_hitbox=True)
            overlay.attach(sim)

            for dx, dy, count in replay["intents"]:
                for _ in range(count):
                    game.updateScreen(sim.player, sim.getLines())
                    overlay.record_step((dx, dy), sim.step((dx, dy)))

            assert sim.getState() == replay["expected"]


class TestReplayRunner:
    def run(self, event_batches):
        """Run the replay overlay loop with scripted events; return positions."""
        positions = []
        instances = []

        class Recording(overlay_module.ReplayPlayer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)

        replay = {
            "version": 1,
            "seed": 1,
            "start_pace": 0,
            "maze": {"rows": 15, "cols": 20, "cell": 22},
            "intents": [[1, 0, 30]],
        }
        batches = iter(event_batches)

        def next_events():
            batch = next(batches)
            positions.append(instances[0].position if instances else 0)
            return batch

        with PygameMockContext(mock_event=False), patch(
            "pygame.event.get", side_effect=next_events
        ), patch("pygame.key.get_pressed", return_value={}), patch.object(
            overlay_module, "ReplayPlayer", Recording
        ), patch.object(
            Game, "cleanup"
        ), patch.object(
            Line, "generateMaze", return_value=[]
        ):
            code = run_replay_overlay(replay)
        return code, positions, instances[0]

    @staticmethod
    def key(code):
        return [MockPygameEvent(pygame.KEYDOWN, key=code)]

    def test_plays_pauses_steps_forward_and_back_then_quits(self):
        batches = [
            [],  # plays: step 1
            [],  # plays: step 2
            self.key(pygame.K_SPACE),  # pause
            self.key(pygame.K_PERIOD),  # +1 -> 3
            self.key(pygame.K_PERIOD),  # +1 -> 4
            self.key(pygame.K_COMMA),  # -1 -> 3
            [],  # still paused
            self.key(pygame.K_ESCAPE),
        ]

        code, _, replay_player = self.run(batches)

        assert code == 0
        assert replay_player.position == 3

    def test_stepping_keys_are_ignored_while_playing(self):
        batches = [
            self.key(pygame.K_PERIOD),
            self.key(pygame.K_COMMA),
            self.key(pygame.K_ESCAPE),
        ]

        _, _, replay_player = self.run(batches)

        # The two playing frames advance one step each; the quit frame does not.
        assert replay_player.position == 2

    def test_closing_the_window_quits(self):
        batches = [[MockPygameEvent(pygame.QUIT)]]

        code, _, _ = self.run(batches)

        assert code == 0


class TestHumanRunner:
    def test_engine_runs_with_a_tracking_simulation_that_is_restored_afterwards(self):
        original = engine.Simulation
        seen = {}

        def fake_maze(seed=None, start_pace=0, record_path=None):
            seen["simulation"] = engine.Simulation
            seen["seed"], seen["start_pace"] = seed, start_pace
            raise SystemExit(0)

        with patch.object(engine, "maze", side_effect=fake_maze):
            assert run_human_overlay(seed=9, start_pace=2) == 0

        assert seen["simulation"] is not original
        assert issubclass(seen["simulation"], original)
        assert (seen["seed"], seen["start_pace"]) == (9, 2)
        assert engine.Simulation is original
