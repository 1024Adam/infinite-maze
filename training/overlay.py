"""Debug overlay showing what the agent observes and is rewarded for.

The overlay is drawn on top of the rendered game through ``Game.setOverlay``; the
game package knows nothing about it. Press F3 to toggle it. Run it with:

    python -m training.overlay --replay path.json [--reward-config path]
    python -m training.overlay --seed 1234        # human play with the overlay

Replay controls: SPACE pauses, ``.`` steps one tick forward and ``,`` one tick back
while paused (going back re-runs the replay from its start), ESC quits.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

import pygame

from infinite_maze.core import engine
from infinite_maze.core.game import Game
from infinite_maze.core.replay import decodeIntents, loadReplay, validateReplay
from infinite_maze.core.simulation import Intent, Simulation, StepResult
from infinite_maze.entities.maze import Line
from infinite_maze.entities.player import Player
from infinite_maze.utils.config import config as game_config

from . import config
from .observation import Observation, build_observation, window_origin
from .reward import RewardBreakdown, RewardConfig, RewardFunction, load_reward_config

CELL_COLOR = (255, 140, 0)
WINDOW_COLOR = (0, 120, 255)
HITBOX_COLOR = (255, 0, 0)
RAY_COLORS = ((0, 170, 0), (170, 0, 170), (0, 150, 170), (200, 120, 0))
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (0, 0, 0)


class Overlay:
    """Callable drawn by ``Game.updateScreen`` after the maze, before the flip."""

    def __init__(
        self,
        reward: Optional[RewardFunction] = None,
        show_hitbox: bool = False,
        enabled: bool = True,
    ) -> None:
        self.reward = reward if reward is not None else RewardFunction()
        self.show_hitbox = show_hitbox
        self.enabled = enabled
        self.sim: Optional[Simulation] = None
        self.last_intent: Optional[Intent] = None
        self.last_breakdown: Optional[RewardBreakdown] = None
        self.cumulative = 0.0
        self._toggle_was_down = False
        self._font: Optional[pygame.font.Font] = None

    def attach(self, sim: Simulation) -> None:
        """Observe ``sim`` and draw on its game's screen."""
        self.sim = sim
        sim.game.setOverlay(self)

    def reset_episode(self) -> None:
        """Forget the reward history (a new episode started)."""
        self.last_intent = None
        self.last_breakdown = None
        self.cumulative = 0.0

    def record_step(self, intent: Intent, result: StepResult) -> None:
        """Account for one ``Simulation.step``."""
        self.last_intent = intent
        self.last_breakdown = self.reward(result)
        self.cumulative += self.last_breakdown.total

    def __call__(self, surface: pygame.Surface) -> None:
        self._poll_toggle()
        if not self.enabled or self.sim is None:
            return
        obs = build_observation(self.sim)
        self._draw_window(surface, obs)
        self._draw_rays(surface, obs)
        if self.show_hitbox:
            self._draw_hitbox(surface)
        self._draw_panel(surface)

    def _poll_toggle(self) -> None:
        """Toggle on the F3 key-down edge (works whoever drains the event queue)."""
        try:
            down = bool(pygame.key.get_pressed()[pygame.K_F3])
        except (KeyError, IndexError):
            down = False
        if down and not self._toggle_was_down:
            self.enabled = not self.enabled
        self._toggle_was_down = down

    def _draw_window(self, surface: pygame.Surface, obs: Observation) -> None:
        assert self.sim is not None
        size = config.OBS_GRID_SIZE
        cell = config.OBS_GRID_CELL_PX
        originX, originY = window_origin(self.sim)
        tint = pygame.Surface((size * cell, size * cell), pygame.SRCALPHA)
        for row, cells in enumerate(obs.grid):
            for col, value in enumerate(cells):
                if value:
                    tint.fill(
                        CELL_COLOR + (config.OVERLAY_CELL_ALPHA,),
                        (col * cell, row * cell, cell, cell),
                    )
        surface.blit(tint, (originX, originY))
        pygame.draw.rect(
            surface, WINDOW_COLOR, (originX, originY, size * cell, size * cell), 1
        )

    def _draw_rays(self, surface: pygame.Surface, obs: Observation) -> None:
        assert self.sim is not None
        player = self.sim.player
        left, top = player.getPosition()
        right = left + player.getWidth()
        bottom = top + player.getHeight()
        centerX = (left + right) / 2
        centerY = (top + bottom) / 2
        cap = config.OBS_RAY_CAP_PX
        right_d, left_d, up_d, down_d = (ray * cap for ray in obs.rays)
        ends = (
            ((right, centerY), (right + right_d, centerY)),
            ((left, centerY), (left - left_d, centerY)),
            ((centerX, top), (centerX, top - up_d)),
            ((centerX, bottom), (centerX, bottom + down_d)),
        )
        for color, (start, end) in zip(RAY_COLORS, ends):
            pygame.draw.line(surface, color, start, end, 1)

    def _draw_hitbox(self, surface: pygame.Surface) -> None:
        assert self.sim is not None
        player = self.sim.player
        x, y = player.getPosition()
        pygame.draw.rect(
            surface, HITBOX_COLOR, (x, y, player.getWidth(), player.getHeight()), 1
        )

    def panel_lines(self) -> List[str]:
        """Text shown in the panel (also used by tests)."""
        assert self.sim is not None
        sim = self.sim
        lines = [
            f"seed {sim.getSeed()}",
            f"tick {sim.getTicks()}  pace {sim.game.getPace()}",
            f"intent {self.last_intent if self.last_intent is not None else '-'}",
        ]
        if self.last_breakdown is None:
            lines.append("reward -")
        else:
            b = self.last_breakdown
            lines.append(f"reward {b.total:+.2f}")
            lines.append(
                f" progress {b.progress:+.2f} step {b.step:+.2f} end {b.terminal:+.2f}"
            )
        lines.append(f"return {self.cumulative:+.2f}")
        return lines

    def _draw_panel(self, surface: pygame.Surface) -> None:
        if self._font is None:
            self._font = pygame.font.SysFont("", config.OVERLAY_FONT_SIZE)
        lines = self.panel_lines()
        margin = config.OVERLAY_PANEL_MARGIN_PX
        width = config.OVERLAY_PANEL_WIDTH_PX
        height = len(lines) * config.OVERLAY_LINE_HEIGHT_PX + 2 * margin
        left = surface.get_width() - width - margin
        background = pygame.Surface((width, height), pygame.SRCALPHA)
        background.fill(PANEL_COLOR + (config.OVERLAY_PANEL_ALPHA,))
        surface.blit(background, (left, margin))
        for index, text in enumerate(lines):
            rendered = self._font.render(text, 1, TEXT_COLOR)
            surface.blit(
                rendered,
                (
                    left + margin,
                    margin * 2 + index * config.OVERLAY_LINE_HEIGHT_PX,
                ),
            )


def make_tracked_simulation(overlay: Overlay) -> Type[Simulation]:
    """Simulation subclass that reports every step and reset to ``overlay``.

    Lets the overlay follow a game created inside ``engine.maze()`` without any
    change to the game package.
    """

    class TrackedSimulation(Simulation):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            overlay.attach(self)

        def step(self, intent: Intent) -> StepResult:
            result = super().step(intent)
            overlay.record_step(intent, result)
            return result

        def reset(self, seed: Optional[int] = None) -> None:
            super().reset(seed)
            overlay.reset_episode()

    return TrackedSimulation


class ReplayPlayer:
    """Plays a replay under an overlay with forward/backward stepping."""

    def __init__(
        self,
        replay: Dict[str, Any],
        overlay: Overlay,
        game: Optional[Game] = None,
        player: Optional[Player] = None,
    ) -> None:
        validateReplay(replay)
        self.overlay = overlay
        self.intents: List[Intent] = decodeIntents(replay["intents"])
        self.position = 0
        self.sim = Simulation(
            seed=replay["seed"],
            start_pace=replay["start_pace"],
            max_steps=None,
            game=game,
            player=player,
        )
        # Regenerating the maze is slow, so rewinding reuses a copy of the original.
        self._pristine = [(line.getStart(), line.getEnd()) for line in self.sim.lines]
        overlay.attach(self.sim)

    def step_forward(self) -> bool:
        """Play the next recorded step; False once the replay is finished."""
        if self.position >= len(self.intents):
            return False
        intent = self.intents[self.position]
        self.overlay.record_step(intent, self.sim.step(intent))
        self.position += 1
        return True

    def seek(self, target: int) -> None:
        """Move to ``target`` steps in, re-running from the start to go back."""
        target = max(0, min(target, len(self.intents)))
        if target < self.position:
            self._rewind()
        while self.position < target:
            self.step_forward()

    def _rewind(self) -> None:
        original = Line.__dict__["generateMaze"]
        pristine = self._pristine

        def fresh(*args: Any, **kwargs: Any) -> List[Line]:
            return [Line(start, end) for start, end in pristine]

        setattr(Line, "generateMaze", staticmethod(fresh))
        try:
            self.sim.reset()
        finally:
            setattr(Line, "generateMaze", original)
        self.overlay.reset_episode()
        self.position = 0


def _handle_replay_key(
    key: int, paused: bool, replay_player: ReplayPlayer
) -> Tuple[bool, bool]:
    """Apply a key press; return (toggle_pause, quit)."""
    if key == pygame.K_ESCAPE:
        return False, True
    if key == pygame.K_SPACE:
        return True, False
    if paused and key == pygame.K_PERIOD:
        replay_player.seek(replay_player.position + 1)
    elif paused and key == pygame.K_COMMA:
        replay_player.seek(replay_player.position - 1)
    return False, False


def run_replay_overlay(
    replay: Dict[str, Any],
    reward_config: Optional[RewardConfig] = None,
    show_hitbox: bool = False,
) -> int:
    """Play ``replay`` in a window with the overlay until ESC or window close."""
    game = Game()
    player = Player(game_config.PLAYER_START_X, game_config.PLAYER_START_Y)
    overlay = Overlay(RewardFunction(reward_config), show_hitbox)
    replay_player = ReplayPlayer(replay, overlay, game, player)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                toggle, quit_ = _handle_replay_key(
                    event.key, game.isPaused(), replay_player
                )
                if toggle:
                    game.changePaused(player)
                running = running and not quit_
        if running and not game.isPaused():
            replay_player.step_forward()
        game.updateScreen(player, replay_player.sim.getLines())
        game.limitFps()
    game.cleanup()
    return 0


def run_human_overlay(
    seed: Optional[int] = None,
    start_pace: int = game_config.START_PACE,
    reward_config: Optional[RewardConfig] = None,
    show_hitbox: bool = False,
) -> int:
    """Play ``engine.maze()`` normally, with the overlay following the game."""
    overlay = Overlay(RewardFunction(reward_config), show_hitbox)
    original = engine.Simulation
    setattr(engine, "Simulation", make_tracked_simulation(overlay))
    try:
        engine.maze(seed=seed, start_pace=start_pace)
    except SystemExit:
        pass
    finally:
        setattr(engine, "Simulation", original)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m training.overlay",
        description="Show the observation and reward overlay (toggle with F3).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", metavar="PATH", help="replay file to play back")
    source.add_argument("--seed", type=int, help="play yourself on this maze seed")
    parser.add_argument("--start-pace", type=int, default=game_config.START_PACE)
    parser.add_argument("--reward-config", metavar="PATH", help="JSON RewardConfig")
    parser.add_argument("--hitbox", action="store_true", help="draw the player hitbox")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    reward_config = (
        load_reward_config(args.reward_config) if args.reward_config else None
    )
    if args.replay:
        return run_replay_overlay(loadReplay(args.replay), reward_config, args.hitbox)
    return run_human_overlay(args.seed, args.start_pace, reward_config, args.hitbox)


if __name__ == "__main__":
    sys.exit(main())
