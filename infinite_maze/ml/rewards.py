"""Reward shaping for the Infinite Maze RL environment.

All functions are pure (stateless). Episode-level state such as
``prev_gap_offset`` and ``consecutive_blocked`` is owned by the environment and
passed in as arguments.

Reward weights are sourced exclusively from ``config.ML_CONFIG``; nothing is
hardcoded in this module.
"""

from ..utils.config import config

_MC = config.MOVEMENT_CONSTANTS
DO_NOTHING = _MC["DO_NOTHING"]
RIGHT = _MC["RIGHT"]
LEFT = _MC["LEFT"]
UP = _MC["UP"]
DOWN = _MC["DOWN"]

_ML = config.ML_CONFIG


def compute_reward(
    action: int,
    blocked_flags: dict,
    terminated: bool,
    game,
    bfs_action: int = -1,
    phase: int = 1,
    player=None,
    recent_actions=None,
    recent_x_positions=None,
    blocked_right: bool = False,
) -> float:
    """Return the shaped reward for one environment step.

    Terminal state takes priority over all per-action rewards.

    Parameters
    ----------
    action : int
        The action taken this step (one of DO_NOTHING / RIGHT / LEFT / UP / DOWN).
    blocked_flags : dict
        Keys ``"right"``, ``"left"``, ``"up"``, ``"down"`` mapping to bools.
        Computed by ``features.is_blocked_*`` before the move is applied.
    terminated : bool
        True when ``player.getX() < game.X_MIN``.
    game : Game
        Live game instance; used for pace in Phase 4 urgency scaling.
        Not used by base rewards — kept in signature for consistency.
    bfs_action : int
        BFS-optimal action for this step, computed by ``features.bfs_optimal_action()``.
        Pass -1 (default) to skip the BFS curriculum bonus.
    phase : int
        Current training phase. BFS bonus only applies for phase >= 3.
    player : Player, optional
        Live player instance. Used for vertical boundary penalty in Phase 1-2.
        Pass None to skip boundary-awareness reward.
    recent_actions : deque | None
        Recent action history from the environment. Used in Phase 3+ to penalise
        LEFT/RIGHT alternation loops (A-B-A).
    recent_x_positions : deque | None
        Recent x-position history from the environment. Used in Phase 3+ to
        penalise repeated UP behavior that does not produce rightward progress.

    Returns
    -------
    float
        Shaped reward value.

    Cases
    -----
    terminal                 → REWARD_TERMINAL        (-10.0)
    RIGHT + not blocked      → REWARD_MOVE_RIGHT      (+1.0)
    RIGHT + blocked          → REWARD_MOVE_RIGHT_BLOCKED (-0.3)
    LEFT  + not blocked      → REWARD_MOVE_LEFT       (-1.5)
    LEFT  + blocked          → REWARD_MOVE_RIGHT_BLOCKED (-0.3)  # symmetric
    DO_NOTHING               → REWARD_DO_NOTHING      (-0.1)
    UP (free)                → REWARD_MOVE_VERTICAL_UP          (0.0)
    DOWN (free)              → REWARD_MOVE_VERTICAL_DOWN         (0.0)
    UP (blocked right)       → REWARD_VERTICAL_WHEN_BLOCKED_UP   (+0.1)  + phase3_shaping gap bonus
    DOWN (blocked right)     → REWARD_VERTICAL_WHEN_BLOCKED_DOWN (+0.1)  + phase3_shaping gap bonus
    + BFS match (phase >= 3) → +REWARD_BFS_MATCH (+0.1) when action == bfs_action
    """
    if terminated:
        return float(_ML["REWARD_TERMINAL"])

    if action == RIGHT:
        if blocked_flags.get("right", False):
            reward = float(_ML["REWARD_MOVE_RIGHT_BLOCKED"])
        else:
            reward = float(_ML["REWARD_MOVE_RIGHT"])

    elif action == LEFT:
        if blocked_flags.get("left", False):
            reward = float(_ML["REWARD_MOVE_RIGHT_BLOCKED"])
        else:
            reward = float(_ML["REWARD_MOVE_LEFT"])

    elif action == DO_NOTHING:
        if phase >= 3:
            reward = float(_ML["REWARD_DO_NOTHING_PHASE3"])
        else:
            reward = float(_ML["REWARD_DO_NOTHING"])
        if phase == 2 and blocked_flags.get("right", False):
            reward += float(_ML["REWARD_DO_NOTHING_WHEN_BLOCKED_RIGHT"])

    else:
        # UP or DOWN
        if blocked_flags.get("right", False):
            if phase >= 3:
                # Phase 3+: keep UP/DOWN symmetric and let BFS/gap shaping guide direction.
                reward = float(_ML["REWARD_VERTICAL_WHEN_BLOCKED_PHASE3"])
            else:
                key = (
                    "REWARD_VERTICAL_WHEN_BLOCKED_UP"
                    if action == UP
                    else "REWARD_VERTICAL_WHEN_BLOCKED_DOWN"
                )
                reward = float(_ML[key])
        else:
            key = (
                "REWARD_MOVE_VERTICAL_UP"
                if action == UP
                else "REWARD_MOVE_VERTICAL_DOWN"
            )
            reward = float(_ML[key])

    # BFS curriculum bonus (Phase 3+): small bonus when action matches BFS-optimal move
    if phase >= 3 and bfs_action >= 0 and action == bfs_action:
        reward += float(_ML["REWARD_BFS_MATCH"])

    # Phase 3+: penalise A-B-A LEFT/RIGHT alternation (e.g. LEFT->RIGHT->LEFT).
    # This suppresses retreat-advance looping without affecting single corrective moves.
    if (
        phase >= 3
        and recent_actions is not None
        and len(recent_actions) >= 2
        and action in (LEFT, RIGHT)
    ):
        prev_action = recent_actions[-1]
        prev_prev_action = recent_actions[-2]
        if (
            prev_prev_action == action
            and prev_action in (LEFT, RIGHT)
            and prev_action != action
        ):
            reward += float(_ML["REWARD_LR_ALTERNATION_PENALTY"])

    # Phase 3+: penalise DOWN when RIGHT is available (asymmetric incentive:
    # prefer RIGHT > DOWN > UP in escape order). This breaks DOWN-dominated local optima.
    if phase >= 3 and action == DOWN and not blocked_right:
        reward += float(_ML["REWARD_PHASE3_UP_NO_PROGRESS_PENALTY"])  # -0.1 penalty

    # Phase 3+: penalise vertical bias when it is not translating into rightward progress.
    if (
        phase >= 3
        and action in (UP, DOWN)
        and recent_x_positions is not None
        and len(recent_x_positions) >= int(_ML["PHASE3_UP_NO_PROGRESS_WINDOW"])
    ):
        x_delta = float(recent_x_positions[-1] - recent_x_positions[0])
        min_delta = float(_ML["PHASE3_UP_NO_PROGRESS_MIN_DELTA"])
        if x_delta < min_delta:
            reward += float(_ML["REWARD_PHASE3_UP_NO_PROGRESS_PENALTY"])

    # Vertical boundary awareness (Phase 1-2 only): discourage getting stuck at top/bottom walls.
    # This reinforces what the wall-grid already shows but accelerates learning in early phases.
    if phase <= 2 and player is not None:
        y_pos = player.getY()
        y_min = game.Y_MIN if hasattr(game, 'Y_MIN') else 40
        y_max = game.Y_MAX if hasattr(game, 'Y_MAX') else 450
        boundary_threshold = float(_ML["Y_BOUNDARY_THRESHOLD"])
        
        if y_pos < y_min + boundary_threshold or y_pos > y_max - boundary_threshold:
            reward += float(_ML["REWARD_APPROACH_Y_BOUNDARY"])

    return reward


def phase3_shaping(
    action: int,
    prev_gap_offset: float,
    new_gap_offset: float,
    prev_blocked_count: int,
    new_blocked_count: int,
    blocked_right: bool,
    prev_x: float,
    new_x: float,
) -> float:
    """Return the Phase 3 directional path-finding bonus.

    This function is **Phase 3 only** and must not be called in Phases 4+.
    The environment enables/disables the call based on the active phase flag.

    Two bonuses (non-exclusive, additive):

    1. **Gap-approach bonus** (+0.3): fires when ``blocked_right=True``,
       the action is vertical (UP or DOWN), and the new gap offset is
       *closer to 0.5* than the previous one (i.e. the agent moved toward
       the nearest right-facing gap).

     2. **Escape bonus** (+0.05): fires whenever ``new_blocked_count`` is
         strictly less than ``prev_blocked_count``, rewarding the agent for
         breaking out of a stuck-right state.

     3. **Progress-on-escape bonus** (``REWARD_PHASE3_PROGRESS_ESCAPE``):
         fires when blocked-right pressure is reduced in the same step and the
         player's x position increases (net rightward progress while escaping).

    Parameters
    ----------
    action : int
    prev_gap_offset : float  Gap offset at the start of this step (from the previous obs).
    new_gap_offset  : float  Gap offset at the end of this step (from the new obs).
    prev_blocked_count : int  ``consecutive_blocked`` before this step.
    new_blocked_count  : int  ``consecutive_blocked`` after this step.
    blocked_right : bool  Whether RIGHT was blocked before the move.
    """
    bonus = 0.0

    # Gap-approach bonus
    if (
        blocked_right
        and action in (UP, DOWN)
        and abs(new_gap_offset - 0.5) < abs(prev_gap_offset - 0.5)
    ):
        bonus += 0.3

    # Escape bonus — agent reduced its stuck counter this step
    if new_blocked_count < prev_blocked_count:
        bonus += 0.05
        if new_x > prev_x:
            bonus += float(_ML["REWARD_PHASE3_PROGRESS_ESCAPE"])

    return bonus


def phase4_direction_shaping(
    action: int,
    gap_offset: float,
    blocked_right: bool,
) -> float:
    """Return Phase 4 directional bonus/penalty for vertical choice.

    When RIGHT is blocked, encourage choosing vertical direction that aligns
    with the nearest right-facing gap:

      - gap_offset < 0.5  -> prefer UP
      - gap_offset > 0.5  -> prefer DOWN

    A small neutral band around 0.5 avoids noisy penalties when the gap is
    effectively level with the player.
    """
    if not blocked_right or action not in (UP, DOWN):
        return 0.0

    neutral = float(_ML["PHASE4_GAP_NEUTRAL_BAND"])
    if abs(gap_offset - 0.5) <= neutral:
        return 0.0

    gap_above = gap_offset < 0.5
    correct_action = UP if gap_above else DOWN
    if action == correct_action:
        return float(_ML["REWARD_PHASE4_GAP_DIRECTION_MATCH"])
    return float(_ML["REWARD_PHASE4_GAP_DIRECTION_MISMATCH"])


def urgency_scaled_terminal(game) -> float:
    """Return the Phase 4 terminal penalty, scaled by current pace.

    Higher pace → heavier penalty, encouraging the agent to reach high
    scores before pace climbs into the unwinnable range.

    Returns ``REWARD_TERMINAL * (1 + pace / MAX_PACE)``.
    """
    scale = 1.0 + game.getPace() / float(_ML["MAX_PACE"])
    return float(_ML["REWARD_TERMINAL"]) * scale
