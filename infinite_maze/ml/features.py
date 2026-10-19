"""Feature extraction for the Infinite Maze RL environment.

Provides headless-safe observation encoding, AABB collision checks
(replicated from core/engine.py — do not import from engine.py), wall distance
scanning, and a local wall-occupancy grid for maze lookahead.

All public functions are pure (stateless). Episode-level state such as
``consecutive_blocked`` is owned by the environment and passed in as a
parameter.
"""

import numpy as np

from ..utils.config import config

_MC = config.MOVEMENT_CONSTANTS
DO_NOTHING = _MC["DO_NOTHING"]
RIGHT = _MC["RIGHT"]
LEFT = _MC["LEFT"]
UP = _MC["UP"]
DOWN = _MC["DOWN"]

_ML = config.ML_CONFIG


# ---------------------------------------------------------------------------
# AABB collision checks — replicated from core/engine.py, no import
# ---------------------------------------------------------------------------


def is_blocked_right(player, lines) -> bool:
    """Return True if the player cannot move right by one speed unit."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py <= line.getYStart()
                and py + ph >= line.getYStart()
                and px + pw + speed == line.getXStart()
            ):
                return True
        else:
            if (
                px + pw <= line.getXStart()
                and px + pw + speed >= line.getXStart()
                and (
                    (py >= line.getYStart() and py <= line.getYEnd())
                    or (py + ph >= line.getYStart() and py + ph <= line.getYEnd())
                )
            ):
                return True
    return False


def is_blocked_right_at_y(player, lines, y_candidate: float) -> bool:
    """RIGHT collision check with y_candidate substituted for player.getY().

    Keeps player.getX(), player.getWidth(), and all line positions fixed.
    Used by nearest_right_gap_offset to probe candidate Y positions.
    """
    px = player.getX()
    pw = player.getWidth()
    ph = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                y_candidate <= line.getYStart()
                and y_candidate + ph >= line.getYStart()
                and px + pw + speed == line.getXStart()
            ):
                return True
        else:
            if (
                px + pw <= line.getXStart()
                and px + pw + speed >= line.getXStart()
                and (
                    (y_candidate >= line.getYStart() and y_candidate <= line.getYEnd())
                    or (
                        y_candidate + ph >= line.getYStart()
                        and y_candidate + ph <= line.getYEnd()
                    )
                )
            ):
                return True
    return False


def is_blocked_left(player, lines) -> bool:
    """Return True if the player cannot move left by one speed unit."""
    px = player.getX()
    py = player.getY()
    ph = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py <= line.getYStart()
                and py + ph >= line.getYStart()
                and px - speed == line.getXEnd()
            ):
                return True
        else:
            if (
                px >= line.getXEnd()
                and px - speed <= line.getXEnd()
                and (
                    (py >= line.getYStart() and py <= line.getYEnd())
                    or (py + ph >= line.getYStart() and py + ph <= line.getYEnd())
                )
            ):
                return True
    return False


def is_blocked_up(player, lines) -> bool:
    """Return True if the player cannot move up by one speed unit."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py >= line.getYStart()
                and py - speed <= line.getYStart()
                and (
                    (px >= line.getXStart() and px <= line.getXEnd())
                    or (px + pw >= line.getXStart() and px + pw <= line.getXEnd())
                )
            ):
                return True
        else:
            if (
                px <= line.getXStart()
                and px + pw >= line.getXStart()
                and py - speed == line.getYEnd()
            ):
                return True
    return False


def is_blocked_down(player, lines) -> bool:
    """Return True if the player cannot move down by one speed unit."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    speed = player.getSpeed()

    for line in lines:
        if line.getIsHorizontal():
            if (
                py + ph <= line.getYStart()
                and py + ph + speed >= line.getYStart()
                and (
                    (px >= line.getXStart() and px <= line.getXEnd())
                    or (px + pw >= line.getXStart() and px + pw <= line.getXEnd())
                )
            ):
                return True
        else:
            if (
                px <= line.getXStart()
                and px + pw >= line.getXStart()
                and py + ph + speed == line.getYStart()
            ):
                return True
    return False


def is_blocked(player, lines, direction: int) -> bool:
    """Return True if the player is blocked in the given direction.

    Parameters
    ----------
    direction : int
        One of the ``config.MOVEMENT_CONSTANTS`` values (RIGHT, LEFT, UP, DOWN).
        DO_NOTHING always returns False.
    """
    if direction == RIGHT:
        return is_blocked_right(player, lines)
    if direction == LEFT:
        return is_blocked_left(player, lines)
    if direction == UP:
        return is_blocked_up(player, lines)
    if direction == DOWN:
        return is_blocked_down(player, lines)
    return False


# ---------------------------------------------------------------------------
# Wall distance scanning
# ---------------------------------------------------------------------------


def _wall_dist_right(player, lines) -> float:
    """Raw pixel distance to the nearest right-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    edge = px + pw
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        dist = line.getXStart() - edge
        if dist < 0:
            continue
        if line.getIsHorizontal():
            if py <= line.getYStart() <= py + ph:
                min_dist = min(min_dist, dist)
        else:
            ys = line.getYStart()
            ye = line.getYEnd()
            if (py >= ys and py <= ye) or (py + ph >= ys and py + ph <= ye):
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_left(player, lines) -> float:
    """Raw pixel distance to the nearest left-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    ph = player.getHeight()
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        dist = px - line.getXEnd()
        if dist < 0:
            continue
        if line.getIsHorizontal():
            if py <= line.getYStart() <= py + ph:
                min_dist = min(min_dist, dist)
        else:
            ys = line.getYStart()
            ye = line.getYEnd()
            if (py >= ys and py <= ye) or (py + ph >= ys and py + ph <= ye):
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_down(player, lines) -> float:
    """Raw pixel distance to the nearest downward-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    edge = py + ph
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        if line.getIsHorizontal():
            dist = line.getYStart() - edge
            if dist < 0:
                continue
            xs = line.getXStart()
            xe = line.getXEnd()
            if (px >= xs and px <= xe) or (px + pw >= xs and px + pw <= xe):
                min_dist = min(min_dist, dist)
        else:
            dist = line.getYStart() - edge
            if dist < 0:
                continue
            xs = line.getXStart()
            if px <= xs <= px + pw:
                min_dist = min(min_dist, dist)
    return min_dist


def _wall_dist_up(player, lines) -> float:
    """Raw pixel distance to the nearest upward-blocking wall, capped at MAX_WALL_SCAN_DIST."""
    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    min_dist = float(_ML["MAX_WALL_SCAN_DIST"])

    for line in lines:
        if line.getIsHorizontal():
            dist = py - line.getYStart()
            if dist < 0:
                continue
            xs = line.getXStart()
            xe = line.getXEnd()
            if (px >= xs and px <= xe) or (px + pw >= xs and px + pw <= xe):
                min_dist = min(min_dist, dist)
        else:
            dist = py - line.getYEnd()
            if dist < 0:
                continue
            xs = line.getXStart()
            if px <= xs <= px + pw:
                min_dist = min(min_dist, dist)
    return min_dist


# ---------------------------------------------------------------------------
# BFS optimal action (Phase 3 curriculum reward)
# ---------------------------------------------------------------------------


def _build_adjacency(player, lines, game):
    """Return a dict mapping (cx, cy) cell coords → set of reachable neighbour cells.

    Uses pixel-level collision checks (is_blocked_right_at_y and equivalents)
    sampled at MAZE_CELL_SIZE intervals to approximate cell-level adjacency.
    """
    CELL = config.MAZE_CELL_SIZE
    y_min = game.Y_MIN
    y_max = game.Y_MAX - player.getHeight()

    # Collect candidate cell centres within the visible window
    # X: player position outward to X_MAX + one extra column for margin
    # Y: full screen height in CELL steps
    px_cell = (player.getX() // CELL) * CELL
    x_start = px_cell - CELL
    x_end = int(game.X_MAX) + CELL * 2

    cells = set()
    for cx in range(x_start, x_end + 1, CELL):
        for cy in range((y_min // CELL) * CELL, (y_max // CELL + 2) * CELL, CELL):
            if y_min <= cy <= y_max:
                cells.add((cx, cy))

    adj = {c: set() for c in cells}

    # Check lateral / vertical adjacency via pixel-level collision checks
    # We reuse existing is_blocked_right_at_y as our wall query
    for cx, cy in cells:
        # Temporarily move player to this cell for collision checks
        orig_x = player.getX()
        orig_y = player.getY()
        player.setX(cx)
        player.setY(cy)

        right_nb = (cx + CELL, cy)
        left_nb = (cx - CELL, cy)
        up_nb = (cx, cy - CELL)
        down_nb = (cx, cy + CELL)

        if right_nb in cells and not is_blocked_right(player, lines):
            adj[(cx, cy)].add(right_nb)
            adj[right_nb].add((cx, cy))
        if left_nb in cells and not is_blocked_left(player, lines):
            adj[(cx, cy)].add(left_nb)
        if up_nb in cells and not is_blocked_up(player, lines):
            adj[(cx, cy)].add(up_nb)
        if down_nb in cells and not is_blocked_down(player, lines):
            adj[(cx, cy)].add(down_nb)

        player.setX(orig_x)
        player.setY(orig_y)

    return adj


def bfs_optimal_action(player, lines, game) -> int:
    """Return the BFS-optimal next action toward the best rightward gap.

    Uses a two-phase search:
    1. Finds all cells from which RIGHT is unblocked (candidate gaps), ordered
       by path distance from the player.
    2. For each candidate (nearest first), measures forward reachability —
       how many additional rightward cells are accessible beyond the gap.
       Skips candidates whose forward reachability is below MIN_FORWARD_CELLS
       (pocket filter), unless no candidate passes the threshold.

    Returns the first action on the shortest path to the selected gap cell,
    or DO_NOTHING if no rightward gap is reachable.

    Parameters
    ----------
    player : Player
    lines  : list[Line]
    game   : Game
    """
    CELL = config.MAZE_CELL_SIZE
    MIN_FORWARD = (
        2  # minimum cells reachable rightward beyond gap to pass pocket filter
    )
    MAX_CANDIDATES = 8  # stop after evaluating this many candidates to bound cost

    player_x = player.getX()
    player_y = player.getY()
    start = ((player_x // CELL) * CELL, (player_y // CELL) * CELL)

    adj = _build_adjacency(player, lines, game)

    if start not in adj:
        return DO_NOTHING

    # Phase 1: BFS from start — record shortest path to every reachable cell
    from collections import deque

    parent = {start: None}
    dist_from_start = {start: 0}
    action_from_start = {start: None}
    queue = deque([start])
    gap_candidates = []  # (dist, cell) for cells where RIGHT is unblocked

    while queue:
        node = queue.popleft()
        dist = dist_from_start[node]

        # Check if RIGHT is unblocked from this cell
        nx, ny = node
        orig_x, orig_y = player.getX(), player.getY()
        player.setX(nx)
        player.setY(ny)
        right_open = not is_blocked_right(player, lines)
        player.setX(orig_x)
        player.setY(orig_y)

        if right_open and node != start:
            gap_candidates.append((dist, node))

        for nb in adj.get(node, set()):
            if nb not in parent:
                parent[nb] = node
                dist_from_start[nb] = dist + 1
                # Record first action taken from start toward this neighbour
                if parent[node] is None:  # nb is directly adjacent to start
                    nx_nb, ny_nb = nb
                    sx, sy = start
                    if nx_nb > sx:
                        first_act = RIGHT
                    elif nx_nb < sx:
                        first_act = LEFT
                    elif ny_nb < sy:
                        first_act = UP
                    else:
                        first_act = DOWN
                    action_from_start[nb] = first_act
                else:
                    action_from_start[nb] = action_from_start.get(node, DO_NOTHING)
                queue.append(nb)

    if not gap_candidates:
        return DO_NOTHING

    # Sort by distance (nearest first)
    gap_candidates.sort(key=lambda x: x[0])

    # Phase 2: forward reachability filter — pick nearest gap that isn't a pocket
    def _forward_reachable(gap_cell) -> int:
        """Count cells reachable rightward of gap_cell (shallow BFS, right-biased)."""
        seen = {gap_cell}
        q = deque([gap_cell])
        count = 0
        while q and count < 10:  # cap to keep cost bounded
            cx, cy = q.popleft()
            for nb in adj.get((cx, cy), set()):
                if nb not in seen and nb[0] >= gap_cell[0]:  # only rightward/same-x
                    seen.add(nb)
                    q.append(nb)
                    count += 1
        return count

    selected = None
    fallback = gap_candidates[0][1]  # nearest gap regardless of reachability

    for i, (_, cell) in enumerate(gap_candidates[:MAX_CANDIDATES]):
        if _forward_reachable(cell) >= MIN_FORWARD:
            selected = cell
            break

    target = selected if selected is not None else fallback

    return action_from_start.get(target, DO_NOTHING)


# ---------------------------------------------------------------------------
# Nearest right-facing gap (Phase 3 shaping feature)
# ---------------------------------------------------------------------------


def nearest_right_gap_offset(player, lines, game) -> float:
    """Return the normalised vertical offset to the nearest Y where RIGHT is unblocked.

    Scans outward from the player's current Y in 5-pixel increments up to
    ``ML_CONFIG["GAP_SCAN_RADIUS"]`` pixels, alternating above and below.

    Returns
    -------
    float in [0.0, 1.0]
        0.5 — gap is at the current Y (or no gap found within radius).
        < 0.5 — nearest gap is above the player.
        > 0.5 — nearest gap is below the player.
    """
    py = player.getY()
    ph = player.getHeight()
    radius = _ML["GAP_SCAN_RADIUS"]
    y_min = game.Y_MIN
    y_max = game.Y_MAX - ph

    # Current Y already unblocked — gap is here
    if not is_blocked_right_at_y(player, lines, py):
        return 0.5

    # Scan outward, alternating up/down, in 5-pixel steps
    for dist in range(5, radius + 1, 5):
        for y_candidate in (py - dist, py + dist):
            if y_candidate < y_min or y_candidate > y_max:
                continue
            if not is_blocked_right_at_y(player, lines, y_candidate):
                offset = (y_candidate - py) / radius * 0.5 + 0.5
                return float(np.clip(offset, 0.0, 1.0))

    return 0.5  # no gap found within scan radius


# ---------------------------------------------------------------------------
# Local wall-occupancy grid
# ---------------------------------------------------------------------------


def get_wall_grid(player, lines) -> np.ndarray:
    """Return a flat binary float32 array encoding wall presence in a local grid.

    The grid covers ``GRID_COLS`` columns × ``GRID_ROWS`` rows centred on the
    player, scanning rightward from the player's right edge. For each cell:

      feature 0 (has_right_wall)  — 1.0 if any vertical wall exists within
                                    this cell's x extent that overlaps the
                                    cell's y range; 0.0 otherwise.
      feature 1 (has_bottom_wall) — 1.0 if any horizontal wall exists within
                                    this cell's y extent that overlaps the
                                    cell's x range; 0.0 otherwise.

    Cells are ordered col 0…COLS-1; within each col, row offsets −HALF…+HALF.

    Shape: (GRID_COLS * GRID_ROWS * 2,) — 40 features with defaults COLS=4, ROWS=5.

    Parameters
    ----------
    player : Player
    lines  : list[Line]
    """
    CELL = config.MAZE_CELL_SIZE
    COLS = _ML["GRID_COLS"]
    ROWS = _ML["GRID_ROWS"]
    half_rows = ROWS // 2

    px = player.getX()
    py = player.getY()
    pw = player.getWidth()
    ph = player.getHeight()
    cy = py + ph // 2  # player centre y

    features = np.zeros(COLS * ROWS * 2, dtype=np.float32)
    idx = 0

    for col in range(COLS):
        left_edge = px + pw + col * CELL
        right_edge = px + pw + (col + 1) * CELL

        for row_offset in range(-half_rows, half_rows + 1):
            top = cy + row_offset * CELL - CELL // 2
            bottom = cy + row_offset * CELL + CELL // 2

            # has_right_wall: vertical wall with x in (left_edge, right_edge]
            #                 whose y-span overlaps [top, bottom]
            has_right = 0.0
            for line in lines:
                if not line.getIsHorizontal():
                    lx = line.getXStart()
                    if left_edge < lx <= right_edge:
                        if line.getYStart() <= bottom and line.getYEnd() >= top:
                            has_right = 1.0
                            break

            # has_bottom_wall: horizontal wall with y in (top, bottom]
            #                  whose x-span overlaps [left_edge, right_edge]
            has_bottom = 0.0
            for line in lines:
                if line.getIsHorizontal():
                    ly = line.getYStart()
                    if top < ly <= bottom:
                        if (
                            line.getXStart() <= right_edge
                            and line.getXEnd() >= left_edge
                        ):
                            has_bottom = 1.0
                            break

            features[idx] = has_right
            features[idx + 1] = has_bottom
            idx += 2

    return features


# ---------------------------------------------------------------------------
# Main observation encoder
# ---------------------------------------------------------------------------


def get_obs(
    player,
    lines,
    game,
    consecutive_blocked: int = 0,
    recent_actions=None,
    recent_x_positions=None,
    recent_blocked_right=None,
    legacy: bool = False,
) -> np.ndarray:
    """Encode the current game state as a flat float32 array.

    Returns shape (57,) by default, or (54,) when ``legacy=True``.
    Use ``legacy=True`` to load models trained before the history features
    (indices 13–15) and gap offset (index 16) were added.

    All values are normalised to [0.0, 1.0].

    Parameters
    ----------
    player : Player
    lines : list[Line]
    game : Game
    consecutive_blocked : int
        Number of consecutive ticks the RIGHT action has been blocked.
        Tracked by the environment and passed in here; this module is stateless.
    recent_actions : deque | None
        Windowed action history for computing action repeat ratio.
        If None or empty, defaults to neutral 0.0 (no repetition).
    recent_x_positions : deque | None
        Windowed x position history for computing x progress.
        If None or has <2 elements, defaults to neutral 0.5 (no progress).
    recent_blocked_right : deque | None
        Windowed blocked-right flags for computing persistence ratio.
        If None or empty, defaults to neutral 0.0 (not persistently blocked).

    Layout
    ------
    [0]     player x (normalised)
    [1]     player y (normalised)
    [2]     blocked right
    [3]     blocked left
    [4]     blocked up
    [5]     blocked down
    [6]     wall distance right (normalised)
    [7]     wall distance left (normalised)
    [8]     wall distance up (normalised)
    [9]     wall distance down (normalised)
    [10]    pace (normalised)
    [11]    distance from death boundary (normalised)
    [12]    consecutive ticks blocked right (normalised)
    [13]    recent action repeat ratio (windowed)
    [14]    signed x-progress over recent window, mapped to [0, 1]
    [15]    blocked-right persistence ratio (windowed)
    [16]    nearest right-facing gap offset (< 0.5 = above, > 0.5 = below)
    [17..56] local wall grid — GRID_COLS × GRID_ROWS × 2 binary features
    """
    ml = _ML
    max_scan = ml["MAX_WALL_SCAN_DIST"]
    x_range = int(game.X_MAX) - game.X_MIN  # int() — X_MAX is WIDTH/2 (float)

    action_repeat = 0.0
    if recent_actions is not None and len(recent_actions) > 0:
        last_action = recent_actions[-1]
        action_repeat = sum(a == last_action for a in recent_actions) / float(
            len(recent_actions)
        )

    signed_progress = 0.5
    if recent_x_positions is not None and len(recent_x_positions) >= 2:
        x_delta = float(recent_x_positions[-1] - recent_x_positions[0])
        signed = np.clip(x_delta / max(x_range, 1), -1.0, 1.0)
        signed_progress = float(np.clip(0.5 + 0.5 * signed, 0.0, 1.0))

    blocked_persistence = 0.0
    if recent_blocked_right is not None and len(recent_blocked_right) > 0:
        blocked_persistence = sum(float(v) for v in recent_blocked_right) / float(
            len(recent_blocked_right)
        )

    # Phase 4: gap offset for directional navigation
    gap_offset = nearest_right_gap_offset(player, lines, game)

    scalars = np.array(
        [
            # [0]  player x
            np.clip(player.getX() / config.SCREEN_WIDTH, 0.0, 1.0),
            # [1]  player y
            np.clip(player.getY() / config.SCREEN_HEIGHT, 0.0, 1.0),
            # [2]  blocked right
            float(is_blocked_right(player, lines)),
            # [3]  blocked left
            float(is_blocked_left(player, lines)),
            # [4]  blocked up
            float(is_blocked_up(player, lines)),
            # [5]  blocked down
            float(is_blocked_down(player, lines)),
            # [6]  wall distance right
            np.clip(_wall_dist_right(player, lines) / max_scan, 0.0, 1.0),
            # [7]  wall distance left
            np.clip(_wall_dist_left(player, lines) / max_scan, 0.0, 1.0),
            # [8]  wall distance up
            np.clip(_wall_dist_up(player, lines) / max_scan, 0.0, 1.0),
            # [9]  wall distance down
            np.clip(_wall_dist_down(player, lines) / max_scan, 0.0, 1.0),
            # [10] pace
            np.clip(game.getPace() / ml["MAX_PACE"], 0.0, 1.0),
            # [11] distance from death boundary
            np.clip((player.getX() - game.X_MIN) / x_range, 0.0, 1.0),
            # [12] consecutive ticks blocked right (normalised)
            np.clip(consecutive_blocked / ml["CONSECUTIVE_BLOCKED_CAP"], 0.0, 1.0),
            # [13] recent action repeat ratio
            np.clip(action_repeat, 0.0, 1.0),
            # [14] signed x progress over recent window
            np.clip(signed_progress, 0.0, 1.0),
            # [15] blocked-right persistence ratio
            np.clip(blocked_persistence, 0.0, 1.0),
            # [16] nearest right-facing gap offset
            np.clip(gap_offset, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    if legacy:
        # Drop indices 13-16 (action_repeat, x_progress, blocked_persistence, gap_offset)
        # to produce the original (53,) observation shape.
        scalars = scalars[:13]
    return np.concatenate([scalars, get_wall_grid(player, lines)])
