# Training tools

ML-side tooling for Infinite Maze. It lives outside `infinite_maze/` so the game
stays free of ML code and dependencies (stdlib + `pygame` only).

| Module | Purpose |
|---|---|
| `config.py` | Observation/reward constants, **fixed seed sets**, evaluation and overlay settings |
| `observation.py` | `build_observation(sim)`: egocentric wall raster, four rays, scalars |
| `reward.py` | `RewardFunction` / `RewardBreakdown` / `RewardConfig` |
| `policies.py` | `RandomPolicy`, `AlwaysRightPolicy`, `HeuristicPolicy` (+ `RasterHeuristicPolicy`) |
| `evaluate.py` | `evaluate(policy, seeds, ...)`, comparison table CLI |
| `overlay.py` | Observation/reward overlay for replays and human play |

## Observation

`build_observation(sim)` returns an `Observation`:

- `grid`: `OBS_GRID_SIZE x OBS_GRID_SIZE` raster (default 9 x 9 cells of 11 px), centred on
  the player; `1` where a wall crosses the cell. Independent of scroll offsets.
- `rays`: distance to the nearest wall right, left, up, down, as a fraction of
  `OBS_RAY_CAP_PX`. Walls touching the player's box count, like the engine's collision.
- `scalars`: x, y (within the play area), pace and ticks fraction, all in `[0, 1]`.

## Reward

Separate from the in-game score (which is floored at 0). Per step:
`progress_weight * moved_x - step_penalty`, minus `death_penalty` on termination and
plus `truncation_bonus` on truncation. `RewardConfig.validate(max_steps)` rejects a
death penalty that does not exceed the step penalty accrued over a whole episode, so
dying can never beat surviving. `RewardBreakdown` keeps the parts visible.

The default `death_penalty` is 100 (not 50) so the check also passes for 5000-step
episodes.

## Evaluating baselines

```bash
poetry run python -m training.evaluate --policy all --seeds validation \
    --max-steps 2000 --out training/runs/baselines.json
```

Options: `--policy {random,right,heuristic,heuristic-raster,all}`,
`--seeds {train,validation,test}`, `--num-seeds N`, `--start-pace N`,
`--reward-config path.json`, `--worst-replay-dir DIR` (writes each policy's
worst-seed episode as a replay you can watch with the overlay).

Seed sets are permanent: `train` 0-99, `validation` 1000-1049, `test` 2000-2049. Do not
tune on `test`.

### Baseline numbers (validation seeds, `--max-steps 2000`, default reward)

| Policy | Mean return | Median | Mean score | Mean steps | Death rate |
|---|---|---|---|---|---|
| random | -102.5 | -101.1 | 0.8 | 231 | 100% |
| right | 38.1 | 24.0 | 58.1 | 2000 | 0% |
| heuristic | 81.1 | 78.0 | 101.1 | 2000 | 0% |

A learned policy has to clear the heuristic's mean return of 81.1 on these seeds.
Episodes are capped at 2000 steps, which is just past the first pace increase (tick
1800), so death rate for the non-random policies is 0%; use `--start-pace` or a larger
`--max-steps` to stress them.

### Raster vs rays

The same heuristic reading only the raster (`heuristic-raster`) scores a mean return
of 63.7 on the same seeds versus 81.1 with rays. The raster's 11 px resolution cannot
see a wall that is 2 px away, so for a first learning run the rays alone are the
stronger signal; the raster is still available for a convolutional model.

## Overlay

```bash
poetry run python -m training.overlay --replay path.json [--reward-config r.json] [--hitbox]
poetry run python -m training.overlay --seed 1234        # play yourself
```

Draws the observation window (orange = wall cells), the four rays, and a panel with
seed, tick, pace, last intent, step reward breakdown and cumulative return.

- `F3`: toggle the overlay.
- Replay mode: `SPACE` pause, `.` step forward and `,` step back (while paused; going
  back re-runs the replay from its start), `ESC` quit.

The only change to `infinite_maze/` for this is `Game.setOverlay(callback)`.
