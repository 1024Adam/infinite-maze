---
name: run-experiment
description: "Prepare and report on a reproducible Infinite Maze training experiment. Use when: launching or reviewing a PPO/RL training run, pinning an experiment config, running baselines and smoke tests before a long run, or summarising a finished run against baselines and the expert ladder."
---
Run an experiment for Infinite Maze following these steps. Follow the `ml-dev` agent conventions and `plans/03-ml-readiness-and-first-training-run.md`.

Inputs (ask for any that are missing): the experiment config path under `training/configs/`, and whether this is a **prepare** run (before launch) or a **report** run (after the user's long run finished, with the run directory under `training/runs/`).

## Prepare

1. **Check the environment**
   - Confirm ML groups are installed: `poetry run python -c "import gymnasium, stable_baselines3, torch"`. If this fails, stop and report the install command (`poetry install --with ml-env,ml-train`).
   - Set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.
2. **Validate the config**
   - Load it and run the reward validation for its `max_steps`; report any failure.
   - Confirm it is committed (`git status` shows no changes to it) and that `training/runs/<name>/` does not already exist (never overwrite a run).
3. **Re-run baselines with the same settings** (`start_pace`, `max_steps`, `frame_skip`, seeds): random, right, heuristic and oracle, using `python -m training.evaluate`. Report the table.
4. **Smoke run**: run the same config with a tiny `total_timesteps` override (a few thousand) into a throwaway run name. Run `python .github/skills/run-experiment/check_run.py training/runs/<throwaway>` to confirm every artefact is written, and confirm the model can be evaluated and replayed with the overlay. Delete the throwaway run directory afterwards.
5. **Give the launch command** for the full run, for example:

   ```bash
   SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
     poetry run python -m training.train --config training/configs/<name>.json
   ```

   Also state the expected duration (from the smoke run's steps per second), the number of CPU cores it will use, and what to send back when it finishes. Do not start the long run yourself.

## Report

1. Run `python .github/skills/run-experiment/check_run.py training/runs/<name>` to verify the run is complete. Read `metadata.json`, `metrics.csv` and `eval.json`; note the commit, seed and library versions.
2. Evaluate the final and best models with `python -m training.evaluate --policy model --model <path> --frame-skip <k>` on the training seed(s) and on the `validation` seeds, using the same settings as the baselines.
3. Fill in [report-template.md](./report-template.md): the setup block, one comparison table for the training seed(s) and one for the validation seeds, and the expert-ladder status for each.
4. Learning curve summary: return and score versus timesteps, and whether it plateaued.
5. Look at the best and worst replays (paths under `training/runs/<name>/replays/`) and describe behaviour: where the agent stalls, any oscillation, wall-hugging or other exploits. Give the overlay command to watch them: `poetry run python -m training.overlay --replay <path>`.
6. **Label the result**: a fixed-seed result is potentially memorised; do not present it as generalisation.
7. Recommend one next step (reward change, curriculum change, more steps, history/recurrent policy, or shaping per the plan's trigger) and list any request for `pygame-dev` or decision needed.

Never use the `test` seed set for tuning or model selection.
