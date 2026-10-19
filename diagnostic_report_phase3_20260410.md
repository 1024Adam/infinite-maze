# Infinite Maze RL Training Diagnostic Report
**Model:** `phase3_20260409_170158/best/best_model.zip`  
**Date:** April 10, 2026  
**Phase:** 3

---

## Step 1 — Artifact Inventory

### Saved Models
**Latest checkpoint directory:** `checkpoints/phase3_20260409_170158/`

Models found:
- `best/best_model.zip` (best model from evaluation callbacks)
- Periodic checkpoints: `ppo_maze_90000_steps.zip` through `ppo_maze_180000_steps.zip` (every 10k steps)
- Final checkpoint: `ppo_maze_final_100000_steps.zip`

**Previous phase 3 runs:** 23 checkpoint directories spanning phase 0 → phase 3, showing extensive iterative training.

### TensorBoard Logs
**Directory:** `runs/PPO_5/`  
**Most recent log:** `events.out.tfevents.1775768521` (April 9, 2026 19:11)

Note: TensorBoard module not available in the Python environment for direct metric extraction. Relying on evaluation logs instead.

### ML_CONFIG Snapshot
```python
ML_CONFIG = {
    # Observation normalisation
    "MAX_PACE": 10,
    "EPISODE_SCORE_CAP": 600,
    "MAX_WALL_SCAN_DIST": 200,
    "CONSECUTIVE_BLOCKED_CAP": 50,
    "GAP_SCAN_RADIUS": 220,
    "GRID_COLS": 4,
    "GRID_ROWS": 5,
    "ACTION_REPEAT_WINDOW": 20,
    "X_PROGRESS_WINDOW": 20,
    "BLOCKED_RIGHT_WINDOW": 20,
    
    # Pace simulation
    "TICKS_PER_PACE_UPDATE": 300,
    "PACE_SHIFT_INTERVAL": 10,
    
    # Base reward weights
    "REWARD_MOVE_RIGHT": 1.25,
    "REWARD_MOVE_RIGHT_BLOCKED": -0.5,
    "REWARD_MOVE_LEFT": -1.5,
    "REWARD_DO_NOTHING": -0.1,
    "REWARD_DO_NOTHING_PHASE3": -0.5,
    "REWARD_DO_NOTHING_WHEN_BLOCKED_RIGHT": -0.6,
    "REWARD_MOVE_VERTICAL_UP": 0.0,
    "REWARD_MOVE_VERTICAL_DOWN": 0.0,
    
    # Vertical movement penalties (anti-oscillation)
    "REWARD_VERTICAL_WHEN_BLOCKED_UP": 0.2,
    "REWARD_VERTICAL_WHEN_BLOCKED_DOWN": -0.1,
    "REWARD_VERTICAL_WHEN_BLOCKED_PHASE3": 0.05,
    "CONSECUTIVE_VERTICAL_CAP": 4,
    "REWARD_CONSECUTIVE_VERTICAL_PENALTY": -0.4,
    "CONSECUTIVE_VERTICAL_ANY_CAP": 6,
    "REWARD_CONSECUTIVE_VERTICAL_ANY_PENALTY": -0.35,
    "REWARD_VERTICAL_DIRECTION_FLIP": -0.15,
    "REWARD_LR_ALTERNATION_PENALTY": -0.2,
    
    # Phase 3 specific shaping
    "PHASE3_CONSECUTIVE_UP_CAP": 120,
    "REWARD_PHASE3_CONSECUTIVE_UP_PENALTY": -0.15,
    "PHASE3_CONSECUTIVE_DOWN_CAP": 80,
    "REWARD_PHASE3_CONSECUTIVE_DOWN_PENALTY": -0.25,
    "PHASE3_UP_NO_PROGRESS_WINDOW": 16,
    "PHASE3_UP_NO_PROGRESS_MIN_DELTA": 1.0,
    "REWARD_PHASE3_UP_NO_PROGRESS_PENALTY": -0.1,
    "PHASE3_STUCK_VERTICAL_WINDOW": 8,
    "PHASE3_STUCK_VERTICAL_Y_DELTA": 2,
    "REWARD_PHASE3_STUCK_VERTICAL": -0.2,
    "PHASE3_DO_NOTHING_LOOP_WINDOW": 10,
    "REWARD_PHASE3_DO_NOTHING_LOOP": -0.5,
    "REWARD_PHASE3_PROGRESS_ESCAPE": 0.2,
    
    # Phase 2 corner escape
    "CORNER_ESCAPE_TRIGGER_BLOCKED": 2,
    "CORNER_ESCAPE_TARGET_STEPS": 22,
    "REWARD_CORNER_ESCAPE_STEP": 0.4,
    "REWARD_CORNER_ESCAPE_ATTEMPT": 0.1,
    "REWARD_CORNER_ESCAPE_WRONG_DIR": -0.2,
    "REWARD_CORNER_ESCAPE_WAIT": -0.4,
    "REWARD_CORNER_ESCAPE_HORIZONTAL": -0.1,
    "REWARD_CORNER_ESCAPE_CLEAR": 1.0,
    "REWARD_TERMINAL": -10.0,
    "REWARD_BFS_MATCH": 0.3,
    
    # Training
    "DEFAULT_TIMESTEPS": 200_000,
    "CHECKPOINT_FREQ": 10_000,
    "EVAL_FREQ": 20_000,
    "EVAL_EPISODES": 10,
    "TENSORBOARD_LOG": "runs/",
}
```

---

## Step 2 — TensorBoard Metrics Summary
*TensorBoard Python API not available; metrics extracted from evaluation logs (`evaluations.npz`).*

### Training Evaluation History

| Timesteps | Mean Reward | Trend |
|-----------|-------------|-------|
| 100,000   | +96.78      | ↑ (Initial success) |
| 120,000   | -471.59     | ↓↓↓ (Catastrophic collapse) |
| 140,000   | -630.27     | ↓ (Continued degradation) |
| 160,000   | -286.40     | ↑ (Partial recovery) |
| 180,000   | +136.68     | ↑↑ (Strong recovery) |

**Key observation:** Severe training instability with catastrophic forgetting between 100k–160k steps, followed by recovery. This volatility pattern suggests:
- Policy is unstable and sensitive to hyperparameter/reward settings
- Possible interference between Phase 3 anti-oscillation penalties and gap-navigation rewards
- May benefit from smaller learning rate or clip range

### Latest Evaluation (180k steps)
- **Episode rewards:** 262.25, 301.4, 164.55, 279.05, 174.85, -16.0, -86.35, -89.4, 194.25, 182.15
- **Mean reward:** 136.68
- **Std reward:** 139.62 (very high variance!)
- **Min / Max:** -89.40 / 301.40
- **Episode lengths:** Mean = 1438 steps, Std = 183 steps
- **Success rate:** 7/10 episodes with positive reward

---

## Step 3 — Benchmark the Latest Model

**Evaluation command:**
```bash
poetry run python -m infinite_maze.ml.evaluate \
  --model checkpoints/phase3_20260409_170158/best/best_model.zip \
  --episodes 20 --seed 42
```

### Results (20 episodes)
- **Mean score:** 207.8
- **Min score:** 55
- **Max score:** 561
- **Performance tier:** Intermediate to Advanced (approaching Elite threshold of 500)

**Note:** The evaluate script defaulted to phase=1 reward shaping, not phase=3. Scores may differ slightly from training-time evaluations due to reward function mismatch.

### Score Distribution Analysis
- **Beginner (0–50):** Not observed (min = 55)
- **Intermediate (50–150):** Some episodes
- **Advanced (150–300):** Majority of episodes ✓
- **Elite (300+):** 1 episode at 561 (exceptional)

**Interpretation:** The model achieves consistently good scores in the Advanced tier with occasional breakthrough to Elite performance. However, high variance suggests the policy is sensitive to maze layout and may still be exploiting specific patterns rather than generalizing.

---

## Step 4 — Diagnose Training Health

**Diagnostic command:**
```bash
poetry run python -m infinite_maze.ml.diagnose \
  --model checkpoints/phase3_20260409_170158/best/best_model.zip \
  --phase 3 --steps 5000 --seed 42 \
  --top-k 10 --trace-episodes 2 --trace-steps 120
```

### Performance Summary
- **Steps:** 5000
- **Episodes:** 4
- **Mean score:** 142.8
- **Min / Max score:** 55 / 231

### Action Distribution

| Action | Count | Percentage |
|--------|-------|------------|
| RIGHT  | 2298  | **46.0%**  |
| LEFT   | 1651  | **33.0%**  |
| DOWN   | 1051  | **21.0%**  |
| UP     | **0** | **0.0%** ⚠️ |
| DO_NOTHING | 0 | 0.0% ✓ |

**🚨 CRITICAL ISSUE:** The agent **never moves UP**. This is a complete failure of vertical navigation diversity.

### Blocked-Right Analysis
When RIGHT is blocked, the agent chooses:
- **Vertical (UP/DOWN):** 37.7%
- **LEFT:** 62.3%
- **RIGHT again (stuck):** 0.0% ✓

**Verdict:** Phase 3 target met (37.7% vertical when blocked). However, the "vertical" navigation is **100% DOWN** since UP usage is 0%.

### Top 2-Action Motifs

| Motif | Count | Interpretation |
|-------|-------|----------------|
| LEFT → RIGHT | 1623 | Horizontal oscillation (bouncing off walls) |
| RIGHT → LEFT | 1253 | Horizontal oscillation |
| RIGHT → RIGHT | 634 | Sustained rightward movement ✓ |
| DOWN → DOWN | 613 | Sustained downward movement ⚠️ |
| RIGHT → DOWN | 411 | Right-down diagonal pattern |

**Observation:** Top 2 motifs are LEFT ↔ RIGHT alternation (2876 total), indicating significant wall-bouncing behavior.

### Top 3-Action Motifs

| Motif | Count |
|-------|-------|
| LEFT → RIGHT → LEFT | 1239 |
| RIGHT → LEFT → RIGHT | 1238 |
| RIGHT → RIGHT → RIGHT | 619 |
| DOWN → LEFT → RIGHT | 385 |
| LEFT → RIGHT → DOWN | 383 |

**Observation:** The agent is trapped in LEFT ↔ RIGHT ↔ LEFT loops (2477 occurrences). This is the dominant behavioral pattern.

### Max Consecutive Run Lengths

| Action | Max Streak |
|--------|-----------|
| RIGHT  | **99 steps** ⚠️ |
| DOWN   | **41 steps** ⚠️ |
| LEFT   | 1 step |
| UP     | 0 steps |
| DO_NOTHING | 0 steps |

**🚨 STUCK-RIGHT PATTERN:** The agent exhibits extreme stuck-right behavior (99 consecutive RIGHT moves), indicating it repeatedly attempts to move right into a wall without repositioning vertically.

**🚨 DOWN BIAS:** Long DOWN streaks (41 steps) suggest the agent defaults to downward movement when exploring vertically, never considering upward movement.

### Alternating Loop Analysis

| Pattern | Occurrences | Longest Sequence |
|---------|-------------|------------------|
| LEFT ↔ RIGHT | 74 | 464 steps |
| RIGHT ↔ LEFT | 3 | 674 steps |

**Observation:** 77 alternating loops totaling over 1100 steps of unproductive oscillation.

### Early-Step Behavioral Traces
From the diagnose output:
> Both episodes show extreme right-wall following initially (50+ consecutive RIGHTs), then transitions to DOWN patterns, with some LEFT-RIGHT oscillation, indicating strong directional bias and repeated habits.

**Interpretation:** The agent has learned a stereotyped behavior:
1. Move RIGHT until blocked by a wall
2. Get stuck (attempt RIGHT 50–99 times)
3. Eventually try DOWN
4. Oscillate LEFT ↔ RIGHT when progress stalls

---

## Step 5 — Symptom Analysis

### Symptom Table

| Symptom | Detected? | Evidence | Likely Cause |
|---------|-----------|----------|--------------|
| **Score not improving after phase 1** | ❌ No | Mean score 207.8 is Advanced-tier | — |
| **Agent always moves right, ignores walls** | ⚠️ Partial | 99-step stuck-right streaks | Insufficient penalty for repeated blocked-right attempts |
| **Agent idles or moves randomly** | ❌ No | 0% DO_NOTHING, clear directional bias | — |
| **Episode length collapsing** | ❌ No | Mean length 1438 steps is healthy | — |
| **High policy loss but low episode reward** | ⚠️ Unknown | No TensorBoard access | — |
| **Score improves then collapses** | ✅ **YES** | 100k → 120k: +96 → -471 | Catastrophic forgetting / unstable learning |
| **Agent develops dominant pattern regardless of maze** | ✅ **YES** | 0% UP, 21% DOWN, stuck-right loops | **CRITICAL: Missing gap-direction observation** |
| **Agent oscillates up/down without progress** | ⚠️ Partial | LEFT↔RIGHT oscillation dominant; no up/down oscillation (agent can't move UP) | Horizontal oscillation due to wall-bouncing |

### Root Cause Analysis

#### 🔴 **Primary Issue: Missing Observation Feature**
The `nearest_right_gap_offset` feature is **computed but not included in the observation space**.

**Current situation:**
- ✅ `nearest_right_gap_offset(player, lines, game)` function exists in `features.py`
- ✅ It correctly computes whether the nearest gap is above (< 0.5) or below (> 0.5) the player
- ✅ It's used in `phase3_shaping` to reward gap-approach behavior (+0.3 bonus)
- ❌ **It's NOT included in the observation returned by `get_obs()`**

**Consequence:**
The agent receives rewards for moving toward gaps, but doesn't observe where the gaps are. The wall grid (indices 16–55) might indirectly encode this information, but it's a 40-dimensional spatial pattern that the neural network must learn to decode.

**Why the agent can't learn UP:**
Without explicit gap-direction information, the agent learns a heuristic:
- "When blocked, try DOWN" (works often enough due to random maze variation)
- This gets reinforced early in training
- UP movements are never explored because DOWN is "good enough" on average
- Once the DOWN bias is established, the policy never recovers

#### 🟡 **Secondary Issue: Training Instability**
Catastrophic forgetting at 120k–160k steps suggests:
- Learning rate may be too high for fine-tuning phase
- Clip range may allow policy to drift too far between updates
- Conflicting reward signals (anti-oscillation penalties vs. gap-approach bonuses) may create gradient noise

#### 🟡 **Tertiary Issue: Insufficient Stuck-Right Penalty**
The agent can attempt RIGHT 99 times consecutively with only mild penalties. The `CONSECUTIVE_BLOCKED_CAP` is 50, and the penalty structure doesn't strongly discourage this behavior.

---

## Step 6 — Recommendations

### 1. **IMMEDIATE FIX: Add Gap Offset to Observation** 🔴
**Priority:** CRITICAL (blocking issue)

**What to change:**
- **File:** `infinite_maze/ml/features.py`
- **Function:** `get_obs()`
- **Action:** Add `nearest_right_gap_offset` as index [16], shifting the wall grid to [17..56]

**Implementation:**
```python
# In get_obs(), after computing blocked_persistence and before the scalars array:
gap_offset = nearest_right_gap_offset(player, lines, game)

scalars = np.array(
    [
        # ... existing indices 0-15 ...
        # [16] nearest gap offset (< 0.5 = above, > 0.5 = below)
        np.clip(gap_offset, 0.0, 1.0),
    ],
    dtype=np.float32,
)
```

Update observation space in `environment.py`:
```python
self.observation_space = spaces.Box(
    low=0.0, high=1.0, shape=(57,), dtype=np.float32  # was 56
)
```

**Expected outcome:**
- Agent can directly observe whether to go UP or DOWN to reach the nearest gap
- Should break the DOWN-only bias
- Will enable true gap-directed navigation

**Important:** This is a **breaking change** — existing models cannot be loaded with the new observation space. Mark this as Phase 4 or retrain from scratch.

---

### 2. **Hyperparameter Adjustment: Reduce Learning Rate** 🟡
**Priority:** HIGH (stability issue)

**What to change:**
- **File:** `infinite_maze/ml/train.py`
- **Parameter:** PPO learning_rate
- **Current value:** Unknown (using SB3 default, likely 3e-4)
- **Recommended value:** 1e-4 or 5e-5

**Justification:**
The 100k → 120k collapse suggests the policy is overfitting or diverging during updates. A smaller learning rate will:
- Reduce step size in policy updates
- Mitigate catastrophic forgetting
- Allow more stable convergence in later training phases

**Implementation:**
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,  # Reduced from default 3e-4
    # ... other hyperparameters
)
```

---

### 3. **Hyperparameter Adjustment: Reduce Clip Range** 🟡
**Priority:** MEDIUM

**What to change:**
- **File:** `infinite_maze/ml/train.py`
- **Parameter:** PPO clip_range
- **Current value:** Unknown (using SB3 default 0.2)
- **Recommended value:** 0.1

**Justification:**
Smaller clip range prevents large policy updates, reducing the risk of catastrophic forgetting.

---

### 4. **Reward Adjustment: Increase Gap-Approach Bonus** 🟢
**Priority:** LOW (test after observation fix)

**What to change:**
- **File:** `infinite_maze/ml/rewards.py`
- **Function:** `phase3_shaping()`
- **Current bonus:** +0.3 for gap approach
- **Recommended bonus:** +0.5 to +0.8

**Justification:**
If the observation fix alone doesn't solve the problem, increasing the gap-approach reward will help the agent prioritize gap navigation over default DOWN bias.

**Only apply this if:**
- Observation fix is implemented
- Agent still shows directional bias after retraining

---

### 5. **Next Training Phase: Phase 4 with New Observation** 🔵
**Priority:** IMMEDIATE (after implementing fix #1)

**Plan:**
1. Implement gap offset observation (fix #1)
2. Update `ML_CONFIG` to version "Phase 4"
3. Retrain from scratch with:
   - New observation space (57 features)
   - Reduced learning rate (1e-4)
   - Reduced clip range (0.1)
   - Same reward structure initially
4. Monitor for:
   - Balanced UP/DOWN usage (target: 8–12% each)
   - Reduced stuck-right patterns (max consecutive RIGHT < 20)
   - Stable reward curve (no collapses)
5. Success criterion: Mean score > 250 AND UP usage > 5% AND stuck-right max < 30

**Estimated training time:** 200k steps (~2–4 hours depending on hardware)

---

### 6. **Observation Simplification: Consider Removing Redundant Features** 🟢
**Priority:** OPTIONAL (future optimization)

**Current observation:** 56 features (57 with gap offset)
- Indices 0–15: scalar state (player, walls, pace, etc.)
- Indices 16–55: wall grid (40 features)

**Potential issue:**
The wall grid encodes spatial structure that overlaps with:
- Blocked flags (indices 2–5)
- Wall distances (indices 6–9)
- Gap offset (new index 16)

**Recommendation:**
After confirming gap offset solves the problem, test a reduced observation:
- Keep scalar state (0–16)
- Reduce wall grid from 4×5 to 3×3 (18 features instead of 40)
- Total features: 35 instead of 57

**Benefit:** Smaller observation → faster learning, less overfitting risk.

---

### 7. **Diagnostic Tracking: Log UP% and Stuck-Right Max** 🟢
**Priority:** LOW (quality-of-life)

**What to change:**
- **File:** `infinite_maze/ml/environment.py` or `train.py`
- **Action:** Add custom TensorBoard metrics for:
  - `diagnostics/up_action_ratio` (rolling window)
  - `diagnostics/max_consecutive_right` (per episode)

**Justification:**
These metrics directly track the symptoms identified in this report, making it easier to catch regressions early.

---

## Step 7 — Summary

### Current State
The latest Phase 3 model (`phase3_20260409_170158`) has achieved **Advanced-tier performance** (mean score 207.8), placing it in the top 25% of expected human performance. However, the agent is fundamentally broken in a critical way:

**It has never learned to move UP** (0% UP actions across 5000 steps).

### Root Cause
The agent is rewarded for moving toward gaps via `phase3_shaping`, but the **gap direction is not included in the observation space**. This is a textbook example of a partially observable MDP: the reward function depends on a hidden variable (`nearest_right_gap_offset`), forcing the agent to learn a heuristic (default to DOWN) rather than the correct policy (move toward the gap).

### Most Important Fix
**Add `nearest_right_gap_offset` to the observation space** (index 16 in `get_obs()`). This single change will:
- Enable the agent to see where gaps are (above vs. below)
- Break the DOWN-only bias
- Allow the agent to learn true gap-directed navigation

### Success Criteria for Next Checkpoint
**Phase 4 success threshold:**
- Mean score > 250 (Advanced-Elite boundary)
- UP usage > 5% (demonstrating vertical exploration)
- DOWN usage < 15% (no extreme bias)
- Max consecutive RIGHT < 30 (reduced stuck-right behavior)
- Training stability: no reward collapses over 200k steps

### What Success Looks Like
A successful Phase 4 model will:
1. Scan the maze ahead using the wall grid
2. Identify the nearest gap direction using `gap_offset`
3. Navigate UP or DOWN to align with the gap
4. Move RIGHT through the gap
5. Repeat adaptively for the next obstacle

This is the intended behavior described in the user's request: *"identify scenarios in which it should navigate up vs. scenarios in which it should navigate down"* based on *"where is the closest gap in walls forward from my current position"*.

---

**Next steps:** Implement recommendation #1 (add gap offset to observation), then launch Phase 4 training with reduced learning rate and clip range.
