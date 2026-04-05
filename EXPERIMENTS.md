# Experiment Log — Unitree RL Lab

Tracking training runs, findings, and tuning decisions for Q1 and other robots.

---

## How to Read This Log

- **Run ID**: timestamp folder name under `logs/rsl_rl/<task>/`
- **Status**: 🟡 running / ✅ done / ❌ failed / 🚫 killed early
- **Key metrics** sampled at end of run or when killed

---

## Q1 (YSXSZ — 10 DOF Bipedal)

### Baseline port from RoboTamer4Qmini

---

#### Run 001 — Initial port, baseline config
- **Date**: 2026-04-05
- **Run ID**: `2026-04-05_13-24-46`
- **Status**: 🟡 running
- **Task**: `Unitree-Q1-Velocity`
- **Command**:
  ```bash
  ./unitree_rl_lab.sh -t --task Unitree-Q1-Velocity --num_envs 4096 --headless
  ```
- **Config changes from default**:
  - First run — no changes, pure baseline port
- **Key hyperparameters**:
  - `decimation = 15` (0.015s control step, matching RoboTamer4Qmini)
  - `dt = 0.001`
  - `num_envs = 4096`
  - Actor: `[512, 256, 128]` hidden dims
  - PD gains: hip_yaw k=30/d=0.5, hip_roll k=60/d=1.5, hip_pitch k=100/d=2.5, knee k=100/d=2.5, ankle k=30/d=0.5
- **Observations at iteration X**:
  - `mean_reward`:
  - `episode_length`:
  - `track_lin_vel_xy`:
  - `bad_orientation` termination %:
- **Notes**:
  - First Isaac Lab training run for Q1
  - Standing height set to 0.7m (estimated — may need tuning)
  - Target base height reward set to 0.55m (verify against real robot)
- **Outcome / next steps**:
  - [ ] Check if robot stays upright past iteration 1000
  - [ ] Verify ankle body name matches contact sensor filter (`ankle_pitch_.*`)
  - [ ] Tune standing height if robot spawns clipping into ground

---

## G1-29dof

### Baseline (unitree_rl_lab default config)

---

#### Run 001 — First successful training run
- **Date**: 2026-04-05
- **Run ID**: `2026-04-05_11-42-18` (approx)
- **Status**: ✅ done (ran to convergence)
- **Task**: `Unitree-G1-29dof-Velocity`
- **Notes**:
  - Fixed `handle_deprecated_rsl_rl_cfg` missing from train.py
  - Fixed robot spawning — switched from USD to URDF (Method 2)
  - `bad_orientation` termination was 99.9% at iteration 236 — normal for early training

---

## Lessons Learned

| Date | Finding |
|---|---|
| 2026-04-05 | Isaac Lab `train.py` needs `handle_deprecated_rsl_rl_cfg` called before `to_dict()` for rsl-rl-lib >= 5.0 |
| 2026-04-05 | G1-29dof config defaults to USD spawn — must switch to URDF for IsaacSim 5.x |
| 2026-04-05 | `./unitree_rl_lab.sh -t` hardcodes `--headless`; use `python scripts/rsl_rl/train.py` directly for GUI |

---

## Tuning Reference

### Reward weight intuition
- Increase `track_lin_vel_xy` weight → more aggressive velocity tracking, less smooth motion
- Increase `action_rate` penalty → smoother actions, slower learning
- Increase `flat_orientation_l2` → more upright posture, may inhibit dynamic leaning
- Increase `alive` weight → robot learns to survive first, velocity tracking second

### Things to try when robot won't walk
1. Reduce `decimation` (shorter control delay)
2. Increase `alive` reward weight temporarily
3. Reduce velocity command range (start slower)
4. Check PD gains — too stiff = jerky, too soft = floppy

### Things to try when reward is noisy / unstable
1. Reduce learning rate (`1e-3` → `5e-4`)
2. Increase `num_mini_batches` (4 → 8)
3. Check for NaN in observations (enable `check_for_nan=True` in runner cfg)
