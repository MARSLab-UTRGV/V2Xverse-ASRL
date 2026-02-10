# Training RL for Adaptive CBF Gammas (V2Xverse-ASRL)

This document describes the current RL pipeline used to learn adaptive CBF gamma values in CARLA.
It matches the code in this repository.

## 1) End-to-end pipeline

```
Sensors -> Perception (frozen) -> Planner (frozen) -> V2X_Controller
       -> RL policy (gamma actions) -> CBFQPFilter -> throttle/brake -> CARLA
```

- RL does not train perception/planning.
- RL learns three gamma values for CBF: vehicle, pedestrian, bicycle.
- CBF currently filters longitudinal behavior (safe desired speed).

## 2) Code map (source of truth)

- Integration: `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`
- Agent loop / skip-frame behavior: `simulation/leaderboard/team_code/pnp_agent_e2e.py`
- RL trainer: `simulation/leaderboard/team_code/rl_trainer.py`
- Policy network: `simulation/leaderboard/team_code/rl_policy.py`
- Buffer: `simulation/leaderboard/team_code/rl_buffer.py`
- CBF filter: `simulation/leaderboard/team_code/cbf_filter.py`
- Evaluator (route loop, best checkpoint, flush): `simulation/leaderboard/leaderboard/leaderboard_evaluator_rl.py`
- Launcher script: `scripts/train_rl_e2e.sh`
- Main config: `simulation/leaderboard/team_code/agent_config/pnp_config_rl_cbf.yaml`

## 3) Runtime cadence: CARLA ticks vs decision steps

Important distinction:

- CARLA world still ticks every frame.
- Agent step function still executes every frame.
- New control (and RL transition collection) happens only every `skip_frames`.
- Skipped frames reuse previous control.

Current behavior:

- `skip_frames` is read from config in infer.
- `pnp_agent_e2e.py` returns `self.infer.prev_control` on skipped frames.
- Therefore RL `step(...)` is called on decision frames, not every CARLA frame.

This means `n_steps` counts decision frames (data points stored in the rollout buffer), not raw CARLA frames.

## 4) RL inputs and outputs

### Observation (11D)

Built in `pnp_infer_action_e2e.py` from ego + nearby actors:

```
[v_ego, dv,
 veh_min_dist, veh_closing_speed, veh_heading_alignment,
 ped_min_dist, ped_closing_speed, ped_heading_alignment,
 bike_min_dist, bike_closing_speed, bike_heading_alignment]
```

- `v_ego`: ego speed magnitude.
- `dv`: `desired_speed - v_ego`.
- Per type, only nearest actor stats are used in the observation.

### Action (3D)

Policy output:

```
[gamma_vehicle, gamma_ped, gamma_bike]
```

Action scaling from policy space `[-1, 1]`:

```
gamma = (a + 1) / 2 * (gmax - gmin) + gmin
```

### Evaluation-only mode (frozen RL)

RL can be switched to inference-only without training updates:

- `rl.eval_only=true`
- `rl.deterministic_eval=true` (recommended for reproducible benchmarks)
- `rl.resume_path=<checkpoint>` (required in eval-only mode)

Behavior in eval-only mode:

- Policy still outputs adaptive gammas from the current observation.
- No rollout-buffer writes, no PPO updates, no pending-buffer flush update.
- Evaluator keeps route/global metrics output (`results.json`) but skips RL best-checkpoint writes.
- If `resume_path` is missing/invalid, initialization fails fast.

### Reward

- Per-step reward: route progress delta (nonnegative clamp).
- Terminal reward: `+1` when route completed.
- If route progress is unavailable in a step payload, that step reward is `0.0`.

### Cost

- Per-step cost: dense risk from proximity + TTC.
- Terminal cost: `cost_terminal_collision` when evaluator reports collision for route end.

## 5) Dense cost formulation

Actor scope:

- Dynamic actors in RL safety classes: vehicle, pedestrian, bicycle.
- Actors farther than `d_safe` are ignored.
- Same actor list is shared across RL obs, dense-cost computation, and CBF call for that decision frame.

Per actor `i`:

```
u_i       = (p_actor - p_ego) / ||p_actor - p_ego||
v_rel_i   = v_actor - v_ego
closing_i = max(0, -dot(v_rel_i, u_i))
```

Proximity:

```
c_prox_i = 0                                 if dist_i >= d_safe
c_prox_i = 1                                 if dist_i <= d_collision
c_prox_i = (d_safe - dist_i)/(d_safe - d_collision) otherwise
```

Guard:

- If `d_safe <= d_collision`, fallback to threshold form to avoid divide-by-zero.

TTC:

```
ttc_i   = dist_i / (closing_i + eps)
c_ttc_i = max(0, 1 - ttc_i / ttc_threshold)  if closing_i > 0 and ttc_i < ttc_threshold
c_ttc_i = 0                                  otherwise
```

Per-actor risk and step cost:

```
risk_i = max(c_prox_i, c_ttc_i)
c_t    = max_i(risk_i)
```

Blend mode:

- Config key exists: `cost_blend_mode`.
- Only `"max"` is implemented right now.
- Unsupported values fallback to `"max"` with a warning.

## 6) Policy and optimizer details

Policy network (`rl_policy.py`):

- Actor MLP: `(obs_dim -> 64 -> 64 -> action_dim)` with `Tanh`.
- Critic MLP: `(obs_dim -> 64 -> 64 -> 1)` with `Tanh`.
- Gaussian policy with learned `logstd` parameter.

Trainer (`rl_trainer.py`):

- Optimizer: Adam (`learning_rate`, `eps=1e-5`).
- Gradient clipping: `max_grad_norm`.
- PPO clipped objective over `update_epochs` and `num_minibatches`.

## 7) PPO/CRPO math

Reward side:

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * gae_lambda * (1 - done_t) * A_{t+1}
R_t     = A_t + V(s_t)
```

Reward advantages are normalized:

```
A <- (A - mean(A)) / (std(A) + eps)
```

Cost side:

```
C_t = c_t + gamma * (1 - done_t) * C_{t+1}
```

Cost returns are used raw (no mean-centering/std normalization).

PPO losses:

```
ratio    = exp(logpi_new - logpi_old)
L_reward = -E[min(ratio*A, clip(ratio,1-e,1+e)*A)]
L_cost   =  E[max(ratio*C, clip(ratio,1-e,1+e)*C)]
L_value  = 0.5 * E[(V - R)^2]
L_total  = L_reward + lambda_mult*L_cost - ent_coef*Entropy + vf_coef*L_value
```

Why `max` for cost clipping:

- `C` is raw discounted cost return (typically nonnegative here).
- We minimize `L_total`; increasing `L_cost` strengthens safety pressure.
- Using `max(...)` is conservative (upper-bound style) for cost and avoids under-penalizing risky updates.
- Using `min(...)` would be optimistic (lower-bound style) and can make constraints easier to satisfy.

## 8) Lagrangian update

Constraint error:

```
avg_cost = mean(c_t)
err      = avg_cost - cost_limit
```

PID-style multiplier update:

- Integral stores raw accumulated error.
- `ki` is applied once in final combination.

Conceptual form:

```
p_int <- max(0, p_int + err)
lambda_mult <- clip(kp*ema_p + ki*p_int + kd*p_der, [0, lagrange_mult_max])
```

## 9) Buffer lifecycle and route boundaries

- `step(...)` appends transition from previous action and emits next action.
- Buffer updates when `size >= n_steps`.
- Route end calls `finish_episode(...)`:
  - terminal reward bonus for completion,
  - terminal collision cost if evaluator says collided.
- Buffer is not reset at route boundaries.
- If run ends with remaining samples, evaluator calls `flush_pending()`.

Collision signal source:

- Evaluator computes collisions from route infraction stats.
- It passes `completed` and `collided` via `agent_instance.on_route_end(...)`.
- Infer forwards this to trainer `finish_episode(...)`.

## 10) Run command and script arguments

Primary launcher:

```
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rl_e2e.sh 24 40002 codriving 7 rl_cbf _2 10 1 331 1
```

`train_rl_e2e.sh` args:

1. `route_id_start`
2. `carla_port`
3. `exp_name`
4. `run_id` (mapped to `results/rl_runs/runXXX`)
5. `agent_config_name` (loads `pnp_config_<name>.yaml`)
6. `scenario_parameter_suffix` (loads `scenario_parameter<suffix>.yaml`)
7. `num_routes`
8. `route_passes`
9. `route_id_max`
10. `reuse_agent` (`1` reuse, `0` recreate per route)
11. `resume_mode` (`1` resume route progress + RL state, `0` fresh run)
12. `rl_resume_ckpt` (optional path override, used when `resume_mode=1`)

`route_passes` semantics:

- Evaluator treats this as exact pass count.
- `0` or `1` means a single pass.
- `2` means exactly two passes.

Resume behavior for route-id loop:

- With `resume_mode=1` (arg 11), evaluator resumes from the same run checkpoint
  (`results/rl_runs/runXXX/eval/results.json`) and continues from the next
  route in the same pass.
- It runs only the remaining routes in that pass until `num_routes` is reached.
- If interruption happened mid-route, that route restarts from route start.

RL checkpoint source when resuming:

- By default, `resume_mode=1` loads:
  - `results/rl_runs/runXXX/rl/checkpoints/ckpt_latest.pt`
- You can override it with arg 12 (`rl_resume_ckpt`).
- This override is applied via env `RL_RESUME_PATH` and takes priority over
  YAML `rl.resume_path`.

Examples:

Fresh run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rl_e2e.sh 24 40002 codriving 7 rl_cbf _2 105 1 331 1 0
```

Continue interrupted run (same `run_id`):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rl_e2e.sh 24 40002 codriving 7 rl_cbf _2 105 1 331 1 1
```

Continue with explicit RL checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rl_e2e.sh 24 40002 codriving 7 rl_cbf _2 105 1 331 1 1 \
  results/rl_runs/run007/rl/checkpoints/ckpt_update_50.pt
```

Continue and extend total passes later (example: previously `route_passes=1`, now continue to `3`):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rl_e2e.sh 24 40002 codriving 7 rl_cbf _2 105 3 331 1 1
```

Operational notes:

- There is no `--force` flag in `train_rl_e2e.sh`.
- For clean fresh experiments, use a new `run_id` instead of reusing an old run with `resume_mode=0`.
- `resume_mode=1` needs both checkpoints from the same run root:
  - evaluator progress: `results/rl_runs/runXXX/eval/results.json`
  - RL policy state: `results/rl_runs/runXXX/rl/checkpoints/ckpt_latest.pt` (or arg12 override)
- If interruption happened mid-route, resume restarts that route from route start.

For frozen-policy evaluation runs, keep the same launcher but set in YAML:

- `rl.eval_only=true`
- `rl.deterministic_eval=true`
- `rl.resume_path=<path_to_ckpt_best_or_latest.pt>`

## 11) Outputs and file layout

Run root:

```
results/rl_runs/runXXX/
  eval/
    results.json
    ego_vehicle_0/results.json
  images/
    passXX_town05_short_rYY_<timestamp>_<microsec>/
      ego_vehicle_0/*.jpg
      ego_vehicle_0/*.json
  rl/
    train_log.csv
    checkpoints/
      ckpt_update_*.pt
      ckpt_latest.pt
      ckpt_best.pt
      ckpt_best_meta.json
```

Notes:

- `RL_LOG_DIR` is injected by `train_rl_e2e.sh` and overrides YAML log dir for run isolation.
- Best checkpoint saving cadence is evaluator arg `--best-every` (default `10`).

## 12) Key config fields (`pnp_config_rl_cbf.yaml`)

Core:

- `rl.enabled`
- `rl.n_steps`
- `rl.update_epochs`
- `rl.num_minibatches`
- `rl.cost_limit`
- `rl.lagrange_mult`
- `rl.pid_gains`
- `rl.eval_only`
- `rl.deterministic_eval`
- `rl.resume_path`

Dense cost:

- `rl.cost_distance` (`d_collision`)
- `rl.cost_d_safe`
- `rl.cost_ttc_threshold`
- `rl.cost_terminal_collision`
- `rl.cost_blend_mode` (`"max"` currently active)
- `rl.cost_w_prox`, `rl.cost_w_ttc` (reserved for future weighted mode)

Logging:

- `rl.log_scale` (readability only)

Safety bypass:

- `cbf.enabled=false`: RL/CBF path bypassed, nominal controller output used.
- `rl.enabled=false` with `cbf.enabled=true`: CBF runs with fixed YAML gammas.

## 13) Reading `train_log.csv`

Columns:

- Raw: `avg_reward`, `avg_cost`, `avg_return`, `avg_cost_return`
- Scaled: `avg_reward_scaled`, `avg_cost_scaled`, `avg_return_scaled`, `avg_cost_return_scaled`
- Multiplier: `lagrange_mult`
- Action means: `mean_action_0..2`

Interpretation:

- Small raw rewards are expected (progress deltas).
- `avg_cost` vs `cost_limit` drives multiplier adaptation.
- If `lagrange_mult` stays near zero for many updates, safety constraint pressure is weak.
- Use eval JSON metrics (driving score, completion, collisions) as primary outcome metrics.

## 14) Reproducibility checklist

- Record run command and config snapshot.
- Use a fresh `runXXX` to avoid mixed CSV schemas.
- Confirm `train_log.csv` header includes scaled columns (new trainer format).
- Track both training metrics and evaluator metrics for conclusions.
- Prefer long horizons (>=100k decision steps) before judging convergence.

## 15) Paper evaluation pipeline (RL-CBF vs baselines)

This repo now includes a reproducible evaluation + plotting pipeline for paper tables/figures.

### 15.1 Run-matrix configs

- Main benchmark matrix: `experiments/paper_eval_town05_main.yaml`
  - protocol: `town05_all_scenarios_2.json`, `105 routes x 3 passes`
  - methods: `rl_cbf_adaptive`, `fixed_cbf`, `pid_only`
- Checkpoint-stage matrix: `experiments/paper_eval_ckpt_stages.yaml`
  - protocol: `105 routes x 1 pass`
  - methods are stage-tagged checkpoints (for sample-efficiency curves)

### 15.2 Orchestrate evaluations

Dry-run first (prints resolved commands, writes no CARLA outputs):

```bash
bash scripts/run_paper_eval.sh --config experiments/paper_eval_town05_main.yaml --dry-run
```

Run all methods:

```bash
bash scripts/run_paper_eval.sh --config experiments/paper_eval_town05_main.yaml
```

Run one method only:

```bash
bash scripts/run_paper_eval.sh --config experiments/paper_eval_town05_main.yaml --method rl_cbf_adaptive
```

Start from pass 2 (skip pass 1):

```bash
bash scripts/run_paper_eval.sh --config experiments/paper_eval_town05_main.yaml --start-pass 2
```

Continue interrupted pass 2:

```bash
bash scripts/run_paper_eval.sh --config experiments/paper_eval_town05_main.yaml --start-pass 2 --continue
```

Notes:

- Results are stored deterministically under:
  - `results/paper_eval/<experiment>/<method>/passXX/...`
- A run manifest is written to:
  - `results/paper_eval/<experiment>/run_manifest.json`
- Each method gets an effective config snapshot:
  - `results/paper_eval/<experiment>/<method>/manifest/agent_config_eval.yaml`

### 15.3 Aggregate CSV tables

From manifest (recommended):

```bash
python tools/paper/aggregate_eval.py \
  --manifest results/paper_eval/town05_main/run_manifest.json \
  --output-dir paper_outputs/main
```

Fallback from directory discovery:

```bash
python tools/paper/aggregate_eval.py \
  --runs-root results/paper_eval/town05_main \
  --output-dir paper_outputs/main
```

Generated tables:

- `paper_outputs/main/route_metrics.csv`
- `paper_outputs/main/method_summary.csv`
- `paper_outputs/main/paired_deltas.csv`

### 15.4 Generate figures

```bash
python tools/paper/plot_eval.py \
  --summary-csv paper_outputs/main/method_summary.csv \
  --paired-csv paper_outputs/main/paired_deltas.csv \
  --output-dir paper_outputs/main/figures
```

Generated figures (PNG and PDF):

- `overall_scores`
- `overall_collisions_per_km`
- `infraction_breakdown`
- `paired_delta_hist_<baseline_method>`
- `checkpoint_stage_curve` (only when stage-tagged methods are present)

### 15.5 Metric conventions

- Paired deltas are computed as:
  - `delta = reference_method - compare_method`
- Collision rate uses:
  - `collisions_per_km = total_collisions / traveled_km`
- 95% CI uses normal approximation:
  - `1.96 * std / sqrt(n)`

### 15.6 Deterministic RL evaluation mode

In `rl_cbf_adaptive`, evaluation uses frozen policy inference:

- `rl.enabled=true`
- `rl.eval_only=true`
- `rl.deterministic_eval=true`
- `rl.resume_path=<trained_checkpoint>`

This avoids optimizer/buffer updates during paper benchmarking.
