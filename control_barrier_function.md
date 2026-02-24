# Control Barrier Function (CBF) in V2Xverse-ASRL

This document explains the current CBF implementation in this repo, with direct correspondence to code.

## 1) Where CBF is used in the control loop

Main integration: `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`

Per decision step:

1. `V2X_Controller` computes nominal `steer`, `desired_speed`.
2. CBF is called: `cbf_filter.step(ego, actors, steer, desired_speed, gammas=...)`.
3. CBF returns:
   - same steer (longitudinal-only filter),
   - filtered safe desired speed,
   - `cbf_info` debug/status dict.
4. Throttle/brake are recomputed from safe desired speed.

If CBF is disabled, nominal output is used directly.

## 2) Implementation files

- CBF implementation: `simulation/leaderboard/team_code/cbf_filter.py`
- CBF integration: `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`
- CBF config: `simulation/leaderboard/team_code/agent_config/pnp_config_rl_cbf.yaml`
- RL-gamma provider: `simulation/leaderboard/team_code/rl_trainer.py` + policy

## 3) Problem solved each step (QP)

The filter solves a quadratic program over one longitudinal acceleration plus slack variables.

Decision variables:

```
x = [a, s_1, s_2, ..., s_N]^T
```

- `a`: longitudinal acceleration command
- `s_i >= 0`: slack for actor-specific barrier constraint `i`

Objective (code-equivalent form):

```
min_x (a - a_nom)^2 + slack_weight * sum_i s_i^2
```

where:

```
a_nom = (desired_speed - ego_speed) / dt
```

This keeps CBF close to nominal behavior while allowing soft violations through penalized slack.

## 4) Actor set and preprocessing

Actors passed to CBF come from:

```
world.get_actors().filter("*vehicle*") + world.get_actors().filter("*walker*")
```

Then CBF itself filters:

- skip `None`, dead actors, ego actor
- skip actors farther than `max_distance`
- skip near-zero-distance numeric edge case

For each retained actor:

- `p_rel = p_actor - p_ego` (2D)
- `d = ||p_rel||`
- `u = p_rel / (d + eps)` (line-of-sight unit vector)
- `v_rel = v_actor - v_ego` (2D)
- `v_rel_line = dot(u, v_rel)`
- `closing_speed = max(0, -v_rel_line)`

## 5) Barrier function used in code

Actor and ego use a circular collision radius approximation from bounding box extents:

```
r = hypot(extent_x, extent_y)
```

Safety distance:

```
d_safe_geom = r_ego + r_actor + margin
```

Barrier value:

```
h = d - d_safe_geom - (closing_speed^2) / (2 * a_brake)
```

Interpretation:

- `h >= 0`: outside the braking-based safety boundary
- `h < 0`: inside safety boundary (already risky)

The term `(closing_speed^2)/(2*a_brake)` is the stopping-distance approximation.

## 6) Per-actor CBF inequality in this implementation

For actor `i`, code builds:

```
a_coeff_i = -0.5 * heading_alignment_i * dt^2
b_i       = -gamma_i * h_i - v_rel_line_i * dt
```

with:

```
heading_alignment_i = dot(ego_heading_unit, u_i)
```

The constrained form is:

```
a_coeff_i * a + s_i >= b_i
s_i >= 0
```

and global acceleration bounds:

```
a_min <= a <= a_max
```

Notes:

- `gamma_i` is actor-type dependent (`vehicle`, `ped`, `bike`) and can be overridden each step by RL.
- Steering is not optimized here; only longitudinal acceleration is optimized.

## 7) Matrix form used by OSQP

For `N` active actor constraints:

- variables: `1 + N`
- rows: `1 + N + N`
  - row 0: acceleration bound variable row
  - next `N`: slack nonnegativity rows
  - last `N`: CBF rows

Quadratic term:

```
P = diag([2, 2*slack_weight, ..., 2*slack_weight])
q = [-2*a_nom, 0, ..., 0]
```

This is exactly the expanded quadratic for `(a-a_nom)^2 + slack_weight*sum s_i^2` up to constants.

## 8) Solver and fallback behavior

Solver: OSQP (`osqp` + `scipy.sparse`).

Status handling:

- if disabled: `status = "disabled"`, pass-through nominal
- if solver deps missing: `status = "missing_solver"`, pass-through nominal
- if ego missing: `status = "no_ego"`, pass-through nominal
- if bad `dt`: `status = "bad_dt"`, pass-through nominal
- if no valid actor constraints: `status = "no_constraints"`, pass-through nominal
- if QP fails (`res.x is None` or bad status): `status = "qp_fail"`, pass-through nominal
- if success: use solved `a` and report OSQP status string

Returned debug info always includes:

- `status`
- `min_barrier_value` (when constraints were evaluated)

## 9) Conversion back to speed and actuation

After solving:

```
v_safe = max(0, ego_speed + a_safe * dt)
```

Then controller recomputes throttle/brake from `(current_speed, v_safe)`.

So CBF output is a safe desired speed, not direct throttle/brake.

## 10) RL interaction with CBF

When RL is enabled:

- RL outputs three gammas each decision step:
  - `gamma_vehicle`
  - `gamma_ped`
  - `gamma_bike`
- CBF receives these via `gammas=...` and overrides internal gamma values for that step.

When RL is disabled:

- CBF uses static gammas from YAML.

Important:

- RL does not control steering here.
- RL affects safety aggressiveness through gamma, while CBF still enforces acceleration bounds and QP structure.

## 11) Parameter guide (mapped to implementation)

From `pnp_config_rl_cbf.yaml -> cbf`:

- `enabled`: master CBF switch
- `max_distance`: actor cutoff radius for constraints
- `margin`: geometric safety buffer
- `a_brake`: braking capability assumption in barrier math
- `a_min`: minimum allowed acceleration (most negative braking)
- `a_max`: maximum allowed acceleration
- `slack_weight`: penalty for violating CBF constraints
- `gamma_vehicle`, `gamma_ped`, `gamma_bike`: per-type barrier gains (used if RL not overriding)

Practical consistency rule:

- Keep `|a_min| >= a_brake` to avoid contradictory assumptions (QP should be able to command braking at least as strong as barrier model assumes).

## 12) Worked numeric example

Assume:

- `d = 5.0 m`
- `closing_speed = 3.0 m/s`
- `r_ego = 1.5 m`
- `r_actor = 1.0 m`
- `margin = 1.0 m`
- `a_brake = 6.0 m/s^2`

Then:

```
d_safe_geom = 1.5 + 1.0 + 1.0 = 3.5
stop_term   = 3^2 / (2*6) = 0.75
h           = 5.0 - 3.5 - 0.75 = 0.75  (safe side)
```

If distance becomes `3.8`:

```
h = 3.8 - 3.5 - 0.75 = -0.45  (unsafe side)
```

Negative `h` increases the required corrective acceleration through the RHS term.

## 13) Known modeling limits

Current CBF is intentionally simple:

- longitudinal-only (no steering optimization)
- circle approximation for actor footprint
- constant `a_brake` model in stop-distance term
- no explicit uncertainty model
- no multi-step trajectory optimization in the QP

These trade complexity for runtime robustness and easy integration with learned gamma modulation.

## 14) Quick diagnostics checklist

When behavior looks wrong, check:

1. `cbf_info.status` in saved planning JSON (`no_constraints`, `qp_fail`, etc.)
2. `min_barrier_value` trend (frequent large negatives indicate late intervention)
3. Config consistency: `a_brake`, `a_min`, `margin`, `max_distance`
4. Whether RL is overriding gammas (presence of `route_info['gammas']` entries)

## 15) Summary

CBF in this repo is a per-step convex QP over longitudinal acceleration with soft actor-specific barrier constraints.  
RL adapts per-type barrier gains online; CBF remains the final safety filter that converts those gains into bounded acceleration and safe desired speed.
