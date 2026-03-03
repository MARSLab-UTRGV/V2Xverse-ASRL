# Control Barrier Function (CBF) in V2Xverse-ASRL

This document describes the current CBF runtime behavior with direct mapping to code.

## 1) Where CBF is used in the loop

Main integration: `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`.

Per decision step:

1. Low-level controller (PID or pure pursuit lateral + shared longitudinal PID) computes nominal `steer`, `desired_speed`.
2. CBF is called:
   - `cbf_filter.step(ego, actors, steer, desired_speed, gammas=...)`
3. CBF returns:
   - unchanged steer (CBF is longitudinal-only),
   - filtered `desired_speed_safe`,
   - `cbf_info` diagnostics.
4. Throttle/brake are recomputed from `(speed, desired_speed_safe)`.

If `cbf.enabled=false`, nominal output passes through unchanged.

## 2) Code locations

- CBF core: `simulation/leaderboard/team_code/cbf_filter.py`
- CBF call site: `simulation/leaderboard/team_code/pnp_infer_action_e2e.py`
- Config: `simulation/leaderboard/team_code/agent_config/pnp_config_rl_cbf.yaml`

## 3) Decision variables and objective (OSQP)

QP variables:

```
x = [a, s_1, s_2, ..., s_N]^T
```

- `a`: ego longitudinal acceleration
- `s_i >= 0`: slack for actor `i` constraint

Objective:

```
min (a - a_nom)^2 + slack_weight * sum_i s_i^2
```

with:

```
a_nom = (desired_speed_nom - ego_speed) / dt
```

So CBF tries to stay close to nominal acceleration, but can violate hard actor constraints using penalized slack.

## 4) Time step used by CBF

`dt` is taken from CBF config (`cbf.dt`) via `CBFQPFilter._get_dt()`.

Recommended setting:

```
cbf.dt ~= simulation.skip_frames * world.fixed_delta_seconds
```

Example:
- world fixed delta = `0.05s`
- `skip_frames=4`
- CBF decision interval is about `0.2s` -> set `cbf.dt: 0.2`

## 5) Actor source, classification, and preprocessing

Actors passed in from integration:

```
world.get_actors().filter("*vehicle*") + world.get_actors().filter("*walker*")
```

Inside CBF:

- skip ego, dead actors, far actors (`> max_distance`)
- actor type for gamma/diagnostics:
  - walker -> `ped`
  - bicycle/diamondback -> `bike`
  - otherwise -> `vehicle`

Base kinematics per actor:

- `p_rel = p_actor - p_ego` (2D)
- `d = ||p_rel||`
- `u = p_rel / (d + eps)`
- `v_rel = v_actor - v_ego`
- `v_rel_line = dot(u, v_rel)`
- `closing_speed = max(0, -v_rel_line)`

## 6) Path-aware actor gating (important)

CBF now filters actors before constraints to reduce false stops from adjacent/opposite lanes.

Computed in ego frame:

- `longitudinal_offset = dot(p_rel, ego_heading)`
- `lateral_offset = dot(p_rel, ego_right)`
- `forward_velocity = dot(v_rel, ego_heading)`
- `lateral_velocity = dot(v_rel, ego_right)`

Rules:

1. **Rear filter**
   - drop actor if `longitudinal_offset < -gate_rear_distance`

2. **Lateral corridor**
   - `lateral_gate = gate_lateral_distance + gate_lateral_speed_gain * ego_speed`
   - if `|lateral_offset| <= lateral_gate`, keep actor

3. **Crossing exception (for far-lateral actors)**
   - only if `gate_enable_crossing=true`
   - actor must move toward path: `lateral_offset * lateral_velocity < 0`
   - lateral TTC:

     ```
     ttc_lat = |lateral_offset| / (|lateral_velocity| + eps)
     ```

   - require `ttc_lat <= gate_crossing_ttc`
   - predict longitudinal crossing point:

     ```
     longitudinal_at_cross = longitudinal_offset + forward_velocity * ttc_lat
     ```

   - keep if:

     ```
     -gate_crossing_rear <= longitudinal_at_cross <= gate_crossing_forward
     ```

This keeps side actors that are truly crossing soon, while rejecting far lateral non-interfering traffic.

## 7) Barrier function

Circle approximation:

```
r = hypot(extent_x, extent_y)
```

Safety distance:

```
d_safe_geom = r_ego + r_actor + margin
```

Barrier:

```
h = d - d_safe_geom - closing_speed^2 / (2 * a_brake)
```

Interpretation:
- `h >= 0`: safer side
- `h < 0`: unsafe side

## 8) Per-actor inequality and gamma effect

Per actor `i`:

```
heading_alignment_i = dot(ego_heading, u_i)
a_coeff_i = -0.5 * heading_alignment_i * dt^2
rhs_i     = -gamma_i * h_i - v_rel_line_i * dt
```

Constraint:

```
a_coeff_i * a + s_i >= rhs_i
s_i >= 0
```

Global bound:

```
a_min <= a <= a_max
```

Gamma behavior:
- if `h_i < 0` (unsafe), higher gamma increases `rhs_i` -> tighter braking pressure
- if `h_i > 0`, higher gamma can relax pressure

## 9) OSQP matrix form

For `N` active actor constraints:

- variables: `1 + N`
- rows: `1 + N + N`
  - accel bounds row
  - `N` slack nonnegativity rows
  - `N` CBF rows

Quadratic and linear terms:

```
P = diag([2, 2*slack_weight, ..., 2*slack_weight])
q = [-2*a_nom, 0, ..., 0]
```

## 10) Fallback/status behavior

Status values:

- `disabled`
- `missing_solver`
- `no_ego`
- `bad_dt`
- `no_constraints`
- `qp_fail`
- OSQP solved status string (e.g., `solved`)

On failure or no constraints, CBF returns nominal desired speed.

## 11) `cbf_info` diagnostics (per step)

Always/commonly reported:

- `status`
- `min_barrier_value`
- `num_constraints`
- `num_considered_actors`
- `num_filtered_behind`
- `num_filtered_lateral`
- `num_crossing_kept`

Worst barrier actor fields (`min_barrier_actor_*`):

- `id`, `type`, `dist`, `closing_speed`, `heading_alignment`
- `gamma`, `rhs`, `a_coeff`

Tightest nominal-pressure actor fields (`tightest_actor_*`):

- same metadata plus:
- `tightest_actor_nominal_residual = rhs - a_coeff * a_nom`

Interpretation:
- `min_barrier_actor`: geometrically most unsafe actor (`min h`)
- `tightest_actor`: actor that most strongly conflicts with nominal acceleration

## 12) Output conversion to controller

After QP:

```
desired_speed_safe = max(0, ego_speed + a_safe * dt)
```

Then longitudinal PID maps speed target to throttle/brake.

## 13) RL interaction

When RL is enabled, per-step gammas are injected:

- `gamma_vehicle`
- `gamma_ped`
- `gamma_bike`

When RL is disabled, static YAML gammas are used.

RL changes only gamma values; CBF structure (QP, bounds, gating) stays the same.

## 14) Config guide (CBF block)

Core:

- `enabled`
- `dt`
- `max_distance`
- `margin`
- `a_brake`
- `a_min`, `a_max`
- `slack_weight`
- `gamma_vehicle`, `gamma_ped`, `gamma_bike`

Path-gating:

- `actor_path_gating`
- `gate_rear_distance`
- `gate_lateral_distance`
- `gate_lateral_speed_gain`
- `gate_enable_crossing`
- `gate_crossing_ttc`
- `gate_crossing_forward`
- `gate_crossing_rear`

Practical consistency checks:

- keep `|a_min| >= a_brake` to avoid impossible braking assumptions
- align `cbf.dt` with actual control decision interval

## 15) Known limits

- longitudinal-only CBF (no steering optimization)
- circular footprint approximation
- no uncertainty model
- one-step QP, no multi-step trajectory optimization
- gating is geometric/kinematic, not full map/lane-topology semantics

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
