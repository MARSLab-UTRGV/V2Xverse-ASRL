import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - handled by runtime check
    osqp = None
    sp = None


class CBFQPFilter:
    """
    Collision-only control barrier filter using a small QP on longitudinal accel.

    The filter keeps steering from the nominal controller and adjusts desired
    speed via a constrained acceleration to avoid imminent collisions.
    """

    def __init__(self, config: Dict):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.max_distance = float(config.get("max_distance", 30.0))
        self.margin = float(config.get("margin", 1.5))
        self.a_brake = float(config.get("a_brake", 6.0))
        self.a_min = float(config.get("a_min", -6.0))
        self.a_max = float(config.get("a_max", 4.0))
        self.slack_weight = float(config.get("slack_weight", 1000.0))
        self.gamma_vehicle = float(config.get("gamma_vehicle", 0.5))
        self.gamma_ped = float(config.get("gamma_ped", 0.5))
        self.gamma_bike = float(config.get("gamma_bike", 0.5))
        self.debug = bool(config.get("debug", False))
        self._eps = 1e-4

    def _actor_gamma(self, actor) -> float:
        type_id = actor.type_id.lower()
        if "walker" in type_id:
            return self.gamma_ped
        if "diamondback" in type_id or "bicycle" in type_id:
            return self.gamma_bike
        return self.gamma_vehicle

    @staticmethod
    def _vec2(carla_vector) -> np.ndarray:
        return np.array([carla_vector.x, carla_vector.y], dtype=np.float32)

    def _get_dt(self, ego) -> float:
        try:
            dt = ego.get_world().get_settings().fixed_delta_seconds
            if dt is None or dt <= 0:
                return 0.05
            return float(dt)
        except Exception:
            return 0.05

    def step(
        self,
        ego,
        actors: List,
        steer_nom: float,
        desired_speed: float,
        gammas: Dict[str, float] = None,
    ):
        info = {"status": "disabled"}
        if not self.enabled:
            return steer_nom, desired_speed, info
        if osqp is None or sp is None:
            info["status"] = "missing_solver"
            return steer_nom, desired_speed, info
        if ego is None:
            info["status"] = "no_ego"
            return steer_nom, desired_speed, info

        # Optional per-step gamma override from RL policy
        if gammas is not None:
            self.gamma_vehicle = float(gammas.get("gamma_vehicle", self.gamma_vehicle))
            self.gamma_ped = float(gammas.get("gamma_ped", self.gamma_ped))
            self.gamma_bike = float(gammas.get("gamma_bike", self.gamma_bike))

        dt = self._get_dt(ego)
        ego_location = ego.get_location()
        ego_velocity = ego.get_velocity()
        ego_speed = float(np.linalg.norm(self._vec2(ego_velocity)))
        if dt <= 1e-3:
            info["status"] = "bad_dt"
            return steer_nom, desired_speed, info

        nominal_longitudinal_accel = (desired_speed - ego_speed) / dt
        ego_yaw_rad = math.radians(ego.get_transform().rotation.yaw)
        ego_heading_vector = np.array(
            [math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)], dtype=np.float32
        )

        ego_bounding_box = ego.bounding_box.extent
        ego_collision_radius = float(
            np.hypot(ego_bounding_box.x, ego_bounding_box.y)
        )

        cbf_constraints = []
        min_barrier_value = float("inf")
        for actor in actors:
            if actor is None or actor.id == ego.id:
                continue
            if not actor.is_alive:
                continue

            actor_location = actor.get_location()
            if actor_location.distance(ego_location) > self.max_distance:
                continue

            relative_position = self._vec2(actor_location) - self._vec2(ego_location)
            relative_distance = float(np.linalg.norm(relative_position))
            if relative_distance < self._eps:
                continue

            relative_direction = relative_position / (relative_distance + self._eps)
            relative_velocity = (
                self._vec2(actor.get_velocity()) - self._vec2(ego_velocity)
            )
            relative_speed_along_line = float(
                np.dot(relative_direction, relative_velocity)
            )

            closing_speed = max(0.0, -relative_speed_along_line)
            try:
                actor_bounding_box = actor.bounding_box.extent
                actor_collision_radius = float(
                    np.hypot(actor_bounding_box.x, actor_bounding_box.y)
                )
            except Exception:
                actor_collision_radius = 0.5

            safety_distance = (
                ego_collision_radius + actor_collision_radius + self.margin
            )
            barrier_now = (
                relative_distance
                - safety_distance
                - (closing_speed**2) / (2.0 * self.a_brake)
            )
            min_barrier_value = min(min_barrier_value, barrier_now)

            heading_alignment = float(np.dot(ego_heading_vector, relative_direction))
            accel_coefficient = -0.5 * heading_alignment * (dt**2)
            barrier_gain = self._actor_gamma(actor)
            constraint_rhs = (
                -barrier_gain * barrier_now - relative_speed_along_line * dt
            )

            cbf_constraints.append((accel_coefficient, constraint_rhs))

        if not cbf_constraints:
            info.update(
                {"status": "no_constraints", "min_barrier_value": min_barrier_value}
            )
            return steer_nom, desired_speed, info

        num_variables = 1 + len(cbf_constraints)  # accel + slacks
        num_rows = 1 + len(cbf_constraints) + len(cbf_constraints)

        # objective: (a - a_nom)^2 + slack_weight * sum(slack^2)
        quadratic_cost = sp.diags(
            [2.0] + [2.0 * self.slack_weight] * len(cbf_constraints)
        ).tocsc()
        linear_cost = np.zeros(num_variables, dtype=np.float32)
        linear_cost[0] = -2.0 * nominal_longitudinal_accel

        constraint_matrix = sp.lil_matrix((num_rows, num_variables), dtype=np.float32)
        lower_bounds = -np.inf * np.ones(num_rows, dtype=np.float32)
        upper_bounds = np.inf * np.ones(num_rows, dtype=np.float32)

        # accel bounds
        constraint_matrix[0, 0] = 1.0
        lower_bounds[0] = self.a_min
        upper_bounds[0] = self.a_max

        # slack >= 0
        for i in range(len(cbf_constraints)):
            row = 1 + i
            constraint_matrix[row, 1 + i] = 1.0
            lower_bounds[row] = 0.0

        # CBF constraints: a_coeff * a + slack_i >= b_val
        for i, (accel_coefficient, constraint_rhs) in enumerate(cbf_constraints):
            row = 1 + len(cbf_constraints) + i
            constraint_matrix[row, 0] = accel_coefficient
            constraint_matrix[row, 1 + i] = 1.0
            lower_bounds[row] = constraint_rhs

        prob = osqp.OSQP()
        prob.setup(
            P=quadratic_cost,
            q=linear_cost,
            A=constraint_matrix.tocsc(),
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            polish=True,
        )
        res = prob.solve()

        if res.x is None or res.info.status_val not in (1, 2):
            info.update({"status": "qp_fail", "min_barrier_value": min_barrier_value})
            return steer_nom, desired_speed, info

        safe_longitudinal_accel = float(res.x[0])
        desired_speed_safe = max(0.0, ego_speed + safe_longitudinal_accel * dt)
        info.update(
            {"status": res.info.status, "min_barrier_value": min_barrier_value}
        )
        return steer_nom, desired_speed_safe, info
