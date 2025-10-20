from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class EgoState:
    pos: np.ndarray
    heading: float
    speed: float
    vel_xy: np.ndarray


@dataclass
class ObstacleState:
    pos: np.ndarray
    yaw: float
    vel_xy: np.ndarray
    size: Tuple[float, float]


class FixedCBFLayer:
    """
    Projects nominal PID commands onto a safe control set defined by
    velocity-aware distance barriers.
    """

    @dataclass
    class Constraint:
        h: float
        Lf: float
        Lg: np.ndarray

    def __init__(self, control_cfg: Dict):
        self.gamma = control_cfg.get('cbf_gamma', 1.0)
        self.d_safe = control_cfg.get('cbf_d_safe', 2.5)
        self.a_brake = control_cfg.get('cbf_a_brake', 3.5)
        self.v_clip = control_cfg.get('cbf_v_clip', 12.0)
        self.delta_max = control_cfg.get('steer_max', 0.4)
        self.k_throttle = control_cfg.get('a_drive', 3.0)
        self.k_brake = control_cfg.get('a_brake', 3.5)
        self.wheelbase = control_cfg.get('wheelbase', 2.8)
        self.w_slack = control_cfg.get('cbf_w_slack', 1000.0)


    def project(
        self,
        nominal_cmd: Dict[str, float],
        route_info: Dict,
    ) -> Tuple[float, float, float, Dict]:
        """
        Clamp nominal (steer, throttle, brake) via QP-based CBF projection.
        """
        ego = self._parse_ego(route_info)
        obstacles = self._parse_obstacles(route_info)

        a_nom = self._nominal_acceleration(nominal_cmd['throttle'], nominal_cmd['brake'])
        delta_nom = self.delta_max * float(np.clip(nominal_cmd['steer'], -1.0, 1.0))

        constraints = [
            self._build_constraint(ego, obs)
            for obs in obstacles
        ]
        constraints = [c for c in constraints if c is not None]

        if not constraints:
            return (
                nominal_cmd['steer'],
                nominal_cmd['throttle'],
                nominal_cmd['brake'],
                {'min_h': None, 'active': 0, 'status': 'inactive'},
            )

        P, q, A, l = self._assemble_qp(a_nom, delta_nom, constraints)
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A, l=l, u=np.full_like(l, np.inf), verbose=False, polish=True)
        qp_res = solver.solve()

        if qp_res.info.status != 'solved':
            return (
                nominal_cmd['steer'],
                nominal_cmd['throttle'],
                nominal_cmd['brake'],
                {'status': qp_res.info.status, 'active': len(constraints)},
            )

        a_star, delta_star = qp_res.x[:2]
        steer_safe, throttle_safe, brake_safe = self._inverse_actuation(a_star, delta_star)
        diag = {
            'status': qp_res.info.status,
            'active': len(constraints),
            'min_h': float(min(c.h for c in constraints)),
            'slack': float(qp_res.x[-1]),
        }
        return steer_safe, throttle_safe, brake_safe, diag

    # Helper routines
    
    def _parse_ego(self, route_info: Dict) -> EgoState:
        pos = np.asarray(route_info.get('ego_pose', (0.0, 0.0)), dtype=np.float32)
        heading = float(route_info.get('ego_heading', 0.0))
        speed = float(route_info.get('ego_speed_f', 0.0))
        vel_xy = np.asarray(route_info.get('ego_vel_xy', (0.0, 0.0)), dtype=np.float32)
        return EgoState(pos=pos, heading=heading, speed=speed, vel_xy=vel_xy)

    def _parse_obstacles(self, route_info: Dict) -> List[ObstacleState]:
        obstacles = []
        for obs in route_info.get('obstacles', []):
            pos = np.asarray(obs.get('pos', (0.0, 0.0)), dtype=np.float32)
            yaw = float(obs.get('yaw', 0.0))
            vel_xy = np.asarray(obs.get('vel_xy', (0.0, 0.0)), dtype=np.float32)
            size = obs.get('size', (2.0, 1.0))
            obstacles.append(ObstacleState(pos=pos, yaw=yaw, vel_xy=vel_xy, size=size))
        return obstacles

    def _nominal_acceleration(self, throttle: float, brake: float) -> float:
        return (
            self.k_throttle * float(np.clip(throttle, 0.0, 1.0))
            - self.k_brake * float(np.clip(brake, 0.0, 1.0))
        )

    def _inverse_actuation(self, a: float, delta: float) -> Tuple[float, float, float]:
        steer = np.clip(delta / self.delta_max, -1.0, 1.0)
        if a >= 0:
            throttle = np.clip(a / self.k_throttle, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-a / self.k_brake, 0.0, 1.0)
        return steer, throttle, brake

    def _build_constraint(self, ego: EgoState, obs: ObstacleState) -> Optional[Constraint]:
        delta_p = ego.pos - obs.pos
        dist = float(np.linalg.norm(delta_p))
        if dist <= 1e-3:
            return None

        c = delta_p / dist
        delta_v = ego.vel_xy - obs.vel_xy
        s = max(0.0, -float(np.dot(delta_v, c)))

        a_max = max(1e-3, self.a_brake * (1.0 - ego.speed / self.v_clip))
        h = dist - (s ** 2) / (2.0 * a_max) - self.d_safe

        dh_dp = c - (s / a_max) * (-delta_v / dist + (s / (dist ** 2)) * delta_p)
        dh_dve = -(s / a_max) * c
        dh_dpsi = -(s / a_max) * np.dot(
            c, ego.speed * np.array([-math.sin(ego.heading), math.cos(ego.heading)], dtype=np.float32)
        )

        p_dot = ego.speed * np.array([math.cos(ego.heading), math.sin(ego.heading)], dtype=np.float32)
        Lf = float(np.dot(dh_dp, p_dot))

        Lg = np.zeros(2, dtype=np.float32)
        Lg[0] = dh_dve @ np.array([1.0, 0.0])
        Lg[1] = dh_dpsi * (ego.speed / self.wheelbase)
        print("CBF h: %.2f, Lf: %.2f, Lg: [%.2f, %.2f]" % (h, Lf, Lg[0], Lg[1]))
        return FixedCBFLayer.Constraint(h=h, Lf=Lf, Lg=Lg)

    def _assemble_qp(
        self,
        a_nom: float,
        delta_nom: float,
        constraints: List[Constraint],
    ) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray]:
        P = sp.diags([2.0, 2.0, 2.0 * self.w_slack])
        q = np.array([-2.0 * a_nom, -2.0 * delta_nom, 0.0])

        rows = []
        lower = []
        for c in constraints:
            row = np.zeros(3, dtype=np.float64)
            row[:2] = c.Lg
            row[2] = 1.0  # slack
            rows.append(row)
            lower.append(-c.Lf - self.gamma * c.h)

        rows.append(np.array([0.0, 0.0, -1.0]))  # slack >= 0
        lower.append(0.0)
        
        A = sp.csc_matrix(np.vstack(rows))
        l = np.array(lower, dtype=np.float64)
        return P, q, A, l
