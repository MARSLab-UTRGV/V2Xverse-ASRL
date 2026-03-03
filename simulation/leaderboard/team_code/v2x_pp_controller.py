import math

import numpy as np

from team_code.v2x_controller import V2X_Controller


class V2X_PP_Controller(V2X_Controller):
    """
    Pure Pursuit lateral controller reusing the existing longitudinal PID logic.
    """

    def __init__(self, config):
        super().__init__(config)
        self.pp_wheelbase_m = float(config.get("pp_wheelbase_m", 2.8))
        self.pp_max_steer_angle_rad = float(config.get("pp_max_steer_angle_rad", 1.22))
        self.pp_lookahead_min_m = float(config.get("pp_lookahead_min_m", 1.8))
        self.pp_lookahead_gain_s = float(config.get("pp_lookahead_gain_s", 0.12))
        self.pp_lookahead_max_m = float(config.get("pp_lookahead_max_m", 3.5))
        self.pp_interp_step_m = float(config.get("pp_interp_step_m", 0.5))
        self.pp_steer_gain = float(config.get("pp_steer_gain", 1.0))
        self.pp_steer_sign = float(config.get("pp_steer_sign", 1.0))
        self.pp_low_speed_steer_zero_mps = float(
            config.get("pp_low_speed_steer_zero_mps", 0.01)
        )

    def _target_pid_fallback_steer(self, target, speed):
        aim = np.asarray(target, dtype=np.float32).reshape(-1)
        if aim.size < 2:
            return 0.0

        theta_tg = np.arctan2(float(aim[0]), float(aim[1]) + 1e-7)
        angle_tg = np.sign(theta_tg) * (180 - np.abs(np.degrees(theta_tg))) / 90.0
        if speed < 0.01:
            angle_tg = 0.0
        steer = self.turn_controller.step(angle_tg)
        return float(np.clip(steer, -1.0, 1.0))

    @staticmethod
    def _valid_waypoints(waypoints):
        wp = np.asarray(waypoints, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            return None
        if wp.shape[0] < 2:
            return None
        if not np.isfinite(wp).all():
            return None
        return wp

    @staticmethod
    def _valid_target(target):
        tgt = np.asarray(target, dtype=np.float32).reshape(-1)
        if tgt.size < 2:
            return None
        if not np.isfinite(tgt[:2]).all():
            return None
        return tgt[:2]

    def _compute_target_pp_steer(self, target_point, speed, lookahead):
        if speed < self.pp_low_speed_steer_zero_mps:
            return 0.0, 0.0, 0.0, 0.0

        x_l = float(target_point[0])
        y_l = float(target_point[1])
        raw_distance = float(np.hypot(x_l, y_l))
        if raw_distance < 1e-4:
            return 0.0, 0.0, 0.0, 0.0

        alpha = math.atan2(x_l, y_l + 1e-6)
        # Use a speed-adaptive minimum geometric distance to avoid over-aggressive
        # steering when the target is very close.
        distance = max(raw_distance, float(lookahead))

        kappa = 2.0 * math.sin(alpha) / distance
        delta = math.atan(self.pp_wheelbase_m * kappa)
        steer_norm = delta / max(1e-3, self.pp_max_steer_angle_rad)
        steer_raw = self.pp_steer_sign * self.pp_steer_gain * steer_norm
        steer = float(np.clip(steer_raw, -1.0, 1.0))
        return steer, alpha, float(steer_raw), raw_distance

    def run_step(self, route_info):
        speed = float(route_info["speed"])
        waypoints = self._valid_waypoints(route_info.get("waypoints", []))
        target = self._valid_target(route_info.get("target", [0.0, 0.0]))

        if speed < 0.2:
            self.stop_steps += 1
        else:
            self.stop_steps = max(0, self.stop_steps - 10)

        lookahead = float(
            np.clip(
                self.pp_lookahead_min_m + self.pp_lookahead_gain_s * speed,
                self.pp_lookahead_min_m,
                self.pp_lookahead_max_m,
            )
        )

        if target is None:
            steer = self._target_pid_fallback_steer(route_info.get("target", [0.0, 0.0]), speed)
            alpha = 0.0
            steer_raw = steer
            target_distance = 0.0
        else:
            steer, alpha, steer_raw, target_distance = self._compute_target_pp_steer(
                target, speed, lookahead
            )

        desired_speed = float(self.compute_desired_speed(waypoints)) if waypoints is not None else 0.0

        throttle, brake = self.compute_throttle_brake(speed, desired_speed)

        meta_info_1 = "speed: %.2f, target_speed: %.2f, alpha: %.2f, Ld: %.2f, target_d: %.2f, steer: %.2f" % (
            speed,
            desired_speed,
            alpha,
            lookahead,
            target_distance,
            steer,
        )
        meta_info_2 = "stop_steps:%d, lateral:pure_pursuit_target" % (self.stop_steps)
        meta_info = {
            1: meta_info_1,
            2: meta_info_2,
            "desired_speed": float(desired_speed),
            "speed": float(speed),
            "pp_alpha": float(alpha),
            "pp_lookahead": float(lookahead),
            "pp_steer_raw": float(steer_raw),
            "pp_target_distance": float(target_distance),
        }

        return steer, throttle, brake, meta_info
