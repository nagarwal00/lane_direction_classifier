"""
Kalman Filter for temporal smoothing of lane curvature coefficient A_fit.

State vector: [A_fit, dA_fit/dt]
Observation:  [A_fit]  (we observe curvature each frame, not its derivative)
"""

import numpy as np


class LaneKalmanFilter:
    """
    1D Kalman filter tracking A_fit (quadratic lane curvature coefficient)
    and its first time derivative.

    Usage:
        kf = LaneKalmanFilter()
        smoothed_A = kf.update(raw_A_fit)   # call once per frame
        is_confident = kf.innovation_ok()   # True if measurement is consistent
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        """
        Args:
            dt: Time between frames in seconds. Default assumes 30 FPS video.
        """
        # --- State transition matrix F  (constant-velocity model) ---
        # [A_new]       [1  dt] [A_old      ]
        # [dA_new]  =   [0   1] [dA_old     ]
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # --- Observation matrix H  (we observe A directly) ---
        # observed = [1, 0] * [A, dA]^T
        self.H = np.array([[1.0, 0.0]])

        # --- Process noise covariance Q ---
        # Tune q: larger → filter trusts measurements more (more responsive)
        #         smaller → filter trusts model more (smoother but lagging)
        q = 1e-5
        self.Q = np.array([[q, 0.0],
                           [0.0, q]])

        # --- Measurement noise covariance R ---
        # Tune r: larger → measurements are noisy (smoother output)
        #         smaller → measurements are trusted (more reactive)
        self.R = np.array([[1e-4]])

        # --- Initial state and covariance ---
        self.x = np.zeros((2, 1))          # [A_fit=0, dA_fit=0]
        self.P = np.eye(2) * 1.0           # large initial uncertainty
        self.initialized = False
        self._innovation = 0.0

    def update(self, measurement: float) -> float:
        """
        Feed one A_fit measurement, return the Kalman-smoothed A_fit.

        Args:
            measurement: Raw A_fit from polynomial fit this frame.

        Returns:
            Smoothed A_fit value.
        """
        z = np.array([[measurement]])

        if not self.initialized:
            # Seed state with first observation
            self.x[0, 0] = measurement
            self.initialized = True
            return measurement

        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- Update (Kalman gain) ---
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self._innovation = float(z[0, 0] - (self.H @ x_pred)[0, 0])
        self.x = x_pred + K * self._innovation
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0])

    def innovation_ok(self, threshold: float = 0.005) -> bool:
        """
        Returns True if the last measurement was consistent with prediction.
        A large innovation (surprise) indicates an unreliable frame.

        Args:
            threshold: Maximum acceptable |innovation| for A_fit.
        """
        return abs(self._innovation) < threshold

    def reset(self):
        """Reset state — call when a new video clip starts."""
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        self.initialized = False
        self._innovation = 0.0

