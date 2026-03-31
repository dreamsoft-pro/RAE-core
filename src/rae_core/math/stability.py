"""
Stability Governor - Control Theory Implementation for RAE.

This module implements MATH-3: Stability Governor using PID Controllers and Kalman Filters
to smooth out fluctuations in memory importance and system metrics.
"""


class PIDController:
    """
    Proportional-Integral-Derivative Controller.
    Used for stabilizing metrics or resource allocation.
    """

    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint

        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, measurement: float, dt: float = 1.0) -> float:
        """
        Calculate control output.
        """
        error = self.setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self._integral += error * dt
        i_term = self.ki * self._integral

        # Derivative term
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        self._prev_error = error

        return p_term + i_term + d_term


class SimpleKalmanFilter:
    """
    1D Kalman Filter for smoothing noisy measurements (e.g. importance scores).
    """

    def __init__(
        self, r: float, q: float, a: float = 1.0, b: float = 0.0, c: float = 1.0
    ):
        """
        Args:
            r: Measurement noise covariance (How noisy is the measurement?)
            q: Process noise covariance (How much does the system change?)
            a: State transition multiplier
            b: Control input multiplier
            c: Measurement multiplier
        """
        self.r = r
        self.q = q
        self.a = a
        self.b = b
        self.c = c

        self.x = 0.0  # Initial state estimate
        self.p = 1.0  # Initial covariance estimate

    def update(self, measurement: float, control_input: float = 0.0) -> float:
        """
        Update the filter with a new measurement.
        Returns the filtered state estimate.
        """
        # Prediction Step
        # x = A*x + B*u
        pred_x = self.a * self.x + self.b * control_input
        # p = A*p*A^T + Q
        pred_p = self.a * self.p * self.a + self.q

        # Update Step
        # Kalman Gain: K = p*C^T * (C*p*C^T + R)^-1
        k = pred_p * self.c / (self.c * pred_p * self.c + self.r)

        # Update state: x = x + K * (z - C*x)
        self.x = pred_x + k * (measurement - self.c * pred_x)

        # Update covariance: p = (I - K*C) * p
        self.p = (1 - k * self.c) * pred_p

        return self.x
