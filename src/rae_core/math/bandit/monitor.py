"""
Bandit Monitor for tracking and anomaly detection

Monitors bandit performance and detects issues:
- Performance degradation
- Excessive exploration
- Arm imbalance
- Anomalous rewards
"""

import time
from collections import deque
from dataclasses import dataclass, field

from .bandit import MultiArmedBandit


@dataclass
class MonitorAlert:
    """
    Alert raised by monitor.

    Attributes:
        severity: Alert severity (info, warning, critical)
        category: Alert category
        message: Human-readable message
        timestamp: When alert was raised
        metadata: Additional context
    """

    severity: str  # info, warning, critical
    category: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class BanditMonitor:
    """
    Monitor for Multi-Armed Bandit.

    Tracks performance and raises alerts for:
    - Performance degradation
    - Excessive exploration
    - Arm imbalance (one arm dominates)
    - Anomalous rewards (outliers)
    - Staleness (no recent updates)
    """

    def __init__(
        self,
        bandit: MultiArmedBandit,
        window_size: int = 100,
        alert_history_size: int = 50,
    ):
        """
        Initialize monitor.

        Args:
            bandit: Bandit to monitor
            window_size: Rolling window for statistics
            alert_history_size: Maximum alerts to keep
        """
        self.bandit = bandit
        self.window_size = window_size
        self.alert_history_size = alert_history_size

        # Alert tracking
        self.alerts: deque[MonitorAlert] = deque(maxlen=alert_history_size)
        self.last_check_time = time.time()

        # Rolling statistics
        self.reward_window: deque[float] = deque(maxlen=window_size)
        self.arm_selection_window: deque[str] = deque(maxlen=window_size)

    def check_health(self) -> list[MonitorAlert]:
        """
        Run all health checks.

        Returns:
            List of new alerts (if any)
        """
        new_alerts = []

        # Check 1: Performance degradation
        alert = self._check_degradation()
        if alert:
            new_alerts.append(alert)

        # Check 2: Excessive exploration
        alert = self._check_excessive_exploration()
        if alert:
            new_alerts.append(alert)

        # Check 3: Arm imbalance
        alert = self._check_arm_imbalance()
        if alert:
            new_alerts.append(alert)

        # Check 4: Reward anomalies
        alert = self._check_reward_anomalies()
        if alert:
            new_alerts.append(alert)

        # Check 5: Staleness
        alert = self._check_staleness()
        if alert:
            new_alerts.append(alert)

        # Store alerts
        for alert in new_alerts:
            self.alerts.append(alert)

        self.last_check_time = time.time()
        return new_alerts

    def _check_degradation(self) -> MonitorAlert | None:
        """Check for performance degradation"""
        is_degraded, drop = self.bandit.check_degradation()

        if is_degraded:
            return MonitorAlert(
                severity="critical",
                category="degradation",
                message=f"Performance degraded by {drop:.1%} below baseline",
                metadata={
                    "drop": drop,
                    "baseline": self.bandit.baseline_mean_reward,
                    "current": (
                        sum(self.bandit.last_100_rewards)
                        / len(self.bandit.last_100_rewards)
                        if self.bandit.last_100_rewards
                        else 0.0
                    ),
                },
            )
        return None

    def _check_excessive_exploration(self) -> MonitorAlert | None:
        """Check if exploration rate is too high"""
        exploration_rate = self.bandit.config.exploration_rate
        max_rate = self.bandit.config.max_exploration_rate

        if exploration_rate > max_rate:
            return MonitorAlert(
                severity="critical",
                category="exploration",
                message=f"Exploration rate {exploration_rate:.1%} exceeds maximum {max_rate:.1%}",
                metadata={
                    "exploration_rate": exploration_rate,
                    "max_rate": max_rate,
                },
            )
        elif exploration_rate > max_rate * 0.8:
            return MonitorAlert(
                severity="warning",
                category="exploration",
                message=f"Exploration rate {exploration_rate:.1%} approaching maximum {max_rate:.1%}",
                metadata={
                    "exploration_rate": exploration_rate,
                    "max_rate": max_rate,
                },
            )
        return None

    def _check_arm_imbalance(self) -> MonitorAlert | None:
        """Check if one arm dominates selection"""
        if self.bandit.total_pulls < 50:
            return None  # Too early to judge

        # Calculate arm selection distribution
        arm_pulls = {arm.arm_id: arm.pulls for arm in self.bandit.arms}
        total_pulls = sum(arm_pulls.values())

        if total_pulls == 0:
            return None

        # Find dominant arm
        max_arm_id = max(arm_pulls.keys(), key=lambda k: arm_pulls[k])
        max_arm_ratio = arm_pulls[max_arm_id] / total_pulls

        # Alert if one arm has > 70% of selections
        if max_arm_ratio > 0.7:
            return MonitorAlert(
                severity="warning",
                category="arm_imbalance",
                message=f"Arm {max_arm_id} dominates with {max_arm_ratio:.1%} of selections",
                metadata={
                    "dominant_arm": max_arm_id,
                    "ratio": max_arm_ratio,
                    "distribution": arm_pulls,
                },
            )
        return None

    def _check_reward_anomalies(self) -> MonitorAlert | None:
        """Check for anomalous rewards (outliers)"""
        if len(self.reward_window) < 20:
            return None

        # Calculate mean and std
        rewards = list(self.reward_window)
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = variance**0.5

        if std == 0:
            return None

        # Check for outliers (> 3 std from mean)
        outliers = [r for r in rewards if abs(r - mean) > 3 * std]

        if outliers:
            return MonitorAlert(
                severity="warning",
                category="reward_anomaly",
                message=f"Detected {len(outliers)} reward outliers (>3 std)",
                metadata={
                    "outlier_count": len(outliers),
                    "total_count": len(rewards),
                    "mean": mean,
                    "std": std,
                },
            )
        return None

    def _check_staleness(self) -> MonitorAlert | None:
        """Check if bandit hasn't been updated recently"""
        # Find most recent pull across all arms
        recent_pulls = [
            arm.last_pulled for arm in self.bandit.arms if arm.last_pulled is not None
        ]

        if not recent_pulls:
            return None

        most_recent = max(recent_pulls)
        age = time.time() - most_recent

        # Alert if no updates in 1 hour
        if age > 3600:
            return MonitorAlert(
                severity="warning",
                category="staleness",
                message=f"No bandit updates in {age / 60:.0f} minutes",
                metadata={
                    "age_seconds": age,
                    "last_update": most_recent,
                },
            )
        return None

    def record_decision(self, arm_id: str, reward: float | None = None):
        """
        Record a decision for monitoring.

        Args:
            arm_id: ID of selected arm
            reward: Reward received (if outcome is available)
        """
        self.arm_selection_window.append(arm_id)

        if reward is not None:
            self.reward_window.append(reward)

    def get_summary(self) -> dict:
        """
        Get monitoring summary.

        Returns:
            Dictionary with monitoring statistics
        """
        # Recent alerts by severity
        recent_alerts = list(self.alerts)
        alert_counts = {
            "critical": sum(1 for a in recent_alerts if a.severity == "critical"),
            "warning": sum(1 for a in recent_alerts if a.severity == "warning"),
            "info": sum(1 for a in recent_alerts if a.severity == "info"),
        }

        # Arm selection distribution (recent window)
        arm_distribution: dict[str, int] = {}
        for arm_id in self.arm_selection_window:
            arm_distribution[arm_id] = arm_distribution.get(arm_id, 0) + 1

        # Reward statistics
        reward_stats = {}
        if self.reward_window:
            rewards = list(self.reward_window)
            reward_stats = {
                "mean": sum(rewards) / len(rewards),
                "min": min(rewards),
                "max": max(rewards),
                "count": len(rewards),
            }

        return {
            "last_check": self.last_check_time,
            "alert_counts": alert_counts,
            "recent_alerts": [a.to_dict() for a in list(self.alerts)[-5:]],
            "arm_distribution": arm_distribution,
            "reward_stats": reward_stats,
            "bandit_stats": self.bandit.get_statistics(),
        }

    def get_health_status(self) -> str:
        """
        Get overall health status.

        Returns:
            Health status: "healthy", "warning", "critical"
        """
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]

        critical_count = sum(1 for a in recent_alerts if a.severity == "critical")
        warning_count = sum(1 for a in recent_alerts if a.severity == "warning")

        if critical_count > 0:
            return "critical"
        elif warning_count > 2:
            return "warning"
        else:
            return "healthy"
