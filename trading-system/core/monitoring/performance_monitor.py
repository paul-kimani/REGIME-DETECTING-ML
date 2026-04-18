"""PerformanceMonitor — daily drift detection and alert dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import requests.exceptions

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_WARNING_THRESHOLD: float = 20.0   # abs drift % that triggers WARNING
_CRITICAL_THRESHOLD: float = 35.0  # abs drift % that triggers CRITICAL
_RETRAIN_CRITICAL_COUNT: int = 2   # number of CRITICAL alerts that triggers RETRAIN

# Metrics where a *negative* drift is bad (higher is always better for these).
# max_drawdown is stored as a positive fraction — lower is better, so positive
# drift (live > baseline) is bad for it.
_INVERSE_METRICS: frozenset[str] = frozenset({"max_drawdown"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """Represents a single performance alert.

    Attributes:
        level:     Severity string — ``"WARNING"``, ``"CRITICAL"``, or ``"RETRAIN"``.
        metric:    Name of the metric that triggered the alert.
        drift_pct: Raw drift percentage (positive = improved, negative = degraded
                   *except* for inverse metrics where positive = degraded).
        message:   Human-readable description of the alert.
    """

    level: str
    metric: str
    drift_pct: float
    message: str


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Tracks rolling live performance vs baseline. Runs daily at 22:00 UTC.

    Rolling windows: last 20, 50, 100 trades.
    Drift thresholds: WARNING > 20%, CRITICAL > 35%.
    RETRAIN trigger: >= 2 CRITICAL metrics.
    """

    def __init__(self, db_manager=None) -> None:
        """Initialise the monitor.

        Args:
            db_manager: Optional :class:`~core.data.db_manager.DatabaseManager`
                        for persisting system events.  Loaded from the caller;
                        not constructed internally.

        Environment variables consumed (Telegram):
            TELEGRAM_BOT_TOKEN: Bot token string.
            TELEGRAM_CHAT_ID:   Numeric or string chat ID.
        """
        self._db_manager = db_manager
        self._telegram_token: Optional[str] = os.environ.get("TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id: Optional[str] = os.environ.get("TELEGRAM_CHAT_ID")

        if self._telegram_token and self._telegram_chat_id:
            logger.info("PerformanceMonitor: Telegram notifications enabled")
        else:
            logger.info(
                "PerformanceMonitor: Telegram not configured "
                "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
            )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_live_metrics(
        self, recent_trades: list, window: int = 50
    ) -> dict:
        """Compute performance metrics from the last *window* trades.

        Args:
            recent_trades: Full list of recent trade objects.  Each trade must
                           expose ``pnl_currency`` and ``r_multiple`` attributes.
            window:        Number of most-recent trades to include (default 50).

        Returns:
            Dict with keys:
            ``win_rate``, ``profit_factor``, ``avg_r_multiple``,
            ``max_drawdown``, ``sharpe_ratio``, ``total_trades``.
            Returns zeroed dict when there are fewer than 2 trades in the window.
        """
        trades = recent_trades[-window:] if len(recent_trades) > window else recent_trades
        n = len(trades)

        empty = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r_multiple": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_trades": 0,
        }

        if n < 2:
            logger.warning(
                "compute_live_metrics: only %d trade(s) in window=%d — "
                "returning zeroed metrics",
                n,
                window,
            )
            return empty

        pnls = np.array([float(getattr(t, "pnl_currency", 0.0)) for t in trades])
        r_mults = np.array([float(getattr(t, "r_multiple", 0.0)) for t in trades])

        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        win_rate = float(len(winners)) / n

        gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        avg_r_multiple = float(np.mean(r_mults))

        # Max drawdown from cumulative PnL curve
        cum_pnl = np.cumsum(pnls)
        running_peak = np.maximum.accumulate(cum_pnl)
        # Avoid divide-by-zero for flat or all-loss sequences
        safe_peak = np.where(running_peak != 0, running_peak, np.nan)
        dd = (cum_pnl - running_peak) / safe_peak
        max_drawdown = float(abs(np.nanmin(dd))) if not np.all(np.isnan(dd)) else 0.0

        # Sharpe ratio from R-multiples (treat each trade as a "period")
        std_r = float(np.std(r_mults, ddof=1)) if n > 1 else 0.0
        if std_r > 0:
            sharpe_ratio = float(np.mean(r_mults) / std_r * np.sqrt(n))
        else:
            sharpe_ratio = 0.0

        metrics = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_r_multiple": avg_r_multiple,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": n,
        }

        logger.info(
            "compute_live_metrics (window=%d): n=%d win_rate=%.2f%% "
            "pf=%.2f avg_R=%.3f max_dd=%.2f%% sharpe=%.2f",
            window,
            n,
            win_rate * 100,
            profit_factor if profit_factor != float("inf") else 9999,
            avg_r_multiple,
            max_drawdown * 100,
            sharpe_ratio,
        )
        return metrics

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    def compute_drift(
        self, live_metrics: dict, baseline_metrics: dict
    ) -> dict:
        """Compute percentage drift for each shared metric.

        Formula: ``(live - baseline) / abs(baseline) * 100``

        For most metrics a positive value means improvement.
        For ``max_drawdown`` a positive value means the drawdown grew (worsened).

        Args:
            live_metrics:     Output of :meth:`compute_live_metrics`.
            baseline_metrics: Reference metrics dict (same keys).

        Returns:
            Dict mapping metric name to drift percentage.
            Missing or zero-baseline metrics are assigned 0.0.
        """
        drift: dict = {}
        for metric, live_val in live_metrics.items():
            baseline_val = baseline_metrics.get(metric)
            if baseline_val is None:
                drift[metric] = 0.0
                continue

            base_abs = abs(float(baseline_val))
            if base_abs == 0.0:
                drift[metric] = 0.0
            else:
                drift[metric] = (float(live_val) - float(baseline_val)) / base_abs * 100.0

        logger.debug("compute_drift: %s", drift)
        return drift

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def check_alerts(self, drift: dict) -> list:
        """Evaluate drift values and generate :class:`Alert` objects.

        A positive drift is good for all metrics *except* ``max_drawdown``,
        where a positive drift means the drawdown increased (got worse).

        Thresholds:
        - |drift| > 20% → WARNING
        - |drift| > 35% → CRITICAL

        A RETRAIN alert is appended when >= 2 CRITICAL alerts are present.

        Args:
            drift: Output of :meth:`compute_drift`.

        Returns:
            List of :class:`Alert` objects, possibly empty.  A RETRAIN alert
            is always last when present.
        """
        alerts: list[Alert] = []

        for metric, drift_pct in drift.items():
            if metric == "total_trades":
                continue  # not a performance quality metric

            abs_drift = abs(drift_pct)

            # For inverse metrics: positive drift is bad, so flip the sign for
            # the "is_degradation" check below.
            if metric in _INVERSE_METRICS:
                is_degradation = drift_pct > 0   # drawdown got larger
            else:
                is_degradation = drift_pct < 0   # metric got worse

            if abs_drift <= _WARNING_THRESHOLD:
                continue

            direction_word = "degraded" if is_degradation else "improved"

            if abs_drift > _CRITICAL_THRESHOLD:
                level = "CRITICAL"
            else:
                level = "WARNING"

            message = (
                f"{metric} has {direction_word} by {abs_drift:.1f}% "
                f"(drift={drift_pct:+.1f}%) — {level}"
            )
            alerts.append(
                Alert(
                    level=level,
                    metric=metric,
                    drift_pct=drift_pct,
                    message=message,
                )
            )
            logger.warning("PerformanceMonitor alert: %s", message)

        # RETRAIN trigger
        critical_count = sum(1 for a in alerts if a.level == "CRITICAL")
        if critical_count >= _RETRAIN_CRITICAL_COUNT:
            retrain_alert = Alert(
                level="RETRAIN",
                metric="system",
                drift_pct=0.0,
                message=(
                    f"RETRAIN triggered: {critical_count} CRITICAL metric(s) detected. "
                    "Model retraining recommended."
                ),
            )
            alerts.append(retrain_alert)
            logger.warning("PerformanceMonitor: %s", retrain_alert.message)

        return alerts

    # ------------------------------------------------------------------
    # Alert dispatch
    # ------------------------------------------------------------------

    def send_alert(self, alert: Alert, db_manager=None) -> None:
        """Persist alert to ``system_events`` and optionally push to Telegram.

        Telegram failures are swallowed (only :class:`requests.RequestException`
        is caught).

        Args:
            alert:      The :class:`Alert` to dispatch.
            db_manager: Optional override for the db_manager set at construction.
        """
        mgr = db_manager or self._db_manager

        # -- Persist to PostgreSQL --
        if mgr is not None:
            try:
                mgr.log_system_event(
                    event_type="PERFORMANCE_ALERT",
                    severity=alert.level,
                    message=alert.message,
                    details={
                        "metric": alert.metric,
                        "drift_pct": alert.drift_pct,
                    },
                )
            except Exception as exc:
                logger.error(
                    "PerformanceMonitor.send_alert: failed to log system event — %s",
                    exc,
                )

        # -- Telegram notification --
        if not (self._telegram_token and self._telegram_chat_id):
            return

        text = (
            f"[{alert.level}] Trading System Alert\n"
            f"Metric : {alert.metric}\n"
            f"Drift  : {alert.drift_pct:+.2f}%\n"
            f"Message: {alert.message}"
        )
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        try:
            response = requests.post(
                url,
                json={"chat_id": self._telegram_chat_id, "text": text},
                timeout=10,
            )
            if not response.ok:
                logger.warning(
                    "PerformanceMonitor: Telegram responded with status %d — %s",
                    response.status_code,
                    response.text[:200],
                )
            else:
                logger.debug(
                    "PerformanceMonitor: Telegram notification sent for %s",
                    alert.metric,
                )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "PerformanceMonitor: Telegram notification failed — %s", exc
            )

    # ------------------------------------------------------------------
    # Daily check entry point
    # ------------------------------------------------------------------

    def run_daily_check(
        self, recent_trades: list, baseline_metrics: dict
    ) -> list:
        """Run the full daily monitoring pipeline.

        Steps:
        1. Compute live metrics over the last 50 trades.
        2. Compute drift against *baseline_metrics*.
        3. Generate alerts.
        4. Dispatch each alert via :meth:`send_alert`.
        5. Return the alert list.

        Args:
            recent_trades:    List of recent closed trade objects.
            baseline_metrics: Reference metrics from the validated backtest.

        Returns:
            List of :class:`Alert` objects generated during this run.
        """
        logger.info(
            "PerformanceMonitor.run_daily_check: starting — %d trades available",
            len(recent_trades),
        )

        live = self.compute_live_metrics(recent_trades, window=50)
        drift = self.compute_drift(live, baseline_metrics)
        alerts = self.check_alerts(drift)

        logger.info(
            "run_daily_check: %d alert(s) generated (%d WARNING, %d CRITICAL, %d RETRAIN)",
            len(alerts),
            sum(1 for a in alerts if a.level == "WARNING"),
            sum(1 for a in alerts if a.level == "CRITICAL"),
            sum(1 for a in alerts if a.level == "RETRAIN"),
        )

        for alert in alerts:
            self.send_alert(alert)

        return alerts
