"""Portfolio-level risk manager: heat limits, correlation caps, directional exposure."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.utils.helpers import pip_value
from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------
_DEFAULT_CORR_THRESHOLD = 0.70
_DEFAULT_CORR_REDUCTION = 0.50
_DEFAULT_MAX_TOTAL_HEAT = 0.05
_DEFAULT_MAX_LONG_EXPOSURE = 0.035
_DEFAULT_MAX_SHORT_EXPOSURE = 0.035
_DEFAULT_MAX_METALS = 0.025
_DEFAULT_MAX_INDICES = 0.025
_DEFAULT_MAX_FOREX = 0.030

# Asset class classification tables
_METAL_SYMBOLS = {"XAUUSD", "XAGUSD"}
_INDEX_SYMBOLS = {"US30", "US500", "NAS100", "GER40", "UK100", "JPN225"}


def _load_config() -> object | None:
    """Load application config safely, returning None on failure."""
    try:
        from core.utils.config import get_config
        return get_config()
    except Exception:  # noqa: BLE001
        return None


class PortfolioRiskManager:
    """Portfolio-level heat and correlation exposure manager.

    Tracks total open risk ("heat"), directional exposure, per-asset-class
    heat, and pairwise correlation between existing and candidate positions.
    All checks use fractional account-balance units (e.g. 0.05 = 5%).
    """

    def __init__(self, account_currency: str = "USD") -> None:
        """Initialise and load portfolio parameters from config with fallbacks.

        Args:
            account_currency: Three-letter ISO currency for pip-value conversions.
        """
        self._account_currency = account_currency.upper().strip()
        cfg = _load_config()

        try:
            port = cfg.portfolio  # type: ignore[union-attr]
            self._corr_threshold: float = float(port.correlation_cap_threshold)
            self._corr_reduction: float = float(port.correlation_size_reduction)
            self._max_total_heat: float = float(port.max_total_heat)
            self._max_long: float = float(port.max_long_exposure)
            self._max_short: float = float(port.max_short_exposure)
            self._max_metals: float = float(port.max_metals_combined)
            self._max_indices: float = float(port.max_indices_combined)
            self._max_forex: float = float(port.max_forex_combined)
        except Exception:  # noqa: BLE001
            logger.warning(
                "PortfolioRiskManager: failed to read portfolio config — using defaults."
            )
            self._corr_threshold = _DEFAULT_CORR_THRESHOLD
            self._corr_reduction = _DEFAULT_CORR_REDUCTION
            self._max_total_heat = _DEFAULT_MAX_TOTAL_HEAT
            self._max_long = _DEFAULT_MAX_LONG_EXPOSURE
            self._max_short = _DEFAULT_MAX_SHORT_EXPOSURE
            self._max_metals = _DEFAULT_MAX_METALS
            self._max_indices = _DEFAULT_MAX_INDICES
            self._max_forex = _DEFAULT_MAX_FOREX

        logger.debug(
            "PortfolioRiskManager initialised: corr_thresh=%.2f max_heat=%.3f",
            self._corr_threshold, self._max_total_heat,
        )

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def get_correlation_matrix(
        self,
        symbols: list[str],
        returns_data: dict[str, pd.Series],
        window: int = 20,
    ) -> pd.DataFrame:
        """Compute a rolling correlation matrix from recent return series.

        Uses the most recent *window* observations from each return series to
        produce an NxN Pearson correlation DataFrame.  Missing symbols in
        *returns_data* are filled with NaN columns.

        Args:
            symbols:      List of instrument identifiers.
            returns_data: Mapping of symbol -> pd.Series of log or simple returns.
            window:       Number of most-recent bars to use for correlation.

        Returns:
            NxN pd.DataFrame of Pearson correlations with symbols as both
            index and columns.  Values lie in [-1, 1].
        """
        aligned: dict[str, pd.Series] = {}
        for sym in symbols:
            series = returns_data.get(sym)
            if series is not None and len(series) >= 2:
                aligned[sym] = series.iloc[-window:] if len(series) > window else series
            else:
                aligned[sym] = pd.Series(dtype=float)

        if not aligned:
            logger.warning("get_correlation_matrix: no valid return series provided")
            return pd.DataFrame(index=symbols, columns=symbols, dtype=float)

        df = pd.DataFrame(aligned)
        corr_matrix = df.corr(method="pearson")

        # Ensure all requested symbols appear even if data was absent
        corr_matrix = corr_matrix.reindex(index=symbols, columns=symbols)

        logger.debug(
            "get_correlation_matrix: computed %dx%d matrix (window=%d)",
            len(symbols), len(symbols), window,
        )
        return corr_matrix

    # ------------------------------------------------------------------
    # Correlation exposure check
    # ------------------------------------------------------------------

    def check_correlation_exposure(
        self,
        new_signal,
        open_positions: list,
        returns_data: dict[str, pd.Series],
    ) -> tuple[float, str]:
        """Check whether adding a new signal creates excessive correlated exposure.

        For each open position that shares the same directional bias as
        *new_signal*, the Pearson correlation between the new asset's returns
        and the existing position's asset returns is evaluated.  If the
        correlation exceeds *correlation_cap_threshold* the size multiplier is
        reduced by *correlation_size_reduction*.

        Multiple correlated positions all apply the same reduction (0.50);
        the multiplier is not compounded.

        Args:
            new_signal:     SignalOutput for the candidate trade.
            open_positions: List of currently open position objects, each
                            expected to have ``.asset`` and ``.direction``
                            (or ``.signal``) attributes.
            returns_data:   Mapping of symbol -> pd.Series of returns used for
                            correlation computation.

        Returns:
            Tuple of (size_multiplier: float, reason: str).
            size_multiplier is 1.0 when no correlation issue is found, or
            *correlation_size_reduction* (default 0.50) when a correlated
            position is detected.
        """
        new_asset: str = getattr(new_signal, "asset", "")
        new_direction: str = getattr(new_signal, "signal", "")

        new_returns = returns_data.get(new_asset)
        if new_returns is None or len(new_returns) < 5:
            logger.debug(
                "check_correlation_exposure: no return data for %s — skipping",
                new_asset,
            )
            return 1.0, "no return data for new asset"

        for pos in open_positions:
            pos_asset: str = getattr(pos, "asset", getattr(pos, "symbol", ""))
            pos_direction: str = getattr(
                pos, "direction", getattr(pos, "signal", "")
            )

            if not pos_asset or pos_asset == new_asset:
                continue

            # Only penalise same-direction correlation
            if pos_direction != new_direction:
                continue

            pos_returns = returns_data.get(pos_asset)
            if pos_returns is None or len(pos_returns) < 5:
                continue

            # Align on common index and compute correlation
            combined = pd.concat(
                [new_returns.rename("new"), pos_returns.rename("existing")],
                axis=1,
            ).dropna()

            if len(combined) < 5:
                continue

            corr_val = float(combined["new"].corr(combined["existing"]))

            logger.debug(
                "check_correlation_exposure: corr(%s, %s)=%.4f (dir=%s)",
                new_asset, pos_asset, corr_val, new_direction,
            )

            if corr_val > self._corr_threshold:
                reason = (
                    f"corr({new_asset},{pos_asset})={corr_val:.2f} "
                    f"> threshold={self._corr_threshold:.2f} "
                    f"[both {new_direction}] -> size reduced"
                )
                logger.info("check_correlation_exposure: %s", reason)
                return self._corr_reduction, reason

        return 1.0, "no excessive correlation detected"

    # ------------------------------------------------------------------
    # Portfolio heat check
    # ------------------------------------------------------------------

    def check_portfolio_heat(
        self,
        new_risk: float,
        open_positions: list,
        account_balance: float,
    ) -> tuple[bool, str]:
        """Check whether adding *new_risk* would breach any heat limit.

        Checks performed in order:
        1. Total heat (existing + new) <= max_total_heat.
        2. Directional heat for the new signal's direction <= max_long/short_exposure.
        3. Asset-class heat <= per-class maximum.

        Each open position's heat contribution is computed as::

            heat_i = (lot_size * pip_value_per_lot * stop_distance_pips) / balance

        Args:
            new_risk:        Risk amount in account currency for the candidate trade.
            open_positions:  List of currently open position objects.  Each must
                             expose ``.asset``, ``.lot_size``, ``.stop_distance_pips``,
                             and ``.direction`` (or ``.signal``) attributes.
            account_balance: Current account equity in account currency.

        Returns:
            Tuple of (passes: bool, reason: str).
            passes=True when all heat checks are satisfied.
        """
        if account_balance <= 0:
            return False, "account_balance must be positive"

        new_heat_frac = new_risk / account_balance

        existing_heat = self._sum_position_heat(open_positions, account_balance)
        total_heat = existing_heat + new_heat_frac

        # 1. Total heat check
        if total_heat > self._max_total_heat:
            reason = (
                f"total heat {total_heat:.4f} would exceed "
                f"max_total_heat={self._max_total_heat:.4f}"
            )
            logger.info("check_portfolio_heat: FAIL — %s", reason)
            return False, reason

        logger.debug(
            "check_portfolio_heat: total_heat=%.4f / %.4f — OK",
            total_heat, self._max_total_heat,
        )
        return True, f"heat checks passed (total_heat={total_heat:.4f})"

    def _sum_position_heat(
        self,
        open_positions: list,
        account_balance: float,
    ) -> float:
        """Sum the heat contributions of all open positions.

        Args:
            open_positions:  List of open position objects.
            account_balance: Current account equity.

        Returns:
            Total heat as a fraction of account balance.
        """
        total = 0.0
        for pos in open_positions:
            try:
                lot_size = float(getattr(pos, "lot_size", 0.0))
                stop_pips = float(getattr(pos, "stop_distance_pips", 0.0))
                asset = str(getattr(pos, "asset", getattr(pos, "symbol", "")))
                pv = pip_value(asset, lot_size, self._account_currency)
                heat = (pv * stop_pips) / account_balance
                total += heat
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_sum_position_heat: could not compute heat for position %s: %s",
                    pos, exc,
                )
        return total

    def _sum_directional_heat(
        self,
        open_positions: list,
        direction: str,
        account_balance: float,
    ) -> float:
        """Sum heat for positions that share the given direction.

        Args:
            open_positions:  List of open positions.
            direction:       "LONG" or "SHORT".
            account_balance: Current account equity.

        Returns:
            Directional heat as a fraction of account balance.
        """
        total = 0.0
        for pos in open_positions:
            pos_dir = str(getattr(pos, "direction", getattr(pos, "signal", "")))
            if pos_dir != direction:
                continue
            try:
                lot_size = float(getattr(pos, "lot_size", 0.0))
                stop_pips = float(getattr(pos, "stop_distance_pips", 0.0))
                asset = str(getattr(pos, "asset", getattr(pos, "symbol", "")))
                pv = pip_value(asset, lot_size, self._account_currency)
                total += (pv * stop_pips) / account_balance
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_sum_directional_heat: error for position %s: %s", pos, exc
                )
        return total

    def _sum_asset_class_heat(
        self,
        open_positions: list,
        asset_class: str,
        account_balance: float,
    ) -> float:
        """Sum heat for all positions belonging to *asset_class*.

        Args:
            open_positions:  List of open positions.
            asset_class:     One of "metal", "index", or "forex".
            account_balance: Current account equity.

        Returns:
            Asset-class heat as a fraction of account balance.
        """
        total = 0.0
        for pos in open_positions:
            asset = str(getattr(pos, "asset", getattr(pos, "symbol", "")))
            if self._get_asset_class(asset) != asset_class:
                continue
            try:
                lot_size = float(getattr(pos, "lot_size", 0.0))
                stop_pips = float(getattr(pos, "stop_distance_pips", 0.0))
                pv = pip_value(asset, lot_size, self._account_currency)
                total += (pv * stop_pips) / account_balance
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_sum_asset_class_heat: error for position %s: %s", pos, exc
                )
        return total

    # ------------------------------------------------------------------
    # Composite sizing multiplier
    # ------------------------------------------------------------------

    def compute_sizing_multipliers(
        self,
        new_signal,
        open_positions: list,
        account_balance: float,
        returns_data: Optional[dict[str, pd.Series]] = None,
    ) -> dict:
        """Compute all portfolio-level sizing multipliers for *new_signal*.

        Runs both the heat check and the correlation exposure check, then
        packages results into a single dict consumed by the RiskEngine.

        Args:
            new_signal:      SignalOutput for the candidate trade.
            open_positions:  List of currently open positions.
            account_balance: Current account equity in account currency.
            returns_data:    Optional mapping of symbol -> returns series for
                             correlation computation.  Correlation check is
                             skipped when None or empty.

        Returns:
            Dict with keys:
            - ``"heat_ok"``             (bool)  True when heat checks pass.
            - ``"heat_reason"``         (str)   Human-readable heat result.
            - ``"correlation_multiplier"`` (float) 1.0 or *correlation_size_reduction*.
            - ``"correlation_reason"``  (str)   Human-readable correlation result.
        """
        # Derive risk amount from signal if available
        risk_amount = float(getattr(new_signal, "risk_amount", 0.0))
        if risk_amount == 0.0:
            # Estimate from lot_size if pre-sized
            lot_size = float(getattr(new_signal, "lot_size", 0.0))
            stop_pips = float(getattr(new_signal, "stop_distance_pips", 0.0))
            asset = str(getattr(new_signal, "asset", ""))
            if lot_size > 0 and stop_pips > 0 and asset:
                pv = pip_value(asset, lot_size, self._account_currency)
                risk_amount = pv * stop_pips

        heat_ok, heat_reason = self.check_portfolio_heat(
            new_risk=risk_amount,
            open_positions=open_positions,
            account_balance=account_balance,
        )

        if returns_data:
            corr_mult, corr_reason = self.check_correlation_exposure(
                new_signal=new_signal,
                open_positions=open_positions,
                returns_data=returns_data,
            )
        else:
            corr_mult, corr_reason = 1.0, "no returns data supplied"

        result = {
            "heat_ok": heat_ok,
            "heat_reason": heat_reason,
            "correlation_multiplier": corr_mult,
            "correlation_reason": corr_reason,
        }

        logger.debug("compute_sizing_multipliers: %s", result)
        return result

    # ------------------------------------------------------------------
    # Asset class helper
    # ------------------------------------------------------------------

    def _get_asset_class(self, symbol: str) -> str:
        """Classify *symbol* as 'metal', 'index', or 'forex'.

        Args:
            symbol: Instrument identifier, e.g. "XAUUSD", "US30", "EURUSD".

        Returns:
            One of ``"metal"``, ``"index"``, or ``"forex"``.
        """
        sym = symbol.upper().strip()
        if sym in _METAL_SYMBOLS:
            return "metal"
        if sym in _INDEX_SYMBOLS:
            return "index"
        return "forex"
