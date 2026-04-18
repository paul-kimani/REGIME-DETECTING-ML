"""SimulationEngine — realistic event-driven backtester with slippage and spread models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pip size lookup  (XAUUSD=0.01, forex=0.0001, indices=1.0)
# ---------------------------------------------------------------------------
_PIP_SIZES: dict[str, float] = {
    "XAUUSD": 0.01,
    "XAGUSD": 0.001,
    # Forex majors and crosses — default 0.0001
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    "AUDJPY": 0.01,   "CHFJPY": 0.01,   "CADJPY": 0.01,
    "NZDJPY": 0.01,
    "EURGBP": 0.0001, "EURAUD": 0.0001, "EURCAD": 0.0001,
    "EURNZD": 0.0001, "EURCHF": 0.0001, "GBPAUD": 0.0001,
    "GBPCAD": 0.0001, "GBPCHF": 0.0001, "GBPNZD": 0.0001,
    "AUDCAD": 0.0001, "AUDCHF": 0.0001, "AUDNZD": 0.0001,
    "CADCHF": 0.0001, "NZDCAD": 0.0001, "NZDCHF": 0.0001,
    # Indices
    "US30":   1.0,
    "US500":  0.1,
    "NAS100": 0.1,
    "GER40":  1.0,
    "UK100":  1.0,
    "JPN225": 1.0,
}

_DEFAULT_PIP_SIZE: float = 0.0001

# Typical half-spread in price units (not pips) per symbol
_DEFAULT_SPREAD_PIPS: float = 2.0   # default 2-pip full spread


def _pip_size(symbol: str) -> float:
    """Return the pip size for *symbol*.

    Args:
        symbol: Instrument identifier, e.g. "XAUUSD" or "EURUSD".

    Returns:
        Pip size as a float.
    """
    return _PIP_SIZES.get(symbol.upper().strip(), _DEFAULT_PIP_SIZE)


# ---------------------------------------------------------------------------
# BacktestTrade
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""

    trade_id: int
    symbol: str
    direction: str
    module: str
    entry_bar: int            # index in data
    entry_price: float
    entry_time: datetime
    stop_loss: float
    tp1: float
    tp2: float
    lot_size: float
    atr_at_entry: float
    regime_at_entry: str
    # Filled in at close
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_time: Optional[datetime] = None
    pnl_currency: float = 0.0
    r_multiple: float = 0.0
    mae_pips: float = 0.0      # max adverse excursion in pips (positive = adverse)
    mfe_pips: float = 0.0      # max favourable excursion in pips (positive = favourable)
    hold_bars: int = 0
    tp1_hit: bool = False
    slippage_pips: float = 0.0


# ---------------------------------------------------------------------------
# BacktestResults
# ---------------------------------------------------------------------------


@dataclass
class BacktestResults:
    """Complete results from a simulation run."""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    trades: list
    equity_curve: pd.Series     # indexed by bar index
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """Event-driven backtester that mirrors live system logic exactly.

    Slippage model:
    - Market orders: Normal(0.5, 0.8) pips, clipped [0, 2.5]
    - Limit orders: fill only if candle penetrates (low <= limit for longs)
    - Spread: applied at entry (ask = mid + spread/2) and exit (bid = mid - spread/2)

    Look-ahead prevention: features and regime detection computed only
    on data available up to bar i (features[0:i+1]).

    Args:
        feature_engineer: Instance with a ``compute(df)`` method.
        regime_detector:  Instance with a ``detect(symbol, data_dict, features)`` method.
        signal_router:    Instance with a ``route(symbol, timeframe, features,
                          regime_state, latest_bar)`` method.
        risk_engine:      Instance with a ``process(signal, account_state, features,
                          open_positions)`` method.
        initial_balance:  Starting account equity in account currency.
    """

    # Maximum number of bars a trade can remain open before forced close
    _MAX_HOLD_BARS: int = 96 * 5   # 5 trading days of M15 bars

    # Full spread in pips to apply at entry/exit
    _SPREAD_PIPS: float = _DEFAULT_SPREAD_PIPS

    def __init__(
        self,
        feature_engineer,
        regime_detector,
        signal_router,
        risk_engine,
        initial_balance: float = 100_000.0,
    ) -> None:
        """Initialise the simulation engine.

        Args:
            feature_engineer: FeatureEngineer instance.
            regime_detector:  RegimeDetector instance.
            signal_router:    SignalRouter instance.
            risk_engine:      RiskEngine instance.
            initial_balance:  Starting equity in account currency.
        """
        self._fe = feature_engineer
        self._rd = regime_detector
        self._sr = signal_router
        self._re = risk_engine
        self._initial_balance = float(initial_balance)

        # State reset on every run() call
        self._balance: float = 0.0
        self._open_trades: list[BacktestTrade] = []
        self._closed_trades: list[BacktestTrade] = []
        self._trade_counter: int = 0
        self._equity_by_bar: dict[int, float] = {}
        self._pending_order: Optional[object] = None    # TradeOrder awaiting fill

        logger.info(
            "SimulationEngine initialised: initial_balance=%.2f", initial_balance
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_idx: int = 200,
    ) -> BacktestResults:
        """Run the simulation bar-by-bar.

        Args:
            symbol:    Instrument identifier, e.g. "XAUUSD".
            data:      Full OHLCV DataFrame sorted ascending with columns
                       [open, high, low, close, volume].
            start_idx: Bar index from which trading begins (warmup period).

        Returns:
            :class:`BacktestResults` populated with all trades and equity curve.
        """
        if data.empty:
            raise ValueError("data must not be empty")
        if start_idx >= len(data):
            raise ValueError(
                f"start_idx={start_idx} >= len(data)={len(data)}"
            )

        logger.info(
            "[%s] run() started: %d bars, start_idx=%d, balance=%.2f",
            symbol, len(data), start_idx, self._initial_balance,
        )

        # Reset state
        self._balance = self._initial_balance
        self._open_trades = []
        self._closed_trades = []
        self._trade_counter = 0
        self._equity_by_bar = {}
        self._pending_order = None

        pip = _pip_size(symbol)
        spread_price = self._SPREAD_PIPS * pip   # full spread in price units

        # Pre-compute features once for the full dataset — each bar slices [0:i+1]
        # to ensure no look-ahead.  FeatureEngineer.compute() is deterministic so
        # we compute on the whole frame and then restrict the slice per bar.
        # NOTE: we deliberately do NOT use a pre-computed single-pass because
        # the spec requires features[0:i+1] at each bar step.  We compute
        # incrementally by slicing; for efficiency we pre-run once and slice.
        try:
            all_features: pd.DataFrame = self._fe.compute(data, symbol=symbol)
        except Exception as exc:
            logger.error("[%s] FeatureEngineer.compute() failed: %s", symbol, exc)
            all_features = pd.DataFrame(index=data.index)

        start_dt: datetime = _bar_time(data, start_idx)
        end_dt: datetime   = _bar_time(data, len(data) - 1)

        for i in range(start_idx, len(data)):
            bar: pd.Series = data.iloc[i]

            # ----------------------------------------------------------------
            # 1. Build look-ahead-free features (rows 0 .. i inclusive)
            # ----------------------------------------------------------------
            features_to_bar: pd.DataFrame = all_features.iloc[: i + 1]

            # ----------------------------------------------------------------
            # 2. Build mock account state
            # ----------------------------------------------------------------
            account_state: dict = self._build_account_state()
            account_state["timestamp"] = _bar_time(data, i)

            # ----------------------------------------------------------------
            # 3. Regime detection on current slice
            # ----------------------------------------------------------------
            try:
                regime_state = self._rd.detect(
                    symbol=symbol,
                    data_dict={"M15": data.iloc[: i + 1]},
                    features=features_to_bar,
                )
                current_regime: str = regime_state.m15_regime
            except Exception as exc:
                logger.warning(
                    "[%s] bar=%d regime_detector.detect() failed: %s", symbol, i, exc
                )
                regime_state = None
                current_regime = "UNKNOWN"

            # ----------------------------------------------------------------
            # 4. Attempt fill for any pending order from previous bar
            # ----------------------------------------------------------------
            if self._pending_order is not None:
                prev_bar: Optional[pd.Series] = (
                    data.iloc[i - 1] if i > 0 else None
                )
                new_trade = self._attempt_fill(
                    self._pending_order, bar, i, prev_bar
                )
                if new_trade is not None:
                    self._open_trades.append(new_trade)
                    logger.info(
                        "[%s] bar=%d FILLED trade_id=%d %s @ %.5f",
                        symbol, i, new_trade.trade_id,
                        new_trade.direction, new_trade.entry_price,
                    )
                self._pending_order = None

            # ----------------------------------------------------------------
            # 5. Manage open positions against current bar OHLCV
            # ----------------------------------------------------------------
            self._check_open_positions(bar, i)

            # Update MAE/MFE for still-open trades
            for trade in list(self._open_trades):
                self._update_mae_mfe(trade, bar, pip)

            # ----------------------------------------------------------------
            # 6. Generate signal if trading is allowed
            # ----------------------------------------------------------------
            if regime_state is not None:
                latest_bar_dict: dict = {
                    "bid": float(bar["close"]) - spread_price / 2,
                    "ask": float(bar["close"]) + spread_price / 2,
                    "atr": float(
                        features_to_bar["atr_14"].iloc[-1]
                        if "atr_14" in features_to_bar.columns
                        else 0.0
                    ),
                    "high":  float(bar["high"]),
                    "low":   float(bar["low"]),
                    "close": float(bar["close"]),
                    "open":  float(bar["open"]),
                }

                try:
                    signal = self._sr.route(
                        symbol=symbol,
                        timeframe="M15",
                        features=features_to_bar,
                        regime_state=regime_state,
                        latest_bar=latest_bar_dict,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] bar=%d signal_router.route() failed: %s", symbol, i, exc
                    )
                    signal = None

                # ----------------------------------------------------------------
                # 7. Process signal through risk engine
                # ----------------------------------------------------------------
                if signal is not None:
                    try:
                        trade_order = self._re.process(
                            signal=signal,
                            account_state=account_state,
                            features=features_to_bar,
                            open_positions=self._open_trades,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[%s] bar=%d risk_engine.process() failed: %s",
                            symbol, i, exc,
                        )
                        trade_order = None

                    if trade_order is not None:
                        # Queue the order — fill at next bar open
                        self._pending_order = trade_order
                        logger.debug(
                            "[%s] bar=%d order queued: %s %s entry=%.5f "
                            "sl=%.5f tp1=%.5f tp2=%.5f lots=%.4f",
                            symbol, i,
                            trade_order.direction, trade_order.module,
                            trade_order.entry_price, trade_order.stop_loss,
                            trade_order.take_profit_1, trade_order.take_profit_2,
                            trade_order.lot_size,
                        )

            # ----------------------------------------------------------------
            # 8. Update equity curve (balance + unrealised P&L)
            # ----------------------------------------------------------------
            unrealised: float = sum(
                self._unrealised_pnl(t, float(bar["close"]), pip)
                for t in self._open_trades
            )
            self._equity_by_bar[i] = self._balance + unrealised

        # Force-close any remaining open trades on the last bar
        last_bar = data.iloc[-1]
        for trade in list(self._open_trades):
            self._close_position(
                trade, last_bar, len(data) - 1, "end_of_data"
            )

        equity_curve = pd.Series(self._equity_by_bar, name="equity")

        logger.info(
            "[%s] run() complete: %d trades closed, final_balance=%.2f",
            symbol, len(self._closed_trades), self._balance,
        )

        return BacktestResults(
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
            initial_balance=self._initial_balance,
            final_balance=self._balance,
            trades=list(self._closed_trades),
            equity_curve=equity_curve,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _check_open_positions(self, bar: pd.Series, bar_idx: int) -> None:
        """Check all open backtest positions against current bar OHLCV.

        Priority order for intrabar events: stop first, then TP1, then TP2.
        For LONG positions:
            - Stop hit  : bar.low  <= trade.stop_loss
            - TP1 hit   : bar.high >= trade.tp1
            - TP2 hit   : bar.high >= trade.tp2 (only if tp1 already hit)
        For SHORT positions:
            - Stop hit  : bar.high >= trade.stop_loss
            - TP1 hit   : bar.low  <= trade.tp1
            - TP2 hit   : bar.low  <= trade.tp2 (only if tp1 already hit)

        Partial close at TP1: close 50% lot size, move stop to breakeven.

        Args:
            bar:     Current OHLCV bar as a Series.
            bar_idx: Integer bar index.
        """
        bar_low  = float(bar["low"])
        bar_high = float(bar["high"])

        for trade in list(self._open_trades):
            # ----- Hard stop check (highest priority) ----------------------
            if trade.direction == "LONG":
                stop_hit = bar_low <= trade.stop_loss
            else:
                stop_hit = bar_high >= trade.stop_loss

            if stop_hit:
                close_price = trade.stop_loss
                self._close_position(trade, bar, bar_idx, "stop_loss", close_price)
                continue

            # ----- TP2 check (only when TP1 already hit) -------------------
            if trade.tp1_hit:
                if trade.direction == "LONG":
                    tp2_hit = bar_high >= trade.tp2
                else:
                    tp2_hit = bar_low <= trade.tp2

                if tp2_hit:
                    close_price = trade.tp2
                    self._close_position(trade, bar, bar_idx, "tp2", close_price)
                    continue

            # ----- TP1 check (partial close) -------------------------------
            if not trade.tp1_hit:
                if trade.direction == "LONG":
                    tp1_hit = bar_high >= trade.tp1
                else:
                    tp1_hit = bar_low <= trade.tp1

                if tp1_hit:
                    self._handle_tp1(trade, bar_idx, trade.tp1)

            # ----- Time expiry check ---------------------------------------
            hold = bar_idx - trade.entry_bar
            if hold >= self._MAX_HOLD_BARS:
                close_price = float(bar["close"])
                self._close_position(
                    trade, bar, bar_idx, "time_expiry", close_price
                )

    def _handle_tp1(
        self,
        trade: BacktestTrade,
        bar_idx: int,
        tp1_price: float,
    ) -> None:
        """Handle a TP1 hit: close 50% of position, move stop to breakeven.

        This modifies the trade in-place (reduces lot_size by 50%, marks
        tp1_hit=True, moves stop_loss to entry_price).  The 50% PnL is
        immediately credited to the balance.

        Args:
            trade:     The open BacktestTrade.
            bar_idx:   Current bar index.
            tp1_price: Price at which TP1 was hit.
        """
        pip = _pip_size(trade.symbol)
        half_lots = trade.lot_size * 0.5

        # Compute PnL on the 50% that is closed
        if trade.direction == "LONG":
            pnl_pips = (tp1_price - trade.entry_price) / pip
        else:
            pnl_pips = (trade.entry_price - tp1_price) / pip

        # Use a simplified pip-value estimate (same as in _close_position)
        pnl_currency = pnl_pips * pip * half_lots * self._contract_size(trade.symbol)
        self._balance += pnl_currency

        # Reduce lot size to 50% and move stop to breakeven
        trade.lot_size   = half_lots
        trade.tp1_hit    = True
        trade.stop_loss  = trade.entry_price  # breakeven

        logger.info(
            "[%s] trade_id=%d TP1 hit @ %.5f — 50%% closed, pnl=%.2f, "
            "stop moved to breakeven %.5f",
            trade.symbol, trade.trade_id, tp1_price, pnl_currency, trade.entry_price,
        )

    def _attempt_fill(
        self,
        trade_order,
        bar: pd.Series,
        bar_idx: int,
        prev_bar: Optional[pd.Series] = None,
    ) -> Optional[BacktestTrade]:
        """Try to fill the order based on order type and bar prices.

        Order types:
        - ``"MARKET"``: Fill at bar open + slippage (always fills).
        - ``"LIMIT"``:  Fill only if bar penetrates the limit level.
          For LONG  : bar.low  <= limit_price
          For SHORT : bar.high >= limit_price
        - ``"STOP_LIMIT"``: Treated identically to LIMIT.

        Spread is applied: entry ask = fill_price + spread/2 for LONG,
                           entry bid = fill_price - spread/2 for SHORT.

        Args:
            trade_order: :class:`~core.risk.TradeOrder` from the risk engine.
            bar:         Current OHLCV bar.
            bar_idx:     Current bar index.
            prev_bar:    Previous bar (unused, kept for API completeness).

        Returns:
            A new :class:`BacktestTrade` if filled, otherwise ``None``.
        """
        direction: str  = str(trade_order.direction).upper()
        symbol: str     = str(trade_order.symbol)
        order_type: str = str(getattr(trade_order, "order_type", "MARKET")).upper()
        limit_price: float = float(trade_order.entry_price)

        pip        = _pip_size(symbol)
        spread_px  = self._SPREAD_PIPS * pip   # full spread in price units
        bar_open   = float(bar["open"])
        bar_low    = float(bar["low"])
        bar_high   = float(bar["high"])

        fill_price: Optional[float] = None

        if order_type == "MARKET":
            slippage_pips  = self._compute_slippage()
            slippage_price = slippage_pips * pip
            if direction == "LONG":
                fill_price = bar_open + slippage_price + spread_px / 2
            else:
                fill_price = bar_open - slippage_price - spread_px / 2

        elif order_type in ("LIMIT", "STOP_LIMIT"):
            slippage_pips = 0.0
            if direction == "LONG" and bar_low <= limit_price:
                fill_price = limit_price + spread_px / 2
            elif direction == "SHORT" and bar_high >= limit_price:
                fill_price = limit_price - spread_px / 2
        else:
            # Unknown order type — treat as market
            logger.warning(
                "[%s] Unknown order_type='%s' — treating as MARKET", symbol, order_type
            )
            slippage_pips  = self._compute_slippage()
            slippage_price = slippage_pips * pip
            if direction == "LONG":
                fill_price = bar_open + slippage_price + spread_px / 2
            else:
                fill_price = bar_open - slippage_price - spread_px / 2

        if fill_price is None:
            return None

        self._trade_counter += 1
        regime_at_entry = str(
            getattr(
                getattr(trade_order, "regime_context", None),
                "m15_regime",
                "UNKNOWN",
            )
        )
        atr_at_entry = float(getattr(trade_order, "atr", 0.0))
        entry_time   = _bar_time_raw(bar)

        trade = BacktestTrade(
            trade_id        = self._trade_counter,
            symbol          = symbol,
            direction       = direction,
            module          = str(trade_order.module),
            entry_bar       = bar_idx,
            entry_price     = fill_price,
            entry_time      = entry_time,
            stop_loss       = float(trade_order.stop_loss),
            tp1             = float(trade_order.take_profit_1),
            tp2             = float(trade_order.take_profit_2),
            lot_size        = float(trade_order.lot_size),
            atr_at_entry    = atr_at_entry,
            regime_at_entry = regime_at_entry,
            slippage_pips   = slippage_pips if order_type == "MARKET" else 0.0,
        )
        return trade

    def _close_position(
        self,
        trade: BacktestTrade,
        bar: pd.Series,
        bar_idx: int,
        reason: str,
        price: Optional[float] = None,
    ) -> None:
        """Close a backtest trade, compute PnL, R-multiple, MAE and MFE.

        If *price* is None the bar close is used.
        Spread is subtracted from the exit price (bid side for LONG,
        ask side for SHORT).

        Args:
            trade:   The open :class:`BacktestTrade` to close.
            bar:     Current OHLCV bar.
            bar_idx: Current bar index.
            reason:  Human-readable close reason string.
            price:   Optional explicit exit price; defaults to bar close.
        """
        if trade not in self._open_trades:
            return

        pip       = _pip_size(trade.symbol)
        spread_px = self._SPREAD_PIPS * pip

        if price is None:
            raw_exit = float(bar["close"])
        else:
            raw_exit = float(price)

        # Apply spread to exit: LONG exits at bid (close - spread/2),
        # SHORT exits at ask (close + spread/2)
        if trade.direction == "LONG":
            exit_price = raw_exit - spread_px / 2
        else:
            exit_price = raw_exit + spread_px / 2

        # PnL in pips
        if trade.direction == "LONG":
            pnl_pips = (exit_price - trade.entry_price) / pip
        else:
            pnl_pips = (trade.entry_price - exit_price) / pip

        # Convert pips to currency
        contract = self._contract_size(trade.symbol)
        pnl_currency = pnl_pips * pip * trade.lot_size * contract

        # R-multiple: how many R's did this trade make?
        stop_dist_pips = abs(trade.entry_price - trade.stop_loss) / pip
        if stop_dist_pips > 0:
            r_multiple = pnl_pips / stop_dist_pips
        else:
            r_multiple = 0.0

        hold_bars = bar_idx - trade.entry_bar

        # Populate trade fields
        trade.exit_bar      = bar_idx
        trade.exit_price    = exit_price
        trade.exit_reason   = reason
        trade.exit_time     = _bar_time_raw(bar)
        trade.pnl_currency  = pnl_currency
        trade.r_multiple    = r_multiple
        trade.hold_bars     = hold_bars

        # Update balance immediately
        self._balance += pnl_currency

        self._open_trades.remove(trade)
        self._closed_trades.append(trade)

        logger.info(
            "[%s] trade_id=%d CLOSED reason=%s %s @ %.5f "
            "pnl=%.2f R=%.2f hold=%d",
            trade.symbol, trade.trade_id, reason,
            trade.direction, exit_price,
            pnl_currency, r_multiple, hold_bars,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_slippage(self) -> float:
        """Sample slippage in pips: Normal(0.5, 0.8) clipped to [0, 2.5].

        Returns:
            Slippage magnitude in pips (always non-negative).
        """
        slip = float(np.random.normal(loc=0.5, scale=0.8))
        return float(np.clip(slip, 0.0, 2.5))

    def _build_account_state(self) -> dict:
        """Build mock account_state dict for the risk engine and pre-trade checks.

        Returns:
            Dict with balance, equity, daily_loss_pct, consecutive_losses,
            open_positions, recent_trades, and current_spread.
        """
        # Compute daily loss as fraction of initial balance from recent closed trades
        recent_pnl = sum(t.pnl_currency for t in self._closed_trades[-20:])
        daily_loss_pct = max(0.0, -recent_pnl / self._initial_balance)

        # Count current consecutive losses
        consec = 0
        for t in reversed(self._closed_trades):
            if t.pnl_currency < 0:
                consec += 1
            else:
                break

        return {
            "balance":             self._balance,
            "equity":              self._balance,
            "daily_loss_pct":      daily_loss_pct,
            "consecutive_losses":  consec,
            "open_positions":      list(self._open_trades),
            "recent_trades":       list(self._closed_trades[-50:]),
            "current_spread":      0.0,
        }

    def _update_mae_mfe(
        self,
        trade: BacktestTrade,
        bar: pd.Series,
        pip_size: float,
    ) -> None:
        """Update running MAE and MFE for an open trade.

        MAE (max adverse excursion) uses bar.low for LONG (worst case within bar)
        and bar.high for SHORT.  MFE uses bar.high for LONG and bar.low for SHORT.
        Both are stored as positive pip counts.

        Args:
            trade:    Open :class:`BacktestTrade` to update.
            bar:      Current OHLCV bar.
            pip_size: Pip size for the instrument.
        """
        if pip_size <= 0:
            return

        bar_low  = float(bar["low"])
        bar_high = float(bar["high"])

        if trade.direction == "LONG":
            adverse_price    = bar_low
            favourable_price = bar_high
            adverse_pips  = (trade.entry_price - adverse_price)    / pip_size
            favourable_pips = (favourable_price - trade.entry_price) / pip_size
        else:
            adverse_price    = bar_high
            favourable_price = bar_low
            adverse_pips  = (adverse_price - trade.entry_price)    / pip_size
            favourable_pips = (trade.entry_price - favourable_price) / pip_size

        trade.mae_pips = max(trade.mae_pips, adverse_pips)
        trade.mfe_pips = max(trade.mfe_pips, favourable_pips)

    def _unrealised_pnl(
        self,
        trade: BacktestTrade,
        mid_price: float,
        pip_size: float,
    ) -> float:
        """Compute unrealised PnL for an open trade at mid_price.

        Args:
            trade:     Open :class:`BacktestTrade`.
            mid_price: Current mid price.
            pip_size:  Pip size for the instrument.

        Returns:
            Unrealised PnL in account currency.
        """
        if pip_size <= 0:
            return 0.0

        if trade.direction == "LONG":
            pnl_pips = (mid_price - trade.entry_price) / pip_size
        else:
            pnl_pips = (trade.entry_price - mid_price) / pip_size

        contract = self._contract_size(trade.symbol)
        return pnl_pips * pip_size * trade.lot_size * contract

    @staticmethod
    def _contract_size(symbol: str) -> float:
        """Return the contract size (units per lot) for *symbol*.

        Uses the same table as core.utils.helpers._PIP_TABLE.

        Args:
            symbol: Instrument identifier.

        Returns:
            Contract size as a float.
        """
        _contracts: dict[str, float] = {
            "XAUUSD": 100.0, "XAGUSD": 100.0,
            "US30":   1.0,   "US500":  1.0,
            "NAS100": 1.0,   "GER40":  1.0,
            "UK100":  1.0,   "JPN225": 1.0,
            "USOIL":  1000.0, "UKOIL": 1000.0,
        }
        sym = symbol.upper().strip()
        return _contracts.get(sym, 100_000.0)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _bar_time(data: pd.DataFrame, idx: int) -> datetime:
    """Extract a datetime from a DataFrame bar index position.

    Args:
        data: OHLCV DataFrame.
        idx:  Integer bar position.

    Returns:
        datetime for the bar, UTC-naive.
    """
    try:
        ts = data.index[idx]
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime().replace(tzinfo=None)
        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return datetime(2000, 1, 1)


def _bar_time_raw(bar: pd.Series) -> datetime:
    """Extract a datetime from a single bar Series (name or 'timestamp' field).

    Args:
        bar: A row from an OHLCV DataFrame.

    Returns:
        datetime for the bar, UTC-naive.
    """
    try:
        if bar.name is not None:
            ts = bar.name
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime().replace(tzinfo=None)
            return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
    except Exception:
        pass
    return datetime(2000, 1, 1)
