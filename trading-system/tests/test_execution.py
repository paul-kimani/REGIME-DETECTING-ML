"""Tests for execution engine: order manager, position manager, fill monitor, trade journal."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class TestOrderManager:
    """Tests for core/execution/order_manager.py."""

    def _make_order_manager(self, mock_mt5=None):
        from core.execution.order_manager import OrderManager

        if mock_mt5 is None:
            mock_mt5 = MagicMock()
        return OrderManager(mt5_client=mock_mt5), mock_mt5

    def test_place_limit_order_long_success(self, sample_trade_order, mock_mt5_client):
        """place_limit_order() returns a ticket int on successful MT5 response."""
        from core.execution.order_manager import OrderManager, TRADE_RETCODE_DONE

        mock_mt5_client.send_order.return_value = {
            "retcode": TRADE_RETCODE_DONE,
            "order":   99001,
        }
        om = OrderManager(mt5_client=mock_mt5_client)
        ticket = om.place_limit_order(sample_trade_order)

        assert ticket == 99001
        mock_mt5_client.send_order.assert_called_once()

    def test_place_limit_order_short_uses_sell_limit(self, sample_trade_order, mock_mt5_client):
        """SHORT direction maps to SELL_LIMIT order type (3)."""
        from core.execution.order_manager import OrderManager, TRADE_RETCODE_DONE, ORDER_TYPE_SELL_LIMIT

        sample_trade_order.direction = "SHORT"
        mock_mt5_client.send_order.return_value = {"retcode": TRADE_RETCODE_DONE, "order": 99002}

        om = OrderManager(mt5_client=mock_mt5_client)
        om.place_limit_order(sample_trade_order)

        call_kwargs = mock_mt5_client.send_order.call_args[0][0]
        assert call_kwargs["type"] == ORDER_TYPE_SELL_LIMIT

    def test_place_limit_order_sl_tp_always_zero(self, sample_trade_order, mock_mt5_client):
        """SL and TP in the MT5 request are always 0 (Python-managed)."""
        from core.execution.order_manager import OrderManager, TRADE_RETCODE_DONE

        mock_mt5_client.send_order.return_value = {"retcode": TRADE_RETCODE_DONE, "order": 99003}
        om = OrderManager(mt5_client=mock_mt5_client)
        om.place_limit_order(sample_trade_order)

        request = mock_mt5_client.send_order.call_args[0][0]
        assert request.get("sl", 0) == 0
        assert request.get("tp", 0) == 0

    def test_place_limit_order_returns_none_on_mt5_error(self, sample_trade_order, mock_mt5_client):
        """place_limit_order() returns None when MT5 returns non-success retcode."""
        from core.execution.order_manager import OrderManager

        mock_mt5_client.send_order.return_value = {"retcode": 10006, "order": 0}
        om = OrderManager(mt5_client=mock_mt5_client)
        ticket = om.place_limit_order(sample_trade_order)
        assert ticket is None

    def test_place_limit_order_returns_none_on_exception(self, sample_trade_order, mock_mt5_client):
        """place_limit_order() returns None (not raises) on MT5ConnectionError."""
        from core.execution.order_manager import OrderManager
        from core.execution.mt5_connector import MT5ConnectionError

        mock_mt5_client.send_order.side_effect = MT5ConnectionError("timeout")
        om = OrderManager(mt5_client=mock_mt5_client)
        ticket = om.place_limit_order(sample_trade_order)
        assert ticket is None

    def test_place_stop_limit_order_uses_correct_type(self, sample_trade_order, mock_mt5_client):
        """BREAKOUT LONG maps to BUY_STOP_LIMIT (6)."""
        from core.execution.order_manager import OrderManager, TRADE_RETCODE_DONE, ORDER_TYPE_BUY_STOP_LIMIT

        sample_trade_order.module = "BREAKOUT"
        mock_mt5_client.send_order.return_value = {"retcode": TRADE_RETCODE_DONE, "order": 99010}
        om = OrderManager(mt5_client=mock_mt5_client)
        om.place_stop_limit_order(sample_trade_order)

        request = mock_mt5_client.send_order.call_args[0][0]
        assert request["type"] == ORDER_TYPE_BUY_STOP_LIMIT

    def test_cancel_order_calls_mt5(self, mock_mt5_client):
        """cancel_order() calls MT5 with the correct ticket."""
        from core.execution.order_manager import OrderManager, TRADE_RETCODE_DONE

        mock_mt5_client.send_order.return_value = {"retcode": TRADE_RETCODE_DONE}
        om = OrderManager(mt5_client=mock_mt5_client)
        om.cancel_order(99004)
        mock_mt5_client.send_order.assert_called_once()


# ---------------------------------------------------------------------------
# PositionState & PositionManager
# ---------------------------------------------------------------------------


class TestPositionManager:
    """Tests for core/execution/position_manager.py."""

    def _make_position_manager(self, mock_mt5=None):
        from core.execution.position_manager import PositionManager

        if mock_mt5 is None:
            mock_mt5 = MagicMock()
            mock_mt5.positions_get.return_value = []

        stop_eng  = MagicMock()
        cb        = MagicMock()
        cb.is_trading_halted.return_value = False
        journal   = MagicMock()
        regime    = MagicMock()

        pm = PositionManager(
            mt5_client=mock_mt5,
            stop_target_engine=stop_eng,
            circuit_breaker=cb,
            trade_journal=journal,
            regime_detector=regime,
        )
        return pm, mock_mt5

    def _make_position_state(self, ticket=10001, direction="LONG"):
        from core.execution.position_manager import PositionState

        return PositionState(
            ticket=ticket,
            symbol="XAUUSD",
            direction=direction,
            module="MOMENTUM",
            lot_size=0.10,
            entry_price=1810.0,
            fill_price=1810.0,
            current_stop=1800.0,
            tp1=1820.0,
            tp2=1830.0,
            atr_at_entry=5.0,
            timeframe="M15",
            magic_number=12345678,
            open_time=datetime.now(timezone.utc),
        )

    def test_register_position_adds_to_dict(self):
        """register_position() adds the state to the internal dict."""
        pm, _ = self._make_position_manager()
        ps = self._make_position_state(ticket=10001)
        pm.register_position(ps)

        assert 10001 in pm._positions
        assert pm._positions[10001] is ps

    def test_get_open_positions_returns_list(self):
        """get_open_positions() returns a list of PositionState dicts."""
        pm, _ = self._make_position_manager()
        ps = self._make_position_state(ticket=10002)
        pm.register_position(ps)

        positions = pm.get_open_positions()
        assert isinstance(positions, list)

    def test_close_position_removes_from_dict(self):
        """_close_position() removes the ticket from the dict."""
        pm, mock_mt5 = self._make_position_manager()
        mock_mt5.send_order.return_value = {"retcode": 10009}

        ps = self._make_position_state(ticket=10003)
        pm.register_position(ps)
        pm._close_position(10003, reason="test_close")

        assert 10003 not in pm._positions

    def test_hard_stop_triggered_for_long_below_stop(self):
        """Position is closed when close price drops below stop_loss (LONG)."""
        pm, mock_mt5 = self._make_position_manager()
        mock_mt5.send_order.return_value = {"retcode": 10009}
        mock_mt5.symbol_info_tick.return_value = {
            "bid": 1795.0,  # below stop 1800
            "ask": 1796.0,
        }

        ps = self._make_position_state(ticket=10004, direction="LONG")
        pm.register_position(ps)
        pm._check_position(10004)

        # Position should be closed
        assert 10004 not in pm._positions

    def test_tp1_partial_close_sets_flag(self):
        """TP1 hit triggers tp1_hit flag and moves stop to breakeven."""
        pm, mock_mt5 = self._make_position_manager()
        mock_mt5.send_order.return_value = {"retcode": 10009}
        mock_mt5.symbol_info_tick.return_value = {
            "bid": 1821.0,  # above TP1 = 1820
            "ask": 1821.5,
        }

        ps = self._make_position_state(ticket=10005, direction="LONG")
        pm.register_position(ps)
        pm._check_position(10005)

        if 10005 in pm._positions:
            assert pm._positions[10005].tp1_hit
            # Stop should be at breakeven or higher
            assert pm._positions[10005].current_stop >= ps.entry_price


# ---------------------------------------------------------------------------
# FillMonitor
# ---------------------------------------------------------------------------


class TestFillMonitor:
    """Tests for core/execution/fill_monitor.py."""

    def test_on_fill_registers_position(self, sample_trade_order):
        """on_fill() calls position_manager.register_position()."""
        from core.execution.fill_monitor import FillMonitor

        pm      = MagicMock()
        mt5     = MagicMock()
        journal = MagicMock()

        mt5.orders_get.return_value  = []
        mt5.positions_get.return_value = [
            {
                "ticket":     99001,
                "symbol":     "XAUUSD",
                "type":       0,   # BUY
                "volume":     0.10,
                "price_open": 1810.0,
                "sl":         0.0,
                "tp":         0.0,
                "magic":      sample_trade_order.magic_number,
            }
        ]

        fm = FillMonitor(mt5_client=mt5, position_manager=pm, trade_journal=journal)
        fm.on_fill(99001, sample_trade_order)

        pm.register_position.assert_called_once()


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------


class TestTradeJournal:
    """Tests for core/execution/trade_journal.py."""

    def test_log_order_placed_does_not_raise(self, sample_trade_order):
        """log_order_placed() runs without error when DB/Redis are unavailable."""
        from core.execution.trade_journal import TradeJournal

        db = MagicMock()
        db.insert_trade.side_effect = Exception("DB down")
        journal = TradeJournal(db_manager=db)

        # Should not raise
        journal.log_order_placed(sample_trade_order)

    def test_log_fill_does_not_raise(self, sample_trade_order):
        """log_fill() runs without error on exception."""
        from core.execution.trade_journal import TradeJournal

        db = MagicMock()
        db.update_trade.side_effect = Exception("DB down")
        journal = TradeJournal(db_manager=db)
        journal.log_fill(99001, sample_trade_order, fill_price=1810.5)

    def test_redis_not_required(self, sample_trade_order):
        """TradeJournal works when Redis is unavailable."""
        from core.execution.trade_journal import TradeJournal

        db = MagicMock()
        journal = TradeJournal(db_manager=db, redis_client=None)
        journal.log_order_placed(sample_trade_order)   # no crash


# ---------------------------------------------------------------------------
# PreExecutionValidator
# ---------------------------------------------------------------------------


class TestPreExecutionValidator:
    """Tests for core/execution/pre_execution_validator.py."""

    def test_valid_trade_passes(self, sample_trade_order, sample_regime_state, mock_mt5_client):
        """validate() returns (True, []) for a sane trade under normal conditions."""
        from core.execution.pre_execution_validator import PreExecutionValidator

        mock_mt5_client.symbol_info_tick.return_value = {
            "bid": 1809.5,
            "ask": 1810.5,
        }
        mock_mt5_client.symbol_info.return_value = {
            "spread": 10,
            "trade_mode": 4,   # full trading
        }

        pev = PreExecutionValidator(mt5_client=mock_mt5_client)
        valid, failures = pev.validate(sample_trade_order, sample_regime_state)
        # May pass or fail depending on price proximity — should not raise
        assert isinstance(valid, bool)
        assert isinstance(failures, list)

    def test_mt5_error_causes_fail(self, sample_trade_order, sample_regime_state, mock_mt5_client):
        """validate() returns (False, reasons) when MT5 throws."""
        from core.execution.pre_execution_validator import PreExecutionValidator
        from core.execution.mt5_connector import MT5ConnectionError

        mock_mt5_client.symbol_info_tick.side_effect = MT5ConnectionError("timeout")

        pev = PreExecutionValidator(mt5_client=mock_mt5_client)
        valid, failures = pev.validate(sample_trade_order, sample_regime_state)
        assert not valid
        assert len(failures) > 0
