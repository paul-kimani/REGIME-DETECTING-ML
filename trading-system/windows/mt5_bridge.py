"""FastAPI bridge server wrapping MetaTrader5 Python library for remote access from Mac."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MT5_BRIDGE_HOST: str = os.getenv("MT5_BRIDGE_HOST", "0.0.0.0")
MT5_BRIDGE_PORT: int = int(os.getenv("MT5_BRIDGE_PORT", "8000"))
MT5_BRIDGE_API_KEY: str = os.getenv("MT5_BRIDGE_API_KEY", "")
MT5_ALLOWED_IP_RAW: str = os.getenv("MT5_ALLOWED_IP", "")
MT5_ALLOWED_IPS: List[str] = (
    [ip.strip() for ip in MT5_ALLOWED_IP_RAW.split(",") if ip.strip()]
    if MT5_ALLOWED_IP_RAW
    else []
)
MT5_LOGIN: Optional[int] = int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None
MT5_PASSWORD: Optional[str] = os.getenv("MT5_PASSWORD")
MT5_SERVER: Optional[str] = os.getenv("MT5_SERVER")

if not MT5_BRIDGE_API_KEY:
    raise RuntimeError(
        "MT5_BRIDGE_API_KEY is required. Set it in the .env file or as an environment variable."
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mt5_bridge")

# ---------------------------------------------------------------------------
# MetaTrader5 import — guarded so Mac doesn't crash at import time
# ---------------------------------------------------------------------------

try:
    import MetaTrader5 as mt5  # type: ignore[import]

    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 module not available on this platform. "
        "Endpoints will return 503 when called."
    )


def _require_mt5() -> Any:
    """Return the mt5 module or raise HTTP 503 if unavailable."""
    if not _MT5_AVAILABLE or mt5 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MetaTrader5 module is not available on this platform.",
        )
    return mt5


# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

_TF_MAP: Dict[str, Any] = {}


def _get_tf_map() -> Dict[str, Any]:
    if _MT5_AVAILABLE and mt5 is not None and not _TF_MAP:
        _TF_MAP.update(
            {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
        )
    return _TF_MAP


def _resolve_tf(tf_str: str) -> Any:
    """Resolve a timeframe string to an mt5 constant, or raise HTTP 400."""
    tf_map = _get_tf_map()
    if tf_str not in tf_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown timeframe '{tf_str}'. Valid values: {list(tf_map.keys())}",
        )
    return tf_map[tf_str]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str
    order_type: str  # BUY / SELL / BUY_LIMIT / SELL_LIMIT / BUY_STOP / SELL_STOP
    volume: float
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 0
    comment: str = ""
    deviation: int = Field(default=20)


class CancelRequest(BaseModel):
    ticket: int


class CloseRequest(BaseModel):
    ticket: int
    volume: Optional[float] = None  # full close if None


class PartialCloseRequest(BaseModel):
    ticket: int
    volume: float


class OrderResult(BaseModel):
    success: bool
    ticket: Optional[int]
    retcode: int
    comment: str


class CloseResult(BaseModel):
    success: bool
    ticket: Optional[int]
    retcode: int
    comment: str


class CancelResult(BaseModel):
    success: bool
    retcode: int
    comment: str


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MT5 Bridge")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Middleware — logging, API key, IP allow-list
# ---------------------------------------------------------------------------


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next: Any) -> Any:
    start = time.perf_counter()

    # IP allow-list check
    if MT5_ALLOWED_IPS:
        client_ip = request.client.host if request.client else ""
        if client_ip not in MT5_ALLOWED_IPS:
            logger.warning("Blocked request from disallowed IP: %s", client_ip)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "IP not allowed."},
            )

    # API key check
    api_key = request.headers.get("X-API-Key", "")
    if api_key != MT5_BRIDGE_API_KEY:
        logger.warning(
            "Invalid API key from %s",
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Invalid or missing API key."},
        )

    logger.info(
        "REQUEST  %s %s from %s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "RESPONSE %s %s → %s  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Background health-check loop
# ---------------------------------------------------------------------------

_health_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]


async def _mt5_health_loop() -> None:
    """Ping MT5 every 30 seconds and log the result."""
    while True:
        await asyncio.sleep(30)
        if not _MT5_AVAILABLE or mt5 is None:
            logger.warning("MT5 health-check: module unavailable")
            continue
        try:
            info = mt5.account_info()
            if info is not None:
                logger.info(
                    "MT5 health-check OK — server=%s balance=%.2f equity=%.2f",
                    info.server,
                    info.balance,
                    info.equity,
                )
            else:
                err = mt5.last_error()
                logger.warning("MT5 health-check: account_info() returned None — %s", err)
        except Exception as exc:  # noqa: BLE001
            logger.exception("MT5 health-check exception: %s", exc)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    global _health_task

    if _MT5_AVAILABLE and mt5 is not None:
        kwargs: Dict[str, Any] = {}
        if MT5_LOGIN:
            kwargs["login"] = MT5_LOGIN
        if MT5_PASSWORD:
            kwargs["password"] = MT5_PASSWORD
        if MT5_SERVER:
            kwargs["server"] = MT5_SERVER

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize() failed: %s", err)
        else:
            info = mt5.account_info()
            if info is not None:
                logger.info(
                    "MT5 initialized — server=%s login=%s balance=%.2f equity=%.2f",
                    info.server,
                    info.login,
                    info.balance,
                    info.equity,
                )
            else:
                logger.warning("MT5 initialized but account_info() returned None")
    else:
        logger.warning("Startup: MetaTrader5 module not available, skipping MT5 init.")

    _health_task = asyncio.create_task(_mt5_health_loop())
    logger.info("MT5 Bridge started on %s:%s", MT5_BRIDGE_HOST, MT5_BRIDGE_PORT)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _health_task
    if _health_task is not None:
        _health_task.cancel()
        try:
            await _health_task
        except asyncio.CancelledError:
            pass

    if _MT5_AVAILABLE and mt5 is not None:
        mt5.shutdown()
        logger.info("MT5 shutdown complete.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _position_to_dict(pos: Any) -> Dict[str, Any]:
    return {
        "ticket": pos.ticket,
        "time": datetime.utcfromtimestamp(pos.time).isoformat(),
        "type": pos.type,
        "magic": pos.magic,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "price_open": pos.price_open,
        "sl": pos.sl,
        "tp": pos.tp,
        "price_current": pos.price_current,
        "profit": pos.profit,
        "comment": pos.comment,
    }


def _order_to_dict(order: Any) -> Dict[str, Any]:
    return {
        "ticket": order.ticket,
        "time_setup": datetime.utcfromtimestamp(order.time_setup).isoformat(),
        "type": order.type,
        "magic": order.magic,
        "symbol": order.symbol,
        "volume_initial": order.volume_initial,
        "volume_current": order.volume_current,
        "price_open": order.price_open,
        "sl": order.sl,
        "tp": order.tp,
        "comment": order.comment,
        "state": order.state,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
@limiter.limit("100/minute")
async def health(request: Request) -> Dict[str, Any]:
    if not _MT5_AVAILABLE or mt5 is None:
        return {
            "status": "error",
            "mt5_connected": False,
            "account_server": "",
            "ping_ms": 0.0,
        }

    t0 = time.perf_counter()
    info = mt5.account_info()
    ping_ms = (time.perf_counter() - t0) * 1000

    if info is None:
        err = mt5.last_error()
        logger.warning("health: account_info() returned None — %s", err)
        return {
            "status": "error",
            "mt5_connected": False,
            "account_server": "",
            "ping_ms": round(ping_ms, 2),
        }

    return {
        "status": "ok",
        "mt5_connected": True,
        "account_server": info.server,
        "ping_ms": round(ping_ms, 2),
    }


@app.get("/candles/{symbol}/{tf}/{count}")
@limiter.limit("100/minute")
async def get_candles(
    request: Request,
    symbol: str,
    tf: str,
    count: int,
) -> List[Dict[str, Any]]:
    _mt5 = _require_mt5()
    timeframe = _resolve_tf(tf)

    rates = _mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        err = _mt5.last_error()
        logger.error(
            "copy_rates_from_pos failed for %s/%s count=%s — %s", symbol, tf, count, err
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching candles: {err}",
        )

    return [
        {
            "time": datetime.utcfromtimestamp(int(r["time"])).isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread": int(r["spread"]),
        }
        for r in rates
    ]


@app.get("/candles/{symbol}/{tf}/range")
@limiter.limit("100/minute")
async def get_candles_range(
    request: Request,
    symbol: str,
    tf: str,
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    _mt5 = _require_mt5()
    timeframe = _resolve_tf(tf)

    try:
        dt_from = datetime.fromisoformat(from_date)
        dt_to = datetime.fromisoformat(to_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {exc}. Use ISO 8601 (e.g. 2024-01-01T00:00:00).",
        ) from exc

    rates = _mt5.copy_rates_range(symbol, timeframe, dt_from, dt_to)
    if rates is None:
        err = _mt5.last_error()
        logger.error(
            "copy_rates_range failed for %s/%s [%s, %s] — %s",
            symbol,
            tf,
            from_date,
            to_date,
            err,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching candles range: {err}",
        )

    return [
        {
            "time": datetime.utcfromtimestamp(int(r["time"])).isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "tick_volume": int(r["tick_volume"]),
            "spread": int(r["spread"]),
        }
        for r in rates
    ]


@app.get("/tick/{symbol}")
@limiter.limit("100/minute")
async def get_tick(request: Request, symbol: str) -> Dict[str, Any]:
    _mt5 = _require_mt5()

    tick = _mt5.symbol_info_tick(symbol)
    if tick is None:
        err = _mt5.last_error()
        logger.error("symbol_info_tick failed for %s — %s", symbol, err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching tick for {symbol}: {err}",
        )

    spread = round((tick.ask - tick.bid) * 1e5, 1) if tick.ask and tick.bid else 0.0
    return {
        "symbol": symbol,
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": spread,
        "time": datetime.utcfromtimestamp(tick.time).isoformat(),
    }


@app.get("/account")
@limiter.limit("100/minute")
async def get_account(request: Request) -> Dict[str, Any]:
    _mt5 = _require_mt5()

    info = _mt5.account_info()
    if info is None:
        err = _mt5.last_error()
        logger.error("account_info() failed — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching account info: {err}",
        )

    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "margin_level": info.margin_level,
        "profit": info.profit,
        "currency": info.currency,
        "leverage": info.leverage,
    }


@app.get("/positions")
@limiter.limit("100/minute")
async def get_positions(request: Request) -> List[Dict[str, Any]]:
    _mt5 = _require_mt5()

    positions = _mt5.positions_get()
    if positions is None:
        err = _mt5.last_error()
        logger.error("positions_get() failed — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching positions: {err}",
        )

    return [_position_to_dict(p) for p in positions]


@app.get("/orders")
@limiter.limit("100/minute")
async def get_orders(request: Request) -> List[Dict[str, Any]]:
    _mt5 = _require_mt5()

    orders = _mt5.orders_get()
    if orders is None:
        err = _mt5.last_error()
        logger.error("orders_get() failed — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching orders: {err}",
        )

    return [_order_to_dict(o) for o in orders]


@app.get("/symbol/{symbol}")
@limiter.limit("100/minute")
async def get_symbol_info(request: Request, symbol: str) -> Dict[str, Any]:
    _mt5 = _require_mt5()

    info = _mt5.symbol_info(symbol)
    if info is None:
        err = _mt5.last_error()
        logger.error("symbol_info() failed for %s — %s", symbol, err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching symbol info for {symbol}: {err}",
        )

    return {
        "symbol": symbol,
        "digits": info.digits,
        "point": info.point,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
        "spread": info.spread,
    }


@app.post("/order/place", response_model=OrderResult)
@limiter.limit("100/minute")
async def place_order(request: Request, req: OrderRequest) -> OrderResult:
    _mt5 = _require_mt5()

    _ORDER_TYPE_MAP: Dict[str, Any] = {
        "BUY": _mt5.ORDER_TYPE_BUY,
        "SELL": _mt5.ORDER_TYPE_SELL,
        "BUY_LIMIT": _mt5.ORDER_TYPE_BUY_LIMIT,
        "SELL_LIMIT": _mt5.ORDER_TYPE_SELL_LIMIT,
        "BUY_STOP": _mt5.ORDER_TYPE_BUY_STOP,
        "SELL_STOP": _mt5.ORDER_TYPE_SELL_STOP,
    }

    if req.order_type not in _ORDER_TYPE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown order_type '{req.order_type}'. "
            f"Valid values: {list(_ORDER_TYPE_MAP.keys())}",
        )

    order_type = _ORDER_TYPE_MAP[req.order_type]
    is_market = req.order_type in ("BUY", "SELL")

    if is_market:
        tick = _mt5.symbol_info_tick(req.symbol)
        if tick is None:
            err = _mt5.last_error()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"MT5 error fetching tick for price: {err}",
            )
        price = tick.ask if req.order_type == "BUY" else tick.bid
        type_filling = _mt5.ORDER_FILLING_IOC
    else:
        if req.price is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="price is required for pending orders.",
            )
        price = req.price
        type_filling = _mt5.ORDER_FILLING_RETURN

    trade_request: Dict[str, Any] = {
        "action": _mt5.TRADE_ACTION_DEAL if is_market else _mt5.TRADE_ACTION_PENDING,
        "symbol": req.symbol,
        "volume": req.volume,
        "type": order_type,
        "price": price,
        "deviation": req.deviation,
        "magic": req.magic,
        "comment": req.comment,
        "type_time": _mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }

    if req.sl is not None:
        trade_request["sl"] = req.sl
    if req.tp is not None:
        trade_request["tp"] = req.tp

    logger.info("Placing order: %s", trade_request)
    result = _mt5.order_send(trade_request)

    if result is None:
        err = _mt5.last_error()
        logger.error("order_send() returned None — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error placing order: {err}",
        )

    success = result.retcode == _mt5.TRADE_RETCODE_DONE
    if not success:
        logger.warning(
            "order_send retcode=%s comment=%s last_error=%s",
            result.retcode,
            result.comment,
            _mt5.last_error(),
        )

    return OrderResult(
        success=success,
        ticket=result.order if success else None,
        retcode=result.retcode,
        comment=result.comment,
    )


@app.post("/order/cancel", response_model=CancelResult)
@limiter.limit("100/minute")
async def cancel_order(request: Request, req: CancelRequest) -> CancelResult:
    _mt5 = _require_mt5()

    trade_request: Dict[str, Any] = {
        "action": _mt5.TRADE_ACTION_REMOVE,
        "order": req.ticket,
    }

    logger.info("Cancelling order ticket=%s", req.ticket)
    result = _mt5.order_send(trade_request)

    if result is None:
        err = _mt5.last_error()
        logger.error("order_send (cancel) returned None — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error cancelling order: {err}",
        )

    success = result.retcode == _mt5.TRADE_RETCODE_DONE
    if not success:
        logger.warning(
            "cancel retcode=%s comment=%s last_error=%s",
            result.retcode,
            result.comment,
            _mt5.last_error(),
        )

    return CancelResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
    )


@app.post("/position/close", response_model=CloseResult)
@limiter.limit("100/minute")
async def close_position(request: Request, req: CloseRequest) -> CloseResult:
    _mt5 = _require_mt5()

    positions = _mt5.positions_get(ticket=req.ticket)
    if not positions:
        err = _mt5.last_error()
        logger.error("position not found for ticket=%s — %s", req.ticket, err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position with ticket {req.ticket} not found.",
        )

    pos = positions[0]
    volume = req.volume if req.volume is not None else pos.volume

    close_type = _mt5.ORDER_TYPE_SELL if pos.type == _mt5.ORDER_TYPE_BUY else _mt5.ORDER_TYPE_BUY
    tick = _mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        err = _mt5.last_error()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching tick for close: {err}",
        )

    price = tick.bid if pos.type == _mt5.ORDER_TYPE_BUY else tick.ask

    trade_request: Dict[str, Any] = {
        "action": _mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": volume,
        "type": close_type,
        "position": req.ticket,
        "price": price,
        "deviation": 20,
        "magic": pos.magic,
        "comment": "close",
        "type_time": _mt5.ORDER_TIME_GTC,
        "type_filling": _mt5.ORDER_FILLING_IOC,
    }

    logger.info("Closing position ticket=%s volume=%s", req.ticket, volume)
    result = _mt5.order_send(trade_request)

    if result is None:
        err = _mt5.last_error()
        logger.error("order_send (close) returned None — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error closing position: {err}",
        )

    success = result.retcode == _mt5.TRADE_RETCODE_DONE
    if not success:
        logger.warning(
            "close retcode=%s comment=%s last_error=%s",
            result.retcode,
            result.comment,
            _mt5.last_error(),
        )

    return CloseResult(
        success=success,
        ticket=result.order if success else None,
        retcode=result.retcode,
        comment=result.comment,
    )


@app.post("/position/partial_close", response_model=CloseResult)
@limiter.limit("100/minute")
async def partial_close_position(request: Request, req: PartialCloseRequest) -> CloseResult:
    _mt5 = _require_mt5()

    positions = _mt5.positions_get(ticket=req.ticket)
    if not positions:
        err = _mt5.last_error()
        logger.error("position not found for ticket=%s — %s", req.ticket, err)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position with ticket {req.ticket} not found.",
        )

    pos = positions[0]

    if req.volume >= pos.volume:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Partial close volume ({req.volume}) must be less than "
                f"position volume ({pos.volume}). Use /position/close for full close."
            ),
        )

    close_type = _mt5.ORDER_TYPE_SELL if pos.type == _mt5.ORDER_TYPE_BUY else _mt5.ORDER_TYPE_BUY
    tick = _mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        err = _mt5.last_error()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error fetching tick for partial close: {err}",
        )

    price = tick.bid if pos.type == _mt5.ORDER_TYPE_BUY else tick.ask

    trade_request: Dict[str, Any] = {
        "action": _mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": req.volume,
        "type": close_type,
        "position": req.ticket,
        "price": price,
        "deviation": 20,
        "magic": pos.magic,
        "comment": "partial_close",
        "type_time": _mt5.ORDER_TIME_GTC,
        "type_filling": _mt5.ORDER_FILLING_IOC,
    }

    logger.info("Partial close position ticket=%s volume=%s", req.ticket, req.volume)
    result = _mt5.order_send(trade_request)

    if result is None:
        err = _mt5.last_error()
        logger.error("order_send (partial_close) returned None — %s", err)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"MT5 error partially closing position: {err}",
        )

    success = result.retcode == _mt5.TRADE_RETCODE_DONE
    if not success:
        logger.warning(
            "partial_close retcode=%s comment=%s last_error=%s",
            result.retcode,
            result.comment,
            _mt5.last_error(),
        )

    return CloseResult(
        success=success,
        ticket=result.order if success else None,
        retcode=result.retcode,
        comment=result.comment,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=MT5_BRIDGE_HOST, port=MT5_BRIDGE_PORT)
