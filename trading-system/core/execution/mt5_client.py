"""DEPRECATED — MT5Client HTTP bridge client.

This file is no longer used. The HTTP bridge has been removed.

Use core/execution/mt5_connector.py instead:

    from core.execution.mt5_connector import MT5Connector, MT5ConnectionError

MT5Connector connects directly to the MetaTrader5 terminal via the
MetaTrader5 Python library — no HTTP, no network, no separate bridge process.
"""

raise ImportError(
    "mt5_client is deprecated. "
    "Use: from core.execution.mt5_connector import MT5Connector, MT5ConnectionError"
)
