"""Centralised config loader — merges YAML files, .env overrides, and validates required fields."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from core.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at: trading-system/core/utils/config.py
# Project root (trading-system/) is 2 levels up from this file's directory.
# parents[0] = utils/  parents[1] = core/  parents[2] = trading-system/
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"
_ENV_FILE: Path = _PROJECT_ROOT / ".env"

# YAML files to load, in the order they are merged into the config dict
_YAML_FILES: list[str] = [
    "assets.yaml",
    "risk_params.yaml",
    "regime_params.yaml",
    "signal_params.yaml",
    "prop_firm.yaml",
]

# Required top-level keys that must be present after loading
_REQUIRED_KEYS: list[str] = [
    "assets",
    "sizing",
    "volatility_scalars",
    "portfolio",
    "rr_minimums",
    "stop_limits",
    "regime_age_multipliers",
    "states",
    "hmm",
    "persistence",
    "mtf",
    "universal_model",
    "momentum",
    "mean_reversion",
    "breakout",
    "hold_time_limits",
    "mode",
    "ftmo",
    "internal_buffers",
    "circuit_breakers",
    "recovery_sizing",
    "news_filter",
]

# Map ENV_VAR_NAME -> dot-separated config path it overrides.
# The convention used: the environment variable name is the uppercased,
# underscore-joined path segments, e.g. MAX_RISK_PER_TRADE maps to
# sizing.max_risk_per_trade.  Additional explicit mappings can be added here.
_ENV_OVERRIDES: dict[str, str] = {
    # sizing
    "BASE_RISK_PER_TRADE": "sizing.base_risk_per_trade",
    "MAX_RISK_PER_TRADE": "sizing.max_risk_per_trade",
    "MIN_RISK_PER_TRADE": "sizing.min_risk_per_trade",
    "KELLY_FRACTION": "sizing.kelly_fraction",
    "KELLY_MIN_CONFIDENCE": "sizing.kelly_min_confidence",
    # portfolio
    "MAX_TOTAL_HEAT": "portfolio.max_total_heat",
    "MAX_LONG_EXPOSURE": "portfolio.max_long_exposure",
    "MAX_SHORT_EXPOSURE": "portfolio.max_short_exposure",
    # prop firm
    "PROP_FIRM_MODE": "mode",
    # logging / misc
    "LOG_DIR": None,  # consumed by logger; not injected into config dict
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
    """Raised when a required configuration key is absent after loading."""


# ---------------------------------------------------------------------------
# _ConfigNode — dot-notation access wrapper
# ---------------------------------------------------------------------------


class _ConfigNode:
    """Wrap a plain dict so that its keys are accessible via attribute access.

    Nested dicts are wrapped recursively, giving transparent dot-notation
    access to arbitrarily deep config trees::

        node.sizing.max_risk_per_trade  # -> float

    Non-dict values are returned as-is.  Unknown attributes raise
    :class:`AttributeError`.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise the node, recursively wrapping nested dicts.

        Args:
            data: A plain :class:`dict` of configuration values.
        """
        # Store raw dict for serialisation / iteration
        object.__setattr__(self, "_data", data)
        for key, value in data.items():
            object.__setattr__(self, key, _wrap(value))

    # ---- attribute access --------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"Config has no attribute '{name}'. "
            f"Available keys: {list(self._data.keys())}"
        )

    def __repr__(self) -> str:
        return f"_ConfigNode({list(self._data.keys())})"

    # ---- dict-like helpers -------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present.

        Args:
            key:     Top-level attribute name.
            default: Fallback value when the key is absent.

        Returns:
            The value stored under *key*, or *default*.
        """
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return the underlying raw dict.

        Returns:
            The original :class:`dict` that backs this node.
        """
        return self._data  # type: ignore[return-value]


def _wrap(value: Any) -> Any:
    """Recursively wrap *value* in :class:`_ConfigNode` if it is a dict."""
    if isinstance(value, dict):
        return _ConfigNode(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Config — main class
# ---------------------------------------------------------------------------


class Config:
    """Load, merge, validate and expose all YAML configuration files.

    Loading sequence
    ----------------
    1. ``.env`` is loaded via :func:`dotenv.load_dotenv` so that environment
       variables are available before any YAML is read.
    2. Each YAML file in :data:`_YAML_FILES` is merged (shallow at the top
       level) into a single dict.
    3. Explicit environment-variable overrides are applied (see
       :data:`_ENV_OVERRIDES`).
    4. Required keys are validated; :class:`ConfigValidationError` is raised
       if any are missing.
    5. Every top-level key is exposed as a :class:`_ConfigNode` attribute,
       giving dot-notation access to nested values.

    Attributes:
        assets (list):                  List of asset configs.
        sizing (_ConfigNode):           Position-sizing parameters.
        volatility_scalars (_ConfigNode): Vol scalar thresholds.
        portfolio (_ConfigNode):        Portfolio-level heat limits.
        rr_minimums (_ConfigNode):      Per-strategy R:R minimums.
        stop_limits (_ConfigNode):      ATR stop-loss multiplier caps.
        regime_age_multipliers (_ConfigNode): Regime-age size scalars.
        states (_ConfigNode):           HMM state integer mappings.
        hmm (_ConfigNode):              HMM training parameters.
        persistence (_ConfigNode):      Regime persistence thresholds.
        mtf (_ConfigNode):              Multi-timeframe weights.
        universal_model (_ConfigNode):  Universal risk-on/off model.
        momentum (_ConfigNode):         Momentum strategy params.
        mean_reversion (_ConfigNode):   Mean-reversion strategy params.
        breakout (_ConfigNode):         Breakout strategy params.
        hold_time_limits (_ConfigNode): Max candle hold times.
        mode (bool):                    Prop-firm mode flag.
        ftmo (_ConfigNode):             FTMO rule parameters.
        internal_buffers (_ConfigNode): Internal safety buffers.
        circuit_breakers (_ConfigNode): Circuit-breaker levels.
        recovery_sizing (_ConfigNode):  Post-loss recovery multipliers.
        news_filter (_ConfigNode):      News-event filter settings.
    """

    def __init__(self) -> None:
        """Load all YAML files, apply env overrides, and validate the config."""
        self._log: logging.Logger = get_logger(__name__)
        self._raw: dict[str, Any] = {}

        self._load_env()
        self._load_yaml_files()
        self._apply_env_overrides()
        self._validate()
        self._expose_attributes()

        self._log.debug(
            "Config loaded. Top-level keys: %s",
            sorted(self._raw.keys()),
        )

    # ---- loading -----------------------------------------------------------

    def _load_env(self) -> None:
        """Load the ``.env`` file if it exists.

        Uses :func:`dotenv.load_dotenv` so that already-set environment
        variables are not overwritten.
        """
        if _ENV_FILE.exists():
            load_dotenv(dotenv_path=_ENV_FILE, override=False)
            self._log.debug("Loaded .env from %s", _ENV_FILE)
        else:
            self._log.debug(".env not found at %s — skipping", _ENV_FILE)

    def _load_yaml_files(self) -> None:
        """Load each YAML file in :data:`_YAML_FILES` and merge into one dict.

        Top-level keys are merged with a last-writer-wins strategy.  Duplicate
        keys across different files result in the later file's values taking
        precedence; a DEBUG warning is emitted.

        Raises:
            FileNotFoundError: If a required YAML file is not found under
                :data:`_CONFIGS_DIR`.
            yaml.YAMLError: If a YAML file contains a syntax error.
        """
        for filename in _YAML_FILES:
            path = _CONFIGS_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Required config file not found: {path}\n"
                    f"Expected directory: {_CONFIGS_DIR}"
                )
            with path.open("r", encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}

            overlap = set(self._raw) & set(data)
            if overlap:
                self._log.debug(
                    "Key overlap when loading %s (keys will be overwritten): %s",
                    filename,
                    sorted(overlap),
                )

            self._raw.update(data)
            self._log.debug(
                "Loaded %s — keys added/updated: %s",
                filename,
                sorted(data.keys()),
            )

    def _apply_env_overrides(self) -> None:
        """Apply environment-variable overrides defined in :data:`_ENV_OVERRIDES`.

        Each mapping ``ENV_VAR -> "section.key"`` is checked; if the
        environment variable is set, its string value is cast to the same type
        as the existing config value (int, float, bool, or str) and written
        into ``self._raw``.

        Unknown section/key paths are logged at WARNING level and skipped.
        """
        for env_var, config_path in _ENV_OVERRIDES.items():
            env_value = os.environ.get(env_var)
            if env_value is None or config_path is None:
                continue

            parts = config_path.split(".")
            # Navigate to the parent dict
            container: Any = self._raw
            try:
                for part in parts[:-1]:
                    container = container[part]
                leaf_key = parts[-1]
            except (KeyError, TypeError):
                self._log.warning(
                    "ENV override %s -> %s: path not found in config; skipping",
                    env_var,
                    config_path,
                )
                continue

            if not isinstance(container, dict):
                self._log.warning(
                    "ENV override %s -> %s: parent is not a dict; skipping",
                    env_var,
                    config_path,
                )
                continue

            # Cast to existing value type where possible
            existing = container.get(leaf_key)
            try:
                if isinstance(existing, bool):
                    cast_value: Any = env_value.lower() in ("1", "true", "yes")
                elif isinstance(existing, int):
                    cast_value = int(env_value)
                elif isinstance(existing, float):
                    cast_value = float(env_value)
                else:
                    cast_value = env_value
            except (ValueError, TypeError):
                self._log.warning(
                    "ENV override %s=%s could not be cast to %s; using str",
                    env_var,
                    env_value,
                    type(existing).__name__,
                )
                cast_value = env_value

            container[leaf_key] = cast_value
            self._log.debug(
                "ENV override applied: %s -> %s = %r",
                env_var,
                config_path,
                cast_value,
            )

    # ---- validation --------------------------------------------------------

    def _validate(self) -> None:
        """Check that all required top-level keys are present.

        Raises:
            ConfigValidationError: If one or more required keys are absent,
                with a clear message listing every missing key.
        """
        missing = [k for k in _REQUIRED_KEYS if k not in self._raw]
        if missing:
            raise ConfigValidationError(
                f"Config validation failed — the following required keys are "
                f"missing after loading all YAML files and env overrides:\n"
                f"  {missing}\n"
                f"Config directory: {_CONFIGS_DIR}\n"
                f"Loaded files: {_YAML_FILES}"
            )

    # ---- attribute exposure ------------------------------------------------

    def _expose_attributes(self) -> None:
        """Wrap each top-level raw dict value in :class:`_ConfigNode` and bind it."""
        for key, value in self._raw.items():
            setattr(self, key, _wrap(value))

    # ---- public helpers ----------------------------------------------------

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Return the top-level config attribute *key*, or *default* if absent.

        This is a safe alternative to direct attribute access for keys whose
        presence is not guaranteed.

        Args:
            key:     Top-level config key (e.g. ``"sizing"``).
            default: Value to return when *key* is not present.

        Returns:
            The value stored under *key*, or *default*.

        Example::

            risk = config.get("sizing")
            exotic = config.get("exotic_params", default={})
        """
        return getattr(self, key, default)

    def reload(self) -> None:
        """Re-read all YAML files and re-apply env overrides in place.

        Useful for long-running processes that need to pick up config changes
        without restarting.  The singleton instance is updated in-place; any
        references to the instance will see the new values immediately.

        Raises:
            FileNotFoundError: If a YAML file has been removed since the last
                load.
            ConfigValidationError: If the reloaded config is no longer valid.
        """
        self._log.info("Reloading configuration from disk...")
        self._raw = {}
        self._load_env()
        self._load_yaml_files()
        self._apply_env_overrides()
        self._validate()
        self._expose_attributes()
        self._log.info(
            "Configuration reloaded. Top-level keys: %s",
            sorted(self._raw.keys()),
        )

    def __repr__(self) -> str:
        return f"Config(keys={sorted(self._raw.keys())})"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[Config] = None


def get_config() -> Config:
    """Return the shared :class:`Config` singleton.

    The instance is created on the first call and reused on all subsequent
    calls, ensuring that YAML files and environment variables are only read
    once per process.

    Returns:
        The application-wide :class:`Config` instance.

    Raises:
        FileNotFoundError:      If a required YAML file is missing.
        ConfigValidationError:  If a required config key is absent.

    Example::

        from core.utils.config import get_config

        config = get_config()
        max_risk = config.sizing.max_risk_per_trade   # 0.015
        assets   = config.assets                      # list of _ConfigNode
        mode     = config.mode                        # False (or True on prop firm)
    """
    global _instance
    if _instance is None:
        _instance = Config()
    return _instance
