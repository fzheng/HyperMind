"""
Hyperliquid Trade Executor

Executes trades on Hyperliquid when consensus signals fire.
Disabled by default - requires explicit configuration to enable.

Phase 3e: Hyperliquid only. Multi-exchange in Phase 4.

@module executor
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import asyncpg


# Hyperliquid API endpoints
HL_INFO_API = os.getenv("HL_INFO_API", "https://api.hyperliquid.xyz/info")
HL_EXCHANGE_API = os.getenv("HL_EXCHANGE_API", "https://api.hyperliquid.xyz/exchange")


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    status: str  # "filled", "rejected", "failed"
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    error_message: Optional[str] = None
    exposure_before: Optional[float] = None
    exposure_after: Optional[float] = None
    position_pct: Optional[float] = None


class HyperliquidExecutor:
    """
    Execute trades on Hyperliquid.

    This is a READ-ONLY implementation for Phase 3e.
    Actual trade execution requires:
    1. Private key configuration (secure storage)
    2. Explicit enable via execution_config
    3. Risk limit checks passing

    For now, this class:
    - Fetches account state (positions, equity)
    - Validates risk limits
    - Logs what would be executed (dry run)
    """

    def __init__(self, address: Optional[str] = None):
        """
        Initialize executor.

        Args:
            address: Hyperliquid wallet address for account state queries
        """
        self.address = address
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_account_state(self) -> Optional[dict]:
        """
        Fetch account state from Hyperliquid.

        Returns:
            Account state dict or None if failed
        """
        if not self.address:
            return None

        try:
            client = await self._get_client()
            payload = {
                "type": "clearinghouseState",
                "user": self.address,
            }
            resp = await client.post(HL_INFO_API, json=payload)
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"[executor] Failed to fetch account state: {e}")
            return None

    async def get_account_value(self) -> float:
        """
        Get current account value.

        Returns:
            Account value in USD, or 0 if unavailable
        """
        state = await self.get_account_state()
        if not state:
            return 0.0

        margin_summary = state.get("marginSummary", {})
        return float(margin_summary.get("accountValue", 0))

    async def get_current_exposure(self) -> float:
        """
        Get current total exposure as fraction of equity.

        Returns:
            Exposure ratio (0-1+), or 0 if unavailable
        """
        state = await self.get_account_state()
        if not state:
            return 0.0

        margin_summary = state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))

        if account_value <= 0:
            return 0.0

        # Calculate total notional exposure
        total_notional = 0.0
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            size = abs(float(pos.get("szi", 0)))
            entry_price = float(pos.get("entryPx", 0))
            total_notional += size * entry_price

        return total_notional / account_value

    async def get_mid_price(self, symbol: str) -> Optional[float]:
        """
        Get current mid price for a symbol.

        Args:
            symbol: Asset symbol (BTC, ETH)

        Returns:
            Mid price or None if unavailable
        """
        try:
            client = await self._get_client()
            payload = {"type": "allMids"}
            resp = await client.post(HL_INFO_API, json=payload)
            if resp.status_code == 200:
                mids = resp.json()
                return float(mids.get(symbol, 0))
            return None
        except Exception as e:
            print(f"[executor] Failed to fetch mid price: {e}")
            return None

    async def validate_execution(
        self,
        symbol: str,
        direction: str,
        config: dict[str, Any],
    ) -> tuple[bool, str, dict]:
        """
        Validate if execution should proceed.

        Checks:
        1. Executor enabled
        2. Hyperliquid enabled
        3. Address configured
        4. Exposure limits
        5. Account has value

        Args:
            symbol: Asset symbol
            direction: Trade direction (long/short)
            config: Execution config from database

        Returns:
            Tuple of (can_execute, reason, context_dict)
        """
        context = {}

        # Check master enable
        if not config.get("enabled"):
            return False, "Auto-trading disabled", context

        # Check Hyperliquid enable
        hl_config = config.get("hyperliquid", {})
        if not hl_config.get("enabled"):
            return False, "Hyperliquid trading disabled", context

        # Check address configured
        address = hl_config.get("address")
        if not address:
            return False, "No Hyperliquid address configured", context

        self.address = address

        # Get account state
        account_value = await self.get_account_value()
        context["account_value"] = account_value

        if account_value <= 0:
            return False, "No account value", context

        # Check exposure limit
        current_exposure = await self.get_current_exposure()
        context["exposure_before"] = current_exposure

        max_exposure = hl_config.get("max_exposure_pct", 10) / 100  # Convert from % to decimal
        if current_exposure >= max_exposure:
            return False, f"Exposure {current_exposure:.1%} >= {max_exposure:.1%} limit", context

        # Get price
        mid_price = await self.get_mid_price(symbol)
        if not mid_price or mid_price <= 0:
            return False, f"Could not get price for {symbol}", context
        context["price"] = mid_price

        # Calculate position size
        max_position_pct = hl_config.get("max_position_pct", 2) / 100
        max_size_usd = account_value * max_position_pct
        size_coin = max_size_usd / mid_price
        context["size_coin"] = size_coin
        context["size_usd"] = max_size_usd
        context["position_pct"] = max_position_pct

        # Calculate new exposure
        new_exposure = current_exposure + (max_size_usd / account_value)
        context["exposure_after"] = new_exposure

        if new_exposure > max_exposure:
            return False, f"Trade would exceed exposure limit ({new_exposure:.1%} > {max_exposure:.1%})", context

        return True, "Validation passed", context

    async def execute_signal(
        self,
        db: asyncpg.Pool,
        decision_id: str,
        symbol: str,
        direction: str,
        config: dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute a consensus signal.

        For Phase 3e, this performs validation and logging only (dry run).
        Actual execution requires private key integration (Phase 4).

        Args:
            db: Database pool for logging
            decision_id: ID of the decision that triggered this
            symbol: Asset symbol (BTC, ETH)
            direction: Trade direction (long, short)
            config: Execution config from database

        Returns:
            ExecutionResult with status and details
        """
        # Validate
        can_execute, reason, context = await self.validate_execution(
            symbol, direction, config
        )

        if not can_execute:
            result = ExecutionResult(
                status="rejected",
                error_message=reason,
                exposure_before=context.get("exposure_before"),
            )
            await self._log_execution(db, decision_id, symbol, direction, config, result)
            return result

        # For Phase 3e: Dry run only
        # In Phase 4, this would call the Hyperliquid SDK to place orders

        result = ExecutionResult(
            status="simulated",  # Would be "filled" with real execution
            fill_price=context.get("price"),
            fill_size=context.get("size_coin"),
            exposure_before=context.get("exposure_before"),
            exposure_after=context.get("exposure_after"),
            position_pct=context.get("position_pct"),
            error_message="Dry run - execution disabled in Phase 3e",
        )

        await self._log_execution(db, decision_id, symbol, direction, config, result)

        print(f"[executor] Simulated {direction} {symbol}: "
              f"size={context.get('size_coin'):.4f}, "
              f"price=${context.get('price'):,.2f}, "
              f"exposure={context.get('exposure_before'):.1%} -> {context.get('exposure_after'):.1%}")

        return result

    async def _log_execution(
        self,
        db: asyncpg.Pool,
        decision_id: str,
        symbol: str,
        direction: str,
        config: dict[str, Any],
        result: ExecutionResult,
    ) -> None:
        """Log execution attempt to database."""
        try:
            hl_config = config.get("hyperliquid", {})
            leverage = hl_config.get("max_leverage", 1)

            async with db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO execution_logs
                    (decision_id, exchange, symbol, side, size, leverage, status,
                     fill_price, fill_size, error_message, account_value,
                     position_pct, exposure_before, exposure_after)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    decision_id,
                    "hyperliquid",
                    symbol,
                    "buy" if direction == "long" else "sell",
                    result.fill_size or 0,
                    leverage,
                    result.status,
                    result.fill_price,
                    result.fill_size,
                    result.error_message,
                    None,  # account_value - could be added
                    result.position_pct,
                    result.exposure_before,
                    result.exposure_after,
                )
        except Exception as e:
            print(f"[executor] Failed to log execution: {e}")


# Global executor instance (initialized on startup)
_executor: Optional[HyperliquidExecutor] = None


def get_executor() -> HyperliquidExecutor:
    """Get or create global executor instance."""
    global _executor
    if _executor is None:
        _executor = HyperliquidExecutor()
    return _executor


async def maybe_execute_signal(
    db: asyncpg.Pool,
    decision_id: str,
    symbol: str,
    direction: str,
) -> Optional[ExecutionResult]:
    """
    Execute a signal if auto-trading is enabled.

    This is the main entry point for signal execution.
    Called from consensus detection after a signal is logged.

    Args:
        db: Database pool
        decision_id: The decision log ID
        symbol: Asset symbol (BTC, ETH)
        direction: Trade direction (long, short)

    Returns:
        ExecutionResult if execution was attempted, None if disabled
    """
    from .portfolio import get_execution_config

    # Get execution config
    config = await get_execution_config(db)

    if not config.get("configured") or not config.get("enabled"):
        return None  # Auto-trading disabled

    # Execute
    executor = get_executor()
    return await executor.execute_signal(db, decision_id, symbol, direction, config)
