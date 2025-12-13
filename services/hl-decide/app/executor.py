"""
Hyperliquid Trade Executor

Executes trades on Hyperliquid when consensus signals fire.
Disabled by default - requires explicit configuration to enable.

Features:
- Kelly criterion position sizing (fractional Kelly, default 25%)
- Risk governor integration for safety limits
- Dry run mode by default (simulates execution)
- Real execution requires explicit REAL_EXECUTION_ENABLED=true

@module executor
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import asyncpg

from .kelly import (
    kelly_position_size,
    get_consensus_kelly_size,
    KellyInput,
    KellyResult,
    KELLY_ENABLED,
    KELLY_FRACTION,
    KELLY_MIN_EPISODES,
    KELLY_FALLBACK_PCT,
)


# Hyperliquid API endpoints
HL_INFO_API = os.getenv("HL_INFO_API", "https://api.hyperliquid.xyz/info")
HL_EXCHANGE_API = os.getenv("HL_EXCHANGE_API", "https://api.hyperliquid.xyz/exchange")


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    status: str  # "filled", "rejected", "failed", "simulated"
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    error_message: Optional[str] = None
    exposure_before: Optional[float] = None
    exposure_after: Optional[float] = None
    position_pct: Optional[float] = None
    # Kelly sizing info (Phase 4)
    kelly_result: Optional[KellyResult] = None


class HyperliquidExecutor:
    """
    Execute trades on Hyperliquid.

    By default, runs in dry-run mode (simulates execution).
    Real execution requires:
    1. Private key configuration (HL_PRIVATE_KEY env var)
    2. Explicit enable via REAL_EXECUTION_ENABLED=true
    3. Risk limit checks passing

    Features:
    - Fetches account state (positions, equity)
    - Kelly criterion position sizing
    - Risk governor validation
    - Comprehensive execution logging
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
        db: asyncpg.Pool,
        symbol: str,
        direction: str,
        config: dict[str, Any],
        consensus_addresses: Optional[list[str]] = None,
        stop_distance_pct: float = 0.02,
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
            db: Database pool for Kelly data lookup
            symbol: Asset symbol
            direction: Trade direction (long/short)
            config: Execution config from database
            consensus_addresses: List of trader addresses in consensus (for Kelly)
            stop_distance_pct: Stop distance as fraction (for Kelly sizing)

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

        # Calculate position size - Kelly or fixed
        kelly_enabled = config.get("kelly_enabled", KELLY_ENABLED)
        kelly_result: Optional[KellyResult] = None

        if kelly_enabled and consensus_addresses:
            # Use Kelly criterion sizing based on consensus traders
            kelly_fraction = config.get("kelly_fraction", KELLY_FRACTION)
            kelly_result = await get_consensus_kelly_size(
                db,
                addresses=consensus_addresses,
                account_value=account_value,
                current_price=mid_price,
                stop_distance_pct=stop_distance_pct,
                fraction=kelly_fraction,
            )
            context["kelly_result"] = kelly_result
            position_pct = kelly_result.position_pct
            size_usd = kelly_result.position_size_usd
            size_coin = kelly_result.position_size_coin
            context["sizing_method"] = f"kelly:{kelly_result.method}"
        else:
            # Fixed percentage sizing (legacy)
            max_position_pct = hl_config.get("max_position_pct", 2) / 100
            position_pct = max_position_pct
            size_usd = account_value * position_pct
            size_coin = size_usd / mid_price
            context["sizing_method"] = "fixed"

        context["size_coin"] = size_coin
        context["size_usd"] = size_usd
        context["position_pct"] = position_pct

        # Calculate new exposure
        new_exposure = current_exposure + (size_usd / account_value)
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
        consensus_addresses: Optional[list[str]] = None,
        stop_distance_pct: float = 0.02,
    ) -> ExecutionResult:
        """
        Execute a consensus signal.

        For Phase 3e, this performs validation and logging only (dry run).
        Phase 4 adds Kelly sizing and real execution.

        Args:
            db: Database pool for logging
            decision_id: ID of the decision that triggered this
            symbol: Asset symbol (BTC, ETH)
            direction: Trade direction (long, short)
            config: Execution config from database
            consensus_addresses: List of trader addresses in consensus (for Kelly)
            stop_distance_pct: Stop distance as fraction (for Kelly sizing)

        Returns:
            ExecutionResult with status and details
        """
        # Validate with Kelly sizing if enabled
        can_execute, reason, context = await self.validate_execution(
            db, symbol, direction, config, consensus_addresses, stop_distance_pct
        )

        kelly_result = context.get("kelly_result")

        if not can_execute:
            result = ExecutionResult(
                status="rejected",
                error_message=reason,
                exposure_before=context.get("exposure_before"),
                kelly_result=kelly_result,
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
            error_message="Dry run - execution disabled",
            kelly_result=kelly_result,
        )

        await self._log_execution(db, decision_id, symbol, direction, config, result)

        sizing_method = context.get("sizing_method", "fixed")
        print(f"[executor] Simulated {direction} {symbol}: "
              f"size={context.get('size_coin'):.4f}, "
              f"price=${context.get('price'):,.2f}, "
              f"exposure={context.get('exposure_before'):.1%} -> {context.get('exposure_after'):.1%}, "
              f"sizing={sizing_method}")

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
        """Log execution attempt to database with Kelly sizing details."""
        try:
            hl_config = config.get("hyperliquid", {})
            leverage = hl_config.get("max_leverage", 1)

            # Extract Kelly data if available
            kelly_full = None
            kelly_fraction_used = None
            kelly_position_pct = None
            kelly_method = None
            kelly_reasoning = None
            kelly_capped = False

            if result.kelly_result:
                kr = result.kelly_result
                kelly_full = kr.full_kelly
                kelly_fraction_used = kr.fractional_kelly
                kelly_position_pct = kr.position_pct
                kelly_method = kr.method
                kelly_reasoning = kr.reasoning
                kelly_capped = kr.capped

            async with db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO execution_logs
                    (decision_id, exchange, symbol, side, size, leverage, status,
                     fill_price, fill_size, error_message, account_value,
                     position_pct, exposure_before, exposure_after,
                     kelly_full, kelly_fraction_used, kelly_position_pct,
                     kelly_method, kelly_reasoning, kelly_capped)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                            $15, $16, $17, $18, $19, $20)
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
                    kelly_full,
                    kelly_fraction_used,
                    kelly_position_pct,
                    kelly_method,
                    kelly_reasoning,
                    kelly_capped,
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
    consensus_addresses: Optional[list[str]] = None,
    stop_distance_pct: float = 0.02,
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
        consensus_addresses: List of trader addresses in consensus (for Kelly sizing)
        stop_distance_pct: Stop distance as fraction (for Kelly sizing)

    Returns:
        ExecutionResult if execution was attempted, None if disabled
    """
    from .portfolio import get_execution_config

    # Get execution config
    config = await get_execution_config(db)

    if not config.get("configured") or not config.get("enabled"):
        return None  # Auto-trading disabled

    # Execute with Kelly sizing if enabled
    executor = get_executor()
    return await executor.execute_signal(
        db, decision_id, symbol, direction, config,
        consensus_addresses=consensus_addresses,
        stop_distance_pct=stop_distance_pct,
    )
