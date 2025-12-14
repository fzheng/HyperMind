"""
Multi-Exchange Trade Executor

Executes trades on configured exchanges when consensus signals fire.
Disabled by default - requires explicit configuration to enable.

Features:
- Kelly criterion position sizing (fractional Kelly, default 25%)
- Risk governor integration for safety limits
- Dry run mode by default (simulates execution)
- Real execution requires explicit REAL_EXECUTION_ENABLED=true
- Multi-exchange support via ExchangeManager (Phase 6)

Supported Exchanges:
- Hyperliquid (DEX) - default
- Aster (DEX)
- Bybit (CEX)

@module executor
"""

import asyncio
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
from .exchanges import (
    ExchangeType,
    ExchangeManager,
    OrderParams,
    OrderResult as ExchangeOrderResult,
    OrderSide,
    get_exchange_manager,
    get_fee_config,
)
from .account_normalizer import get_account_normalizer


# Hyperliquid API endpoints
HL_INFO_API = os.getenv("HL_INFO_API", "https://api.hyperliquid.xyz/info")
HL_EXCHANGE_API = os.getenv("HL_EXCHANGE_API", "https://api.hyperliquid.xyz/exchange")

# Retry configuration for account state fetch
ACCOUNT_STATE_MAX_RETRIES = int(os.getenv("ACCOUNT_STATE_MAX_RETRIES", "3"))
ACCOUNT_STATE_BASE_DELAY_MS = int(os.getenv("ACCOUNT_STATE_BASE_DELAY_MS", "500"))


# Safety block metrics (lazy-loaded to avoid circular imports)
_safety_block_counter = None


def get_safety_block_counter():
    """Get or create the safety block counter (lazy load)."""
    global _safety_block_counter
    if _safety_block_counter is None:
        try:
            from prometheus_client import Counter
            # Import registry from main to share metrics
            from .main import registry
            _safety_block_counter = Counter(
                "decide_safety_block_total",
                "Execution blocked by safety checks",
                labelnames=["guard"],  # kill_switch, account_state, risk_governor, circuit_breaker
                registry=registry,
            )
        except Exception as e:
            print(f"[executor] Failed to initialize safety metrics: {e}")
            # Return a dummy counter that does nothing
            class DummyCounter:
                def labels(self, **kwargs):
                    return self
                def inc(self):
                    pass
            _safety_block_counter = DummyCounter()
    return _safety_block_counter


def increment_safety_block(guard: str):
    """Increment safety block counter for a specific guard."""
    try:
        get_safety_block_counter().labels(guard=guard).inc()
    except Exception:
        pass  # Don't let metrics failures affect execution


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

    async def get_account_state(
        self,
        exchange_type: Optional[ExchangeType] = None,
    ) -> Optional[dict]:
        """
        Fetch account state from exchange.

        If ExchangeManager has the exchange connected, uses that.
        Falls back to direct Hyperliquid API for backward compatibility.

        Args:
            exchange_type: Target exchange (None = Hyperliquid)

        Returns:
            Account state dict or None if failed
        """
        # Try ExchangeManager first (multi-exchange path)
        if exchange_type is not None:
            exchange_manager = get_exchange_manager()
            exchange = exchange_manager.get_exchange(exchange_type)
            if exchange and exchange.is_connected:
                try:
                    balance = await exchange.get_balance()
                    positions = await exchange.get_positions()
                    if balance:
                        # Convert to Hyperliquid-like format for backward compatibility
                        return self._to_hl_account_state(balance, positions)
                except Exception as e:
                    print(f"[executor] Failed to fetch from {exchange_type.value}: {e}")
                    # Fall through to legacy path

        # Legacy Hyperliquid API path
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

    def _to_hl_account_state(self, balance, positions: list) -> dict:
        """
        Convert ExchangeManager balance/positions to Hyperliquid-like format.

        This enables backward compatibility with existing risk checks.
        All values are normalized to USD using AccountNormalizer (Phase 6.1.5).
        """
        # Normalize balance to USD (handles USDT → USD conversion for Bybit)
        normalizer = get_account_normalizer()
        normalized = normalizer.normalize_balance_sync(balance)

        # Build assetPositions array
        asset_positions = []
        for pos in positions:
            # Normalize position notional to USD
            norm_pos = normalizer.normalize_position_sync(pos, quote_currency=balance.currency)
            asset_positions.append({
                "position": {
                    "coin": pos.symbol,
                    "szi": str(pos.size),
                    "entryPx": str(pos.entry_price),
                    "leverage": {"type": "isolated", "value": pos.leverage},
                    "liquidationPx": str(pos.liquidation_price) if pos.liquidation_price else None,
                    "unrealizedPnl": str(pos.unrealized_pnl * normalized.conversion_rate),
                    "notionalValueUsd": str(norm_pos.notional_value_usd),  # Added for clarity
                }
            })

        return {
            "marginSummary": {
                "accountValue": str(normalized.total_equity_usd),  # USD-normalized
                "totalMarginUsed": str(normalized.margin_used_usd),  # USD-normalized
                "totalNtlPos": str(sum(
                    normalizer.normalize_position_sync(p, balance.currency).notional_value_usd
                    for p in positions
                )),
            },
            "assetPositions": asset_positions,
            # Normalization metadata for audit trail
            "_normalization": {
                "original_currency": balance.currency,
                "conversion_rate": normalized.conversion_rate,
                "conversion_source": normalized.conversion_source,
                "is_depeg_warning": normalized.is_depeg_warning,
            },
        }

    async def get_account_state_with_retry(
        self,
        exchange_type: Optional[ExchangeType] = None,
    ) -> dict:
        """
        Fetch account state with exponential backoff retry.

        Retries up to ACCOUNT_STATE_MAX_RETRIES times with exponential backoff.
        Raises exception if all retries fail (fail-closed behavior).

        Args:
            exchange_type: Target exchange (None = Hyperliquid)

        Returns:
            Account state dict

        Raises:
            Exception: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(ACCOUNT_STATE_MAX_RETRIES):
            try:
                result = await self.get_account_state(exchange_type)
                if result is not None:
                    return result
                # get_account_state returned None - treat as failure
                last_error = Exception("Account state returned None")
            except Exception as e:
                last_error = e

            # Exponential backoff: 500ms, 1000ms, 2000ms
            if attempt < ACCOUNT_STATE_MAX_RETRIES - 1:
                delay_ms = ACCOUNT_STATE_BASE_DELAY_MS * (2 ** attempt)
                exchange_name = exchange_type.value if exchange_type else "hyperliquid"
                print(f"[executor] Account state fetch from {exchange_name} attempt {attempt + 1} failed, retrying in {delay_ms}ms...")
                await asyncio.sleep(delay_ms / 1000)

        # All retries exhausted
        raise Exception(f"Account state fetch failed after {ACCOUNT_STATE_MAX_RETRIES} attempts: {last_error}")

    async def get_account_value(
        self,
        exchange_type: Optional[ExchangeType] = None,
    ) -> float:
        """
        Get current account value.

        Args:
            exchange_type: Target exchange (None = Hyperliquid)

        Returns:
            Account value in USD, or 0 if unavailable
        """
        state = await self.get_account_state(exchange_type)
        if not state:
            return 0.0

        margin_summary = state.get("marginSummary", {})
        return float(margin_summary.get("accountValue", 0))

    async def get_current_exposure(
        self,
        exchange_type: Optional[ExchangeType] = None,
    ) -> float:
        """
        Get current total exposure as fraction of equity.

        Args:
            exchange_type: Target exchange (None = Hyperliquid)

        Returns:
            Exposure ratio (0-1+), or 0 if unavailable
        """
        state = await self.get_account_state(exchange_type)
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

    async def get_mid_price(
        self,
        symbol: str,
        exchange_type: Optional[ExchangeType] = None,
    ) -> Optional[float]:
        """
        Get current mid price for a symbol.

        Args:
            symbol: Asset symbol (BTC, ETH)
            exchange_type: Target exchange (None = Hyperliquid)

        Returns:
            Mid price or None if unavailable
        """
        # Try ExchangeManager first (multi-exchange path)
        if exchange_type is not None:
            exchange_manager = get_exchange_manager()
            exchange = exchange_manager.get_exchange(exchange_type)
            if exchange and exchange.is_connected:
                try:
                    # Format symbol for exchange
                    formatted_symbol = exchange.format_symbol(symbol)
                    price = await exchange.get_market_price(formatted_symbol)
                    if price:
                        return price
                except Exception as e:
                    print(f"[executor] Failed to fetch price from {exchange_type.value}: {e}")
                    # Fall through to legacy path

        # Legacy Hyperliquid API path
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
        2. Exchange enabled and configured
        3. Exposure limits
        4. Account has value
        5. Risk governor and circuit breaker

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

        # Determine target exchange (Phase 6: multi-exchange support)
        exchange_type_str = config.get("exchange", "hyperliquid").lower()
        try:
            exchange_type = ExchangeType(exchange_type_str)
        except ValueError:
            exchange_type = ExchangeType.HYPERLIQUID
        context["exchange"] = exchange_type.value

        # Check if using ExchangeManager (multi-exchange path)
        exchange_manager = get_exchange_manager()
        exchange = exchange_manager.get_exchange(exchange_type)
        using_exchange_manager = exchange is not None and exchange.is_connected

        # Fall back to legacy Hyperliquid config if ExchangeManager not available
        if not using_exchange_manager:
            # Check Hyperliquid enable (legacy path)
            hl_config = config.get("hyperliquid", {})
            if not hl_config.get("enabled"):
                return False, f"Exchange {exchange_type.value} not enabled", context

            # Check address configured (legacy path)
            address = hl_config.get("address")
            if not address:
                return False, f"No address configured for {exchange_type.value}", context

            self.address = address
            context["using_exchange_manager"] = False
        else:
            context["using_exchange_manager"] = True
            # For multi-exchange, get config from exchange-specific section
            exchange_config = config.get(exchange_type.value, config.get("hyperliquid", {}))
            # No address needed - ExchangeManager has credentials

        # Get account state (uses ExchangeManager if available)
        account_value = await self.get_account_value(exchange_type if using_exchange_manager else None)
        context["account_value"] = account_value

        if account_value <= 0:
            return False, f"No account value on {exchange_type.value}", context

        # Fetch account state with retry (kill switch check, sizing, circuit breaker)
        try:
            account_state = await self.get_account_state_with_retry(
                exchange_type if using_exchange_manager else None
            )
            context["account_state"] = account_state

            # Quick kill switch check (does NOT validate proposed size - that comes after sizing)
            from .risk_governor import get_risk_governor
            governor = get_risk_governor(db)
            if governor.is_kill_switch_active():
                increment_safety_block("kill_switch")
                return False, "Risk governor: Kill switch active", context

            # Update risk governor with exchange-specific state (Phase 6)
            governor.update_positions_from_account_state(account_state, exchange_type.value)
        except Exception as e:
            # Fail-closed: block execution when account state unavailable after retries
            print(f"[executor] Account state fetch from {exchange_type.value} failed after retries: {e}")
            increment_safety_block("account_state")
            return False, f"Account state unavailable: {e}", context

        # Check exposure limit
        current_exposure = await self.get_current_exposure(exchange_type if using_exchange_manager else None)
        context["exposure_before"] = current_exposure

        # Get exposure config from exchange-specific or hyperliquid section
        exchange_config = config.get(exchange_type.value, config.get("hyperliquid", {}))
        max_exposure = exchange_config.get("max_exposure_pct", 10) / 100  # Convert from % to decimal
        if current_exposure >= max_exposure:
            return False, f"Exposure {current_exposure:.1%} >= {max_exposure:.1%} limit on {exchange_type.value}", context

        # Get price (uses ExchangeManager if available)
        mid_price = await self.get_mid_price(symbol, exchange_type if using_exchange_manager else None)
        if not mid_price or mid_price <= 0:
            return False, f"Could not get price for {symbol} on {exchange_type.value}", context
        context["price"] = mid_price

        # Calculate position size - Kelly or fixed
        kelly_enabled = config.get("kelly_enabled", KELLY_ENABLED)
        kelly_result: Optional[KellyResult] = None

        if kelly_enabled and consensus_addresses:
            # Use Kelly criterion sizing based on consensus traders
            kelly_fraction = config.get("kelly_fraction", KELLY_FRACTION)

            # Adjust Kelly fraction for market regime
            try:
                from .regime import get_regime_detector, get_regime_adjusted_kelly, MarketRegime
                regime_detector = get_regime_detector(db)
                regime_analysis = await regime_detector.detect_regime(symbol)
                regime = regime_analysis.regime if regime_analysis else MarketRegime.UNKNOWN
                kelly_fraction = get_regime_adjusted_kelly(kelly_fraction, regime)
                context["regime"] = regime.value if regime else "unknown"
            except Exception as e:
                print(f"[executor] Failed to get regime for Kelly adjustment: {e}")
                context["regime"] = "unknown"

            # Get per-exchange fee config for Kelly sizing (Phase 6)
            fee_config = get_fee_config(exchange_type)
            # Round-trip cost as fraction (e.g., 10 bps = 0.001)
            round_trip_fee_pct = fee_config.round_trip_cost_bps() / 10000
            context["round_trip_fee_bps"] = fee_config.round_trip_cost_bps()

            kelly_result = await get_consensus_kelly_size(
                db,
                addresses=consensus_addresses,
                account_value=account_value,
                current_price=mid_price,
                stop_distance_pct=stop_distance_pct,
                fraction=kelly_fraction,
                round_trip_fee_pct=round_trip_fee_pct,
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

        # Re-calculate slippage with actual Kelly-sized position (Phase 6.1 Gap Fix)
        # Consensus detection uses reference $10k size; we now know actual size.
        try:
            from .consensus import get_slippage_estimate_bps_sync, calculate_ev, get_exchange_fees_bps
            from .consensus import get_funding_cost_bps_sync, CONSENSUS_EV_MIN_R
            from .consensus import DEFAULT_AVG_WIN_R, DEFAULT_AVG_LOSS_R, get_dynamic_hold_hours_sync

            asset = symbol.split("-")[0] if "-" in symbol else symbol.replace("USDC", "")
            actual_slippage_bps = get_slippage_estimate_bps_sync(
                asset=asset,
                exchange=exchange_type.value,
                order_size_usd=size_usd,  # Use Kelly-sized position
            )
            context["actual_slippage_bps"] = actual_slippage_bps

            # Get dynamic hold time from episode data (Phase 6.1)
            hold_hours = get_dynamic_hold_hours_sync(asset)
            context["hold_hours"] = hold_hours

            # Re-calculate EV with actual slippage to validate signal still passes
            fees_bps = get_exchange_fees_bps(exchange_type.value)
            funding_bps = get_funding_cost_bps_sync(
                asset=asset,
                exchange=exchange_type.value,
                hold_hours=hold_hours,
                side=direction,
            )

            # Calculate stop price for EV calculation
            stop_pct = stop_distance_pct or 0.02
            if direction == "long":
                stop_price = mid_price * (1 - stop_pct)
            else:
                stop_price = mid_price * (1 + stop_pct)

            # Estimate p_win from Kelly result if available
            p_win = 0.55  # Default conservative estimate
            if kelly_result and kelly_result.full_kelly > 0:
                # Reverse-engineer p_win from Kelly formula: f = p - (1-p)/R
                # With R = avg_win/avg_loss, approximate p_win
                p_win = min(0.70, 0.50 + kelly_result.full_kelly * 0.5)

            ev_result = calculate_ev(
                p_win=p_win,
                entry_px=mid_price,
                stop_px=stop_price,
                avg_win_r=DEFAULT_AVG_WIN_R,
                avg_loss_r=DEFAULT_AVG_LOSS_R,
                fees_bps=fees_bps,
                slip_bps=actual_slippage_bps,
                funding_bps=funding_bps,
            )
            context["actual_ev_net_r"] = ev_result["ev_net_r"]
            context["actual_ev_cost_r"] = ev_result["ev_cost_r"]

            # Reject if actual slippage pushes EV below minimum
            if ev_result["ev_net_r"] < CONSENSUS_EV_MIN_R:
                return False, f"EV {ev_result['ev_net_r']:.3f}R < minimum {CONSENSUS_EV_MIN_R:.3f}R after Kelly sizing (slippage={actual_slippage_bps:.1f}bps)", context

            print(f"[executor] Slippage recalc: reference→actual: ${size_usd:.0f} order = {actual_slippage_bps:.1f}bps, EV={ev_result['ev_net_r']:.3f}R")
        except Exception as e:
            # Non-fatal: continue with execution if slippage recalc fails
            print(f"[executor] Slippage recalculation failed (non-fatal): {e}")

        # Calculate new exposure
        new_exposure = current_exposure + (size_usd / account_value)
        context["exposure_after"] = new_exposure

        if new_exposure > max_exposure:
            return False, f"Trade would exceed exposure limit ({new_exposure:.1%} > {max_exposure:.1%})", context

        # Re-run risk governor with actual proposed size (reuse cached account_state)
        try:
            from .risk_governor import check_risk_before_trade
            account_state = context.get("account_state")
            if account_state:
                risk_result = await check_risk_before_trade(db, account_state, proposed_size_usd=size_usd)
                if not risk_result.allowed:
                    reason_str = risk_result.reason or "Risk governor blocked (size check)"
                    context["risk_governor_reason"] = reason_str
                    context["risk_governor_warnings"] = risk_result.warnings
                    increment_safety_block("risk_governor")
                    return False, f"Risk governor: {reason_str}", context
        except Exception as e:
            # Fail-closed: block execution when risk check fails
            print(f"[executor] Risk governor size check failed: {e}")
            increment_safety_block("risk_governor")
            return False, f"Risk governor check failed: {e}", context

        # Circuit breaker check (applies to both real and simulated execution)
        try:
            from .risk_governor import get_risk_governor
            governor = get_risk_governor(db)

            # Update position counts from account state using public method (reuse cached)
            # Note: Already updated earlier in validate_execution with exchange info
            exchange_name = context.get("exchange", "hyperliquid")

            # Pass per-symbol count for future multi-position support
            symbol_position_count = governor.get_symbol_position_count(symbol)
            cb_result = governor.run_circuit_breaker_checks(symbol, symbol_position_count)
            if not cb_result.allowed:
                context["circuit_breaker_reason"] = cb_result.reason
                increment_safety_block("circuit_breaker")
                return False, f"Circuit breaker: {cb_result.reason}", context
        except Exception as e:
            # Fail-closed: block execution when circuit breaker check fails
            increment_safety_block("circuit_breaker")
            print(f"[executor] Circuit breaker check failed: {e}")
            return False, f"Circuit breaker check failed: {e}", context

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

        By default, runs in dry-run mode (simulates execution).
        When REAL_EXECUTION_ENABLED=true, places real orders via hl_exchange.

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

        # Check if real execution is enabled
        from .hl_exchange import REAL_EXECUTION_ENABLED

        if REAL_EXECUTION_ENABLED:
            # REAL EXECUTION PATH
            # Note: Circuit breaker check already done in validate_execution()

            # Get exchange type from config (default: Hyperliquid for backward compatibility)
            exchange_type_str = config.get("exchange", "hyperliquid").lower()
            try:
                exchange_type = ExchangeType(exchange_type_str)
            except ValueError:
                exchange_type = ExchangeType.HYPERLIQUID

            # Use ExchangeManager for multi-exchange support
            exchange_manager = get_exchange_manager()
            exchange = exchange_manager.get_exchange(exchange_type)

            # Fall back to legacy hl_exchange if manager not initialized
            if exchange is None:
                from .hl_exchange import get_exchange as get_hl_exchange
                legacy_exchange = get_hl_exchange()
                if not legacy_exchange.can_execute:
                    exchange = None

            if exchange is not None and exchange.is_connected:
                try:
                    is_buy = (direction == "long")
                    size_coin = context.get("size_coin", 0)
                    stop_price = None
                    take_profit_price = None

                    # Calculate stop/take profit if stop_distance provided
                    mid_price = context.get("price", 0)
                    if mid_price and stop_distance_pct:
                        if direction == "long":
                            stop_price = mid_price * (1 - stop_distance_pct)
                            take_profit_price = mid_price * (1 + stop_distance_pct * 2)  # 2:1 RR
                        else:
                            stop_price = mid_price * (1 + stop_distance_pct)
                            take_profit_price = mid_price * (1 - stop_distance_pct * 2)

                    # Create order params for exchange adapter
                    order_params = OrderParams(
                        symbol=symbol,
                        side=OrderSide.BUY if is_buy else OrderSide.SELL,
                        size=size_coin,
                        stop_loss=stop_price,
                        take_profit=take_profit_price,
                    )

                    order_result = await exchange_manager.open_position(exchange_type, order_params)

                    if order_result.success:
                        result = ExecutionResult(
                            status="filled",
                            fill_price=order_result.fill_price,
                            fill_size=order_result.fill_size,
                            exposure_before=context.get("exposure_before"),
                            exposure_after=context.get("exposure_after"),
                            position_pct=context.get("position_pct"),
                            kelly_result=kelly_result,
                        )

                        # Register stop with StopManager (Phase 6: exchange-aware)
                        try:
                            from .stop_manager import get_stop_manager
                            stop_manager = get_stop_manager(db)
                            stop_pct = stop_distance_pct or 0.02
                            await stop_manager.register_stop(
                                decision_id=decision_id,
                                symbol=symbol,
                                direction=direction,
                                entry_price=order_result.fill_price,
                                entry_size=order_result.fill_size,
                                stop_distance_pct=stop_pct,
                                exchange=exchange_type.value,
                            )
                        except Exception as e:
                            print(f"[executor] Failed to register stop: {e}")

                        slippage = order_result.slippage_actual or 0
                        print(f"[executor] FILLED {direction} {symbol} on {exchange_type.value}: "
                              f"size={order_result.fill_size:.4f} @ ${order_result.fill_price:,.2f}, "
                              f"slippage={slippage:.3f}%")
                    else:
                        result = ExecutionResult(
                            status="failed",
                            error_message=order_result.error,
                            exposure_before=context.get("exposure_before"),
                            kelly_result=kelly_result,
                        )
                        print(f"[executor] FAILED {direction} {symbol} on {exchange_type.value}: {order_result.error}")

                    await self._log_execution(db, decision_id, symbol, direction, config, result, exchange_type.value)
                    return result

                except Exception as e:
                    result = ExecutionResult(
                        status="failed",
                        error_message=f"Execution error: {str(e)}",
                        exposure_before=context.get("exposure_before"),
                        kelly_result=kelly_result,
                    )
                    await self._log_execution(db, decision_id, symbol, direction, config, result)
                    return result
            else:
                # No exchange available - try legacy path
                from .hl_exchange import get_exchange as get_hl_exchange, execute_market_order
                legacy_exchange = get_hl_exchange()
                if legacy_exchange.can_execute:
                    try:
                        is_buy = (direction == "long")
                        size_coin = context.get("size_coin", 0)

                        order_result = await execute_market_order(
                            asset=symbol,
                            is_buy=is_buy,
                            size=size_coin,
                        )

                        if order_result.success:
                            result = ExecutionResult(
                                status="filled",
                                fill_price=order_result.fill_price,
                                fill_size=order_result.fill_size,
                                exposure_before=context.get("exposure_before"),
                                exposure_after=context.get("exposure_after"),
                                position_pct=context.get("position_pct"),
                                kelly_result=kelly_result,
                            )

                            # Register stop with StopManager (legacy path uses hyperliquid)
                            try:
                                from .stop_manager import get_stop_manager
                                stop_manager = get_stop_manager(db)
                                stop_pct = stop_distance_pct or 0.02
                                await stop_manager.register_stop(
                                    decision_id=decision_id,
                                    symbol=symbol,
                                    direction=direction,
                                    entry_price=order_result.fill_price,
                                    entry_size=order_result.fill_size,
                                    stop_distance_pct=stop_pct,
                                    exchange="hyperliquid",  # Legacy path
                                )
                            except Exception as e:
                                print(f"[executor] Failed to register stop: {e}")

                            print(f"[executor] FILLED {direction} {symbol}: "
                                  f"size={order_result.fill_size:.4f} @ ${order_result.fill_price:,.2f}, "
                                  f"slippage={order_result.slippage_actual:.3f}%")
                        else:
                            result = ExecutionResult(
                                status="failed",
                                error_message=order_result.error,
                                exposure_before=context.get("exposure_before"),
                                kelly_result=kelly_result,
                            )
                            print(f"[executor] FAILED {direction} {symbol}: {order_result.error}")

                        await self._log_execution(db, decision_id, symbol, direction, config, result)
                        return result

                    except Exception as e:
                        result = ExecutionResult(
                            status="failed",
                            error_message=f"Execution error: {str(e)}",
                            exposure_before=context.get("exposure_before"),
                            kelly_result=kelly_result,
                        )
                        await self._log_execution(db, decision_id, symbol, direction, config, result)
                        return result

        # DRY RUN PATH (default)
        result = ExecutionResult(
            status="simulated",
            fill_price=context.get("price"),
            fill_size=context.get("size_coin"),
            exposure_before=context.get("exposure_before"),
            exposure_after=context.get("exposure_after"),
            position_pct=context.get("position_pct"),
            error_message="Dry run - real execution disabled",
            kelly_result=kelly_result,
        )

        await self._log_execution(db, decision_id, symbol, direction, config, result)

        sizing_method = context.get("sizing_method", "fixed")
        regime = context.get("regime", "unknown")
        print(f"[executor] Simulated {direction} {symbol}: "
              f"size={context.get('size_coin'):.4f}, "
              f"price=${context.get('price'):,.2f}, "
              f"exposure={context.get('exposure_before'):.1%} -> {context.get('exposure_after'):.1%}, "
              f"sizing={sizing_method}, regime={regime}")

        return result

    async def _log_execution(
        self,
        db: asyncpg.Pool,
        decision_id: str,
        symbol: str,
        direction: str,
        config: dict[str, Any],
        result: ExecutionResult,
        exchange: str = "hyperliquid",
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
                    exchange,  # Now uses the provided exchange parameter
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
