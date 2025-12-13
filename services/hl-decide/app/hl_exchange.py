"""
Hyperliquid Exchange API Integration

Phase 4.2: Real Trade Execution

Handles order placement, cancellation, and fill confirmation on Hyperliquid.
Uses the official hyperliquid-python-sdk for EIP-712 signing.

Security:
- Private key loaded from environment variable
- Never logged or stored in database
- Subaccount support for trade isolation

IMPORTANT: This module is disabled by default. Enable only when ready for
real trading with proper risk controls in place.

@module hl_exchange
"""

import os
from dataclasses import dataclass
from typing import Optional

import httpx

# Conditional import for SDK - allows testing without full SDK
try:
    from eth_account import Account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants as hl_constants
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    Exchange = None
    Info = None
    Account = None
    hl_constants = None


# Exchange API endpoint (for direct API calls)
HL_EXCHANGE_API = os.getenv("HL_EXCHANGE_API", "https://api.hyperliquid.xyz/exchange")
HL_INFO_API = os.getenv("HL_INFO_API", "https://api.hyperliquid.xyz/info")

# Safety: Real execution must be explicitly enabled
REAL_EXECUTION_ENABLED = os.getenv("REAL_EXECUTION_ENABLED", "false").lower() == "true"

# Default slippage for market orders (0.5%)
DEFAULT_SLIPPAGE_PCT = float(os.getenv("HL_SLIPPAGE_PCT", "0.50"))

# Use testnet by default for safety
USE_TESTNET = os.getenv("HL_USE_TESTNET", "true").lower() == "true"


@dataclass
class OrderParams:
    """Parameters for placing an order."""

    asset: str  # Asset symbol (BTC, ETH)
    is_buy: bool  # True for long entry/short exit
    size: float  # Size in coin units
    price: Optional[float] = None  # None for market order
    reduce_only: bool = False  # True for closing positions
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT  # Slippage tolerance for market orders


@dataclass
class OrderResult:
    """Result of an order placement."""

    success: bool
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    slippage_actual: Optional[float] = None
    error: Optional[str] = None
    raw_response: Optional[dict] = None


class HyperliquidExchange:
    """
    Exchange API wrapper for Hyperliquid order placement.

    Uses the official hyperliquid-python-sdk for proper EIP-712 signing.
    This handles all the complexities of signing L1 actions.

    Security Notes:
    - Private key is loaded from environment variable, never stored
    - All order placement is gated by REAL_EXECUTION_ENABLED flag
    - Uses testnet by default (HL_USE_TESTNET=true)

    Usage:
        exchange = HyperliquidExchange()
        result = await exchange.place_market_order(
            OrderParams(asset="BTC", is_buy=True, size=0.01)
        )
    """

    def __init__(
        self,
        private_key_env: str = "HL_PRIVATE_KEY",
        account_address_env: str = "HL_ACCOUNT_ADDRESS",
        use_testnet: bool = USE_TESTNET,
    ):
        """
        Initialize exchange connection.

        Args:
            private_key_env: Name of environment variable containing private key
            account_address_env: Name of env var containing account address (for API keys)
            use_testnet: Whether to use testnet (safer for development)

        Note: If using an API key (not main wallet), HL_ACCOUNT_ADDRESS must be
        the main wallet's public address, not the API key's address.
        """
        self._private_key_env = private_key_env
        self._account_address_env = account_address_env
        self._private_key = os.getenv(private_key_env)
        self._account_address = os.getenv(account_address_env)
        self._use_testnet = use_testnet
        self._http_client: Optional[httpx.AsyncClient] = None

        # SDK components (initialized lazily)
        self._exchange: Optional[Exchange] = None
        self._info: Optional[Info] = None
        self._wallet = None

    @property
    def is_configured(self) -> bool:
        """Check if exchange is properly configured for real trading."""
        return bool(self._private_key) and SDK_AVAILABLE

    @property
    def can_execute(self) -> bool:
        """Check if real execution is enabled and configured."""
        return REAL_EXECUTION_ENABLED and self.is_configured

    def _init_sdk(self) -> bool:
        """Initialize the Hyperliquid SDK components."""
        if not SDK_AVAILABLE:
            print("[hl_exchange] SDK not available - install hyperliquid-python-sdk")
            return False

        if not self._private_key:
            print(f"[hl_exchange] Private key not found in {self._private_key_env}")
            return False

        try:
            # Create wallet from private key
            self._wallet = Account.from_key(self._private_key)

            # Select API URL based on network
            if self._use_testnet:
                base_url = hl_constants.TESTNET_API_URL
                print("[hl_exchange] Using TESTNET - orders will not use real funds")
            else:
                base_url = hl_constants.MAINNET_API_URL
                print("[hl_exchange] WARNING: Using MAINNET - orders will use real funds")

            # Initialize Info API (read-only)
            self._info = Info(base_url, skip_ws=True)

            # Initialize Exchange API (for orders)
            # If using API key, pass account_address as the main wallet's address
            account_addr = self._account_address or self._wallet.address
            self._exchange = Exchange(
                self._wallet,
                base_url=base_url,
                account_address=account_addr,
            )

            print(f"[hl_exchange] SDK initialized for {account_addr[:10]}...")
            return True

        except Exception as e:
            print(f"[hl_exchange] Failed to initialize SDK: {e}")
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for direct API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_mid_price(self, asset: str) -> Optional[float]:
        """
        Get current mid price for an asset.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            Mid price or None if unavailable
        """
        try:
            client = await self._get_client()
            payload = {"type": "allMids"}
            resp = await client.post(HL_INFO_API, json=payload)
            if resp.status_code == 200:
                mids = resp.json()
                return float(mids.get(asset, 0))
            return None
        except Exception as e:
            print(f"[hl_exchange] Failed to fetch mid price: {e}")
            return None

    async def place_market_order(self, params: OrderParams) -> OrderResult:
        """
        Place a market order on Hyperliquid.

        Uses the SDK's market_open method which handles:
        - EIP-712 signing
        - Slippage-adjusted limit pricing
        - Immediate-or-Cancel (IoC) execution

        Args:
            params: Order parameters

        Returns:
            OrderResult with fill details or error
        """
        # Safety check
        if not self.can_execute:
            return OrderResult(
                success=False,
                error="Real execution disabled. Set REAL_EXECUTION_ENABLED=true and configure HL_PRIVATE_KEY",
            )

        # Initialize SDK if needed
        if self._exchange is None:
            if not self._init_sdk():
                return OrderResult(
                    success=False,
                    error="Failed to initialize Hyperliquid SDK",
                )

        # Get current price for slippage tracking
        mid_price = await self.get_mid_price(params.asset)
        if not mid_price:
            return OrderResult(
                success=False,
                error=f"Could not get price for {params.asset}",
            )

        try:
            # Convert slippage from percentage to decimal
            slippage = params.slippage_pct / 100

            # Place market order using SDK
            # SDK handles: signing, slippage pricing, IoC order type
            result = self._exchange.market_open(
                coin=params.asset,
                is_buy=params.is_buy,
                sz=params.size,
                px=None,  # Market order
                slippage=slippage,
            )

            print(f"[hl_exchange] Order result: {result}")

            # Parse response
            if result.get("status") == "ok":
                response_data = result.get("response", {}).get("data", {})
                statuses = response_data.get("statuses", [])

                if statuses:
                    status = statuses[0]

                    # Check for fill
                    if "filled" in status:
                        filled = status["filled"]
                        fill_price = float(filled.get("avgPx", 0))
                        fill_size = float(filled.get("totalSz", 0))
                        slippage_actual = abs(fill_price - mid_price) / mid_price * 100

                        return OrderResult(
                            success=True,
                            order_id=str(filled.get("oid", "")),
                            fill_price=fill_price,
                            fill_size=fill_size,
                            slippage_actual=slippage_actual,
                            raw_response=result,
                        )

                    # Check for resting order (partial fill)
                    elif "resting" in status:
                        resting = status["resting"]
                        return OrderResult(
                            success=True,
                            order_id=str(resting.get("oid", "")),
                            raw_response=result,
                        )

                    # Check for error
                    elif "error" in status:
                        return OrderResult(
                            success=False,
                            error=status["error"],
                            raw_response=result,
                        )

            # Handle error response
            return OrderResult(
                success=False,
                error=f"Unexpected response: {result}",
                raw_response=result,
            )

        except Exception as e:
            return OrderResult(
                success=False,
                error=f"Order placement failed: {str(e)}",
            )

    async def close_position(self, asset: str) -> OrderResult:
        """
        Close entire position for an asset.

        Uses SDK's market_close which handles reduce_only automatically.

        Args:
            asset: Asset symbol to close

        Returns:
            OrderResult with fill details or error
        """
        # Safety check
        if not self.can_execute:
            return OrderResult(
                success=False,
                error="Real execution disabled",
            )

        # Initialize SDK if needed
        if self._exchange is None:
            if not self._init_sdk():
                return OrderResult(
                    success=False,
                    error="Failed to initialize Hyperliquid SDK",
                )

        try:
            # Get current price for tracking
            mid_price = await self.get_mid_price(asset)

            # SDK's market_close handles:
            # - Fetching current position
            # - Determining close direction
            # - Setting reduce_only=True
            result = self._exchange.market_close(
                coin=asset,
                slippage=DEFAULT_SLIPPAGE_PCT / 100,
            )

            print(f"[hl_exchange] Close result: {result}")

            # Parse response (same format as market_open)
            if result.get("status") == "ok":
                response_data = result.get("response", {}).get("data", {})
                statuses = response_data.get("statuses", [])

                if statuses:
                    status = statuses[0]
                    if "filled" in status:
                        filled = status["filled"]
                        fill_price = float(filled.get("avgPx", 0))
                        fill_size = float(filled.get("totalSz", 0))
                        slippage_actual = (
                            abs(fill_price - mid_price) / mid_price * 100
                            if mid_price
                            else None
                        )

                        return OrderResult(
                            success=True,
                            order_id=str(filled.get("oid", "")),
                            fill_price=fill_price,
                            fill_size=fill_size,
                            slippage_actual=slippage_actual,
                            raw_response=result,
                        )

            return OrderResult(
                success=False,
                error=f"Close failed: {result}",
                raw_response=result,
            )

        except Exception as e:
            # Handle "no position" gracefully
            if "position" in str(e).lower():
                return OrderResult(
                    success=True,
                    fill_size=0,
                    error="No position to close",
                )
            return OrderResult(
                success=False,
                error=f"Close position failed: {str(e)}",
            )

    async def get_position(self, asset: str) -> Optional[dict]:
        """
        Get current position for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Position dict with size, entryPx, etc. or None
        """
        if not self._exchange and not self._init_sdk():
            return None

        try:
            # Get user state using SDK
            account = self._account_address or (self._wallet.address if self._wallet else None)
            if not account:
                return None

            user_state = self._info.user_state(account)
            positions = user_state.get("assetPositions", [])

            for pos in positions:
                position_data = pos.get("position", {})
                if position_data.get("coin") == asset:
                    return {
                        "size": float(position_data.get("szi", 0)),
                        "entry_price": float(position_data.get("entryPx", 0)),
                        "unrealized_pnl": float(position_data.get("unrealizedPnl", 0)),
                        "liquidation_price": position_data.get("liquidationPx"),
                        "leverage": position_data.get("leverage", {}),
                    }

            return {"size": 0}  # No position

        except Exception as e:
            print(f"[hl_exchange] Failed to get position: {e}")
            return None

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get current status of an order.

        Args:
            order_id: Order ID from placement

        Returns:
            Order status dict or None
        """
        if not order_id:
            return None

        if not self._info and not self._init_sdk():
            return None

        try:
            account = self._account_address or (self._wallet.address if self._wallet else None)
            if not account:
                return None

            # Get open orders
            open_orders = self._info.open_orders(account)
            for order in open_orders:
                if str(order.get("oid")) == str(order_id):
                    return {
                        "status": "open",
                        "order": order,
                    }

            # Check order history for fills
            # Note: SDK doesn't have direct fill query, would need additional API call
            return {"status": "filled_or_cancelled"}

        except Exception as e:
            print(f"[hl_exchange] Failed to get order status: {e}")
            return None

    async def cancel_order(self, asset: str, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            asset: Asset symbol
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.can_execute:
            return False

        if self._exchange is None:
            if not self._init_sdk():
                return False

        try:
            result = self._exchange.cancel(asset, int(order_id))
            return result.get("status") == "ok"
        except Exception as e:
            print(f"[hl_exchange] Failed to cancel order: {e}")
            return False

    async def get_account_value(self) -> Optional[float]:
        """
        Get total account value in USD.

        Returns:
            Account value or None
        """
        if not self._info and not self._init_sdk():
            return None

        try:
            account = self._account_address or (self._wallet.address if self._wallet else None)
            if not account:
                return None

            user_state = self._info.user_state(account)
            margin_summary = user_state.get("marginSummary", {})
            return float(margin_summary.get("accountValue", 0))

        except Exception as e:
            print(f"[hl_exchange] Failed to get account value: {e}")
            return None


# Global exchange instance
_exchange: Optional[HyperliquidExchange] = None


def get_exchange(
    private_key_env: str = "HL_PRIVATE_KEY",
    account_address_env: str = "HL_ACCOUNT_ADDRESS",
    use_testnet: bool = USE_TESTNET,
) -> HyperliquidExchange:
    """
    Get or create global exchange instance.

    Args:
        private_key_env: Environment variable name for private key
        account_address_env: Environment variable name for account address
        use_testnet: Whether to use testnet

    Returns:
        HyperliquidExchange instance
    """
    global _exchange
    if _exchange is None:
        _exchange = HyperliquidExchange(private_key_env, account_address_env, use_testnet)
    return _exchange


async def execute_market_order(
    asset: str,
    is_buy: bool,
    size: float,
    reduce_only: bool = False,
) -> OrderResult:
    """
    Convenience function to execute a market order.

    Args:
        asset: Asset symbol (BTC, ETH)
        is_buy: True for buy, False for sell
        size: Order size in coins
        reduce_only: Whether this is a position close

    Returns:
        OrderResult with fill details
    """
    exchange = get_exchange()

    if reduce_only:
        return await exchange.close_position(asset)

    params = OrderParams(
        asset=asset,
        is_buy=is_buy,
        size=size,
        reduce_only=reduce_only,
    )
    return await exchange.place_market_order(params)
