"""
Hyperliquid Exchange Adapter

Implements ExchangeInterface for Hyperliquid perp DEX.

Features:
- EIP-712 signed orders via hyperliquid-python-sdk
- Agent wallet support (security best practice)
- Testnet/mainnet switching
- Automatic precision handling

@module exchanges.hyperliquid_adapter
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .interface import (
    ExchangeInterface,
    ExchangeConfig,
    ExchangeType,
    OrderParams,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    Balance,
    MarketData,
    MarginMode,
)

# Conditional SDK import
try:
    from eth_account import Account
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants as hl_constants
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    Exchange = None  # type: ignore
    Info = None  # type: ignore
    Account = None  # type: ignore
    hl_constants = None  # type: ignore


# Default API endpoints
HL_MAINNET_API = "https://api.hyperliquid.xyz"
HL_TESTNET_API = "https://api.hyperliquid-testnet.xyz"


class HyperliquidAdapter(ExchangeInterface):
    """
    Hyperliquid exchange adapter.

    Uses the official hyperliquid-python-sdk for order signing and placement.
    Supports both main wallet and agent wallet (API key) configurations.

    Configuration:
        config = ExchangeConfig(
            exchange_type=ExchangeType.HYPERLIQUID,
            testnet=True,
            private_key_env="HL_PRIVATE_KEY",
            account_address_env="HL_ACCOUNT_ADDRESS",  # Main wallet if using agent
        )
        adapter = HyperliquidAdapter(config)
    """

    def __init__(self, config: ExchangeConfig):
        """Initialize Hyperliquid adapter."""
        super().__init__(config)

        self._http_client: Optional[httpx.AsyncClient] = None
        self._exchange: Optional[Exchange] = None
        self._info: Optional[Info] = None
        self._wallet = None

        # Symbol metadata cache
        self._meta_cache: dict[str, dict] = {}

    @property
    def is_configured(self) -> bool:
        """Check if exchange is properly configured."""
        return SDK_AVAILABLE and bool(self.config.get_private_key())

    @property
    def _base_url(self) -> str:
        """Get API base URL based on testnet setting."""
        if self.config.testnet:
            return HL_TESTNET_API if not hl_constants else hl_constants.TESTNET_API_URL
        return HL_MAINNET_API if not hl_constants else hl_constants.MAINNET_API_URL

    @property
    def _info_url(self) -> str:
        """Get info API URL."""
        return f"{self._base_url}/info"

    async def connect(self) -> bool:
        """Connect to Hyperliquid API."""
        if not SDK_AVAILABLE:
            print("[hyperliquid] SDK not available - install hyperliquid-python-sdk")
            return False

        private_key = self.config.get_private_key()
        if not private_key:
            print(f"[hyperliquid] Private key not found in {self.config.private_key_env}")
            return False

        try:
            # Create wallet from private key
            self._wallet = Account.from_key(private_key)

            # Log network
            if self.config.testnet:
                print("[hyperliquid] Using TESTNET - orders will not use real funds")
            else:
                print("[hyperliquid] WARNING: Using MAINNET - orders will use real funds")

            # Initialize Info API (read-only)
            self._info = Info(self._base_url, skip_ws=True)

            # Initialize Exchange API (for orders)
            # If using agent wallet, account_address is the main wallet
            account_addr = self.config.get_account_address() or self._wallet.address
            self._exchange = Exchange(
                self._wallet,
                base_url=self._base_url,
                account_address=account_addr,
            )

            # Create HTTP client for direct API calls
            self._http_client = httpx.AsyncClient(timeout=30)

            # Cache symbol metadata
            await self._load_metadata()

            self._connected = True
            print(f"[hyperliquid] Connected as {account_addr[:10]}...")
            return True

        except Exception as e:
            print(f"[hyperliquid] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Hyperliquid API."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._exchange = None
        self._info = None
        self._wallet = None
        self._connected = False

    async def _load_metadata(self) -> None:
        """Load and cache symbol metadata."""
        if not self._info:
            return

        try:
            meta = self._info.meta()
            universe = meta.get("universe", [])
            for asset in universe:
                name = asset.get("name", "")
                self._meta_cache[name] = {
                    "sz_decimals": asset.get("szDecimals", 3),
                    "max_leverage": asset.get("maxLeverage", 50),
                }
        except Exception as e:
            print(f"[hyperliquid] Failed to load metadata: {e}")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for direct API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30)
        return self._http_client

    def _get_account_address(self) -> Optional[str]:
        """Get account address for queries."""
        return self.config.get_account_address() or (
            self._wallet.address if self._wallet else None
        )

    # Account operations

    async def get_balance(self) -> Optional[Balance]:
        """Get account balance."""
        if not self._info:
            return None

        account = self._get_account_address()
        if not account:
            return None

        try:
            user_state = self._info.user_state(account)
            margin_summary = user_state.get("marginSummary", {})

            total_equity = float(margin_summary.get("accountValue", 0))
            margin_used = float(margin_summary.get("totalMarginUsed", 0))
            available = float(margin_summary.get("totalRawUsd", 0))

            # Calculate unrealized PnL from positions
            unrealized_pnl = 0.0
            for ap in user_state.get("assetPositions", []):
                pos = ap.get("position", {})
                unrealized_pnl += float(pos.get("unrealizedPnl", 0))

            return Balance(
                total_equity=total_equity,
                available_balance=available,
                margin_used=margin_used,
                unrealized_pnl=unrealized_pnl,
                currency="USD",
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            print(f"[hyperliquid] Failed to get balance: {e}")
            return None

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        if not self._info:
            return []

        account = self._get_account_address()
        if not account:
            return []

        try:
            user_state = self._info.user_state(account)
            positions = []

            for ap in user_state.get("assetPositions", []):
                pos = ap.get("position", {})
                size = float(pos.get("szi", 0))

                if size == 0:
                    continue

                entry_price = float(pos.get("entryPx", 0))
                position_value = float(pos.get("positionValue", 0))
                mark_price = position_value / abs(size) if size != 0 else entry_price

                positions.append(Position(
                    symbol=pos.get("coin", ""),
                    side=PositionSide.LONG if size > 0 else PositionSide.SHORT,
                    size=size,
                    entry_price=entry_price,
                    mark_price=mark_price,
                    liquidation_price=float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None,
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    leverage=int(pos.get("leverage", {}).get("value", 1)),
                    margin_mode=MarginMode.CROSS if pos.get("leverage", {}).get("type") == "cross" else MarginMode.ISOLATED,
                    margin_used=float(pos.get("marginUsed", 0)),
                    timestamp=datetime.now(timezone.utc),
                ))

            return positions

        except Exception as e:
            print(f"[hyperliquid] Failed to get positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    # Trading operations

    async def open_position(self, params: OrderParams) -> OrderResult:
        """Open a new position."""
        if not self._exchange:
            return OrderResult(success=False, error="Exchange not connected")

        # Format symbol for exchange
        symbol = self.format_symbol(params.symbol)

        # Get current price for slippage tracking
        mid_price = await self.get_market_price(symbol)
        if not mid_price:
            return OrderResult(success=False, error=f"Could not get price for {symbol}")

        try:
            # Set leverage if specified
            if params.leverage:
                await self.set_leverage(symbol, params.leverage)

            # Convert slippage to decimal
            slippage = params.slippage_pct / 100

            # Place market order using SDK
            is_buy = params.side == OrderSide.BUY
            result = self._exchange.market_open(
                coin=symbol,
                is_buy=is_buy,
                sz=params.size,
                px=None,  # Market order
                slippage=slippage,
            )

            return self._parse_order_result(result, mid_price)

        except Exception as e:
            return OrderResult(success=False, error=f"Order failed: {str(e)}")

    async def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
    ) -> OrderResult:
        """Close position partially or fully."""
        if not self._exchange:
            return OrderResult(success=False, error="Exchange not connected")

        # Format symbol for exchange
        formatted_symbol = self.format_symbol(symbol)

        try:
            mid_price = await self.get_market_price(formatted_symbol)

            if size is None:
                # Full close using SDK
                result = self._exchange.market_close(
                    coin=formatted_symbol,
                    slippage=self.config.default_slippage_pct / 100,
                )
            else:
                # Partial close - need to determine direction
                position = await self.get_position(formatted_symbol)
                if not position:
                    return OrderResult(success=True, fill_size=0, error="No position to close")

                is_buy = position.size < 0  # Buy to close short, sell to close long
                result = self._exchange.market_open(
                    coin=formatted_symbol,
                    is_buy=is_buy,
                    sz=abs(size),
                    px=None,
                    slippage=self.config.default_slippage_pct / 100,
                    reduce_only=True,
                )

            return self._parse_order_result(result, mid_price)

        except Exception as e:
            # Handle "no position" gracefully
            if "position" in str(e).lower():
                return OrderResult(success=True, fill_size=0, error="No position to close")
            return OrderResult(success=False, error=f"Close failed: {str(e)}")

    async def place_order(self, params: OrderParams) -> OrderResult:
        """Place a generic order."""
        if params.order_type == OrderType.MARKET:
            if params.reduce_only:
                return await self.close_position(params.symbol, params.size)
            return await self.open_position(params)

        # Limit order
        if not self._exchange:
            return OrderResult(success=False, error="Exchange not connected")

        if params.price is None:
            return OrderResult(success=False, error="Price required for limit order")

        try:
            is_buy = params.side == OrderSide.BUY
            result = self._exchange.order(
                coin=params.symbol,
                is_buy=is_buy,
                sz=params.size,
                limit_px=params.price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=params.reduce_only,
            )
            return self._parse_order_result(result, params.price)

        except Exception as e:
            return OrderResult(success=False, error=f"Order failed: {str(e)}")

    def _parse_order_result(
        self,
        result: dict,
        reference_price: Optional[float] = None,
    ) -> OrderResult:
        """Parse SDK response into OrderResult."""
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
                        abs(fill_price - reference_price) / reference_price * 100
                        if reference_price and reference_price > 0
                        else None
                    )

                    return OrderResult(
                        success=True,
                        order_id=str(filled.get("oid", "")),
                        fill_price=fill_price,
                        fill_size=fill_size,
                        filled_pct=100.0,
                        status="filled",
                        slippage_actual=slippage_actual,
                        raw_response=result,
                    )

                elif "resting" in status:
                    resting = status["resting"]
                    return OrderResult(
                        success=True,
                        order_id=str(resting.get("oid", "")),
                        status="pending",
                        raw_response=result,
                    )

                elif "error" in status:
                    return OrderResult(
                        success=False,
                        error=status["error"],
                        status="rejected",
                        raw_response=result,
                    )

        return OrderResult(
            success=False,
            error=f"Unexpected response: {result}",
            raw_response=result,
        )

    # Order management

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get status of an order."""
        if not self._info:
            return None

        account = self._get_account_address()
        if not account:
            return None

        try:
            open_orders = self._info.open_orders(account)
            for order in open_orders:
                if str(order.get("oid")) == str(order_id):
                    return {"status": "open", "order": order}
            return {"status": "filled_or_cancelled"}

        except Exception as e:
            print(f"[hyperliquid] Failed to get order status: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order."""
        if not self._exchange:
            return False

        try:
            result = self._exchange.cancel(symbol, int(order_id))
            return result.get("status") == "ok"
        except Exception as e:
            print(f"[hyperliquid] Failed to cancel order: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        if not self._exchange or not self._info:
            return 0

        account = self._get_account_address()
        if not account:
            return 0

        try:
            open_orders = self._info.open_orders(account)
            cancelled = 0

            for order in open_orders:
                order_symbol = order.get("coin", "")
                if symbol is None or order_symbol == symbol:
                    if await self.cancel_order(order_symbol, str(order.get("oid"))):
                        cancelled += 1

            return cancelled

        except Exception as e:
            print(f"[hyperliquid] Failed to cancel orders: {e}")
            return 0

    # Risk management

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self._exchange:
            return False

        try:
            result = self._exchange.update_leverage(
                leverage=leverage,
                coin=symbol,
                is_cross=self.config.default_margin_mode == MarginMode.CROSS,
            )
            return result.get("status") == "ok"
        except Exception as e:
            print(f"[hyperliquid] Failed to set leverage: {e}")
            return False

    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        size: Optional[float] = None,
    ) -> OrderResult:
        """Set stop-loss order for position."""
        if not self._exchange:
            return OrderResult(success=False, error="Exchange not connected")

        position = await self.get_position(symbol)
        if not position:
            return OrderResult(success=False, error="No position for stop-loss")

        try:
            # Determine direction based on position
            is_buy = position.size < 0  # Buy to close short
            order_size = abs(size if size else position.size)

            result = self._exchange.order(
                coin=symbol,
                is_buy=is_buy,
                sz=order_size,
                limit_px=stop_price,
                order_type={
                    "trigger": {
                        "triggerPx": stop_price,
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                reduce_only=True,
            )
            return self._parse_order_result(result, stop_price)

        except Exception as e:
            return OrderResult(success=False, error=f"Stop-loss failed: {str(e)}")

    async def set_take_profit(
        self,
        symbol: str,
        take_profit_price: float,
        size: Optional[float] = None,
    ) -> OrderResult:
        """Set take-profit order for position."""
        if not self._exchange:
            return OrderResult(success=False, error="Exchange not connected")

        position = await self.get_position(symbol)
        if not position:
            return OrderResult(success=False, error="No position for take-profit")

        try:
            is_buy = position.size < 0
            order_size = abs(size if size else position.size)

            result = self._exchange.order(
                coin=symbol,
                is_buy=is_buy,
                sz=order_size,
                limit_px=take_profit_price,
                order_type={
                    "trigger": {
                        "triggerPx": take_profit_price,
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                reduce_only=True,
            )
            return self._parse_order_result(result, take_profit_price)

        except Exception as e:
            return OrderResult(success=False, error=f"Take-profit failed: {str(e)}")

    # Market data

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market mid price."""
        try:
            client = await self._get_http_client()
            payload = {"type": "allMids"}
            resp = await client.post(self._info_url, json=payload)
            if resp.status_code == 200:
                mids = resp.json()
                return float(mids.get(symbol, 0)) or None
            return None
        except Exception as e:
            print(f"[hyperliquid] Failed to get price: {e}")
            return None

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get full market data."""
        try:
            client = await self._get_http_client()

            # Get mids
            mid_price = await self.get_market_price(symbol)
            if not mid_price:
                return None

            # Get L2 orderbook for bid/ask
            payload = {"type": "l2Book", "coin": symbol}
            resp = await client.post(self._info_url, json=payload)

            if resp.status_code == 200:
                book = resp.json()
                levels = book.get("levels", [[], []])
                bids = levels[0] if len(levels) > 0 else []
                asks = levels[1] if len(levels) > 1 else []

                best_bid = float(bids[0]["px"]) if bids else mid_price
                best_ask = float(asks[0]["px"]) if asks else mid_price

                return MarketData(
                    symbol=symbol,
                    bid=best_bid,
                    ask=best_ask,
                    last=mid_price,
                    mark_price=mid_price,
                    timestamp=datetime.now(timezone.utc),
                )

            return None

        except Exception as e:
            print(f"[hyperliquid] Failed to get market data: {e}")
            return None

    # Utility methods

    def format_symbol(self, symbol: str) -> str:
        """Format symbol for Hyperliquid (no change needed)."""
        return symbol.upper()

    def format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity to exchange precision."""
        meta = self._meta_cache.get(symbol, {})
        decimals = meta.get("sz_decimals", 3)
        factor = 10 ** decimals
        return int(quantity * factor) / factor

    def format_price(self, symbol: str, price: float) -> float:
        """Format price to 5 significant figures (Hyperliquid requirement)."""
        if price <= 0:
            return 0.0

        # Round to 5 significant figures
        from math import floor, log10
        sig_figs = 5
        magnitude = floor(log10(abs(price)))
        factor = 10 ** (sig_figs - magnitude - 1)
        return round(price * factor) / factor
