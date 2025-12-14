"""
Slippage Estimation Provider

Estimates expected slippage based on order size and orderbook depth.

Slippage is critical for large orders that may move the market. This provider
fetches orderbook data when available and falls back to static estimates.

@module slippage_provider
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

import httpx


# Configuration
SLIPPAGE_CACHE_TTL_SECONDS = int(os.getenv("SLIPPAGE_CACHE_TTL_SECONDS", "60"))  # 1 minute
SLIPPAGE_API_TIMEOUT_SECONDS = int(os.getenv("SLIPPAGE_API_TIMEOUT_SECONDS", "5"))

# API URLs
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com")
BYBIT_TESTNET_API_URL = os.getenv("BYBIT_TESTNET_API_URL", "https://api-testnet.bybit.com")
HYPERLIQUID_API_URL = os.getenv("HYPERLIQUID_API_URL", "https://api.hyperliquid.xyz")

# Default slippage estimates (bps) when orderbook unavailable
# These are conservative estimates based on typical market conditions
DEFAULT_SLIPPAGE_BPS = {
    "hyperliquid": {
        "BTC": {"small": 1.0, "medium": 2.0, "large": 5.0},  # <$10k, $10-50k, >$50k
        "ETH": {"small": 1.5, "medium": 3.0, "large": 7.0},
    },
    "aster": {
        "BTC": {"small": 1.0, "medium": 2.0, "large": 5.0},
        "ETH": {"small": 1.5, "medium": 3.0, "large": 7.0},
    },
    "bybit": {
        "BTC": {"small": 0.5, "medium": 1.5, "large": 3.0},  # CEX typically tighter
        "ETH": {"small": 1.0, "medium": 2.0, "large": 5.0},
    },
}

# Order size thresholds (USD)
SIZE_THRESHOLD_SMALL = float(os.getenv("SIZE_THRESHOLD_SMALL", "10000"))
SIZE_THRESHOLD_LARGE = float(os.getenv("SIZE_THRESHOLD_LARGE", "50000"))

# Slippage warning threshold (bps)
SLIPPAGE_WARNING_THRESHOLD_BPS = float(os.getenv("SLIPPAGE_WARNING_THRESHOLD_BPS", "10.0"))


@dataclass
class OrderbookLevel:
    """Single orderbook level."""
    price: float
    size: float  # In base asset units


@dataclass
class OrderbookData:
    """Orderbook snapshot."""
    asset: str
    exchange: str
    bids: List[OrderbookLevel]  # Sorted by price descending (best bid first)
    asks: List[OrderbookLevel]  # Sorted by price ascending (best ask first)
    mid_price: float
    spread_bps: float
    fetched_at: datetime = None
    source: str = "api"

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc)

    @property
    def is_expired(self) -> bool:
        """Check if data has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > SLIPPAGE_CACHE_TTL_SECONDS

    def get_bid_depth_usd(self, levels: int = 5) -> float:
        """Get total bid depth in USD for top N levels."""
        return sum(
            b.price * b.size for b in self.bids[:levels]
        )

    def get_ask_depth_usd(self, levels: int = 5) -> float:
        """Get total ask depth in USD for top N levels."""
        return sum(
            a.price * a.size for a in self.asks[:levels]
        )


@dataclass
class SlippageEstimate:
    """Slippage estimation result."""
    asset: str
    exchange: str
    order_size_usd: float
    side: str  # "buy" or "sell"
    estimated_slippage_bps: float
    expected_fill_price: float
    mid_price: float
    impact_bps: float  # Price impact from market order
    is_warning: bool  # True if slippage exceeds threshold
    source: str  # "orderbook" or "static"
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now(timezone.utc)


@dataclass
class CachedOrderbook:
    """Cached orderbook data."""
    data: OrderbookData
    fetched_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > SLIPPAGE_CACHE_TTL_SECONDS


class SlippageProvider:
    """
    Multi-exchange slippage estimation provider with caching.

    Fetches orderbook data when available, estimates slippage based on
    order size, and provides warnings for high-slippage scenarios.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize slippage provider.

        Args:
            testnet: Whether to use testnet APIs
        """
        self.testnet = testnet
        self._cache: Dict[str, CachedOrderbook] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=SLIPPAGE_API_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_cache_key(self, exchange: str, asset: str) -> str:
        """Get cache key for exchange/asset pair."""
        return f"{exchange.lower()}:{asset.upper()}"

    def _is_cache_valid(self, exchange: str, asset: str) -> bool:
        """Check if cached orderbook is still valid."""
        key = self._get_cache_key(exchange, asset)
        if key not in self._cache:
            return False
        return not self._cache[key].is_expired

    async def get_orderbook(
        self,
        asset: str,
        exchange: str = "hyperliquid",
        force_refresh: bool = False,
    ) -> Optional[OrderbookData]:
        """
        Get orderbook data for an asset on an exchange.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Exchange name
            force_refresh: Force API refresh even if cache valid

        Returns:
            OrderbookData or None if unavailable
        """
        key = self._get_cache_key(exchange, asset)
        asset_upper = asset.upper()
        exchange_lower = exchange.lower()

        # Return cached if valid and not forcing refresh
        if not force_refresh and self._is_cache_valid(exchange, asset):
            return self._cache[key].data

        # Try to fetch from API
        data = None

        if exchange_lower == "bybit":
            data = await self._fetch_bybit_orderbook(asset_upper)
        elif exchange_lower in ("hyperliquid", "aster"):
            data = await self._fetch_hyperliquid_orderbook(asset_upper)

        # Cache result if successful
        if data is not None:
            self._cache[key] = CachedOrderbook(
                data=data,
                fetched_at=datetime.now(timezone.utc),
            )

        return data

    async def estimate_slippage(
        self,
        asset: str,
        exchange: str,
        order_size_usd: float,
        side: str = "buy",
        force_refresh: bool = False,
    ) -> SlippageEstimate:
        """
        Estimate slippage for an order.

        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Exchange name
            order_size_usd: Order size in USD
            side: "buy" or "sell"
            force_refresh: Force orderbook refresh

        Returns:
            SlippageEstimate with expected slippage
        """
        asset_upper = asset.upper()
        exchange_lower = exchange.lower()

        # Try to get orderbook for accurate estimate
        orderbook = await self.get_orderbook(asset_upper, exchange_lower, force_refresh)

        if orderbook is not None:
            return self._estimate_from_orderbook(
                orderbook, order_size_usd, side
            )

        # Fall back to static estimates
        return self._estimate_static(
            asset_upper, exchange_lower, order_size_usd, side
        )

    def _estimate_from_orderbook(
        self,
        orderbook: OrderbookData,
        order_size_usd: float,
        side: str,
    ) -> SlippageEstimate:
        """
        Estimate slippage from orderbook depth.

        Simulates walking the orderbook to fill the order.
        """
        mid_price = orderbook.mid_price
        levels = orderbook.asks if side == "buy" else orderbook.bids

        if not levels:
            # No orderbook data, use spread as estimate
            slippage_bps = orderbook.spread_bps / 2
            expected_price = mid_price * (1 + slippage_bps / 10000) if side == "buy" else mid_price * (1 - slippage_bps / 10000)
            return SlippageEstimate(
                asset=orderbook.asset,
                exchange=orderbook.exchange,
                order_size_usd=order_size_usd,
                side=side,
                estimated_slippage_bps=slippage_bps,
                expected_fill_price=expected_price,
                mid_price=mid_price,
                impact_bps=slippage_bps,
                is_warning=slippage_bps > SLIPPAGE_WARNING_THRESHOLD_BPS,
                source="orderbook",
            )

        # Walk the orderbook
        remaining_usd = order_size_usd
        total_filled = 0.0
        total_cost = 0.0

        for level in levels:
            level_usd = level.price * level.size

            if remaining_usd <= 0:
                break

            if level_usd >= remaining_usd:
                # Partial fill at this level
                fill_amount = remaining_usd / level.price
                total_filled += fill_amount
                total_cost += remaining_usd
                remaining_usd = 0
            else:
                # Take whole level
                total_filled += level.size
                total_cost += level_usd
                remaining_usd -= level_usd

        if total_filled > 0:
            avg_fill_price = total_cost / total_filled
            impact_bps = abs(avg_fill_price - mid_price) / mid_price * 10000
        else:
            avg_fill_price = mid_price
            impact_bps = 0

        # Add half-spread for expected slippage
        slippage_bps = impact_bps + orderbook.spread_bps / 2

        return SlippageEstimate(
            asset=orderbook.asset,
            exchange=orderbook.exchange,
            order_size_usd=order_size_usd,
            side=side,
            estimated_slippage_bps=slippage_bps,
            expected_fill_price=avg_fill_price,
            mid_price=mid_price,
            impact_bps=impact_bps,
            is_warning=slippage_bps > SLIPPAGE_WARNING_THRESHOLD_BPS,
            source="orderbook",
        )

    def _estimate_static(
        self,
        asset: str,
        exchange: str,
        order_size_usd: float,
        side: str,
    ) -> SlippageEstimate:
        """
        Estimate slippage using static defaults.

        Falls back to conservative estimates based on typical market conditions.
        """
        # Determine size bucket
        if order_size_usd < SIZE_THRESHOLD_SMALL:
            size_bucket = "small"
        elif order_size_usd < SIZE_THRESHOLD_LARGE:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        # Get static slippage estimate
        exchange_rates = DEFAULT_SLIPPAGE_BPS.get(exchange, DEFAULT_SLIPPAGE_BPS["hyperliquid"])
        asset_rates = exchange_rates.get(asset, exchange_rates.get("BTC", {"small": 2, "medium": 4, "large": 10}))
        slippage_bps = asset_rates.get(size_bucket, 5.0)

        return SlippageEstimate(
            asset=asset,
            exchange=exchange,
            order_size_usd=order_size_usd,
            side=side,
            estimated_slippage_bps=slippage_bps,
            expected_fill_price=0,  # Unknown without orderbook
            mid_price=0,
            impact_bps=slippage_bps,
            is_warning=slippage_bps > SLIPPAGE_WARNING_THRESHOLD_BPS,
            source="static",
        )

    async def _fetch_bybit_orderbook(self, asset: str) -> Optional[OrderbookData]:
        """
        Fetch orderbook from Bybit API.

        Uses v5/market/orderbook endpoint.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            OrderbookData or None if unavailable
        """
        try:
            client = await self._get_client()
            symbol = f"{asset}USDT"
            base_url = BYBIT_TESTNET_API_URL if self.testnet else BYBIT_API_URL

            response = await client.get(
                f"{base_url}/v5/market/orderbook",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "limit": 25,  # Top 25 levels
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})

            # Parse bids and asks
            bids = [
                OrderbookLevel(price=float(b[0]), size=float(b[1]))
                for b in result.get("b", [])
            ]
            asks = [
                OrderbookLevel(price=float(a[0]), size=float(a[1]))
                for a in result.get("a", [])
            ]

            if not bids or not asks:
                return None

            best_bid = bids[0].price
            best_ask = asks[0].price
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid_price * 10000

            return OrderbookData(
                asset=asset,
                exchange="bybit",
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread_bps=spread_bps,
                source="api",
            )

        except Exception as e:
            print(f"[slippage:bybit] Error fetching orderbook for {asset}: {e}")

        return None

    async def _fetch_hyperliquid_orderbook(self, asset: str) -> Optional[OrderbookData]:
        """
        Fetch orderbook from Hyperliquid API.

        Uses the l2Book endpoint.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            OrderbookData or None if unavailable
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{HYPERLIQUID_API_URL}/info",
                json={
                    "type": "l2Book",
                    "coin": asset,
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            levels = data.get("levels", [[], []])

            if len(levels) < 2:
                return None

            # levels[0] = bids, levels[1] = asks
            # Each level: {"px": "price", "sz": "size", "n": num_orders}
            bids = [
                OrderbookLevel(price=float(b["px"]), size=float(b["sz"]))
                for b in levels[0]
            ]
            asks = [
                OrderbookLevel(price=float(a["px"]), size=float(a["sz"]))
                for a in levels[1]
            ]

            if not bids or not asks:
                return None

            best_bid = bids[0].price
            best_ask = asks[0].price
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid_price * 10000

            return OrderbookData(
                asset=asset,
                exchange="hyperliquid",
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread_bps=spread_bps,
                source="api",
            )

        except Exception as e:
            print(f"[slippage:hyperliquid] Error fetching orderbook for {asset}: {e}")

        return None

    def clear_cache(self) -> None:
        """Clear all cached orderbook data."""
        self._cache.clear()

    def get_cache_status(self) -> Dict[str, dict]:
        """
        Get cache status for all cached orderbooks.

        Returns:
            Dict mapping cache key to cache info
        """
        result = {}
        for key, cached in self._cache.items():
            data = cached.data
            result[key] = {
                "source": data.source,
                "mid_price": data.mid_price,
                "spread_bps": data.spread_bps,
                "bid_depth_usd": data.get_bid_depth_usd(),
                "ask_depth_usd": data.get_ask_depth_usd(),
                "is_expired": cached.is_expired,
            }
        return result


# Global singleton
_slippage_provider: Optional[SlippageProvider] = None


def get_slippage_provider() -> SlippageProvider:
    """Get the global slippage provider singleton."""
    global _slippage_provider
    if _slippage_provider is None:
        _slippage_provider = SlippageProvider()
    return _slippage_provider


def init_slippage_provider(testnet: bool = True) -> SlippageProvider:
    """
    Initialize the global slippage provider.

    Args:
        testnet: Whether to use testnet APIs

    Returns:
        Configured SlippageProvider
    """
    global _slippage_provider
    _slippage_provider = SlippageProvider(testnet=testnet)
    print(f"[slippage-provider] Initialized with testnet={testnet}, cache_ttl={SLIPPAGE_CACHE_TTL_SECONDS}s")
    return _slippage_provider
