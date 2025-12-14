"""
Bybit ATR Provider

Provides ATR data from Bybit via the REST API klines endpoint.

@module atr_provider.bybit
"""

import os
from datetime import datetime, timezone
from typing import Optional, List

import httpx

from .interface import (
    ATRProviderInterface,
    ATRData,
    Candle,
    calculate_atr,
    ATR_PERIOD,
)


# Bybit API configuration
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com")
BYBIT_TESTNET_API_URL = os.getenv("BYBIT_TESTNET_API_URL", "https://api-testnet.bybit.com")
BYBIT_ATR_TIMEOUT_SECONDS = int(os.getenv("BYBIT_ATR_TIMEOUT_SECONDS", "10"))


class BybitATRProvider(ATRProviderInterface):
    """
    ATR provider using Bybit klines API.

    Fetches 1-minute candles from Bybit for ATR calculation.
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize Bybit ATR provider.

        Args:
            testnet: Whether to use testnet API
        """
        super().__init__("bybit")
        self.testnet = testnet
        self.base_url = BYBIT_TESTNET_API_URL if testnet else BYBIT_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        # Bybit public API doesn't require credentials for klines
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=BYBIT_ATR_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _format_symbol(self, asset: str) -> str:
        """Format asset to Bybit symbol format."""
        asset_upper = asset.upper()
        if asset_upper in ("BTC", "ETH"):
            return f"{asset_upper}USDT"
        return asset_upper

    async def get_atr(self, asset: str, price: Optional[float] = None) -> ATRData:
        """
        Get ATR data for an asset from Bybit klines API.

        Args:
            asset: Asset symbol (BTC, ETH)
            price: Current price (optional, will fetch from API)

        Returns:
            ATRData with ATR values
        """
        asset_upper = asset.upper()

        # Check cache first
        cached = self._get_cached(asset_upper)
        if cached is not None:
            if price is not None and price != cached.price and price > 0:
                cached = ATRData(
                    asset=asset_upper,
                    atr=cached.atr,
                    atr_pct=cached.atr / price * 100,
                    price=price,
                    multiplier=cached.multiplier,
                    stop_distance_pct=cached.atr / price * 100 * cached.multiplier,
                    timestamp=cached.timestamp,
                    source=cached.source,
                    exchange=cached.exchange,
                )
            return cached

        # Fetch candles and calculate ATR
        try:
            candles = await self.get_candles(asset_upper, ATR_PERIOD + 5)

            if len(candles) >= ATR_PERIOD + 1:
                atr = calculate_atr(candles, ATR_PERIOD)
                if atr is not None and atr > 0:
                    current_price = price or float(candles[0].close)
                    multiplier = self._get_multiplier(asset_upper)

                    atr_data = ATRData(
                        asset=asset_upper,
                        atr=atr,
                        atr_pct=atr / current_price * 100 if current_price > 0 else 0,
                        price=current_price,
                        multiplier=multiplier,
                        stop_distance_pct=atr / current_price * 100 * multiplier if current_price > 0 else 0,
                        timestamp=candles[0].ts,
                        source="api",
                        exchange="bybit",
                    )
                    self._set_cached(asset_upper, atr_data)
                    return atr_data

        except Exception as e:
            print(f"[atr:bybit] Failed to fetch ATR for {asset}: {e}")

        # Fallback
        return self._fallback_atr(asset_upper, price)

    async def get_candles(
        self,
        asset: str,
        count: int = ATR_PERIOD + 5,
    ) -> List[Candle]:
        """
        Fetch OHLC candles from Bybit klines API.

        Uses v5 market kline endpoint.

        Args:
            asset: Asset symbol (BTC, ETH)
            count: Number of candles to fetch

        Returns:
            List of Candle objects, newest first
        """
        try:
            client = await self._get_client()
            symbol = self._format_symbol(asset)

            # Bybit v5 klines endpoint
            # https://bybit-exchange.github.io/docs/v5/market/kline
            response = await client.get(
                f"{self.base_url}/v5/market/kline",
                params={
                    "category": "linear",  # USDT perpetual
                    "symbol": symbol,
                    "interval": "1",  # 1 minute
                    "limit": count,
                },
            )

            if response.status_code != 200:
                print(f"[atr:bybit] API error {response.status_code}: {response.text}")
                return []

            data = response.json()
            if data.get("retCode") != 0:
                print(f"[atr:bybit] API returned error: {data.get('retMsg')}")
                return []

            # Parse kline data
            # Format: [startTime, open, high, low, close, volume, turnover]
            result = data.get("result", {})
            klines = result.get("list", [])

            candles = []
            for kline in klines:
                if len(kline) >= 5:
                    ts_ms = int(kline[0])
                    candles.append(
                        Candle(
                            ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                        )
                    )

            # Bybit returns newest first, which is what we want
            return candles

        except httpx.TimeoutException:
            print(f"[atr:bybit] Timeout fetching klines for {asset}")
            return []
        except Exception as e:
            print(f"[atr:bybit] Error fetching klines for {asset}: {e}")
            return []

    async def get_market_price(self, asset: str) -> Optional[float]:
        """
        Get current market price from Bybit.

        Args:
            asset: Asset symbol (BTC, ETH)

        Returns:
            Current price or None
        """
        try:
            client = await self._get_client()
            symbol = self._format_symbol(asset)

            response = await client.get(
                f"{self.base_url}/v5/market/tickers",
                params={
                    "category": "linear",
                    "symbol": symbol,
                },
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if data.get("retCode") != 0:
                return None

            result = data.get("result", {})
            tickers = result.get("list", [])

            if tickers and len(tickers) > 0:
                return float(tickers[0].get("lastPrice", 0))

            return None

        except Exception as e:
            print(f"[atr:bybit] Error fetching price for {asset}: {e}")
            return None
