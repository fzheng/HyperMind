"""
Tests for Account Normalizer

@module tests.test_account_normalizer
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.account_normalizer import (
    AccountNormalizer,
    CurrencyRate,
    NormalizedBalance,
    NormalizedPosition,
    get_account_normalizer,
    init_account_normalizer,
    USDT_RATE_CACHE_TTL_SECONDS,
    DEFAULT_USDT_USD_RATE,
    DEPEG_WARNING_THRESHOLD,
)
from app.exchanges.interface import Balance, Position, PositionSide, MarginMode


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def normalizer():
    """Create fresh account normalizer for testing."""
    return AccountNormalizer()


@pytest.fixture
def usd_balance():
    """Create USD-denominated balance."""
    return Balance(
        total_equity=100000.0,
        available_balance=80000.0,
        margin_used=20000.0,
        unrealized_pnl=500.0,
        realized_pnl_today=-200.0,
        currency="USD",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def usdt_balance():
    """Create USDT-denominated balance."""
    return Balance(
        total_equity=100000.0,
        available_balance=80000.0,
        margin_used=20000.0,
        unrealized_pnl=500.0,
        realized_pnl_today=-200.0,
        currency="USDT",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def btc_position():
    """Create BTC position."""
    return Position(
        symbol="BTC",
        side=PositionSide.LONG,
        size=1.0,
        entry_price=50000.0,
        mark_price=51000.0,
        unrealized_pnl=1000.0,
        leverage=1,
        margin_mode=MarginMode.CROSS,
        timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# CurrencyRate Tests
# =============================================================================


class TestCurrencyRate:
    """Tests for CurrencyRate dataclass."""

    def test_is_expired_fresh(self):
        """Fresh rate is not expired."""
        rate = CurrencyRate(
            rate=0.9995,
            source="api",
            fetched_at=datetime.now(timezone.utc),
        )
        assert rate.is_expired is False

    def test_is_expired_old(self):
        """Old rate is expired."""
        rate = CurrencyRate(
            rate=0.9995,
            source="api",
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=USDT_RATE_CACHE_TTL_SECONDS + 10),
        )
        assert rate.is_expired is True

    def test_age_seconds(self):
        """Age calculation is correct."""
        rate = CurrencyRate(
            rate=1.0,
            source="api",
            fetched_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        assert 29 <= rate.age_seconds <= 31


# =============================================================================
# NormalizedBalance Tests
# =============================================================================


class TestNormalizedBalance:
    """Tests for NormalizedBalance dataclass."""

    def test_margin_ratio(self):
        """Margin ratio calculated correctly."""
        balance = Balance(
            total_equity=100000.0,
            available_balance=80000.0,
            margin_used=20000.0,
            currency="USD",
        )
        normalized = NormalizedBalance(
            original=balance,
            total_equity_usd=100000.0,
            available_balance_usd=80000.0,
            margin_used_usd=20000.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_today_usd=0.0,
            conversion_rate=1.0,
            conversion_source="identity",
        )
        assert normalized.margin_ratio == 0.2  # 20%

    def test_margin_ratio_zero_equity(self):
        """Margin ratio is 0 when equity is 0."""
        balance = Balance(
            total_equity=0.0,
            available_balance=0.0,
            margin_used=0.0,
            currency="USD",
        )
        normalized = NormalizedBalance(
            original=balance,
            total_equity_usd=0.0,
            available_balance_usd=0.0,
            margin_used_usd=0.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_today_usd=0.0,
            conversion_rate=1.0,
            conversion_source="identity",
        )
        assert normalized.margin_ratio == 0.0

    def test_depeg_warning_identity(self):
        """No depeg warning for USD (identity conversion)."""
        balance = Balance(total_equity=100000.0, available_balance=80000.0, margin_used=20000.0, currency="USD")
        normalized = NormalizedBalance(
            original=balance,
            total_equity_usd=100000.0,
            available_balance_usd=80000.0,
            margin_used_usd=20000.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_today_usd=0.0,
            conversion_rate=1.0,
            conversion_source="identity",
        )
        assert normalized.is_depeg_warning is False

    def test_depeg_warning_stable(self):
        """No depeg warning for rate close to 1.0."""
        balance = Balance(total_equity=100000.0, available_balance=80000.0, margin_used=20000.0, currency="USDT")
        normalized = NormalizedBalance(
            original=balance,
            total_equity_usd=100020.0,
            available_balance_usd=80016.0,
            margin_used_usd=20004.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_today_usd=0.0,
            conversion_rate=1.0002,  # Within threshold
            conversion_source="api",
        )
        assert normalized.is_depeg_warning is False

    def test_depeg_warning_triggered(self):
        """Depeg warning when rate deviates significantly."""
        balance = Balance(total_equity=100000.0, available_balance=80000.0, margin_used=20000.0, currency="USDT")
        normalized = NormalizedBalance(
            original=balance,
            total_equity_usd=99000.0,
            available_balance_usd=79200.0,
            margin_used_usd=19800.0,
            unrealized_pnl_usd=0.0,
            realized_pnl_today_usd=0.0,
            conversion_rate=0.99,  # 1% depeg
            conversion_source="api",
        )
        assert normalized.is_depeg_warning is True


# =============================================================================
# AccountNormalizer Tests
# =============================================================================


class TestAccountNormalizer:
    """Tests for AccountNormalizer class."""

    @pytest.mark.asyncio
    async def test_normalize_usd_balance(self, normalizer, usd_balance):
        """USD balance normalizes with identity conversion."""
        normalized = await normalizer.normalize_balance(usd_balance)

        assert normalized.total_equity_usd == usd_balance.total_equity
        assert normalized.available_balance_usd == usd_balance.available_balance
        assert normalized.margin_used_usd == usd_balance.margin_used
        assert normalized.conversion_rate == 1.0
        assert normalized.conversion_source == "identity"

    @pytest.mark.asyncio
    async def test_normalize_usdt_balance_api(self, normalizer, usdt_balance):
        """USDT balance normalizes with API rate."""
        # Mock the API call
        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=0.9998):
            normalized = await normalizer.normalize_balance(usdt_balance)

            assert normalized.conversion_rate == 0.9998
            assert normalized.conversion_source == "api"
            assert normalized.total_equity_usd == pytest.approx(99980.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_normalize_usdt_balance_fallback(self, normalizer, usdt_balance):
        """USDT balance uses fallback when API unavailable."""
        # Mock API failure
        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=None):
            normalized = await normalizer.normalize_balance(usdt_balance)

            assert normalized.conversion_rate == DEFAULT_USDT_USD_RATE
            assert normalized.conversion_source == "fallback"

    @pytest.mark.asyncio
    async def test_caching_prevents_api_calls(self, normalizer, usdt_balance):
        """Cached rate prevents repeated API calls."""
        mock_fetch = AsyncMock(return_value=0.9999)
        with patch.object(normalizer, "_fetch_usdt_rate", mock_fetch):
            # First call
            await normalizer.normalize_balance(usdt_balance)
            # Second call should use cache
            await normalizer.normalize_balance(usdt_balance)

            assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self, normalizer, usdt_balance):
        """Force refresh fetches new rate."""
        mock_fetch = AsyncMock(return_value=0.9999)
        with patch.object(normalizer, "_fetch_usdt_rate", mock_fetch):
            # First call
            await normalizer.normalize_balance(usdt_balance)
            # Force refresh
            await normalizer.normalize_balance(usdt_balance, force_refresh_rate=True)

            assert mock_fetch.call_count == 2

    def test_normalize_balance_sync_usd(self, normalizer, usd_balance):
        """Sync normalization works for USD."""
        normalized = normalizer.normalize_balance_sync(usd_balance)

        assert normalized.total_equity_usd == usd_balance.total_equity
        assert normalized.conversion_source == "identity"

    def test_normalize_balance_sync_usdt_fallback(self, normalizer, usdt_balance):
        """Sync normalization uses fallback for USDT without cache."""
        normalized = normalizer.normalize_balance_sync(usdt_balance)

        # No cache, uses fallback
        assert normalized.conversion_rate == DEFAULT_USDT_USD_RATE
        assert normalized.conversion_source == "fallback"

    @pytest.mark.asyncio
    async def test_normalize_balance_sync_usdt_cached(self, normalizer, usdt_balance):
        """Sync normalization uses cached rate for USDT."""
        # Pre-populate cache via async call
        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=0.9997):
            await normalizer.get_usdt_usd_rate()

        # Sync call should use cache
        normalized = normalizer.normalize_balance_sync(usdt_balance)

        assert normalized.conversion_rate == 0.9997
        assert normalized.conversion_source == "api"

    @pytest.mark.asyncio
    async def test_normalize_position_usd(self, normalizer, btc_position):
        """Position with USD quote normalizes correctly."""
        normalized = await normalizer.normalize_position(btc_position, quote_currency="USD")

        assert normalized.notional_value_usd == btc_position.notional_value
        assert normalized.conversion_rate == 1.0
        assert normalized.conversion_source == "identity"

    @pytest.mark.asyncio
    async def test_normalize_position_usdt(self, normalizer, btc_position):
        """Position with USDT quote normalizes correctly."""
        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=0.9998):
            normalized = await normalizer.normalize_position(btc_position, quote_currency="USDT")

            # BTC position: 1.0 * 51000 mark_price = 51000 notional
            expected_usd = 51000.0 * 0.9998
            assert normalized.notional_value_usd == pytest.approx(expected_usd, rel=0.001)

    def test_normalize_position_sync(self, normalizer, btc_position):
        """Sync position normalization works."""
        normalized = normalizer.normalize_position_sync(btc_position, quote_currency="USD")

        assert normalized.notional_value_usd == btc_position.notional_value

    def test_get_conversion_rate_usd(self, normalizer):
        """USD conversion is identity."""
        rate, source = normalizer.get_conversion_rate("USD")
        assert rate == 1.0
        assert source == "identity"

    def test_get_conversion_rate_usdt_no_cache(self, normalizer):
        """USDT without cache returns fallback."""
        rate, source = normalizer.get_conversion_rate("USDT")
        assert rate == DEFAULT_USDT_USD_RATE
        assert source == "fallback"

    def test_get_conversion_rate_unknown(self, normalizer):
        """Unknown currency assumes 1:1."""
        rate, source = normalizer.get_conversion_rate("BTC")
        assert rate == 1.0
        assert source == "assumed"

    def test_clear_cache(self, normalizer):
        """Cache can be cleared."""
        # Add to cache
        normalizer._rate_cache["USDT_USD"] = CurrencyRate(rate=0.999, source="api")

        normalizer.clear_cache()

        assert len(normalizer._rate_cache) == 0

    def test_get_cache_status(self, normalizer):
        """Cache status reports correctly."""
        normalizer._rate_cache["USDT_USD"] = CurrencyRate(
            rate=0.999,
            source="api",
            fetched_at=datetime.now(timezone.utc),
        )

        status = normalizer.get_cache_status()

        assert "USDT_USD" in status
        assert status["USDT_USD"]["rate"] == 0.999
        assert status["USDT_USD"]["source"] == "api"
        assert status["USDT_USD"]["is_expired"] is False

    @pytest.mark.asyncio
    async def test_close_client(self, normalizer):
        """HTTP client can be closed."""
        # Create client
        await normalizer._get_client()
        assert normalizer._client is not None

        await normalizer.close()
        assert normalizer._client is None


class TestCoinGeckoAPI:
    """Tests for CoinGecko API integration."""

    @pytest.mark.asyncio
    async def test_fetch_usdt_rate_success(self, normalizer):
        """Successfully parses CoinGecko API response."""
        with patch.object(normalizer, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value={"tether": {"usd": 0.9997}})

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            rate = await normalizer._fetch_usdt_rate()

            assert rate == pytest.approx(0.9997)

    @pytest.mark.asyncio
    async def test_fetch_usdt_rate_api_error(self, normalizer):
        """Handles API errors gracefully."""
        with patch.object(normalizer, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            rate = await normalizer._fetch_usdt_rate()

            assert rate is None

    @pytest.mark.asyncio
    async def test_fetch_usdt_rate_missing_data(self, normalizer):
        """Handles missing data in response."""
        with patch.object(normalizer, "_get_client") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value={})  # Missing tether data

            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http

            rate = await normalizer._fetch_usdt_rate()

            assert rate is None

    @pytest.mark.asyncio
    async def test_fetch_usdt_rate_exception(self, normalizer):
        """Handles exceptions gracefully."""
        with patch.object(normalizer, "_get_client") as mock_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.return_value = mock_http

            rate = await normalizer._fetch_usdt_rate()

            assert rate is None


class TestAccountNormalizerSingleton:
    """Tests for global singleton."""

    def test_get_account_normalizer_singleton(self):
        """get_account_normalizer returns same instance."""
        # Reset singleton for test
        import app.account_normalizer as module
        module._account_normalizer = None

        normalizer1 = get_account_normalizer()
        normalizer2 = get_account_normalizer()

        assert normalizer1 is normalizer2

    @pytest.mark.asyncio
    async def test_init_account_normalizer(self):
        """init_account_normalizer creates new instance and pre-fetches rate."""
        import app.account_normalizer as module
        module._account_normalizer = None

        with patch.object(AccountNormalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=0.9999):
            normalizer = await init_account_normalizer()

            assert normalizer is not None
            assert get_account_normalizer() is normalizer

            # Rate should be cached
            rate = normalizer._rate_cache.get("USDT_USD")
            assert rate is not None
            assert rate.rate == 0.9999


class TestIntegration:
    """Integration-style tests."""

    @pytest.mark.asyncio
    async def test_aggregate_multi_exchange_exposure(self):
        """Aggregate exposure across multiple exchanges."""
        normalizer = AccountNormalizer()

        # Simulate HL balance (USD)
        hl_balance = Balance(
            total_equity=50000.0,
            available_balance=40000.0,
            margin_used=10000.0,
            currency="USD",
        )

        # Simulate Bybit balance (USDT)
        bybit_balance = Balance(
            total_equity=30000.0,
            available_balance=25000.0,
            margin_used=5000.0,
            currency="USDT",
        )

        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=0.999):
            hl_normalized = await normalizer.normalize_balance(hl_balance)
            bybit_normalized = await normalizer.normalize_balance(bybit_balance)

            # Aggregate totals
            total_equity = hl_normalized.total_equity_usd + bybit_normalized.total_equity_usd
            total_margin = hl_normalized.margin_used_usd + bybit_normalized.margin_used_usd

            # Expected: 50000 + 30000*0.999 = 50000 + 29970 = 79970
            assert total_equity == pytest.approx(79970.0, rel=0.01)
            # Expected: 10000 + 5000*0.999 = 10000 + 4995 = 14995
            assert total_margin == pytest.approx(14995.0, rel=0.01)

        await normalizer.close()

    @pytest.mark.asyncio
    async def test_risk_check_with_normalization(self):
        """Risk check with normalized values across venues."""
        normalizer = AccountNormalizer()

        # 10% max position size rule
        MAX_POSITION_PCT = 0.10

        # Bybit position (USDT quote)
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.5,
            entry_price=100000.0,
            mark_price=100000.0,  # $50k notional
        )

        # Bybit balance
        balance = Balance(
            total_equity=200000.0,
            available_balance=150000.0,
            margin_used=50000.0,
            currency="USDT",
        )

        with patch.object(normalizer, "_fetch_usdt_rate", new_callable=AsyncMock, return_value=1.0):
            norm_balance = await normalizer.normalize_balance(balance)
            norm_position = await normalizer.normalize_position(position, quote_currency="USDT")

            # Position size as % of equity
            position_pct = norm_position.notional_value_usd / norm_balance.total_equity_usd

            # 50000 / 200000 = 25%
            assert position_pct == 0.25
            assert position_pct > MAX_POSITION_PCT  # Would fail risk check

        await normalizer.close()
