# Multi-Exchange Architecture Design

*Phase 3e Technical Specification - December 2025*

---

## 1. Overview

This document specifies the architecture for extending SigmaPilot beyond Hyperliquid to support multiple cryptocurrency exchanges. The design prioritizes:

- **Clean abstraction**: Exchange-agnostic core logic
- **Safety**: Credential encryption, rate limiting, error handling
- **Extensibility**: Easy to add new exchanges
- **Configurability**: Per-exchange limits via UI

---

## 2. Exchange Adapter Interface

### Core Interface (`packages/ts-lib/src/exchange/types.ts`)

```typescript
/**
 * Unified exchange adapter interface
 * All exchange-specific implementations must satisfy this contract
 */
export interface ExchangeAdapter {
  // Identity
  readonly name: ExchangeName;
  readonly type: 'cex' | 'dex';

  // Connection
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;

  // Account
  getBalance(): Promise<Balance>;
  getPositions(): Promise<Position[]>;
  getClosedPnL(since: Date): Promise<ClosedPosition[]>;

  // Market Data
  getMarkPrice(symbol: string): Promise<number>;
  getOrderBook(symbol: string, depth?: number): Promise<OrderBook>;
  subscribePrice(symbol: string, callback: (price: number) => void): () => void;

  // Orders
  openLong(params: OpenOrderParams): Promise<Order>;
  openShort(params: OpenOrderParams): Promise<Order>;
  closeLong(symbol: string, size?: number): Promise<Order>;
  closeShort(symbol: string, size?: number): Promise<Order>;
  cancelOrder(orderId: string): Promise<void>;
  cancelAllOrders(symbol?: string): Promise<void>;

  // Risk Management
  setStopLoss(symbol: string, price: number): Promise<Order>;
  setTakeProfit(symbol: string, price: number): Promise<Order>;
  cancelStopLoss(symbol: string): Promise<void>;
  cancelTakeProfit(symbol: string): Promise<void>;

  // Utilities
  formatQuantity(symbol: string, quantity: number): string;
  formatPrice(symbol: string, price: number): string;
  getSymbolInfo(symbol: string): Promise<SymbolInfo>;
}

// Type definitions
export type ExchangeName = 'hyperliquid' | 'binance' | 'bybit' | 'okx';

export interface Balance {
  total: number;          // Total equity
  available: number;      // Available for trading
  margin: number;         // Used margin
  unrealizedPnl: number;  // Unrealized P&L
  currency: string;       // Base currency (USD, USDT, etc.)
}

export interface Position {
  exchange: ExchangeName;
  symbol: string;
  side: 'long' | 'short';
  size: number;           // Position size in base currency
  entryPrice: number;
  markPrice: number;
  unrealizedPnl: number;
  leverage: number;
  marginUsed: number;
  liquidationPrice: number | null;
  timestamp: Date;
}

export interface ClosedPosition {
  exchange: ExchangeName;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  exitPrice: number;
  realizedPnl: number;
  openedAt: Date;
  closedAt: Date;
}

export interface OpenOrderParams {
  symbol: string;
  size: number;           // Size in quote currency (USD)
  leverage: number;
  stopLoss?: number;      // Optional SL price
  takeProfit?: number;    // Optional TP price
  reduceOnly?: boolean;
}

export interface Order {
  id: string;
  exchange: ExchangeName;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'take_profit';
  size: number;
  price?: number;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  filledSize: number;
  avgFillPrice: number;
  timestamp: Date;
}

export interface SymbolInfo {
  symbol: string;
  baseCurrency: string;
  quoteCurrency: string;
  minSize: number;
  maxSize: number;
  sizeStep: number;
  minPrice: number;
  maxPrice: number;
  priceStep: number;
  maxLeverage: number;
}
```

---

## 3. Exchange Implementations

### 3.1 Hyperliquid Adapter (`packages/ts-lib/src/exchange/hyperliquid.ts`)

```typescript
import { Hyperliquid } from 'hyperliquid';
import { ExchangeAdapter, ExchangeName, Balance, Position, ... } from './types';

export class HyperliquidAdapter implements ExchangeAdapter {
  readonly name: ExchangeName = 'hyperliquid';
  readonly type = 'dex' as const;

  private client: Hyperliquid | null = null;
  private readonly config: HyperliquidConfig;

  constructor(config: HyperliquidConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    this.client = new Hyperliquid({
      privateKey: this.config.privateKey,
      // ... other config
    });
    await this.client.connect();
  }

  async getBalance(): Promise<Balance> {
    const account = await this.client!.info.perpetuals.getClearinghouseState(
      this.config.address
    );
    return {
      total: parseFloat(account.marginSummary.accountValue),
      available: parseFloat(account.marginSummary.withdrawable),
      margin: parseFloat(account.marginSummary.totalMarginUsed),
      unrealizedPnl: parseFloat(account.marginSummary.totalNtlPos),
      currency: 'USD',
    };
  }

  async getPositions(): Promise<Position[]> {
    const state = await this.client!.info.perpetuals.getClearinghouseState(
      this.config.address
    );
    return state.assetPositions
      .filter(p => parseFloat(p.position.szi) !== 0)
      .map(p => ({
        exchange: this.name,
        symbol: p.position.coin,
        side: parseFloat(p.position.szi) > 0 ? 'long' : 'short',
        size: Math.abs(parseFloat(p.position.szi)),
        entryPrice: parseFloat(p.position.entryPx),
        markPrice: parseFloat(p.position.positionValue) / Math.abs(parseFloat(p.position.szi)),
        unrealizedPnl: parseFloat(p.position.unrealizedPnl),
        leverage: parseFloat(p.position.leverage.value),
        marginUsed: parseFloat(p.position.marginUsed),
        liquidationPrice: p.position.liquidationPx ? parseFloat(p.position.liquidationPx) : null,
        timestamp: new Date(),
      }));
  }

  async openLong(params: OpenOrderParams): Promise<Order> {
    const result = await this.client!.exchange.marketOpen({
      coin: params.symbol,
      isBuy: true,
      sz: this.formatQuantity(params.symbol, params.size / await this.getMarkPrice(params.symbol)),
      reduceOnly: params.reduceOnly ?? false,
    });
    // Convert to Order type...
  }

  // ... implement remaining methods
}

interface HyperliquidConfig {
  privateKey: string;
  address: string;
  testnet?: boolean;
}
```

### 3.2 Binance Futures Adapter (`packages/ts-lib/src/exchange/binance.ts`)

```typescript
import Binance from 'binance-api-node';
import { ExchangeAdapter, ExchangeName, Balance, Position, ... } from './types';

export class BinanceFuturesAdapter implements ExchangeAdapter {
  readonly name: ExchangeName = 'binance';
  readonly type = 'cex' as const;

  private client: ReturnType<typeof Binance> | null = null;
  private readonly config: BinanceConfig;

  constructor(config: BinanceConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    this.client = Binance({
      apiKey: this.config.apiKey,
      apiSecret: this.config.apiSecret,
      futures: true,
    });
    // Test connection
    await this.client.futuresAccountInfo();
  }

  async getBalance(): Promise<Balance> {
    const account = await this.client!.futuresAccountInfo();
    return {
      total: parseFloat(account.totalWalletBalance),
      available: parseFloat(account.availableBalance),
      margin: parseFloat(account.totalInitialMargin),
      unrealizedPnl: parseFloat(account.totalUnrealizedProfit),
      currency: 'USDT',
    };
  }

  async getPositions(): Promise<Position[]> {
    const positions = await this.client!.futuresPositionRisk();
    return positions
      .filter(p => parseFloat(p.positionAmt) !== 0)
      .map(p => ({
        exchange: this.name,
        symbol: p.symbol.replace('USDT', ''),  // Normalize symbol
        side: parseFloat(p.positionAmt) > 0 ? 'long' : 'short',
        size: Math.abs(parseFloat(p.positionAmt)),
        entryPrice: parseFloat(p.entryPrice),
        markPrice: parseFloat(p.markPrice),
        unrealizedPnl: parseFloat(p.unRealizedProfit),
        leverage: parseInt(p.leverage),
        marginUsed: parseFloat(p.initialMargin),
        liquidationPrice: parseFloat(p.liquidationPrice) || null,
        timestamp: new Date(),
      }));
  }

  async openLong(params: OpenOrderParams): Promise<Order> {
    // Set leverage first
    await this.client!.futuresLeverage({
      symbol: `${params.symbol}USDT`,
      leverage: params.leverage,
    });

    const result = await this.client!.futuresOrder({
      symbol: `${params.symbol}USDT`,
      side: 'BUY',
      type: 'MARKET',
      quantity: this.formatQuantity(params.symbol, params.size),
    });
    // Convert to Order type...
  }

  // ... implement remaining methods
}

interface BinanceConfig {
  apiKey: string;
  apiSecret: string;
  testnet?: boolean;
}
```

---

## 4. Exchange Manager

### Unified Manager (`packages/ts-lib/src/exchange/manager.ts`)

```typescript
import { ExchangeAdapter, ExchangeName, Position, Balance } from './types';
import { HyperliquidAdapter } from './hyperliquid';
import { BinanceFuturesAdapter } from './binance';

export class ExchangeManager {
  private adapters: Map<ExchangeName, ExchangeAdapter> = new Map();

  /**
   * Register an exchange adapter
   */
  register(adapter: ExchangeAdapter): void {
    this.adapters.set(adapter.name, adapter);
  }

  /**
   * Get a specific adapter
   */
  get(name: ExchangeName): ExchangeAdapter | undefined {
    return this.adapters.get(name);
  }

  /**
   * Connect all registered exchanges
   */
  async connectAll(): Promise<Map<ExchangeName, Error | null>> {
    const results = new Map<ExchangeName, Error | null>();

    await Promise.all(
      Array.from(this.adapters.entries()).map(async ([name, adapter]) => {
        try {
          await adapter.connect();
          results.set(name, null);
        } catch (error) {
          results.set(name, error as Error);
        }
      })
    );

    return results;
  }

  /**
   * Get aggregated balance across all exchanges
   */
  async getTotalBalance(): Promise<AggregatedBalance> {
    const balances = await Promise.all(
      Array.from(this.adapters.entries()).map(async ([name, adapter]) => {
        try {
          const balance = await adapter.getBalance();
          return { exchange: name, balance, error: null };
        } catch (error) {
          return { exchange: name, balance: null, error: error as Error };
        }
      })
    );

    const successful = balances.filter(b => b.balance !== null);

    return {
      total: successful.reduce((sum, b) => sum + b.balance!.total, 0),
      available: successful.reduce((sum, b) => sum + b.balance!.available, 0),
      margin: successful.reduce((sum, b) => sum + b.balance!.margin, 0),
      unrealizedPnl: successful.reduce((sum, b) => sum + b.balance!.unrealizedPnl, 0),
      byExchange: Object.fromEntries(
        balances.map(b => [b.exchange, b.balance])
      ),
      errors: Object.fromEntries(
        balances.filter(b => b.error).map(b => [b.exchange, b.error!.message])
      ),
    };
  }

  /**
   * Get all positions across all exchanges
   */
  async getAllPositions(): Promise<Position[]> {
    const positionArrays = await Promise.all(
      Array.from(this.adapters.values()).map(adapter =>
        adapter.getPositions().catch(() => [] as Position[])
      )
    );
    return positionArrays.flat();
  }

  /**
   * Get connected exchange names
   */
  getConnectedExchanges(): ExchangeName[] {
    return Array.from(this.adapters.entries())
      .filter(([_, adapter]) => adapter.isConnected())
      .map(([name, _]) => name);
  }
}

interface AggregatedBalance {
  total: number;
  available: number;
  margin: number;
  unrealizedPnl: number;
  byExchange: Record<ExchangeName, Balance | null>;
  errors: Record<ExchangeName, string>;
}
```

---

## 5. Auto-Trade Router

### Signal Router (`services/hl-decide/app/router.py`)

```python
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

class RoutingMode(Enum):
    SINGLE = "single"      # Execute on one exchange only
    MULTI = "multi"        # Execute on multiple exchanges

@dataclass
class ExchangeConfig:
    enabled: bool
    max_leverage: int
    max_position_pct: float  # % of that exchange's equity
    symbol_limits: Dict[str, float]  # Per-symbol limits

@dataclass
class AutoTradeConfig:
    enabled: bool
    routing_mode: RoutingMode
    exchanges: Dict[str, ExchangeConfig]
    require_approval: bool  # Human-in-loop for large trades

@dataclass
class RoutingDecision:
    exchange: str
    size_usd: float
    leverage: int
    reason: str

class SignalRouter:
    """
    Routes consensus signals to appropriate exchanges based on configuration.
    """

    def __init__(self, config: AutoTradeConfig):
        self.config = config

    def route(
        self,
        signal: ConsensusSignal,
        balances: Dict[str, float],
        positions: Dict[str, List[Position]],
    ) -> List[RoutingDecision]:
        """
        Decide which exchanges to execute on and with what size.

        Returns a list of routing decisions (empty if no execution).
        """
        if not self.config.enabled:
            return []

        decisions = []

        for exchange, ex_config in self.config.exchanges.items():
            if not ex_config.enabled:
                continue

            balance = balances.get(exchange, 0)
            if balance <= 0:
                continue

            # Check existing exposure
            existing_exposure = self._calculate_exposure(
                positions.get(exchange, []), signal.symbol
            )

            # Calculate max allowed size
            max_size = min(
                balance * ex_config.max_position_pct,
                ex_config.symbol_limits.get(signal.symbol, float('inf')),
            )

            # Reduce by existing exposure
            available_size = max(0, max_size - existing_exposure)

            if available_size < 100:  # Minimum $100 position
                continue

            # Determine actual size based on signal strength
            target_size = min(
                available_size,
                self._size_from_confidence(signal.confidence, balance),
            )

            decisions.append(RoutingDecision(
                exchange=exchange,
                size_usd=target_size,
                leverage=min(ex_config.max_leverage, self._leverage_from_ev(signal.ev)),
                reason=self._generate_reason(exchange, target_size, signal),
            ))

            # In SINGLE mode, only route to first viable exchange
            if self.config.routing_mode == RoutingMode.SINGLE:
                break

        return decisions

    def _calculate_exposure(
        self, positions: List[Position], symbol: str
    ) -> float:
        """Calculate current exposure for a symbol."""
        return sum(
            abs(p.size * p.entry_price)
            for p in positions
            if p.symbol == symbol
        )

    def _size_from_confidence(self, confidence: float, balance: float) -> float:
        """
        Scale position size by confidence.
        55% confidence -> 0.5% of balance
        75% confidence -> 2% of balance (max)
        """
        pct = 0.005 + (confidence - 0.55) * (0.015 / 0.20)
        return balance * min(pct, 0.02)

    def _leverage_from_ev(self, ev: float) -> int:
        """
        Scale leverage by expected value.
        EV 0.2R -> 1x
        EV 0.5R -> 3x
        EV 1.0R -> 5x (max)
        """
        if ev < 0.3:
            return 1
        elif ev < 0.5:
            return 2
        elif ev < 0.7:
            return 3
        elif ev < 0.9:
            return 4
        else:
            return 5

    def _generate_reason(
        self, exchange: str, size: float, signal: ConsensusSignal
    ) -> str:
        """Generate human-readable routing reason."""
        return (
            f"Routing to {exchange}: ${size:.0f} position "
            f"(confidence={signal.confidence:.0%}, EV={signal.ev:.2f}R)"
        )
```

---

## 6. Credential Management

### Encrypted Storage (`packages/ts-lib/src/exchange/credentials.ts`)

```typescript
import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const KEY_LENGTH = 32;
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

export class CredentialManager {
  private masterKey: Buffer;

  constructor(masterKeyHex: string) {
    if (masterKeyHex.length !== 64) {
      throw new Error('Master key must be 64 hex characters (32 bytes)');
    }
    this.masterKey = Buffer.from(masterKeyHex, 'hex');
  }

  /**
   * Encrypt credentials for storage
   */
  encrypt(plaintext: string): string {
    const iv = crypto.randomBytes(IV_LENGTH);
    const cipher = crypto.createCipheriv(ALGORITHM, this.masterKey, iv);

    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    // Format: iv:authTag:ciphertext
    return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
  }

  /**
   * Decrypt credentials from storage
   */
  decrypt(ciphertext: string): string {
    const parts = ciphertext.split(':');
    if (parts.length !== 3) {
      throw new Error('Invalid ciphertext format');
    }

    const iv = Buffer.from(parts[0], 'hex');
    const authTag = Buffer.from(parts[1], 'hex');
    const encrypted = parts[2];

    const decipher = crypto.createDecipheriv(ALGORITHM, this.masterKey, iv);
    decipher.setAuthTag(authTag);

    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }
}

// Database schema for credentials
export interface StoredCredential {
  id: string;
  exchange: ExchangeName;
  label: string;           // User-friendly name
  encryptedApiKey: string;
  encryptedApiSecret: string;
  encryptedPrivateKey?: string;  // For DEX
  address?: string;        // For DEX (public, not encrypted)
  testnet: boolean;
  createdAt: Date;
  lastUsedAt: Date;
}
```

---

## 7. Rate Limiting

### Per-Exchange Limiters (`packages/ts-lib/src/exchange/rate-limiter.ts`)

```typescript
import Bottleneck from 'bottleneck';

const EXCHANGE_LIMITS: Record<ExchangeName, { maxPerSecond: number; maxPerMinute: number }> = {
  hyperliquid: { maxPerSecond: 2, maxPerMinute: 100 },
  binance: { maxPerSecond: 10, maxPerMinute: 1200 },
  bybit: { maxPerSecond: 10, maxPerMinute: 600 },
  okx: { maxPerSecond: 10, maxPerMinute: 600 },
};

export class ExchangeRateLimiter {
  private limiters: Map<ExchangeName, Bottleneck> = new Map();

  constructor() {
    for (const [exchange, limits] of Object.entries(EXCHANGE_LIMITS)) {
      this.limiters.set(exchange as ExchangeName, new Bottleneck({
        maxConcurrent: 1,
        minTime: 1000 / limits.maxPerSecond,
        reservoir: limits.maxPerMinute,
        reservoirRefreshAmount: limits.maxPerMinute,
        reservoirRefreshInterval: 60 * 1000,
      }));
    }
  }

  /**
   * Wrap an async function with rate limiting
   */
  wrap<T>(exchange: ExchangeName, fn: () => Promise<T>): Promise<T> {
    const limiter = this.limiters.get(exchange);
    if (!limiter) {
      return fn();
    }
    return limiter.schedule(fn);
  }

  /**
   * Get current usage stats
   */
  getStats(exchange: ExchangeName): { running: number; queued: number } {
    const limiter = this.limiters.get(exchange);
    if (!limiter) {
      return { running: 0, queued: 0 };
    }
    const counts = limiter.counts();
    return { running: counts.RUNNING, queued: counts.QUEUED };
  }
}
```

---

## 8. Database Schema

### Migration (`db/migrations/050_exchange_credentials.sql`)

```sql
-- Exchange credentials (encrypted)
CREATE TABLE IF NOT EXISTS exchange_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exchange VARCHAR(32) NOT NULL,
    label VARCHAR(128) NOT NULL,
    encrypted_api_key TEXT NOT NULL,
    encrypted_api_secret TEXT,
    encrypted_private_key TEXT,
    address VARCHAR(64),  -- Public address for DEX
    testnet BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,

    UNIQUE(exchange, label)
);

-- Auto-trade configuration
CREATE TABLE IF NOT EXISTS autotrade_config (
    id SERIAL PRIMARY KEY,
    enabled BOOLEAN DEFAULT false,
    routing_mode VARCHAR(16) DEFAULT 'single',
    require_approval BOOLEAN DEFAULT true,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Per-exchange auto-trade settings
CREATE TABLE IF NOT EXISTS autotrade_exchange_config (
    exchange VARCHAR(32) PRIMARY KEY,
    enabled BOOLEAN DEFAULT false,
    max_leverage INT DEFAULT 3,
    max_position_pct DECIMAL(5,4) DEFAULT 0.02,
    symbol_limits JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize default config
INSERT INTO autotrade_config (enabled, routing_mode, require_approval)
VALUES (false, 'single', true)
ON CONFLICT DO NOTHING;

-- Initialize exchange defaults
INSERT INTO autotrade_exchange_config (exchange, enabled, max_leverage, max_position_pct)
VALUES
    ('hyperliquid', false, 5, 0.02),
    ('binance', false, 3, 0.01)
ON CONFLICT DO NOTHING;
```

---

## 9. API Endpoints

### Exchange Management (`services/hl-stream/src/routes/exchanges.ts`)

```typescript
router.get('/api/exchanges', async (req, res) => {
  // List configured exchanges (no secrets)
  const credentials = await db.query(
    `SELECT id, exchange, label, address, testnet, created_at, last_used_at
     FROM exchange_credentials`
  );
  res.json(credentials.rows);
});

router.post('/api/exchanges', async (req, res) => {
  // Add new exchange credentials
  const { exchange, label, apiKey, apiSecret, privateKey, address, testnet } = req.body;

  // Validate exchange type
  if (!SUPPORTED_EXCHANGES.includes(exchange)) {
    return res.status(400).json({ error: 'Unsupported exchange' });
  }

  // Encrypt credentials
  const encApiKey = credentialManager.encrypt(apiKey);
  const encApiSecret = apiSecret ? credentialManager.encrypt(apiSecret) : null;
  const encPrivateKey = privateKey ? credentialManager.encrypt(privateKey) : null;

  // Test connection before saving
  const adapter = createAdapter(exchange, { apiKey, apiSecret, privateKey, address, testnet });
  try {
    await adapter.connect();
  } catch (error) {
    return res.status(400).json({ error: 'Connection test failed', details: error.message });
  }

  // Save to database
  const result = await db.query(
    `INSERT INTO exchange_credentials
     (exchange, label, encrypted_api_key, encrypted_api_secret, encrypted_private_key, address, testnet)
     VALUES ($1, $2, $3, $4, $5, $6, $7)
     RETURNING id`,
    [exchange, label, encApiKey, encApiSecret, encPrivateKey, address, testnet]
  );

  res.json({ id: result.rows[0].id, exchange, label });
});

router.delete('/api/exchanges/:id', async (req, res) => {
  await db.query('DELETE FROM exchange_credentials WHERE id = $1', [req.params.id]);
  res.json({ success: true });
});

router.post('/api/exchanges/:id/test', async (req, res) => {
  // Test connection for existing credentials
  const cred = await getDecryptedCredentials(req.params.id);
  const adapter = createAdapter(cred.exchange, cred);

  try {
    await adapter.connect();
    const balance = await adapter.getBalance();
    res.json({ success: true, balance });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});
```

---

## 10. Environment Variables

```bash
# Master key for credential encryption (generate with: openssl rand -hex 32)
CREDENTIAL_MASTER_KEY=your-64-char-hex-key-here

# Default exchange (for backward compatibility)
DEFAULT_EXCHANGE=hyperliquid

# Rate limiting overrides (optional)
HL_SDK_CALLS_PER_SECOND=2
BINANCE_CALLS_PER_SECOND=10
```

---

## 11. Migration Notes

### For Existing Users

1. **Generate master key**: `openssl rand -hex 32`
2. **Add to `.env`**: `CREDENTIAL_MASTER_KEY=<your-key>`
3. **Run migration**: Automatic on service startup
4. **Re-add credentials**: Via new Settings UI (old env vars deprecated)

### Breaking Changes

- `HYPERLIQUID_PRIVATE_KEY` env var deprecated (use encrypted storage)
- `HYPERLIQUID_ADDRESS` env var deprecated (use encrypted storage)
- New database tables required

### Backward Compatibility

During transition period, system will:
1. Check for new encrypted credentials in database
2. Fall back to env vars if no database credentials exist
3. Log deprecation warning if using env vars

---

## 12. Future Extensions

### Planned Exchanges (P2)
- Bybit (uses Binance-like API)
- OKX
- dYdX

### Planned Features
- Unified order history
- Cross-exchange arbitrage detection
- Portfolio rebalancing

---

*This design enables SigmaPilot to grow beyond Hyperliquid while maintaining security and simplicity.*
