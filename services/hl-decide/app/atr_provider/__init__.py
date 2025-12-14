"""
Multi-Exchange ATR Provider Module

Provides ATR data from multiple exchange sources for accurate stop calculations
when executing on different venues.

@module atr_provider
"""

from .interface import ATRProviderInterface, ATRData, Candle
from .hyperliquid import HyperliquidATRProvider
from .bybit import BybitATRProvider
from .manager import ATRManager, get_atr_manager, init_atr_manager

__all__ = [
    "ATRProviderInterface",
    "ATRData",
    "Candle",
    "HyperliquidATRProvider",
    "BybitATRProvider",
    "ATRManager",
    "get_atr_manager",
    "init_atr_manager",
]
