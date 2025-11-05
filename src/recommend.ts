import { clamp, nowIso } from './utils';
import type { Recommendation } from './types';

export function computeRecommendation(
  address: string,
  exposureBtc: number,
  priceUsd: number
): Recommendation {
  // Simple heuristic:
  // - Score is sign(exposure) * tanh(|exposure| / 5)
  //   (mild for small exposure, saturates for large exposure)
  const strength = Math.tanh(Math.abs(exposureBtc) / 5);
  const sign = Math.sign(exposureBtc);
  const score = clamp(sign * strength, -1, 1);

  let text: string;
  if (exposureBtc === 0) {
    text = `No open BTC perp exposure.`;
  } else if (exposureBtc > 0) {
    text = `Net long ${exposureBtc.toFixed(4)} BTC perp. Consider downside risk if price falls.`;
  } else {
    text = `Net short ${Math.abs(exposureBtc).toFixed(4)} BTC perp. Consider upside risk if price rises.`;
  }

  return {
    address,
    exposureBtc,
    priceUsd,
    score,
    text,
    timestamp: nowIso()
  };
}
