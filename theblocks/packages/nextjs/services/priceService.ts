"use client";

/**
 * @title Price Service
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Dynamic price fetching with fallback to oracles
 * @dev Fetches prices from multiple sources with caching
 */

export interface TokenPrice {
  symbol: string;
  price: number;
  change24h: number;
  lastUpdated: number;
  source: "coingecko" | "oracle" | "fallback";
}

// CoinGecko token IDs mapping
const TOKEN_IDS: Record<string, string> = {
  ETH: "ethereum",
  WETH: "ethereum",
  USDC: "usd-coin",
  USDT: "tether",
  DAI: "dai",
  WBTC: "wrapped-bitcoin",
  LINK: "chainlink",
  BTC: "bitcoin",
};

// Fallback prices (used when APIs fail)
const FALLBACK_PRICES: Record<string, number> = {
  ETH: 2500,
  WETH: 2500,
  USDC: 1,
  USDT: 1,
  DAI: 1,
  WBTC: 45000,
  LINK: 15,
  BTC: 45000,
};

// Price cache
const priceCache: Map<string, TokenPrice> = new Map();
const CACHE_DURATION = 30000; // 30 seconds

// Price update listeners
const priceListeners: Set<(prices: Map<string, TokenPrice>) => void> = new Set();

/**
 * Subscribe to price updates
 */
export function subscribeToPrices(callback: (prices: Map<string, TokenPrice>) => void): () => void {
  priceListeners.add(callback);
  // Send current cache immediately
  if (priceCache.size > 0) {
    callback(priceCache);
  }
  return () => priceListeners.delete(callback);
}

/**
 * Notify all listeners of price updates
 */
function notifyPriceListeners() {
  priceListeners.forEach(callback => callback(priceCache));
}

/**
 * Fetch prices from CoinGecko API
 */
async function fetchFromCoinGecko(symbols: string[]): Promise<Map<string, TokenPrice>> {
  const prices = new Map<string, TokenPrice>();

  try {
    const ids = symbols
      .map(s => TOKEN_IDS[s])
      .filter(Boolean)
      .join(",");

    if (!ids) return prices;

    const response = await fetch(
      `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_24hr_change=true`,
      { next: { revalidate: 30 } },
    );

    if (!response.ok) throw new Error("CoinGecko API error");

    const data = await response.json();

    for (const symbol of symbols) {
      const id = TOKEN_IDS[symbol];
      if (id && data[id]) {
        prices.set(symbol, {
          symbol,
          price: data[id].usd,
          change24h: data[id].usd_24h_change || 0,
          lastUpdated: Date.now(),
          source: "coingecko",
        });
      }
    }
  } catch (error) {
    console.warn("CoinGecko fetch failed:", error);
  }

  return prices;
}

/**
 * Get prices with simulation variance (for realistic demo)
 */
function getSimulatedPrice(basePrice: number): number {
  // Add small random variance (±0.5%) for realism
  const variance = (Math.random() - 0.5) * 0.01;
  return basePrice * (1 + variance);
}

/**
 * Get current price for a token
 */
export async function getPrice(symbol: string): Promise<TokenPrice> {
  // Check cache first
  const cached = priceCache.get(symbol);
  if (cached && Date.now() - cached.lastUpdated < CACHE_DURATION) {
    return cached;
  }

  // Try to fetch fresh prices
  const freshPrices = await fetchFromCoinGecko([symbol]);
  if (freshPrices.has(symbol)) {
    const price = freshPrices.get(symbol)!;
    priceCache.set(symbol, price);
    return price;
  }

  // Use fallback with simulated variance
  const fallbackPrice = FALLBACK_PRICES[symbol] || 1;
  const simulatedPrice = getSimulatedPrice(fallbackPrice);

  const price: TokenPrice = {
    symbol,
    price: simulatedPrice,
    change24h: (Math.random() - 0.5) * 10, // Random ±5% change
    lastUpdated: Date.now(),
    source: "fallback",
  };

  priceCache.set(symbol, price);
  return price;
}

/**
 * Get prices for multiple tokens
 */
export async function getPrices(symbols: string[]): Promise<Map<string, TokenPrice>> {
  const now = Date.now();
  const needsFetch: string[] = [];
  const result = new Map<string, TokenPrice>();

  // Check which tokens need fresh prices
  for (const symbol of symbols) {
    const cached = priceCache.get(symbol);
    if (cached && now - cached.lastUpdated < CACHE_DURATION) {
      result.set(symbol, cached);
    } else {
      needsFetch.push(symbol);
    }
  }

  // Fetch missing prices
  if (needsFetch.length > 0) {
    const freshPrices = await fetchFromCoinGecko(needsFetch);

    for (const symbol of needsFetch) {
      if (freshPrices.has(symbol)) {
        const price = freshPrices.get(symbol)!;
        priceCache.set(symbol, price);
        result.set(symbol, price);
      } else {
        // Use fallback
        const price = await getPrice(symbol);
        result.set(symbol, price);
      }
    }
  }

  notifyPriceListeners();
  return result;
}

/**
 * Calculate swap quote with dynamic pricing
 */
export async function calculateSwapQuote(
  tokenInSymbol: string,
  tokenOutSymbol: string,
  amountIn: number,
  feePercent: number = 0.3,
): Promise<{
  amountOut: number;
  fee: number;
  priceImpact: number;
  rate: number;
  priceIn: number;
  priceOut: number;
}> {
  const [priceIn, priceOut] = await Promise.all([getPrice(tokenInSymbol), getPrice(tokenOutSymbol)]);

  const valueIn = amountIn * priceIn.price;
  const rawAmountOut = valueIn / priceOut.price;

  // Calculate fee
  const fee = rawAmountOut * (feePercent / 100);
  const amountOut = rawAmountOut - fee;

  // Calculate price impact (based on trade size)
  // Larger trades have higher impact
  const priceImpact = Math.min(amountIn * 0.05, 5); // Max 5%

  // Apply price impact
  const finalAmountOut = amountOut * (1 - priceImpact / 100);

  return {
    amountOut: finalAmountOut,
    fee,
    priceImpact,
    rate: finalAmountOut / amountIn,
    priceIn: priceIn.price,
    priceOut: priceOut.price,
  };
}

/**
 * Start background price updates
 */
export function startPriceUpdates(symbols: string[], intervalMs: number = 30000): () => void {
  const update = () => getPrices(symbols);

  // Initial fetch
  update();

  // Set up interval
  const intervalId = setInterval(update, intervalMs);

  // Return cleanup function
  return () => clearInterval(intervalId);
}

/**
 * Format price for display
 */
export function formatPrice(price: number, decimals: number = 2): string {
  if (price >= 1000) {
    return price.toLocaleString("en-US", { maximumFractionDigits: 2 });
  } else if (price >= 1) {
    return price.toFixed(decimals);
  } else {
    return price.toFixed(6);
  }
}

/**
 * Format price change
 */
export function formatPriceChange(change: number): { text: string; color: string } {
  const formatted = change.toFixed(2);
  if (change > 0) {
    return { text: `+${formatted}%`, color: "text-green-400" };
  } else if (change < 0) {
    return { text: `${formatted}%`, color: "text-red-400" };
  }
  return { text: "0.00%", color: "text-gray-400" };
}

