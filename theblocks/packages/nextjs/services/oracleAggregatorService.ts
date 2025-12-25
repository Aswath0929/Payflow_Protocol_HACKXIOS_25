"use client";

/**
 * @title Oracle Aggregator Service
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Real-time price aggregation from Chainlink and Pyth oracles
 * @dev Fetches live prices from multiple oracle sources with fallback
 */

import { createPublicClient, http, formatUnits } from "viem";
import { sepolia } from "viem/chains";

// ========================================
// TYPES
// ========================================

export interface OraclePrice {
  price: number;
  timestamp: number;
  source: "chainlink" | "pyth" | "aggregated";
  confidence: number; // 0-100
  pair: string;
}

export interface AggregatedPrice {
  chainlinkPrice: number | null;
  pythPrice: number | null;
  aggregatedPrice: number;
  deviation: number; // % difference between sources
  timestamp: number;
  confidence: number;
  pair: string;
  sources: number; // How many sources responded
}

export interface OracleHealth {
  chainlink: {
    healthy: boolean;
    lastUpdate: number;
    consecutiveFailures: number;
  };
  pyth: {
    healthy: boolean;
    lastUpdate: number;
    consecutiveFailures: number;
  };
}

// ========================================
// CHAINLINK SEPOLIA FEEDS
// ========================================

export const CHAINLINK_FEEDS: Record<string, { address: `0x${string}`; decimals: number }> = {
  "ETH/USD": { address: "0x694AA1769357215DE4FAC081bf1f309aDC325306", decimals: 8 },
  "BTC/USD": { address: "0x1b44F3514812d835EB1BDB0acB33d3fA3351Ee43", decimals: 8 },
  "LINK/USD": { address: "0xc59E3633BAAC79493d908e63626716e204A45EdF", decimals: 8 },
  "DAI/USD": { address: "0x14866185B1962B63C3Ea9E03Bc1da838bab34C19", decimals: 8 },
  "USDC/USD": { address: "0xA2F78ab2355fe2f984D808B5CeE7FD0A93D5270E", decimals: 8 },
  "EUR/USD": { address: "0x1a81afB8146aeFfCFc5E50e8479e826E7D55b910", decimals: 8 },
  "GBP/USD": { address: "0x91FAB41F5f3bE955963a986366edAcff1aaeaa83", decimals: 8 },
  "JPY/USD": { address: "0x8A6af2B75F23831ADc973ce6288e5329F63D86c6", decimals: 8 },
};

// Chainlink AggregatorV3 ABI (minimal)
const CHAINLINK_ABI = [
  {
    inputs: [],
    name: "latestRoundData",
    outputs: [
      { name: "roundId", type: "uint80" },
      { name: "answer", type: "int256" },
      { name: "startedAt", type: "uint256" },
      { name: "updatedAt", type: "uint256" },
      { name: "answeredInRound", type: "uint80" },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [],
    name: "decimals",
    outputs: [{ name: "", type: "uint8" }],
    stateMutability: "view",
    type: "function",
  },
] as const;

// ========================================
// PYTH NETWORK FEEDS
// ========================================

const PYTH_CONTRACT = "0xDd24F84d36BF92C65F92307595335bdFab5Bbd21";

export const PYTH_FEED_IDS: Record<string, string> = {
  "ETH/USD": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
  "BTC/USD": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
  "LINK/USD": "0x8ac0c70fff57e9aefdf5edf44b51d62c2d433653cbb2cf5cc06bb115af04d221",
  "USDC/USD": "0xeaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a",
  "DAI/USD": "0xb0948a5e5313200c632b51bb5ca32f6de0d36e9950a942d19751e833f70dabfd",
  "EUR/USD": "0xa995d00bb36a63cef7fd2c287dc105fc8f3d93779f062f09551b0af3e81ec30b",
  "GBP/USD": "0x84c2dde9633d93d1bcad84e244848f8d1f9d5bfb4b4dbcf3a7f8b5b3e1b4f0e4",
  "SOL/USD": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
  "AVAX/USD": "0x93da3352f9f1d105fdfe4971cfa80e9dd777bfc5d0f683ebb6e1294b92137bb7",
  "MATIC/USD": "0x5de33440f6c205f2fb7c1ed88e51adc3e64ad89d0b3d4f10a18bd79b5f4abfae",
};

// Pyth ABI (minimal)
const PYTH_ABI = [
  {
    inputs: [{ name: "id", type: "bytes32" }],
    name: "getPriceUnsafe",
    outputs: [
      {
        components: [
          { name: "price", type: "int64" },
          { name: "conf", type: "uint64" },
          { name: "expo", type: "int32" },
          { name: "publishTime", type: "uint256" },
        ],
        name: "price",
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "id", type: "bytes32" }],
    name: "getPriceNoOlderThan",
    outputs: [
      {
        components: [
          { name: "price", type: "int64" },
          { name: "conf", type: "uint64" },
          { name: "expo", type: "int32" },
          { name: "publishTime", type: "uint256" },
        ],
        name: "price",
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
] as const;

// ========================================
// VIEM CLIENT
// ========================================

const publicClient = createPublicClient({
  chain: sepolia,
  transport: http("https://eth-sepolia.g.alchemy.com/v2/demo"),
});

// ========================================
// ORACLE HEALTH TRACKING
// ========================================

const oracleHealth: OracleHealth = {
  chainlink: { healthy: true, lastUpdate: 0, consecutiveFailures: 0 },
  pyth: { healthy: true, lastUpdate: 0, consecutiveFailures: 0 },
};

// ========================================
// PRICE FETCHING FUNCTIONS
// ========================================

/**
 * Fetch price from Chainlink oracle
 */
export async function fetchChainlinkPrice(pair: string): Promise<OraclePrice | null> {
  const feed = CHAINLINK_FEEDS[pair];
  if (!feed) return null;

  try {
    const result = await publicClient.readContract({
      address: feed.address,
      abi: CHAINLINK_ABI,
      functionName: "latestRoundData",
    });

    const [, answer, , updatedAt] = result;
    const price = Number(formatUnits(answer, feed.decimals));
    const timestamp = Number(updatedAt) * 1000;

    // Check staleness (1 hour threshold)
    const isStale = Date.now() - timestamp > 3600000;

    // Update health
    oracleHealth.chainlink.healthy = !isStale;
    oracleHealth.chainlink.lastUpdate = timestamp;
    oracleHealth.chainlink.consecutiveFailures = 0;

    return {
      price,
      timestamp,
      source: "chainlink",
      confidence: isStale ? 50 : 95,
      pair,
    };
  } catch (error) {
    console.warn(`Chainlink fetch failed for ${pair}:`, error);
    oracleHealth.chainlink.consecutiveFailures++;
    if (oracleHealth.chainlink.consecutiveFailures >= 3) {
      oracleHealth.chainlink.healthy = false;
    }
    return null;
  }
}

/**
 * Fetch price from Pyth oracle
 */
export async function fetchPythPrice(pair: string): Promise<OraclePrice | null> {
  const feedId = PYTH_FEED_IDS[pair];
  if (!feedId) return null;

  try {
    const result = await publicClient.readContract({
      address: PYTH_CONTRACT,
      abi: PYTH_ABI,
      functionName: "getPriceUnsafe",
      args: [feedId as `0x${string}`],
    });

    const { price: priceRaw, expo, publishTime, conf } = result;
    const price = Number(priceRaw) * Math.pow(10, Number(expo));
    const timestamp = Number(publishTime) * 1000;
    const confidence = Math.min(100, 100 - Number(conf) / Number(priceRaw) * 100);

    // Check staleness (5 minutes for Pyth - it updates more frequently)
    const isStale = Date.now() - timestamp > 300000;

    // Update health
    oracleHealth.pyth.healthy = !isStale;
    oracleHealth.pyth.lastUpdate = timestamp;
    oracleHealth.pyth.consecutiveFailures = 0;

    return {
      price: Math.abs(price), // Ensure positive
      timestamp,
      source: "pyth",
      confidence: isStale ? 40 : Math.max(80, confidence),
      pair,
    };
  } catch (error) {
    console.warn(`Pyth fetch failed for ${pair}:`, error);
    oracleHealth.pyth.consecutiveFailures++;
    if (oracleHealth.pyth.consecutiveFailures >= 3) {
      oracleHealth.pyth.healthy = false;
    }
    return null;
  }
}

/**
 * Aggregate prices from both oracles
 */
export async function getAggregatedPrice(pair: string): Promise<AggregatedPrice> {
  const [chainlinkResult, pythResult] = await Promise.all([
    fetchChainlinkPrice(pair),
    fetchPythPrice(pair),
  ]);

  const chainlinkPrice = chainlinkResult?.price ?? null;
  const pythPrice = pythResult?.price ?? null;

  let aggregatedPrice: number;
  let deviation = 0;
  let confidence = 0;
  let sources = 0;

  if (chainlinkPrice !== null && pythPrice !== null) {
    // Both sources available - use weighted average (Chainlink weighted higher)
    aggregatedPrice = chainlinkPrice * 0.6 + pythPrice * 0.4;
    deviation = Math.abs(chainlinkPrice - pythPrice) / aggregatedPrice * 100;
    confidence = deviation < 1 ? 98 : deviation < 2 ? 90 : deviation < 5 ? 75 : 50;
    sources = 2;
  } else if (chainlinkPrice !== null) {
    // Only Chainlink available
    aggregatedPrice = chainlinkPrice;
    confidence = chainlinkResult?.confidence ?? 85;
    sources = 1;
  } else if (pythPrice !== null) {
    // Only Pyth available
    aggregatedPrice = pythPrice;
    confidence = pythResult?.confidence ?? 80;
    sources = 1;
  } else {
    // No sources - return fallback
    aggregatedPrice = getFallbackPrice(pair);
    confidence = 20;
    sources = 0;
  }

  return {
    chainlinkPrice,
    pythPrice,
    aggregatedPrice,
    deviation,
    timestamp: Date.now(),
    confidence,
    pair,
    sources,
  };
}

/**
 * Get multiple aggregated prices at once
 */
export async function getMultipleAggregatedPrices(pairs: string[]): Promise<Map<string, AggregatedPrice>> {
  const results = await Promise.all(pairs.map(pair => getAggregatedPrice(pair)));
  const priceMap = new Map<string, AggregatedPrice>();
  
  results.forEach((result, index) => {
    priceMap.set(pairs[index], result);
  });
  
  return priceMap;
}

/**
 * Fallback prices when oracles are unavailable
 */
function getFallbackPrice(pair: string): number {
  const fallbacks: Record<string, number> = {
    "ETH/USD": 2500,
    "BTC/USD": 45000,
    "LINK/USD": 15,
    "DAI/USD": 1,
    "USDC/USD": 1,
    "EUR/USD": 1.08,
    "GBP/USD": 1.27,
    "JPY/USD": 0.0067,
    "SOL/USD": 100,
    "AVAX/USD": 35,
    "MATIC/USD": 0.85,
  };
  return fallbacks[pair] ?? 1;
}

/**
 * Get oracle health status
 */
export function getOracleHealth(): OracleHealth {
  return { ...oracleHealth };
}

/**
 * Get all available pairs
 */
export function getAvailablePairs(): string[] {
  const chainlinkPairs = Object.keys(CHAINLINK_FEEDS);
  const pythPairs = Object.keys(PYTH_FEED_IDS);
  return [...new Set([...chainlinkPairs, ...pythPairs])];
}

/**
 * Check if a pair has Chainlink support
 */
export function hasChainlinkSupport(pair: string): boolean {
  return pair in CHAINLINK_FEEDS;
}

/**
 * Check if a pair has Pyth support
 */
export function hasPythSupport(pair: string): boolean {
  return pair in PYTH_FEED_IDS;
}

// ========================================
// PRICE SUBSCRIPTION (Real-time updates)
// ========================================

type PriceCallback = (prices: Map<string, AggregatedPrice>) => void;
const subscribers: Set<PriceCallback> = new Set();
let updateInterval: NodeJS.Timeout | null = null;

/**
 * Subscribe to real-time price updates
 */
export function subscribeToPriceUpdates(
  pairs: string[],
  callback: PriceCallback,
  intervalMs: number = 10000
): () => void {
  subscribers.add(callback);

  // Start updates if not already running
  if (!updateInterval) {
    const updatePrices = async () => {
      const prices = await getMultipleAggregatedPrices(pairs);
      subscribers.forEach(cb => cb(prices));
    };

    // Initial fetch
    updatePrices();

    // Set up interval
    updateInterval = setInterval(updatePrices, intervalMs);
  }

  // Return unsubscribe function
  return () => {
    subscribers.delete(callback);
    if (subscribers.size === 0 && updateInterval) {
      clearInterval(updateInterval);
      updateInterval = null;
    }
  };
}

/**
 * Format price for display
 */
export function formatOraclePrice(price: number, pair: string): string {
  if (pair.includes("JPY")) {
    return price.toFixed(6);
  } else if (price >= 1000) {
    return price.toLocaleString("en-US", { maximumFractionDigits: 2 });
  } else if (price >= 1) {
    return price.toFixed(4);
  } else {
    return price.toFixed(6);
  }
}

/**
 * Get confidence color for UI
 */
export function getConfidenceColor(confidence: number): string {
  if (confidence >= 90) return "text-green-400";
  if (confidence >= 70) return "text-yellow-400";
  if (confidence >= 50) return "text-orange-400";
  return "text-red-400";
}

/**
 * Get source badge for UI
 */
export function getSourceBadge(sources: number): { text: string; color: string } {
  if (sources === 2) return { text: "Dual Oracle", color: "bg-green-500/20 text-green-400" };
  if (sources === 1) return { text: "Single Oracle", color: "bg-yellow-500/20 text-yellow-400" };
  return { text: "Fallback", color: "bg-red-500/20 text-red-400" };
}

