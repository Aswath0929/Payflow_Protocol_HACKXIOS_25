"use client";

/**
 * @title Trade Store
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Shared state management for trades across components
 * @dev Uses localStorage for persistence across sessions
 */

export interface Trade {
  id: string;
  type: "swap" | "add_liquidity" | "remove_liquidity";
  tokenIn: string;
  tokenOut: string;
  amountIn: string;
  amountOut: string;
  priceIn: number;
  priceOut: number;
  fee: string;
  priceImpact: string;
  status: "completed" | "pending" | "failed";
  timestamp: number;
  txHash: string;
  userAddress: string;
}

const STORAGE_KEY = "blocks_trade_history";

// Event emitter for cross-component updates
const tradeListeners: Set<() => void> = new Set();

/**
 * Subscribe to trade updates
 */
export function subscribeToTrades(callback: () => void): () => void {
  tradeListeners.add(callback);
  return () => tradeListeners.delete(callback);
}

/**
 * Notify all listeners of trade updates
 */
function notifyListeners() {
  tradeListeners.forEach(callback => callback());
}

/**
 * Get all trades from localStorage
 */
export function getTrades(userAddress?: string): Trade[] {
  if (typeof window === "undefined") return [];

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const trades: Trade[] = stored ? JSON.parse(stored) : [];

    // Filter by user if address provided
    if (userAddress) {
      return trades.filter(t => t.userAddress.toLowerCase() === userAddress.toLowerCase());
    }

    return trades;
  } catch (error) {
    console.error("Error reading trades:", error);
    return [];
  }
}

/**
 * Add a new trade to history
 */
export function addTrade(trade: Omit<Trade, "id" | "timestamp">): Trade {
  const newTrade: Trade = {
    ...trade,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
    timestamp: Date.now(),
  };

  try {
    const trades = getTrades();
    trades.unshift(newTrade); // Add to beginning

    // Keep only last 100 trades per user
    const userTrades = trades.filter(t => t.userAddress === trade.userAddress).slice(0, 100);
    const otherTrades = trades.filter(t => t.userAddress !== trade.userAddress);

    localStorage.setItem(STORAGE_KEY, JSON.stringify([...userTrades, ...otherTrades]));
    notifyListeners();
  } catch (error) {
    console.error("Error saving trade:", error);
  }

  return newTrade;
}

/**
 * Update trade status (e.g., pending -> completed)
 */
export function updateTradeStatus(tradeId: string, status: Trade["status"], txHash?: string): void {
  try {
    const trades = getTrades();
    const index = trades.findIndex(t => t.id === tradeId);

    if (index !== -1) {
      trades[index].status = status;
      if (txHash) trades[index].txHash = txHash;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trades));
      notifyListeners();
    }
  } catch (error) {
    console.error("Error updating trade:", error);
  }
}

/**
 * Calculate trade statistics for a user
 */
export function getTradeStats(userAddress: string): {
  totalTrades: number;
  totalVolume: number;
  totalFees: number;
  avgTradeSize: number;
} {
  const trades = getTrades(userAddress).filter(t => t.status === "completed");

  if (trades.length === 0) {
    return {
      totalTrades: 0,
      totalVolume: 0,
      totalFees: 0,
      avgTradeSize: 0,
    };
  }

  const totalVolume = trades.reduce((sum, t) => {
    const valueIn = parseFloat(t.amountIn) * t.priceIn;
    return sum + valueIn;
  }, 0);

  const totalFees = trades.reduce((sum, t) => {
    return sum + parseFloat(t.fee) * t.priceOut;
  }, 0);

  return {
    totalTrades: trades.length,
    totalVolume,
    totalFees,
    avgTradeSize: totalVolume / trades.length,
  };
}

/**
 * Clear all trades for a user
 */
export function clearTrades(userAddress: string): void {
  try {
    const trades = getTrades();
    const filtered = trades.filter(t => t.userAddress.toLowerCase() !== userAddress.toLowerCase());
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    notifyListeners();
  } catch (error) {
    console.error("Error clearing trades:", error);
  }
}

/**
 * Export trades as JSON for backup
 */
export function exportTrades(userAddress: string): string {
  const trades = getTrades(userAddress);
  return JSON.stringify(trades, null, 2);
}

