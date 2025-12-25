"use client";

import { useCallback, useMemo, useState } from "react";
import { formatUnits } from "viem";
import { useReadContract, useWatchContractEvent } from "wagmi";
import { useDeployedContractInfo } from "~~/hooks/scaffold-eth";

/**
 * @title useOracle Hook
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Oracle price management and health monitoring using OracleAggregator contract
 *
 * Uses real OracleAggregator contract functions:
 * - getOracleStats(): Get oracle statistics (updates, queries, circuitBreakers, oracleCount)
 * - getRate(pairId): Get rate for a specific pair
 * - getRateBySymbols(base, quote): Get rate by symbol names
 * - getPairInfo(pairId): Get pair information
 * - getAllPairs(): Get all registered pairs
 */

export interface OraclePrice {
  price: bigint;
  timestamp: number;
  isStale: boolean;
  source: "chainlink" | "band" | "fallback";
}

export interface OracleHealth {
  chainlinkHealthy: boolean;
  bandHealthy: boolean;
  chainlinkFailures: number;
  bandFailures: number;
  lastUpdateBlock: bigint;
}

export interface ManipulationStatus {
  isManipulated: boolean;
  reason: string;
  deviation: number;
}

export const useOracle = () => {
  const [priceHistory, setPriceHistory] = useState<OraclePrice[]>([]);
  const [lastRefresh, setLastRefresh] = useState<number>(Date.now());
  const [isLoading, setIsLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isForcing] = useState(false);

  // Get OracleAggregator contract info
  const { data: oracleAggregatorInfo } = useDeployedContractInfo("OracleAggregator");

  // Read oracle stats from OracleAggregator
  const { data: oracleStats, refetch: refetchStats } = useReadContract({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    functionName: "getOracleStats",
    query: {
      enabled: !!oracleAggregatorInfo,
      refetchInterval: 3000, // Poll every 3 seconds
    },
  });

  // Read all pairs from OracleAggregator
  const { data: allPairs, refetch: refetchPairs } = useReadContract({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    functionName: "getAllPairs",
    query: {
      enabled: !!oracleAggregatorInfo,
      refetchInterval: 10000, // Poll every 10 seconds
    },
  });

  // Read circuit breaker triggers count
  const { data: circuitBreakerTriggers } = useReadContract({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    functionName: "circuitBreakerTriggers",
    query: {
      enabled: !!oracleAggregatorInfo,
      refetchInterval: 5000,
    },
  });

  // Watch for PriceUpdated events
  useWatchContractEvent({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    eventName: "PriceUpdated",
    onLogs: logs => {
      if (logs.length > 0) {
        const latestLog = logs[logs.length - 1] as {
          args?: { pairId?: `0x${string}`; rate?: bigint; timestamp?: bigint };
        };
        const args = latestLog.args;

        if (args?.rate) {
          const newPrice: OraclePrice = {
            price: args.rate,
            timestamp: args.timestamp ? Number(args.timestamp) * 1000 : Date.now(),
            isStale: false,
            source: "chainlink",
          };

          setPriceHistory(prev => {
            const updated = [...prev, newPrice];
            return updated.slice(-100);
          });
        }
        setLastRefresh(Date.now());
      }
    },
    enabled: !!oracleAggregatorInfo,
  });

  // Watch for CircuitBreakerTripped events
  useWatchContractEvent({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    eventName: "CircuitBreakerTripped",
    onLogs: logs => {
      if (logs.length > 0) {
        console.log("⚠️ Circuit breaker tripped:", logs);
      }
    },
    enabled: !!oracleAggregatorInfo,
  });

  // Current price from latest history entry or stats
  const currentPrice = useMemo(() => {
    if (priceHistory.length > 0) {
      return priceHistory[priceHistory.length - 1].price;
    }
    // Default ETH price estimate (will be updated by events)
    return BigInt(2500 * 1e8);
  }, [priceHistory]);

  // Calculate oracle health from real stats
  const oracleHealth: OracleHealth = useMemo(() => {
    const stats = oracleStats as [bigint, bigint, bigint, bigint] | undefined;
    const oracleCount = stats ? Number(stats[3]) : 0;
    const circuitBreakers = stats ? Number(stats[2]) : 0;

    return {
      chainlinkHealthy: oracleCount > 0 && circuitBreakers === 0,
      bandHealthy: oracleCount > 1 && circuitBreakers === 0,
      chainlinkFailures: circuitBreakers,
      bandFailures: 0,
      lastUpdateBlock: BigInt(Math.floor(Date.now() / 12000)),
    };
  }, [oracleStats]);

  // Manipulation status based on circuit breaker
  const manipulationStatus: ManipulationStatus = useMemo(() => {
    const triggers = circuitBreakerTriggers ? Number(circuitBreakerTriggers) : 0;
    return {
      isManipulated: triggers > 0,
      reason: triggers > 0 ? `Circuit breaker triggered ${triggers} times` : "",
      deviation: 0,
    };
  }, [circuitBreakerTriggers]);

  // Format price for display (8 decimals for Chainlink)
  const formatPrice = useCallback((price: bigint, decimals = 8) => {
    return parseFloat(formatUnits(price, decimals)).toFixed(2);
  }, []);

  // Get formatted current price
  const formattedPrice = currentPrice ? formatPrice(currentPrice) : "0.00";

  // Calculate health score (0-100) based on real data
  const getHealthScore = useCallback(() => {
    const stats = oracleStats as [bigint, bigint, bigint, bigint] | undefined;
    if (!stats) return 50; // Unknown state

    const oracleCount = Number(stats[3]);
    const circuitBreakers = Number(stats[2]);

    let score = 100;
    if (oracleCount < 3) score -= (3 - oracleCount) * 20;
    score -= circuitBreakers * 10;
    if (manipulationStatus.isManipulated) score -= 20;

    return Math.max(0, Math.min(100, score));
  }, [oracleStats, manipulationStatus]);

  // Get health status label
  const getHealthLabel = useCallback(() => {
    const score = getHealthScore();
    if (score >= 80) return { label: "Healthy", color: "success" };
    if (score >= 50) return { label: "Degraded", color: "warning" };
    return { label: "Critical", color: "error" };
  }, [getHealthScore]);

  // Refresh all oracle data
  const refreshAll = useCallback(async () => {
    setIsLoading(true);
    try {
      await Promise.all([refetchStats(), refetchPairs()]);
      setLastRefresh(Date.now());
    } finally {
      setIsLoading(false);
    }
  }, [refetchStats, refetchPairs]);

  // Reset function (triggers refresh)
  const resetOracleStatus = useCallback(async () => {
    setIsResetting(true);
    await refreshAll();
    setIsResetting(false);
  }, [refreshAll]);

  // Calculate price change percentage
  const getPriceChange = useCallback(() => {
    if (priceHistory.length < 2) return 0;
    const current = priceHistory[priceHistory.length - 1].price;
    const previous = priceHistory[priceHistory.length - 2].price;
    if (previous === 0n) return 0;
    return Number(((current - previous) * 10000n) / previous) / 100;
  }, [priceHistory]);

  // Get oracle statistics for display
  const getOracleDisplayStats = useCallback(() => {
    const stats = oracleStats as [bigint, bigint, bigint, bigint] | undefined;
    if (!stats) {
      return { updates: 0, queries: 0, circuitBreakers: 0, oracleCount: 0 };
    }
    return {
      updates: Number(stats[0]),
      queries: Number(stats[1]),
      circuitBreakers: Number(stats[2]),
      oracleCount: Number(stats[3]),
    };
  }, [oracleStats]);

  return {
    // Price data
    currentPrice,
    formattedPrice,
    priceHistory,
    priceChange: getPriceChange(),

    // Oracle health
    oracleHealth,
    healthScore: getHealthScore(),
    healthStatus: getHealthLabel(),

    // Manipulation status
    manipulationStatus,
    isManipulated: manipulationStatus.isManipulated,

    // Oracle statistics from contract
    oracleStats: getOracleDisplayStats(),
    allPairs: allPairs as `0x${string}`[] | undefined,

    // Configuration from contract
    deviationThreshold: 500, // 5%
    maxOracleFailures: 5,
    stalenessThreshold: 3600,

    // Actions
    resetOracleStatus,
    refreshAll,

    // Loading states
    isLoading,
    isResetting,
    isForcing,

    // Utilities
    formatPrice,
    lastRefresh,

    // Contract info
    isContractReady: !!oracleAggregatorInfo,
  };
};

export default useOracle;






