"use client";

import { useCallback, useRef, useState } from "react";
import { formatEther } from "viem";
import { useScaffoldEventHistory, useScaffoldWatchContractEvent } from "~~/hooks/scaffold-eth";

/**
 * @title useContractEvents Hook
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Real-time contract event monitoring hook
 *
 * FEATURES:
 * - Live event subscription
 * - Event history retrieval
 * - Event filtering
 * - Event aggregation for statistics
 */

export interface ContractEvent {
  id: string;
  eventName: string;
  blockNumber: bigint;
  transactionHash: string;
  timestamp: number;
  args: Record<string, unknown>;
  formatted: string;
}

export interface EventStats {
  totalSettlementsCreated: number;
  totalDeposits: number;
  totalExecutions: number;
  totalDisputes: number;
  totalVolume: bigint;
  last24hEvents: number;
}

const MAX_EVENTS = 100;

export const useContractEvents = () => {
  const [events, setEvents] = useState<ContractEvent[]>([]);
  const [isSubscribed, setIsSubscribed] = useState(true);
  const eventIdCounter = useRef(0);

  // Track stats
  const [stats, setStats] = useState<EventStats>({
    totalSettlementsCreated: 0,
    totalDeposits: 0,
    totalExecutions: 0,
    totalDisputes: 0,
    totalVolume: 0n,
    last24hEvents: 0,
  });

  // Format event for display (before addEvent since it's used in addEvent)
  const formatEventLog = (eventName: string, args: Record<string, unknown>): string => {
    switch (eventName) {
      case "PaymentCreated":
        return `Payment created: ${(args.paymentId as string)?.slice(0, 10) || "N/A"}... Amount: ${formatEther((args.amount as bigint) || 0n)} ETH`;
      case "PaymentExecuted":
        return `Payment executed: ${(args.paymentId as string)?.slice(0, 10) || "N/A"}...`;
      case "PaymentFailed":
        return `Payment failed: ${(args.paymentId as string)?.slice(0, 10) || "N/A"}... Reason: ${args.reason || "Unknown"}`;
      case "PaymentApproved":
        return `Payment approved: ${(args.paymentId as string)?.slice(0, 10) || "N/A"}...`;
      case "PaymentCancelled":
        return `Payment cancelled: ${(args.paymentId as string)?.slice(0, 10) || "N/A"}...`;
      case "ComplianceVerified":
        return `Compliance verified for ${(args.party as string)?.slice(0, 8) || "N/A"}...`;
      case "CrossBorderSettlement":
        return `Cross-border settlement: ${formatEther((args.amount as bigint) || 0n)} ETH`;
      case "Paused":
        return `Protocol paused by ${(args.account as string)?.slice(0, 8) || "N/A"}...`;
      case "Unpaused":
        return `Protocol unpaused by ${(args.account as string)?.slice(0, 8) || "N/A"}...`;
      default:
        return `${eventName}: ${JSON.stringify(args)}`;
    }
  };

  // Update statistics based on event (before addEvent since it's used in addEvent)
  const updateStats = useCallback((eventName: string, args: Record<string, unknown>) => {
    setStats(prev => {
      const updated = { ...prev };
      updated.last24hEvents++;

      switch (eventName) {
        case "PaymentCreated":
          updated.totalSettlementsCreated++;
          updated.totalVolume += (args.amount as bigint) || 0n;
          break;
        case "PaymentExecuted":
          updated.totalExecutions++;
          break;
        case "PaymentFailed":
          updated.totalDisputes++;
          break;
      }

      return updated;
    });
  }, []);

  // Helper to add event
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const addEvent = useCallback(
    (eventName: string, log: { blockNumber?: bigint; transactionHash?: string; args?: any }) => {
      if (!isSubscribed) return;

      const args = (log.args ? { ...log.args } : {}) as Record<string, unknown>;
      const newEvent: ContractEvent = {
        id: `${eventName}-${eventIdCounter.current++}`,
        eventName,
        blockNumber: log.blockNumber || 0n,
        transactionHash: log.transactionHash || "",
        timestamp: Date.now(),
        args,
        formatted: formatEventLog(eventName, args),
      };

      setEvents(prev => {
        const updated = [newEvent, ...prev];
        return updated.slice(0, MAX_EVENTS);
      });

      // Update stats
      updateStats(eventName, args);
    },
    [isSubscribed, updateStats, formatEventLog],
  );

  // Watch PayFlowCore Events - using actual contract events
  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "PaymentCreated",
    onLogs: logs => {
      logs.forEach(log => addEvent("PaymentCreated", log));
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "PaymentExecuted",
    onLogs: logs => {
      logs.forEach(log => addEvent("PaymentExecuted", log));
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "PaymentFailed",
    onLogs: logs => {
      logs.forEach(log => addEvent("PaymentFailed", log));
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "ComplianceVerified",
    onLogs: logs => {
      logs.forEach(log => addEvent("ComplianceVerified", log));
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "Paused",
    onLogs: logs => {
      logs.forEach(log => addEvent("Paused", log));
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "Unpaused",
    onLogs: logs => {
      logs.forEach(log => addEvent("Unpaused", log));
    },
  });

  // Event history (past events)
  const { data: paymentCreatedHistory } = useScaffoldEventHistory({
    contractName: "PayFlowCore",
    eventName: "PaymentCreated",
    fromBlock: 0n,
    blockData: true,
  });

  const { data: paymentExecutedHistory } = useScaffoldEventHistory({
    contractName: "PayFlowCore",
    eventName: "PaymentExecuted",
    fromBlock: 0n,
    blockData: true,
  });

  // Filter functions
  const filterByEventName = useCallback((eventName: string) => events.filter(e => e.eventName === eventName), [events]);

  const filterBySettlement = useCallback(
    (paymentId: string) => events.filter(e => e.args.paymentId === paymentId),
    [events],
  );

  const filterByTimeRange = useCallback(
    (startTime: number, endTime: number = Date.now()) =>
      events.filter(e => e.timestamp >= startTime && e.timestamp <= endTime),
    [events],
  );

  // Get recent events
  const getRecentEvents = useCallback((count: number = 10) => events.slice(0, count), [events]);

  // Clear events
  const clearEvents = useCallback(() => {
    setEvents([]);
    setStats({
      totalSettlementsCreated: 0,
      totalDeposits: 0,
      totalExecutions: 0,
      totalDisputes: 0,
      totalVolume: 0n,
      last24hEvents: 0,
    });
  }, []);

  // Toggle subscription
  const toggleSubscription = useCallback(() => {
    setIsSubscribed(prev => !prev);
  }, []);

  // Get event color based on type
  const getEventColor = useCallback((eventName: string): string => {
    const colors: Record<string, string> = {
      PaymentCreated: "badge-primary",
      PaymentExecuted: "badge-success",
      PaymentFailed: "badge-error",
      PaymentApproved: "badge-info",
      PaymentCancelled: "badge-warning",
      ComplianceVerified: "badge-accent",
      CrossBorderSettlement: "badge-secondary",
      Paused: "badge-error",
      Unpaused: "badge-success",
    };
    return colors[eventName] || "badge-ghost";
  }, []);

  // Get event icon
  const getEventIcon = useCallback((eventName: string): string => {
    const icons: Record<string, string> = {
      PaymentCreated: "📝",
      PaymentExecuted: "✅",
      PaymentFailed: "❌",
      PaymentApproved: "👍",
      PaymentCancelled: "🚫",
      ComplianceVerified: "✓",
      CrossBorderSettlement: "🌍",
      Paused: "⏸️",
      Unpaused: "▶️",
    };
    return icons[eventName] || "📌";
  }, []);

  return {
    // Event data
    events,
    stats,
    eventCount: events.length,

    // Historical data
    paymentCreatedHistory,
    paymentExecutedHistory,

    // Subscription
    isSubscribed,
    toggleSubscription,

    // Filters
    filterByEventName,
    filterBySettlement,
    filterByTimeRange,
    getRecentEvents,

    // Actions
    clearEvents,

    // Utilities
    getEventColor,
    getEventIcon,
  };
};

export default useContractEvents;






