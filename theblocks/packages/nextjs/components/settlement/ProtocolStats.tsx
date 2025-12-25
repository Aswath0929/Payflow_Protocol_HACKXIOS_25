"use client";

import { useCallback, useEffect, useState } from "react";
import { useScaffoldReadContract } from "~~/hooks/scaffold-eth";
import { useContractEvents } from "~~/hooks/settlement";

/**
 * @title ProtocolStats
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Real-time protocol statistics dashboard
 *
 * FEATURES:
 * - Total payments & volume
 * - Queue status
 * - Success/failure rates
 * - Live event feed
 * - Performance metrics
 *
 * NOTE: Uses actual PayFlowCore functions
 */

export const ProtocolStats = () => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [refreshKey, setRefreshKey] = useState(0);

  const { events, stats, getEventColor, getEventIcon, getRecentEvents } = useContractEvents();

  // Protocol reads - using actual PayFlowCore functions
  const { data: totalPaymentsCreated, refetch: refetchCreated } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsCreated",
  });

  const { data: totalPaymentsExecuted, refetch: refetchExecuted } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsExecuted",
  });

  const { data: averageSettlementTime, refetch: refetchAvgTime } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "averageSettlementTime",
  });

  const { data: isPaused } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "paused",
  });

  // Demo values for features not yet in contract
  const totalVolume = 125.5; // ETH - demo value
  const pendingInQueue = 3; // Demo value
  const currentQueueHead = 1; // Demo value

  // Calculated stats
  const totalPayments = totalPaymentsCreated ? Number(totalPaymentsCreated) : 0;
  const executedPayments = totalPaymentsExecuted ? Number(totalPaymentsExecuted) : 0;
  const avgTime = averageSettlementTime ? Number(averageSettlementTime) : 0;

  // Success rate calculation
  const getSuccessRate = () => {
    if (totalPayments === 0) return 0;
    return Math.round((executedPayments * 100) / totalPayments);
  };

  // Refresh all data
  const handleRefresh = useCallback(() => {
    refetchCreated();
    refetchExecuted();
    refetchAvgTime();
    setRefreshKey(prev => prev + 1);
  }, [refetchCreated, refetchExecuted, refetchAvgTime]);

  // Auto-refresh every 3 seconds for real-time updates
  useEffect(() => {
    const interval = setInterval(handleRefresh, 3000);
    return () => clearInterval(interval);
  }, [handleRefresh]);

  const recentEvents = getRecentEvents(8);

  return (
    <div className="bg-base-100 rounded-3xl shadow-lg p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold">📊 Protocol Statistics</h2>
          <span className={`badge ${isPaused ? "badge-error" : "badge-success"} badge-sm`}>
            {isPaused ? "🔴 Paused" : "🟢 Active"}
          </span>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={handleRefresh}>
          🔄 Refresh
        </button>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-primary/20 to-primary/5 rounded-2xl p-4">
          <p className="text-xs opacity-60 mb-1">Total Payments</p>
          <p className="text-3xl font-bold">{totalPayments}</p>
          <p className="text-xs opacity-60 mt-1">Created</p>
        </div>

        <div className="bg-gradient-to-br from-secondary/20 to-secondary/5 rounded-2xl p-4">
          <p className="text-xs opacity-60 mb-1">Total Volume</p>
          <p className="text-3xl font-bold">{totalVolume.toFixed(2)}</p>
          <p className="text-xs opacity-60 mt-1">ETH Settled</p>
        </div>

        <div className="bg-gradient-to-br from-accent/20 to-accent/5 rounded-2xl p-4">
          <p className="text-xs opacity-60 mb-1">Executed</p>
          <p className="text-3xl font-bold">{executedPayments}</p>
          <p className="text-xs opacity-60 mt-1">Completed</p>
        </div>

        <div className="bg-gradient-to-br from-success/20 to-success/5 rounded-2xl p-4">
          <p className="text-xs opacity-60 mb-1">Success Rate</p>
          <p className="text-3xl font-bold">{getSuccessRate()}%</p>
          <p className="text-xs opacity-60 mt-1">Finalized</p>
        </div>
      </div>

      {/* Detailed Stats */}
      <div className="bg-base-200 rounded-2xl p-4 mb-6">
        <h3 className="font-semibold mb-3">📈 Breakdown</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-success">{executedPayments}</p>
            <p className="text-xs opacity-60">Executed</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-warning">{pendingInQueue}</p>
            <p className="text-xs opacity-60">Pending</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-info">{avgTime}s</p>
            <p className="text-xs opacity-60">Avg Time</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-accent">{currentQueueHead}</p>
            <p className="text-xs opacity-60">Queue Head</p>
          </div>
        </div>
      </div>

      {/* Live Event Feed */}
      <div className="bg-base-200 rounded-2xl p-4">
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-semibold">📡 Live Events</h3>
          <span className="badge badge-ghost badge-sm">{events.length} total</span>
        </div>

        {recentEvents.length === 0 ? (
          <div className="text-center py-6 opacity-60">
            <span className="text-2xl">📭</span>
            <p className="text-sm mt-2">No events yet. Waiting for activity...</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {recentEvents.map(event => (
              <div key={event.id} className="flex items-start gap-3 p-2 bg-base-100 rounded-lg text-sm">
                <span className="text-lg">{getEventIcon(event.eventName)}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`badge ${getEventColor(event.eventName)} badge-xs`}>{event.eventName}</span>
                    <span className="text-xs opacity-50">Block #{event.blockNumber.toString()}</span>
                  </div>
                  <p className="text-xs opacity-70 truncate mt-1">{event.formatted}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Session Stats */}
      <div className="mt-6 grid grid-cols-4 gap-3 text-center">
        <div className="bg-base-200 rounded-xl p-3">
          <p className="text-lg font-bold text-primary">{stats.totalSettlementsCreated}</p>
          <p className="text-[10px] opacity-60">Session Created</p>
        </div>
        <div className="bg-base-200 rounded-xl p-3">
          <p className="text-lg font-bold text-secondary">{stats.totalDeposits}</p>
          <p className="text-[10px] opacity-60">Session Deposits</p>
        </div>
        <div className="bg-base-200 rounded-xl p-3">
          <p className="text-lg font-bold text-accent">{stats.totalExecutions}</p>
          <p className="text-[10px] opacity-60">Session Executions</p>
        </div>
        <div className="bg-base-200 rounded-xl p-3">
          <p className="text-lg font-bold text-warning">{stats.totalDisputes}</p>
          <p className="text-[10px] opacity-60">Session Disputes</p>
        </div>
      </div>
    </div>
  );
};

export default ProtocolStats;






