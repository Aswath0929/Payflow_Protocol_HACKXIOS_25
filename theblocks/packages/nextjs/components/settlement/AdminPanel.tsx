"use client";

import { useAccount, useReadContract } from "wagmi";
import { useDeployedContractInfo, useScaffoldReadContract, useScaffoldWriteContract } from "~~/hooks/scaffold-eth";
import { notification } from "~~/utils/scaffold-eth";

/**
 * @title AdminPanel
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Admin controls for protocol management
 *
 * Uses real PayFlowCore contract functions:
 * - paused(): Check if protocol is paused
 * - pause(): Pause the protocol (admin only)
 * - unpause(): Unpause the protocol (admin only)
 * - getProtocolStats(): Get protocol statistics
 * - hasRole(): Check if address has admin role
 */

// Default admin role for PayFlowCore
const DEFAULT_ADMIN_ROLE = "0x0000000000000000000000000000000000000000000000000000000000000000" as const;

export const AdminPanel = () => {
  const { address: connectedAddress } = useAccount();

  // Get contract info
  const { data: payFlowCoreInfo } = useDeployedContractInfo("PayFlowCore");
  const { data: oracleAggregatorInfo } = useDeployedContractInfo("OracleAggregator");

  // Read pause status
  const { data: isPaused, refetch: refetchPaused } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "paused",
  });

  // Read protocol stats from PayFlowCore
  const { data: protocolStats } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "getProtocolStats",
  });

  // Check if connected wallet has admin role
  const { data: isAdminData } = useReadContract({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "hasRole",
    args: connectedAddress ? [DEFAULT_ADMIN_ROLE, connectedAddress] : undefined,
    query: {
      enabled: !!payFlowCoreInfo && !!connectedAddress,
    },
  });
  const isAdmin = isAdminData as boolean | undefined;

  // Read oracle stats
  const { data: oracleStats } = useReadContract({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    functionName: "getOracleStats",
    query: {
      enabled: !!oracleAggregatorInfo,
      refetchInterval: 10000,
    },
  });

  // Read circuit breaker triggers
  const { data: circuitBreakerTriggers } = useReadContract({
    address: oracleAggregatorInfo?.address,
    abi: oracleAggregatorInfo?.abi,
    functionName: "circuitBreakerTriggers",
    query: {
      enabled: !!oracleAggregatorInfo,
      refetchInterval: 5000,
    },
  });

  // Parse protocol stats
  const stats = protocolStats as [bigint, bigint, bigint, bigint] | undefined;
  const totalCreated = stats ? Number(stats[0]) : 0;
  const totalExecuted = stats ? Number(stats[1]) : 0;
  const totalVolume = stats ? stats[2] : 0n;

  // Parse oracle stats
  const oStats = oracleStats as [bigint, bigint, bigint, bigint] | undefined;
  const oracleUpdates = oStats ? Number(oStats[0]) : 0;
  const oracleQueries = oStats ? Number(oStats[1]) : 0;
  const oracleCount = oStats ? Number(oStats[3]) : 0;
  const cbTriggers = circuitBreakerTriggers ? Number(circuitBreakerTriggers) : 0;

  // Contract writes
  const { writeContractAsync: pauseContract } = useScaffoldWriteContract("PayFlowCore");
  const { writeContractAsync: unpauseContract } = useScaffoldWriteContract("PayFlowCore");

  const handlePause = async () => {
    try {
      await pauseContract({ functionName: "pause" });
      notification.success("Protocol paused");
      refetchPaused();
    } catch (error: unknown) {
      notification.error((error as Error)?.message || "Failed to pause");
    }
  };

  const handleUnpause = async () => {
    try {
      await unpauseContract({ functionName: "unpause" });
      notification.success("Protocol unpaused");
      refetchPaused();
    } catch (error: unknown) {
      notification.error((error as Error)?.message || "Failed to unpause");
    }
  };

  return (
    <div className="bg-base-100 rounded-3xl shadow-lg p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2 text-center">🛡️ Admin Panel</h2>
      <p className="text-center text-sm opacity-70 mb-6">Protocol management and monitoring</p>

      {/* Admin Status */}
      <div className={`alert ${isAdmin ? "alert-success" : "alert-warning"} mb-6`}>
        {isAdmin ? (
          <span>✅ You are the admin. Full access granted.</span>
        ) : (
          <span>⚠️ You are not the admin. View-only access.</span>
        )}
      </div>

      {/* System Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="stat bg-base-200 rounded-xl">
          <div className="stat-title">Payments Created</div>
          <div className="stat-value">{totalCreated}</div>
          <div className="stat-desc">Total created</div>
        </div>
        <div className="stat bg-base-200 rounded-xl">
          <div className="stat-title">Payments Executed</div>
          <div className="stat-value">{totalExecuted}</div>
          <div className="stat-desc">Successfully settled</div>
        </div>
        <div className="stat bg-base-200 rounded-xl">
          <div className="stat-title">Total Volume</div>
          <div className="stat-value text-lg">{(Number(totalVolume) / 1e18).toFixed(2)} ETH</div>
          <div className="stat-desc">Processed volume</div>
        </div>
      </div>

      {/* Protocol Status */}
      <div className="bg-base-200 rounded-xl p-4 mb-6">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="font-semibold">Protocol Status</h3>
            <div className="flex items-center gap-2 mt-1">
              <div className={`badge ${isPaused ? "badge-error" : "badge-success"} badge-lg`}>
                {isPaused ? "🔴 PAUSED" : "🟢 ACTIVE"}
              </div>
            </div>
          </div>
          {isAdmin && (
            <div className="space-x-2">
              {isPaused ? (
                <button className="btn btn-success" onClick={handleUnpause}>
                  ▶️ Unpause
                </button>
              ) : (
                <button className="btn btn-error" onClick={handlePause}>
                  ⏸️ Pause
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Oracle Health Dashboard */}
      <div className="bg-base-200 rounded-xl p-4 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-semibold">🔮 Oracle Statistics</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Oracle Updates */}
          <div className="p-4 rounded-lg border border-info bg-info/10">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">📊</span>
              <span className="font-semibold">Price Updates</span>
            </div>
            <p className="text-2xl font-bold">{oracleUpdates.toLocaleString()}</p>
            <p className="text-sm opacity-70">Total price updates received</p>
          </div>

          {/* Oracle Queries */}
          <div className="p-4 rounded-lg border border-info bg-info/10">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🔍</span>
              <span className="font-semibold">Oracle Queries</span>
            </div>
            <p className="text-2xl font-bold">{oracleQueries.toLocaleString()}</p>
            <p className="text-sm opacity-70">Total rate queries</p>
          </div>

          {/* Active Oracles */}
          <div className="p-4 rounded-lg border border-success bg-success/10">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">🔗</span>
              <span className="font-semibold">Active Oracles</span>
            </div>
            <p className="text-2xl font-bold">{oracleCount}</p>
            <p className="text-sm opacity-70">Registered oracles</p>
          </div>

          {/* Circuit Breaker */}
          <div
            className={`p-4 rounded-lg border ${cbTriggers > 0 ? "border-warning bg-warning/10" : "border-success bg-success/10"}`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">⚡</span>
              <span className="font-semibold">Circuit Breaker</span>
            </div>
            <p className="text-2xl font-bold">{cbTriggers}</p>
            <p className="text-sm opacity-70">Times triggered</p>
          </div>
        </div>
      </div>

      {/* Invariants Status */}
      <div className="bg-base-200 rounded-xl p-4">
        <h3 className="font-semibold mb-4">📋 Active Invariants</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="badge badge-success">✓</span>
            <span>Conservation of Value</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge badge-success">✓</span>
            <span>No Double Settlement</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge badge-success">✓</span>
            <span>Oracle Freshness (60s max staleness)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge badge-success">✓</span>
            <span>Timeout & Liveness</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="badge badge-success">✓</span>
            <span>Partial Finality Continuity</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminPanel;
