"use client";

import { useMemo, useState } from "react";
import { formatEther } from "viem";
import { useAccount, useReadContract } from "wagmi";
import { useDeployedContractInfo, useScaffoldReadContract } from "~~/hooks/scaffold-eth";
import { notification } from "~~/utils/scaffold-eth";

/**
 * @title SettlementMonitor
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Real-time monitoring dashboard for payment status
 *
 * Uses real PayFlowCore contract functions:
 * - totalPaymentsCreated(): Get total created payments
 * - totalPaymentsExecuted(): Get total executed payments
 * - getUserPayments(address): Get user's sent and received payment IDs
 * - getPayment(paymentId): Get payment details
 */

// Payment status enum from PayFlowCore
const STATUS_NAMES = ["PENDING", "APPROVED", "EXECUTED", "CANCELLED", "FAILED"];
const STATUS_COLORS: Record<number, string> = {
  0: "badge-warning", // PENDING
  1: "badge-info", // APPROVED
  2: "badge-success", // EXECUTED
  3: "badge-ghost", // CANCELLED
  4: "badge-error", // FAILED
};

const STATUS_ICONS: Record<number, string> = {
  0: "⏳", // PENDING
  1: "✓", // APPROVED
  2: "✅", // EXECUTED
  3: "❌", // CANCELLED
  4: "⚠️", // FAILED
};

interface PaymentView {
  paymentId: `0x${string}`;
  sender: `0x${string}`;
  recipient: `0x${string}`;
  token: `0x${string}`;
  amount: bigint;
  targetToken: `0x${string}`;
  targetAmount: bigint;
  status: number;
  createdAt: bigint;
  executedAt: bigint;
  approvalCount: bigint;
  requiredApprovals: bigint;
  referenceId: `0x${string}`;
}

export const SettlementMonitor = () => {
  const { address, isConnected } = useAccount();
  const [selectedPaymentId, setSelectedPaymentId] = useState<string>("");

  // Get contract info
  const { data: payFlowCoreInfo } = useDeployedContractInfo("PayFlowCore");

  // Read total payments count from contract
  const { data: totalPayments, refetch: refetchTotal } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsCreated",
  });

  const { data: totalExecuted, refetch: refetchExecuted } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsExecuted",
  });

  // Read user payments from contract
  const { data: userPayments, refetch: refetchUserPayments } = useReadContract({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "getUserPayments",
    args: address ? [address] : undefined,
    query: {
      enabled: !!payFlowCoreInfo && !!address,
      refetchInterval: 5000,
    },
  });

  // Parse user payments
  const { sentPaymentIds, receivedPaymentIds, allPaymentIds } = useMemo(() => {
    const payments = userPayments as [`0x${string}`[], `0x${string}`[]] | undefined;
    const sent = payments ? payments[0] : [];
    const received = payments ? payments[1] : [];
    return {
      sentPaymentIds: sent,
      receivedPaymentIds: received,
      allPaymentIds: [...sent, ...received],
    };
  }, [userPayments]);

  // Read first 5 payment details
  const paymentQueries = allPaymentIds.slice(0, 5).map(paymentId => ({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "getPayment" as const,
    args: [paymentId] as const,
  }));

  const { data: payment1 } = useReadContract({
    ...paymentQueries[0],
    query: { enabled: !!paymentQueries[0]?.address && allPaymentIds.length > 0 },
  });
  const { data: payment2 } = useReadContract({
    ...paymentQueries[1],
    query: { enabled: !!paymentQueries[1]?.address && allPaymentIds.length > 1 },
  });
  const { data: payment3 } = useReadContract({
    ...paymentQueries[2],
    query: { enabled: !!paymentQueries[2]?.address && allPaymentIds.length > 2 },
  });
  const { data: payment4 } = useReadContract({
    ...paymentQueries[3],
    query: { enabled: !!paymentQueries[3]?.address && allPaymentIds.length > 3 },
  });
  const { data: payment5 } = useReadContract({
    ...paymentQueries[4],
    query: { enabled: !!paymentQueries[4]?.address && allPaymentIds.length > 4 },
  });

  const payments = [payment1, payment2, payment3, payment4, payment5].filter(Boolean) as PaymentView[];

  // Calculate stats from real data
  const pendingCount = payments.filter(p => p.status === 0 || p.status === 1).length;
  const completedCount = payments.filter(p => p.status === 2).length;

  // Handle refresh
  const handleRefresh = async () => {
    await Promise.all([refetchTotal(), refetchExecuted(), refetchUserPayments()]);
    notification.success("Data refreshed");
  };

  return (
    <div className="bg-base-100 rounded-3xl shadow-lg p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-6 text-center">📊 Payment Monitor</h2>

      {/* Protocol Stats */}
      <div className="bg-base-200 rounded-xl p-4 mb-6">
        <h3 className="font-semibold mb-2">📋 Protocol Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="opacity-70">Total Created:</span>
            <span className="font-bold ml-2">{totalPayments?.toString() || "0"}</span>
          </div>
          <div>
            <span className="opacity-70">Total Executed:</span>
            <span className="font-bold ml-2">{totalExecuted?.toString() || "0"}</span>
          </div>
          <div>
            <span className="opacity-70">Your Sent:</span>
            <span className="font-bold ml-2 text-primary">{sentPaymentIds.length}</span>
          </div>
          <div>
            <span className="opacity-70">Your Received:</span>
            <span className="font-bold ml-2 text-success">{receivedPaymentIds.length}</span>
          </div>
        </div>
      </div>

      {/* Payment ID Selector */}
      <div className="flex gap-4 items-end mb-6">
        <div className="form-control flex-1">
          <label className="label">
            <span className="label-text">Payment ID (Optional)</span>
          </label>
          <input
            type="text"
            placeholder="Enter payment ID to track"
            className="input input-bordered"
            value={selectedPaymentId}
            onChange={e => setSelectedPaymentId(e.target.value)}
          />
        </div>
        <button className="btn btn-secondary" onClick={handleRefresh}>
          🔄 Refresh
        </button>
      </div>

      {/* Recent Payments List */}
      <div className="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-xl p-6 mb-6">
        <h3 className="text-xl font-bold mb-4">Your Recent Payments</h3>
        {payments.length > 0 ? (
          <div className="space-y-4">
            {payments.map((payment, idx) => (
              <div
                key={idx}
                className="bg-base-100 rounded-lg p-4 flex justify-between items-center cursor-pointer hover:bg-base-200 transition-colors"
                onClick={() => setSelectedPaymentId(payment.paymentId)}
              >
                <div>
                  <code className="text-sm font-mono">
                    {payment.paymentId.slice(0, 10)}...{payment.paymentId.slice(-8)}
                  </code>
                  <p className="text-xs opacity-70 mt-1">
                    {payment.sender.slice(0, 8)}... → {payment.recipient.slice(0, 8)}...
                  </p>
                </div>
                <div className="text-right">
                  <span className="font-bold">{formatEther(payment.amount)} tokens</span>
                  <div className={`badge ${STATUS_COLORS[payment.status] || "badge-ghost"} ml-2`}>
                    {STATUS_ICONS[payment.status] || "?"} {STATUS_NAMES[payment.status] || "UNKNOWN"}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <span className="text-4xl">📭</span>
            <p className="mt-2 opacity-70">
              {isConnected ? "No payments found for your address" : "Connect wallet to see your payments"}
            </p>
          </div>
        )}
      </div>

      {/* Payment Stats Summary */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-warning/10 rounded-xl p-4 text-center">
          <span className="text-3xl">{pendingCount}</span>
          <p className="text-sm opacity-70">Pending</p>
        </div>
        <div className="bg-success/10 rounded-xl p-4 text-center">
          <span className="text-3xl">{completedCount}</span>
          <p className="text-sm opacity-70">Completed</p>
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 bg-info/10 rounded-xl p-4">
        <h4 className="font-semibold text-info">ℹ️ How Payments Work</h4>
        <ul className="text-sm opacity-70 mt-2 space-y-1">
          <li>• Create a payment with recipient address and amount</li>
          <li>• Compliance verification happens automatically</li>
          <li>• Payments execute through the oracle-verified settlement engine</li>
          <li>• Track status in real-time on this dashboard</li>
        </ul>
      </div>

      {!isConnected && (
        <div className="mt-4 text-center text-warning">Connect your wallet to view and track your payments</div>
      )}
    </div>
  );
};

export default SettlementMonitor;
