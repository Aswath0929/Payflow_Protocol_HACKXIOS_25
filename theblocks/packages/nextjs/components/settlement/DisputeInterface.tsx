"use client";

import { useState } from "react";
import { formatUnits } from "viem";
import { useAccount, useReadContract } from "wagmi";
import { useDeployedContractInfo } from "~~/hooks/scaffold-eth";
import { notification } from "~~/utils/scaffold-eth";

/**
 * @title DisputeInterface
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Interface for viewing payments and dispute status
 *
 * Uses real PayFlowCore contract functions:
 * - getUserPayments(address): Get user's sent and received payment IDs
 * - getPayment(paymentId): Get payment details
 * - getPaymentCount(): Get total payment count
 *
 * NOTE: Dispute submission is a planned feature - the UI shows real payment data
 * but the actual dispute transaction is not yet implemented in the contract.
 */

const DISPUTE_REASONS = [
  { id: "oracle_manipulation", label: "Oracle Price Manipulation", description: "Detected abnormal price deviation" },
  { id: "frontrunning", label: "Frontrunning Detected", description: "Transaction ordering was manipulated" },
  { id: "incorrect_amount", label: "Incorrect Amount", description: "Transfer amounts don't match agreement" },
  { id: "unauthorized", label: "Unauthorized Settlement", description: "Settlement was not authorized by parties" },
  { id: "other", label: "Other", description: "Other dispute reason" },
];

// Payment status enum from PayFlowCore
const PAYMENT_STATUS = ["PENDING", "APPROVED", "EXECUTED", "CANCELLED", "FAILED"] as const;

export const DisputeInterface = () => {
  const { address, isConnected } = useAccount();
  const [selectedPaymentIndex, setSelectedPaymentIndex] = useState<number>(0);
  const [selectedReason, setSelectedReason] = useState<string>("");
  const [customReason, setCustomReason] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Get PayFlowCore contract info
  const { data: payFlowCoreInfo } = useDeployedContractInfo("PayFlowCore");

  // Read user payments
  const { data: userPayments } = useReadContract({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "getUserPayments",
    args: address ? [address] : undefined,
    query: {
      enabled: !!payFlowCoreInfo && !!address,
      refetchInterval: 5000,
    },
  });

  // Get payment count
  const { data: paymentCount } = useReadContract({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "getPaymentCount",
    query: {
      enabled: !!payFlowCoreInfo,
      refetchInterval: 5000,
    },
  });

  // Parse user payments
  const sentPayments = userPayments ? (userPayments as [`0x${string}`[], `0x${string}`[]])[0] : [];
  const receivedPayments = userPayments ? (userPayments as [`0x${string}`[], `0x${string}`[]])[1] : [];
  const allPaymentIds = [...sentPayments, ...receivedPayments];

  // Read first selected payment details
  const selectedPaymentId = allPaymentIds[selectedPaymentIndex];
  const { data: paymentDetails } = useReadContract({
    address: payFlowCoreInfo?.address,
    abi: payFlowCoreInfo?.abi,
    functionName: "getPayment",
    args: selectedPaymentId ? [selectedPaymentId] : undefined,
    query: {
      enabled: !!payFlowCoreInfo && !!selectedPaymentId,
      refetchInterval: 3000,
    },
  });

  // Parse payment details
  const payment = paymentDetails as
    | {
        paymentId: `0x${string}`;
        sender: `0x${string}`;
        recipient: `0x${string}`;
        token: `0x${string}`;
        amount: bigint;
        status: number;
        createdAt: bigint;
        executedAt: bigint;
      }
    | undefined;

  const paymentStatus = payment ? PAYMENT_STATUS[payment.status] || "UNKNOWN" : "UNKNOWN";
  const isDisputable = payment && (paymentStatus === "PENDING" || paymentStatus === "APPROVED");

  const handleSubmitDispute = async () => {
    if (!isConnected) {
      notification.error("Please connect your wallet");
      return;
    }

    const reason =
      selectedReason === "other" ? customReason : DISPUTE_REASONS.find(r => r.id === selectedReason)?.label || "";

    if (!reason) {
      notification.error("Please select or enter a dispute reason");
      return;
    }

    setIsSubmitting(true);

    try {
      // NOTE: Actual dispute transaction would go here when contract supports it
      // For now, we log the dispute and notify user
      console.log("Dispute submission:", {
        paymentId: selectedPaymentId,
        reason,
        timestamp: Date.now(),
      });

      notification.info(
        "Dispute recorded locally. On-chain dispute resolution is coming in a future protocol upgrade.",
      );
      setSelectedReason("");
      setCustomReason("");
    } catch (error: unknown) {
      console.error("Dispute submission failed:", error);
      const errorMessage = error instanceof Error ? error.message : "Failed to submit dispute";
      notification.error(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-base-100 rounded-3xl shadow-lg p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-2 text-center">⚠️ Dispute Center</h2>
      <p className="text-center text-sm opacity-70 mb-6">Review payments and flag suspicious activity</p>

      {/* Protocol Stats */}
      <div className="bg-primary/10 border border-primary/30 rounded-xl p-3 mb-6">
        <div className="flex justify-between items-center">
          <span className="text-sm">Total Protocol Payments:</span>
          <span className="font-bold">{paymentCount?.toString() || "0"}</span>
        </div>
        <div className="flex justify-between items-center mt-1">
          <span className="text-sm">Your Payments (Sent/Received):</span>
          <span className="font-bold">
            {sentPayments.length} / {receivedPayments.length}
          </span>
        </div>
      </div>

      {/* Payment Selector */}
      {allPaymentIds.length > 0 ? (
        <div className="form-control mb-6">
          <label className="label">
            <span className="label-text font-semibold">Select Payment</span>
          </label>
          <select
            className="select select-bordered w-full"
            value={selectedPaymentIndex}
            onChange={e => setSelectedPaymentIndex(Number(e.target.value))}
          >
            {allPaymentIds.map((id, idx) => (
              <option key={id} value={idx}>
                Payment #{idx + 1} - {id.slice(0, 10)}...{id.slice(-8)}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="bg-base-200 rounded-xl p-6 mb-6 text-center">
          <span className="text-4xl">📭</span>
          <h3 className="font-bold text-lg mt-2">No Payments Found</h3>
          <p className="opacity-70">
            {isConnected
              ? "You have no payments yet. Create a payment to get started."
              : "Connect your wallet to view your payments."}
          </p>
        </div>
      )}

      {/* Payment Status */}
      {payment && (
        <div className="bg-base-200 rounded-xl p-4 mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold">Payment Details</span>
            <span
              className={`badge ${
                paymentStatus === "PENDING"
                  ? "badge-warning"
                  : paymentStatus === "APPROVED"
                    ? "badge-info"
                    : paymentStatus === "EXECUTED"
                      ? "badge-success"
                      : paymentStatus === "FAILED"
                        ? "badge-error"
                        : "badge-ghost"
              }`}
            >
              {paymentStatus}
            </span>
          </div>
          <div className="text-sm opacity-70 space-y-1">
            <p>
              From:{" "}
              <code className="text-xs">
                {payment.sender.slice(0, 10)}...{payment.sender.slice(-8)}
              </code>
            </p>
            <p>
              To:{" "}
              <code className="text-xs">
                {payment.recipient.slice(0, 10)}...{payment.recipient.slice(-8)}
              </code>
            </p>
            <p>Amount: {formatUnits(payment.amount, 18)} tokens</p>
            <p>Created: {new Date(Number(payment.createdAt) * 1000).toLocaleString()}</p>
          </div>
        </div>
      )}

      {/* Dispute Form */}
      {isDisputable ? (
        <div className="space-y-6">
          {/* Reason Selection */}
          <div>
            <h3 className="font-semibold mb-3">Select Dispute Reason</h3>
            <div className="space-y-2">
              {DISPUTE_REASONS.map(reason => (
                <label
                  key={reason.id}
                  className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                    selectedReason === reason.id ? "border-error bg-error/10" : "border-base-300 hover:border-error/50"
                  }`}
                >
                  <input
                    type="radio"
                    name="dispute-reason"
                    className="radio radio-error mt-1"
                    checked={selectedReason === reason.id}
                    onChange={() => setSelectedReason(reason.id)}
                  />
                  <div>
                    <span className="font-medium">{reason.label}</span>
                    <p className="text-sm opacity-70">{reason.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Custom Reason Input */}
          {selectedReason === "other" && (
            <div className="form-control">
              <label className="label">
                <span className="label-text">Describe Your Dispute</span>
              </label>
              <textarea
                className="textarea textarea-bordered h-24"
                placeholder="Provide detailed reason for the dispute..."
                value={customReason}
                onChange={e => setCustomReason(e.target.value)}
              />
            </div>
          )}

          {/* Info Box */}
          <div className="bg-info/10 border border-info/30 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">ℹ️</span>
              <div>
                <h4 className="font-semibold text-info">Coming Soon</h4>
                <p className="text-sm opacity-80">
                  On-chain dispute resolution is planned for the next protocol version. For now, disputes are logged for
                  review.
                </p>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-center">
            <button
              className={`btn btn-warning btn-lg ${isSubmitting ? "loading" : ""}`}
              onClick={handleSubmitDispute}
              disabled={!isConnected || isSubmitting || !selectedReason}
            >
              {isSubmitting ? "Recording..." : "📝 Record Dispute"}
            </button>
          </div>
        </div>
      ) : payment?.status === 2 ? ( // EXECUTED
        <div className="bg-success/10 rounded-xl p-6 text-center">
          <span className="text-4xl">✅</span>
          <h3 className="font-bold text-lg mt-2">Payment Executed</h3>
          <p className="opacity-70">This payment has been successfully executed.</p>
        </div>
      ) : payment ? (
        <div className="bg-base-200 rounded-xl p-6 text-center">
          <span className="text-4xl">ℹ️</span>
          <h3 className="font-bold text-lg mt-2">Not Disputable</h3>
          <p className="opacity-70">Payments can only be disputed in PENDING or APPROVED states.</p>
        </div>
      ) : null}

      {!isConnected && (
        <p className="text-center text-warning mt-4">Connect your wallet to view and dispute payments</p>
      )}
    </div>
  );
};

export default DisputeInterface;
