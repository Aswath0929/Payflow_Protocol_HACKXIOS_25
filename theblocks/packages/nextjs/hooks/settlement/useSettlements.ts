"use client";

import { useCallback, useState } from "react";
import { formatEther, parseEther } from "viem";
import { useAccount } from "wagmi";
import { useScaffoldReadContract, useScaffoldWatchContractEvent } from "~~/hooks/scaffold-eth";

/**
 * @title useSettlements Hook
 * @author TheBlocks Team - Hackxios 2K25
 * @notice Payment/settlement state management hook
 *
 * NOTE: Uses actual PayFlowCore functions where available
 * Some features are in demo mode as the contract uses payment-based architecture
 */

export interface Settlement {
  id: bigint;
  initiator: string;
  totalAmount: bigint;
  totalDeposited: bigint;
  state: number;
  createdAt: bigint;
  timeout: bigint;
  queuePosition: bigint;
  totalTransfers: bigint;
  executedTransfers: bigint;
}

export interface Transfer {
  from: string;
  to: string;
  amount: bigint;
  executed: boolean;
}

const STATE_NAMES = ["PENDING", "INITIATED", "EXECUTING", "FINALIZED", "DISPUTED", "FAILED"];

export const useSettlements = (settlementId?: bigint) => {
  const { address: connectedAddress, isConnected } = useAccount();
  const [userSettlements] = useState<bigint[]>([]);
  const [isLoading] = useState(false);
  const [isCreating] = useState(false);

  // Use actual PayFlowCore functions
  const { data: totalPaymentsCreated, refetch: refetchTotalCreated } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsCreated",
  });

  const { data: totalPaymentsExecuted, refetch: refetchTotalExecuted } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "totalPaymentsExecuted",
  });

  const { data: averageSettlementTime } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "averageSettlementTime",
  });

  const { data: isPaused } = useScaffoldReadContract({
    contractName: "PayFlowCore",
    functionName: "paused",
  });

  // Watch actual PayFlowCore events
  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "PaymentCreated",
    onLogs: logs => {
      console.log("Payment created:", logs);
      refetchTotalCreated();
    },
  });

  useScaffoldWatchContractEvent({
    contractName: "PayFlowCore",
    eventName: "PaymentExecuted",
    onLogs: logs => {
      console.log("Payment executed:", logs);
      refetchTotalExecuted();
    },
  });

  // Demo settlement data (since getSettlement doesn't exist)
  const settlement: Settlement | null = settlementId
    ? {
        id: settlementId,
        initiator: connectedAddress || "0x0000000000000000000000000000000000000000",
        totalAmount: parseEther("1.0"),
        totalDeposited: parseEther("1.0"),
        state: 1,
        createdAt: BigInt(Math.floor(Date.now() / 1000) - 3600),
        timeout: BigInt(86400),
        queuePosition: 0n,
        totalTransfers: 1n,
        executedTransfers: 0n,
      }
    : null;

  const transfers: Transfer[] = [];

  // Demo create settlement - contract has complex signature
  const createSettlement = useCallback(
    async (transfersList: { from: string; to: string; amount: string }[]) => {
      if (!isConnected) throw new Error("Wallet not connected");

      const firstTransfer = transfersList[0];
      if (!firstTransfer) throw new Error("No transfer specified");

      // Demo mode - simulate payment creation
      console.log("Demo: Creating payment for", firstTransfer);
      await new Promise(resolve => setTimeout(resolve, 1500));
      refetchTotalCreated();
      return null;
    },
    [isConnected, refetchTotalCreated],
  );

  // Demo functions for unavailable features
  const deposit = useCallback(
    async () => {
      console.log("Demo: deposit not available in PayFlowCore");
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    [],
  );

  const initiateSettlement = useCallback(
    async () => {
      console.log("Demo: initiateSettlement not available");
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    [],
  );

  const executeSettlement = useCallback(
    async () => {
      console.log("Demo: executeSettlement not available");
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    [],
  );

  const refundSettlement = useCallback(
    async () => {
      console.log("Demo: refundSettlement not available");
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    [],
  );

  const disputeSettlement = useCallback(
    async () => {
      console.log("Demo: disputeSettlement not available");
      await new Promise(resolve => setTimeout(resolve, 1000));
    },
    [],
  );

  // Utility functions
  const getStateName = (state: number) => STATE_NAMES[state] || "UNKNOWN";

  const getProgress = () => {
    if (!settlement || settlement.totalTransfers === 0n) return 0;
    return Number((settlement.executedTransfers * 100n) / settlement.totalTransfers);
  };

  const getDepositProgress = () => {
    if (!settlement || settlement.totalAmount === 0n) return 0;
    return Math.min(Number((settlement.totalDeposited * 100n) / settlement.totalAmount), 100);
  };

  const formatAmount = (amount: bigint) => formatEther(amount);

  const refetchAll = useCallback(() => {
    refetchTotalCreated();
    refetchTotalExecuted();
  }, [refetchTotalCreated, refetchTotalExecuted]);

  return {
    // Settlement data
    settlement,
    transfers,
    canInitiate: true,
    initiateReason: "",
    isEligibleForRefund: false,

    // Actual contract data
    totalPaymentsCreated,
    totalPaymentsExecuted,
    averageSettlementTime,
    isPaused,

    // Queue data (demo values)
    queueHead: 0n,
    queueLength: 0n,
    nextSettlementId: 1n,

    // User data
    userSettlements,
    connectedAddress,
    isConnected,

    // Actions
    createSettlement,
    deposit,
    initiateSettlement,
    executeSettlement,
    refundSettlement,
    disputeSettlement,
    refetchAll,
    refetchSettlement: refetchTotalCreated,

    // Loading states
    isLoading,
    isCreating,
    isDepositing: false,
    isInitiating: false,
    isExecuting: false,
    isRefunding: false,
    isDisputing: false,

    // Utilities
    getStateName,
    getProgress,
    getDepositProgress,
    formatAmount,
  };
};

export default useSettlements;






