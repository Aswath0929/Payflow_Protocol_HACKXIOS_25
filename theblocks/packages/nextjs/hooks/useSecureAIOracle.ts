import { useState, useCallback } from "react";
import { useAccount, useWriteContract, useReadContract } from "wagmi";
import { keccak256, toHex, encodeAbiParameters, parseAbiParameters } from "viem";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║                    SECURE AI ORACLE HOOK                                      ║
 * ║                                                                               ║
 * ║   React hook for secure AI fraud detection                                   ║
 * ║   - Calls off-chain AI oracle securely                                       ║
 * ║   - Submits signed results to blockchain                                     ║
 * ║   - Verifies signatures before trusting results                              ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 *
 * ARCHITECTURE:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                          FRONTEND (This Hook)                               │
 * │                                                                             │
 * │  1. Send tx data to backend ───▶ Backend (API key safe)                    │
 * │  2. Receive SIGNED result ◀──── Backend calls AI API                       │
 * │  3. Submit to blockchain ─────▶ Smart contract verifies signature          │
 * │  4. Get on-chain result ◀────── Contract stores assessment                 │
 * │                                                                             │
 * └─────────────────────────────────────────────────────────────────────────────┘
 */

// Contract ABI (partial)
const SECURE_AI_ORACLE_ABI = [
  {
    inputs: [
      { name: "transactionId", type: "bytes32" },
      { name: "riskScore", type: "uint8" },
      { name: "approved", type: "bool" },
      { name: "explanation", type: "string" },
      { name: "confidence", type: "uint256" },
      { name: "model", type: "string" },
      { name: "timestamp", type: "uint256" },
      { name: "signature", type: "bytes" },
    ],
    name: "submitAssessment",
    outputs: [],
    stateMutability: "nonpayable",
    type: "function",
  },
  {
    inputs: [{ name: "transactionId", type: "bytes32" }],
    name: "isTransactionApproved",
    outputs: [
      { name: "approved", type: "bool" },
      { name: "riskScore", type: "uint8" },
      { name: "reason", type: "string" },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [{ name: "transactionId", type: "bytes32" }],
    name: "getAssessment",
    outputs: [
      {
        components: [
          { name: "transactionId", type: "bytes32" },
          { name: "riskScore", type: "uint8" },
          { name: "riskLevel", type: "uint8" },
          { name: "approved", type: "bool" },
          { name: "explanation", type: "string" },
          { name: "confidence", type: "uint256" },
          { name: "model", type: "string" },
          { name: "timestamp", type: "uint256" },
          { name: "oracle", type: "address" },
          { name: "signature", type: "bytes" },
        ],
        type: "tuple",
      },
    ],
    stateMutability: "view",
    type: "function",
  },
  {
    inputs: [],
    name: "getStatistics",
    outputs: [
      { name: "_totalAssessments", type: "uint256" },
      { name: "_blockedTransactions", type: "uint256" },
      { name: "_flaggedTransactions", type: "uint256" },
      { name: "_oracleCount", type: "uint256" },
    ],
    stateMutability: "view",
    type: "function",
  },
] as const;

// Types
interface TransactionData {
  sender: string;
  recipient: string;
  amount: number;
  senderTxCount?: number;
  senderAvgAmount?: number;
  senderTotalVolume?: number;
}

interface SignedAnalysisResult {
  transaction_id: string;
  risk_score: number;
  risk_level: string;
  approved: boolean;
  explanation: string;
  confidence: number;
  model: string;
  processing_time_ms: number;
  signature: string;
  oracle_address: string;
  signed_at: number;
  message_hash: string;
  message: string;
  // Neural Network fields (PRIMARY detection)
  neural_net_score?: number;
  neural_net_confidence?: number;
  neural_net_risk_level?: string;
  neural_net_is_anomaly?: boolean;
  neural_net_explanation?: string;
}

interface AnalysisState {
  isLoading: boolean;
  isSubmitting: boolean;
  result: SignedAnalysisResult | null;
  error: string | null;
  txHash: string | null;
}

type AIProvider = "openai" | "anthropic" | "local";

/**
 * Hook for secure AI fraud detection
 */
export function useSecureAIOracle(
  contractAddress: string,
  oracleApiUrl: string = "http://localhost:8000"
) {
  const { address } = useAccount();
  const { writeContractAsync } = useWriteContract();

  const [state, setState] = useState<AnalysisState>({
    isLoading: false,
    isSubmitting: false,
    result: null,
    error: null,
    txHash: null,
  });

  /**
   * Generate unique transaction ID
   */
  const generateTransactionId = useCallback(
    (data: TransactionData): `0x${string}` => {
      const timestamp = Date.now();
      const payload = `${data.sender}-${data.recipient}-${data.amount}-${timestamp}`;
      return keccak256(toHex(payload));
    },
    []
  );

  /**
   * Call AI oracle backend (API key stays server-side)
   */
  const callAIOracle = useCallback(
    async (
      transactionId: string,
      data: TransactionData,
      provider: AIProvider = "local"
    ): Promise<SignedAnalysisResult> => {
      // All providers now use the unified /analyze endpoint
      // Qwen3 MoE is used as the AI backend (GPT-4 removed)
      const endpoint = "/analyze";

      const response = await fetch(`${oracleApiUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          transaction_id: transactionId,
          sender: data.sender,
          recipient: data.recipient,
          amount: data.amount,
          sender_tx_count: data.senderTxCount || 0,
          sender_avg_amount: data.senderAvgAmount || 0,
          sender_total_volume: data.senderTotalVolume || 0,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Oracle request failed");
      }

      return response.json();
    },
    [oracleApiUrl]
  );

  /**
   * Submit signed assessment to blockchain
   */
  const submitToBlockchain = useCallback(
    async (result: SignedAnalysisResult): Promise<string> => {
      // Convert transaction_id to bytes32
      const transactionIdBytes = result.transaction_id.startsWith("0x")
        ? (result.transaction_id as `0x${string}`)
        : (`0x${result.transaction_id}` as `0x${string}`);

      // Convert confidence to basis points (0-10000)
      const confidenceBps = BigInt(Math.floor(result.confidence * 10000));

      // Ensure signature is proper hex
      const signatureBytes = result.signature.startsWith("0x")
        ? (result.signature as `0x${string}`)
        : (`0x${result.signature}` as `0x${string}`);

      const hash = await writeContractAsync({
        address: contractAddress as `0x${string}`,
        abi: SECURE_AI_ORACLE_ABI,
        functionName: "submitAssessment",
        args: [
          transactionIdBytes,
          result.risk_score,
          result.approved,
          result.explanation,
          confidenceBps,
          result.model,
          BigInt(result.signed_at),
          signatureBytes,
        ],
      });

      return hash;
    },
    [contractAddress, writeContractAsync]
  );

  /**
   * Main function: Analyze transaction with AI and submit to blockchain
   */
  const analyzeAndSubmit = useCallback(
    async (
      data: TransactionData,
      provider: AIProvider = "local",
      submitOnChain: boolean = true
    ) => {
      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
      }));

      try {
        // 1. Generate transaction ID
        const transactionId = generateTransactionId(data);

        // 2. Call AI oracle (secure - API key on server)
        const result = await callAIOracle(transactionId, data, provider);

        setState((prev) => ({
          ...prev,
          isLoading: false,
          result,
        }));

        // 3. Optionally submit to blockchain
        if (submitOnChain) {
          setState((prev) => ({
            ...prev,
            isSubmitting: true,
          }));

          const txHash = await submitToBlockchain(result);

          setState((prev) => ({
            ...prev,
            isSubmitting: false,
            txHash,
          }));

          return { result, txHash };
        }

        return { result, txHash: null };
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Analysis failed";
        setState((prev) => ({
          ...prev,
          isLoading: false,
          isSubmitting: false,
          error: errorMessage,
        }));
        throw error;
      }
    },
    [generateTransactionId, callAIOracle, submitToBlockchain]
  );

  /**
   * Analyze only (without blockchain submission)
   */
  const analyzeOnly = useCallback(
    async (data: TransactionData, provider: AIProvider = "local") => {
      return analyzeAndSubmit(data, provider, false);
    },
    [analyzeAndSubmit]
  );

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    setState({
      isLoading: false,
      isSubmitting: false,
      result: null,
      error: null,
      txHash: null,
    });
  }, []);

  return {
    ...state,
    analyzeAndSubmit,
    analyzeOnly,
    reset,
    generateTransactionId,
  };
}

/**
 * Hook to read oracle statistics
 */
export function useOracleStatistics(contractAddress: string) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: contractAddress as `0x${string}`,
    abi: SECURE_AI_ORACLE_ABI,
    functionName: "getStatistics",
  });

  return {
    totalAssessments: data ? Number(data[0]) : 0,
    blockedTransactions: data ? Number(data[1]) : 0,
    flaggedTransactions: data ? Number(data[2]) : 0,
    oracleCount: data ? Number(data[3]) : 0,
    isLoading,
    error: error?.message,
    refetch,
  };
}

/**
 * Hook to check if a specific transaction is approved
 */
export function useTransactionApproval(
  contractAddress: string,
  transactionId?: string
) {
  const { data, isLoading, error, refetch } = useReadContract({
    address: contractAddress as `0x${string}`,
    abi: SECURE_AI_ORACLE_ABI,
    functionName: "isTransactionApproved",
    args: transactionId ? [transactionId as `0x${string}`] : undefined,
    query: {
      enabled: !!transactionId,
    },
  });

  return {
    approved: data ? data[0] : null,
    riskScore: data ? Number(data[1]) : null,
    reason: data ? data[2] : null,
    isLoading,
    error: error?.message,
    refetch,
  };
}

export default useSecureAIOracle;
