"use client";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                     PAYFLOW AI FRAUD MONITORING DASHBOARD                             ║
 * ║                                                                                       ║
 * ║   Real-time Transaction Monitoring with AI Analysis                                  ║
 * ║                                                                                       ║
 * ║   Features:                                                                           ║
 * ║   • Live WebSocket connection to AI Oracle                                           ║
 * ║   • Real-time transaction analysis stream                                            ║
 * ║   • Risk score visualization with gauges                                             ║
 * ║   • Alert management and review workflow                                             ║
 * ║   • Statistics and performance metrics                                               ║
 * ║                                                                                       ║
 * ║   Hackxios 2K25 - PayFlow Protocol                                                   ║
 * ╚═══════════════════════════════════════════════════════════════════════════════════════╝
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { useAccount } from "wagmi";

// Types
interface TransactionAnalysis {
  transaction_id: string;
  sender: string;
  recipient: string;
  amount: number;
  overall_score: number;
  velocity_score: number;
  amount_score: number;
  pattern_score: number;
  graph_score: number;
  timing_score: number;
  ai_score: number;
  // Neural Network fields (PRIMARY)
  neural_net_score: number;
  neural_net_confidence: number;
  neural_net_risk_level: string;
  neural_net_is_anomaly: boolean;
  neural_net_explanation: string;
  // Classification
  risk_level: "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  approved: boolean;
  flagged: boolean;
  blocked: boolean;
  alerts: string[];
  ai_explanation: string;
  analysis_time_ms: number;
  signature: string;
  signer_address: string;
  timestamp: string;
}

// Expert AI Oracle Types (NEW - v3.0) - Matches actual API response
interface ExpertVerdict {
  transaction: {
    id: string;
    sender: string;
    recipient: string;
    amount: number;
    timestamp: number;
  };
  risk_assessment: {
    score: number;
    level: string;
    emoji: string;
    confidence: number;
  };
  model_scores: {
    neural_ensemble: number;
    typology_detector: number;
    qwen3_llm: number;
    compliance_risk: number;
  };
  analysis: {
    features_extracted: number;
    primary_typology: string | null;
    detected_typologies: DetectedTypology[];
    compliance_status: string;
  };
  explainability: {
    explanation: string;
    key_risk_factors: string[];
    recommendations: string[];
  };
  signature: {
    signature: string | null;
    signer_address: string | null;
  };
  performance: {
    total_time_ms: number;
    breakdown: {
      feature_extraction: number;
      neural_ensemble: number;
      typology_detection: number;
      qwen3_llm: number;
      compliance_check: number;
    };
    meets_latency_requirement: boolean;
  };
  metadata: {
    model_version: string;
    engine_name: string;
    timestamp: string;
  };
}

interface DetectedTypology {
  typology: string;
  typology_name: string;
  confidence: number;
  severity: string;
  triggered_rules: string[];
  evidence: Record<string, unknown>;
  market_impact_reference: string;
}

interface ExpertMetrics {
  overview: {
    total_transactions: number;
    total_fraud_detected: number;
  };
  latency: {
    avg_ms: number;
    p95_ms: number;
    p99_ms: number;
  };
  accuracy: {
    accuracy_pct: number;
    false_positive_rate_pct: number;
  };
  judge_requirements: {
    visa_latency_met: boolean;
    paypal_fpr_met: boolean;
    accuracy_target_met: boolean;
  };
  risk_distribution: {
    safe: number;
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
}

interface OracleStats {
  total_analyses: number;
  total_blocked: number;
  total_flagged: number;
  avg_latency_ms: number;
  gpt4_calls: number;
  neural_net_predictions: number;  // NEW: Neural network predictions count
  cache_hits: number;
  total_profiles: number;
  blacklist_size: number;
  oracle_address: string;
  gpt4_enabled: boolean;
  ml_trained: boolean;
  websocket_connections: number;
  // Expert Oracle Stats (v3.0)
  expert_version?: string;
  expert_ensemble_trained?: boolean;
  expert_typologies_count?: number;
  expert_llm_model?: string;
}

interface WebSocketMessage {
  type: "connected" | "analysis" | "blacklist_update" | "stats" | "pong" | "analysis_result";
  data: any;
}

// Risk level colors
const RISK_COLORS = {
  SAFE: { bg: "bg-green-500", text: "text-green-500", gradient: "from-green-400 to-green-600" },
  LOW: { bg: "bg-blue-500", text: "text-blue-500", gradient: "from-blue-400 to-blue-600" },
  MEDIUM: { bg: "bg-yellow-500", text: "text-yellow-500", gradient: "from-yellow-400 to-yellow-600" },
  HIGH: { bg: "bg-orange-500", text: "text-orange-500", gradient: "from-orange-400 to-orange-600" },
  CRITICAL: { bg: "bg-red-500", text: "text-red-500", gradient: "from-red-400 to-red-600" },
};

// API Base URL
const API_URL = process.env.NEXT_PUBLIC_AI_ORACLE_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_AI_ORACLE_WS || "ws://localhost:8000/ws";

// ═══════════════════════════════════════════════════════════════════════════════
//                              COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Animated Risk Gauge Component
 */
function RiskGauge({ score, size = "lg" }: { score: number; size?: "sm" | "md" | "lg" }) {
  const radius = size === "lg" ? 80 : size === "md" ? 60 : 40;
  const stroke = size === "lg" ? 12 : size === "md" ? 8 : 6;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;

  const getColor = () => {
    if (score <= 20) return "#22c55e";
    if (score <= 40) return "#3b82f6";
    if (score <= 60) return "#eab308";
    if (score <= 80) return "#f97316";
    return "#ef4444";
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        className="transform -rotate-90"
        width={(radius + stroke) * 2}
        height={(radius + stroke) * 2}
      >
        {/* Background circle */}
        <circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-base-300"
        />
        {/* Progress circle */}
        <circle
          cx={radius + stroke}
          cy={radius + stroke}
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth={stroke}
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={`font-bold ${size === "lg" ? "text-3xl" : size === "md" ? "text-2xl" : "text-lg"}`}>
          {score}
        </span>
        <span className="text-xs opacity-60">Risk</span>
      </div>
    </div>
  );
}

/**
 * Transaction Card Component
 */
function TransactionCard({ analysis }: { analysis: TransactionAnalysis }) {
  const colors = RISK_COLORS[analysis.risk_level];
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`card bg-base-200 shadow-lg hover:shadow-xl transition-all duration-300 ${
        analysis.blocked ? "border-2 border-error" : analysis.flagged ? "border-2 border-warning" : ""
      }`}
    >
      <div className="card-body p-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <RiskGauge score={analysis.overall_score} size="sm" />
            <div>
              <div className="flex items-center gap-2">
                <span className={`badge ${colors.bg} text-white font-semibold`}>
                  {analysis.risk_level}
                </span>
                {analysis.blocked && <span className="badge badge-error">BLOCKED</span>}
                {analysis.flagged && <span className="badge badge-warning">REVIEW</span>}
                {analysis.approved && !analysis.flagged && !analysis.blocked && (
                  <span className="badge badge-success">APPROVED</span>
                )}
              </div>
              <p className="text-xs opacity-60 mt-1">{analysis.analysis_time_ms.toFixed(1)}ms</p>
            </div>
          </div>
          <div className="text-right">
            <p className="font-bold text-lg">${analysis.amount.toLocaleString()}</p>
            <p className="text-xs opacity-60">USDC</p>
          </div>
        </div>

        {/* Addresses */}
        <div className="mt-3 space-y-1">
          <div className="flex items-center gap-2 text-sm">
            <span className="opacity-60">From:</span>
            <code className="text-xs bg-base-300 px-2 py-1 rounded">
              {analysis.sender.slice(0, 10)}...{analysis.sender.slice(-8)}
            </code>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="opacity-60">To:</span>
            <code className="text-xs bg-base-300 px-2 py-1 rounded">
              {analysis.recipient.slice(0, 10)}...{analysis.recipient.slice(-8)}
            </code>
          </div>
        </div>

        {/* Alerts */}
        {analysis.alerts.length > 0 && (
          <div className="mt-3">
            <div className="flex flex-wrap gap-1">
              {analysis.alerts.map((alert, i) => (
                <span key={i} className="badge badge-outline badge-sm">
                  ⚠️ {alert.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Expand button */}
        <button
          className="btn btn-ghost btn-xs mt-2"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show Less" : "Show Details"}
        </button>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 space-y-3">
            {/* Neural Network Status (PRIMARY) */}
            <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/30 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-purple-400">🧠 NEURAL NETWORK (Primary)</span>
                <span className={`text-xs font-bold ${
                  analysis.neural_net_is_anomaly ? "text-red-400" : "text-green-400"
                }`}>
                  {analysis.neural_net_is_anomaly ? "⚠️ ANOMALY DETECTED" : "✓ NORMAL"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-lg font-bold text-purple-400">{analysis.neural_net_score || 0}</p>
                  <p className="text-xs opacity-60">NN Score</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-blue-400">{((analysis.neural_net_confidence || 0) * 100).toFixed(0)}%</p>
                  <p className="text-xs opacity-60">Confidence</p>
                </div>
                <div>
                  <p className={`text-sm font-bold ${
                    analysis.neural_net_risk_level === "SAFE" ? "text-green-400" :
                    analysis.neural_net_risk_level === "LOW" ? "text-blue-400" :
                    analysis.neural_net_risk_level === "MEDIUM" ? "text-yellow-400" :
                    analysis.neural_net_risk_level === "HIGH" ? "text-orange-400" :
                    "text-red-400"
                  }`}>{analysis.neural_net_risk_level || "N/A"}</p>
                  <p className="text-xs opacity-60">NN Risk</p>
                </div>
              </div>
            </div>

            {/* Score breakdown */}
            <div className="grid grid-cols-3 gap-2">
              <ScorePill label="Velocity" score={analysis.velocity_score} />
              <ScorePill label="Amount" score={analysis.amount_score} />
              <ScorePill label="Pattern" score={analysis.pattern_score} />
              <ScorePill label="Graph" score={analysis.graph_score} />
              <ScorePill label="Timing" score={analysis.timing_score} />
              <ScorePill label="GPT-4" score={analysis.ai_score} />
            </div>

            {/* AI Explanation */}
            {analysis.ai_explanation && (
              <div className="bg-base-300 p-3 rounded-lg">
                <p className="text-xs font-semibold mb-1">🤖 Combined AI Analysis</p>
                <p className="text-sm">{analysis.ai_explanation}</p>
              </div>
            )}

            {/* Neural Network Explanation */}
            {analysis.neural_net_explanation && (
              <div className="bg-purple-900/20 border border-purple-500/20 p-3 rounded-lg">
                <p className="text-xs font-semibold mb-1 text-purple-400">🧠 Neural Network Reasoning</p>
                <p className="text-sm text-purple-200">{analysis.neural_net_explanation}</p>
              </div>
            )}

            {/* Signature */}
            <div className="text-xs opacity-60">
              <p>Signed by: {analysis.signer_address}</p>
              <p className="truncate">Signature: {analysis.signature.slice(0, 40)}...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Score Pill Component
 */
function ScorePill({ label, score }: { label: string; score: number }) {
  const getColor = () => {
    if (score <= 20) return "text-green-500";
    if (score <= 40) return "text-blue-500";
    if (score <= 60) return "text-yellow-500";
    if (score <= 80) return "text-orange-500";
    return "text-red-500";
  };

  return (
    <div className="bg-base-300 rounded-lg p-2 text-center">
      <p className={`font-bold ${getColor()}`}>{score}</p>
      <p className="text-xs opacity-60">{label}</p>
    </div>
  );
}

/**
 * Statistics Card
 */
function StatsCard({ stats }: { stats: OracleStats }) {
  return (
    <div className="stats stats-vertical lg:stats-horizontal shadow bg-base-200 w-full">
      <div className="stat">
        <div className="stat-figure text-primary">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div className="stat-title">Total Analyses</div>
        <div className="stat-value text-primary">{stats.total_analyses.toLocaleString()}</div>
        <div className="stat-desc">🧠 Neural: {stats.neural_net_predictions || 0} | 🤖 GPT-4: {stats.gpt4_calls}</div>
      </div>
      
      <div className="stat">
        <div className="stat-figure text-error">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
          </svg>
        </div>
        <div className="stat-title">Blocked</div>
        <div className="stat-value text-error">{stats.total_blocked}</div>
        <div className="stat-desc">
          {((stats.total_blocked / Math.max(stats.total_analyses, 1)) * 100).toFixed(1)}% block rate
        </div>
      </div>
      
      <div className="stat">
        <div className="stat-figure text-warning">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div className="stat-title">Flagged</div>
        <div className="stat-value text-warning">{stats.total_flagged}</div>
        <div className="stat-desc">Pending review</div>
      </div>
      
      <div className="stat">
        <div className="stat-figure text-info">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <div className="stat-title">Avg Latency</div>
        <div className="stat-value text-info">{stats.avg_latency_ms.toFixed(0)}ms</div>
        <div className="stat-desc">Cache hits: {stats.cache_hits}</div>
      </div>
    </div>
  );
}

/**
 * Connection Status Badge
 */
function ConnectionStatus({ connected, oracleAddress }: { connected: boolean; oracleAddress: string }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${connected ? "bg-success/20" : "bg-error/20"}`}>
      <div className={`w-3 h-3 rounded-full ${connected ? "bg-success animate-pulse" : "bg-error"}`} />
      <span className="text-sm font-medium">
        {connected ? "Live" : "Disconnected"}
      </span>
      {connected && oracleAddress && (
        <code className="text-xs opacity-60 hidden md:inline">
          {oracleAddress.slice(0, 10)}...
        </code>
      )}
    </div>
  );
}

/**
 * Transaction Analyzer Form
 */
function TransactionAnalyzer({ 
  onAnalyze, 
  onExpertAnalyze,
  useExpert,
  onToggleMode 
}: { 
  onAnalyze: (tx: any) => Promise<void>;
  onExpertAnalyze: (tx: any) => Promise<void>;
  useExpert: boolean;
  onToggleMode: () => void;
}) {
  const [sender, setSender] = useState("");
  const [recipient, setRecipient] = useState("");
  const [amount, setAmount] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sender || !recipient || !amount) return;

    setLoading(true);
    try {
      if (useExpert) {
        await onExpertAnalyze({ sender, recipient, amount: parseFloat(amount) });
      } else {
        await onAnalyze({ sender, recipient, amount: parseFloat(amount) });
      }
    } finally {
      setLoading(false);
    }
  };

  // Quick test buttons for demo
  const runQuickTest = async (testCase: string) => {
    setLoading(true);
    try {
      let tx;
      switch (testCase) {
        case "tornado":
          tx = { sender: "0xUser123", recipient: "0x722122df12d4e14e13ac3b6895a86e84145b6967", amount: 50000 };
          break;
        case "ofac":
          tx = { sender: "0xSuspect", recipient: "0x8576acc5c05d6ce88f4e49bf65bdf0c62f91353c", amount: 10000 };
          break;
        case "clean":
          tx = { sender: "0xAlice123", recipient: "0xBob456", amount: 100 };
          break;
        case "highvalue":
          tx = { sender: "0xWhale", recipient: "0xExchange", amount: 500000 };
          break;
        default:
          tx = { sender: "0xTest", recipient: "0xReceiver", amount: 1000 };
      }
      if (useExpert) {
        await onExpertAnalyze(tx);
      } else {
        await onAnalyze(tx);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="card bg-base-200">
      <div className="card-body">
        <div className="flex items-center justify-between">
          <h3 className="card-title text-lg">🔍 Analyze Transaction</h3>
          <label className="swap swap-flip">
            <input type="checkbox" checked={useExpert} onChange={onToggleMode} />
            <div className="swap-on badge badge-primary">Expert AI v3</div>
            <div className="swap-off badge badge-ghost">Legacy</div>
          </label>
        </div>
        
        {useExpert && (
          <div className="alert alert-info py-2 mt-2">
            <span className="text-xs">🧠 Expert Mode: 34 features • 5-model ensemble • Qwen3 LLM • 15 typologies</span>
          </div>
        )}
        
        <div className="form-control">
          <label className="label">
            <span className="label-text">Sender Address</span>
          </label>
          <input
            type="text"
            placeholder="0x..."
            className="input input-bordered"
            value={sender}
            onChange={(e) => setSender(e.target.value)}
          />
        </div>
        
        <div className="form-control">
          <label className="label">
            <span className="label-text">Recipient Address</span>
          </label>
          <input
            type="text"
            placeholder="0x..."
            className="input input-bordered"
            value={recipient}
            onChange={(e) => setRecipient(e.target.value)}
          />
        </div>
        
        <div className="form-control">
          <label className="label">
            <span className="label-text">Amount (USDC)</span>
          </label>
          <input
            type="number"
            placeholder="1000.00"
            className="input input-bordered"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </div>
        
        <button type="submit" className={`btn btn-primary mt-4 ${loading ? "loading" : ""}`} disabled={loading}>
          {loading ? "Analyzing..." : useExpert ? "🧠 Expert Analysis" : "Analyze Transaction"}
        </button>
        
        {/* Quick Test Buttons */}
        <div className="divider text-xs">Quick Tests</div>
        <div className="grid grid-cols-2 gap-2">
          <button type="button" className="btn btn-xs btn-error" onClick={() => runQuickTest("tornado")} disabled={loading}>
            🌪️ Tornado Cash
          </button>
          <button type="button" className="btn btn-xs btn-warning" onClick={() => runQuickTest("ofac")} disabled={loading}>
            🚫 OFAC Address
          </button>
          <button type="button" className="btn btn-xs btn-success" onClick={() => runQuickTest("clean")} disabled={loading}>
            ✅ Clean Transfer
          </button>
          <button type="button" className="btn btn-xs btn-info" onClick={() => runQuickTest("highvalue")} disabled={loading}>
            💰 High Value
          </button>
        </div>
      </div>
    </form>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function AIFraudDashboard() {
  const { address } = useAccount();
  const [connected, setConnected] = useState(false);
  const [oracleAddress, setOracleAddress] = useState("");
  const [stats, setStats] = useState<OracleStats | null>(null);
  const [analyses, setAnalyses] = useState<TransactionAnalysis[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expertVerdict, setExpertVerdict] = useState<ExpertVerdict | null>(null);
  const [expertMetrics, setExpertMetrics] = useState<ExpertMetrics | null>(null);
  const [useExpertMode, setUseExpertMode] = useState(true); // Default to Expert mode
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch Expert Metrics on mount
  useEffect(() => {
    const fetchExpertMetrics = async () => {
      try {
        const response = await fetch(`${API_URL}/expert/metrics`);
        if (response.ok) {
          const metrics = await response.json();
          setExpertMetrics(metrics);
        }
      } catch (e) {
        console.log("Expert metrics not available yet");
      }
    };
    fetchExpertMetrics();
    const interval = setInterval(fetchExpertMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket connection with improved error handling
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log("🔌 WebSocket connected");
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          switch (message.type) {
            case "connected":
              setOracleAddress(message.data.oracle_address);
              setStats(message.data.stats);
              break;
            
            case "analysis":
              const analysis = message.data as TransactionAnalysis;
              setAnalyses((prev) => [analysis, ...prev.slice(0, 49)]);
              break;
            
            case "stats":
              setStats(message.data);
              break;
            
            case "analysis_result":
              const result = message.data as TransactionAnalysis;
              setAnalyses((prev) => [result, ...prev.slice(0, 49)]);
              break;
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onerror = () => {
        // Don't show error if we're in expert mode - WebSocket is optional
        if (!useExpertMode) {
          setError("WebSocket unavailable. Using Expert API mode.");
        }
      };

      ws.onclose = () => {
        console.log("🔌 WebSocket disconnected");
        setConnected(false);
        
        // Reconnect after 5 seconds (less aggressive)
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 5000);
      };

      wsRef.current = ws;
    } catch (e) {
      console.error("Failed to connect WebSocket:", e);
      // Don't set error - Expert mode still works via REST API
    }
  }, [useExpertMode]);

  // Initialize WebSocket on mount
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Ping to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000);

    return () => clearInterval(pingInterval);
  }, []);

  // Request stats periodically
  useEffect(() => {
    const statsInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "get_stats" }));
      }
    }, 10000);

    return () => clearInterval(statsInterval);
  }, []);

  // Analyze transaction via REST API
  const analyzeTransaction = async (tx: { sender: string; recipient: string; amount: number }) => {
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sender: tx.sender,
          recipient: tx.recipient,
          amount: tx.amount,
          use_ai: true,
        }),
      });

      if (!response.ok) throw new Error("Analysis failed");

      const result = await response.json();
      setAnalyses((prev) => [result, ...prev.slice(0, 49)]);
    } catch (e) {
      console.error("Analysis error:", e);
      setError("Failed to analyze transaction");
    }
  };

  // Expert AI Oracle Analysis (v3.0 - 34 features, 5-model ensemble, Qwen3 LLM)
  const analyzeWithExpert = async (tx: { sender: string; recipient: string; amount: number }) => {
    try {
      const response = await fetch(`${API_URL}/expert/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sender: tx.sender,
          recipient: tx.recipient,
          amount: tx.amount,
        }),
      });

      if (!response.ok) throw new Error("Expert analysis failed");

      const verdict: ExpertVerdict = await response.json();
      
      // Convert ExpertVerdict (nested API response) to TransactionAnalysis format for display
      const analysis: TransactionAnalysis = {
        transaction_id: verdict.transaction.id,
        sender: verdict.transaction.sender,
        recipient: verdict.transaction.recipient,
        amount: verdict.transaction.amount,
        overall_score: verdict.risk_assessment.score,
        velocity_score: Math.round(verdict.model_scores.neural_ensemble * 0.3),
        amount_score: Math.round(verdict.model_scores.neural_ensemble * 0.3),
        pattern_score: Math.round(verdict.model_scores.typology_detector),
        graph_score: Math.round(verdict.model_scores.compliance_risk),
        timing_score: Math.round(verdict.model_scores.qwen3_llm * 0.5),
        ai_score: Math.round(verdict.model_scores.qwen3_llm),
        neural_net_score: verdict.model_scores.neural_ensemble,
        neural_net_confidence: verdict.risk_assessment.confidence,
        neural_net_risk_level: verdict.risk_assessment.level.toUpperCase(),
        neural_net_is_anomaly: verdict.risk_assessment.score >= 60,
        neural_net_explanation: verdict.explainability.explanation,
        risk_level: verdict.risk_assessment.level.toUpperCase() as "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
        approved: verdict.risk_assessment.score < 60,
        flagged: verdict.risk_assessment.score >= 60 && verdict.risk_assessment.score < 80,
        blocked: verdict.risk_assessment.score >= 80,
        alerts: verdict.explainability.key_risk_factors || [],
        ai_explanation: verdict.explainability.explanation,
        analysis_time_ms: verdict.performance.total_time_ms,
        signature: verdict.signature.signature || "",
        signer_address: verdict.signature.signer_address || "",
        timestamp: new Date(verdict.transaction.timestamp * 1000).toISOString(),
      };
      
      setAnalyses((prev) => [analysis, ...prev.slice(0, 49)]);
      setExpertVerdict(verdict);
    } catch (e) {
      console.error("Expert analysis error:", e);
      setError("Failed to analyze with Expert AI Oracle");
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">🛡️</span>
            AI Fraud Detection
          </h1>
          <p className="text-base-content/60 mt-1">
            {useExpertMode 
              ? "Expert AI Oracle v3.0 - 34 features, 5-model ensemble, Qwen3 LLM" 
              : "Real-time transaction monitoring powered by GPT-4 + ML"}
          </p>
        </div>
        <ConnectionStatus connected={connected} oracleAddress={oracleAddress} />
      </div>

      {/* Expert Metrics Banner */}
      {useExpertMode && expertMetrics && (
        <div className="alert bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-500/30">
          <div className="flex flex-wrap gap-6 w-full justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🧠</span>
              <div>
                <p className="font-bold text-purple-400">Expert AI Oracle</p>
                <p className="text-xs opacity-70">PayFlow-ExpertAI-v3.0.0-Ensemble</p>
              </div>
            </div>
            <div className="flex gap-6 text-center">
              <div>
                <p className="text-lg font-bold text-green-400">{expertMetrics.latency.avg_ms}ms</p>
                <p className="text-xs opacity-60">Avg Latency</p>
              </div>
              <div>
                <p className="text-lg font-bold text-blue-400">{expertMetrics.accuracy.accuracy_pct}%</p>
                <p className="text-xs opacity-60">Accuracy</p>
              </div>
              <div>
                <p className={`text-lg font-bold ${expertMetrics.judge_requirements.visa_latency_met ? "text-green-400" : "text-red-400"}`}>
                  {expertMetrics.judge_requirements.visa_latency_met ? "✓" : "✗"}
                </p>
                <p className="text-xs opacity-60">Visa &lt;300ms</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Expert Verdict Detail (when available) */}
      {expertVerdict && useExpertMode && (
        <div className="card bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-500/20">
          <div className="card-body">
            <h3 className="card-title text-purple-400">🧠 Latest Expert Verdict</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="stat bg-base-300 rounded-lg p-3">
                <div className="stat-title text-xs">Risk Score</div>
                <div className={`stat-value text-2xl ${
                  expertVerdict.risk_assessment.score >= 80 ? "text-red-500" :
                  expertVerdict.risk_assessment.score >= 60 ? "text-orange-500" :
                  expertVerdict.risk_assessment.score >= 40 ? "text-yellow-500" :
                  "text-green-500"
                }`}>
                  {expertVerdict.risk_assessment.score}/100 {expertVerdict.risk_assessment.emoji}
                </div>
              </div>
              <div className="stat bg-base-300 rounded-lg p-3">
                <div className="stat-title text-xs">Primary Typology</div>
                <div className="stat-value text-lg">{expertVerdict.analysis.primary_typology || "None"}</div>
              </div>
              <div className="stat bg-base-300 rounded-lg p-3">
                <div className="stat-title text-xs">Confidence</div>
                <div className="stat-value text-2xl text-blue-400">{(expertVerdict.risk_assessment.confidence * 100).toFixed(0)}%</div>
              </div>
              <div className="stat bg-base-300 rounded-lg p-3">
                <div className="stat-title text-xs">Total Latency</div>
                <div className="stat-value text-2xl text-green-400">{expertVerdict.performance.total_time_ms.toFixed(1)}ms</div>
              </div>
            </div>
            
            {/* Component Breakdown */}
            <div className="mt-4 grid grid-cols-5 gap-2 text-center text-xs">
              <div className="bg-base-300 rounded p-2">
                <p className="font-bold">{expertVerdict.performance.breakdown.feature_extraction.toFixed(1)}ms</p>
                <p className="opacity-60">Features</p>
              </div>
              <div className="bg-base-300 rounded p-2">
                <p className="font-bold">{expertVerdict.performance.breakdown.neural_ensemble.toFixed(1)}ms</p>
                <p className="opacity-60">Ensemble</p>
              </div>
              <div className="bg-base-300 rounded p-2">
                <p className="font-bold">{expertVerdict.performance.breakdown.typology_detection.toFixed(1)}ms</p>
                <p className="opacity-60">Typology</p>
              </div>
              <div className="bg-base-300 rounded p-2">
                <p className="font-bold">{expertVerdict.performance.breakdown.qwen3_llm.toFixed(1)}ms</p>
                <p className="opacity-60">Qwen3 LLM</p>
              </div>
              <div className="bg-base-300 rounded p-2">
                <p className="font-bold">{expertVerdict.performance.breakdown.compliance_check.toFixed(1)}ms</p>
                <p className="opacity-60">Compliance</p>
              </div>
            </div>

            {/* Model Scores */}
            <div className="mt-4 grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-gradient-to-b from-blue-900/30 to-blue-800/10 rounded p-2 border border-blue-500/20">
                <p className="font-bold text-blue-400">{expertVerdict.model_scores.neural_ensemble.toFixed(1)}</p>
                <p className="opacity-60">Neural</p>
              </div>
              <div className="bg-gradient-to-b from-purple-900/30 to-purple-800/10 rounded p-2 border border-purple-500/20">
                <p className="font-bold text-purple-400">{expertVerdict.model_scores.typology_detector.toFixed(1)}</p>
                <p className="opacity-60">Typology</p>
              </div>
              <div className="bg-gradient-to-b from-green-900/30 to-green-800/10 rounded p-2 border border-green-500/20">
                <p className="font-bold text-green-400">{expertVerdict.model_scores.qwen3_llm.toFixed(1)}</p>
                <p className="opacity-60">Qwen3</p>
              </div>
              <div className="bg-gradient-to-b from-red-900/30 to-red-800/10 rounded p-2 border border-red-500/20">
                <p className="font-bold text-red-400">{expertVerdict.model_scores.compliance_risk.toFixed(1)}</p>
                <p className="opacity-60">Compliance</p>
              </div>
            </div>

            {/* Explanation */}
            {expertVerdict.explainability.explanation && (
              <div className="mt-4 bg-base-300 p-3 rounded-lg">
                <p className="text-xs font-semibold mb-1">🤖 AI Explanation</p>
                <p className="text-sm">{expertVerdict.explainability.explanation}</p>
              </div>
            )}

            {/* Risk Factors */}
            {expertVerdict.explainability.key_risk_factors && expertVerdict.explainability.key_risk_factors.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold mb-2">⚠️ Key Risk Factors</p>
                <div className="flex flex-wrap gap-1">
                  {expertVerdict.explainability.key_risk_factors.map((factor, i) => (
                    <span key={i} className="badge badge-warning badge-sm">{factor}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Detected Typologies */}
            {expertVerdict.analysis.detected_typologies && expertVerdict.analysis.detected_typologies.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold mb-2">🎯 Detected Typologies</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {expertVerdict.analysis.detected_typologies.map((typ, i) => (
                    <div key={i} className={`p-2 rounded border ${
                      typ.severity === "CRITICAL" ? "bg-red-900/20 border-red-500/30" :
                      typ.severity === "HIGH" ? "bg-orange-900/20 border-orange-500/30" :
                      typ.severity === "MEDIUM" ? "bg-yellow-900/20 border-yellow-500/30" :
                      "bg-green-900/20 border-green-500/30"
                    }`}>
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-sm">{typ.typology_name}</span>
                        <span className={`badge badge-sm ${
                          typ.severity === "CRITICAL" ? "badge-error" :
                          typ.severity === "HIGH" ? "badge-warning" :
                          typ.severity === "MEDIUM" ? "badge-info" : "badge-success"
                        }`}>{(typ.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <p className="text-xs opacity-70 mt-1">{typ.triggered_rules?.join(", ") || "Pattern detected"}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {expertVerdict.explainability.recommendations && expertVerdict.explainability.recommendations.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-semibold mb-2">📋 Recommendations</p>
                <ul className="text-xs space-y-1">
                  {expertVerdict.explainability.recommendations.slice(0, 3).map((rec, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-green-500">▸</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Alert - only show if not in expert mode or if it's a critical error */}
      {error && !useExpertMode && (
        <div className="alert alert-warning">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
          </svg>
          <div className="flex-1">
            <span>{error}</span>
            <p className="text-xs opacity-70 mt-1">
              Make sure the AI Oracle API is running: <code className="bg-base-300 px-1 rounded">python -m uvicorn api:app --port 8000</code>
            </p>
          </div>
        </div>
      )}

      {/* Expert Mode Info Banner when WebSocket is disconnected but Expert mode works */}
      {!connected && useExpertMode && !error && (
        <div className="alert bg-blue-900/20 border border-blue-500/30">
          <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-blue-300">Expert Mode Active - Use the &quot;🧠 Expert Analysis&quot; button or Quick Tests to analyze transactions via REST API</span>
        </div>
      )}

      {/* Stats */}
      {stats && <StatsCard stats={stats} />}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Transaction Analyzer */}
        <div className="lg:col-span-1">
          <TransactionAnalyzer 
            onAnalyze={analyzeTransaction} 
            onExpertAnalyze={analyzeWithExpert}
            useExpert={useExpertMode}
            onToggleMode={() => setUseExpertMode(!useExpertMode)}
          />
          
          {/* Oracle Info */}
          {stats && (
            <div className="card bg-base-200 mt-4">
              <div className="card-body">
                <h3 className="card-title text-lg">🤖 Oracle Status</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="opacity-60">GPT-4 Enabled</span>
                    <span className={stats.gpt4_enabled ? "text-success" : "text-error"}>
                      {stats.gpt4_enabled ? "Yes" : "No"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="opacity-60">ML Model Trained</span>
                    <span className={stats.ml_trained ? "text-success" : "text-warning"}>
                      {stats.ml_trained ? "Yes" : "Training..."}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="opacity-60">Active Profiles</span>
                    <span>{stats.total_profiles}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="opacity-60">Blacklisted</span>
                    <span className="text-error">{stats.blacklist_size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="opacity-60">WebSocket Clients</span>
                    <span>{stats.websocket_connections}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Transaction Feed */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">📊 Live Transaction Feed</h2>
            <span className="badge badge-ghost">{analyses.length} transactions</span>
          </div>
          
          {analyses.length === 0 ? (
            <div className="card bg-base-200">
              <div className="card-body items-center text-center py-12">
                <span className="text-6xl mb-4">📡</span>
                <h3 className="text-lg font-semibold">Waiting for transactions...</h3>
                <p className="text-base-content/60">
                  Analyze a transaction or wait for real-time updates
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4 max-h-[800px] overflow-y-auto pr-2">
              {analyses.map((analysis, i) => (
                <TransactionCard key={`${analysis.transaction_id}-${i}`} analysis={analysis} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
