"use client";

import { useState, useEffect, useCallback } from "react";
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
  risk_level: "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  approved: boolean;
  flagged: boolean;
  blocked: boolean;
  explanations: string[];
  analysis_time_ms: number;
}

interface WalletRisk {
  address: string;
  risk_score: number;
  risk_level: string;
  peak_risk_score: number;
  is_blacklisted: boolean;
  is_whitelisted: boolean;
  transaction_count: number;
  total_volume: number;
  avg_amount: number;
}

interface Statistics {
  total_analyses: number;
  total_blocked: number;
  total_flagged: number;
  total_profiles: number;
  blacklist_size: number;
  whitelist_size: number;
  model_version: string;
  block_rate: number;
  flag_rate: number;
}

// Risk level colors
const riskColors = {
  SAFE: "bg-green-500",
  LOW: "bg-blue-500",
  MEDIUM: "bg-yellow-500",
  HIGH: "bg-orange-500",
  CRITICAL: "bg-red-500",
};

const riskTextColors = {
  SAFE: "text-green-500",
  LOW: "text-blue-500",
  MEDIUM: "text-yellow-500",
  HIGH: "text-orange-500",
  CRITICAL: "text-red-500",
};

// Score gauge component
const ScoreGauge = ({
  label,
  score,
  size = "sm",
}: {
  label: string;
  score: number;
  size?: "sm" | "lg";
}) => {
  const getColor = (s: number) => {
    if (s <= 20) return "stroke-green-500";
    if (s <= 40) return "stroke-blue-500";
    if (s <= 60) return "stroke-yellow-500";
    if (s <= 80) return "stroke-orange-500";
    return "stroke-red-500";
  };

  const circumference = 2 * Math.PI * 40;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const sizeClass = size === "lg" ? "w-24 h-24" : "w-16 h-16";
  const textSize = size === "lg" ? "text-lg font-bold" : "text-xs";

  return (
    <div className="flex flex-col items-center">
      <div className={`relative ${sizeClass}`}>
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" strokeWidth="8" className="text-gray-700" />
          <circle
            cx="50"
            cy="50"
            r="40"
            fill="none"
            strokeWidth="8"
            strokeLinecap="round"
            className={`transition-all duration-500 ${getColor(score)}`}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
          />
        </svg>
        <span className={`absolute inset-0 flex items-center justify-center ${textSize}`}>{score}</span>
      </div>
      <span className="text-xs mt-1 text-gray-400">{label}</span>
    </div>
  );
};

// Alert card component
const AlertCard = ({ analysis }: { analysis: TransactionAnalysis }) => {
  return (
    <div
      className={`rounded-lg border p-4 ${
        analysis.blocked
          ? "border-red-500 bg-red-500/10"
          : analysis.flagged
            ? "border-yellow-500 bg-yellow-500/10"
            : "border-green-500 bg-green-500/10"
      }`}
    >
      <div className="flex justify-between items-start mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 rounded text-xs font-bold ${
                analysis.blocked ? "bg-red-500" : analysis.flagged ? "bg-yellow-500 text-black" : "bg-green-500"
              }`}
            >
              {analysis.blocked ? "BLOCKED" : analysis.flagged ? "FLAGGED" : "APPROVED"}
            </span>
            <span className={`text-xs ${riskTextColors[analysis.risk_level]}`}>{analysis.risk_level}</span>
          </div>
          <p className="text-xs text-gray-400 mt-1 font-mono">{analysis.transaction_id.slice(0, 20)}...</p>
        </div>
        <ScoreGauge label="Risk" score={analysis.overall_score} size="lg" />
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm mb-3">
        <div>
          <span className="text-gray-400">From:</span>
          <p className="font-mono text-xs truncate">{analysis.sender}</p>
        </div>
        <div>
          <span className="text-gray-400">To:</span>
          <p className="font-mono text-xs truncate">{analysis.recipient}</p>
        </div>
        <div>
          <span className="text-gray-400">Amount:</span>
          <p className="font-bold">${analysis.amount.toLocaleString()}</p>
        </div>
        <div>
          <span className="text-gray-400">Analysis Time:</span>
          <p>{analysis.analysis_time_ms.toFixed(2)}ms</p>
        </div>
      </div>

      <div className="flex justify-between gap-2 mb-3">
        <ScoreGauge label="Velocity" score={analysis.velocity_score} />
        <ScoreGauge label="Amount" score={analysis.amount_score} />
        <ScoreGauge label="Pattern" score={analysis.pattern_score} />
        <ScoreGauge label="Graph" score={analysis.graph_score} />
        <ScoreGauge label="Timing" score={analysis.timing_score} />
      </div>

      {analysis.explanations.length > 0 && (
        <div className="border-t border-gray-700 pt-2">
          <p className="text-xs text-gray-400 mb-1">Analysis Notes:</p>
          <ul className="text-xs space-y-1">
            {analysis.explanations.slice(0, 3).map((exp, i) => (
              <li key={i} className="text-gray-300">
                • {exp}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// Main Dashboard Component
export default function FraudDashboard() {
  const { address } = useAccount();
  const [stats, setStats] = useState<Statistics | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<TransactionAnalysis[]>([]);
  const [walletRisk, setWalletRisk] = useState<WalletRisk | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiUrl] = useState("http://localhost:8000");

  // Form state for testing
  const [testForm, setTestForm] = useState({
    sender: "",
    recipient: "",
    amount: 1000,
  });

  // Fetch statistics
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${apiUrl}/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.log("API not available:", e);
    }
  }, [apiUrl]);

  // Fetch wallet risk
  const fetchWalletRisk = useCallback(async () => {
    if (!address) return;
    try {
      const res = await fetch(`${apiUrl}/wallet/${address}`);
      if (res.ok) {
        const data = await res.json();
        setWalletRisk(data);
      }
    } catch (e) {
      console.log("Error fetching wallet risk:", e);
    }
  }, [address, apiUrl]);

  // Analyze transaction
  const analyzeTransaction = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transaction_id: `0x${Date.now().toString(16)}`,
          sender: testForm.sender || address || "0xTest",
          recipient: testForm.recipient || "0xRecipient",
          amount: testForm.amount,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setRecentAnalyses((prev) => [data, ...prev].slice(0, 10));
        await fetchStats();
      }
    } catch (e) {
      console.log("Error analyzing transaction:", e);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchStats();
    fetchWalletRisk();
  }, [fetchStats, fetchWalletRisk]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">🛡️ PayFlow AI Fraud Detection</h1>
          <p className="text-gray-400">Real-time ML-based risk scoring for stablecoin transactions</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Total Analyses</p>
            <p className="text-2xl font-bold">{stats?.total_analyses || 0}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Blocked</p>
            <p className="text-2xl font-bold text-red-500">{stats?.total_blocked || 0}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Flagged</p>
            <p className="text-2xl font-bold text-yellow-500">{stats?.total_flagged || 0}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Block Rate</p>
            <p className="text-2xl font-bold">{stats?.block_rate?.toFixed(1) || 0}%</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Blacklisted</p>
            <p className="text-2xl font-bold">{stats?.blacklist_size || 0}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs">Model Version</p>
            <p className="text-sm font-mono truncate">{stats?.model_version || "N/A"}</p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Transaction Analyzer */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4">🔍 Analyze Transaction</h2>

            <div className="space-y-4">
              <div>
                <label className="text-xs text-gray-400">Sender Address</label>
                <input
                  type="text"
                  placeholder={address || "0x..."}
                  className="w-full bg-gray-700 rounded px-3 py-2 text-sm font-mono"
                  value={testForm.sender}
                  onChange={(e) => setTestForm({ ...testForm, sender: e.target.value })}
                />
              </div>

              <div>
                <label className="text-xs text-gray-400">Recipient Address</label>
                <input
                  type="text"
                  placeholder="0x..."
                  className="w-full bg-gray-700 rounded px-3 py-2 text-sm font-mono"
                  value={testForm.recipient}
                  onChange={(e) => setTestForm({ ...testForm, recipient: e.target.value })}
                />
              </div>

              <div>
                <label className="text-xs text-gray-400">Amount (USD)</label>
                <input
                  type="number"
                  className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
                  value={testForm.amount}
                  onChange={(e) => setTestForm({ ...testForm, amount: parseFloat(e.target.value) })}
                />
              </div>

              <button
                onClick={analyzeTransaction}
                disabled={isLoading}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded py-2 font-bold transition-colors"
              >
                {isLoading ? "Analyzing..." : "Analyze Transaction"}
              </button>
            </div>

            {/* Quick Test Buttons */}
            <div className="mt-4 pt-4 border-t border-gray-700">
              <p className="text-xs text-gray-400 mb-2">Quick Tests:</p>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setTestForm({ sender: "0xAlice", recipient: "0xBob", amount: 100 })}
                  className="px-3 py-1 bg-green-600/20 border border-green-600 rounded text-xs"
                >
                  ✓ Normal
                </button>
                <button
                  onClick={() => setTestForm({ sender: "0xTest", recipient: "0xReceiver", amount: 9999 })}
                  className="px-3 py-1 bg-yellow-600/20 border border-yellow-600 rounded text-xs"
                >
                  ⚠ Structuring
                </button>
                <button
                  onClick={() => setTestForm({ sender: "0xTest", recipient: "0xBadActor", amount: 50000 })}
                  className="px-3 py-1 bg-red-600/20 border border-red-600 rounded text-xs"
                >
                  ✗ High Risk
                </button>
              </div>
            </div>
          </div>

          {/* Your Wallet Risk */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4">👤 Your Wallet Risk</h2>

            {address ? (
              walletRisk ? (
                <div className="space-y-4">
                  <div className="flex justify-center mb-4">
                    <ScoreGauge label="Risk Score" score={walletRisk.risk_score} size="lg" />
                  </div>

                  <div className="text-center mb-4">
                    <span
                      className={`px-3 py-1 rounded ${riskColors[walletRisk.risk_level as keyof typeof riskColors] || "bg-gray-500"}`}
                    >
                      {walletRisk.risk_level}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-gray-700 rounded p-2">
                      <p className="text-xs text-gray-400">Transactions</p>
                      <p className="font-bold">{walletRisk.transaction_count}</p>
                    </div>
                    <div className="bg-gray-700 rounded p-2">
                      <p className="text-xs text-gray-400">Volume</p>
                      <p className="font-bold">${walletRisk.total_volume?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-gray-700 rounded p-2">
                      <p className="text-xs text-gray-400">Peak Risk</p>
                      <p className="font-bold">{walletRisk.peak_risk_score}</p>
                    </div>
                    <div className="bg-gray-700 rounded p-2">
                      <p className="text-xs text-gray-400">Status</p>
                      <p className="font-bold">
                        {walletRisk.is_blacklisted ? (
                          <span className="text-red-500">Blacklisted</span>
                        ) : walletRisk.is_whitelisted ? (
                          <span className="text-green-500">Whitelisted</span>
                        ) : (
                          <span className="text-gray-400">Normal</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-400">
                  <p>No risk profile yet</p>
                  <p className="text-xs mt-2">Analyze a transaction to create your profile</p>
                </div>
              )
            ) : (
              <div className="text-center text-gray-400">
                <p>Connect your wallet to see risk profile</p>
              </div>
            )}
          </div>

          {/* System Health */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-bold mb-4">📊 System Health</h2>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">AI Model</span>
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                  Online
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">FraudOracle</span>
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                  Deployed
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Profiles Tracked</span>
                <span>{stats?.total_profiles || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Avg Analysis Time</span>
                <span>&lt;50ms</span>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-700">
              <h3 className="text-sm font-bold mb-2">Risk Distribution</h3>
              <div className="flex gap-1">
                <div className="flex-1 h-2 bg-green-500 rounded-l" style={{ width: "40%" }}></div>
                <div className="flex-1 h-2 bg-blue-500" style={{ width: "30%" }}></div>
                <div className="flex-1 h-2 bg-yellow-500" style={{ width: "15%" }}></div>
                <div className="flex-1 h-2 bg-orange-500" style={{ width: "10%" }}></div>
                <div className="flex-1 h-2 bg-red-500 rounded-r" style={{ width: "5%" }}></div>
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Safe</span>
                <span>Low</span>
                <span>Med</span>
                <span>High</span>
                <span>Crit</span>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Analyses */}
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">📋 Recent Analyses</h2>

          {recentAnalyses.length > 0 ? (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recentAnalyses.map((analysis, i) => (
                <AlertCard key={i} analysis={analysis} />
              ))}
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
              <p className="text-4xl mb-2">🔍</p>
              <p>No analyses yet</p>
              <p className="text-sm mt-2">Use the analyzer above to test transactions</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 pt-4 border-t border-gray-700 text-center text-gray-400 text-sm">
          <p>PayFlow AI Fraud Detection • Hackxios 2K25</p>
          <p className="text-xs mt-1">Making stablecoins safer than traditional banking</p>
        </div>
      </div>
    </div>
  );
}
