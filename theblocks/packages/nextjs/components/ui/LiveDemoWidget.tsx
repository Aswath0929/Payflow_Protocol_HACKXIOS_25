"use client";

import { useEffect, useState } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// LIVE DEMO WIDGETS - Animated demo visualizations for different pages
// ═══════════════════════════════════════════════════════════════════════════

interface DemoStep {
  label: string;
  icon: string;
  status?: "pending" | "active" | "success" | "complete";
}

// ============================================
// PAYMENT FLOW DEMO - For Dashboard
// ============================================

export function PaymentFlowDemo() {
  const [step, setStep] = useState(0);
  const [amount, setAmount] = useState(0);

  const steps: DemoStep[] = [
    { label: "Compliance Check", icon: "🔍" },
    { label: "KYC Verified", icon: "✓" },
    { label: "AML Cleared", icon: "🛡️" },
    { label: "FX Rate Locked", icon: "📈" },
    { label: "Escrow Funded", icon: "🔒" },
    { label: "Settlement", icon: "⚡" },
    { label: "Complete", icon: "💎" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep(prev => (prev + 1) % steps.length);
    }, 1200); // Faster animation for real-time feel
    return () => clearInterval(interval);
  }, [steps.length]);

  useEffect(() => {
    if (step === 5) {
      const duration = 800;
      const start = Date.now();
      const animate = () => {
        const elapsed = Date.now() - start;
        const progress = Math.min(elapsed / duration, 1);
        setAmount(Math.floor(progress * 2500000));
        if (progress < 1) requestAnimationFrame(animate);
      };
      animate();
    }
  }, [step]);

  return (
    <div className="relative bg-[#0d0d15] border border-white/10 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
          </div>
          <span className="text-xs text-zinc-500 font-medium">Live Payment Flow</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
          </span>
          <span className="text-xs text-green-400">Live</span>
        </div>
      </div>

      {/* Content */}
      <div className="p-5 space-y-4">
        {/* Transfer Route */}
        <div className="flex items-center justify-between p-3 rounded-xl bg-white/[0.03] border border-white/5">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-xs">
              🏦
            </div>
            <div className="text-xs">
              <div className="text-zinc-500">From</div>
              <div className="font-medium text-white">JPMorgan</div>
            </div>
          </div>
          <div className="text-violet-400 animate-pulse text-lg">→</div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-right">
              <div className="text-zinc-500">To</div>
              <div className="font-medium text-white">HSBC</div>
            </div>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center text-xs">
              🏛️
            </div>
          </div>
        </div>

        {/* Amount */}
        <div className="text-center py-2">
          <div className="text-xs text-zinc-500 mb-1">Settlement Amount</div>
          <div className="text-2xl font-bold font-mono text-white">
            ${(step >= 5 ? amount : 2500000).toLocaleString()}
            <span className="text-sm text-zinc-500 ml-1">USDC</span>
          </div>
          <div className="text-xs text-cyan-400 mt-1">≈ 8.2 seconds</div>
        </div>

        {/* Progress */}
        <div className="space-y-2">
          {steps.slice(0, Math.min(step + 1, steps.length)).map((s, i) => (
            <div
              key={i}
              className={`flex items-center gap-2 p-2 rounded-lg text-xs transition-all duration-300 ${
                i === step
                  ? "bg-violet-500/10 border border-violet-500/30"
                  : i < step
                    ? "bg-green-500/5 border border-green-500/20"
                    : "bg-white/5"
              }`}
            >
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center ${
                  i < step
                    ? "bg-green-500/20 text-green-400"
                    : i === step
                      ? "bg-violet-500/20 text-violet-400"
                      : "bg-white/10"
                }`}
              >
                {s.icon}
              </div>
              <span className={i <= step ? "text-white" : "text-zinc-500"}>{s.label}</span>
              {i < step && <span className="ml-auto text-green-400 text-[10px]">✓</span>}
              {i === step && <span className="ml-auto text-violet-400 animate-pulse text-[10px]">Processing...</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================
// ORACLE CONSENSUS DEMO - For Oracle Dashboard
// ============================================

export function OracleConsensusDemo() {
  const [phase, setPhase] = useState(0);
  const [prices, setPrices] = useState({ chainlink: 0, pyth: 0, consensus: 0 });

  const basePrice = 3245.67;
  const oracles = [
    { name: "Chainlink", icon: "🔗", color: "from-blue-500 to-blue-600", variance: 0.12 },
    { name: "Pyth", icon: "🔮", color: "from-purple-500 to-purple-600", variance: -0.08 },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(prev => (prev + 1) % 5);
    }, 1200); // Faster refresh for real-time feel
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (phase >= 1 && phase <= 2) {
      const oracle = oracles[phase - 1];
      const targetPrice = basePrice * (1 + oracle.variance / 100);
      const duration = 600;
      const start = Date.now();
      const animate = () => {
        const elapsed = Date.now() - start;
        const progress = Math.min(elapsed / duration, 1);
        const currentPrice = basePrice + (targetPrice - basePrice) * progress;
        setPrices(prev => ({
          ...prev,
          [phase === 1 ? "chainlink" : "pyth"]: currentPrice,
        }));
        if (progress < 1) requestAnimationFrame(animate);
      };
      animate();
    } else if (phase === 3) {
      // Calculate consensus (weighted average: 60% Chainlink, 40% Pyth)
      const consensusPrice = prices.chainlink * 0.6 + prices.pyth * 0.4;
      setPrices(prev => ({ ...prev, consensus: consensusPrice }));
    } else if (phase === 4) {
      // Reset for next cycle
      setTimeout(() => {
        setPrices({ chainlink: 0, pyth: 0, consensus: 0 });
      }, 1500);
    }
  }, [phase]);

  const phases = [
    { label: "Fetching Prices", icon: "🔄" },
    { label: "Chainlink Response", icon: "🔗" },
    { label: "Pyth Response", icon: "🔮" },
    { label: "Weighted Consensus", icon: "🧠" },
    { label: "Price Verified", icon: "✓" },
  ];

  return (
    <div className="relative bg-[#0d0d15] border border-white/10 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
          </div>
          <span className="text-xs text-zinc-500 font-medium">Dual-Oracle Consensus</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
          </span>
          <span className="text-xs text-cyan-400">ETH/USD</span>
        </div>
      </div>

      {/* Content */}
      <div className="p-5 space-y-4">
        {/* Current Phase */}
        <div className="text-center p-3 rounded-xl bg-gradient-to-r from-violet-500/10 to-cyan-500/10 border border-white/5">
          <div className="text-2xl mb-1">{phases[phase]?.icon}</div>
          <div className="text-sm font-medium text-white">{phases[phase]?.label}</div>
        </div>

        {/* Oracle Prices */}
        <div className="space-y-2">
          {oracles.map((oracle, i) => {
            const priceKey = oracle.name.toLowerCase() as "chainlink" | "pyth";
            const price = prices[priceKey];
            const isActive = phase === i + 1;
            const isDone = phase > i + 1;

            return (
              <div
                key={oracle.name}
                className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
                  isActive
                    ? "bg-gradient-to-r from-violet-500/10 to-transparent border border-violet-500/30"
                    : isDone
                      ? "bg-green-500/5 border border-green-500/20"
                      : "bg-white/[0.02] border border-white/5"
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-8 h-8 rounded-lg bg-gradient-to-br ${oracle.color} flex items-center justify-center`}
                  >
                    {oracle.icon}
                  </div>
                  <span className="text-sm font-medium text-white">{oracle.name}</span>
                </div>
                <div className="text-right">
                  {price > 0 ? (
                    <>
                      <div className="text-sm font-mono text-white">${price.toFixed(2)}</div>
                      <div className={`text-[10px] ${oracle.variance >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {oracle.variance >= 0 ? "+" : ""}
                        {oracle.variance}%
                      </div>
                    </>
                  ) : (
                    <div className="text-xs text-zinc-500">{isActive ? "Fetching..." : "Waiting"}</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Consensus Result */}
        {phase >= 3 && prices.consensus > 0 && (
          <div className="p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 text-center animate-pulse">
            <div className="text-xs text-green-400 mb-1">🧠 Weighted Consensus Price</div>
            <div className="text-2xl font-bold font-mono text-green-400">${prices.consensus.toFixed(2)}</div>
            <div className="text-[10px] text-zinc-400 mt-1">2/2 oracles • 60/40 weight • 99.8% confidence</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// SECURITY SHIELD DEMO - For Attack Simulator
// ============================================

export function SecurityShieldDemo() {
  const [attackPhase, setAttackPhase] = useState(0);
  const [blocked, setBlocked] = useState(0);

  const attacks = [
    { type: "Flash Loan Attack", icon: "⚡", severity: "Critical" },
    { type: "Price Manipulation", icon: "📉", severity: "High" },
    { type: "Sandwich Attack", icon: "🥪", severity: "Medium" },
    { type: "Oracle Spoofing", icon: "👻", severity: "High" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setAttackPhase(prev => (prev + 1) % 8);
    }, 1000); // Faster for real-time security monitoring feel
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (attackPhase >= 1 && attackPhase <= 4) {
      setBlocked(prev => prev + 1);
    } else if (attackPhase === 0) {
      setBlocked(0);
    }
  }, [attackPhase]);

  const defenseSteps = [
    { label: "Monitoring", icon: "👁️" },
    { label: "Threat Detected", icon: "🚨" },
    { label: "Analyzing Pattern", icon: "🔍" },
    { label: "Defense Activated", icon: "🛡️" },
    { label: "Attack Blocked", icon: "🚫" },
    { label: "System Secured", icon: "✓" },
  ];

  return (
    <div className="relative bg-[#0d0d15] border border-white/10 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
          </div>
          <span className="text-xs text-zinc-500 font-medium">Security Shield Demo</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`relative flex h-2 w-2 ${attackPhase > 0 && attackPhase < 5 ? "animate-pulse" : ""}`}>
            <span
              className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${
                attackPhase > 0 && attackPhase < 5 ? "bg-red-400 animate-ping" : "bg-green-400"
              }`}
            />
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                attackPhase > 0 && attackPhase < 5 ? "bg-red-500" : "bg-green-500"
              }`}
            />
          </span>
          <span className={`text-xs ${attackPhase > 0 && attackPhase < 5 ? "text-red-400" : "text-green-400"}`}>
            {attackPhase > 0 && attackPhase < 5 ? "Threat Active" : "Secure"}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="p-5 space-y-4">
        {/* Stats */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-white/[0.02] border border-white/5 text-center">
            <div className="text-xl font-bold text-green-400">{blocked}</div>
            <div className="text-[10px] text-zinc-500">Attacks Blocked</div>
          </div>
          <div className="p-3 rounded-lg bg-white/[0.02] border border-white/5 text-center">
            <div className="text-xl font-bold text-cyan-400">100%</div>
            <div className="text-[10px] text-zinc-500">Success Rate</div>
          </div>
        </div>

        {/* Current Attack */}
        {attackPhase > 0 && attackPhase <= 4 && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 animate-pulse">
            <div className="flex items-center gap-2">
              <span className="text-xl">{attacks[attackPhase - 1]?.icon}</span>
              <div>
                <div className="text-sm font-medium text-red-400">{attacks[attackPhase - 1]?.type}</div>
                <div className="text-[10px] text-red-400/70">Severity: {attacks[attackPhase - 1]?.severity}</div>
              </div>
            </div>
          </div>
        )}

        {/* Defense Progress */}
        <div className="space-y-2">
          {defenseSteps
            .slice(0, Math.min(attackPhase > 0 ? attackPhase + 1 : 1, defenseSteps.length))
            .map((step, i) => (
              <div
                key={i}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs transition-all duration-300 ${
                  i === attackPhase
                    ? "bg-cyan-500/10 border border-cyan-500/30"
                    : i < attackPhase
                      ? "bg-green-500/5 border border-green-500/20"
                      : "bg-white/5 border border-transparent"
                }`}
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center ${
                    i < attackPhase
                      ? "bg-green-500/20 text-green-400"
                      : i === attackPhase
                        ? "bg-cyan-500/20 text-cyan-400"
                        : "bg-white/10"
                  }`}
                >
                  {step.icon}
                </div>
                <span className={i <= attackPhase ? "text-white" : "text-zinc-500"}>{step.label}</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

// ============================================
// COMPLIANCE VERIFICATION DEMO - For Compliance Tab
// ============================================

export function ComplianceDemo() {
  const [phase, setPhase] = useState(0);

  const checks = [
    { label: "Identity Verification", icon: "🪪", time: "2.1s" },
    { label: "KYC/AML Check", icon: "🔍", time: "1.8s" },
    { label: "Sanctions Screening", icon: "🌍", time: "0.9s" },
    { label: "Risk Assessment", icon: "📊", time: "1.2s" },
    { label: "Compliance Score", icon: "✓", time: "0.5s" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase(prev => (prev + 1) % (checks.length + 2));
    }, 900); // Faster compliance checks
    return () => clearInterval(interval);
  }, [checks.length]);

  return (
    <div className="relative bg-[#0d0d15] border border-white/10 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
          </div>
          <span className="text-xs text-zinc-500 font-medium">Real-time Compliance</span>
        </div>
        <div className="text-xs text-violet-400">Institutional Tier</div>
      </div>

      {/* Content */}
      <div className="p-5 space-y-4">
        {/* Progress Steps */}
        <div className="space-y-2">
          {checks.map((check, i) => {
            const isActive = phase === i;
            const isDone = phase > i;

            return (
              <div
                key={i}
                className={`flex items-center justify-between p-3 rounded-lg transition-all duration-300 ${
                  isActive
                    ? "bg-violet-500/10 border border-violet-500/30"
                    : isDone
                      ? "bg-green-500/5 border border-green-500/20"
                      : "bg-white/[0.02] border border-white/5"
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                      isDone ? "bg-green-500/20" : isActive ? "bg-violet-500/20" : "bg-white/10"
                    }`}
                  >
                    {check.icon}
                  </div>
                  <span className={`text-sm ${isDone || isActive ? "text-white" : "text-zinc-500"}`}>
                    {check.label}
                  </span>
                </div>
                <div className="text-right">
                  {isDone ? (
                    <span className="text-xs text-green-400">✓ {check.time}</span>
                  ) : isActive ? (
                    <span className="text-xs text-violet-400 animate-pulse">Verifying...</span>
                  ) : (
                    <span className="text-xs text-zinc-600">Pending</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Final Score */}
        {phase >= checks.length && (
          <div className="p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 text-center">
            <div className="text-xs text-green-400 mb-1">Compliance Score</div>
            <div className="text-3xl font-bold text-green-400">98.5%</div>
            <div className="text-[10px] text-zinc-400 mt-1">✓ Approved for Unlimited Transfers</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// SETTLEMENT SPEED DEMO
// ============================================

export function SettlementSpeedDemo() {
  const [seconds, setSeconds] = useState(0);
  const [isRunning, setIsRunning] = useState(true);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setSeconds(prev => {
        if (prev >= 12) {
          setIsRunning(false);
          setTimeout(() => {
            setSeconds(0);
            setIsRunning(true);
          }, 3000);
          return 12;
        }
        return prev + 0.1;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isRunning]);

  const progress = Math.min(seconds / 12, 1) * 100;

  return (
    <div className="relative bg-[#0d0d15] border border-white/10 rounded-2xl overflow-hidden p-5">
      <div className="text-center mb-4">
        <div className="text-xs text-zinc-500 mb-2">Cross-Border Settlement Time</div>
        <div className="text-5xl font-bold font-mono">
          <span className={seconds >= 12 ? "text-green-400" : "text-white"}>{seconds.toFixed(1)}</span>
          <span className="text-xl text-zinc-500">s</span>
        </div>
        <div className="text-xs mt-2 overflow-hidden">
          <span
            className="inline-block transition-all duration-500"
            style={{
              opacity: seconds >= 12 ? 0 : 0.5,
              transform: seconds >= 12 ? "translateY(-10px)" : "translateY(0)",
              color: "#71717a",
            }}
          >
            vs 3-5 days (traditional)
          </span>
          <span
            className="inline-block transition-all duration-700"
            style={{
              opacity: seconds >= 12 ? 1 : 0,
              transform: seconds >= 12 ? "translateY(0)" : "translateY(10px)",
              fontWeight: seconds >= 12 ? 600 : 400,
              marginLeft: seconds >= 12 ? "0" : "-100%",
              position: seconds >= 12 ? "relative" : "absolute",
              color: "#4ade80",
            }}
          >
            250x faster than SWIFT
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-100 ${
            seconds >= 12
              ? "bg-gradient-to-r from-green-500 to-emerald-500"
              : "bg-gradient-to-r from-violet-500 to-cyan-500"
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Milestones */}
      <div className="flex justify-between mt-2 text-[10px] text-zinc-500">
        <span>Initiated</span>
        <span className={seconds >= 4 ? "text-green-400" : ""}>Compliance ✓</span>
        <span className={seconds >= 8 ? "text-green-400" : ""}>Settled ✓</span>
        <span className={seconds >= 12 ? "text-green-400" : ""}>Complete</span>
      </div>

      {seconds >= 12 && (
        <div className="mt-4 text-center text-green-400 text-sm animate-pulse">✓ $10,000,000 settled successfully</div>
      )}
    </div>
  );
}
