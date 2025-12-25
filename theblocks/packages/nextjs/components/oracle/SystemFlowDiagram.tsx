"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

interface StepData {
  id: number;
  title: string;
  icon: string;
  description: string;
  details: string[];
  color: string;
  glowColor: string;
}

interface OracleData {
  name: string;
  status: "live" | "faulty" | "backup" | "fallback";
  icon: string;
  color: string;
}

const oracles: OracleData[] = [
  { name: "Chainlink", status: "live", icon: "⛓️", color: "from-blue-500 to-blue-600" },
  { name: "Pyth", status: "live", icon: "🔮", color: "from-purple-500 to-purple-600" },
  { name: "SyncedFeed", status: "backup", icon: "🔄", color: "from-cyan-500 to-teal-500" },
];

const steps: StepData[] = [
  {
    id: 1,
    title: "STALENESS CHECK",
    icon: "⏱️",
    description: "Filter out outdated oracle data",
    details: [
      "Chainlink: MAX_STALENESS = 3600s (1 hour)",
      "Pyth: MAX_STALENESS = 60s (real-time)",
      "SyncedFeed: Backup when primary stale",
    ],
    color: "from-yellow-400 to-orange-500",
    glowColor: "yellow",
  },
  {
    id: 2,
    title: "DEVIATION CHECK",
    icon: "📊",
    description: "Detect manipulation attempts",
    details: [
      "Calculate median of all valid prices",
      "Reject any price deviating >5% from median",
      "Prevents flash loan & sandwich attacks",
    ],
    color: "from-cyan-400 to-blue-500",
    glowColor: "cyan",
  },
  {
    id: 3,
    title: "WEIGHTED CONSENSUS",
    icon: "🛡️",
    description: "Fault-tolerant price calculation",
    details: [
      "Chainlink: 60% weight (industry standard)",
      "Pyth: 40% weight (real-time)",
      "Resistant to single oracle failure",
    ],
    color: "from-purple-400 to-pink-500",
    glowColor: "purple",
  },
  {
    id: 4,
    title: "CONFIDENCE WEIGHTING",
    icon: "⚖️",
    description: "Weighted average by reliability",
    details: [
      "Higher weight: fresh data, reliable oracles",
      "Lower weight: stale data, flagged oracles",
      "Final Price = Σ(price × weight) / Σ(weights)",
    ],
    color: "from-green-400 to-emerald-500",
    glowColor: "green",
  },
];

// Animated particle component - uses fixed offset to avoid hydration mismatch
function Particle({ delay, offset }: { delay: number; offset: number }) {
  return (
    <motion.div
      className="absolute w-1 h-1 bg-cyan-400 rounded-full"
      initial={{ opacity: 0, y: 0 }}
      animate={{
        opacity: [0, 1, 0],
        y: [0, -100],
        x: [0, offset],
      }}
      transition={{
        duration: 2,
        delay,
        repeat: Infinity,
        ease: "easeOut",
      }}
    />
  );
}

// Data flow line animation
function DataFlowLine({ isActive }: { isActive: boolean }) {
  return (
    <div className="relative h-12 flex items-center justify-center">
      <div className="absolute w-0.5 h-full bg-gradient-to-b from-white/20 to-white/5" />
      <motion.div
        className="absolute w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_15px_5px_rgba(34,211,238,0.5)]"
        animate={
          isActive
            ? {
                y: [-20, 20],
                opacity: [0, 1, 0],
              }
            : {}
        }
        transition={{
          duration: 0.8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="text-cyan-400 text-xl"
        animate={{ y: [0, 5, 0] }}
        transition={{ duration: 1, repeat: Infinity }}
      >
        ▼
      </motion.div>
    </div>
  );
}

// Oracle card component
function OracleCard({
  oracle,
  isSelected,
  onSelect,
}: {
  oracle: OracleData;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const statusColors = {
    live: "border-green-500 shadow-green-500/30",
    faulty: "border-red-500 shadow-red-500/30 animate-pulse",
    backup: "border-yellow-500 shadow-yellow-500/30",
    fallback: "border-cyan-500 shadow-cyan-500/30",
  };

  const statusLabels = {
    live: { text: "LIVE", bg: "bg-green-500" },
    faulty: { text: "FAULTY", bg: "bg-red-500" },
    backup: { text: "BACKUP", bg: "bg-yellow-500" },
    fallback: { text: "FALLBACK", bg: "bg-cyan-500" },
  };

  return (
    <motion.div
      className={`relative cursor-pointer`}
      whileHover={{ scale: 1.05, y: -5 }}
      whileTap={{ scale: 0.95 }}
      onClick={onSelect}
    >
      {/* Glow effect */}
      <motion.div
        className={`absolute -inset-1 bg-gradient-to-r ${oracle.color} rounded-2xl blur-lg`}
        animate={{ opacity: isSelected ? 0.6 : 0.2 }}
      />

      {/* Card */}
      <div
        className={`relative bg-black/60 backdrop-blur-xl border-2 ${statusColors[oracle.status]} rounded-2xl p-4 shadow-lg`}
      >
        {/* Status badge */}
        <div
          className={`absolute -top-2 -right-2 ${statusLabels[oracle.status].bg} text-white text-[10px] font-bold px-2 py-0.5 rounded-full`}
        >
          {statusLabels[oracle.status].text}
        </div>

        {/* Icon */}
        <motion.div
          className="text-4xl mb-2 text-center"
          animate={oracle.status === "faulty" ? { rotate: [0, -10, 10, 0] } : {}}
          transition={{ duration: 0.5, repeat: oracle.status === "faulty" ? Infinity : 0, repeatDelay: 1 }}
        >
          {oracle.icon}
        </motion.div>

        {/* Name */}
        <div className="text-center">
          <div className="font-bold text-white text-sm">{oracle.name}</div>
        </div>

        {/* Pulse ring for live oracles */}
        {oracle.status === "live" && (
          <motion.div
            className="absolute inset-0 border-2 border-green-400 rounded-2xl"
            animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
      </div>
    </motion.div>
  );
}

// Step card component
function StepCard({ step, isActive, onClick }: { step: StepData; isActive: boolean; onClick: () => void }) {
  return (
    <motion.div className="relative cursor-pointer" whileHover={{ scale: 1.02 }} onClick={onClick} layout>
      {/* Animated glow */}
      <motion.div
        className={`absolute -inset-1 bg-gradient-to-r ${step.color} rounded-2xl blur-xl`}
        animate={{ opacity: isActive ? 0.5 : 0.1 }}
        transition={{ duration: 0.3 }}
      />

      {/* Card */}
      <motion.div
        className={`relative bg-black/60 backdrop-blur-xl border rounded-2xl overflow-hidden transition-colors ${
          isActive ? "border-white/40" : "border-white/10"
        }`}
        animate={{ borderColor: isActive ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.1)" }}
      >
        {/* Header */}
        <div className={`bg-gradient-to-r ${step.color} p-4`}>
          <div className="flex items-center gap-3">
            <motion.span
              className="text-3xl"
              animate={isActive ? { rotate: [0, -10, 10, 0], scale: [1, 1.2, 1] } : {}}
              transition={{ duration: 0.5, repeat: isActive ? Infinity : 0, repeatDelay: 2 }}
            >
              {step.icon}
            </motion.span>
            <div>
              <div className="text-xs text-white/70 font-medium">STEP {step.id}</div>
              <div className="font-bold text-white">{step.title}</div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-4">
          <p className="text-white/70 text-sm mb-3">{step.description}</p>

          <AnimatePresence>
            {isActive && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="space-y-2 pt-2 border-t border-white/10">
                  {step.details.map((detail, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ x: -20, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: idx * 0.1 }}
                      className="flex items-start gap-2"
                    >
                      <span className="text-cyan-400 mt-1">•</span>
                      <span className="text-xs text-white/80">{detail}</span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Progress indicator */}
        <motion.div
          className={`h-1 bg-gradient-to-r ${step.color}`}
          initial={{ width: "0%" }}
          animate={{ width: isActive ? "100%" : "0%" }}
          transition={{ duration: 2, repeat: isActive ? Infinity : 0 }}
        />
      </motion.div>
    </motion.div>
  );
}

export function SystemFlowDiagram() {
  const [activeStep, setActiveStep] = useState(1);
  const [selectedOracle, setSelectedOracle] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(true);
  const [mounted, setMounted] = useState(false);

  // Generate fixed particle positions once on mount
  const particlePositions = useMemo(
    () =>
      Array.from({ length: 20 }, (_, i) => ({
        left: (i * 17 + 5) % 100,
        top: (i * 23 + 10) % 100,
        offset: ((i * 7) % 40) - 20,
      })),
    [],
  );

  // Set mounted to true after hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Auto-advance through steps
  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setActiveStep(prev => (prev % 4) + 1);
    }, 4000);
    return () => clearInterval(interval);
  }, [isAnimating]);

  return (
    <div className="relative">
      {/* Background particles - only render after mount to avoid hydration mismatch */}
      {mounted && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {particlePositions.map((pos, i) => (
            <div key={i} className="absolute" style={{ left: `${pos.left}%`, top: `${pos.top}%` }}>
              <Particle delay={i * 0.3} offset={pos.offset} />
            </div>
          ))}
        </div>
      )}

      {/* Main container */}
      <div className="relative z-10">
        {/* Header */}
        <motion.div className="text-center mb-8" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 rounded-full mb-4">
            <motion.span animate={{ rotate: 360 }} transition={{ duration: 3, repeat: Infinity, ease: "linear" }}>
              ⚡
            </motion.span>
            <span className="text-cyan-300 font-medium">INTELLIGENT MULTI-ORACLE AGGREGATION</span>
            <motion.span animate={{ rotate: 360 }} transition={{ duration: 3, repeat: Infinity, ease: "linear" }}>
              ⚡
            </motion.span>
          </div>

          <div className="flex items-center justify-center gap-4 mt-4">
            <button
              onClick={() => setIsAnimating(!isAnimating)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                isAnimating
                  ? "bg-green-500/20 border border-green-500/50 text-green-300"
                  : "bg-white/10 border border-white/20 text-white/60"
              }`}
            >
              {isAnimating ? "🔄 Auto-Play ON" : "⏸️ Auto-Play OFF"}
            </button>
          </div>
        </motion.div>

        {/* Oracle Sources */}
        <div className="mb-6">
          <div className="text-center mb-4">
            <span className="text-white/50 text-xs uppercase tracking-wider">Data Sources</span>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            {oracles.map(oracle => (
              <OracleCard
                key={oracle.name}
                oracle={oracle}
                isSelected={selectedOracle === oracle.name}
                onSelect={() => setSelectedOracle(selectedOracle === oracle.name ? null : oracle.name)}
              />
            ))}
          </div>
        </div>

        {/* Data flow animation */}
        <DataFlowLine isActive={isAnimating} />

        {/* Processing Steps */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {steps.map(step => (
            <StepCard
              key={step.id}
              step={step}
              isActive={activeStep === step.id}
              onClick={() => {
                setIsAnimating(false);
                setActiveStep(step.id);
              }}
            />
          ))}
        </div>

        {/* Data flow to final */}
        <DataFlowLine isActive={isAnimating} />

        {/* Final Output */}
        <motion.div
          className="relative max-w-md mx-auto"
          animate={{ scale: activeStep === 4 ? [1, 1.05, 1] : 1 }}
          transition={{ duration: 0.5 }}
        >
          {/* Pulsing glow */}
          <motion.div
            className="absolute -inset-2 bg-gradient-to-r from-green-500 via-cyan-500 to-purple-500 rounded-3xl blur-xl"
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
          />

          {/* Card */}
          <div className="relative bg-black/70 backdrop-blur-xl border-2 border-green-500/50 rounded-3xl p-6 text-center">
            <motion.div
              className="text-5xl mb-3"
              animate={{ scale: [1, 1.2, 1], rotate: [0, 5, -5, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              ✅
            </motion.div>
            <div className="text-2xl font-bold text-white mb-1">FINAL PRICE</div>
            <div className="text-green-400 font-medium">Byzantine Fault Tolerant</div>
            <div className="mt-3 flex items-center justify-center gap-2">
              <span className="px-3 py-1 bg-green-500/20 border border-green-500/30 rounded-full text-green-300 text-sm">
                🛡️ Secure
              </span>
              <span className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/30 rounded-full text-cyan-300 text-sm">
                ⚡ Real-time
              </span>
              <span className="px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full text-purple-300 text-sm">
                🔒 Verified
              </span>
            </div>
          </div>
        </motion.div>

        {/* Legend */}
        <div className="mt-8 flex flex-wrap justify-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-white/60">Live Oracle</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="text-white/60">Faulty (Excluded)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-white/60">Backup Oracle</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-cyan-500" />
            <span className="text-white/60">Fallback Feed</span>
          </div>
        </div>
      </div>
    </div>
  );
}
