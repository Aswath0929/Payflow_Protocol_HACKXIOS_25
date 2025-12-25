"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import type { NextPage } from "next";
import { ContractStatus } from "~~/components/oracle/ContractStatus";
import { SecurityMetrics } from "~~/components/oracle/SecurityMetrics";
import { ThreeOracleDashboard } from "~~/components/oracle/ThreeOracleDashboard";
import { OracleConsensusDemo, SecurityShieldDemo } from "~~/components/ui/LiveDemoWidget";
import { ScrollReveal } from "~~/components/ui/ScrollReveal";

/**
 * 🏆 PAYFLOW ORACLE DASHBOARD - NARRATIVE-DRIVEN SECURITY VISUALIZATION
 * Hackxios 2K25 - PayPal & Visa Track
 *
 * Not just monitoring oracles—demonstrating why this innovation matters.
 * BFT consensus that would make even Visa's fraud team jealous.
 *
 * Features:
 * - 3-Oracle BFT Consensus (Byzantine Fault Tolerant)
 * - GuardianOracleV2 Security Layer
 * - SmartOracleSelector AI Scoring
 * - Multi-Asset Price Explorer (50+ feeds)
 */

type ViewMode = "dashboard" | "metrics" | "contracts";

// Inner component that uses useSearchParams
const OraclePageContent = () => {
  const searchParams = useSearchParams();
  const initialView = (searchParams.get("view") as ViewMode) || "dashboard";
  const [activeView, setActiveView] = useState<ViewMode>(initialView);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    // Update view if URL param changes
    const viewParam = searchParams.get("view") as ViewMode;
    if (viewParam && ["dashboard", "metrics", "contracts"].includes(viewParam)) {
      setActiveView(viewParam);
    }
  }, [searchParams]);

  const views = [
    { id: "dashboard" as ViewMode, label: "3-Oracle BFT + AI", icon: "🧠", color: "from-violet-500 to-cyan-500" },
    { id: "metrics" as ViewMode, label: "Security", icon: "🛡️", color: "from-green-500 to-emerald-500" },
    { id: "contracts" as ViewMode, label: "Contracts", icon: "📜", color: "from-pink-500 to-rose-500" },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl animate-pulse" />
        <div
          className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        />
        <div
          className="absolute top-1/2 left-1/2 w-64 h-64 bg-blue-500/5 rounded-full blur-2xl"
          style={{ animation: "bounce 4s ease-in-out infinite" }}
        />

        {/* Grid Pattern */}
        <div
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)`,
            backgroundSize: "80px 80px",
          }}
        />
      </div>

      {/* Header */}
      <header
        className={`relative z-10 pt-8 pb-6 transition-all duration-1000 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-10"
        }`}
      >
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
            {/* Logo & Title */}
            <div className="text-center lg:text-left">
              <div className="flex items-center justify-center lg:justify-start gap-3 mb-2">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-xl flex items-center justify-center text-2xl font-bold shadow-lg shadow-cyan-500/30">
                    ⬡
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-ping" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full" />
                </div>
                <div>
                  <h1 className="text-3xl font-black bg-gradient-to-r from-violet-400 via-cyan-400 to-green-400 bg-clip-text text-transparent">
                    PayFlow Oracle
                  </h1>
                  <p className="text-xs text-zinc-400 tracking-widest uppercase">3-Oracle BFT Security System</p>
                </div>
              </div>
            </div>

            {/* Network Badge */}
            <div className="flex items-center gap-4">
              <div className="px-4 py-2 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/30 rounded-full flex items-center gap-2">
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-yellow-300">Sepolia Testnet</span>
              </div>
              <div className="px-4 py-2 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-full flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-green-300">Live</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav
        className={`relative z-10 py-4 transition-all duration-1000 delay-200 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
        }`}
      >
        <div className="container mx-auto px-4">
          <div className="flex flex-wrap justify-center gap-3">
            {views.map(view => (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                className={`group relative px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  activeView === view.id
                    ? `bg-gradient-to-r ${view.color} text-white shadow-lg scale-105`
                    : "bg-white/5 text-zinc-300 hover:bg-white/10 border border-white/5"
                }`}
              >
                <span className="flex items-center gap-2">
                  <span className="text-xl">{view.icon}</span>
                  <span>{view.label}</span>
                </span>
                {activeView === view.id && <div className="absolute inset-0 rounded-xl bg-white/20 animate-pulse" />}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main
        className={`relative z-10 container mx-auto px-4 py-8 transition-all duration-1000 delay-400 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
        }`}
      >
        {/* Narrative Banner - Why Oracle Security Matters */}
        <ScrollReveal className="mb-8">
          <div className="relative p-6 rounded-2xl border border-cyan-500/20 bg-gradient-to-r from-cyan-900/20 to-blue-900/20 overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl" />
            <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex-1">
                <div className="text-cyan-400 text-sm font-medium uppercase tracking-wider mb-2">Why This Matters</div>
                <h3 className="text-xl font-bold text-white mb-2">The First Line of Defense</h3>
                <p className="text-zinc-400 text-sm">
                  Flash loan attacks have stolen <span className="text-red-400 font-bold">$1.5B+</span> by manipulating
                  single-source price feeds. Our 3-Oracle BFT consensus with{" "}
                  <span className="text-cyan-400 font-bold">AI scoring</span> and{" "}
                  <span className="text-cyan-400 font-bold">outlier detection</span> makes such attacks mathematically
                  impossible.
                  <span className="text-zinc-500 block mt-1 italic">
                    If one oracle lies, consensus prevails. Try it yourself in Attack Sim.
                  </span>
                </p>
              </div>
              <div className="flex-shrink-0 flex gap-6 text-center">
                <div>
                  <div className="text-3xl font-bold text-red-400">$1.5B+</div>
                  <div className="text-xs text-zinc-500">
                    Stolen via oracle
                    <br />
                    manipulation
                  </div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-green-400">$0</div>
                  <div className="text-xs text-zinc-500">
                    Possible with
                    <br />
                    PayFlow BFT
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ScrollReveal>

        {/* Live Demo Widgets */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <ScrollReveal delay={100} direction="left">
            <div>
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <span className="text-xl">🧠</span> Live Oracle Consensus
              </h3>
              <OracleConsensusDemo />
            </div>
          </ScrollReveal>
          <ScrollReveal delay={200} direction="right">
            <div>
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <span className="text-xl">🛡️</span> Security Shield
              </h3>
              <SecurityShieldDemo />
            </div>
          </ScrollReveal>
        </div>

        {activeView === "dashboard" && <ThreeOracleDashboard />}
        {activeView === "metrics" && <SecurityMetrics />}
        {activeView === "contracts" && <ContractStatus />}
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 border-t border-white/5">
        <div className="container mx-auto px-4 text-center">
          <p className="text-zinc-500 text-sm">💎 PayFlow Protocol • Hackxios 2K25 • Built for PayPal & Visa</p>
        </div>
      </footer>
    </div>
  );
};

// Loading fallback for Suspense
const LoadingFallback = () => (
  <div className="min-h-screen bg-[#0a0a0f] text-white flex items-center justify-center">
    <div className="text-center">
      <div className="w-16 h-16 bg-gradient-to-br from-violet-500 to-cyan-500 rounded-xl flex items-center justify-center text-3xl font-bold shadow-lg shadow-violet-500/30 mx-auto mb-4 animate-pulse">
        💎
      </div>
      <p className="text-zinc-400">Loading Oracle Dashboard...</p>
    </div>
  </div>
);

// Main page component wrapped in Suspense
const OraclePage: NextPage = () => {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <OraclePageContent />
    </Suspense>
  );
};

export default OraclePage;
