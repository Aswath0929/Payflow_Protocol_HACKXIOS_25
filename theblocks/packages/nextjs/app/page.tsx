"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { RainbowKitCustomConnectButton } from "~~/components/scaffold-eth";
import { ScrollProgress, SectionNav } from "~~/components/ui/AdvancedAnimations";
import {
  AnimatedCounter,
  GlassmorphicCard,
  MorphingBlob,
  PulseRing,
  TypewriterText,
} from "~~/components/ui/ModernEffects";
import {
  MissionImpactCard,
  NARRATIVE,
  NarrativeBadge,
  NarrativeDivider,
  PainPointCard,
  QuoteBlock,
  StatCard,
  StoryBlock,
  TimelineItem,
} from "~~/components/ui/Narrative";
import { MagneticButton, ParticleField } from "~~/components/ui/ParticleField";
import { ScrollReveal } from "~~/components/ui/ScrollReveal";

// ═══════════════════════════════════════════════════════════════════════════
// PAYFLOW PROTOCOL - NARRATIVE-DRIVEN LANDING PAGE
// A story that wins hearts and minds. Not just technically—emotionally.
// Built for Hackxios 2K25 • PayPal & Visa Track
// ═══════════════════════════════════════════════════════════════════════════

export default function Home() {
  const router = useRouter();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [scrollY, setScrollY] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);

  // Track mouse for gradient effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("scroll", handleScroll);

    // Trigger entrance animations
    setTimeout(() => setIsLoaded(true), 100);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // Live demo animation state
  const [demoStep, setDemoStep] = useState(0);
  const [amount, setAmount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDemoStep(prev => (prev + 1) % 8);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (demoStep === 3) {
      const duration = 1000;
      const start = Date.now();
      const animate = () => {
        const elapsed = Date.now() - start;
        const progress = Math.min(elapsed / duration, 1);
        setAmount(Math.floor(progress * 10000000));
        if (progress < 1) requestAnimationFrame(animate);
      };
      animate();
    }
  }, [demoStep]);

  const demoSteps = [
    { label: "Initiating", status: "pending", icon: "🔄" },
    { label: "KYC Verified", status: "success", icon: "✓" },
    { label: "AML Cleared", status: "success", icon: "✓" },
    { label: "Sanctions OK", status: "success", icon: "✓" },
    { label: "FX Rate Locked", status: "success", icon: "📈" },
    { label: "Transferring", status: "active", icon: "⚡" },
    { label: "Settled", status: "success", icon: "💎" },
    { label: "Complete", status: "complete", icon: "🎉" },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-hidden">
      {/* Scroll Progress Bar */}
      <ScrollProgress />

      {/* Section Navigation */}
      <SectionNav
        sections={[
          { id: "hero", label: "Home" },
          { id: "problem", label: "The Problem" },
          { id: "solution", label: "Solution" },
          { id: "features", label: "Features" },
        ]}
      />

      {/* Animated Background Gradient */}
      <div
        className="fixed inset-0 pointer-events-none transition-opacity duration-1000"
        style={{
          background: `
            radial-gradient(800px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(139, 92, 246, 0.07), transparent 40%),
            radial-gradient(600px circle at ${mousePosition.x + 200}px ${mousePosition.y - 100}px, rgba(6, 182, 212, 0.05), transparent 40%)
          `,
        }}
      />

      {/* Grid Background */}
      <div
        className="fixed inset-0 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: "100px 100px",
          transform: `translateY(${scrollY * 0.1}px)`,
        }}
      />

      {/* Floating Orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div
          className="absolute w-[600px] h-[600px] rounded-full opacity-20 blur-[100px]"
          style={{
            background: "linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)",
            top: "-200px",
            right: "-200px",
            animation: "float 20s ease-in-out infinite",
          }}
        />
        <div
          className="absolute w-[400px] h-[400px] rounded-full opacity-15 blur-[80px]"
          style={{
            background: "linear-gradient(135deg, #22c55e 0%, #06b6d4 100%)",
            bottom: "-100px",
            left: "-100px",
            animation: "float 15s ease-in-out infinite reverse",
          }}
        />
      </div>

      {/* Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrollY > 50 ? "bg-[#0a0a0f]/80 backdrop-blur-xl border-b border-white/5" : ""
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link
              href="/"
              className={`flex items-center gap-3 transition-all duration-700 ${isLoaded ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"}`}
            >
              <div className="relative">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
                  <span className="text-lg font-bold">💎</span>
                </div>
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-500 blur-lg opacity-50" />
              </div>
              <span className="text-xl font-bold tracking-tight">PayFlow</span>
            </Link>

            {/* Nav Links */}
            <div
              className={`hidden md:flex items-center gap-6 transition-all duration-700 delay-100 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"}`}
            >
              <Link
                href="/dashboard"
                className="text-sm text-zinc-400 hover:text-white transition-colors duration-300 relative group"
              >
                🏦 Dashboard
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-violet-500 to-cyan-500 transition-all duration-300 group-hover:w-full" />
              </Link>
              <Link
                href="/oracle-dashboard"
                className="text-sm text-zinc-400 hover:text-white transition-colors duration-300 relative group"
              >
                📊 Oracles
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-violet-500 to-cyan-500 transition-all duration-300 group-hover:w-full" />
              </Link>
              <a
                href="#problem"
                className="text-sm text-zinc-400 hover:text-white transition-colors duration-300 relative group"
              >
                The Problem
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-violet-500 to-cyan-500 transition-all duration-300 group-hover:w-full" />
              </a>
              <a
                href="#solution"
                className="text-sm text-zinc-400 hover:text-white transition-colors duration-300 relative group"
              >
                Our Solution
                <span className="absolute -bottom-1 left-0 w-0 h-px bg-gradient-to-r from-violet-500 to-cyan-500 transition-all duration-300 group-hover:w-full" />
              </a>
            </div>

            {/* CTA Buttons */}
            <div
              className={`flex items-center gap-3 transition-all duration-700 delay-200 ${isLoaded ? "opacity-100 translate-x-0" : "opacity-0 translate-x-4"}`}
            >
              <Link
                href="/dashboard"
                prefetch
                className="relative px-5 py-2 rounded-full text-sm font-medium overflow-hidden group hidden sm:block focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-zinc-900"
              >
                <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600 transition-all duration-300 group-hover:scale-105" />
                <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600 blur-xl opacity-50 transition-all duration-300 group-hover:opacity-80" />
                <span className="relative text-white text-sm">Launch App →</span>
              </Link>
              <RainbowKitCustomConnectButton />
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" ref={heroRef} className="relative min-h-screen flex items-center justify-center pt-20">
        <div className="max-w-7xl mx-auto px-6 py-20">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left: Text Content */}
            <div className="space-y-8">
              {/* Badge */}
              <div
                className={`transition-all duration-700 delay-300 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
              >
                <NarrativeBadge text="Live on Sepolia • PayPal & Visa Track" />
              </div>

              {/* Main Headline - Emotional Hook with Typewriter Effect */}
              <h1
                className={`text-5xl md:text-7xl font-bold leading-[1.1] tracking-tight transition-all duration-700 delay-400 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
              >
                <span className="text-zinc-400 text-2xl md:text-3xl font-normal block mb-4">What if money could</span>
                <TypewriterText
                  text="carry its own rules?"
                  className="bg-gradient-to-r from-violet-400 via-cyan-400 to-green-400 bg-clip-text text-transparent"
                  speed={80}
                  showCursor={false}
                />
              </h1>

              {/* Subheadline - The Tension */}
              <p
                className={`text-xl text-zinc-400 max-w-lg leading-relaxed transition-all duration-700 delay-500 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
              >
                <span className="text-white font-medium">$10 million from New York to Tokyo in 12 seconds.</span> Not 5
                days. Not with hidden fees. Not with compliance uncertainty.
                <br />
                <br />
                <span className="text-zinc-500 italic">
                  This is what institutional payments should have always been.
                </span>
              </p>

              {/* CTA Buttons */}
              <div
                className={`flex flex-wrap gap-4 transition-all duration-700 delay-600 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
              >
                <button
                  onClick={() => router.push("/dashboard")}
                  className="group relative px-8 py-4 rounded-xl font-medium overflow-hidden"
                >
                  <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600" />
                  <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600 blur-xl opacity-50 group-hover:opacity-80 transition-opacity" />
                  <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    <span className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-violet-600" />
                  </span>
                  <span className="relative flex items-center gap-2 text-white">
                    Experience the Future
                    <svg
                      className="w-4 h-4 transition-transform group-hover:translate-x-1"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </span>
                </button>

                <a
                  href="#problem"
                  className="group px-8 py-4 rounded-xl font-medium border border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/10 transition-all duration-300 flex items-center gap-2"
                >
                  Read Our Story ↓
                </a>
              </div>

              {/* Stats with Context - Animated Counters */}
              <div
                className={`flex gap-8 pt-8 border-t border-white/5 transition-all duration-700 delay-700 ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
              >
                <div className="space-y-1">
                  <div className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                    <AnimatedCounter value={12} suffix="s" duration={2000} />
                  </div>
                  <div className="text-sm text-zinc-400">Settlement</div>
                  <div className="text-xs text-zinc-600">Not 5 days</div>
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                    <AnimatedCounter value={0.1} decimals={1} suffix="%" duration={2000} />
                  </div>
                  <div className="text-sm text-zinc-400">Total Fees</div>
                  <div className="text-xs text-zinc-600">Not 3-7%</div>
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                    <AnimatedCounter value={100} suffix="%" duration={2000} />
                  </div>
                  <div className="text-sm text-zinc-400">Compliant</div>
                  <div className="text-xs text-zinc-600">Always</div>
                </div>
              </div>
            </div>

            {/* Right: Live Demo Card - Glassmorphic */}
            <div
              className={`relative transition-all duration-1000 delay-500 ${isLoaded ? "opacity-100 translate-x-0 rotate-0" : "opacity-0 translate-x-8 rotate-3"}`}
            >
              {/* Morphing Blob Background */}
              <MorphingBlob className="absolute -inset-20 opacity-30" color1="#8b5cf6" color2="#06b6d4" />

              {/* Glassmorphic Card */}
              <GlassmorphicCard className="relative" glowColor="rgba(139, 92, 246, 0.3)">
                {/* Card Header */}
                <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-500/80" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                      <div className="w-3 h-3 rounded-full bg-green-500/80" />
                    </div>
                    <span className="text-sm text-zinc-500">Live Demo — Watch Money Move</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-500">
                    <PulseRing color="green" size={8} />
                    Sepolia Testnet
                  </div>
                </div>

                {/* Card Content */}
                <div className="p-6 space-y-6">
                  {/* Transfer Info - Institutional Context */}
                  <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-sm font-bold">
                        🇺🇸
                      </div>
                      <div>
                        <div className="text-sm text-zinc-400">JPMorgan NYC</div>
                        <div className="font-medium">Treasury Desk</div>
                      </div>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="text-2xl animate-pulse">→</div>
                      <div className="text-xs text-cyan-400 font-mono">12 sec</div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div>
                        <div className="text-sm text-zinc-400">MUFG Tokyo</div>
                        <div className="font-medium">Settlement Ops</div>
                      </div>
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center text-sm font-bold">
                        🇯🇵
                      </div>
                    </div>
                  </div>

                  {/* Amount Display */}
                  <div className="text-center py-4">
                    <div className="text-sm text-zinc-500 mb-2">Institutional Transfer</div>
                    <div className="text-4xl font-bold font-mono tracking-tight">
                      ${demoStep >= 3 ? amount.toLocaleString() : "10,000,000"}
                      <span className="text-xl text-zinc-500 ml-2">USDC</span>
                    </div>
                    <div className="text-sm text-green-400 mt-2">✓ Full compliance embedded</div>
                  </div>

                  {/* Progress Steps */}
                  <div className="space-y-3">
                    {demoSteps.slice(0, Math.min(demoStep + 1, 8)).map((step, i) => (
                      <div
                        key={i}
                        className={`flex items-center gap-3 p-3 rounded-lg transition-all duration-500 ${
                          i === demoStep
                            ? "bg-violet-500/10 border border-violet-500/30"
                            : i < demoStep
                              ? "bg-green-500/5 border border-green-500/20"
                              : "bg-white/5 border border-transparent"
                        }`}
                        style={
                          i === demoStep
                            ? {
                                animationName: "pulse",
                                animationDuration: "2s",
                                animationTimingFunction: "ease-in-out",
                                animationIterationCount: "infinite",
                                animationDelay: `${i * 0.1}s`,
                              }
                            : {}
                        }
                      >
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                            i < demoStep
                              ? "bg-green-500/20 text-green-400"
                              : i === demoStep
                                ? "bg-violet-500/20 text-violet-400"
                                : "bg-white/10 text-zinc-500"
                          }`}
                        >
                          {step.icon}
                        </div>
                        <span className={i <= demoStep ? "text-white" : "text-zinc-500"}>{step.label}</span>
                        {i < demoStep && <span className="ml-auto text-green-400 text-sm">✓</span>}
                        {i === demoStep && (
                          <span className="ml-auto">
                            <span className="flex h-2 w-2">
                              <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-violet-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500"></span>
                            </span>
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </GlassmorphicCard>
            </div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2">
            <div className="w-1.5 h-3 bg-white/50 rounded-full animate-pulse" />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE PROBLEM - Setting the Stakes */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="problem" className="relative py-32">
        <div className="max-w-7xl mx-auto px-6">
          <StoryBlock
            headline={NARRATIVE.problem.headline}
            subheadline={NARRATIVE.problem.subheadline}
            story={NARRATIVE.problem.story}
          />

          {/* Problem Stats */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
            {NARRATIVE.problem.stats.map((stat, i) => (
              <StatCard key={i} index={i} value={stat.value} label={stat.label} context={stat.context} />
            ))}
          </div>

          <NarrativeDivider text="The Pain Points" />

          {/* Pain Point Cards */}
          <div className="grid md:grid-cols-2 gap-6">
            {NARRATIVE.problem.painPoints.map((point, i) => (
              <PainPointCard
                key={i}
                index={i}
                title={point.title}
                description={point.description}
                emotion={point.emotion}
              />
            ))}
          </div>

          <QuoteBlock
            quote="The global payments system is a $150 trillion market running on 1970s infrastructure. Every day of delay costs the global economy billions in trapped capital and missed opportunities."
            attribution="The Opportunity"
          />
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE VISION - Our Answer */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section
        id="solution"
        className="relative py-32 bg-gradient-to-b from-transparent via-violet-950/10 to-transparent"
      >
        <div className="max-w-7xl mx-auto px-6">
          <StoryBlock
            headline={NARRATIVE.vision.headline}
            subheadline={NARRATIVE.vision.subheadline}
            story={NARRATIVE.vision.story}
          />

          {/* Vision Principles */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
            {NARRATIVE.vision.principles.map((principle, i) => (
              <ScrollReveal key={i} delay={i * 100} direction="up">
                <div className="group relative p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-violet-500/30 transition-all duration-500 h-full">
                  <div className="absolute -inset-px rounded-2xl bg-gradient-to-r from-violet-500 to-cyan-500 opacity-0 group-hover:opacity-10 transition-opacity duration-500 blur-xl" />
                  <div className="relative">
                    <div className="text-4xl mb-4">{principle.icon}</div>
                    <h3 className="text-lg font-bold text-white mb-2">{principle.title}</h3>
                    <p className="text-zinc-400 text-sm leading-relaxed">{principle.description}</p>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE INNOVATION - Technical Depth */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section id="features" className="relative py-32">
        <div className="max-w-7xl mx-auto px-6">
          <StoryBlock
            headline={NARRATIVE.innovation.headline}
            subheadline={NARRATIVE.innovation.subheadline}
            story={NARRATIVE.innovation.story}
          />

          {/* Innovation Differentiators */}
          <div className="grid md:grid-cols-2 gap-8 mt-16">
            {NARRATIVE.innovation.differentiators.map((diff, i) => (
              <ScrollReveal key={i} delay={i * 150} direction={i % 2 === 0 ? "left" : "right"}>
                <div className="group relative p-8 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-cyan-500/30 transition-all duration-500 h-full">
                  <div className="absolute -inset-px rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-500 opacity-0 group-hover:opacity-10 transition-opacity duration-500 blur-xl" />
                  <div className="relative">
                    <h3 className="text-xl font-bold text-white mb-3">{diff.title}</h3>
                    <p className="text-zinc-400 leading-relaxed mb-4">{diff.description}</p>
                    <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                      <span className="text-xs text-cyan-400 font-mono">{diff.technical}</span>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE MARKET - Why Now */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="relative py-32 bg-gradient-to-b from-transparent via-cyan-950/10 to-transparent">
        <div className="max-w-7xl mx-auto px-6">
          <StoryBlock
            headline={NARRATIVE.market.headline}
            subheadline={NARRATIVE.market.subheadline}
            story={NARRATIVE.market.story}
            align="left"
          />

          {/* Market Convergence Timeline */}
          <div className="mt-16 max-w-2xl">
            {NARRATIVE.market.convergencePoints.map((point, i) => (
              <TimelineItem
                key={i}
                index={i}
                player={point.player}
                action={point.action}
                implication={point.implication}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THE MISSION - Why We Care */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="relative py-32">
        <div className="max-w-7xl mx-auto px-6">
          <StoryBlock
            headline={NARRATIVE.mission.headline}
            subheadline={NARRATIVE.mission.subheadline}
            story={NARRATIVE.mission.story}
          />

          {/* Mission Impact Areas */}
          <div className="grid md:grid-cols-2 gap-6 mt-16">
            {NARRATIVE.mission.impactAreas.map((area, i) => (
              <MissionImpactCard key={i} index={i} area={area.area} problem={area.problem} solution={area.solution} />
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* CREDIBILITY - Proof We're Real */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="relative py-32 bg-gradient-to-b from-transparent via-green-950/10 to-transparent">
        <div className="max-w-7xl mx-auto px-6">
          <ScrollReveal className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <span className="text-white">Deployed. Verified. </span>
              <span className="bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                Ready.
              </span>
            </h2>
            <p className="text-zinc-400">All contracts live on Sepolia and verified on Etherscan</p>
          </ScrollReveal>

          {/* Contract Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {NARRATIVE.credibility.contracts.map((contract, i) => (
              <ScrollReveal key={i} delay={i * 100} direction="up">
                <a
                  href={`https://sepolia.etherscan.io/address/${contract.address}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block group p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-green-500/30 transition-all duration-300"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white">{contract.name}</span>
                    <span className="text-xs text-green-400">✓ Verified</span>
                  </div>
                  <div className="text-xs font-mono text-zinc-500 mb-2 truncate">{contract.address}</div>
                  <div className="text-sm text-zinc-400">{contract.purpose}</div>
                </a>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* CTA - The Invitation */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section className="relative py-32 overflow-hidden">
        {/* Interactive Particle Field */}
        <ParticleField particleCount={80} connectionDistance={120} speed={0.3} interactive={true} />

        <div className="max-w-4xl mx-auto px-6 text-center relative z-10">
          {/* Glow */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-[600px] h-[600px] bg-gradient-to-r from-violet-600/20 to-cyan-600/20 rounded-full blur-[100px] animate-pulse" />
          </div>

          <ScrollReveal className="relative space-y-8">
            <h2 className="text-4xl md:text-6xl font-bold">
              The future of money
              <br />
              <span className="bg-gradient-to-r from-violet-400 via-cyan-400 to-green-400 bg-clip-text text-transparent animate-[text-shimmer_3s_ease-in-out_infinite]">
                is programmable.
              </span>
            </h2>

            <p className="text-xl text-zinc-400 max-w-2xl mx-auto">
              We didn&apos;t just build faster rails. We built rails that think. Join us in redefining how the world
              moves value.
            </p>

            <div className="flex flex-wrap justify-center gap-4 pt-4">
              <MagneticButton
                onClick={() => router.push("/dashboard")}
                className="group relative px-10 py-5 rounded-xl font-medium text-lg overflow-hidden"
              >
                <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600" />
                <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-600 blur-xl opacity-50 group-hover:opacity-80 transition-opacity" />
                <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-all duration-500 bg-gradient-to-r from-cyan-600 to-violet-600" />
                <span className="relative text-white flex items-center gap-2">
                  Launch PayFlow
                  <svg
                    className="w-5 h-5 transition-transform group-hover:translate-x-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </span>
              </MagneticButton>

              <a
                href="https://sepolia.etherscan.io/address/0x4c9489812a9D971b431B9d99049a42B437347dBC"
                target="_blank"
                rel="noopener noreferrer"
                className="px-10 py-5 rounded-xl font-medium text-lg border border-white/10 hover:border-white/30 bg-white/5 hover:bg-white/10 transition-all duration-300 backdrop-blur-sm"
              >
                View Smart Contracts
              </a>
            </div>

            <div className="pt-8 text-sm text-zinc-500">{NARRATIVE.credibility.hackathonContext}</div>
          </ScrollReveal>
        </div>
      </section>

      {/* Footer */}
      <ScrollReveal>
        <footer className="border-t border-white/5 py-12">
          <div className="max-w-7xl mx-auto px-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center text-sm">
                  💎
                </div>
                <span className="font-bold">PayFlow Protocol</span>
              </div>

              <div className="text-sm text-zinc-500 text-center">
                <span className="text-violet-400">Built with conviction.</span> The financial system should work for
                everyone.
              </div>

              <div className="flex items-center gap-6 text-sm text-zinc-400">
                <Link href="/dashboard" className="hover:text-white transition-colors">
                  Dashboard
                </Link>
                <Link href="/oracle-dashboard" className="hover:text-white transition-colors">
                  Oracles
                </Link>
                <a
                  href="https://sepolia.etherscan.io/address/0x4c9489812a9D971b431B9d99049a42B437347dBC"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors"
                >
                  Etherscan
                </a>
              </div>
            </div>
          </div>
        </footer>
      </ScrollReveal>
    </div>
  );
}
