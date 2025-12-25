"use client";

import { ReactNode } from "react";
import { StrikethroughFade } from "./ComparisonReveal";
import { ScrollReveal } from "./ScrollReveal";

// ═══════════════════════════════════════════════════════════════════════════
// PAYFLOW NARRATIVE SYSTEM
// Storytelling components that capture emotions and convey the mission
// ═══════════════════════════════════════════════════════════════════════════

// ============================================
// THE PAYFLOW STORY - Core Narrative Content
// ============================================

export const NARRATIVE = {
  // The Problem - Pain Points
  problem: {
    headline: "The Broken Promise of Global Finance",
    subheadline: "A $150 Trillion Problem Hiding in Plain Sight",
    story: `Every day, businesses lose millions waiting for money that should have arrived yesterday. A manufacturer in Detroit waits 5 days for payment from Tokyo. A healthcare startup in Bangalore watches $2M sit frozen in compliance limbo. A family in Mexico receives remittances minus 7% in hidden fees.

This isn't just inefficiency—it's a tax on the global economy.`,
    stats: [
      { value: "$150T", label: "Annual Cross-Border Volume", context: "Moving at the speed of 1970" },
      { value: "3-5 Days", label: "Average Settlement Time", context: "In an instant world" },
      { value: "$120B", label: "Lost to Fees Annually", context: "Extracted, not earned" },
      { value: "47%", label: "Failed Compliance Checks", context: "Blocking legitimate trade" },
    ],
    painPoints: [
      {
        title: "The Waiting Game",
        description:
          "Cross-border payments take 3-5 business days. In that time, exchange rates shift, opportunities vanish, and capital sits idle.",
        emotion: "frustration",
      },
      {
        title: "Compliance Theater",
        description:
          "Banks run the same KYC checks repeatedly. Each intermediary adds delay, cost, and failure points. Legitimate businesses suffer while bad actors find workarounds.",
        emotion: "absurdity",
      },
      {
        title: "The Hidden Tax",
        description:
          "Correspondent banking fees, FX spreads, lifting fees, receiving fees. By the time money arrives, 3-7% has disappeared into the ether.",
        emotion: "injustice",
      },
      {
        title: "The Trust Deficit",
        description:
          "Without real-time visibility, businesses can't plan. Treasurers hedge against uncertainty. Suppliers demand upfront payment. Everyone pays for lack of trust.",
        emotion: "anxiety",
      },
    ],
  },

  // The Vision - What We're Building
  vision: {
    headline: "Money That Carries Its Own Rules",
    subheadline: "Programmable Payment Rails for the Next Era of Global Commerce",
    story: `Imagine if every dollar knew its destination, its constraints, and its compliance status before it moved. Not after—before.

PayFlow isn't just faster payments. It's a fundamental reimagining of how value moves across borders. We embed the rules into the money itself.`,
    principles: [
      {
        title: "Compliance by Design",
        description:
          "KYC, AML, and sanctions checks happen once, verified on-chain, and travel with every transaction. No redundant checks. No delays. No excuses.",
        icon: "🔐",
      },
      {
        title: "Settlement in Seconds",
        description:
          "12 seconds. That's how long it takes to move $10 million from New York to Tokyo. Not because we cut corners—because we eliminated them.",
        icon: "⚡",
      },
      {
        title: "Transparent by Nature",
        description:
          "Every fee visible. Every step traceable. Every participant verified. The opacity that enabled hidden costs becomes impossible.",
        icon: "🔍",
      },
      {
        title: "Programmable by Default",
        description:
          "Escrow releases on delivery confirmation. Payments execute only if compliance clears. Multi-sig approvals for large transfers. The logic lives in the rails.",
        icon: "🧠",
      },
    ],
  },

  // The Innovation - What Sets Us Apart
  innovation: {
    headline: "Not Just Faster. Fundamentally Different.",
    subheadline: "The Technology That Makes Impossible Possible",
    story: `Other solutions speed up the old system. We replaced it.

Traditional cross-border payments require 6-7 intermediaries, each adding delay and extracting fees. PayFlow requires zero. Compliance that takes 48 hours happens in 2 seconds. FX rates that can shift 3% during settlement are locked at initiation.

This isn't incremental improvement. This is infrastructure for the next century.`,
    differentiators: [
      {
        title: "Multi-Source Oracle Aggregation",
        description:
          "We don't trust any single price source. Our OracleAggregator provides weighted averaging with circuit breakers. 20% deviation triggers emergency halt. Flash loan attacks? Impossible.",
        technical:
          "Multi-oracle support with staleness checks (1 hour threshold), TWAP calculation, and automatic circuit breaker on extreme price movements",
      },
      {
        title: "On-Chain Compliance Engine",
        description:
          "5-tier verification from basic email to full institutional KYC. ComplianceEngine.checkPaymentCompliance() is called before every payment execution.",
        technical:
          "Integrated compliance verification with tiered limits, sanctions checking, and Travel Rule data hashing (FATF R.16 compliant)",
      },
      {
        title: "Programmable Escrow (All 4 Types)",
        description:
          "Money that waits for the right conditions. TIME_BASED auto-release, APPROVAL sign-off, ORACLE verification, and MULTI_SIG corporate approvals—all fully implemented.",
        technical:
          "SmartEscrow with 4 release mechanisms: time-locks, beneficiary approval, external oracle triggers, and M-of-N multi-signature",
      },
      {
        title: "Immutable Audit Trail",
        description:
          "Every compliance check, every approval, every transfer—permanently recorded. Travel Rule compliant. Regulator-queryable. Tamper-proof.",
        technical:
          "AuditRegistry with LOGGER_ROLE access control, logPaymentExecuted events, and regulatory export capabilities",
      },
    ],
  },

  // The Market Reality - Why Now
  market: {
    headline: "The Convergence Has Arrived",
    subheadline: "Fintech × Web3 × Regulatory Clarity = The Moment",
    story: `For years, traditional finance and crypto existed in parallel universes. Banks moved slowly but safely. Crypto moved fast but recklessly. Neither could solve cross-border payments alone.

Now, the worlds are colliding. PayPal holds $1B in crypto. Visa settles in USDC. JPMorgan runs a private blockchain. The question isn't whether institutional finance goes on-chain—it's who builds the rails.`,
    convergencePoints: [
      {
        player: "PayPal",
        action: "Launched PYUSD stablecoin, processing crypto payments for 435M users",
        implication: "Payment giants are moving to programmable money",
      },
      {
        player: "Visa",
        action: "Processing USDC settlements, piloting tokenized assets",
        implication: "Card networks see blockchain as infrastructure, not competition",
      },
      {
        player: "BlackRock",
        action: "Launched tokenized Treasury fund, pushing for Bitcoin ETF",
        implication: "The world's largest asset manager is betting on tokenization",
      },
      {
        player: "Regulators",
        action: "MiCA in Europe, stablecoin frameworks emerging globally",
        implication: "Compliance-first projects have regulatory tailwind",
      },
    ],
  },

  // The Mission - Why We Built This
  mission: {
    headline: "We Built This Because It Matters",
    subheadline: "Financial Infrastructure Should Work for Everyone",
    story: `Every day, 1.4 billion adults remain unbanked—not because they don't want accounts, but because the system wasn't built for them. Small businesses in emerging markets pay 10% fees on cross-border invoices. Migrant workers lose a month's wages to remittance costs.

PayFlow isn't just about moving money faster. It's about building infrastructure that doesn't discriminate based on geography or account size. When compliance is automated and settlement is instant, the small player and the institution stand on equal ground.

We built this because the financial system should work for everyone, not just everyone who can afford the friction.`,
    impactAreas: [
      {
        area: "Small Business Trade",
        problem: "SMBs pay 5-15x more for cross-border transactions than enterprises",
        solution: "Same rails, same fees, same speed—regardless of volume",
      },
      {
        area: "Remittances",
        problem: "Migrant workers lose 6.5% of every dollar sent home",
        solution: "Near-zero fees with instant settlement to any wallet",
      },
      {
        area: "Supply Chain Finance",
        problem: "120-day payment terms strangle supplier cash flow",
        solution: "Programmable escrow releases on verified delivery",
      },
      {
        area: "Treasury Operations",
        problem: "Multi-day uncertainty forces expensive hedging",
        solution: "Real-time settlement enables just-in-time treasury",
      },
    ],
  },

  // Social Proof & Credibility
  credibility: {
    contracts: [
      {
        name: "PayFlowCore",
        address: "0x4c9489812a9D971b431B9d99049a42B437347dBC",
        purpose: "Central payment orchestration",
      },
      {
        name: "ComplianceEngine",
        address: "0xB7829739a4BceA372f391E3Cc638B7ccFdCBF911",
        purpose: "5-tier verification system",
      },
      {
        name: "SmartEscrow",
        address: "0x24DB2e1B585f90Adb27b71a36734aBe00A206CEf",
        purpose: "Programmable conditional payments",
      },
      {
        name: "OracleAggregator",
        address: "0x4205e920770412d94b90f58fc295c80a8D264d81",
        purpose: "BFT price feed consensus",
      },
      {
        name: "AuditRegistry",
        address: "0xDcC0BDF68c3A0568D7C2C9e88D9c6eB574173846",
        purpose: "Immutable compliance logs",
      },
    ],
    hackathonContext: "Built for Hackxios 2K25 • PayPal & Visa Sponsored Track",
  },
};

// ============================================
// NARRATIVE COMPONENTS
// ============================================

interface StoryBlockProps {
  headline: string;
  subheadline?: string;
  story?: string;
  children?: ReactNode;
  variant?: "hero" | "section" | "quote";
  align?: "left" | "center";
}

export function StoryBlock({
  headline,
  subheadline,
  story,
  children,
  variant = "section",
  align = "center",
}: StoryBlockProps) {
  const alignClass = align === "center" ? "text-center mx-auto" : "text-left";

  return (
    <ScrollReveal className={`max-w-4xl ${alignClass}`}>
      {subheadline && (
        <p className="text-sm uppercase tracking-widest text-violet-400 mb-4 font-medium">{subheadline}</p>
      )}
      <h2 className={`font-bold mb-6 ${variant === "hero" ? "text-5xl md:text-7xl" : "text-3xl md:text-5xl"}`}>
        <span className="bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
          {headline}
        </span>
      </h2>
      {story && <p className="text-lg md:text-xl text-zinc-400 leading-relaxed whitespace-pre-line">{story}</p>}
      {children}
    </ScrollReveal>
  );
}

interface PainPointCardProps {
  title: string;
  description: string;
  emotion: string;
  index: number;
}

export function PainPointCard({ title, description, emotion, index }: PainPointCardProps) {
  const emotionColors: Record<string, string> = {
    frustration: "from-red-500 to-orange-500",
    absurdity: "from-yellow-500 to-amber-500",
    injustice: "from-rose-500 to-pink-500",
    anxiety: "from-purple-500 to-violet-500",
  };

  return (
    <ScrollReveal delay={index * 100} direction="up">
      <div className="group relative p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-all duration-500 h-full">
        <div
          className={`absolute -inset-px rounded-2xl bg-gradient-to-r ${emotionColors[emotion] || emotionColors.frustration} opacity-0 group-hover:opacity-10 transition-opacity duration-500 blur-xl`}
        />
        <div className="relative">
          <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
          <p className="text-zinc-400 leading-relaxed">{description}</p>
        </div>
      </div>
    </ScrollReveal>
  );
}

interface StatCardProps {
  value: string;
  label: string;
  context?: string;
  index: number;
}

export function StatCard({ value, label, context, index }: StatCardProps) {
  return (
    <ScrollReveal delay={index * 100} direction="up">
      <div className="text-center p-6 rounded-2xl bg-white/[0.02] border border-white/5">
        <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent mb-2">
          {value}
        </div>
        <div className="text-white font-medium mb-1">{label}</div>
        {context && <div className="text-sm text-zinc-500 italic">{context}</div>}
      </div>
    </ScrollReveal>
  );
}

interface QuoteBlockProps {
  quote: string;
  attribution?: string;
}

export function QuoteBlock({ quote, attribution }: QuoteBlockProps) {
  return (
    <ScrollReveal>
      <blockquote className="relative max-w-3xl mx-auto text-center py-12">
        <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-6xl text-violet-500/30">&quot;</div>
        <p className="text-2xl md:text-3xl font-light text-zinc-300 leading-relaxed italic">{quote}</p>
        {attribution && <footer className="mt-6 text-zinc-500">— {attribution}</footer>}
      </blockquote>
    </ScrollReveal>
  );
}

interface TimelineItemProps {
  player: string;
  action: string;
  implication: string;
  index: number;
}

export function TimelineItem({ player, action, implication, index }: TimelineItemProps) {
  return (
    <ScrollReveal delay={index * 150} direction="left">
      <div className="relative pl-8 pb-8 border-l border-white/10 last:border-l-transparent">
        <div className="absolute -left-2 top-0 w-4 h-4 rounded-full bg-gradient-to-r from-violet-500 to-cyan-500" />
        <div className="text-lg font-bold text-white mb-1">{player}</div>
        <div className="text-zinc-400 mb-2">{action}</div>
        <div className="text-sm text-cyan-400 italic">→ {implication}</div>
      </div>
    </ScrollReveal>
  );
}

interface MissionImpactCardProps {
  area: string;
  problem: string;
  solution: string;
  index: number;
}

export function MissionImpactCard({ area, problem, solution, index }: MissionImpactCardProps) {
  return (
    <ScrollReveal delay={index * 100} direction="up">
      <div className="group relative p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-green-500/30 transition-all duration-500 h-full">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-r from-green-500 to-emerald-500 opacity-0 group-hover:opacity-10 transition-opacity duration-500 blur-xl" />
        <div className="relative">
          <div className="text-sm uppercase tracking-wider text-green-400 mb-2">{area}</div>
          <div className="text-zinc-400 mb-4">
            <StrikethroughFade problem={problem} solution={solution} />
          </div>
        </div>
      </div>
    </ScrollReveal>
  );
}

// ============================================
// SECTION DIVIDERS WITH NARRATIVE
// ============================================

export function NarrativeDivider({ text }: { text: string }) {
  return (
    <div className="relative py-16">
      <div className="absolute inset-0 flex items-center">
        <div className="w-full border-t border-white/5" />
      </div>
      <div className="relative flex justify-center">
        <span className="bg-[#0a0a0f] px-6 text-zinc-500 text-sm uppercase tracking-widest">{text}</span>
      </div>
    </div>
  );
}

// ============================================
// HERO NARRATIVE BADGE
// ============================================

export function NarrativeBadge({ text, pulse = true }: { text: string; pulse?: boolean }) {
  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm">
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
        </span>
      )}
      <span className="text-sm text-zinc-300">{text}</span>
    </div>
  );
}
