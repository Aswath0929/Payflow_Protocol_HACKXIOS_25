import AIFraudDashboard from "~~/components/fraud/AIFraudDashboard";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Fraud Detection | PayFlow",
  description: "Real-time AI-powered fraud detection for stablecoin transactions with Qwen3 MoE and ML",
};

export default function FraudPage() {
  return <AIFraudDashboard />;
}
