import AIFraudDashboard from "~~/components/fraud/AIFraudDashboard";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Fraud Detection | PayFlow",
  description: "Real-time AI-powered fraud detection for stablecoin transactions with GPT-4 and ML",
};

export default function FraudPage() {
  return <AIFraudDashboard />;
}
