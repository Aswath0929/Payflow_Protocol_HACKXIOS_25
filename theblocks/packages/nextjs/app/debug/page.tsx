import { DebugContracts } from "./_components/DebugContracts";
import type { NextPage } from "next";
import { PageBackground } from "~~/components/ui/PageBackground";
import { getMetadata } from "~~/utils/scaffold-eth/getMetadata";

export const metadata = getMetadata({
  title: "Debug Contracts",
  description: "Debug your deployed 🏗 Scaffold-ETH 2 contracts in an easy way",
});

const Debug: NextPage = () => {
  return (
    <div className="min-h-screen relative">
      {/* Matrix-Style Debug Background */}
      <PageBackground theme="debug" intensity="low" />

      <div className="relative z-10">
        <DebugContracts />
        <div className="text-center mt-8 bg-secondary/80 backdrop-blur-sm p-10 border border-green-500/20">
          <h1 className="text-4xl my-0">Debug Contracts</h1>
          <p className="text-neutral">
            You can debug & interact with your deployed contracts here.
            <br /> Check{" "}
            <code className="italic bg-base-300 text-base font-bold [word-spacing:-0.5rem] px-1">
              packages / nextjs / app / debug / page.tsx
            </code>{" "}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Debug;
