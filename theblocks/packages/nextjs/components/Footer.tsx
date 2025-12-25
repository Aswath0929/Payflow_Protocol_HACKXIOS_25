import React from "react";
import Link from "next/link";
import { useFetchNativeCurrencyPrice } from "@scaffold-ui/hooks";
import { hardhat } from "viem/chains";
import { CurrencyDollarIcon, MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import { SwitchTheme } from "~~/components/SwitchTheme";
import { Faucet } from "~~/components/scaffold-eth";
import { useTargetNetwork } from "~~/hooks/scaffold-eth/useTargetNetwork";

/**
 * Site footer - Premium PayFlow Design
 */
export const Footer = () => {
  const { targetNetwork } = useTargetNetwork();
  const isLocalNetwork = targetNetwork.id === hardhat.id;
  const { price: nativeCurrencyPrice } = useFetchNativeCurrencyPrice();

  return (
    <div className="min-h-0 py-5 px-1 mb-11 lg:mb-0 bg-[#0a0a0f]">
      <div>
        <div className="fixed flex justify-between items-center w-full z-10 p-4 bottom-0 left-0 pointer-events-none">
          <div className="flex flex-col md:flex-row gap-2 pointer-events-auto">
            {nativeCurrencyPrice > 0 && (
              <div>
                <div className="px-4 py-2 rounded-xl bg-zinc-900/80 backdrop-blur-sm border border-white/10 text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <CurrencyDollarIcon className="h-4 w-4 text-green-400" />
                  <span>{nativeCurrencyPrice.toFixed(2)}</span>
                </div>
              </div>
            )}
            {isLocalNetwork && (
              <>
                <Faucet />
                <Link
                  href="/blockexplorer"
                  passHref
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-violet-600/20 to-cyan-600/20 border border-violet-500/30 text-sm font-medium text-white flex items-center gap-2 hover:opacity-80 transition-opacity"
                >
                  <MagnifyingGlassIcon className="h-4 w-4" />
                  <span>Block Explorer</span>
                </Link>
              </>
            )}
          </div>
          <SwitchTheme className={`pointer-events-auto ${isLocalNetwork ? "self-end md:self-auto" : ""}`} />
        </div>
      </div>
      <div className="w-full border-t border-white/5 pt-4">
        <div className="flex justify-center items-center gap-4 text-sm text-zinc-500">
          <span className="bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent font-semibold">
            💎 PayFlow Protocol
          </span>
          <span>•</span>
          <span>Hackxios 2K25</span>
          <span>•</span>
          <span>Built for PayPal & Visa</span>
          <span>•</span>
          <a
            href="https://sepolia.etherscan.io/address/0x4c9489812a9D971b431B9d99049a42B437347dBC"
            target="_blank"
            rel="noreferrer"
            className="text-violet-400 hover:text-violet-300 transition-colors"
          >
            View on Etherscan ↗
          </a>
        </div>
      </div>
    </div>
  );
};
