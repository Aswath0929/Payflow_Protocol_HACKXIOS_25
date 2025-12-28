import { HardhatRuntimeEnvironment } from "hardhat/types";
import { DeployFunction } from "hardhat-deploy/types";

/**
 * Deploy SecureAIOracle - On-chain signature verification for AI fraud detection
 * 
 * This contract:
 * 1. Verifies cryptographic signatures from off-chain AI oracles
 * 2. Stores risk assessments on-chain
 * 3. Blocks/flags high-risk transactions
 * 
 * SETUP AFTER DEPLOYMENT:
 * 1. Register your oracle address: secureOracle.registerOracle(oracleAddress, trustScore)
 * 2. Configure thresholds if needed
 */
const deploySecureAIOracle: DeployFunction = async function (hre: HardhatRuntimeEnvironment) {
  const { deployer } = await hre.getNamedAccounts();
  const { deploy, log } = hre.deployments;

  log("\n" + "=".repeat(60));
  log("🔐 Deploying SecureAIOracle...");
  log("=".repeat(60));

  const secureOracle = await deploy("SecureAIOracle", {
    from: deployer,
    args: [],
    log: true,
    autoMine: true,
  });

  if (secureOracle.newlyDeployed) {
    log("\n✅ SecureAIOracle deployed at:", secureOracle.address);
    log("\n📋 Post-deployment steps:");
    log("1. Generate oracle private key (if not done):");
    log('   python -c "from eth_account import Account; a = Account.create(); print(f\'Private: {a.key.hex()}\\nAddress: {a.address}\')"');
    log("\n2. Register oracle in contract:");
    log(`   const oracle = await ethers.getContractAt("SecureAIOracle", "${secureOracle.address}")`);
    log('   await oracle.registerOracle("0xYOUR_ORACLE_ADDRESS", 8000)');
    log("\n3. Add to .env.local:");
    log("   ORACLE_PRIVATE_KEY=0x...");
    log("   OPENAI_API_KEY=sk-...");
    log("\n4. Start oracle backend:");
    log("   cd packages/nextjs/services");
    log("   uvicorn secureAiOracle:app --port 8001");
    log("\n" + "=".repeat(60));
  }

  // Try to verify on Etherscan
  if (hre.network.name !== "localhost" && hre.network.name !== "hardhat") {
    try {
      log("\n🔍 Verifying on Etherscan...");
      await hre.run("verify:verify", {
        address: secureOracle.address,
        constructorArguments: [],
      });
      log("✅ Contract verified on Etherscan");
    } catch (error: any) {
      if (error.message.includes("Already Verified")) {
        log("✅ Contract already verified");
      } else {
        log("⚠️ Verification failed:", error.message);
      }
    }
  }

  return true;
};

deploySecureAIOracle.tags = ["SecureAIOracle"];
deploySecureAIOracle.id = "deploy_secure_ai_oracle";

export default deploySecureAIOracle;
