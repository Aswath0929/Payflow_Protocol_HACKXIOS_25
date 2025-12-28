import { HardhatRuntimeEnvironment } from "hardhat/types";
import { DeployFunction } from "hardhat-deploy/types";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                     PAYFLOW SECURE FRAUD ORACLE DEPLOYMENT                            ║
 * ║                                                                                       ║
 * ║   Deploys the complete AI Fraud Detection infrastructure:                            ║
 * ║   • SecureFraudOracle - On-chain signature verification                              ║
 * ║   • PayFlowFraudGateway - Integration with PayFlowCore                               ║
 * ║                                                                                       ║
 * ║   Post-deployment:                                                                    ║
 * ║   1. Register AI Oracle signing address                                              ║
 * ║   2. Configure thresholds                                                            ║
 * ║   3. Start Python AI service                                                         ║
 * ║                                                                                       ║
 * ║   Hackxios 2K25 - PayFlow Protocol                                                   ║
 * ╚═══════════════════════════════════════════════════════════════════════════════════════╝
 */

const deploySecureFraudOracle: DeployFunction = async function (hre: HardhatRuntimeEnvironment) {
  const { deployer } = await hre.getNamedAccounts();
  const { deploy, get, execute } = hre.deployments;

  console.log("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
  console.log("║                   DEPLOYING SECURE AI FRAUD ORACLE SYSTEM                    ║");
  console.log("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

  console.log("📍 Deployer:", deployer);
  console.log("🌐 Network:", hre.network.name);
  console.log("");

  // ═══════════════════════════════════════════════════════════════════════════════
  //                         DEPLOY SECURE FRAUD ORACLE
  // ═══════════════════════════════════════════════════════════════════════════════

  console.log("🔐 Deploying SecureFraudOracle...");
  
  const secureFraudOracle = await deploy("SecureFraudOracle", {
    from: deployer,
    args: [deployer], // Admin address
    log: true,
    autoMine: true,
    waitConfirmations: hre.network.name === "localhost" ? 1 : 2,
  });

  console.log("✅ SecureFraudOracle deployed at:", secureFraudOracle.address);
  console.log("   Gas used:", secureFraudOracle.receipt?.gasUsed?.toString());

  // ═══════════════════════════════════════════════════════════════════════════════
  //                         DEPLOY FRAUD GATEWAY
  // ═══════════════════════════════════════════════════════════════════════════════

  console.log("\n🌉 Deploying PayFlowFraudGateway...");

  // Get PayFlowCore and AuditRegistry addresses
  let payFlowCoreAddress: string;
  let auditRegistryAddress: string;

  try {
    const payFlowCore = await get("PayFlowCore");
    payFlowCoreAddress = payFlowCore.address;
    console.log("   Found PayFlowCore at:", payFlowCoreAddress);
  } catch (e) {
    console.log("   ⚠️  PayFlowCore not found, using zero address");
    payFlowCoreAddress = "0x0000000000000000000000000000000000000000";
  }

  try {
    const auditRegistry = await get("AuditRegistry");
    auditRegistryAddress = auditRegistry.address;
    console.log("   Found AuditRegistry at:", auditRegistryAddress);
  } catch (e) {
    console.log("   ⚠️  AuditRegistry not found, using zero address");
    auditRegistryAddress = "0x0000000000000000000000000000000000000000";
  }

  const fraudGateway = await deploy("PayFlowFraudGateway", {
    from: deployer,
    args: [
      secureFraudOracle.address,  // Fraud Oracle
      payFlowCoreAddress,          // PayFlowCore
      auditRegistryAddress,        // AuditRegistry
      deployer,                    // Admin
    ],
    log: true,
    autoMine: true,
    waitConfirmations: hre.network.name === "localhost" ? 1 : 2,
  });

  console.log("✅ PayFlowFraudGateway deployed at:", fraudGateway.address);
  console.log("   Gas used:", fraudGateway.receipt?.gasUsed?.toString());

  // ═══════════════════════════════════════════════════════════════════════════════
  //                         CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════════════════

  console.log("\n⚙️  Configuring SecureFraudOracle...");

  // The AI Oracle's signing address - in production, this would be configured
  // For now, we'll use the deployer as the oracle
  // You should replace this with the actual oracle address from the Python service
  const ORACLE_ADDRESS = process.env.AI_ORACLE_ADDRESS || deployer;

  console.log("   Registering oracle address:", ORACLE_ADDRESS);
  
  try {
    await execute(
      "SecureFraudOracle",
      { from: deployer, log: true },
      "registerOracle",
      ORACLE_ADDRESS
    );
    console.log("   ✅ Oracle registered successfully");
  } catch (e: any) {
    if (e.message.includes("Already registered")) {
      console.log("   ℹ️  Oracle already registered");
    } else {
      console.log("   ⚠️  Failed to register oracle:", e.message);
    }
  }

  // Set thresholds
  console.log("   Setting thresholds (Block: 80, Review: 60)...");
  try {
    await execute(
      "SecureFraudOracle",
      { from: deployer, log: true },
      "updateThresholds",
      80, // Block threshold
      60  // Review threshold
    );
    console.log("   ✅ Thresholds configured");
  } catch (e: any) {
    console.log("   ⚠️  Failed to set thresholds:", e.message);
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //                         SUMMARY
  // ═══════════════════════════════════════════════════════════════════════════════

  console.log("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
  console.log("║                         DEPLOYMENT SUMMARY                                   ║");
  console.log("╠═══════════════════════════════════════════════════════════════════════════════╣");
  console.log(`║  SecureFraudOracle:    ${secureFraudOracle.address}  ║`);
  console.log(`║  PayFlowFraudGateway:  ${fraudGateway.address}  ║`);
  console.log("╠═══════════════════════════════════════════════════════════════════════════════╣");
  console.log("║                          NEXT STEPS                                          ║");
  console.log("╠═══════════════════════════════════════════════════════════════════════════════╣");
  console.log("║  1. Set ORACLE_PRIVATE_KEY in .env for Python service                        ║");
  console.log("║  2. Set OPENAI_API_KEY in .env for GPT-4 analysis                           ║");
  console.log("║  3. Start AI Oracle: cd packages/nextjs/services/ai && python api.py        ║");
  console.log("║  4. Update frontend with contract addresses                                  ║");
  console.log("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

  // Verify contracts on Etherscan (if not localhost)
  if (hre.network.name !== "localhost" && hre.network.name !== "hardhat") {
    console.log("🔍 Verifying contracts on Etherscan...");
    
    try {
      await hre.run("verify:verify", {
        address: secureFraudOracle.address,
        constructorArguments: [deployer],
      });
      console.log("   ✅ SecureFraudOracle verified");
    } catch (e: any) {
      console.log("   ⚠️  Verification failed:", e.message);
    }

    try {
      await hre.run("verify:verify", {
        address: fraudGateway.address,
        constructorArguments: [
          secureFraudOracle.address,
          payFlowCoreAddress,
          auditRegistryAddress,
          deployer,
        ],
      });
      console.log("   ✅ PayFlowFraudGateway verified");
    } catch (e: any) {
      console.log("   ⚠️  Verification failed:", e.message);
    }
  }

  return true;
};

export default deploySecureFraudOracle;

deploySecureFraudOracle.id = "deploy_secure_fraud_oracle";
deploySecureFraudOracle.tags = ["SecureFraudOracle", "PayFlowFraudGateway", "AI", "Fraud"];
deploySecureFraudOracle.dependencies = ["PayFlowProtocol"];
