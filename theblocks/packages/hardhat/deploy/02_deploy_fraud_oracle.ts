import { HardhatRuntimeEnvironment } from "hardhat/types";
import { DeployFunction } from "hardhat-deploy/types";

/**
 * Deploy FraudOracle - AI-Powered Fraud Detection for Stablecoin Ecosystems
 * 
 * This contract receives risk scores from off-chain ML models and enforces
 * fraud prevention directly in payment flows.
 */
const deployFraudOracle: DeployFunction = async function (hre: HardhatRuntimeEnvironment) {
  const { deployer } = await hre.getNamedAccounts();
  const { deploy } = hre.deployments;

  console.log("\n🛡️ Deploying FraudOracle - AI Fraud Detection...\n");

  const fraudOracle = await deploy("FraudOracle", {
    from: deployer,
    args: [],
    log: true,
    autoMine: true,
  });

  console.log(`\n✅ FraudOracle deployed at: ${fraudOracle.address}`);

  // Get the deployed contract for configuration
  const FraudOracle = await hre.ethers.getContractAt("FraudOracle", fraudOracle.address);

  // Log initial configuration
  const blockThreshold = await FraudOracle.blockThreshold();
  const reviewThreshold = await FraudOracle.reviewThreshold();
  const monitorThreshold = await FraudOracle.monitorThreshold();
  const modelVersion = await FraudOracle.currentModelVersion();

  console.log("\n📊 FraudOracle Configuration:");
  console.log(`   Block Threshold:   ${blockThreshold} (transactions above this are blocked)`);
  console.log(`   Review Threshold:  ${reviewThreshold} (transactions above this are flagged)`);
  console.log(`   Monitor Threshold: ${monitorThreshold} (transactions above this get enhanced monitoring)`);
  console.log(`   Model Version:     ${modelVersion}`);

  // Add some known bad actor addresses (examples - in production these come from Chainalysis)
  console.log("\n🚨 Adding example known bad actors...");
  
  // These are example addresses - in production, you'd fetch from Chainalysis, Elliptic, etc.
  const knownBadActors = [
    "0x0000000000000000000000000000000000000BAD", // Example
    "0x00000000000000000000000000000000DEADBEEF", // Example
  ];

  for (const badActor of knownBadActors) {
    try {
      // Hash the address to add to known bad actors
      const actorHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(badActor.toLowerCase()));
      await FraudOracle.addKnownBadActor(actorHash);
      console.log(`   Added: ${badActor.slice(0, 10)}...`);
    } catch {
      console.log(`   Skipped: ${badActor.slice(0, 10)}... (may already exist)`);
    }
  }

  console.log("\n🎉 FraudOracle deployment complete!");
  console.log("\n📋 Next Steps:");
  console.log("   1. Start the Python fraud detection service: uvicorn fraudApi:app --port 8000");
  console.log("   2. Configure the AI Oracle role for your backend service");
  console.log("   3. Integrate with PayFlowCore for automatic fraud checking");
  console.log("");
};

export default deployFraudOracle;

deployFraudOracle.tags = ["FraudOracle"];
