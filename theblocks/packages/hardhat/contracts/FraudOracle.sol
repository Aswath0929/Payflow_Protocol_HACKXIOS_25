// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║                         FRAUD ORACLE                                          ║
 * ║              AI-Powered Fraud Detection for Stablecoin Ecosystems            ║
 * ╠═══════════════════════════════════════════════════════════════════════════════╣
 * ║  "Making stablecoins safer than traditional banking"                          ║
 * ║                                                                               ║
 * ║  FraudOracle receives AI-computed risk scores from off-chain ML models       ║
 * ║  and enforces fraud prevention directly in payment flows:                     ║
 * ║  • Real-time transaction risk scoring                                         ║
 * ║  • Behavioral anomaly detection                                               ║
 * ║  • Wallet clustering & mixing service detection                               ║
 * ║  • Transaction velocity monitoring                                            ║
 * ║  • Graph-based relationship analysis                                          ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

/**
 * @title FraudOracle
 * @notice AI-powered fraud detection oracle for stablecoin payment infrastructure
 * @dev Receives off-chain ML risk scores and enforces on-chain fraud prevention
 * 
 * ARCHITECTURE:
 * ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
 * │  Transaction    │───▶│   AI/ML Model   │───▶│  FraudOracle    │
 * │  (mempool/new)  │    │  (off-chain)    │    │  (on-chain)     │
 * └─────────────────┘    └─────────────────┘    └────────┬────────┘
 *                                                        │
 *                        ┌───────────────────────────────┘
 *                        ▼
 *         ┌──────────────────────────────────────┐
 *         │  PayFlowCore / ComplianceEngine      │
 *         │  (Block or Allow Transaction)        │
 *         └──────────────────────────────────────┘
 */
contract FraudOracle is AccessControl, ReentrancyGuard {
    
    // ═══════════════════════════════════════════════════════════════════════════
    //                              ROLES
    // ═══════════════════════════════════════════════════════════════════════════
    
    bytes32 public constant AI_ORACLE_ROLE = keccak256("AI_ORACLE_ROLE");
    bytes32 public constant FRAUD_ANALYST_ROLE = keccak256("FRAUD_ANALYST_ROLE");
    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");

    // ═══════════════════════════════════════════════════════════════════════════
    //                              TYPES
    // ═══════════════════════════════════════════════════════════════════════════

    enum RiskLevel {
        SAFE,           // 0-20: Normal transaction
        LOW,            // 21-40: Minor anomalies
        MEDIUM,         // 41-60: Requires monitoring
        HIGH,           // 61-80: Requires review
        CRITICAL        // 81-100: Block transaction
    }

    enum AlertType {
        VELOCITY_ANOMALY,       // Unusual transaction frequency
        AMOUNT_ANOMALY,         // Unusual transaction amounts
        PATTERN_ANOMALY,        // Unusual behavior patterns
        MIXING_DETECTED,        // Potential mixing service
        SANCTIONED_INTERACTION, // Interacted with sanctioned address
        DUST_ATTACK,            // Dust attack detected
        SYBIL_ATTACK,           // Multiple wallets, same entity
        FLASH_LOAN_ATTACK,      // Flash loan manipulation
        WASH_TRADING,           // Wash trading detected
        LAYERING                // Transaction layering (money laundering)
    }

    struct RiskProfile {
        address wallet;
        uint8 currentRiskScore;         // 0-100
        RiskLevel riskLevel;
        
        // Historical scores (for trend analysis)
        uint8 avg7DayScore;
        uint8 avg30DayScore;
        uint8 peakScore;
        
        // Flags
        bool isBlacklisted;
        bool isWhitelisted;
        bool underInvestigation;
        
        // ML Model info
        bytes32 lastModelVersion;       // Which model computed the score
        uint256 lastUpdated;
        uint256 scoreCount;             // Number of times scored
        
        // Behavioral baseline
        uint256 avgTxAmount;
        uint256 avgTxFrequency;         // Avg seconds between transactions
        uint256 totalVolume;
        uint256 transactionCount;
    }

    struct FraudAlert {
        bytes32 alertId;
        address subject;
        AlertType alertType;
        uint8 severity;                 // 0-100
        
        // Context
        bytes32 transactionId;
        uint256 amount;
        address counterparty;
        
        // AI analysis
        string aiExplanation;           // Human-readable explanation
        bytes32 evidenceHash;           // Hash of detailed evidence
        
        // Status
        bool acknowledged;
        bool resolved;
        address resolvedBy;
        string resolution;
        
        // Timestamps
        uint256 detectedAt;
        uint256 resolvedAt;
    }

    struct TransactionAnalysis {
        bytes32 transactionId;
        address sender;
        address recipient;
        uint256 amount;
        
        // Risk scores
        uint8 senderRiskScore;
        uint8 recipientRiskScore;
        uint8 transactionRiskScore;
        uint8 overallRiskScore;
        
        // Feature scores (0-100 each)
        uint8 velocityScore;            // Transaction frequency risk
        uint8 amountScore;              // Amount anomaly risk
        uint8 patternScore;             // Behavioral pattern risk
        uint8 graphScore;               // Wallet relationship risk
        uint8 timingScore;              // Time-of-day/week risk
        
        // Flags
        bool approved;
        bool flaggedForReview;
        bool blocked;
        
        // Metadata
        uint256 analysisTimestamp;
        bytes32 modelVersion;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                              STATE
    // ═══════════════════════════════════════════════════════════════════════════

    // Risk profiles
    mapping(address => RiskProfile) public riskProfiles;
    address[] public monitoredWallets;
    
    // Fraud alerts
    mapping(bytes32 => FraudAlert) public alerts;
    bytes32[] public allAlerts;
    mapping(address => bytes32[]) public walletAlerts;
    
    // Transaction analyses
    mapping(bytes32 => TransactionAnalysis) public analyses;
    bytes32[] public allAnalyses;
    
    // Blacklists & Whitelists
    mapping(address => bool) public blacklistedAddresses;
    mapping(address => bool) public whitelistedAddresses;
    
    // Known bad actors (from public lists + internal detection)
    mapping(bytes32 => bool) public knownBadActorHashes;
    
    // Configuration
    uint8 public blockThreshold = 80;           // Block transactions above this score
    uint8 public reviewThreshold = 60;          // Flag for review above this
    uint8 public monitorThreshold = 40;         // Enhanced monitoring above this
    
    // Statistics
    uint256 public totalAnalyses;
    uint256 public totalBlocked;
    uint256 public totalAlerts;
    uint256 public totalFalsePositives;
    
    // Model tracking
    bytes32 public currentModelVersion;
    uint256 public modelUpdatedAt;

    // ═══════════════════════════════════════════════════════════════════════════
    //                              EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event RiskScoreUpdated(
        address indexed wallet,
        uint8 oldScore,
        uint8 newScore,
        RiskLevel riskLevel,
        bytes32 modelVersion
    );

    event TransactionAnalyzed(
        bytes32 indexed transactionId,
        address indexed sender,
        address indexed recipient,
        uint8 overallRisk,
        bool approved
    );

    event FraudAlertRaised(
        bytes32 indexed alertId,
        address indexed subject,
        AlertType alertType,
        uint8 severity,
        bytes32 transactionId
    );

    event FraudAlertResolved(
        bytes32 indexed alertId,
        address indexed resolver,
        bool wasFraud,
        string resolution
    );

    event WalletBlacklisted(
        address indexed wallet,
        string reason,
        address indexed blacklistedBy
    );

    event WalletWhitelisted(
        address indexed wallet,
        address indexed whitelistedBy
    );

    event TransactionBlocked(
        bytes32 indexed transactionId,
        address indexed sender,
        address indexed recipient,
        uint8 riskScore,
        string reason
    );

    event ModelVersionUpdated(
        bytes32 oldVersion,
        bytes32 newVersion,
        uint256 timestamp
    );

    event ThresholdsUpdated(
        uint8 blockThreshold,
        uint8 reviewThreshold,
        uint8 monitorThreshold
    );

    // ═══════════════════════════════════════════════════════════════════════════
    //                              CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(AI_ORACLE_ROLE, msg.sender);
        _grantRole(FRAUD_ANALYST_ROLE, msg.sender);
        _grantRole(EMERGENCY_ROLE, msg.sender);
        
        currentModelVersion = keccak256("PayFlow-FraudML-v1.0.0");
        modelUpdatedAt = block.timestamp;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         AI ORACLE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Update risk score for a wallet (called by AI oracle)
     * @param wallet The wallet address to update
     * @param riskScore The new risk score (0-100)
     * @param modelVersion The ML model version that computed this score
     */
    function updateRiskScore(
        address wallet,
        uint8 riskScore,
        bytes32 modelVersion
    ) external onlyRole(AI_ORACLE_ROLE) {
        require(wallet != address(0), "Invalid wallet");
        require(riskScore <= 100, "Score must be 0-100");
        
        RiskProfile storage profile = riskProfiles[wallet];
        uint8 oldScore = profile.currentRiskScore;
        
        // Initialize if new wallet
        if (profile.wallet == address(0)) {
            profile.wallet = wallet;
            monitoredWallets.push(wallet);
        }
        
        // Update score
        profile.currentRiskScore = riskScore;
        profile.riskLevel = _scoreToLevel(riskScore);
        profile.lastModelVersion = modelVersion;
        profile.lastUpdated = block.timestamp;
        profile.scoreCount++;
        
        // Update peak
        if (riskScore > profile.peakScore) {
            profile.peakScore = riskScore;
        }
        
        emit RiskScoreUpdated(wallet, oldScore, riskScore, profile.riskLevel, modelVersion);
        
        // Auto-actions based on score
        if (riskScore >= blockThreshold && !profile.isWhitelisted) {
            _autoBlacklist(wallet, "AI detected critical risk");
        }
    }

    /**
     * @notice Submit batch risk score updates (gas efficient)
     */
    function batchUpdateRiskScores(
        address[] calldata wallets,
        uint8[] calldata scores,
        bytes32 modelVersion
    ) external onlyRole(AI_ORACLE_ROLE) {
        require(wallets.length == scores.length, "Array length mismatch");
        require(wallets.length <= 100, "Batch too large");
        
        for (uint256 i = 0; i < wallets.length; i++) {
            if (wallets[i] != address(0) && scores[i] <= 100) {
                RiskProfile storage profile = riskProfiles[wallets[i]];
                uint8 oldScore = profile.currentRiskScore;
                
                if (profile.wallet == address(0)) {
                    profile.wallet = wallets[i];
                    monitoredWallets.push(wallets[i]);
                }
                
                profile.currentRiskScore = scores[i];
                profile.riskLevel = _scoreToLevel(scores[i]);
                profile.lastModelVersion = modelVersion;
                profile.lastUpdated = block.timestamp;
                profile.scoreCount++;
                
                emit RiskScoreUpdated(wallets[i], oldScore, scores[i], profile.riskLevel, modelVersion);
            }
        }
    }

    /**
     * @notice Analyze a transaction and return risk assessment
     * @param transactionId Unique transaction identifier
     * @param sender Transaction sender
     * @param recipient Transaction recipient
     * @param amount Transaction amount
     * @param velocityScore AI-computed velocity risk (0-100)
     * @param amountScore AI-computed amount anomaly risk (0-100)
     * @param patternScore AI-computed pattern risk (0-100)
     * @param graphScore AI-computed graph/relationship risk (0-100)
     * @param timingScore AI-computed timing risk (0-100)
     */
    function analyzeTransaction(
        bytes32 transactionId,
        address sender,
        address recipient,
        uint256 amount,
        uint8 velocityScore,
        uint8 amountScore,
        uint8 patternScore,
        uint8 graphScore,
        uint8 timingScore
    ) external onlyRole(AI_ORACLE_ROLE) returns (bool approved, uint8 overallRisk) {
        require(sender != address(0) && recipient != address(0), "Invalid addresses");
        
        // Get sender and recipient risk profiles
        uint8 senderRisk = riskProfiles[sender].currentRiskScore;
        uint8 recipientRisk = riskProfiles[recipient].currentRiskScore;
        
        // Calculate transaction risk (weighted average of feature scores)
        uint8 txRisk = uint8(
            (uint256(velocityScore) * 25 +
             uint256(amountScore) * 25 +
             uint256(patternScore) * 20 +
             uint256(graphScore) * 20 +
             uint256(timingScore) * 10) / 100
        );
        
        // Overall risk combines sender, recipient, and transaction risk
        overallRisk = uint8(
            (uint256(senderRisk) * 30 +
             uint256(recipientRisk) * 30 +
             uint256(txRisk) * 40) / 100
        );
        
        // Determine action
        approved = true;
        bool flagged = false;
        bool blocked = false;
        
        // Check blacklists
        if (blacklistedAddresses[sender] || blacklistedAddresses[recipient]) {
            approved = false;
            blocked = true;
            overallRisk = 100;
        }
        // Check thresholds
        else if (overallRisk >= blockThreshold) {
            approved = false;
            blocked = true;
        } else if (overallRisk >= reviewThreshold) {
            flagged = true;
        }
        
        // Whitelist override (but still flag for audit)
        if (whitelistedAddresses[sender] && whitelistedAddresses[recipient]) {
            approved = true;
            blocked = false;
        }
        
        // Store analysis
        TransactionAnalysis storage analysis = analyses[transactionId];
        analysis.transactionId = transactionId;
        analysis.sender = sender;
        analysis.recipient = recipient;
        analysis.amount = amount;
        analysis.senderRiskScore = senderRisk;
        analysis.recipientRiskScore = recipientRisk;
        analysis.transactionRiskScore = txRisk;
        analysis.overallRiskScore = overallRisk;
        analysis.velocityScore = velocityScore;
        analysis.amountScore = amountScore;
        analysis.patternScore = patternScore;
        analysis.graphScore = graphScore;
        analysis.timingScore = timingScore;
        analysis.approved = approved;
        analysis.flaggedForReview = flagged;
        analysis.blocked = blocked;
        analysis.analysisTimestamp = block.timestamp;
        analysis.modelVersion = currentModelVersion;
        
        allAnalyses.push(transactionId);
        totalAnalyses++;
        
        if (blocked) {
            totalBlocked++;
            emit TransactionBlocked(
                transactionId,
                sender,
                recipient,
                overallRisk,
                "AI risk threshold exceeded"
            );
        }
        
        emit TransactionAnalyzed(transactionId, sender, recipient, overallRisk, approved);
        
        return (approved, overallRisk);
    }

    /**
     * @notice Raise a fraud alert
     */
    function raiseAlert(
        address subject,
        AlertType alertType,
        uint8 severity,
        bytes32 transactionId,
        uint256 amount,
        address counterparty,
        string calldata explanation,
        bytes32 evidenceHash
    ) external onlyRole(AI_ORACLE_ROLE) returns (bytes32 alertId) {
        alertId = keccak256(abi.encodePacked(
            subject,
            alertType,
            transactionId,
            block.timestamp,
            totalAlerts
        ));
        
        FraudAlert storage alert = alerts[alertId];
        alert.alertId = alertId;
        alert.subject = subject;
        alert.alertType = alertType;
        alert.severity = severity;
        alert.transactionId = transactionId;
        alert.amount = amount;
        alert.counterparty = counterparty;
        alert.aiExplanation = explanation;
        alert.evidenceHash = evidenceHash;
        alert.detectedAt = block.timestamp;
        
        allAlerts.push(alertId);
        walletAlerts[subject].push(alertId);
        totalAlerts++;
        
        // Update risk profile
        RiskProfile storage profile = riskProfiles[subject];
        if (severity > profile.currentRiskScore) {
            profile.currentRiskScore = severity;
            profile.riskLevel = _scoreToLevel(severity);
        }
        profile.underInvestigation = true;
        
        emit FraudAlertRaised(alertId, subject, alertType, severity, transactionId);
        
        return alertId;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ANALYST FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Resolve a fraud alert
     */
    function resolveAlert(
        bytes32 alertId,
        bool wasFraud,
        string calldata resolution
    ) external onlyRole(FRAUD_ANALYST_ROLE) {
        FraudAlert storage alert = alerts[alertId];
        require(alert.alertId != bytes32(0), "Alert not found");
        require(!alert.resolved, "Already resolved");
        
        alert.resolved = true;
        alert.resolvedBy = msg.sender;
        alert.resolution = resolution;
        alert.resolvedAt = block.timestamp;
        
        // Update profile
        RiskProfile storage profile = riskProfiles[alert.subject];
        profile.underInvestigation = false;
        
        if (!wasFraud) {
            totalFalsePositives++;
            // Consider reducing risk score for false positives
            if (profile.currentRiskScore >= 20) {
                profile.currentRiskScore -= 20;
                profile.riskLevel = _scoreToLevel(profile.currentRiskScore);
            }
        }
        
        emit FraudAlertResolved(alertId, msg.sender, wasFraud, resolution);
    }

    /**
     * @notice Manually blacklist a wallet
     */
    function blacklistWallet(
        address wallet,
        string calldata reason
    ) external onlyRole(FRAUD_ANALYST_ROLE) {
        require(wallet != address(0), "Invalid wallet");
        blacklistedAddresses[wallet] = true;
        
        RiskProfile storage profile = riskProfiles[wallet];
        profile.isBlacklisted = true;
        profile.currentRiskScore = 100;
        profile.riskLevel = RiskLevel.CRITICAL;
        
        emit WalletBlacklisted(wallet, reason, msg.sender);
    }

    /**
     * @notice Whitelist a trusted wallet
     */
    function whitelistWallet(
        address wallet
    ) external onlyRole(FRAUD_ANALYST_ROLE) {
        require(wallet != address(0), "Invalid wallet");
        whitelistedAddresses[wallet] = true;
        
        RiskProfile storage profile = riskProfiles[wallet];
        profile.isWhitelisted = true;
        
        emit WalletWhitelisted(wallet, msg.sender);
    }

    /**
     * @notice Remove from blacklist
     */
    function removeFromBlacklist(address wallet) external onlyRole(FRAUD_ANALYST_ROLE) {
        blacklistedAddresses[wallet] = false;
        riskProfiles[wallet].isBlacklisted = false;
    }

    /**
     * @notice Remove from whitelist
     */
    function removeFromWhitelist(address wallet) external onlyRole(FRAUD_ANALYST_ROLE) {
        whitelistedAddresses[wallet] = false;
        riskProfiles[wallet].isWhitelisted = false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Update ML model version
     */
    function updateModelVersion(bytes32 newVersion) external onlyRole(DEFAULT_ADMIN_ROLE) {
        bytes32 oldVersion = currentModelVersion;
        currentModelVersion = newVersion;
        modelUpdatedAt = block.timestamp;
        
        emit ModelVersionUpdated(oldVersion, newVersion, block.timestamp);
    }

    /**
     * @notice Update risk thresholds
     */
    function updateThresholds(
        uint8 _blockThreshold,
        uint8 _reviewThreshold,
        uint8 _monitorThreshold
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_blockThreshold > _reviewThreshold, "Block must be higher than review");
        require(_reviewThreshold > _monitorThreshold, "Review must be higher than monitor");
        
        blockThreshold = _blockThreshold;
        reviewThreshold = _reviewThreshold;
        monitorThreshold = _monitorThreshold;
        
        emit ThresholdsUpdated(_blockThreshold, _reviewThreshold, _monitorThreshold);
    }

    /**
     * @notice Add known bad actor hash (from external lists)
     */
    function addKnownBadActor(bytes32 actorHash) external onlyRole(FRAUD_ANALYST_ROLE) {
        knownBadActorHashes[actorHash] = true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Check if a transaction should be approved
     */
    function shouldApproveTransaction(
        address sender,
        address recipient,
        uint256 /* amount */
    ) external view returns (bool approved, uint8 riskScore, string memory reason) {
        // Check blacklists first
        if (blacklistedAddresses[sender]) {
            return (false, 100, "Sender is blacklisted");
        }
        if (blacklistedAddresses[recipient]) {
            return (false, 100, "Recipient is blacklisted");
        }
        
        // Get risk scores
        uint8 senderRisk = riskProfiles[sender].currentRiskScore;
        uint8 recipientRisk = riskProfiles[recipient].currentRiskScore;
        riskScore = (senderRisk + recipientRisk) / 2;
        
        // Whitelist override
        if (whitelistedAddresses[sender] && whitelistedAddresses[recipient]) {
            return (true, riskScore, "Whitelisted parties");
        }
        
        // Check thresholds
        if (riskScore >= blockThreshold) {
            return (false, riskScore, "Risk score exceeds block threshold");
        }
        if (riskScore >= reviewThreshold) {
            return (true, riskScore, "Approved but flagged for review");
        }
        
        return (true, riskScore, "Transaction approved");
    }

    /**
     * @notice Get wallet risk profile
     */
    function getWalletRisk(address wallet) external view returns (
        uint8 riskScore,
        RiskLevel riskLevel,
        bool isBlacklisted,
        bool isWhitelisted,
        bool underInvestigation
    ) {
        RiskProfile storage profile = riskProfiles[wallet];
        return (
            profile.currentRiskScore,
            profile.riskLevel,
            profile.isBlacklisted,
            profile.isWhitelisted,
            profile.underInvestigation
        );
    }

    /**
     * @notice Get transaction analysis
     */
    function getTransactionAnalysis(bytes32 transactionId) external view returns (
        uint8 overallRisk,
        bool approved,
        bool flagged,
        bool blocked,
        uint8 velocityScore,
        uint8 amountScore,
        uint8 patternScore,
        uint8 graphScore,
        uint8 timingScore
    ) {
        TransactionAnalysis storage analysis = analyses[transactionId];
        return (
            analysis.overallRiskScore,
            analysis.approved,
            analysis.flaggedForReview,
            analysis.blocked,
            analysis.velocityScore,
            analysis.amountScore,
            analysis.patternScore,
            analysis.graphScore,
            analysis.timingScore
        );
    }

    /**
     * @notice Get fraud statistics
     */
    function getStatistics() external view returns (
        uint256 _totalAnalyses,
        uint256 _totalBlocked,
        uint256 _totalAlerts,
        uint256 _totalFalsePositives,
        uint256 _monitoredWallets
    ) {
        return (
            totalAnalyses,
            totalBlocked,
            totalAlerts,
            totalFalsePositives,
            monitoredWallets.length
        );
    }

    /**
     * @notice Get recent alerts (for dashboard)
     */
    function getRecentAlerts(uint256 count) external view returns (bytes32[] memory) {
        uint256 total = allAlerts.length;
        if (count > total) count = total;
        
        bytes32[] memory recent = new bytes32[](count);
        for (uint256 i = 0; i < count; i++) {
            recent[i] = allAlerts[total - 1 - i];
        }
        return recent;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function _scoreToLevel(uint8 score) internal pure returns (RiskLevel) {
        if (score <= 20) return RiskLevel.SAFE;
        if (score <= 40) return RiskLevel.LOW;
        if (score <= 60) return RiskLevel.MEDIUM;
        if (score <= 80) return RiskLevel.HIGH;
        return RiskLevel.CRITICAL;
    }

    function _autoBlacklist(address wallet, string memory reason) internal {
        if (!blacklistedAddresses[wallet] && !whitelistedAddresses[wallet]) {
            blacklistedAddresses[wallet] = true;
            riskProfiles[wallet].isBlacklisted = true;
            
            emit WalletBlacklisted(wallet, reason, address(this));
        }
    }
}
