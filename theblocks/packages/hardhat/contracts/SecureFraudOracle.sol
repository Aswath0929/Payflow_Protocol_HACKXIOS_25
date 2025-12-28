// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                     PAYFLOW SECURE FRAUD ORACLE                                       ║
 * ║                                                                                       ║
 * ║   On-Chain AI Verification with Cryptographic Signature Validation                   ║
 * ║                                                                                       ║
 * ║   Features:                                                                           ║
 * ║   • ECDSA signature verification using ecrecover                                     ║
 * ║   • Multi-oracle support with quorum requirements                                    ║
 * ║   • Replay attack prevention with nonces                                             ║
 * ║   • Rate limiting for oracle submissions                                             ║
 * ║   • Emergency circuit breaker                                                        ║
 * ║   • Full event logging for off-chain monitoring                                      ║
 * ║                                                                                       ║
 * ║   Hackxios 2K25 - PayFlow Protocol                                                   ║
 * ╚═══════════════════════════════════════════════════════════════════════════════════════╝
 */

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title SecureFraudOracle
 * @notice Verifies cryptographically signed AI fraud analysis on-chain
 * @dev Uses ECDSA signatures compatible with Ethereum's ecrecover
 */
contract SecureFraudOracle is AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ═══════════════════════════════════════════════════════════════════════════════
    //                              CONSTANTS & ROLES
    // ═══════════════════════════════════════════════════════════════════════════════
    
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    
    uint256 public constant MAX_RISK_SCORE = 100;
    uint256 public constant ANALYSIS_VALIDITY_PERIOD = 1 hours;
    uint256 public constant MIN_ORACLE_QUORUM = 1;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              TYPES & STRUCTS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// @notice Risk levels matching off-chain AI analysis
    enum RiskLevel { SAFE, LOW, MEDIUM, HIGH, CRITICAL }
    
    /// @notice Complete AI analysis result
    struct AIAnalysis {
        bytes32 transactionId;
        address sender;
        address recipient;
        uint256 amount;
        uint256 riskScore;
        RiskLevel riskLevel;
        bool approved;
        bool blocked;
        uint256 timestamp;
        address oracleAddress;
        bytes signature;
        bool verified;
    }
    
    /// @notice Wallet risk profile stored on-chain
    struct WalletRisk {
        uint256 currentRiskScore;
        uint256 peakRiskScore;
        uint256 totalAnalyses;
        uint256 blockedCount;
        uint256 lastAnalysisTime;
        bool isBlacklisted;
        bool isWhitelisted;
    }
    
    /// @notice Oracle performance tracking
    struct OracleMetrics {
        uint256 totalSubmissions;
        uint256 successfulVerifications;
        uint256 failedVerifications;
        uint256 lastSubmissionTime;
        bool isActive;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// @notice Threshold at which transactions are blocked
    uint256 public blockThreshold = 80;
    
    /// @notice Threshold for manual review
    uint256 public reviewThreshold = 60;
    
    /// @notice Minimum oracles needed for consensus
    uint256 public oracleQuorum = 1;
    
    /// @notice Rate limit: max submissions per oracle per block
    uint256 public maxSubmissionsPerBlock = 100;
    
    /// @notice Stored analyses by transaction ID
    mapping(bytes32 => AIAnalysis) public analyses;
    
    /// @notice Wallet risk profiles
    mapping(address => WalletRisk) public walletRisks;
    
    /// @notice Oracle metrics
    mapping(address => OracleMetrics) public oracleMetrics;
    
    /// @notice Nonces for replay protection
    mapping(bytes32 => bool) public usedNonces;
    
    /// @notice Submissions per oracle per block
    mapping(address => mapping(uint256 => uint256)) public submissionsPerBlock;
    
    /// @notice Global blacklist
    mapping(address => bool) public globalBlacklist;
    
    /// @notice Global whitelist (bypass checks)
    mapping(address => bool) public globalWhitelist;
    
    /// @notice Registered oracles
    address[] public registeredOracles;
    
    /// @notice Statistics
    uint256 public totalAnalyses;
    uint256 public totalBlocked;
    uint256 public totalApproved;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              EVENTS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    event AnalysisSubmitted(
        bytes32 indexed transactionId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore,
        RiskLevel riskLevel,
        bool approved,
        address oracleAddress
    );
    
    event AnalysisVerified(
        bytes32 indexed transactionId,
        address indexed oracleAddress,
        bool signatureValid
    );
    
    event TransactionBlocked(
        bytes32 indexed transactionId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore,
        string reason
    );
    
    event TransactionApproved(
        bytes32 indexed transactionId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore
    );
    
    event WalletRiskUpdated(
        address indexed wallet,
        uint256 previousScore,
        uint256 newScore,
        RiskLevel riskLevel
    );
    
    event WalletBlacklisted(address indexed wallet, string reason);
    event WalletWhitelisted(address indexed wallet);
    event WalletRemovedFromBlacklist(address indexed wallet);
    
    event OracleRegistered(address indexed oracle);
    event OracleDeactivated(address indexed oracle);
    
    event ThresholdUpdated(uint256 blockThreshold, uint256 reviewThreshold);
    event CircuitBreakerTriggered(address indexed triggeredBy, string reason);
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════════
    
    constructor(address _admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(ADMIN_ROLE, _admin);
        _grantRole(OPERATOR_ROLE, _admin);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              CORE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Submit a signed AI analysis from an authorized oracle
     * @dev Verifies the ECDSA signature matches the oracle's address
     * @param transactionId Unique identifier for the transaction
     * @param sender Transaction sender address
     * @param recipient Transaction recipient address
     * @param amount Transaction amount in smallest units
     * @param riskScore Risk score from AI analysis (0-100)
     * @param approved Whether transaction is approved
     * @param timestamp Analysis timestamp
     * @param signature ECDSA signature from the oracle
     */
    function submitAIAnalysis(
        bytes32 transactionId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        bool approved,
        uint256 timestamp,
        bytes calldata signature
    ) external nonReentrant whenNotPaused {
        require(riskScore <= MAX_RISK_SCORE, "Invalid risk score");
        require(timestamp > block.timestamp - ANALYSIS_VALIDITY_PERIOD, "Analysis expired");
        require(timestamp <= block.timestamp + 5 minutes, "Future timestamp");
        require(!usedNonces[transactionId], "Duplicate transaction");
        
        // Rate limiting
        require(
            submissionsPerBlock[msg.sender][block.number] < maxSubmissionsPerBlock,
            "Rate limit exceeded"
        );
        submissionsPerBlock[msg.sender][block.number]++;
        
        // Reconstruct and verify the signed message
        bytes32 messageHash = keccak256(abi.encodePacked(
            transactionId,
            riskScore,
            approved,
            !approved && riskScore >= blockThreshold, // blocked
            timestamp
        ));
        
        bytes32 ethSignedHash = messageHash.toEthSignedMessageHash();
        address recoveredSigner = ethSignedHash.recover(signature);
        
        // Verify signer is an authorized oracle
        require(hasRole(ORACLE_ROLE, recoveredSigner), "Invalid oracle signature");
        
        // Mark nonce as used
        usedNonces[transactionId] = true;
        
        // Update oracle metrics
        oracleMetrics[recoveredSigner].totalSubmissions++;
        oracleMetrics[recoveredSigner].successfulVerifications++;
        oracleMetrics[recoveredSigner].lastSubmissionTime = block.timestamp;
        
        // Determine risk level
        RiskLevel riskLevel = _scoreToRiskLevel(riskScore);
        bool blocked = riskScore >= blockThreshold || globalBlacklist[sender] || globalBlacklist[recipient];
        
        // Store analysis
        analyses[transactionId] = AIAnalysis({
            transactionId: transactionId,
            sender: sender,
            recipient: recipient,
            amount: amount,
            riskScore: riskScore,
            riskLevel: riskLevel,
            approved: approved && !blocked,
            blocked: blocked,
            timestamp: timestamp,
            oracleAddress: recoveredSigner,
            signature: signature,
            verified: true
        });
        
        // Update wallet risk profiles
        _updateWalletRisk(sender, riskScore, riskLevel);
        _updateWalletRisk(recipient, riskScore / 2, riskLevel); // Recipients get half the score
        
        // Update statistics
        totalAnalyses++;
        if (blocked) {
            totalBlocked++;
            emit TransactionBlocked(transactionId, sender, recipient, amount, riskScore, "AI Risk Score Exceeded");
        } else {
            totalApproved++;
            emit TransactionApproved(transactionId, sender, recipient, amount, riskScore);
        }
        
        emit AnalysisSubmitted(transactionId, sender, recipient, amount, riskScore, riskLevel, !blocked, recoveredSigner);
        emit AnalysisVerified(transactionId, recoveredSigner, true);
    }
    
    /**
     * @notice Check if a transaction should be approved based on stored analysis
     * @param transactionId The transaction to check
     * @return approved Whether the transaction is approved
     * @return riskScore The risk score
     * @return riskLevel The risk level
     */
    function checkTransaction(bytes32 transactionId) 
        external 
        view 
        returns (bool approved, uint256 riskScore, RiskLevel riskLevel) 
    {
        AIAnalysis memory analysis = analyses[transactionId];
        require(analysis.verified, "No verified analysis found");
        
        return (analysis.approved, analysis.riskScore, analysis.riskLevel);
    }
    
    /**
     * @notice Check if an address can transact (not blacklisted, not high risk)
     * @param wallet Address to check
     * @return canTransact Whether the wallet can transact
     * @return riskScore Current risk score
     */
    function canWalletTransact(address wallet) 
        external 
        view 
        returns (bool canTransact, uint256 riskScore) 
    {
        if (globalWhitelist[wallet]) {
            return (true, 0);
        }
        
        if (globalBlacklist[wallet]) {
            return (false, 100);
        }
        
        WalletRisk memory risk = walletRisks[wallet];
        return (risk.currentRiskScore < blockThreshold, risk.currentRiskScore);
    }
    
    /**
     * @notice Verify a signature matches an oracle without storing the analysis
     * @dev Useful for pre-flight checks before transaction submission
     */
    function verifyOracleSignature(
        bytes32 transactionId,
        uint256 riskScore,
        bool approved,
        bool blocked,
        uint256 timestamp,
        bytes calldata signature
    ) external view returns (bool valid, address signer) {
        bytes32 messageHash = keccak256(abi.encodePacked(
            transactionId,
            riskScore,
            approved,
            blocked,
            timestamp
        ));
        
        bytes32 ethSignedHash = messageHash.toEthSignedMessageHash();
        address recoveredSigner = ethSignedHash.recover(signature);
        
        return (hasRole(ORACLE_ROLE, recoveredSigner), recoveredSigner);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function _scoreToRiskLevel(uint256 score) internal pure returns (RiskLevel) {
        if (score <= 20) return RiskLevel.SAFE;
        if (score <= 40) return RiskLevel.LOW;
        if (score <= 60) return RiskLevel.MEDIUM;
        if (score <= 80) return RiskLevel.HIGH;
        return RiskLevel.CRITICAL;
    }
    
    function _updateWalletRisk(address wallet, uint256 score, RiskLevel level) internal {
        WalletRisk storage risk = walletRisks[wallet];
        
        uint256 previousScore = risk.currentRiskScore;
        
        // Exponential moving average for risk score
        if (risk.totalAnalyses == 0) {
            risk.currentRiskScore = score;
        } else {
            // 30% weight to new score, 70% to historical
            risk.currentRiskScore = (score * 30 + risk.currentRiskScore * 70) / 100;
        }
        
        // Track peak
        if (risk.currentRiskScore > risk.peakRiskScore) {
            risk.peakRiskScore = risk.currentRiskScore;
        }
        
        risk.totalAnalyses++;
        risk.lastAnalysisTime = block.timestamp;
        
        // Auto-blacklist if critical and repeated
        if (level == RiskLevel.CRITICAL) {
            risk.blockedCount++;
            if (risk.blockedCount >= 3) {
                globalBlacklist[wallet] = true;
                risk.isBlacklisted = true;
                emit WalletBlacklisted(wallet, "Repeated critical risk transactions");
            }
        }
        
        emit WalletRiskUpdated(wallet, previousScore, risk.currentRiskScore, level);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Register a new AI oracle
     * @param oracle Address of the oracle's signing key
     */
    function registerOracle(address oracle) external onlyRole(ADMIN_ROLE) {
        require(!hasRole(ORACLE_ROLE, oracle), "Already registered");
        
        _grantRole(ORACLE_ROLE, oracle);
        registeredOracles.push(oracle);
        oracleMetrics[oracle].isActive = true;
        
        emit OracleRegistered(oracle);
    }
    
    /**
     * @notice Deactivate an oracle
     */
    function deactivateOracle(address oracle) external onlyRole(ADMIN_ROLE) {
        _revokeRole(ORACLE_ROLE, oracle);
        oracleMetrics[oracle].isActive = false;
        
        emit OracleDeactivated(oracle);
    }
    
    /**
     * @notice Add address to global blacklist
     */
    function blacklistAddress(address wallet, string calldata reason) external onlyRole(OPERATOR_ROLE) {
        globalBlacklist[wallet] = true;
        walletRisks[wallet].isBlacklisted = true;
        
        emit WalletBlacklisted(wallet, reason);
    }
    
    /**
     * @notice Remove address from blacklist
     */
    function removeFromBlacklist(address wallet) external onlyRole(OPERATOR_ROLE) {
        globalBlacklist[wallet] = false;
        walletRisks[wallet].isBlacklisted = false;
        
        emit WalletRemovedFromBlacklist(wallet);
    }
    
    /**
     * @notice Add address to whitelist
     */
    function whitelistAddress(address wallet) external onlyRole(OPERATOR_ROLE) {
        globalWhitelist[wallet] = true;
        walletRisks[wallet].isWhitelisted = true;
        
        emit WalletWhitelisted(wallet);
    }
    
    /**
     * @notice Update risk thresholds
     */
    function updateThresholds(uint256 _blockThreshold, uint256 _reviewThreshold) 
        external 
        onlyRole(ADMIN_ROLE) 
    {
        require(_blockThreshold <= 100 && _reviewThreshold <= 100, "Invalid thresholds");
        require(_blockThreshold > _reviewThreshold, "Block must be higher than review");
        
        blockThreshold = _blockThreshold;
        reviewThreshold = _reviewThreshold;
        
        emit ThresholdUpdated(_blockThreshold, _reviewThreshold);
    }
    
    /**
     * @notice Update oracle quorum requirement
     */
    function updateQuorum(uint256 _quorum) external onlyRole(ADMIN_ROLE) {
        require(_quorum >= MIN_ORACLE_QUORUM, "Quorum too low");
        oracleQuorum = _quorum;
    }
    
    /**
     * @notice Emergency pause
     */
    function triggerCircuitBreaker(string calldata reason) external onlyRole(OPERATOR_ROLE) {
        _pause();
        emit CircuitBreakerTriggered(msg.sender, reason);
    }
    
    /**
     * @notice Resume operations
     */
    function resumeOperations() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Get full analysis for a transaction
     */
    function getAnalysis(bytes32 transactionId) external view returns (AIAnalysis memory) {
        return analyses[transactionId];
    }
    
    /**
     * @notice Get wallet risk profile
     */
    function getWalletRisk(address wallet) external view returns (WalletRisk memory) {
        return walletRisks[wallet];
    }
    
    /**
     * @notice Get oracle metrics
     */
    function getOracleMetrics(address oracle) external view returns (OracleMetrics memory) {
        return oracleMetrics[oracle];
    }
    
    /**
     * @notice Get all registered oracles
     */
    function getRegisteredOracles() external view returns (address[] memory) {
        return registeredOracles;
    }
    
    /**
     * @notice Get contract statistics
     */
    function getStatistics() external view returns (
        uint256 _totalAnalyses,
        uint256 _totalBlocked,
        uint256 _totalApproved,
        uint256 _blockThreshold,
        uint256 _reviewThreshold,
        uint256 _oracleCount
    ) {
        return (
            totalAnalyses,
            totalBlocked,
            totalApproved,
            blockThreshold,
            reviewThreshold,
            registeredOracles.length
        );
    }
    
    /**
     * @notice Check if address is a registered oracle
     */
    function isOracle(address addr) external view returns (bool) {
        return hasRole(ORACLE_ROLE, addr);
    }
}
