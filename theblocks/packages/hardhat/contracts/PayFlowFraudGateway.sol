// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                     PAYFLOW FRAUD GATEWAY                                             ║
 * ║                                                                                       ║
 * ║   Integration Layer Between PayFlowCore and SecureFraudOracle                        ║
 * ║                                                                                       ║
 * ║   This contract provides:                                                             ║
 * ║   • Pre-transaction fraud screening                                                  ║
 * ║   • Real-time risk assessment integration                                            ║
 * ║   • Automatic blocking of high-risk transactions                                     ║
 * ║   • Compliance audit trail generation                                                ║
 * ║                                                                                       ║
 * ║   Hackxios 2K25 - PayFlow Protocol                                                   ║
 * ╚═══════════════════════════════════════════════════════════════════════════════════════╝
 */

interface ISecureFraudOracle {
    enum RiskLevel { SAFE, LOW, MEDIUM, HIGH, CRITICAL }
    
    struct WalletRisk {
        uint256 currentRiskScore;
        uint256 peakRiskScore;
        uint256 totalAnalyses;
        uint256 blockedCount;
        uint256 lastAnalysisTime;
        bool isBlacklisted;
        bool isWhitelisted;
    }
    
    function checkTransaction(bytes32 transactionId) 
        external view returns (bool approved, uint256 riskScore, RiskLevel riskLevel);
    
    function canWalletTransact(address wallet) 
        external view returns (bool canTransact, uint256 riskScore);
    
    function getWalletRisk(address wallet) external view returns (WalletRisk memory);
    
    function verifyOracleSignature(
        bytes32 transactionId,
        uint256 riskScore,
        bool approved,
        bool blocked,
        uint256 timestamp,
        bytes calldata signature
    ) external view returns (bool valid, address signer);
    
    function submitAIAnalysis(
        bytes32 transactionId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        bool approved,
        uint256 timestamp,
        bytes calldata signature
    ) external;
}

interface IPayFlowCore {
    function getPaymentStatus(bytes32 paymentId) external view returns (uint8);
}

interface IAuditRegistry {
    function recordEvent(
        bytes32 entityId,
        string calldata eventType,
        bytes calldata eventData
    ) external;
}

/**
 * @title PayFlowFraudGateway
 * @notice Integrates AI fraud detection with PayFlow payments
 */
contract PayFlowFraudGateway is AccessControl, ReentrancyGuard, Pausable {
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              CONSTANTS & ROLES
    // ═══════════════════════════════════════════════════════════════════════════════
    
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════════
    
    ISecureFraudOracle public fraudOracle;
    IPayFlowCore public payFlowCore;
    IAuditRegistry public auditRegistry;
    
    /// @notice Risk score threshold for blocking (0-100)
    uint256 public blockThreshold = 80;
    
    /// @notice Risk score threshold for manual review (0-100)
    uint256 public reviewThreshold = 60;
    
    /// @notice Minimum amount requiring AI screening (in USD, 6 decimals)
    uint256 public minScreeningAmount = 100 * 10**6; // $100 USDC
    
    /// @notice Whether real-time AI screening is enabled
    bool public realTimeScreeningEnabled = true;
    
    /// @notice Screening results cache
    mapping(bytes32 => ScreeningResult) public screeningResults;
    
    /// @notice Transaction to screening mapping
    mapping(bytes32 => bytes32) public paymentToScreening;
    
    /// @notice Flagged payments pending review
    bytes32[] public flaggedPayments;
    mapping(bytes32 => bool) public isFlagged;
    
    /// @notice Statistics
    uint256 public totalScreened;
    uint256 public totalBlocked;
    uint256 public totalFlagged;
    uint256 public totalApproved;
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              TYPES
    // ═══════════════════════════════════════════════════════════════════════════════
    
    struct ScreeningResult {
        bytes32 paymentId;
        address sender;
        address recipient;
        uint256 amount;
        uint256 riskScore;
        ISecureFraudOracle.RiskLevel riskLevel;
        bool approved;
        bool blocked;
        bool flaggedForReview;
        uint256 screenedAt;
        bytes32 oracleTransactionId;
        bytes signature;
    }
    
    struct BatchScreeningRequest {
        bytes32 paymentId;
        address sender;
        address recipient;
        uint256 amount;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              EVENTS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    event PaymentScreened(
        bytes32 indexed paymentId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore,
        ISecureFraudOracle.RiskLevel riskLevel,
        bool approved
    );
    
    event PaymentBlocked(
        bytes32 indexed paymentId,
        address indexed sender,
        uint256 riskScore,
        string reason
    );
    
    event PaymentFlagged(
        bytes32 indexed paymentId,
        address indexed sender,
        uint256 riskScore,
        string reason
    );
    
    event FlaggedPaymentReviewed(
        bytes32 indexed paymentId,
        address indexed reviewer,
        bool approved
    );
    
    event WalletBlocked(address indexed wallet, string reason);
    event WalletCleared(address indexed wallet, address indexed clearedBy);
    
    event OracleUpdated(address indexed newOracle);
    event ThresholdUpdated(uint256 blockThreshold, uint256 reviewThreshold);
    event ScreeningConfigUpdated(bool enabled, uint256 minAmount);
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════════
    
    constructor(
        address _fraudOracle,
        address _payFlowCore,
        address _auditRegistry,
        address _admin
    ) {
        fraudOracle = ISecureFraudOracle(_fraudOracle);
        payFlowCore = IPayFlowCore(_payFlowCore);
        auditRegistry = IAuditRegistry(_auditRegistry);
        
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(OPERATOR_ROLE, _admin);
        _grantRole(COMPLIANCE_ROLE, _admin);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              SCREENING FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Screen a payment before execution
     * @dev Should be called by PayFlowCore before executing payment
     * @param paymentId The PayFlow payment ID
     * @param sender Payment sender
     * @param recipient Payment recipient
     * @param amount Payment amount
     * @param oracleRiskScore Risk score from AI oracle
     * @param oracleApproved Oracle approval status
     * @param oracleTimestamp Oracle analysis timestamp
     * @param oracleSignature Oracle's cryptographic signature
     * @return approved Whether payment should proceed
     * @return riskScore The final risk score
     */
    function screenPayment(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 oracleRiskScore,
        bool oracleApproved,
        uint256 oracleTimestamp,
        bytes calldata oracleSignature
    ) external nonReentrant whenNotPaused returns (bool approved, uint256 riskScore) {
        require(paymentToScreening[paymentId] == bytes32(0), "Already screened");
        
        // Skip screening for small amounts
        if (amount < minScreeningAmount) {
            return _approvePayment(paymentId, sender, recipient, amount, 0, "Below threshold");
        }
        
        // Generate oracle transaction ID
        bytes32 oracleTxId = keccak256(abi.encodePacked(
            paymentId, sender, recipient, amount, block.timestamp
        ));
        
        // Verify oracle signature
        bool blocked = oracleRiskScore >= blockThreshold;
        (bool signatureValid, ) = fraudOracle.verifyOracleSignature(
            oracleTxId,
            oracleRiskScore,
            oracleApproved,
            blocked,
            oracleTimestamp,
            oracleSignature
        );
        
        require(signatureValid, "Invalid oracle signature");
        
        // Submit to oracle for on-chain recording
        fraudOracle.submitAIAnalysis(
            oracleTxId,
            sender,
            recipient,
            amount,
            oracleRiskScore,
            oracleApproved,
            oracleTimestamp,
            oracleSignature
        );
        
        // Determine screening result
        ISecureFraudOracle.RiskLevel riskLevel = _scoreToLevel(oracleRiskScore);
        
        if (oracleRiskScore >= blockThreshold) {
            return _blockPayment(paymentId, sender, recipient, amount, oracleRiskScore, riskLevel, oracleTxId, oracleSignature);
        } else if (oracleRiskScore >= reviewThreshold) {
            return _flagPayment(paymentId, sender, recipient, amount, oracleRiskScore, riskLevel, oracleTxId, oracleSignature);
        } else {
            return _approvePaymentFull(paymentId, sender, recipient, amount, oracleRiskScore, riskLevel, oracleTxId, oracleSignature);
        }
    }
    
    /**
     * @notice Quick check if a payment should be blocked based on cached wallet data
     * @dev Fast path check before full screening
     */
    function quickCheck(address sender, address recipient) 
        external 
        view 
        returns (bool canProceed, string memory reason) 
    {
        // Check sender
        (bool senderCanTransact, uint256 senderScore) = fraudOracle.canWalletTransact(sender);
        if (!senderCanTransact) {
            return (false, "Sender blocked by fraud oracle");
        }
        
        // Check recipient
        (bool recipientCanTransact, uint256 recipientScore) = fraudOracle.canWalletTransact(recipient);
        if (!recipientCanTransact) {
            return (false, "Recipient blocked by fraud oracle");
        }
        
        // Combined risk
        if (senderScore + recipientScore > 120) {
            return (false, "Combined risk too high");
        }
        
        return (true, "OK");
    }
    
    /**
     * @notice Screen multiple payments in batch
     */
    function batchScreenPayments(
        BatchScreeningRequest[] calldata requests,
        uint256[] calldata riskScores,
        bool[] calldata approvals,
        uint256[] calldata timestamps,
        bytes[] calldata signatures
    ) external nonReentrant whenNotPaused returns (bool[] memory results) {
        require(
            requests.length == riskScores.length &&
            requests.length == approvals.length &&
            requests.length == timestamps.length &&
            requests.length == signatures.length,
            "Array length mismatch"
        );
        
        results = new bool[](requests.length);
        
        for (uint256 i = 0; i < requests.length; i++) {
            (bool approved, ) = this.screenPayment(
                requests[i].paymentId,
                requests[i].sender,
                requests[i].recipient,
                requests[i].amount,
                riskScores[i],
                approvals[i],
                timestamps[i],
                signatures[i]
            );
            results[i] = approved;
        }
        
        return results;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function _scoreToLevel(uint256 score) internal pure returns (ISecureFraudOracle.RiskLevel) {
        if (score <= 20) return ISecureFraudOracle.RiskLevel.SAFE;
        if (score <= 40) return ISecureFraudOracle.RiskLevel.LOW;
        if (score <= 60) return ISecureFraudOracle.RiskLevel.MEDIUM;
        if (score <= 80) return ISecureFraudOracle.RiskLevel.HIGH;
        return ISecureFraudOracle.RiskLevel.CRITICAL;
    }
    
    function _approvePayment(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        string memory reason
    ) internal returns (bool, uint256) {
        totalApproved++;
        
        emit PaymentScreened(
            paymentId,
            sender,
            recipient,
            amount,
            riskScore,
            ISecureFraudOracle.RiskLevel.SAFE,
            true
        );
        
        return (true, riskScore);
    }
    
    function _approvePaymentFull(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        ISecureFraudOracle.RiskLevel riskLevel,
        bytes32 oracleTxId,
        bytes calldata signature
    ) internal returns (bool, uint256) {
        bytes32 screeningId = keccak256(abi.encodePacked(paymentId, block.timestamp));
        
        screeningResults[screeningId] = ScreeningResult({
            paymentId: paymentId,
            sender: sender,
            recipient: recipient,
            amount: amount,
            riskScore: riskScore,
            riskLevel: riskLevel,
            approved: true,
            blocked: false,
            flaggedForReview: false,
            screenedAt: block.timestamp,
            oracleTransactionId: oracleTxId,
            signature: signature
        });
        
        paymentToScreening[paymentId] = screeningId;
        totalScreened++;
        totalApproved++;
        
        emit PaymentScreened(paymentId, sender, recipient, amount, riskScore, riskLevel, true);
        
        return (true, riskScore);
    }
    
    function _blockPayment(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        ISecureFraudOracle.RiskLevel riskLevel,
        bytes32 oracleTxId,
        bytes calldata signature
    ) internal returns (bool, uint256) {
        bytes32 screeningId = keccak256(abi.encodePacked(paymentId, block.timestamp));
        
        screeningResults[screeningId] = ScreeningResult({
            paymentId: paymentId,
            sender: sender,
            recipient: recipient,
            amount: amount,
            riskScore: riskScore,
            riskLevel: riskLevel,
            approved: false,
            blocked: true,
            flaggedForReview: false,
            screenedAt: block.timestamp,
            oracleTransactionId: oracleTxId,
            signature: signature
        });
        
        paymentToScreening[paymentId] = screeningId;
        totalScreened++;
        totalBlocked++;
        
        emit PaymentBlocked(paymentId, sender, riskScore, "AI risk score exceeded threshold");
        
        // Record in audit registry
        if (address(auditRegistry) != address(0)) {
            auditRegistry.recordEvent(
                paymentId,
                "PAYMENT_BLOCKED_BY_AI",
                abi.encode(sender, recipient, amount, riskScore)
            );
        }
        
        return (false, riskScore);
    }
    
    function _flagPayment(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        uint256 riskScore,
        ISecureFraudOracle.RiskLevel riskLevel,
        bytes32 oracleTxId,
        bytes calldata signature
    ) internal returns (bool, uint256) {
        bytes32 screeningId = keccak256(abi.encodePacked(paymentId, block.timestamp));
        
        screeningResults[screeningId] = ScreeningResult({
            paymentId: paymentId,
            sender: sender,
            recipient: recipient,
            amount: amount,
            riskScore: riskScore,
            riskLevel: riskLevel,
            approved: false,
            blocked: false,
            flaggedForReview: true,
            screenedAt: block.timestamp,
            oracleTransactionId: oracleTxId,
            signature: signature
        });
        
        paymentToScreening[paymentId] = screeningId;
        flaggedPayments.push(paymentId);
        isFlagged[paymentId] = true;
        totalScreened++;
        totalFlagged++;
        
        emit PaymentFlagged(paymentId, sender, riskScore, "Risk score requires manual review");
        
        return (false, riskScore);  // Not approved yet, pending review
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              COMPLIANCE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Review and approve/reject a flagged payment
     */
    function reviewFlaggedPayment(bytes32 paymentId, bool approve) 
        external 
        onlyRole(COMPLIANCE_ROLE) 
    {
        require(isFlagged[paymentId], "Payment not flagged");
        
        bytes32 screeningId = paymentToScreening[paymentId];
        ScreeningResult storage result = screeningResults[screeningId];
        
        result.approved = approve;
        result.flaggedForReview = false;
        isFlagged[paymentId] = false;
        
        if (approve) {
            totalApproved++;
        } else {
            totalBlocked++;
            result.blocked = true;
        }
        
        emit FlaggedPaymentReviewed(paymentId, msg.sender, approve);
    }
    
    /**
     * @notice Get all flagged payments pending review
     */
    function getFlaggedPayments() external view returns (bytes32[] memory) {
        return flaggedPayments;
    }
    
    /**
     * @notice Get screening result for a payment
     */
    function getScreeningResult(bytes32 paymentId) 
        external 
        view 
        returns (ScreeningResult memory) 
    {
        bytes32 screeningId = paymentToScreening[paymentId];
        return screeningResults[screeningId];
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function setFraudOracle(address _oracle) external onlyRole(DEFAULT_ADMIN_ROLE) {
        fraudOracle = ISecureFraudOracle(_oracle);
        emit OracleUpdated(_oracle);
    }
    
    function setThresholds(uint256 _block, uint256 _review) external onlyRole(OPERATOR_ROLE) {
        require(_block > _review && _block <= 100, "Invalid thresholds");
        blockThreshold = _block;
        reviewThreshold = _review;
        emit ThresholdUpdated(_block, _review);
    }
    
    function setScreeningConfig(bool _enabled, uint256 _minAmount) external onlyRole(OPERATOR_ROLE) {
        realTimeScreeningEnabled = _enabled;
        minScreeningAmount = _minAmount;
        emit ScreeningConfigUpdated(_enabled, _minAmount);
    }
    
    function pause() external onlyRole(OPERATOR_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    //                              VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════════
    
    function getStatistics() external view returns (
        uint256 _totalScreened,
        uint256 _totalBlocked,
        uint256 _totalFlagged,
        uint256 _totalApproved,
        uint256 _pendingReview
    ) {
        uint256 pending = 0;
        for (uint256 i = 0; i < flaggedPayments.length; i++) {
            if (isFlagged[flaggedPayments[i]]) {
                pending++;
            }
        }
        
        return (
            totalScreened,
            totalBlocked,
            totalFlagged,
            totalApproved,
            pending
        );
    }
}
