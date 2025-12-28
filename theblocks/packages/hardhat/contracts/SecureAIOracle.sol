// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║                         SECURE AI ORACLE CONTRACT                             ║
 * ║                                                                               ║
 * ║   On-chain verification of off-chain AI oracle signatures                    ║
 * ║   - Verifies cryptographic signatures from authorized oracles                 ║
 * ║   - Prevents manipulation of AI analysis results                              ║
 * ║   - Multi-oracle support for decentralization                                ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 *
 * SECURITY MODEL:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                          OFF-CHAIN (Private)                                 │
 * │  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
 * │  │   AI API      │────▶│   Oracle      │────▶│   Sign with   │             │
 * │  │  (OpenAI)     │     │   Service     │     │  Private Key  │             │
 * │  └───────────────┘     └───────────────┘     └───────┬───────┘             │
 * └──────────────────────────────────────────────────────┼──────────────────────┘
 *                                                         │
 *                                                         ▼
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                          ON-CHAIN (This Contract)                           │
 * │  ┌───────────────────────────────────────────────────────────────────────┐  │
 * │  │ 1. Receive signed data                                                │  │
 * │  │ 2. Verify signature matches authorized oracle                         │  │
 * │  │ 3. Check timestamp is recent (prevent replay)                         │  │
 * │  │ 4. Store result if valid                                              │  │
 * │  │ 5. Block transaction if risk too high                                 │  │
 * │  └───────────────────────────────────────────────────────────────────────┘  │
 * └─────────────────────────────────────────────────────────────────────────────┘
 */

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

contract SecureAIOracle is AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ═══════════════════════════════════════════════════════════════════════════
    //                              ROLES
    // ═══════════════════════════════════════════════════════════════════════════
    
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    // ═══════════════════════════════════════════════════════════════════════════
    //                              STRUCTS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @dev Risk assessment result from AI oracle
     */
    struct RiskAssessment {
        bytes32 transactionId;
        uint8 riskScore;          // 0-100
        RiskLevel riskLevel;
        bool approved;
        string explanation;
        uint256 confidence;       // Basis points (10000 = 100%)
        string model;
        uint256 timestamp;
        address oracle;
        bytes signature;
    }

    enum RiskLevel {
        SAFE,       // 0-20
        LOW,        // 21-40
        MEDIUM,     // 41-60
        HIGH,       // 61-80
        CRITICAL    // 81-100
    }

    /**
     * @dev Oracle configuration
     */
    struct OracleConfig {
        bool isActive;
        uint256 trustScore;       // 0-10000 (basis points)
        uint256 totalAssessments;
        uint256 disputedAssessments;
        uint256 registeredAt;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                              STATE
    // ═══════════════════════════════════════════════════════════════════════════

    // Oracle management
    mapping(address => OracleConfig) public oracles;
    address[] public oracleList;
    uint256 public minOracleConsensus = 1;  // Minimum oracles needed for consensus
    
    // Risk assessments
    mapping(bytes32 => RiskAssessment) public assessments;
    mapping(bytes32 => bool) public processedTransactions;
    
    // Signature replay prevention
    mapping(bytes32 => bool) public usedSignatures;
    
    // Configuration
    uint256 public signatureValidityPeriod = 5 minutes;
    uint8 public autoBlockThreshold = 80;  // Block transactions with risk >= 80
    uint8 public manualReviewThreshold = 50;  // Flag for review if risk >= 50
    
    // Statistics
    uint256 public totalAssessments;
    uint256 public blockedTransactions;
    uint256 public flaggedTransactions;

    // ═══════════════════════════════════════════════════════════════════════════
    //                              EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event OracleRegistered(address indexed oracle, uint256 trustScore);
    event OracleDeactivated(address indexed oracle);
    event OracleTrustUpdated(address indexed oracle, uint256 newTrustScore);
    
    event AssessmentSubmitted(
        bytes32 indexed transactionId,
        address indexed oracle,
        uint8 riskScore,
        bool approved,
        string model
    );
    
    event TransactionBlocked(
        bytes32 indexed transactionId,
        uint8 riskScore,
        string reason
    );
    
    event TransactionFlagged(
        bytes32 indexed transactionId,
        uint8 riskScore,
        string reason
    );
    
    event SignatureVerified(
        bytes32 indexed transactionId,
        address indexed oracle,
        bool valid
    );

    // ═══════════════════════════════════════════════════════════════════════════
    //                              ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error InvalidSignature();
    error SignatureExpired();
    error SignatureAlreadyUsed();
    error OracleNotAuthorized();
    error OracleNotActive();
    error TransactionAlreadyProcessed();
    error InvalidRiskScore();
    error AssessmentNotFound();

    // ═══════════════════════════════════════════════════════════════════════════
    //                              CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ORACLE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @dev Register a new oracle address
     * @param oracle The oracle's Ethereum address (derived from their private key)
     * @param trustScore Initial trust score (0-10000)
     */
    function registerOracle(
        address oracle,
        uint256 trustScore
    ) external onlyRole(ADMIN_ROLE) {
        require(trustScore <= 10000, "Trust score must be <= 10000");
        require(!oracles[oracle].isActive, "Oracle already registered");
        
        oracles[oracle] = OracleConfig({
            isActive: true,
            trustScore: trustScore,
            totalAssessments: 0,
            disputedAssessments: 0,
            registeredAt: block.timestamp
        });
        
        oracleList.push(oracle);
        _grantRole(ORACLE_ROLE, oracle);
        
        emit OracleRegistered(oracle, trustScore);
    }

    /**
     * @dev Deactivate an oracle
     */
    function deactivateOracle(address oracle) external onlyRole(ADMIN_ROLE) {
        oracles[oracle].isActive = false;
        _revokeRole(ORACLE_ROLE, oracle);
        emit OracleDeactivated(oracle);
    }

    /**
     * @dev Update oracle trust score
     */
    function updateOracleTrust(
        address oracle,
        uint256 newTrustScore
    ) external onlyRole(ADMIN_ROLE) {
        require(newTrustScore <= 10000, "Trust score must be <= 10000");
        oracles[oracle].trustScore = newTrustScore;
        emit OracleTrustUpdated(oracle, newTrustScore);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         SIGNATURE VERIFICATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @dev Verify that a message was signed by an authorized oracle
     * 
     * This is the core security function. It:
     * 1. Reconstructs the message hash
     * 2. Recovers the signer from the signature
     * 3. Verifies the signer is an authorized oracle
     */
    function verifyOracleSignature(
        bytes32 transactionId,
        uint8 riskScore,
        bool approved,
        uint256 timestamp,
        string memory model,
        bytes memory signature
    ) public view returns (bool valid, address signer) {
        // Reconstruct the message that was signed
        // Must match exactly what the off-chain oracle signed
        string memory message = string(abi.encodePacked(
            '{"transaction_id":"',
            _bytes32ToString(transactionId),
            '","risk_score":',
            _uint8ToString(riskScore),
            ',"approved":',
            approved ? 'true' : 'false',
            ',"timestamp":',
            _uint256ToString(timestamp),
            ',"model":"',
            model,
            '"}'
        ));
        
        // Hash the message (EIP-191 personal sign)
        bytes32 messageHash = keccak256(bytes(message));
        bytes32 ethSignedMessageHash = messageHash.toEthSignedMessageHash();
        
        // Recover signer
        signer = ethSignedMessageHash.recover(signature);
        
        // Verify signer is authorized oracle
        valid = oracles[signer].isActive;
        
        return (valid, signer);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         SUBMIT ASSESSMENT
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @dev Submit a signed risk assessment from an oracle
     * 
     * @param transactionId Unique transaction identifier
     * @param riskScore Risk score 0-100
     * @param approved Whether transaction is approved
     * @param explanation Brief explanation
     * @param confidence Confidence level in basis points
     * @param model AI model used
     * @param timestamp When assessment was created
     * @param signature Cryptographic signature from oracle
     */
    function submitAssessment(
        bytes32 transactionId,
        uint8 riskScore,
        bool approved,
        string memory explanation,
        uint256 confidence,
        string memory model,
        uint256 timestamp,
        bytes memory signature
    ) external nonReentrant whenNotPaused {
        // Validate inputs
        if (riskScore > 100) revert InvalidRiskScore();
        if (processedTransactions[transactionId]) revert TransactionAlreadyProcessed();
        
        // Prevent signature replay
        bytes32 sigHash = keccak256(signature);
        if (usedSignatures[sigHash]) revert SignatureAlreadyUsed();
        
        // Check signature freshness
        if (block.timestamp > timestamp + signatureValidityPeriod) revert SignatureExpired();
        
        // Verify signature
        (bool valid, address signer) = verifyOracleSignature(
            transactionId,
            riskScore,
            approved,
            timestamp,
            model,
            signature
        );
        
        if (!valid) revert InvalidSignature();
        if (!oracles[signer].isActive) revert OracleNotActive();
        
        emit SignatureVerified(transactionId, signer, true);
        
        // Mark signature as used
        usedSignatures[sigHash] = true;
        
        // Determine risk level
        RiskLevel level = _calculateRiskLevel(riskScore);
        
        // Store assessment
        assessments[transactionId] = RiskAssessment({
            transactionId: transactionId,
            riskScore: riskScore,
            riskLevel: level,
            approved: approved,
            explanation: explanation,
            confidence: confidence,
            model: model,
            timestamp: block.timestamp,
            oracle: signer,
            signature: signature
        });
        
        processedTransactions[transactionId] = true;
        
        // Update oracle stats
        oracles[signer].totalAssessments++;
        totalAssessments++;
        
        emit AssessmentSubmitted(transactionId, signer, riskScore, approved, model);
        
        // Auto-block or flag based on risk
        if (riskScore >= autoBlockThreshold) {
            blockedTransactions++;
            emit TransactionBlocked(transactionId, riskScore, explanation);
        } else if (riskScore >= manualReviewThreshold) {
            flaggedTransactions++;
            emit TransactionFlagged(transactionId, riskScore, explanation);
        }
    }

    /**
     * @dev Check if a transaction is approved
     */
    function isTransactionApproved(
        bytes32 transactionId
    ) external view returns (bool approved, uint8 riskScore, string memory reason) {
        if (!processedTransactions[transactionId]) {
            return (true, 0, "Not yet assessed");
        }
        
        RiskAssessment storage assessment = assessments[transactionId];
        
        if (assessment.riskScore >= autoBlockThreshold) {
            return (false, assessment.riskScore, assessment.explanation);
        }
        
        return (assessment.approved, assessment.riskScore, assessment.explanation);
    }

    /**
     * @dev Get full assessment details
     */
    function getAssessment(
        bytes32 transactionId
    ) external view returns (RiskAssessment memory) {
        if (!processedTransactions[transactionId]) revert AssessmentNotFound();
        return assessments[transactionId];
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    function setSignatureValidityPeriod(uint256 period) external onlyRole(ADMIN_ROLE) {
        signatureValidityPeriod = period;
    }

    function setAutoBlockThreshold(uint8 threshold) external onlyRole(ADMIN_ROLE) {
        require(threshold <= 100, "Invalid threshold");
        autoBlockThreshold = threshold;
    }

    function setManualReviewThreshold(uint8 threshold) external onlyRole(ADMIN_ROLE) {
        require(threshold <= 100, "Invalid threshold");
        manualReviewThreshold = threshold;
    }

    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function getOracleCount() external view returns (uint256) {
        return oracleList.length;
    }

    function getActiveOracles() external view returns (address[] memory) {
        uint256 activeCount = 0;
        for (uint256 i = 0; i < oracleList.length; i++) {
            if (oracles[oracleList[i]].isActive) {
                activeCount++;
            }
        }
        
        address[] memory active = new address[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < oracleList.length; i++) {
            if (oracles[oracleList[i]].isActive) {
                active[index] = oracleList[i];
                index++;
            }
        }
        
        return active;
    }

    function getStatistics() external view returns (
        uint256 _totalAssessments,
        uint256 _blockedTransactions,
        uint256 _flaggedTransactions,
        uint256 _oracleCount
    ) {
        return (totalAssessments, blockedTransactions, flaggedTransactions, oracleList.length);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function _calculateRiskLevel(uint8 score) internal pure returns (RiskLevel) {
        if (score <= 20) return RiskLevel.SAFE;
        if (score <= 40) return RiskLevel.LOW;
        if (score <= 60) return RiskLevel.MEDIUM;
        if (score <= 80) return RiskLevel.HIGH;
        return RiskLevel.CRITICAL;
    }

    function _bytes32ToString(bytes32 _bytes32) internal pure returns (string memory) {
        bytes memory bytesArray = new bytes(64);
        for (uint256 i = 0; i < 32; i++) {
            bytes1 b = _bytes32[i];
            bytes1 hi = bytes1(uint8(b) / 16);
            bytes1 lo = bytes1(uint8(b) - 16 * uint8(hi));
            bytesArray[i * 2] = _char(hi);
            bytesArray[i * 2 + 1] = _char(lo);
        }
        return string(bytesArray);
    }

    function _char(bytes1 b) internal pure returns (bytes1) {
        if (uint8(b) < 10) return bytes1(uint8(b) + 0x30);
        else return bytes1(uint8(b) + 0x57);
    }

    function _uint8ToString(uint8 value) internal pure returns (string memory) {
        return _uint256ToString(uint256(value));
    }

    function _uint256ToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) return "0";
        
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        
        return string(buffer);
    }
}
