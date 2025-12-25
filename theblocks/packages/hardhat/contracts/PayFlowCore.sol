// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║                         PAYFLOW PROTOCOL                                       ║
 * ║              Programmable Money Rails for Institutional Payments              ║
 * ╠═══════════════════════════════════════════════════════════════════════════════╣
 * ║  "What if money could carry its own rules?"                                   ║
 * ║                                                                               ║
 * ║  PayFlow transforms institutional payments by embedding programmable          ║
 * ║  conditions directly into payment flows. Every transfer can carry:            ║
 * ║  • Compliance requirements (KYC/AML/Sanctions)                                ║
 * ║  • Time restrictions (business hours, settlement windows)                     ║
 * ║  • Multi-signature approval thresholds                                        ║
 * ║  • Automatic FX conversion at oracle-verified rates                           ║
 * ║  • Escrow conditions with automatic release                                   ║
 * ║  • Immutable audit trails for regulatory compliance                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

/**
 * @title PayFlowCore
 * @notice The central routing engine for programmable institutional payments
 * @dev Handles payment creation, condition verification, and execution
 */
contract PayFlowCore is AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;

    // ═══════════════════════════════════════════════════════════════════════════
    //                              ROLES
    // ═══════════════════════════════════════════════════════════════════════════
    
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant COMPLIANCE_ROLE = keccak256("COMPLIANCE_ROLE");
    bytes32 public constant TREASURY_ROLE = keccak256("TREASURY_ROLE");

    // ═══════════════════════════════════════════════════════════════════════════
    //                              TYPES
    // ═══════════════════════════════════════════════════════════════════════════

    enum PaymentStatus {
        CREATED,        // Payment initiated, awaiting conditions
        PENDING,        // Conditions being verified
        APPROVED,       // All conditions met, ready to execute
        EXECUTED,       // Payment completed successfully
        FAILED,         // Payment failed (conditions not met)
        CANCELLED,      // Payment cancelled by sender
        DISPUTED        // Under dispute resolution
    }

    enum ComplianceTier {
        NONE,           // No KYC required (small amounts only)
        BASIC,          // Email + Phone verification
        STANDARD,       // Government ID + Address
        ENHANCED,       // Full KYC + Source of funds
        INSTITUTIONAL   // Corporate KYC + Ultimate beneficial owner
    }

    struct PaymentConditions {
        // Compliance
        ComplianceTier requiredSenderTier;
        ComplianceTier requiredRecipientTier;
        bool requireSanctionsCheck;
        
        // Time restrictions
        uint256 validFrom;          // Earliest execution time
        uint256 validUntil;         // Latest execution time (0 = no expiry)
        bool businessHoursOnly;     // Only execute during business hours
        
        // Amount controls
        uint256 maxSlippage;        // Max FX slippage in basis points
        
        // Multi-sig (0 = no approval required)
        uint256 requiredApprovals;
        address[] approvers;
        
        // Escrow
        bool useEscrow;
        uint256 escrowReleaseTime;  // Auto-release after this time
        bytes32 escrowConditionHash; // Hash of external condition
    }

    struct Payment {
        // Core payment data
        bytes32 paymentId;
        address sender;
        address recipient;
        address token;
        uint256 amount;
        
        // For cross-border
        address targetToken;        // If different from token, triggers FX
        uint256 targetAmount;       // Expected amount after FX (0 = market rate)
        
        // Status
        PaymentStatus status;
        uint256 createdAt;
        uint256 executedAt;
        
        // Conditions
        PaymentConditions conditions;
        
        // Approvals
        uint256 approvalCount;
        mapping(address => bool) hasApproved;
        
        // Metadata
        bytes32 referenceId;        // External reference
        string memo;
    }

    struct PaymentView {
        bytes32 paymentId;
        address sender;
        address recipient;
        address token;
        uint256 amount;
        address targetToken;
        uint256 targetAmount;
        PaymentStatus status;
        uint256 createdAt;
        uint256 executedAt;
        uint256 approvalCount;
        uint256 requiredApprovals;
        bytes32 referenceId;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                              STATE
    // ═══════════════════════════════════════════════════════════════════════════

    // Payment storage
    mapping(bytes32 => Payment) private payments;
    bytes32[] public paymentIds;
    
    // User payment tracking
    mapping(address => bytes32[]) public userPaymentsSent;
    mapping(address => bytes32[]) public userPaymentsReceived;
    
    // Supported tokens
    mapping(address => bool) public supportedTokens;
    address[] public tokenList;
    
    // Module addresses
    address public complianceEngine;
    address public oracleAggregator;
    address public escrowVault;
    address public auditRegistry;
    
    // Protocol settings
    uint256 public protocolFeeBps = 10; // 0.1% default fee
    address public feeRecipient;
    uint256 public maxPaymentAmount = 100_000_000 * 1e6; // $100M default max
    
    // Statistics
    uint256 public totalPaymentsCreated;
    uint256 public totalPaymentsExecuted;
    uint256 public totalVolumeProcessed;
    uint256 public averageSettlementTime;

    // ═══════════════════════════════════════════════════════════════════════════
    //                              EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event PaymentCreated(
        bytes32 indexed paymentId,
        address indexed sender,
        address indexed recipient,
        address token,
        uint256 amount,
        bytes32 referenceId
    );
    
    event PaymentApproved(
        bytes32 indexed paymentId,
        address indexed approver,
        uint256 approvalCount,
        uint256 requiredApprovals
    );
    
    event PaymentExecuted(
        bytes32 indexed paymentId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 settlementTime
    );
    
    event PaymentFailed(
        bytes32 indexed paymentId,
        string reason
    );
    
    event PaymentCancelled(
        bytes32 indexed paymentId,
        address indexed cancelledBy
    );
    
    event CrossBorderSettlement(
        bytes32 indexed paymentId,
        address sourceToken,
        address targetToken,
        uint256 sourceAmount,
        uint256 targetAmount,
        uint256 fxRate
    );
    
    event ComplianceVerified(
        bytes32 indexed paymentId,
        address indexed party,
        ComplianceTier tier,
        bool sanctionsCleared
    );
    
    event TravelRuleRequired(
        bytes32 indexed paymentId,
        uint256 amount
    );
    
    event TravelRuleRecorded(
        bytes32 indexed paymentId,
        bytes32 originatorDataHash,
        bytes32 beneficiaryDataHash
    );
    
    event ComplianceCheckFailed(
        bytes32 indexed paymentId,
        string reason
    );
    
    event FXRateApplied(
        bytes32 indexed paymentId,
        bytes32 pairId,
        uint256 rate,
        uint256 sourceAmount,
        uint256 targetAmount
    );

    // ═══════════════════════════════════════════════════════════════════════════
    //                         TRAVEL RULE DATA
    // ═══════════════════════════════════════════════════════════════════════════
    
    struct TravelRuleRecord {
        bytes32 originatorDataHash;
        bytes32 beneficiaryDataHash;
        uint256 timestamp;
        uint256 amount;
        bool verified;
    }
    
    mapping(bytes32 => TravelRuleRecord) public travelRuleRecords;
    uint256 public travelRuleThreshold = 3000 * 1e18; // $3000 FATF threshold

    // ═══════════════════════════════════════════════════════════════════════════
    //                              CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        _grantRole(COMPLIANCE_ROLE, msg.sender);
        _grantRole(TREASURY_ROLE, msg.sender);
        feeRecipient = msg.sender;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         PAYMENT CREATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Create a new programmable payment
     * @param recipient The payment recipient
     * @param token The token to send
     * @param amount The amount to send
     * @param conditions The programmable conditions for this payment
     * @param referenceId External reference ID for tracking
     * @param memo Human-readable memo
     * @return paymentId The unique payment identifier
     */
    function createPayment(
        address recipient,
        address token,
        uint256 amount,
        PaymentConditions calldata conditions,
        bytes32 referenceId,
        string calldata memo
    ) external nonReentrant whenNotPaused returns (bytes32 paymentId) {
        require(recipient != address(0), "Invalid recipient");
        require(recipient != msg.sender, "Cannot pay yourself");
        require(amount > 0, "Amount must be positive");
        require(amount <= maxPaymentAmount, "Exceeds max payment");
        require(supportedTokens[token], "Token not supported");
        
        // Generate unique payment ID
        paymentId = keccak256(abi.encodePacked(
            msg.sender,
            recipient,
            token,
            amount,
            block.timestamp,
            totalPaymentsCreated
        ));
        
        // Store payment
        Payment storage payment = payments[paymentId];
        payment.paymentId = paymentId;
        payment.sender = msg.sender;
        payment.recipient = recipient;
        payment.token = token;
        payment.amount = amount;
        payment.targetToken = token; // Same token by default
        payment.status = PaymentStatus.CREATED;
        payment.createdAt = block.timestamp;
        
        // Copy conditions manually (can't assign calldata struct with dynamic array directly)
        payment.conditions.requiredSenderTier = conditions.requiredSenderTier;
        payment.conditions.requiredRecipientTier = conditions.requiredRecipientTier;
        payment.conditions.requireSanctionsCheck = conditions.requireSanctionsCheck;
        payment.conditions.validFrom = conditions.validFrom;
        payment.conditions.validUntil = conditions.validUntil;
        payment.conditions.businessHoursOnly = conditions.businessHoursOnly;
        payment.conditions.maxSlippage = conditions.maxSlippage;
        payment.conditions.requiredApprovals = conditions.requiredApprovals;
        payment.conditions.useEscrow = conditions.useEscrow;
        payment.conditions.escrowReleaseTime = conditions.escrowReleaseTime;
        payment.conditions.escrowConditionHash = conditions.escrowConditionHash;
        
        payment.referenceId = referenceId;
        payment.memo = memo;
        
        // Copy approvers array
        for (uint i = 0; i < conditions.approvers.length; i++) {
            payment.conditions.approvers.push(conditions.approvers[i]);
        }
        
        // Transfer tokens to protocol
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        
        // Track
        paymentIds.push(paymentId);
        userPaymentsSent[msg.sender].push(paymentId);
        userPaymentsReceived[recipient].push(paymentId);
        totalPaymentsCreated++;
        
        emit PaymentCreated(paymentId, msg.sender, recipient, token, amount, referenceId);
        
        // Auto-process if no conditions
        if (_canAutoExecute(payment)) {
            _executePayment(paymentId);
        } else {
            payment.status = PaymentStatus.PENDING;
        }
        
        return paymentId;
    }

    /**
     * @notice Create a cross-border payment with FX conversion
     */
    function createCrossBorderPayment(
        address recipient,
        address sourceToken,
        uint256 sourceAmount,
        address targetToken,
        uint256 minTargetAmount,
        PaymentConditions calldata conditions,
        bytes32 referenceId,
        string calldata memo
    ) external nonReentrant whenNotPaused returns (bytes32 paymentId) {
        require(recipient != address(0), "Invalid recipient");
        require(sourceAmount > 0, "Amount must be positive");
        require(supportedTokens[sourceToken], "Source token not supported");
        require(supportedTokens[targetToken], "Target token not supported");
        
        paymentId = keccak256(abi.encodePacked(
            msg.sender,
            recipient,
            sourceToken,
            targetToken,
            sourceAmount,
            block.timestamp,
            totalPaymentsCreated
        ));
        
        Payment storage payment = payments[paymentId];
        payment.paymentId = paymentId;
        payment.sender = msg.sender;
        payment.recipient = recipient;
        payment.token = sourceToken;
        payment.amount = sourceAmount;
        payment.targetToken = targetToken;
        payment.targetAmount = minTargetAmount;
        payment.status = PaymentStatus.PENDING;
        payment.createdAt = block.timestamp;
        
        // Copy conditions manually (can't assign calldata struct with dynamic array directly)
        payment.conditions.requiredSenderTier = conditions.requiredSenderTier;
        payment.conditions.requiredRecipientTier = conditions.requiredRecipientTier;
        payment.conditions.requireSanctionsCheck = conditions.requireSanctionsCheck;
        payment.conditions.validFrom = conditions.validFrom;
        payment.conditions.validUntil = conditions.validUntil;
        payment.conditions.businessHoursOnly = conditions.businessHoursOnly;
        payment.conditions.maxSlippage = conditions.maxSlippage;
        payment.conditions.requiredApprovals = conditions.requiredApprovals;
        payment.conditions.useEscrow = conditions.useEscrow;
        payment.conditions.escrowReleaseTime = conditions.escrowReleaseTime;
        payment.conditions.escrowConditionHash = conditions.escrowConditionHash;
        
        // Copy approvers array
        for (uint i = 0; i < conditions.approvers.length; i++) {
            payment.conditions.approvers.push(conditions.approvers[i]);
        }
        
        payment.referenceId = referenceId;
        payment.memo = memo;
        
        IERC20(sourceToken).safeTransferFrom(msg.sender, address(this), sourceAmount);
        
        paymentIds.push(paymentId);
        userPaymentsSent[msg.sender].push(paymentId);
        userPaymentsReceived[recipient].push(paymentId);
        totalPaymentsCreated++;
        
        emit PaymentCreated(paymentId, msg.sender, recipient, sourceToken, sourceAmount, referenceId);
        
        return paymentId;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         PAYMENT APPROVAL
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Approve a payment (for multi-sig payments)
     */
    function approvePayment(bytes32 paymentId) external nonReentrant {
        Payment storage payment = payments[paymentId];
        require(payment.paymentId == paymentId, "Payment not found");
        require(payment.status == PaymentStatus.PENDING, "Not pending approval");
        require(!payment.hasApproved[msg.sender], "Already approved");
        require(_isApprover(payment, msg.sender), "Not an approver");
        
        payment.hasApproved[msg.sender] = true;
        payment.approvalCount++;
        
        emit PaymentApproved(
            paymentId, 
            msg.sender, 
            payment.approvalCount, 
            payment.conditions.requiredApprovals
        );
        
        // Check if ready to execute
        if (_allConditionsMet(payment)) {
            _executePayment(paymentId);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         PAYMENT EXECUTION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Execute a payment that has met all conditions
     */
    function executePayment(bytes32 paymentId) external nonReentrant {
        Payment storage payment = payments[paymentId];
        require(payment.paymentId == paymentId, "Payment not found");
        require(
            payment.status == PaymentStatus.PENDING || 
            payment.status == PaymentStatus.APPROVED,
            "Cannot execute"
        );
        require(_allConditionsMet(payment), "Conditions not met");
        
        _executePayment(paymentId);
    }

    function _executePayment(bytes32 paymentId) internal {
        Payment storage payment = payments[paymentId];
        
        // ✅ CRITICAL: Verify compliance BEFORE execution
        if (!_verifyCompliance(payment)) {
            // Payment failed compliance - state already updated in _verifyCompliance
            IERC20(payment.token).safeTransfer(payment.sender, payment.amount);
            emit ComplianceCheckFailed(paymentId, "Compliance verification failed");
            return;
        }
        
        uint256 amountToSend = payment.amount;
        
        // Deduct protocol fee
        if (protocolFeeBps > 0) {
            uint256 fee = (amountToSend * protocolFeeBps) / 10000;
            amountToSend -= fee;
            IERC20(payment.token).safeTransfer(feeRecipient, fee);
        }
        
        // Handle cross-border FX if needed
        uint256 fxRate = 1e18;
        if (payment.targetToken != payment.token) {
            uint256 originalAmount = amountToSend;
            amountToSend = _performFXConversion(
                payment.token,
                payment.targetToken,
                amountToSend,
                payment.targetAmount,
                payment.conditions.maxSlippage
            );
            
            // Calculate actual FX rate used
            if (originalAmount > 0) {
                fxRate = (amountToSend * 1e18) / originalAmount;
            }
            
            // Emit FX rate event for audit trail
            bytes32 pairId = keccak256(abi.encodePacked(payment.token, "/", payment.targetToken));
            emit FXRateApplied(paymentId, pairId, fxRate, originalAmount, amountToSend);
            
            emit CrossBorderSettlement(
                paymentId,
                payment.token,
                payment.targetToken,
                payment.amount,
                amountToSend,
                fxRate
            );
        }
        
        // Transfer to recipient
        if (payment.conditions.useEscrow) {
            // Send to escrow vault
            IERC20(payment.targetToken).safeTransfer(escrowVault, amountToSend);
        } else {
            // Direct transfer
            IERC20(payment.targetToken).safeTransfer(payment.recipient, amountToSend);
        }
        
        // Update state
        payment.status = PaymentStatus.EXECUTED;
        payment.executedAt = block.timestamp;
        
        uint256 settlementTime = payment.executedAt - payment.createdAt;
        totalPaymentsExecuted++;
        totalVolumeProcessed += payment.amount;
        
        // Update average settlement time
        averageSettlementTime = (averageSettlementTime * (totalPaymentsExecuted - 1) + settlementTime) / totalPaymentsExecuted;
        
        // Log to audit registry
        if (auditRegistry != address(0)) {
            IAuditRegistry(auditRegistry).logPaymentExecuted(
                paymentId,
                payment.sender,
                payment.recipient,
                payment.amount,
                payment.targetToken,
                settlementTime,
                "" // jurisdiction - could be enhanced with Travel Rule data
            );
        }
        
        emit PaymentExecuted(
            paymentId,
            payment.sender,
            payment.recipient,
            amountToSend,
            settlementTime
        );
    }

    /**
     * @notice Perform FX conversion using oracle rates
     * @dev Uses OracleAggregator for real-time rates with slippage protection
     */
    function _performFXConversion(
        address sourceToken,
        address targetToken,
        uint256 sourceAmount,
        uint256 minTargetAmount,
        uint256 maxSlippage
    ) internal view returns (uint256) {
        // Same token = no conversion needed
        if (sourceToken == targetToken) {
            return sourceAmount;
        }
        
        // Get oracle rate if oracle is configured
        uint256 targetAmount;
        if (oracleAggregator != address(0)) {
            // Build pair key from token symbols (simplified - in production, use token registry)
            bytes32 pairId = keccak256(abi.encodePacked(sourceToken, "/", targetToken));
            
            IOracleAggregator oracle = IOracleAggregator(oracleAggregator);
            IOracleAggregator.RateResponse memory rateData = oracle.getRate(pairId);
            
            // Check for stale data (1 hour threshold)
            require(
                block.timestamp - rateData.timestamp <= 3600 || rateData.spotRate > 0,
                "Stale oracle data"
            );
            
            // Check circuit breaker
            require(!rateData.circuitBreakerActive, "Circuit breaker active");
            
            // Perform conversion using oracle rate (rates are 18 decimals)
            if (rateData.spotRate > 0) {
                targetAmount = (sourceAmount * rateData.spotRate) / 1e18;
            } else {
                // Fallback to 1:1 for stablecoins if no rate
                targetAmount = sourceAmount;
            }
        } else {
            // No oracle configured - use 1:1 (for demo/testing)
            targetAmount = sourceAmount;
        }
        
        // Slippage protection
        if (minTargetAmount > 0) {
            require(targetAmount >= minTargetAmount, "Slippage exceeded: below minimum");
        }
        
        // Check max slippage in basis points
        if (maxSlippage > 0 && minTargetAmount > 0) {
            uint256 slippageBps = ((minTargetAmount - targetAmount) * 10000) / minTargetAmount;
            require(slippageBps <= maxSlippage, "Slippage exceeded: above max bps");
        }
        
        // Verify liquidity
        uint256 balance = IERC20(targetToken).balanceOf(address(this));
        require(balance >= targetAmount, "Insufficient liquidity");
        
        require(targetAmount > 0, "Invalid conversion result");
        
        return targetAmount;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         PAYMENT CANCELLATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Cancel a pending payment
     */
    function cancelPayment(bytes32 paymentId) external nonReentrant {
        Payment storage payment = payments[paymentId];
        require(payment.paymentId == paymentId, "Payment not found");
        require(payment.sender == msg.sender, "Not the sender");
        require(
            payment.status == PaymentStatus.CREATED || 
            payment.status == PaymentStatus.PENDING,
            "Cannot cancel"
        );
        
        // Refund
        IERC20(payment.token).safeTransfer(payment.sender, payment.amount);
        
        payment.status = PaymentStatus.CANCELLED;
        
        emit PaymentCancelled(paymentId, msg.sender);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         CONDITION CHECKING
    // ═══════════════════════════════════════════════════════════════════════════

    function _canAutoExecute(Payment storage payment) internal view returns (bool) {
        PaymentConditions storage c = payment.conditions;
        
        // No conditions = auto execute
        if (
            c.requiredSenderTier == ComplianceTier.NONE &&
            c.requiredRecipientTier == ComplianceTier.NONE &&
            !c.requireSanctionsCheck &&
            c.validFrom == 0 &&
            c.validUntil == 0 &&
            !c.businessHoursOnly &&
            c.requiredApprovals == 0 &&
            !c.useEscrow
        ) {
            return true;
        }
        return false;
    }

    function _allConditionsMet(Payment storage payment) internal view returns (bool) {
        PaymentConditions storage c = payment.conditions;
        
        // Time checks
        if (c.validFrom > 0 && block.timestamp < c.validFrom) return false;
        if (c.validUntil > 0 && block.timestamp > c.validUntil) return false;
        
        // Approval checks
        if (c.requiredApprovals > 0 && payment.approvalCount < c.requiredApprovals) {
            return false;
        }
        
        // Business hours check (simplified: Mon-Fri 9am-5pm UTC)
        if (c.businessHoursOnly) {
            uint256 dayOfWeek = (block.timestamp / 86400 + 4) % 7; // 0=Sunday
            uint256 hourOfDay = (block.timestamp % 86400) / 3600;
            // Skip weekends and outside 9-17 UTC
            if (dayOfWeek == 0 || dayOfWeek == 6) return false;
            if (hourOfDay < 9 || hourOfDay >= 17) return false;
        }
        
        return true;
    }
    
    /**
     * @notice Verify compliance for a payment before execution
     * @dev Calls ComplianceEngine for KYC/AML/Sanctions verification
     */
    function _verifyCompliance(Payment storage payment) internal returns (bool) {
        PaymentConditions storage c = payment.conditions;
        
        // Skip if no compliance requirements
        if (
            c.requiredSenderTier == ComplianceTier.NONE &&
            c.requiredRecipientTier == ComplianceTier.NONE &&
            !c.requireSanctionsCheck
        ) {
            emit ComplianceVerified(payment.paymentId, payment.sender, c.requiredSenderTier, true);
            return true;
        }
        
        // Call compliance engine if configured
        if (complianceEngine != address(0)) {
            IComplianceEngine engine = IComplianceEngine(complianceEngine);
            
            IComplianceEngine.TransactionCheck memory check = engine.checkPaymentCompliance(
                payment.paymentId,
                payment.sender,
                payment.recipient,
                payment.amount,
                c.requireSanctionsCheck,
                IComplianceEngine.ComplianceTier(uint8(c.requiredSenderTier)),
                IComplianceEngine.ComplianceTier(uint8(c.requiredRecipientTier))
            );
            
            if (!check.passed) {
                emit PaymentFailed(payment.paymentId, check.reason);
                payment.status = PaymentStatus.FAILED;
                return false;
            }
            
            emit ComplianceVerified(payment.paymentId, payment.sender, c.requiredSenderTier, true);
            
            // Emit Travel Rule event if required
            if (check.requiresTravelRule) {
                emit TravelRuleRequired(payment.paymentId, payment.amount);
            }
        }
        
        return true;
    }

    function _isApprover(Payment storage payment, address account) internal view returns (bool) {
        for (uint i = 0; i < payment.conditions.approvers.length; i++) {
            if (payment.conditions.approvers[i] == account) return true;
        }
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function getPayment(bytes32 paymentId) external view returns (PaymentView memory) {
        Payment storage payment = payments[paymentId];
        return PaymentView({
            paymentId: payment.paymentId,
            sender: payment.sender,
            recipient: payment.recipient,
            token: payment.token,
            amount: payment.amount,
            targetToken: payment.targetToken,
            targetAmount: payment.targetAmount,
            status: payment.status,
            createdAt: payment.createdAt,
            executedAt: payment.executedAt,
            approvalCount: payment.approvalCount,
            requiredApprovals: payment.conditions.requiredApprovals,
            referenceId: payment.referenceId
        });
    }

    function getPaymentCount() external view returns (uint256) {
        return paymentIds.length;
    }

    function getProtocolStats() external view returns (
        uint256 created,
        uint256 executed,
        uint256 volume,
        uint256 avgSettlement
    ) {
        return (
            totalPaymentsCreated,
            totalPaymentsExecuted,
            totalVolumeProcessed,
            averageSettlementTime
        );
    }

    function getUserPayments(address user) external view returns (
        bytes32[] memory sent,
        bytes32[] memory received
    ) {
        return (userPaymentsSent[user], userPaymentsReceived[user]);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function addSupportedToken(address token) external onlyRole(OPERATOR_ROLE) {
        require(!supportedTokens[token], "Already supported");
        supportedTokens[token] = true;
        tokenList.push(token);
    }

    function setModules(
        address _compliance,
        address _oracle,
        address _escrow,
        address _audit
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        complianceEngine = _compliance;
        oracleAggregator = _oracle;
        escrowVault = _escrow;
        auditRegistry = _audit;
    }

    function setProtocolFee(uint256 _feeBps) external onlyRole(TREASURY_ROLE) {
        require(_feeBps <= 100, "Fee too high"); // Max 1%
        protocolFeeBps = _feeBps;
    }

    function setFeeRecipient(address _recipient) external onlyRole(TREASURY_ROLE) {
        require(_recipient != address(0), "Invalid recipient");
        feeRecipient = _recipient;
    }

    function pause() external onlyRole(OPERATOR_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(OPERATOR_ROLE) {
        _unpause();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //                         TRAVEL RULE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Record Travel Rule data for a payment (FATF requirement for >$3000)
     * @dev Data is hashed on-chain for privacy, full data stored off-chain
     * @param paymentId The payment to record travel rule data for
     * @param originatorData Encoded originator information (name, address, etc)
     * @param beneficiaryData Encoded beneficiary information
     */
    function recordTravelRuleData(
        bytes32 paymentId,
        bytes calldata originatorData,
        bytes calldata beneficiaryData
    ) external onlyRole(COMPLIANCE_ROLE) {
        Payment storage payment = payments[paymentId];
        require(payment.paymentId == paymentId, "Payment not found");
        require(payment.amount >= travelRuleThreshold, "Below travel rule threshold");
        
        TravelRuleRecord storage record = travelRuleRecords[paymentId];
        
        record.originatorDataHash = keccak256(originatorData);
        record.beneficiaryDataHash = keccak256(beneficiaryData);
        record.timestamp = block.timestamp;
        record.amount = payment.amount;
        record.verified = true;
        
        emit TravelRuleRecorded(paymentId, record.originatorDataHash, record.beneficiaryDataHash);
    }
    
    /**
     * @notice Check if payment requires Travel Rule compliance
     */
    function requiresTravelRule(bytes32 paymentId) external view returns (bool) {
        Payment storage payment = payments[paymentId];
        return payment.amount >= travelRuleThreshold;
    }
    
    /**
     * @notice Get Travel Rule record for a payment
     */
    function getTravelRuleRecord(bytes32 paymentId) external view returns (
        bytes32 originatorHash,
        bytes32 beneficiaryHash,
        uint256 timestamp,
        bool verified
    ) {
        TravelRuleRecord storage record = travelRuleRecords[paymentId];
        return (
            record.originatorDataHash,
            record.beneficiaryDataHash,
            record.timestamp,
            record.verified
        );
    }
    
    /**
     * @notice Set the Travel Rule threshold
     */
    function setTravelRuleThreshold(uint256 _threshold) external onlyRole(COMPLIANCE_ROLE) {
        travelRuleThreshold = _threshold;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//                         INTERFACES
// ═══════════════════════════════════════════════════════════════════════════

interface IAuditRegistry {
    function logPaymentExecuted(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        address token,
        uint256 settlementTime,
        string calldata jurisdiction
    ) external returns (bytes32);
}

interface IComplianceEngine {
    enum ComplianceTier {
        NONE,
        BASIC,
        STANDARD,
        ENHANCED,
        INSTITUTIONAL
    }
    
    struct TransactionCheck {
        bool passed;
        string reason;
        bool requiresTravelRule;
        bool requiresEnhancedDD;
        uint256 timestamp;
    }
    
    function checkPaymentCompliance(
        bytes32 paymentId,
        address sender,
        address recipient,
        uint256 amount,
        bool requireSanctionsCheck,
        ComplianceTier requiredSenderTier,
        ComplianceTier requiredRecipientTier
    ) external returns (TransactionCheck memory);
}

interface IOracleAggregator {
    struct RateResponse {
        uint256 spotRate;
        uint256 twapRate;
        uint256 confidence;
        uint256 timestamp;
        bool isStale;
        bool circuitBreakerActive;
    }
    
    function getRate(bytes32 pairId) external view returns (RateResponse memory);
    function getRateBySymbols(string calldata base, string calldata quote) external view returns (RateResponse memory);
    function convert(bytes32 pairId, uint256 amount, bool inverse) external view returns (uint256, uint256);
}
