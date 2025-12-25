// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title SmartEscrow
 * @notice Programmable escrow with automatic release conditions
 * @dev Supports time-based, oracle-based, and multi-sig release conditions
 * 
 * USE CASES:
 * - Cross-border trade finance (release on delivery confirmation)
 * - Milestone-based payments (release on approval)
 * - Dispute-protected transactions (automatic release after period)
 */
contract SmartEscrow is AccessControl, ReentrancyGuard {
    using SafeERC20 for IERC20;

    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant ARBITER_ROLE = keccak256("ARBITER_ROLE");

    enum EscrowStatus {
        ACTIVE,         // Funds locked, awaiting conditions
        RELEASED,       // Funds released to recipient
        REFUNDED,       // Funds returned to sender
        DISPUTED        // Under dispute resolution
    }

    enum ReleaseCondition {
        TIME_BASED,     // Auto-release after timestamp
        APPROVAL,       // Manual approval by parties
        ORACLE,         // External oracle confirmation
        MULTI_SIG       // Multiple signatures required
    }

    struct Escrow {
        bytes32 escrowId;
        bytes32 paymentId;      // Link to PayFlowCore payment
        
        address depositor;
        address beneficiary;
        address arbiter;        // Dispute resolver
        
        address token;
        uint256 amount;
        uint256 fee;
        
        EscrowStatus status;
        ReleaseCondition conditionType;
        
        // Time conditions
        uint256 releaseTime;    // Auto-release timestamp
        uint256 disputeDeadline; // Deadline to raise dispute
        
        // Approval tracking
        bool depositorApproved;
        bool beneficiaryApproved;
        
        // Multi-sig
        uint256 requiredSignatures;
        uint256 signatureCount;
        mapping(address => bool) hasSigned;
        
        // Oracle condition
        bytes32 oracleConditionId;
        bool oracleConditionMet;
        
        // Metadata
        uint256 createdAt;
        string description;
    }

    struct EscrowView {
        bytes32 escrowId;
        bytes32 paymentId;
        address depositor;
        address beneficiary;
        address token;
        uint256 amount;
        EscrowStatus status;
        ReleaseCondition conditionType;
        uint256 releaseTime;
        bool depositorApproved;
        bool beneficiaryApproved;
        uint256 createdAt;
    }

    // State
    mapping(bytes32 => Escrow) private escrows;
    bytes32[] public escrowIds;
    
    mapping(address => bytes32[]) public userEscrowsAsDepositor;
    mapping(address => bytes32[]) public userEscrowsAsBeneficiary;
    
    uint256 public escrowFeeBps = 5; // 0.05%
    address public feeRecipient;
    address public payFlowCore;
    
    // Statistics
    uint256 public totalEscrowsCreated;
    uint256 public totalValueLocked;
    uint256 public totalReleased;

    // Events
    event EscrowCreated(
        bytes32 indexed escrowId,
        bytes32 indexed paymentId,
        address depositor,
        address beneficiary,
        uint256 amount
    );
    
    event EscrowApproved(
        bytes32 indexed escrowId,
        address indexed approver
    );
    
    event EscrowReleased(
        bytes32 indexed escrowId,
        address indexed beneficiary,
        uint256 amount
    );
    
    event EscrowRefunded(
        bytes32 indexed escrowId,
        address indexed depositor,
        uint256 amount
    );
    
    event EscrowDisputed(
        bytes32 indexed escrowId,
        address indexed disputedBy,
        string reason
    );
    
    event DisputeResolved(
        bytes32 indexed escrowId,
        address indexed resolvedBy,
        address recipient,
        uint256 amount
    );

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        _grantRole(ARBITER_ROLE, msg.sender);
        feeRecipient = msg.sender;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ESCROW CREATION
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Create a new time-based escrow
     */
    function createTimeBasedEscrow(
        bytes32 paymentId,
        address beneficiary,
        address token,
        uint256 amount,
        uint256 releaseTime,
        uint256 disputeWindow,
        string calldata description
    ) external nonReentrant returns (bytes32 escrowId) {
        require(beneficiary != address(0), "Invalid beneficiary");
        require(amount > 0, "Amount must be positive");
        require(releaseTime > block.timestamp, "Release time must be future");
        
        escrowId = _createEscrow(
            paymentId,
            msg.sender,
            beneficiary,
            token,
            amount,
            ReleaseCondition.TIME_BASED,
            releaseTime,
            disputeWindow,
            description
        );
        
        return escrowId;
    }

    /**
     * @notice Create an approval-based escrow
     */
    function createApprovalEscrow(
        bytes32 paymentId,
        address beneficiary,
        address token,
        uint256 amount,
        uint256 disputeDeadline,
        string calldata description
    ) external nonReentrant returns (bytes32 escrowId) {
        require(beneficiary != address(0), "Invalid beneficiary");
        require(amount > 0, "Amount must be positive");
        
        escrowId = _createEscrow(
            paymentId,
            msg.sender,
            beneficiary,
            token,
            amount,
            ReleaseCondition.APPROVAL,
            0, // No auto-release
            disputeDeadline,
            description
        );
        
        return escrowId;
    }

    function _createEscrow(
        bytes32 paymentId,
        address depositor,
        address beneficiary,
        address token,
        uint256 amount,
        ReleaseCondition conditionType,
        uint256 releaseTime,
        uint256 disputeWindow,
        string calldata description
    ) internal returns (bytes32 escrowId) {
        escrowId = keccak256(abi.encodePacked(
            depositor,
            beneficiary,
            token,
            amount,
            block.timestamp,
            totalEscrowsCreated
        ));
        
        // Calculate fee
        uint256 fee = (amount * escrowFeeBps) / 10000;
        
        // Transfer tokens
        IERC20(token).safeTransferFrom(depositor, address(this), amount);
        
        // Store escrow
        Escrow storage escrow = escrows[escrowId];
        escrow.escrowId = escrowId;
        escrow.paymentId = paymentId;
        escrow.depositor = depositor;
        escrow.beneficiary = beneficiary;
        escrow.arbiter = msg.sender; // Default arbiter, can be changed
        escrow.token = token;
        escrow.amount = amount - fee;
        escrow.fee = fee;
        escrow.status = EscrowStatus.ACTIVE;
        escrow.conditionType = conditionType;
        escrow.releaseTime = releaseTime;
        escrow.disputeDeadline = block.timestamp + disputeWindow;
        escrow.createdAt = block.timestamp;
        escrow.description = description;
        
        // Track
        escrowIds.push(escrowId);
        userEscrowsAsDepositor[depositor].push(escrowId);
        userEscrowsAsBeneficiary[beneficiary].push(escrowId);
        totalEscrowsCreated++;
        totalValueLocked += escrow.amount;
        
        // Send fee
        if (fee > 0) {
            IERC20(token).safeTransfer(feeRecipient, fee);
        }
        
        emit EscrowCreated(escrowId, paymentId, depositor, beneficiary, escrow.amount);
        
        return escrowId;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ESCROW APPROVAL & RELEASE
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Approve escrow release (for approval-based escrows)
     */
    function approveRelease(bytes32 escrowId) external nonReentrant {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(escrow.status == EscrowStatus.ACTIVE, "Not active");
        
        if (msg.sender == escrow.depositor) {
            require(!escrow.depositorApproved, "Already approved");
            escrow.depositorApproved = true;
        } else if (msg.sender == escrow.beneficiary) {
            require(!escrow.beneficiaryApproved, "Already approved");
            escrow.beneficiaryApproved = true;
        } else {
            revert("Not a party to this escrow");
        }
        
        emit EscrowApproved(escrowId, msg.sender);
        
        // Check if both approved
        if (escrow.depositorApproved && escrow.conditionType == ReleaseCondition.APPROVAL) {
            _releaseEscrow(escrowId);
        }
    }

    /**
     * @notice Release escrow to beneficiary
     */
    function releaseEscrow(bytes32 escrowId) external nonReentrant {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(escrow.status == EscrowStatus.ACTIVE, "Not active");
        require(_canRelease(escrow), "Conditions not met");
        
        _releaseEscrow(escrowId);
    }

    function _releaseEscrow(bytes32 escrowId) internal {
        Escrow storage escrow = escrows[escrowId];
        
        escrow.status = EscrowStatus.RELEASED;
        totalValueLocked -= escrow.amount;
        totalReleased += escrow.amount;
        
        IERC20(escrow.token).safeTransfer(escrow.beneficiary, escrow.amount);
        
        emit EscrowReleased(escrowId, escrow.beneficiary, escrow.amount);
    }

    function _canRelease(Escrow storage escrow) internal view returns (bool) {
        if (escrow.conditionType == ReleaseCondition.TIME_BASED) {
            return block.timestamp >= escrow.releaseTime;
        }
        
        if (escrow.conditionType == ReleaseCondition.APPROVAL) {
            return escrow.depositorApproved;
        }
        
        if (escrow.conditionType == ReleaseCondition.ORACLE) {
            return escrow.oracleConditionMet;
        }
        
        if (escrow.conditionType == ReleaseCondition.MULTI_SIG) {
            return escrow.signatureCount >= escrow.requiredSignatures;
        }
        
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         DISPUTE HANDLING
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @notice Raise a dispute on an escrow
     */
    function raiseDispute(bytes32 escrowId, string calldata reason) external {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(escrow.status == EscrowStatus.ACTIVE, "Not active");
        require(
            msg.sender == escrow.depositor || msg.sender == escrow.beneficiary,
            "Not a party"
        );
        require(block.timestamp <= escrow.disputeDeadline, "Dispute window closed");
        
        escrow.status = EscrowStatus.DISPUTED;
        
        emit EscrowDisputed(escrowId, msg.sender, reason);
    }

    /**
     * @notice Resolve a dispute (arbiter only)
     */
    function resolveDispute(
        bytes32 escrowId,
        address recipient,
        uint256 amount
    ) external onlyRole(ARBITER_ROLE) {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(escrow.status == EscrowStatus.DISPUTED, "Not disputed");
        require(
            recipient == escrow.depositor || recipient == escrow.beneficiary,
            "Invalid recipient"
        );
        require(amount <= escrow.amount, "Amount too high");
        
        // Transfer decided amount to recipient
        if (amount > 0) {
            IERC20(escrow.token).safeTransfer(recipient, amount);
        }
        
        // Refund remainder to other party
        uint256 remainder = escrow.amount - amount;
        if (remainder > 0) {
            address other = recipient == escrow.beneficiary ? escrow.depositor : escrow.beneficiary;
            IERC20(escrow.token).safeTransfer(other, remainder);
        }
        
        escrow.status = EscrowStatus.RELEASED;
        totalValueLocked -= escrow.amount;
        
        emit DisputeResolved(escrowId, msg.sender, recipient, amount);
    }

    /**
     * @notice Request refund (depositor only, before release)
     */
    function requestRefund(bytes32 escrowId) external nonReentrant {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(msg.sender == escrow.depositor, "Not depositor");
        require(escrow.status == EscrowStatus.ACTIVE, "Not active");
        
        // For approval escrows, beneficiary must agree
        if (escrow.conditionType == ReleaseCondition.APPROVAL) {
            require(escrow.beneficiaryApproved, "Beneficiary must agree to refund");
        }
        
        escrow.status = EscrowStatus.REFUNDED;
        totalValueLocked -= escrow.amount;
        
        IERC20(escrow.token).safeTransfer(escrow.depositor, escrow.amount);
        
        emit EscrowRefunded(escrowId, escrow.depositor, escrow.amount);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function getEscrow(bytes32 escrowId) external view returns (EscrowView memory) {
        Escrow storage escrow = escrows[escrowId];
        return EscrowView({
            escrowId: escrow.escrowId,
            paymentId: escrow.paymentId,
            depositor: escrow.depositor,
            beneficiary: escrow.beneficiary,
            token: escrow.token,
            amount: escrow.amount,
            status: escrow.status,
            conditionType: escrow.conditionType,
            releaseTime: escrow.releaseTime,
            depositorApproved: escrow.depositorApproved,
            beneficiaryApproved: escrow.beneficiaryApproved,
            createdAt: escrow.createdAt
        });
    }

    function getProtocolStats() external view returns (
        uint256 created,
        uint256 locked,
        uint256 released
    ) {
        return (totalEscrowsCreated, totalValueLocked, totalReleased);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         ADMIN
    // ═══════════════════════════════════════════════════════════════════════════

    function setPayFlowCore(address _core) external onlyRole(DEFAULT_ADMIN_ROLE) {
        payFlowCore = _core;
    }

    function setFeeRecipient(address _recipient) external onlyRole(DEFAULT_ADMIN_ROLE) {
        feeRecipient = _recipient;
    }

    function setEscrowFee(uint256 _feeBps) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_feeBps <= 50, "Max 0.5%");
        escrowFeeBps = _feeBps;
    }

    /**
     * @notice Oracle callback to mark condition as met
     */
    function setOracleConditionMet(bytes32 escrowId) external onlyRole(OPERATOR_ROLE) {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.conditionType == ReleaseCondition.ORACLE, "Not oracle escrow");
        escrow.oracleConditionMet = true;
        
        emit OracleConditionMet(escrowId);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //                         MULTI-SIG ESCROW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Create a multi-signature escrow requiring multiple approvals
     */
    function createMultiSigEscrow(
        bytes32 paymentId,
        address beneficiary,
        address token,
        uint256 amount,
        uint256 requiredSigs,
        address[] calldata signers,
        uint256 disputeWindow,
        string calldata description
    ) external nonReentrant returns (bytes32 escrowId) {
        require(beneficiary != address(0), "Invalid beneficiary");
        require(amount > 0, "Amount must be positive");
        require(requiredSigs > 0 && requiredSigs <= signers.length, "Invalid sig requirement");
        
        escrowId = _createEscrow(
            paymentId,
            msg.sender,
            beneficiary,
            token,
            amount,
            ReleaseCondition.MULTI_SIG,
            0,
            disputeWindow,
            description
        );
        
        Escrow storage escrow = escrows[escrowId];
        escrow.requiredSignatures = requiredSigs;
        
        // Store signers in mapping
        for (uint i = 0; i < signers.length; i++) {
            escrowSigners[escrowId][signers[i]] = true;
        }
        
        emit MultiSigEscrowCreated(escrowId, requiredSigs, signers.length);
        
        return escrowId;
    }
    
    /**
     * @notice Sign a multi-sig escrow for release
     */
    function signMultiSigRelease(bytes32 escrowId) external nonReentrant {
        Escrow storage escrow = escrows[escrowId];
        require(escrow.escrowId == escrowId, "Escrow not found");
        require(escrow.status == EscrowStatus.ACTIVE, "Not active");
        require(escrow.conditionType == ReleaseCondition.MULTI_SIG, "Not multi-sig escrow");
        require(escrowSigners[escrowId][msg.sender], "Not an authorized signer");
        require(!escrow.hasSigned[msg.sender], "Already signed");
        
        escrow.hasSigned[msg.sender] = true;
        escrow.signatureCount++;
        
        emit MultiSigSigned(escrowId, msg.sender, escrow.signatureCount, escrow.requiredSignatures);
        
        // Auto-release if threshold met
        if (escrow.signatureCount >= escrow.requiredSignatures) {
            _releaseEscrow(escrowId);
        }
    }
    
    /**
     * @notice Create an oracle-based escrow with condition ID
     */
    function createOracleEscrow(
        bytes32 paymentId,
        address beneficiary,
        address token,
        uint256 amount,
        bytes32 oracleConditionId,
        uint256 disputeWindow,
        string calldata description
    ) external nonReentrant returns (bytes32 escrowId) {
        require(beneficiary != address(0), "Invalid beneficiary");
        require(amount > 0, "Amount must be positive");
        
        escrowId = _createEscrow(
            paymentId,
            msg.sender,
            beneficiary,
            token,
            amount,
            ReleaseCondition.ORACLE,
            0,
            disputeWindow,
            description
        );
        
        Escrow storage escrow = escrows[escrowId];
        escrow.oracleConditionId = oracleConditionId;
        
        emit OracleEscrowCreated(escrowId, oracleConditionId);
        
        return escrowId;
    }
    
    // Multi-sig signer tracking
    mapping(bytes32 => mapping(address => bool)) public escrowSigners;
    
    // Events for new escrow types
    event MultiSigEscrowCreated(bytes32 indexed escrowId, uint256 requiredSigs, uint256 totalSigners);
    event MultiSigSigned(bytes32 indexed escrowId, address indexed signer, uint256 currentSigs, uint256 requiredSigs);
    event OracleEscrowCreated(bytes32 indexed escrowId, bytes32 conditionId);
    event OracleConditionMet(bytes32 indexed escrowId);
}
