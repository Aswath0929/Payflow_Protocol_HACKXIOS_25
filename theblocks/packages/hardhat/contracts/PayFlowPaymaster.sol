// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  PAYFLOW PAYMASTER - VISA-STYLE GASLESS TRANSACTIONS                     ║
 * ║  ERC-4337 Account Abstraction Implementation                             ║
 * ╠══════════════════════════════════════════════════════════════════════════╣
 * ║  Built for Hackxios 2K25 - PayPal & Visa Track                           ║
 * ║                                                                           ║
 * ║  Features:                                                                ║
 * ║  • Gasless PYUSD/USDC transfers (users pay in stablecoin)               ║
 * ║  • Corporate sponsorship model (Visa/PayPal style)                       ║
 * ║  • Whitelist-based access control                                        ║
 * ║  • Rate limiting for abuse prevention                                    ║
 * ║  • Automatic gas recoupment from stablecoin                              ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 */

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title IEntryPoint
 * @notice Simplified interface for ERC-4337 EntryPoint
 */
interface IEntryPoint {
    struct UserOperation {
        address sender;
        uint256 nonce;
        bytes initCode;
        bytes callData;
        uint256 callGasLimit;
        uint256 verificationGasLimit;
        uint256 preVerificationGas;
        uint256 maxFeePerGas;
        uint256 maxPriorityFeePerGas;
        bytes paymasterAndData;
        bytes signature;
    }
}

/**
 * @title PayFlowPaymaster
 * @notice Visa-style gasless transaction sponsorship for stablecoin transfers
 * @dev Implements ERC-4337 Paymaster pattern for PYUSD/USDC payments
 * 
 * Key Innovation: Users can send stablecoins without holding ETH
 * - Transaction fees are paid from the transfer amount (tiny % deduction)
 * - OR sponsored by corporate partners (Visa/PayPal model)
 * - Zero friction onboarding for enterprise users
 */
contract PayFlowPaymaster is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;

    // ═══════════════════════════════════════════════════════════════════════
    // STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @notice Supported stablecoins for gas payment
    mapping(address => bool) public supportedTokens;
    
    /// @notice Token to ETH exchange rates (scaled by 1e18)
    mapping(address => uint256) public tokenToEthRate;
    
    /// @notice Whitelisted addresses for sponsored transactions
    mapping(address => bool) public whitelistedUsers;
    
    /// @notice Corporate sponsors who fund gas for users
    mapping(address => uint256) public sponsorBalances;
    
    /// @notice User transaction count for rate limiting
    mapping(address => uint256) public userTxCount;
    
    /// @notice Last transaction timestamp per user
    mapping(address => uint256) public lastTxTimestamp;
    
    /// @notice Gas sponsorship deposits
    uint256 public totalDeposits;
    
    /// @notice Fee percentage for gas recoupment (basis points, 100 = 1%)
    uint256 public feePercentage = 10; // 0.1% default
    
    /// @notice Maximum fee cap in USD (scaled by 1e6 for USDC/PYUSD decimals)
    uint256 public maxFeeCap = 5 * 1e6; // $5 max
    
    /// @notice Rate limit: max transactions per user per hour
    uint256 public rateLimitPerHour = 100;
    
    /// @notice Minimum deposit required to become a sponsor
    uint256 public minSponsorDeposit = 0.1 ether;

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════
    
    event GasSponsored(
        address indexed user,
        address indexed sponsor,
        uint256 ethAmount,
        uint256 tokenAmount
    );
    
    event UserWhitelisted(address indexed user, bool status);
    
    event SponsorDeposited(address indexed sponsor, uint256 amount);
    
    event SponsorWithdrawn(address indexed sponsor, uint256 amount);
    
    event TokenSupported(address indexed token, uint256 rate);
    
    event FeeCollected(
        address indexed user,
        address indexed token,
        uint256 amount
    );
    
    event GaslessTransferExecuted(
        address indexed sender,
        address indexed recipient,
        address indexed token,
        uint256 amount,
        uint256 feeDeducted
    );

    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════
    
    constructor() Ownable(msg.sender) {
        // Initialize with typical ETH price (~$2000)
        // Rate = how many token units (6 decimals) per 1 ETH
        // At $2000/ETH, 1 ETH = 2000 USDC/PYUSD
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SPONSOR FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Deposit ETH to sponsor user transactions
     * @dev Corporate partners use this to fund gasless transactions
     */
    function depositAsSponsore() external payable nonReentrant {
        require(msg.value >= minSponsorDeposit, "Below minimum deposit");
        
        sponsorBalances[msg.sender] += msg.value;
        totalDeposits += msg.value;
        
        emit SponsorDeposited(msg.sender, msg.value);
    }
    
    /**
     * @notice Withdraw sponsor balance
     * @param amount Amount to withdraw
     */
    function withdrawSponsorBalance(uint256 amount) external nonReentrant {
        require(sponsorBalances[msg.sender] >= amount, "Insufficient balance");
        
        sponsorBalances[msg.sender] -= amount;
        totalDeposits -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "ETH transfer failed");
        
        emit SponsorWithdrawn(msg.sender, amount);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GASLESS TRANSFER FUNCTIONS (CORE INNOVATION)
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Execute a gasless stablecoin transfer
     * @dev User signs intent, relayer submits tx, Paymaster covers gas
     * @param token Stablecoin address (PYUSD, USDC, etc.)
     * @param recipient Transfer recipient
     * @param amount Amount to transfer
     * @param sponsor Address of sponsor covering gas (or address(0) for fee deduction)
     * @return netAmount Amount received by recipient after any fees
     */
    function executeGaslessTransfer(
        address token,
        address recipient,
        uint256 amount,
        address sponsor
    ) external nonReentrant whenNotPaused returns (uint256 netAmount) {
        require(supportedTokens[token], "Token not supported");
        require(recipient != address(0), "Invalid recipient");
        require(amount > 0, "Amount must be positive");
        
        // Rate limiting check
        _checkRateLimit(msg.sender);
        
        uint256 gasFee = 0;
        
        if (sponsor != address(0)) {
            // Sponsored transaction - deduct from sponsor balance
            gasFee = _estimateGasCost();
            require(sponsorBalances[sponsor] >= gasFee, "Sponsor insufficient balance");
            
            sponsorBalances[sponsor] -= gasFee;
            totalDeposits -= gasFee;
            
            netAmount = amount; // Full amount to recipient
            
            emit GasSponsored(msg.sender, sponsor, gasFee, 0);
        } else if (whitelistedUsers[msg.sender]) {
            // Whitelisted user - free transaction
            netAmount = amount;
        } else {
            // Fee deduction mode - take small % from transfer
            uint256 tokenFee = _calculateTokenFee(token, amount);
            netAmount = amount - tokenFee;
            gasFee = tokenFee;
            
            // Transfer fee to contract for later withdrawal
            IERC20(token).safeTransferFrom(msg.sender, address(this), tokenFee);
            
            emit FeeCollected(msg.sender, token, tokenFee);
        }
        
        // Execute the actual transfer
        IERC20(token).safeTransferFrom(msg.sender, recipient, netAmount);
        
        emit GaslessTransferExecuted(msg.sender, recipient, token, amount, gasFee);
        
        return netAmount;
    }
    
    /**
     * @notice Check if a user qualifies for sponsored/free transactions
     * @param user Address to check
     * @return isSponsored True if user has a sponsor or is whitelisted
     * @return sponsor Address of sponsor (if any)
     */
    function checkSponsorshipStatus(address user) 
        external 
        view 
        returns (bool isSponsored, address sponsor) 
    {
        if (whitelistedUsers[user]) {
            return (true, address(0));
        }
        
        // In a full implementation, we'd check sponsor mappings
        return (false, address(0));
    }
    
    /**
     * @notice Estimate gas cost in ETH for a typical transfer
     * @return Estimated gas cost in wei
     */
    function _estimateGasCost() internal view returns (uint256) {
        // Typical ERC20 transfer: ~65,000 gas
        // With Paymaster overhead: ~150,000 gas
        uint256 gasEstimate = 150000;
        uint256 gasPrice = tx.gasprice > 0 ? tx.gasprice : 20 gwei;
        
        return gasEstimate * gasPrice;
    }
    
    /**
     * @notice Calculate fee in token terms
     * @param token Token address
     * @param amount Transfer amount
     * @return Fee amount in token units
     */
    function _calculateTokenFee(address token, uint256 amount) 
        internal 
        view 
        returns (uint256) 
    {
        // Calculate percentage fee
        uint256 percentageFee = (amount * feePercentage) / 10000;
        
        // Cap at maximum
        if (percentageFee > maxFeeCap) {
            return maxFeeCap;
        }
        
        // Minimum fee to cover gas
        uint256 gasCostInEth = _estimateGasCost();
        uint256 rate = tokenToEthRate[token];
        uint256 minFee = rate > 0 ? (gasCostInEth * rate) / 1e18 : 0;
        
        return percentageFee > minFee ? percentageFee : minFee;
    }
    
    /**
     * @notice Check and update rate limiting
     * @param user User address
     */
    function _checkRateLimit(address user) internal {
        uint256 currentHour = block.timestamp / 1 hours;
        uint256 lastHour = lastTxTimestamp[user] / 1 hours;
        
        if (currentHour > lastHour) {
            // New hour, reset counter
            userTxCount[user] = 1;
        } else {
            // Same hour, increment and check
            userTxCount[user]++;
            require(userTxCount[user] <= rateLimitPerHour, "Rate limit exceeded");
        }
        
        lastTxTimestamp[user] = block.timestamp;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ERC-4337 PAYMASTER INTERFACE (Simplified)
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Validate a UserOperation for gas sponsorship
     * @dev Called by EntryPoint before executing UserOp
     * @param userOp The UserOperation to validate
     * @param maxCost Maximum cost the Paymaster agrees to pay
     * @return context Arbitrary data to pass to postOp
     * @return validationData Signature validity period
     */
    function validatePaymasterUserOp(
        IEntryPoint.UserOperation calldata userOp,
        bytes32, // userOpHash - unused in simplified version
        uint256 maxCost
    ) external view returns (bytes memory context, uint256 validationData) {
        // Check if sender is whitelisted or has sponsor
        require(
            whitelistedUsers[userOp.sender] || totalDeposits >= maxCost,
            "Paymaster: not sponsored"
        );
        
        // Return encoded context for postOp
        context = abi.encode(userOp.sender, maxCost);
        validationData = 0; // Valid immediately, no time restriction
        
        return (context, validationData);
    }
    
    /**
     * @notice Post-operation hook for gas accounting
     * @dev Called by EntryPoint after UserOp execution
     * @param mode 0 = success, 1 = revert in op, 2 = revert in postOp
     * @param context Data from validatePaymasterUserOp
     * @param actualGasCost Actual gas used
     */
    function postOp(
        uint8 mode,
        bytes calldata context,
        uint256 actualGasCost
    ) external {
        // Decode context
        (address user, ) = abi.decode(context, (address, uint256));
        
        if (mode == 0) {
            // Success - log the sponsorship
            emit GasSponsored(user, address(this), actualGasCost, 0);
        }
        // On failure, gas is still consumed but we don't need special handling
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Add or remove supported token
     * @param token Token address
     * @param rate Token to ETH exchange rate (tokens per ETH, scaled by 1e18)
     */
    function setSupportedToken(address token, uint256 rate) external onlyOwner {
        supportedTokens[token] = rate > 0;
        tokenToEthRate[token] = rate;
        
        emit TokenSupported(token, rate);
    }
    
    /**
     * @notice Whitelist or remove user from free transactions
     * @param user User address
     * @param status Whitelist status
     */
    function setWhitelistStatus(address user, bool status) external onlyOwner {
        whitelistedUsers[user] = status;
        emit UserWhitelisted(user, status);
    }
    
    /**
     * @notice Batch whitelist multiple users
     * @param users Array of user addresses
     * @param status Whitelist status for all
     */
    function batchWhitelist(address[] calldata users, bool status) external onlyOwner {
        for (uint256 i = 0; i < users.length; i++) {
            whitelistedUsers[users[i]] = status;
            emit UserWhitelisted(users[i], status);
        }
    }
    
    /**
     * @notice Update fee percentage
     * @param newFee New fee in basis points (100 = 1%)
     */
    function setFeePercentage(uint256 newFee) external onlyOwner {
        require(newFee <= 100, "Fee too high"); // Max 1%
        feePercentage = newFee;
    }
    
    /**
     * @notice Update rate limit
     * @param newLimit New transactions per hour limit
     */
    function setRateLimit(uint256 newLimit) external onlyOwner {
        require(newLimit >= 10, "Limit too low");
        rateLimitPerHour = newLimit;
    }
    
    /**
     * @notice Pause contract in emergency
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @notice Unpause contract
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @notice Withdraw collected fees
     * @param token Token to withdraw
     * @param to Recipient address
     * @param amount Amount to withdraw
     */
    function withdrawFees(address token, address to, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(to, amount);
    }
    
    /**
     * @notice Emergency ETH withdrawal
     */
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Get estimated fee for a transfer
     * @param token Token address
     * @param amount Transfer amount
     * @return Estimated fee in token units
     */
    function estimateFee(address token, uint256 amount) 
        external 
        view 
        returns (uint256) 
    {
        if (!supportedTokens[token]) return 0;
        return _calculateTokenFee(token, amount);
    }
    
    /**
     * @notice Check if user is rate limited
     * @param user User address
     * @return limited True if user is rate limited
     * @return remaining Transactions remaining this hour
     */
    function isRateLimited(address user) 
        external 
        view 
        returns (bool limited, uint256 remaining) 
    {
        uint256 currentHour = block.timestamp / 1 hours;
        uint256 lastHour = lastTxTimestamp[user] / 1 hours;
        
        if (currentHour > lastHour) {
            return (false, rateLimitPerHour);
        }
        
        uint256 count = userTxCount[user];
        if (count >= rateLimitPerHour) {
            return (true, 0);
        }
        
        return (false, rateLimitPerHour - count);
    }
    
    /**
     * @notice Get contract info
     */
    function getPaymasterInfo() 
        external 
        view 
        returns (
            uint256 deposits,
            uint256 fee,
            uint256 maxFee,
            uint256 rateLimit
        ) 
    {
        return (totalDeposits, feePercentage, maxFeeCap, rateLimitPerHour);
    }
    
    // Allow receiving ETH
    receive() external payable {
        sponsorBalances[msg.sender] += msg.value;
        totalDeposits += msg.value;
        emit SponsorDeposited(msg.sender, msg.value);
    }
}
