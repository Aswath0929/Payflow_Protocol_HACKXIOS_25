/**
 * ╔═══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                   AI SERVICE GUARD - Stability & Error Handling                       ║
 * ║                                                                                       ║
 * ║   Safety guards for AI service interactions:                                          ║
 * ║   • Automatic retry with exponential backoff                                          ║
 * ║   • Circuit breaker pattern for failing services                                      ║
 * ║   • Graceful degradation when AI is unavailable                                       ║
 * ║   • Request deduplication                                                             ║
 * ║   • Timeout management                                                                ║
 * ║                                                                                       ║
 * ║   Hackxios 2K25 - PayFlow Protocol                                                    ║
 * ╚═══════════════════════════════════════════════════════════════════════════════════════╝
 */

interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number; // ms before attempting to close circuit
}

interface RetryConfig {
  maxRetries: number;
  baseDelay: number; // ms
  maxDelay: number; // ms
  backoffMultiplier: number;
}

interface AIServiceConfig {
  baseUrl: string;
  timeout: number;
  circuitBreaker: CircuitBreakerConfig;
  retry: RetryConfig;
}

type CircuitState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

interface RequestCache<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

/**
 * AI Service Guard - Provides resilience patterns for AI API calls
 */
export class AIServiceGuard {
  private config: AIServiceConfig;
  private circuitState: CircuitState = 'CLOSED';
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private requestCache: Map<string, RequestCache<unknown>> = new Map();
  private pendingRequests: Map<string, Promise<unknown>> = new Map();

  constructor(config: Partial<AIServiceConfig> = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:8000',
      timeout: config.timeout || 30000,
      circuitBreaker: {
        failureThreshold: config.circuitBreaker?.failureThreshold || 5,
        successThreshold: config.circuitBreaker?.successThreshold || 2,
        timeout: config.circuitBreaker?.timeout || 60000,
      },
      retry: {
        maxRetries: config.retry?.maxRetries || 3,
        baseDelay: config.retry?.baseDelay || 1000,
        maxDelay: config.retry?.maxDelay || 10000,
        backoffMultiplier: config.retry?.backoffMultiplier || 2,
      },
    };
  }

  /**
   * Check if circuit breaker allows the request
   */
  private canMakeRequest(): boolean {
    const now = Date.now();
    
    switch (this.circuitState) {
      case 'CLOSED':
        return true;
        
      case 'OPEN':
        // Check if timeout has elapsed
        if (now - this.lastFailureTime >= this.config.circuitBreaker.timeout) {
          this.circuitState = 'HALF_OPEN';
          console.log('[AIServiceGuard] Circuit breaker moving to HALF_OPEN');
          return true;
        }
        return false;
        
      case 'HALF_OPEN':
        return true;
        
      default:
        return true;
    }
  }

  /**
   * Record a successful request
   */
  private recordSuccess(): void {
    this.failureCount = 0;
    
    if (this.circuitState === 'HALF_OPEN') {
      this.successCount++;
      if (this.successCount >= this.config.circuitBreaker.successThreshold) {
        this.circuitState = 'CLOSED';
        this.successCount = 0;
        console.log('[AIServiceGuard] Circuit breaker CLOSED - service recovered');
      }
    }
  }

  /**
   * Record a failed request
   */
  private recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.circuitState === 'HALF_OPEN') {
      this.circuitState = 'OPEN';
      console.log('[AIServiceGuard] Circuit breaker OPEN - service still failing');
    } else if (this.failureCount >= this.config.circuitBreaker.failureThreshold) {
      this.circuitState = 'OPEN';
      console.log('[AIServiceGuard] Circuit breaker OPEN - threshold reached');
    }
  }

  /**
   * Calculate delay for retry with exponential backoff
   */
  private getRetryDelay(attempt: number): number {
    const delay = this.config.retry.baseDelay * Math.pow(
      this.config.retry.backoffMultiplier,
      attempt
    );
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.3 * delay;
    return Math.min(delay + jitter, this.config.retry.maxDelay);
  }

  /**
   * Sleep for specified duration
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate cache key for request deduplication
   */
  private getCacheKey(endpoint: string, data: unknown): string {
    return `${endpoint}:${JSON.stringify(data)}`;
  }

  /**
   * Main request function with all safety guards
   */
  async request<T>(
    endpoint: string,
    data: unknown,
    options: {
      method?: 'GET' | 'POST';
      cacheTtl?: number; // ms, 0 to disable caching
      skipRetry?: boolean;
    } = {}
  ): Promise<T> {
    const { method = 'POST', cacheTtl = 0, skipRetry = false } = options;
    const cacheKey = this.getCacheKey(endpoint, data);
    
    // Check cache first
    if (cacheTtl > 0) {
      const cached = this.requestCache.get(cacheKey) as RequestCache<T> | undefined;
      if (cached && Date.now() - cached.timestamp < cached.ttl) {
        console.log('[AIServiceGuard] Cache hit for', endpoint);
        return cached.data;
      }
    }
    
    // Check for pending identical request (deduplication)
    const pending = this.pendingRequests.get(cacheKey);
    if (pending) {
      console.log('[AIServiceGuard] Deduplicating request to', endpoint);
      return pending as Promise<T>;
    }
    
    // Check circuit breaker
    if (!this.canMakeRequest()) {
      throw new Error(`AI service unavailable (circuit breaker OPEN). Retry after ${
        Math.ceil((this.config.circuitBreaker.timeout - (Date.now() - this.lastFailureTime)) / 1000)
      } seconds.`);
    }
    
    // Create request promise
    const requestPromise = this.executeWithRetry<T>(endpoint, data, method, skipRetry);
    
    // Store for deduplication
    this.pendingRequests.set(cacheKey, requestPromise);
    
    try {
      const result = await requestPromise;
      
      // Cache result if TTL specified
      if (cacheTtl > 0) {
        this.requestCache.set(cacheKey, {
          data: result,
          timestamp: Date.now(),
          ttl: cacheTtl,
        });
      }
      
      return result;
    } finally {
      // Clean up pending request
      this.pendingRequests.delete(cacheKey);
    }
  }

  /**
   * Execute request with retry logic
   */
  private async executeWithRetry<T>(
    endpoint: string,
    data: unknown,
    method: 'GET' | 'POST',
    skipRetry: boolean
  ): Promise<T> {
    let lastError: Error | null = null;
    const maxAttempts = skipRetry ? 1 : this.config.retry.maxRetries + 1;
    
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Wait before retry (not on first attempt)
        if (attempt > 0) {
          const delay = this.getRetryDelay(attempt - 1);
          console.log(`[AIServiceGuard] Retry attempt ${attempt}/${maxAttempts - 1} after ${delay}ms`);
          await this.sleep(delay);
        }
        
        const result = await this.executeRequest<T>(endpoint, data, method);
        this.recordSuccess();
        return result;
        
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        // Don't retry on client errors (4xx)
        if (lastError.message.includes('4') && lastError.message.includes('00')) {
          this.recordFailure();
          throw lastError;
        }
        
        console.warn(`[AIServiceGuard] Request failed (attempt ${attempt + 1}):`, lastError.message);
      }
    }
    
    this.recordFailure();
    throw lastError || new Error('Request failed after all retries');
  }

  /**
   * Execute single request with timeout
   */
  private async executeRequest<T>(
    endpoint: string,
    data: unknown,
    method: 'GET' | 'POST'
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
    
    try {
      const url = `${this.config.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: method === 'POST' ? JSON.stringify(data) : undefined,
        signal: controller.signal,
      });
      
      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorBody}`);
      }
      
      return await response.json() as T;
      
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw new Error(`Request timed out after ${this.config.timeout}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get current circuit breaker state
   */
  getCircuitState(): { state: CircuitState; failureCount: number; lastFailure: number } {
    return {
      state: this.circuitState,
      failureCount: this.failureCount,
      lastFailure: this.lastFailureTime,
    };
  }

  /**
   * Manually reset circuit breaker
   */
  resetCircuit(): void {
    this.circuitState = 'CLOSED';
    this.failureCount = 0;
    this.successCount = 0;
    console.log('[AIServiceGuard] Circuit breaker manually reset');
  }

  /**
   * Clear all cached requests
   */
  clearCache(): void {
    this.requestCache.clear();
    console.log('[AIServiceGuard] Cache cleared');
  }

  /**
   * Check if AI service is healthy
   */
  async healthCheck(): Promise<{ healthy: boolean; latency: number; error?: string }> {
    const start = Date.now();
    try {
      await this.request('/health', null, { method: 'GET', skipRetry: true });
      return {
        healthy: true,
        latency: Date.now() - start,
      };
    } catch (error) {
      return {
        healthy: false,
        latency: Date.now() - start,
        error: error instanceof Error ? error.message : 'Health check failed',
      };
    }
  }
}

// Singleton instance for global use
let globalGuard: AIServiceGuard | null = null;

/**
 * Get or create the global AI Service Guard instance
 */
export function getAIServiceGuard(config?: Partial<AIServiceConfig>): AIServiceGuard {
  if (!globalGuard || config) {
    globalGuard = new AIServiceGuard(config);
  }
  return globalGuard;
}

/**
 * Graceful degradation wrapper - returns fallback on failure
 */
export async function withFallback<T>(
  operation: () => Promise<T>,
  fallback: T,
  onError?: (error: Error) => void
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (onError) {
      onError(error instanceof Error ? error : new Error(String(error)));
    }
    console.warn('[AIServiceGuard] Operation failed, using fallback:', error);
    return fallback;
  }
}

/**
 * Default fallback response for fraud analysis
 */
export const FALLBACK_FRAUD_RESPONSE = {
  transaction_id: 'fallback',
  risk_score: 30, // Conservative medium-low score
  risk_level: 'MEDIUM',
  approved: true, // Allow transaction but flag for review
  explanation: 'AI service temporarily unavailable. Transaction allowed with enhanced monitoring.',
  confidence: 0.5,
  model: 'fallback',
  processing_time_ms: 0,
  signature: '',
  oracle_address: '',
  signed_at: Date.now(),
  message_hash: '',
  message: '',
  is_fallback: true,
};

export default AIServiceGuard;
