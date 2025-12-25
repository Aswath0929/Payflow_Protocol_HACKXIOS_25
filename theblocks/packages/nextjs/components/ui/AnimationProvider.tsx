"use client";

import { createContext, useContext, ReactNode, useState, useEffect } from "react";

// ============================================
// GLOBAL ANIMATION CONTEXT
// ============================================

interface AnimationConfig {
  enabled: boolean;
  reducedMotion: boolean;
  defaultDuration: number;
  defaultDelay: number;
}

interface AnimationContextType {
  config: AnimationConfig;
  isPageLoaded: boolean;
  setEnabled: (enabled: boolean) => void;
}

const defaultConfig: AnimationConfig = {
  enabled: true,
  reducedMotion: false,
  defaultDuration: 700,
  defaultDelay: 0,
};

const AnimationContext = createContext<AnimationContextType>({
  config: defaultConfig,
  isPageLoaded: false,
  setEnabled: () => {},
});

export function useAnimation() {
  return useContext(AnimationContext);
}

// ============================================
// ANIMATION PROVIDER
// ============================================

interface AnimationProviderProps {
  children: ReactNode;
}

/**
 * AnimationProvider - Global animation context provider
 * Wrap your app with this to enable universal scroll animations
 * 
 * Features:
 * - Respects user's reduced motion preference
 * - Global enable/disable toggle
 * - Page load detection for initial animations
 * 
 * @example
 * // In your layout.tsx or _app.tsx
 * <AnimationProvider>
 *   <App />
 * </AnimationProvider>
 */
export function AnimationProvider({ children }: AnimationProviderProps) {
  const [config, setConfig] = useState<AnimationConfig>(defaultConfig);
  const [isPageLoaded, setIsPageLoaded] = useState(false);

  useEffect(() => {
    // Check for reduced motion preference
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    
    const updateReducedMotion = () => {
      setConfig(prev => ({
        ...prev,
        reducedMotion: mediaQuery.matches,
        enabled: !mediaQuery.matches,
      }));
    };

    updateReducedMotion();
    mediaQuery.addEventListener("change", updateReducedMotion);

    // Mark page as loaded after a brief delay
    const timer = setTimeout(() => setIsPageLoaded(true), 100);

    return () => {
      mediaQuery.removeEventListener("change", updateReducedMotion);
      clearTimeout(timer);
    };
  }, []);

  const setEnabled = (enabled: boolean) => {
    setConfig(prev => ({ ...prev, enabled }));
  };

  return (
    <AnimationContext.Provider value={{ config, isPageLoaded, setEnabled }}>
      {children}
    </AnimationContext.Provider>
  );
}

// ============================================
// MOTION WRAPPER - Simple animated div
// ============================================

interface MotionProps {
  children: ReactNode;
  className?: string;
  animate?: "fadeIn" | "slideUp" | "slideDown" | "slideLeft" | "slideRight" | "scale" | "blur";
  delay?: number;
  duration?: number;
}

/**
 * Motion - Simple animated wrapper that uses global animation context
 * Automatically respects user's reduced motion preference
 * 
 * @example
 * <Motion animate="fadeIn" delay={200}>
 *   <div>This will fade in after 200ms</div>
 * </Motion>
 */
export function Motion({ 
  children, 
  className = "", 
  animate = "fadeIn",
  delay = 0,
  duration = 700 
}: MotionProps) {
  const { config, isPageLoaded } = useAnimation();
  const [hasAnimated, setHasAnimated] = useState(false);

  useEffect(() => {
    if (isPageLoaded && config.enabled) {
      const timer = setTimeout(() => setHasAnimated(true), delay);
      return () => clearTimeout(timer);
    } else if (!config.enabled) {
      setHasAnimated(true);
    }
  }, [isPageLoaded, config.enabled, delay]);

  const getStyles = () => {
    if (!config.enabled || config.reducedMotion) {
      return {}; // No animations for reduced motion
    }

    const baseStyles = {
      transition: `opacity ${duration}ms ease-out, transform ${duration}ms ease-out, filter ${duration}ms ease-out`,
    };

    const animations: Record<string, { initial: React.CSSProperties; final: React.CSSProperties }> = {
      fadeIn: {
        initial: { opacity: 0 },
        final: { opacity: 1 },
      },
      slideUp: {
        initial: { opacity: 0, transform: "translateY(30px)" },
        final: { opacity: 1, transform: "translateY(0)" },
      },
      slideDown: {
        initial: { opacity: 0, transform: "translateY(-30px)" },
        final: { opacity: 1, transform: "translateY(0)" },
      },
      slideLeft: {
        initial: { opacity: 0, transform: "translateX(30px)" },
        final: { opacity: 1, transform: "translateX(0)" },
      },
      slideRight: {
        initial: { opacity: 0, transform: "translateX(-30px)" },
        final: { opacity: 1, transform: "translateX(0)" },
      },
      scale: {
        initial: { opacity: 0, transform: "scale(0.95)" },
        final: { opacity: 1, transform: "scale(1)" },
      },
      blur: {
        initial: { opacity: 0, filter: "blur(10px)" },
        final: { opacity: 1, filter: "blur(0px)" },
      },
    };

    const animation = animations[animate] || animations.fadeIn;
    return {
      ...baseStyles,
      ...(hasAnimated ? animation.final : animation.initial),
    };
  };

  return (
    <div className={className} style={getStyles()}>
      {children}
    </div>
  );
}

// ============================================
// REVEAL ON SCROLL - Scroll-triggered animation
// ============================================

interface RevealProps {
  children: ReactNode;
  className?: string;
  direction?: "up" | "down" | "left" | "right" | "fade" | "scale";
  delay?: number;
  threshold?: number;
}

/**
 * Reveal - Scroll-triggered animation wrapper
 * Uses IntersectionObserver to trigger animations when element enters viewport
 * 
 * @example
 * <Reveal direction="up" delay={100}>
 *   <Card>Content that reveals on scroll</Card>
 * </Reveal>
 */
export function Reveal({ 
  children, 
  className = "", 
  direction = "up",
  delay = 0,
  threshold = 0.1 
}: RevealProps) {
  const { config } = useAnimation();
  const [isVisible, setIsVisible] = useState(!config.enabled);
  const [ref, setRef] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!config.enabled || config.reducedMotion) {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (ref) observer.unobserve(ref);
        }
      },
      { threshold, rootMargin: "0px 0px -50px 0px" }
    );

    if (ref) observer.observe(ref);
    return () => { if (ref) observer.unobserve(ref); };
  }, [ref, threshold, config.enabled, config.reducedMotion]);

  const getTransform = () => {
    switch (direction) {
      case "up": return "translateY(40px)";
      case "down": return "translateY(-40px)";
      case "left": return "translateX(40px)";
      case "right": return "translateX(-40px)";
      case "scale": return "scale(0.95)";
      case "fade": return "none";
      default: return "translateY(40px)";
    }
  };

  const styles: React.CSSProperties = config.enabled && !config.reducedMotion ? {
    opacity: isVisible ? 1 : 0,
    transform: isVisible ? "none" : getTransform(),
    transition: `opacity 600ms ease-out, transform 600ms ease-out`,
    transitionDelay: `${delay}ms`,
  } : {};

  return (
    <div ref={setRef} className={className} style={styles}>
      {children}
    </div>
  );
}






