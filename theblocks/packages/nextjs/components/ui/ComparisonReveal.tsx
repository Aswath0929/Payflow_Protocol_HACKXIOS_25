"use client";

import { ReactNode, useEffect, useRef, useState } from "react";

// ============================================
// COMPARISON REVEAL - Scroll-triggered transformation
// Old data fades out, new data fades in and gets bolder
// ============================================

interface ComparisonRevealProps {
  oldText: string;
  newText: string;
  className?: string;
  oldClassName?: string;
  newClassName?: string;
  threshold?: number;
  fadeOutDuration?: number;
  fadeInDuration?: number;
  staggerDelay?: number;
}

/**
 * ComparisonReveal - Elegant scroll-triggered comparison animation
 *
 * When the element enters viewport:
 * 1. Old text (in red) fades out smoothly
 * 2. New text (in green) fades in and becomes bolder
 *
 * @example
 * <ComparisonReveal
 *   oldText="3-5 days"
 *   newText="12 seconds"
 * />
 */
export function ComparisonReveal({
  oldText,
  newText,
  className = "",
  oldClassName = "",
  newClassName = "",
  threshold = 0.5,
  fadeOutDuration = 600,
  fadeInDuration = 800,
  staggerDelay = 200,
}: ComparisonRevealProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [animationPhase, setAnimationPhase] = useState<"idle" | "fadeOut" | "fadeIn" | "complete">("idle");

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting && !isVisible) {
            setIsVisible(true);
            // Start animation sequence
            setAnimationPhase("fadeOut");

            // After fade out, start fade in
            setTimeout(() => {
              setAnimationPhase("fadeIn");
            }, fadeOutDuration + staggerDelay);

            // Complete animation
            setTimeout(
              () => {
                setAnimationPhase("complete");
              },
              fadeOutDuration + staggerDelay + fadeInDuration,
            );
          }
        });
      },
      { threshold, rootMargin: "0px 0px -50px 0px" },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [isVisible, threshold, fadeOutDuration, fadeInDuration, staggerDelay]);

  const getOldTextStyles = (): React.CSSProperties => {
    const baseStyles: React.CSSProperties = {
      transition: `all ${fadeOutDuration}ms cubic-bezier(0.4, 0, 0.2, 1)`,
      display: "inline-block",
      position: "relative",
    };

    switch (animationPhase) {
      case "idle":
        return {
          ...baseStyles,
          opacity: 1,
          transform: "translateY(0) scale(1)",
          filter: "blur(0px)",
        };
      case "fadeOut":
      case "fadeIn":
      case "complete":
        return {
          ...baseStyles,
          opacity: 0,
          transform: "translateY(-8px) scale(0.95)",
          filter: "blur(4px)",
        };
    }
  };

  const getNewTextStyles = (): React.CSSProperties => {
    const baseStyles: React.CSSProperties = {
      transition: `all ${fadeInDuration}ms cubic-bezier(0.16, 1, 0.3, 1)`,
      display: "inline-block",
    };

    switch (animationPhase) {
      case "idle":
      case "fadeOut":
        return {
          ...baseStyles,
          opacity: 0,
          transform: "translateY(8px) scale(0.9)",
          fontWeight: 400,
          filter: "blur(4px)",
        };
      case "fadeIn":
        return {
          ...baseStyles,
          opacity: 0.8,
          transform: "translateY(2px) scale(0.98)",
          fontWeight: 600,
          filter: "blur(1px)",
        };
      case "complete":
        return {
          ...baseStyles,
          opacity: 1,
          transform: "translateY(0) scale(1)",
          fontWeight: 700,
          filter: "blur(0px)",
          textShadow: "0 0 30px currentColor",
        };
    }
  };

  return (
    <span ref={ref} className={`relative inline-block ${className}`}>
      {/* Old text - fades out */}
      <span
        className={`text-red-400 ${oldClassName}`}
        style={getOldTextStyles()}
        aria-hidden={animationPhase !== "idle"}
      >
        {oldText}
      </span>

      {/* New text - positioned absolutely to overlay, then takes space */}
      <span
        className={`text-green-400 absolute left-0 top-0 whitespace-nowrap ${newClassName}`}
        style={{
          ...getNewTextStyles(),
          position: animationPhase === "complete" ? "relative" : "absolute",
        }}
      >
        {newText}
      </span>
    </span>
  );
}

// ============================================
// INLINE COMPARISON - For inline text comparisons
// Shows both with transition
// ============================================

interface InlineComparisonProps {
  children: ReactNode;
  threshold?: number;
}

export function InlineComparison({ children, threshold = 0.3 }: InlineComparisonProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            // Animate progress from 0 to 1
            const duration = 1200;
            const startTime = performance.now();

            const animate = (currentTime: number) => {
              const elapsed = currentTime - startTime;
              const newProgress = Math.min(elapsed / duration, 1);

              // Easing function for smooth deceleration
              const eased = 1 - Math.pow(1 - newProgress, 3);
              setProgress(eased);

              if (newProgress < 1) {
                requestAnimationFrame(animate);
              }
            };

            requestAnimationFrame(animate);
          }
        });
      },
      { threshold, rootMargin: "0px 0px -30px 0px" },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [threshold]);

  return (
    <span ref={ref} className="inline-block" style={{ "--comparison-progress": progress } as React.CSSProperties}>
      {children}
    </span>
  );
}

// ============================================
// COMPARISON TEXT - Individual animated text element
// ============================================

interface ComparisonTextProps {
  type: "old" | "new";
  children: ReactNode;
  className?: string;
}

export function ComparisonText({ type, children, className = "" }: ComparisonTextProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [animationComplete, setAnimationComplete] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting && !isVisible) {
            setIsVisible(true);

            // Mark animation as complete after duration
            setTimeout(
              () => {
                setAnimationComplete(true);
              },
              type === "old" ? 600 : 1200,
            );
          }
        });
      },
      { threshold: 0.5 },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [isVisible, type]);

  if (type === "old") {
    return (
      <span
        ref={ref}
        className={`inline-block transition-all duration-500 ease-out ${className}`}
        style={{
          opacity: isVisible ? 0 : 1,
          transform: isVisible ? "translateY(-4px) scale(0.95)" : "translateY(0) scale(1)",
          filter: isVisible ? "blur(3px)" : "blur(0)",
          color: "#f87171", // red-400
          textDecoration: isVisible ? "none" : "line-through",
        }}
      >
        {children}
      </span>
    );
  }

  return (
    <span
      ref={ref}
      className={`inline-block transition-all duration-700 ease-out ${className}`}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "translateY(0) scale(1)" : "translateY(8px) scale(0.9)",
        filter: isVisible ? "blur(0)" : "blur(4px)",
        fontWeight: animationComplete ? 700 : isVisible ? 600 : 400,
        color: "#4ade80", // green-400
        textShadow: animationComplete ? "0 0 20px rgba(74, 222, 128, 0.5)" : "none",
      }}
    >
      {children}
    </span>
  );
}

// ============================================
// STRIKETHROUGH FADE - Static strikethrough effect
// ============================================

interface StrikethroughFadeProps {
  problem: string;
  solution: string;
  className?: string;
}

export function StrikethroughFade({ problem, solution, className = "" }: StrikethroughFadeProps) {
  return (
    <span className={`relative inline-flex items-center gap-1 ${className}`}>
      {/* Problem text - always visible in red */}
      <span className="relative inline-block">
        <span className="text-red-400">{problem}</span>
      </span>

      {/* Arrow */}
      <span className="text-zinc-500 mx-1">→</span>

      {/* Solution text */}
      <span className="text-green-400 font-bold" style={{ textShadow: "0 0 25px rgba(74, 222, 128, 0.6)" }}>
        {solution}
      </span>
    </span>
  );
}

export default ComparisonReveal;
