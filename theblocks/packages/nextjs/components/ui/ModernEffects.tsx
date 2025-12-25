"use client";

import { useEffect, useRef, useState } from "react";

interface GlassmorphicCardProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
  hoverScale?: number;
  enableTilt?: boolean;
  borderGradient?: boolean;
}

/**
 * GlassmorphicCard - Modern glass effect card with 3D tilt
 * Responds to mouse movement with perspective transforms
 */
export function GlassmorphicCard({
  children,
  className = "",
  glowColor = "violet",
  hoverScale = 1.02,
  enableTilt = true,
  borderGradient = true,
}: GlassmorphicCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const [glowPosition, setGlowPosition] = useState({ x: 50, y: 50 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current || !enableTilt) return;

    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const tiltX = ((y - centerY) / centerY) * 10;
    const tiltY = ((centerX - x) / centerX) * 10;

    setTilt({ x: tiltX, y: tiltY });
    setGlowPosition({
      x: (x / rect.width) * 100,
      y: (y / rect.height) * 100,
    });
  };

  const handleMouseLeave = () => {
    setTilt({ x: 0, y: 0 });
    setIsHovered(false);
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const glowColors: Record<string, string> = {
    violet: "139, 92, 246",
    cyan: "34, 211, 238",
    green: "34, 197, 94",
    orange: "249, 115, 22",
    pink: "236, 72, 153",
  };

  const rgb = glowColors[glowColor] || glowColors.violet;

  return (
    <div
      ref={cardRef}
      className={`relative group ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      style={{
        transform: `perspective(1000px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg) scale(${isHovered ? hoverScale : 1})`,
        transition: "transform 0.2s ease-out",
        transformStyle: "preserve-3d",
      }}
    >
      {/* Animated border gradient */}
      {borderGradient && (
        <div
          className="absolute -inset-[1px] rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: `linear-gradient(135deg, 
              rgba(${rgb}, 0.5) 0%, 
              rgba(34, 211, 238, 0.5) 50%, 
              rgba(${rgb}, 0.5) 100%)`,
            backgroundSize: "200% 200%",
            animation: "gradient-rotate 3s ease infinite",
          }}
        />
      )}

      {/* Mouse follow glow */}
      <div
        className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{
          background: `radial-gradient(circle at ${glowPosition.x}% ${glowPosition.y}%, rgba(${rgb}, 0.15) 0%, transparent 50%)`,
        }}
      />

      {/* Glass background */}
      <div
        className="relative rounded-2xl overflow-hidden"
        style={{
          background: "rgba(18, 18, 26, 0.8)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          boxShadow: isHovered
            ? `0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 50px rgba(${rgb}, 0.1)`
            : "0 10px 40px -10px rgba(0, 0, 0, 0.3)",
          transition: "box-shadow 0.3s ease-out",
        }}
      >
        {/* Content with 3D depth */}
        <div style={{ transform: "translateZ(20px)" }}>{children}</div>
      </div>

      {/* Reflection/shine effect */}
      <div
        className="absolute inset-0 rounded-2xl pointer-events-none overflow-hidden"
        style={{
          background: `linear-gradient(105deg, 
            rgba(255, 255, 255, 0) 40%, 
            rgba(255, 255, 255, 0.03) 45%, 
            rgba(255, 255, 255, 0) 50%)`,
          transform: `translateX(${isHovered ? "100%" : "-100%"})`,
          transition: "transform 0.6s ease-out",
        }}
      />
    </div>
  );
}

/**
 * FloatingElement - Adds floating animation to any element
 */
export function FloatingElement({
  children,
  delay = 0,
  duration = 6,
  distance = 20,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  distance?: number;
  className?: string;
}) {
  return (
    <div
      className={className}
      style={{
        animation: `float-smooth ${duration}s ease-in-out infinite`,
        animationDelay: `${delay}s`,
      }}
    >
      {children}
      <style jsx>{`
        @keyframes float-smooth {
          0%,
          100% {
            transform: translateY(0) rotate(0deg);
          }
          25% {
            transform: translateY(-${distance * 0.5}px) rotate(1deg);
          }
          50% {
            transform: translateY(-${distance}px) rotate(0deg);
          }
          75% {
            transform: translateY(-${distance * 0.5}px) rotate(-1deg);
          }
        }
      `}</style>
    </div>
  );
}

/**
 * MorphingBlob - Animated background blob with morphing shape
 */
export function MorphingBlob({
  color1 = "#8b5cf6",
  color2 = "#06b6d4",
  size = 400,
  className = "",
}: {
  color1?: string;
  color2?: string;
  size?: number;
  className?: string;
}) {
  return (
    <div
      className={`absolute blur-[100px] opacity-30 ${className}`}
      style={{
        width: size,
        height: size,
        background: `linear-gradient(135deg, ${color1}, ${color2})`,
        borderRadius: "30% 70% 70% 30% / 30% 30% 70% 70%",
        animation: "morph 15s ease-in-out infinite",
      }}
    >
      <style jsx>{`
        @keyframes morph {
          0%,
          100% {
            border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
            transform: rotate(0deg) scale(1);
          }
          25% {
            border-radius: 58% 42% 75% 25% / 76% 46% 54% 24%;
          }
          50% {
            border-radius: 50% 50% 33% 67% / 55% 27% 73% 45%;
            transform: rotate(180deg) scale(1.1);
          }
          75% {
            border-radius: 33% 67% 58% 42% / 63% 68% 32% 37%;
          }
        }
      `}</style>
    </div>
  );
}

/**
 * AnimatedCounter - Smooth number animation
 */
export function AnimatedCounter({
  value,
  prefix = "",
  suffix = "",
  duration = 2000,
  decimals = 0,
  className = "",
}: {
  value: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  decimals?: number;
  className?: string;
}) {
  const [displayValue, setDisplayValue] = useState(0);
  const [hasAnimated, setHasAnimated] = useState(false);
  const elementRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting && !hasAnimated) {
          setHasAnimated(true);
          const startTime = Date.now();
          const startValue = 0;

          const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function - ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startValue + (value - startValue) * eased;

            setDisplayValue(current);

            if (progress < 1) {
              requestAnimationFrame(animate);
            }
          };

          requestAnimationFrame(animate);
        }
      },
      { threshold: 0.5 },
    );

    if (elementRef.current) {
      observer.observe(elementRef.current);
    }

    return () => observer.disconnect();
  }, [value, duration, hasAnimated]);

  return (
    <span ref={elementRef} className={className}>
      {prefix}
      {displayValue.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
      {suffix}
    </span>
  );
}

/**
 * PulseRing - Animated pulse effect for status indicators
 */
export function PulseRing({
  color = "green",
  size = 12,
  className = "",
}: {
  color?: "green" | "yellow" | "red" | "blue" | "violet";
  size?: number;
  className?: string;
}) {
  const colors = {
    green: { bg: "bg-green-500", ring: "bg-green-400" },
    yellow: { bg: "bg-yellow-500", ring: "bg-yellow-400" },
    red: { bg: "bg-red-500", ring: "bg-red-400" },
    blue: { bg: "bg-blue-500", ring: "bg-blue-400" },
    violet: { bg: "bg-violet-500", ring: "bg-violet-400" },
  };

  return (
    <span className={`relative inline-flex ${className}`} style={{ width: size, height: size }}>
      <span
        className={`animate-ping absolute inline-flex h-full w-full rounded-full ${colors[color].ring} opacity-75`}
      />
      <span className={`relative inline-flex rounded-full h-full w-full ${colors[color].bg}`} />
    </span>
  );
}

/**
 * TypewriterText - Typewriter effect for text
 */
export function TypewriterText({
  text,
  speed = 50,
  delay = 0,
  className = "",
  showCursor = true,
  onComplete,
}: {
  text: string;
  speed?: number;
  delay?: number;
  className?: string;
  showCursor?: boolean;
  onComplete?: () => void;
}) {
  const [displayText, setDisplayText] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    let timeout: NodeJS.Timeout;
    let charIndex = 0;

    const startTyping = () => {
      if (charIndex < text.length) {
        setDisplayText(text.slice(0, charIndex + 1));
        charIndex++;
        timeout = setTimeout(startTyping, speed);
      } else {
        setIsComplete(true);
        onComplete?.();
      }
    };

    timeout = setTimeout(startTyping, delay);

    return () => clearTimeout(timeout);
  }, [text, speed, delay, onComplete]);

  return (
    <span className={className}>
      {displayText}
      {showCursor && !isComplete && <span className="animate-pulse">|</span>}
    </span>
  );
}
