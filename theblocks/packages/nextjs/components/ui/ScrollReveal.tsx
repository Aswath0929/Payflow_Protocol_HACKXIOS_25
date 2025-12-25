"use client";

import { ReactNode, createContext, useContext, useEffect, useRef, useState } from "react";

// ============================================
// ANIMATION PRESETS - Easy-to-use configurations
// ============================================

export const ANIMATION_PRESETS = {
  // Hero sections - dramatic entrance
  hero: { direction: "up" as const, duration: 800, delay: 0 },
  // Cards - subtle slide up
  card: { direction: "up" as const, duration: 600, delay: 0 },
  // Side content - slide from side
  slideLeft: { direction: "left" as const, duration: 600, delay: 0 },
  slideRight: { direction: "right" as const, duration: 600, delay: 0 },
  // Stats/metrics - quick fade
  stat: { direction: "fade" as const, duration: 500, delay: 0 },
  // Lists - staggered items
  listItem: { direction: "up" as const, duration: 400, delay: 50 },
  // Page sections
  section: { direction: "up" as const, duration: 700, delay: 100 },
};

// ============================================
// SCROLL REVEAL CONTEXT - For nested animations
// ============================================

interface ScrollRevealContextType {
  isParentVisible: boolean;
}

const ScrollRevealContext = createContext<ScrollRevealContextType>({ isParentVisible: true });

export function useParentReveal() {
  return useContext(ScrollRevealContext);
}

// ============================================
// SCROLL REVEAL - Core component
// ============================================

interface ScrollRevealProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  direction?: "up" | "down" | "left" | "right" | "fade" | "scale" | "blur";
  duration?: number;
  threshold?: number;
  once?: boolean;
  preset?: keyof typeof ANIMATION_PRESETS;
  disabled?: boolean;
}

/**
 * ScrollReveal - Cursor-style scroll-triggered animations
 * Reveals content when it enters the viewport
 *
 * @example
 * // Basic usage
 * <ScrollReveal>Content here</ScrollReveal>
 *
 * // With preset
 * <ScrollReveal preset="hero">Hero content</ScrollReveal>
 *
 * // With custom options
 * <ScrollReveal direction="left" delay={200}>Slide from left</ScrollReveal>
 */
export function ScrollReveal({
  children,
  className = "",
  delay,
  direction,
  duration,
  threshold = 0.1,
  once = true,
  preset,
  disabled = false,
}: ScrollRevealProps) {
  const [isVisible, setIsVisible] = useState(disabled);
  const ref = useRef<HTMLDivElement>(null);

  // Apply preset if provided
  const presetConfig = preset ? ANIMATION_PRESETS[preset] : null;
  const finalDirection = direction ?? presetConfig?.direction ?? "up";
  const finalDuration = duration ?? presetConfig?.duration ?? 700;
  const finalDelay = delay ?? presetConfig?.delay ?? 0;

  useEffect(() => {
    if (disabled) {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (once && ref.current) {
            observer.unobserve(ref.current);
          }
        } else if (!once) {
          setIsVisible(false);
        }
      },
      {
        threshold,
        rootMargin: "0px 0px -50px 0px",
      },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [threshold, once, disabled]);

  const getTransform = () => {
    switch (finalDirection) {
      case "up":
        return "translateY(40px)";
      case "down":
        return "translateY(-40px)";
      case "left":
        return "translateX(40px)";
      case "right":
        return "translateX(-40px)";
      case "scale":
        return "scale(0.95)";
      case "blur":
      case "fade":
        return "translateY(0)";
      default:
        return "translateY(40px)";
    }
  };

  const getFilter = () => {
    if (finalDirection === "blur") {
      return isVisible ? "blur(0px)" : "blur(10px)";
    }
    return undefined;
  };

  return (
    <ScrollRevealContext.Provider value={{ isParentVisible: isVisible }}>
      <div
        ref={ref}
        className={className}
        style={{
          opacity: isVisible ? 1 : 0,
          transform: isVisible ? "translateY(0) translateX(0) scale(1)" : getTransform(),
          filter: getFilter(),
          transition: `opacity ${finalDuration}ms ease-out, transform ${finalDuration}ms ease-out, filter ${finalDuration}ms ease-out`,
          transitionDelay: `${finalDelay}ms`,
        }}
      >
        {children}
      </div>
    </ScrollRevealContext.Provider>
  );
}

// ============================================
// PAGE TRANSITION - For page-level animations
// ============================================

interface PageTransitionProps {
  children: ReactNode;
  className?: string;
}

/**
 * PageTransition - Wraps entire page content for smooth entrance
 * Use at the top level of each page component
 *
 * @example
 * export default function MyPage() {
 *   return (
 *     <PageTransition>
 *       <div>Page content here</div>
 *     </PageTransition>
 *   );
 * }
 */
export function PageTransition({ children, className = "" }: PageTransitionProps) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Small delay to ensure DOM is ready
    const timer = setTimeout(() => setIsLoaded(true), 50);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className={className}
      style={{
        opacity: isLoaded ? 1 : 0,
        transform: isLoaded ? "translateY(0)" : "translateY(20px)",
        transition: "opacity 600ms ease-out, transform 600ms ease-out",
      }}
    >
      {children}
    </div>
  );
}

// ============================================
// SECTION REVEAL - For major page sections
// ============================================

interface SectionRevealProps {
  children: ReactNode;
  className?: string;
  title?: string;
  description?: string;
  delay?: number;
}

/**
 * SectionReveal - Reveals a major page section with optional title
 * Includes built-in spacing and styling
 *
 * @example
 * <SectionReveal title="Features" description="What we offer">
 *   <FeatureGrid />
 * </SectionReveal>
 */
export function SectionReveal({ children, className = "", title, description, delay = 0 }: SectionRevealProps) {
  return (
    <ScrollReveal preset="section" delay={delay} className={className}>
      {(title || description) && (
        <div className="mb-8">
          {title && <h2 className="text-2xl font-bold text-white mb-2">{title}</h2>}
          {description && <p className="text-zinc-400">{description}</p>}
        </div>
      )}
      {children}
    </ScrollReveal>
  );
}

// ============================================
// ANIMATED CARD - For card-based content
// ============================================

interface AnimatedCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  hoverEffect?: boolean;
}

/**
 * AnimatedCard - A card with scroll reveal and optional hover effects
 *
 * @example
 * <AnimatedCard hoverEffect>
 *   <h3>Card Title</h3>
 *   <p>Card content</p>
 * </AnimatedCard>
 */
export function AnimatedCard({ children, className = "", delay = 0, hoverEffect = true }: AnimatedCardProps) {
  return (
    <ScrollReveal preset="card" delay={delay}>
      <div
        className={`
          bg-white/[0.02] border border-white/5 rounded-xl p-6
          ${hoverEffect ? "hover:border-white/10 hover:-translate-y-1 transition-all duration-300" : ""}
          ${className}
        `}
      >
        {children}
      </div>
    </ScrollReveal>
  );
}

// ============================================
// STAGGER CONTAINER - For staggered child animations
// ============================================

interface StaggerContainerProps {
  children: ReactNode;
  className?: string;
  staggerDelay?: number;
  direction?: "up" | "down" | "left" | "right" | "fade";
}

/**
 * StaggerContainer - Animates children with staggered delays
 * Great for lists, grids, and navigation items
 *
 * @example
 * <StaggerContainer staggerDelay={100}>
 *   {items.map(item => <div key={item.id}>{item.name}</div>)}
 * </StaggerContainer>
 */
export function StaggerContainer({
  children,
  className = "",
  staggerDelay = 100,
  direction = "up",
}: StaggerContainerProps) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const childArray = Array.isArray(children) ? children : [children];

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (ref.current) {
            observer.unobserve(ref.current);
          }
        }
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, []);

  const getTransform = () => {
    switch (direction) {
      case "up":
        return "translateY(30px)";
      case "down":
        return "translateY(-30px)";
      case "left":
        return "translateX(30px)";
      case "right":
        return "translateX(-30px)";
      case "fade":
        return "translateY(0)";
      default:
        return "translateY(30px)";
    }
  };

  return (
    <div ref={ref} className={className}>
      {childArray.map((child, index) => (
        <div
          key={index}
          style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? "translateY(0) translateX(0)" : getTransform(),
            transition: "opacity 500ms ease-out, transform 500ms ease-out",
            transitionDelay: `${index * staggerDelay}ms`,
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}

/**
 * useScrollReveal - Hook for custom scroll reveal logic
 * Returns a ref and visibility state for custom animations
 *
 * @example
 * function MyComponent() {
 *   const { ref, isVisible } = useScrollReveal();
 *   return (
 *     <div ref={ref} style={{ opacity: isVisible ? 1 : 0 }}>
 *       Custom animated content
 *     </div>
 *   );
 * }
 */
export function useScrollReveal(threshold = 0.1) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (ref.current) {
            observer.unobserve(ref.current);
          }
        }
      },
      {
        threshold,
        rootMargin: "0px 0px -50px 0px",
      },
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [threshold]);

  return { ref, isVisible };
}

// Keep ScrollRevealGroup as an alias for backward compatibility
export const ScrollRevealGroup = StaggerContainer;
