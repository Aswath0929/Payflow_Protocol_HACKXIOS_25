"use client";

import { useEffect, useRef, useState } from "react";

/**
 * MagicCursor - Ultra-responsive custom cursor
 * Optimized for zero lag with direct DOM manipulation
 * No React state updates in animation loop = maximum performance
 */
export function MagicCursor() {
  const cursorRingRef = useRef<HTMLDivElement>(null);
  const cursorDotRef = useRef<HTMLDivElement>(null);
  const isVisibleRef = useRef(false);
  const [isMounted, setIsMounted] = useState(false);

  // Prevent hydration mismatch by only rendering on client
  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    // Check for touch device
    if (window.matchMedia("(pointer: coarse)").matches) {
      return;
    }

    if (!isMounted) return;

    const cursorRing = cursorRingRef.current;
    const cursorDot = cursorDotRef.current;
    if (!cursorRing || !cursorDot) return;

    // State variables (no React state for max performance)
    let mouseX = -100;
    let mouseY = -100;
    let ringX = -100;
    let ringY = -100;
    let isClicking = false;
    let animationId: number;

    // Direct mouse tracking - no lag
    const onMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      if (!isVisibleRef.current) {
        isVisibleRef.current = true;
        cursorRing.style.opacity = "1";
        cursorDot.style.opacity = "1";
      }

      // Dot follows instantly
      cursorDot.style.left = `${mouseX}px`;
      cursorDot.style.top = `${mouseY}px`;
    };

    const onMouseDown = () => {
      isClicking = true;
      cursorRing.classList.add("clicking");
      cursorDot.classList.add("clicking");
    };

    const onMouseUp = () => {
      isClicking = false;
      cursorRing.classList.remove("clicking");
      cursorDot.classList.remove("clicking");
    };

    const onMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "BUTTON" ||
        target.tagName === "A" ||
        target.closest("button") ||
        target.closest("a") ||
        target.classList.contains("hoverable")
      ) {
        cursorRing.classList.add("hovering");
        cursorDot.classList.add("hovering");
      }
    };

    const onMouseOut = () => {
      cursorRing.classList.remove("hovering");
      cursorDot.classList.remove("hovering");
    };

    const onMouseLeave = () => {
      isVisibleRef.current = false;
      cursorRing.style.opacity = "0";
      cursorDot.style.opacity = "0";
    };

    // Smooth ring animation - uses lerp for gentle follow
    const animate = () => {
      // Lerp factor - higher = faster follow (0.25 is snappy but smooth)
      const lerp = 0.25;

      ringX += (mouseX - ringX) * lerp;
      ringY += (mouseY - ringY) * lerp;

      cursorRing.style.left = `${ringX}px`;
      cursorRing.style.top = `${ringY}px`;

      animationId = requestAnimationFrame(animate);
    };

    // Event listeners
    document.addEventListener("mousemove", onMouseMove, { passive: true });
    document.addEventListener("mousedown", onMouseDown, { passive: true });
    document.addEventListener("mouseup", onMouseUp, { passive: true });
    document.addEventListener("mouseover", onMouseOver, { passive: true });
    document.addEventListener("mouseout", onMouseOut, { passive: true });
    document.documentElement.addEventListener("mouseleave", onMouseLeave, { passive: true });

    // Start animation
    animationId = requestAnimationFrame(animate);

    return () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mouseout", onMouseOut);
      document.documentElement.removeEventListener("mouseleave", onMouseLeave);
      cancelAnimationFrame(animationId);
    };
  }, [isMounted]);

  // Don't render on server or before hydration
  if (!isMounted) {
    return null;
  }

  return (
    <>
      {/* Cursor Ring - follows with slight delay */}
      <div
        ref={cursorRingRef}
        className="magic-cursor-ring"
        style={{
          position: "fixed",
          width: "40px",
          height: "40px",
          borderRadius: "50%",
          border: "2px solid rgba(139, 92, 246, 0.6)",
          pointerEvents: "none",
          zIndex: 99999,
          transform: "translate(-50%, -50%)",
          opacity: 0,
          transition: "width 0.15s, height 0.15s, border-color 0.15s, background-color 0.15s",
          boxShadow: "0 0 20px rgba(139, 92, 246, 0.3)",
          mixBlendMode: "screen",
          willChange: "left, top",
        }}
      />

      {/* Cursor Dot - instant follow */}
      <div
        ref={cursorDotRef}
        className="magic-cursor-dot"
        style={{
          position: "fixed",
          width: "8px",
          height: "8px",
          borderRadius: "50%",
          backgroundColor: "white",
          pointerEvents: "none",
          zIndex: 99999,
          transform: "translate(-50%, -50%)",
          opacity: 0,
          transition: "width 0.1s, height 0.1s, background-color 0.1s",
          boxShadow: "0 0 10px rgba(255, 255, 255, 0.8)",
          willChange: "left, top",
        }}
      />

      {/* Cursor styles */}
      <style jsx global>{`
        * {
          cursor: none !important;
        }

        .magic-cursor-ring.hovering {
          width: 60px;
          height: 60px;
          border-color: rgba(34, 211, 238, 0.8);
          background-color: rgba(34, 211, 238, 0.1);
          box-shadow: 0 0 30px rgba(34, 211, 238, 0.4);
        }

        .magic-cursor-ring.clicking {
          width: 30px;
          height: 30px;
          border-color: rgba(167, 139, 250, 0.9);
          background-color: rgba(167, 139, 250, 0.2);
        }

        .magic-cursor-dot.hovering {
          width: 12px;
          height: 12px;
          background-color: rgb(34, 211, 238);
        }

        .magic-cursor-dot.clicking {
          width: 6px;
          height: 6px;
          background-color: rgb(167, 139, 250);
        }
      `}</style>
    </>
  );
}
