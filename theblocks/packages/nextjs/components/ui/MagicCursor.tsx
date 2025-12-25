"use client";

import { useEffect, useRef, useState } from "react";

interface Trail {
  x: number;
  y: number;
  opacity: number;
  scale: number;
  hue: number;
}

/**
 * MagicCursor - AI-era custom cursor with particle trails
 * Creates an immersive, futuristic feel
 */
export function MagicCursor() {
  const cursorRef = useRef<HTMLDivElement>(null);
  const cursorDotRef = useRef<HTMLDivElement>(null);
  const trailsRef = useRef<Trail[]>([]);
  const [isHovering, setIsHovering] = useState(false);
  const [isClicking, setIsClicking] = useState(false);
  const positionRef = useRef({ x: -100, y: -100 });
  const requestRef = useRef<number>(0);
  const previousTimeRef = useRef<number>(0);

  useEffect(() => {
    let cursorX = -100;
    let cursorY = -100;
    let targetX = -100;
    let targetY = -100;
    let hue = 260; // Start with purple

    const handleMouseMove = (e: MouseEvent) => {
      targetX = e.clientX;
      targetY = e.clientY;
      positionRef.current = { x: e.clientX, y: e.clientY };

      // Add trail particle
      trailsRef.current.push({
        x: e.clientX,
        y: e.clientY,
        opacity: 0.8,
        scale: 1,
        hue: hue,
      });

      // Keep trail length manageable
      if (trailsRef.current.length > 20) {
        trailsRef.current.shift();
      }

      // Cycle hue for rainbow effect
      hue = (hue + 0.5) % 360;
    };

    const handleMouseDown = () => setIsClicking(true);
    const handleMouseUp = () => setIsClicking(false);

    const handleHoverStart = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "BUTTON" ||
        target.tagName === "A" ||
        target.closest("button") ||
        target.closest("a") ||
        target.classList.contains("hoverable")
      ) {
        setIsHovering(true);
      }
    };

    const handleHoverEnd = () => setIsHovering(false);

    // Smooth animation loop
    const animate = (time: number) => {
      if (previousTimeRef.current !== undefined) {
        // Smooth cursor follow
        cursorX += (targetX - cursorX) * 0.15;
        cursorY += (targetY - cursorY) * 0.15;

        if (cursorRef.current) {
          cursorRef.current.style.transform = `translate(${cursorX - 20}px, ${cursorY - 20}px)`;
        }

        if (cursorDotRef.current) {
          cursorDotRef.current.style.transform = `translate(${targetX - 4}px, ${targetY - 4}px)`;
        }

        // Update trails
        trailsRef.current = trailsRef.current
          .map(trail => ({
            ...trail,
            opacity: trail.opacity * 0.92,
            scale: trail.scale * 0.95,
          }))
          .filter(trail => trail.opacity > 0.01);
      }

      previousTimeRef.current = time;
      requestRef.current = requestAnimationFrame(animate);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("mouseover", handleHoverStart);
    window.addEventListener("mouseout", handleHoverEnd);
    requestRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("mouseover", handleHoverStart);
      window.removeEventListener("mouseout", handleHoverEnd);
      cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // Hide on mobile/touch devices
  if (typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches) {
    return null;
  }

  return (
    <>
      {/* Trail particles */}
      <div className="fixed inset-0 pointer-events-none z-[9998]">
        {trailsRef.current.map((trail, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 rounded-full"
            style={{
              left: trail.x - 4,
              top: trail.y - 4,
              opacity: trail.opacity,
              transform: `scale(${trail.scale})`,
              background: `hsl(${trail.hue}, 70%, 60%)`,
              boxShadow: `0 0 10px hsl(${trail.hue}, 70%, 60%)`,
            }}
          />
        ))}
      </div>

      {/* Main cursor ring */}
      <div
        ref={cursorRef}
        className={`fixed pointer-events-none z-[9999] w-10 h-10 rounded-full border-2 transition-all duration-200 ${
          isHovering
            ? "border-cyan-400 scale-150 bg-cyan-400/10"
            : isClicking
              ? "border-violet-400 scale-75 bg-violet-400/20"
              : "border-violet-500/50"
        }`}
        style={{
          boxShadow: isHovering
            ? "0 0 30px rgba(34, 211, 238, 0.4), inset 0 0 20px rgba(34, 211, 238, 0.1)"
            : "0 0 20px rgba(139, 92, 246, 0.3)",
          mixBlendMode: "screen",
        }}
      />

      {/* Center dot */}
      <div
        ref={cursorDotRef}
        className={`fixed pointer-events-none z-[9999] w-2 h-2 rounded-full transition-all duration-100 ${
          isHovering ? "bg-cyan-400 scale-150" : isClicking ? "bg-violet-400 scale-200" : "bg-white"
        }`}
        style={{
          boxShadow: "0 0 10px currentColor",
        }}
      />

      {/* Hide default cursor */}
      <style jsx global>{`
        * {
          cursor: none !important;
        }
      `}</style>
    </>
  );
}
