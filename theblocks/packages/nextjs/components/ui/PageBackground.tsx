"use client";

import { useEffect, useRef } from "react";

type BackgroundTheme = "dashboard" | "oracle" | "explorer" | "debug" | "minimal";

interface PageBackgroundProps {
  theme?: BackgroundTheme;
  intensity?: "low" | "medium" | "high";
}

/**
 * PageBackground - Creative animated backgrounds for different pages
 * Each theme has unique visual identity while maintaining brand consistency
 */
export function PageBackground({ theme = "minimal", intensity = "medium" }: PageBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Canvas-based animations for dashboard and oracle themes
  useEffect(() => {
    if (theme !== "dashboard" && theme !== "oracle") return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let width = window.innerWidth;
    let height = window.innerHeight;

    // Set canvas size
    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };
    resize();
    window.addEventListener("resize", resize);

    // Particle system for dashboard theme
    if (theme === "dashboard") {
      interface DataParticle {
        x: number;
        y: number;
        vx: number;
        vy: number;
        size: number;
        alpha: number;
        color: string;
        type: "data" | "flow" | "pulse";
      }

      const particles: DataParticle[] = [];
      const particleCount = intensity === "high" ? 80 : intensity === "medium" ? 50 : 30;
      const colors = ["#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"];

      // Initialize particles
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5 + 0.3, // Slight downward bias for "data flow" feel
          size: Math.random() * 3 + 1,
          alpha: Math.random() * 0.5 + 0.2,
          color: colors[Math.floor(Math.random() * colors.length)],
          type: Math.random() > 0.7 ? "pulse" : Math.random() > 0.5 ? "flow" : "data",
        });
      }

      // Data stream lines
      interface DataStream {
        x: number;
        y: number;
        length: number;
        speed: number;
        alpha: number;
      }
      const streams: DataStream[] = [];
      for (let i = 0; i < 15; i++) {
        streams.push({
          x: Math.random() * width,
          y: Math.random() * height,
          length: Math.random() * 100 + 50,
          speed: Math.random() * 2 + 1,
          alpha: Math.random() * 0.3 + 0.1,
        });
      }

      const animate = () => {
        ctx.fillStyle = "rgba(10, 10, 15, 0.1)";
        ctx.fillRect(0, 0, width, height);

        // Draw data streams
        streams.forEach(stream => {
          const gradient = ctx.createLinearGradient(stream.x, stream.y, stream.x, stream.y + stream.length);
          gradient.addColorStop(0, `rgba(139, 92, 246, 0)`);
          gradient.addColorStop(0.5, `rgba(139, 92, 246, ${stream.alpha})`);
          gradient.addColorStop(1, `rgba(139, 92, 246, 0)`);

          ctx.beginPath();
          ctx.strokeStyle = gradient;
          ctx.lineWidth = 1;
          ctx.moveTo(stream.x, stream.y);
          ctx.lineTo(stream.x, stream.y + stream.length);
          ctx.stroke();

          stream.y += stream.speed;
          if (stream.y > height) {
            stream.y = -stream.length;
            stream.x = Math.random() * width;
          }
        });

        // Draw and update particles
        particles.forEach(p => {
          ctx.beginPath();

          if (p.type === "pulse") {
            // Pulsing circle - ensure radius is always positive
            const pulseSize = Math.max(0.5, p.size + Math.sin(Date.now() * 0.005 + p.x) * 2);
            ctx.arc(p.x, p.y, pulseSize, 0, Math.PI * 2);
            ctx.fillStyle = p.color.replace(")", `, ${p.alpha})`).replace("rgb", "rgba");
            ctx.shadowBlur = 10;
            ctx.shadowColor = p.color;
          } else if (p.type === "flow") {
            // Data flow line
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 10, p.y + p.vy * 10);
            ctx.strokeStyle = p.color.replace(")", `, ${p.alpha})`).replace("rgb", "rgba");
            ctx.lineWidth = p.size / 2;
            ctx.stroke();
          } else {
            // Standard data particle
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = p.color.replace(")", `, ${p.alpha})`).replace("rgb", "rgba");
          }

          ctx.fill();
          ctx.shadowBlur = 0;

          // Update position
          p.x += p.vx;
          p.y += p.vy;

          // Wrap around screen
          if (p.x < 0) p.x = width;
          if (p.x > width) p.x = 0;
          if (p.y < 0) p.y = height;
          if (p.y > height) p.y = 0;
        });

        animationId = requestAnimationFrame(animate);
      };

      animate();
    }

    // Dynamic graph network animation for oracle theme with traversing flash balls
    if (theme === "oracle") {
      interface GraphNode {
        x: number;
        y: number;
        radius: number;
        connections: number[];
        brightness: number;
      }

      interface FlashBall {
        fromNode: number;
        toNode: number;
        progress: number;
        speed: number;
        color: string;
        size: number;
        trail: { x: number; y: number; alpha: number }[];
      }

      const nodes: GraphNode[] = [];
      const flashBalls: FlashBall[] = [];
      const gridSpacing = intensity === "high" ? 50 : intensity === "medium" ? 65 : 80;
      const rows = Math.ceil(height / gridSpacing) + 2;
      const cols = Math.ceil(width / gridSpacing) + 2;
      const colors = ["#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899"];

      // Create a proper tight grid mesh (no random jitter for taut mesh)
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          nodes.push({
            x: col * gridSpacing,
            y: row * gridSpacing,
            radius: 1.5,
            connections: [],
            brightness: 0.12 + Math.random() * 0.08,
          });
        }
      }

      // Build proper grid connections (horizontal, vertical, and diagonal for mesh look)
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const idx = row * cols + col;
          // Connect to right neighbor
          if (col < cols - 1) {
            const rightIdx = idx + 1;
            nodes[idx].connections.push(rightIdx);
            nodes[rightIdx].connections.push(idx);
          }
          // Connect to bottom neighbor
          if (row < rows - 1) {
            const bottomIdx = idx + cols;
            nodes[idx].connections.push(bottomIdx);
            nodes[bottomIdx].connections.push(idx);
          }
          // Connect to bottom-right diagonal (creates triangular mesh)
          if (col < cols - 1 && row < rows - 1) {
            const diagIdx = idx + cols + 1;
            nodes[idx].connections.push(diagIdx);
            nodes[diagIdx].connections.push(idx);
          }
          // Connect to bottom-left diagonal (completes the mesh triangles)
          if (col > 0 && row < rows - 1) {
            const diagLeftIdx = idx + cols - 1;
            nodes[idx].connections.push(diagLeftIdx);
            nodes[diagLeftIdx].connections.push(idx);
          }
        }
      }

      // Spawn new flash balls periodically
      const spawnFlashBall = () => {
        if (flashBalls.length < 15) {
          const startNode = Math.floor(Math.random() * nodes.length);
          if (nodes[startNode].connections.length > 0) {
            const targetIdx = Math.floor(Math.random() * nodes[startNode].connections.length);
            const endNode = nodes[startNode].connections[targetIdx];
            flashBalls.push({
              fromNode: startNode,
              toNode: endNode,
              progress: 0,
              speed: 0.015 + Math.random() * 0.015,
              color: colors[Math.floor(Math.random() * colors.length)],
              size: 3 + Math.random() * 3,
              trail: [],
            });
          }
        }
      };

      // Spawn flash balls at intervals
      const spawnInterval = setInterval(spawnFlashBall, 150);

      const animate = () => {
        // Subtle fade for trails
        ctx.fillStyle = "rgba(10, 10, 15, 0.08)";
        ctx.fillRect(0, 0, width, height);

        // Draw mesh connections (very subtle)
        ctx.strokeStyle = "rgba(100, 100, 140, 0.08)";
        ctx.lineWidth = 0.5;
        nodes.forEach((node, i) => {
          node.connections.forEach(j => {
            if (j > i) {
              ctx.beginPath();
              ctx.moveTo(node.x, node.y);
              ctx.lineTo(nodes[j].x, nodes[j].y);
              ctx.stroke();
            }
          });
        });

        // Draw nodes (very subtle)
        nodes.forEach(node => {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(139, 92, 246, ${node.brightness})`;
          ctx.fill();
        });

        // Update and draw flash balls
        for (let i = flashBalls.length - 1; i >= 0; i--) {
          const ball = flashBalls[i];
          const fromNode = nodes[ball.fromNode];
          const toNode = nodes[ball.toNode];

          // Calculate current position
          const x = fromNode.x + (toNode.x - fromNode.x) * ball.progress;
          const y = fromNode.y + (toNode.y - fromNode.y) * ball.progress;

          // Add to trail
          ball.trail.push({ x, y, alpha: 1 });
          if (ball.trail.length > 12) ball.trail.shift();

          // Draw trail
          ball.trail.forEach((point, idx) => {
            const trailAlpha = (idx / ball.trail.length) * 0.6;
            const trailSize = (idx / ball.trail.length) * ball.size * 0.8;
            ctx.beginPath();
            ctx.arc(point.x, point.y, Math.max(0.5, trailSize), 0, Math.PI * 2);
            ctx.fillStyle = ball.color.replace(")", `, ${trailAlpha})`).replace("#", "rgba(").replace(/([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i, (_, r, g, b) => `${parseInt(r, 16)}, ${parseInt(g, 16)}, ${parseInt(b, 16)}`);
            ctx.fill();
          });

          // Draw main flash ball with glow
          ctx.beginPath();
          ctx.arc(x, y, ball.size, 0, Math.PI * 2);
          ctx.fillStyle = ball.color;
          ctx.shadowBlur = 15;
          ctx.shadowColor = ball.color;
          ctx.fill();
          ctx.shadowBlur = 0;

          // Brighten nodes near the ball
          [ball.fromNode, ball.toNode].forEach(nodeIdx => {
            nodes[nodeIdx].brightness = Math.min(0.8, nodes[nodeIdx].brightness + 0.02);
          });

          // Update progress
          ball.progress += ball.speed;

          // When ball reaches destination, continue to next node or remove
          if (ball.progress >= 1) {
            const arrivedNode = nodes[ball.toNode];
            if (arrivedNode.connections.length > 0 && Math.random() > 0.3) {
              // Continue to next node
              const nextConnections = arrivedNode.connections.filter(n => n !== ball.fromNode);
              if (nextConnections.length > 0) {
                ball.fromNode = ball.toNode;
                ball.toNode = nextConnections[Math.floor(Math.random() * nextConnections.length)];
                ball.progress = 0;
                ball.trail = [];
              } else {
                flashBalls.splice(i, 1);
              }
            } else {
              flashBalls.splice(i, 1);
            }
          }
        }

        // Fade node brightness back to normal
        nodes.forEach(node => {
          node.brightness = Math.max(0.15, node.brightness - 0.005);
        });

        animationId = requestAnimationFrame(animate);
      };

      animate();

      // Cleanup interval on unmount
      return () => {
        window.removeEventListener("resize", resize);
        cancelAnimationFrame(animationId);
        clearInterval(spawnInterval);
      };
    }

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationId);
    };
  }, [theme, intensity]);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
      {/* Base gradient for all themes */}
      <div className="absolute inset-0 bg-[#0a0a0f]" />

      {/* Canvas for dashboard/oracle themes */}
      {(theme === "dashboard" || theme === "oracle") && (
        <canvas ref={canvasRef} className="absolute inset-0 opacity-80" />
      )}

      {/* Dashboard theme: Data flow aesthetic */}
      {theme === "dashboard" && (
        <>
          {/* Top gradient accent */}
          <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-violet-500/10 via-transparent to-transparent" />

          {/* Side accents */}
          <div className="absolute top-1/4 -left-48 w-96 h-96 bg-violet-500/20 rounded-full blur-[100px] animate-pulse" />
          <div
            className="absolute bottom-1/4 -right-48 w-96 h-96 bg-cyan-500/20 rounded-full blur-[100px] animate-pulse"
            style={{ animationDelay: "1.5s" }}
          />

          {/* Floating data grid */}
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `
                linear-gradient(rgba(139, 92, 246, 0.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139, 92, 246, 0.5) 1px, transparent 1px)
              `,
              backgroundSize: "60px 60px",
              animation: "grid-scroll 20s linear infinite",
            }}
          />
        </>
      )}

      {/* Oracle theme: Hexagonal network feel */}
      {theme === "oracle" && (
        <>
          {/* Cyan accent glow */}
          <div className="absolute top-0 left-1/3 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[150px]" />
          <div className="absolute bottom-0 right-1/3 w-[600px] h-[600px] bg-violet-500/10 rounded-full blur-[150px]" />

          {/* Security shield overlay */}
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] opacity-10"
            style={{
              background: "conic-gradient(from 0deg, transparent, rgba(6, 182, 212, 0.3), transparent)",
              animation: "spin 10s linear infinite",
            }}
          />
        </>
      )}

      {/* Explorer theme: Block chain aesthetic */}
      {theme === "explorer" && (
        <>
          <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a0f] via-[#0f0a15] to-[#0a0f15]" />

          {/* Floating blocks */}
          <div className="absolute inset-0">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute bg-gradient-to-br from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-lg"
                style={{
                  width: `${20 + Math.random() * 40}px`,
                  height: `${20 + Math.random() * 40}px`,
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animation: `float ${10 + Math.random() * 10}s ease-in-out infinite`,
                  animationDelay: `${Math.random() * 5}s`,
                }}
              />
            ))}
          </div>

          {/* Chain links */}
          <div
            className="absolute left-1/2 top-0 bottom-0 w-px opacity-20"
            style={{
              background: "repeating-linear-gradient(to bottom, transparent, transparent 40px, #f59e0b 40px, #f59e0b 80px)",
              animation: "chain-flow 4s linear infinite",
            }}
          />

          {/* Glowing accents */}
          <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-amber-500/10 rounded-full blur-[80px] animate-pulse" />
          <div
            className="absolute bottom-1/4 right-1/4 w-72 h-72 bg-orange-500/10 rounded-full blur-[80px] animate-pulse"
            style={{ animationDelay: "2s" }}
          />
        </>
      )}

      {/* Debug theme: Terminal/code aesthetic */}
      {theme === "debug" && (
        <>
          <div className="absolute inset-0 bg-[#0a0f0a]" />

          {/* Matrix rain effect - CSS only */}
          <div className="absolute inset-0 overflow-hidden opacity-30">
            {[...Array(30)].map((_, i) => (
              <div
                key={i}
                className="absolute text-green-400/80 font-mono text-xs"
                style={{
                  left: `${i * 3.5}%`,
                  animation: `matrix-fall ${5 + Math.random() * 10}s linear infinite`,
                  animationDelay: `${Math.random() * 5}s`,
                }}
              >
                {[...Array(20)].map((_, j) => (
                  <div key={j} className="opacity-[0.7]" style={{ opacity: 1 - j * 0.05 }}>
                    {String.fromCharCode(0x30a0 + Math.random() * 96)}
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* Terminal glow */}
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-green-500/5 rounded-lg blur-[60px]" />

          {/* Scan line */}
          <div
            className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-400/30 to-transparent"
            style={{
              animation: "scan-line 3s ease-in-out infinite",
            }}
          />
        </>
      )}

      {/* Minimal theme: Subtle but elegant */}
      {theme === "minimal" && (
        <>
          <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a0f] to-[#0f0a18]" />
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-violet-500/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-cyan-500/5 rounded-full blur-[120px]" />
        </>
      )}

      {/* Vignette for all themes */}
      <div
        className="absolute inset-0"
        style={{
          background: "radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.4) 100%)",
        }}
      />

      {/* Additional animations keyframes */}
      <style jsx>{`
        @keyframes grid-scroll {
          0% {
            transform: translateY(0);
          }
          100% {
            transform: translateY(60px);
          }
        }

        @keyframes chain-flow {
          0% {
            transform: translateY(0);
          }
          100% {
            transform: translateY(80px);
          }
        }

        @keyframes matrix-fall {
          0% {
            transform: translateY(-100%);
          }
          100% {
            transform: translateY(100vh);
          }
        }

        @keyframes scan-line {
          0%,
          100% {
            top: 0;
            opacity: 0;
          }
          50% {
            opacity: 1;
          }
          100% {
            top: 100%;
          }
        }

        @keyframes float {
          0%,
          100% {
            transform: translateY(0) rotate(0deg);
          }
          50% {
            transform: translateY(-20px) rotate(5deg);
          }
        }

        @keyframes spin {
          from {
            transform: translate(-50%, -50%) rotate(0deg);
          }
          to {
            transform: translate(-50%, -50%) rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
