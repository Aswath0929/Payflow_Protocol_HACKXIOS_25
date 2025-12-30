"use client";

import { useEffect, useRef } from "react";

type BackgroundTheme = "dashboard" | "oracle" | "explorer" | "debug" | "minimal" | "fraud";

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
    if (theme !== "dashboard" && theme !== "oracle" && theme !== "fraud") return;

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

    // Particle system for dashboard theme - Enhanced with Neural Network Connections
    if (theme === "dashboard") {
      interface DataParticle {
        x: number;
        y: number;
        vx: number;
        vy: number;
        size: number;
        alpha: number;
        color: string;
        type: "data" | "flow" | "pulse" | "neuron";
      }

      const particles: DataParticle[] = [];
      const particleCount = intensity === "high" ? 100 : intensity === "medium" ? 70 : 40;
      const colors = ["#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#d946ef"];

      // Initialize particles with more neuron types
      for (let i = 0; i < particleCount; i++) {
        const typeRoll = Math.random();
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.4,
          vy: (Math.random() - 0.5) * 0.4 + 0.2,
          size: Math.random() * 3 + 1.5,
          alpha: Math.random() * 0.6 + 0.2,
          color: colors[Math.floor(Math.random() * colors.length)],
          type: typeRoll > 0.8 ? "neuron" : typeRoll > 0.6 ? "pulse" : typeRoll > 0.4 ? "flow" : "data",
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
      for (let i = 0; i < 20; i++) {
        streams.push({
          x: Math.random() * width,
          y: Math.random() * height,
          length: Math.random() * 120 + 60,
          speed: Math.random() * 2.5 + 1,
          alpha: Math.random() * 0.25 + 0.1,
        });
      }

      // Neural connection threshold
      const connectionDistance = 150;

      const animate = () => {
        ctx.fillStyle = "rgba(10, 10, 15, 0.08)";
        ctx.fillRect(0, 0, width, height);

        // Draw neural network connections between nearby particles
        for (let i = 0; i < particles.length; i++) {
          for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < connectionDistance) {
              const alpha = (1 - distance / connectionDistance) * 0.15;
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }

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

          // Convert hex to rgba helper
          const hexToRgba = (hex: string, alpha: number) => {
            const match = hex.match(/^#([0-9a-f]{6})$/i);
            if (match) {
              const r = parseInt(match[1].slice(0, 2), 16);
              const g = parseInt(match[1].slice(2, 4), 16);
              const b = parseInt(match[1].slice(4, 6), 16);
              return `rgba(${r}, ${g}, ${b}, ${alpha})`;
            }
            return `rgba(139, 92, 246, ${alpha})`;
          };

          if (p.type === "neuron") {
            // Neuron node with glow
            const glowSize = p.size * 2;
            const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowSize);
            gradient.addColorStop(0, hexToRgba(p.color, p.alpha));
            gradient.addColorStop(0.5, hexToRgba(p.color, p.alpha * 0.5));
            gradient.addColorStop(1, "transparent");
            ctx.arc(p.x, p.y, glowSize, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            // Core
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = hexToRgba(p.color, p.alpha + 0.3);
            ctx.fill();
          } else if (p.type === "pulse") {
            // Pulsing circle
            const pulseSize = Math.max(0.5, p.size + Math.sin(Date.now() * 0.005 + p.x) * 2);
            ctx.arc(p.x, p.y, pulseSize, 0, Math.PI * 2);
            ctx.fillStyle = hexToRgba(p.color, p.alpha);
            ctx.shadowBlur = 10;
            ctx.shadowColor = p.color;
            ctx.fill();
            ctx.shadowBlur = 0;
          } else if (p.type === "flow") {
            // Data flow line
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p.x + p.vx * 12, p.y + p.vy * 12);
            ctx.strokeStyle = hexToRgba(p.color, p.alpha);
            ctx.lineWidth = p.size / 2;
            ctx.stroke();
          } else {
            // Standard data particle
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = hexToRgba(p.color, p.alpha);
            ctx.fill();
          }

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

    // FRAUD THEME: Advanced Qwen3 MoE Neural Architecture Visualization
    if (theme === "fraud") {
      // ═══════════════════════════════════════════════════════════════
      // QWEN3 MIXTURE OF EXPERTS - IMMERSIVE AI VISUALIZATION
      // ═══════════════════════════════════════════════════════════════
      
      interface NeuralNode {
        x: number;
        y: number;
        layer: number;
        radius: number;
        activation: number;
        targetActivation: number;
        color: string;
        pulsePhase: number;
      }

      interface Synapse {
        from: number;
        to: number;
        weight: number;
        signal: number;
        signalProgress: number;
        active: boolean;
      }

      interface ExpertModule {
        x: number;
        y: number;
        radius: number;
        rotation: number;
        active: boolean;
        color: string;
        name: string;
        pulsePhase: number;
        expertId: number;
        confidence: number;
      }
      
      interface DataParticle {
        x: number;
        y: number;
        vx: number;
        vy: number;
        size: number;
        alpha: number;
        color: string;
        life: number;
        maxLife: number;
      }
      
      interface AttentionBeam {
        startX: number;
        startY: number;
        endX: number;
        endY: number;
        progress: number;
        color: string;
        width: number;
      }

      const neurons: NeuralNode[] = [];
      const synapses: Synapse[] = [];
      const experts: ExpertModule[] = [];
      const dataParticles: DataParticle[] = [];
      const attentionBeams: AttentionBeam[] = [];
      
      // Qwen3 MoE Color Palette - Deep Cybernetic Theme
      const moeColors = {
        primary: "#8b5cf6",    // Violet - Main accent
        secondary: "#06b6d4",  // Cyan - Secondary
        tertiary: "#d946ef",   // Fuchsia - Tertiary
        success: "#10b981",    // Emerald - Success/Output
        warning: "#f59e0b",    // Amber - Warning
        danger: "#ef4444",     // Red - Danger/Alert
        info: "#0ea5e9",       // Sky - Info
      };
      
      const layerColors = [
        "#06b6d4", // Input - Cyan
        "#0ea5e9", // Embed - Sky Blue  
        "#6366f1", // Hidden 1 - Indigo
        "#8b5cf6", // Expert Router - Violet
        "#d946ef", // MoE Layer - Fuchsia
        "#10b981", // Output - Emerald
      ];

      // Create MoE architecture neural layers
      const layerConfigs = [
        { nodes: 8, x: width * 0.05, name: "Input" },
        { nodes: 12, x: width * 0.2, name: "Embed" },
        { nodes: 16, x: width * 0.35, name: "Attention" },
        { nodes: 8, x: width * 0.65, name: "Router" },
        { nodes: 6, x: width * 0.8, name: "FFN" },
        { nodes: 3, x: width * 0.95, name: "Output" },
      ];

      // Initialize neurons with staggered positions
      layerConfigs.forEach((layer, layerIdx) => {
        const verticalMargin = 80;
        const spacing = (height - verticalMargin * 2) / (layer.nodes + 1);
        for (let i = 0; i < layer.nodes; i++) {
          neurons.push({
            x: layer.x + (Math.random() - 0.5) * 15,
            y: verticalMargin + spacing * (i + 1) + (Math.random() - 0.5) * 8,
            layer: layerIdx,
            radius: layerIdx === 3 ? 7 : layerIdx === 4 ? 6 : 4,
            activation: Math.random() * 0.2,
            targetActivation: 0,
            color: layerColors[layerIdx],
            pulsePhase: Math.random() * Math.PI * 2,
          });
        }
      });

      // Create sparse MoE-style synapses
      let startIdx = 0;
      for (let l = 0; l < layerConfigs.length - 1; l++) {
        const currentLayerSize = layerConfigs[l].nodes;
        const nextLayerSize = layerConfigs[l + 1].nodes;
        const nextStartIdx = startIdx + currentLayerSize;

        for (let i = 0; i < currentLayerSize; i++) {
          // Sparse connections (MoE style - not fully connected)
          const connectionCount = Math.min(nextLayerSize, l === 3 ? 2 : 3 + Math.floor(Math.random() * 2));
          const targetIndices = new Set<number>();
          while (targetIndices.size < connectionCount) {
            targetIndices.add(Math.floor(Math.random() * nextLayerSize));
          }
          
          targetIndices.forEach(j => {
            synapses.push({
              from: startIdx + i,
              to: nextStartIdx + j,
              weight: 0.15 + Math.random() * 0.35,
              signal: 0,
              signalProgress: -1,
              active: Math.random() > 0.3,
            });
          });
        }
        startIdx = nextStartIdx;
      }

      // Create 8 Expert Modules (representing Qwen3's 8 experts per MoE layer)
      const expertNames = ["Velocity", "Pattern", "Graph", "Timing", "Compliance", "Behavior", "Anomaly", "Risk"];
      const expertColors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#0ea5e9", "#8b5cf6", "#d946ef", "#ec4899"];
      const centerX = width * 0.5;
      const centerY = height * 0.5;
      const expertRadius = Math.min(width, height) * 0.28;
      
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2 - Math.PI / 2;
        experts.push({
          x: centerX + Math.cos(angle) * expertRadius,
          y: centerY + Math.sin(angle) * expertRadius,
          radius: 28 + Math.random() * 8,
          rotation: Math.random() * Math.PI * 2,
          active: i < 2, // Top-k=2 experts active by default
          color: expertColors[i],
          name: expertNames[i],
          pulsePhase: Math.random() * Math.PI * 2,
          expertId: i,
          confidence: 0.3 + Math.random() * 0.7,
        });
      }

      // Initialize data particles (representing tokens/data flow)
      const createParticle = () => {
        const colors = [moeColors.primary, moeColors.secondary, moeColors.tertiary, moeColors.info];
        return {
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.8,
          vy: (Math.random() - 0.5) * 0.8 + 0.2,
          size: 1 + Math.random() * 2,
          alpha: 0.1 + Math.random() * 0.4,
          color: colors[Math.floor(Math.random() * colors.length)],
          life: 0,
          maxLife: 200 + Math.random() * 300,
        };
      };
      
      for (let i = 0; i < 80; i++) {
        dataParticles.push(createParticle());
      }

      // Animation state
      let lastSignalTime = 0;
      let lastExpertToggle = 0;
      let lastBeamTime = 0;
      const signalInterval = 600;
      const expertToggleInterval = 2000;
      const beamInterval = 300;

      const animate = () => {
        // Deep space fade with subtle gradient
        const bgGradient = ctx.createLinearGradient(0, 0, width, height);
        bgGradient.addColorStop(0, "rgba(2, 6, 23, 0.12)");
        bgGradient.addColorStop(0.5, "rgba(5, 5, 15, 0.1)");
        bgGradient.addColorStop(1, "rgba(2, 6, 23, 0.12)");
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        const now = Date.now();
        const time = now * 0.001;

        // ─────────────────────────────────────────────
        // 1. HEXAGONAL GRID BACKGROUND (Tech/AI feel)
        // ─────────────────────────────────────────────
        const hexSize = 40;
        const hexHeight = hexSize * Math.sqrt(3);
        ctx.strokeStyle = `rgba(99, 102, 241, 0.04)`;
        ctx.lineWidth = 0.5;
        
        for (let row = 0; row < height / hexHeight + 2; row++) {
          for (let col = 0; col < width / (hexSize * 1.5) + 2; col++) {
            const offsetX = (row % 2) * hexSize * 0.75;
            const x = col * hexSize * 1.5 + offsetX;
            const y = row * hexHeight * 0.5;
            
            ctx.beginPath();
            for (let i = 0; i < 6; i++) {
              const angle = (Math.PI / 3) * i + Math.PI / 6;
              const hx = x + hexSize * 0.4 * Math.cos(angle);
              const hy = y + hexSize * 0.4 * Math.sin(angle);
              if (i === 0) ctx.moveTo(hx, hy);
              else ctx.lineTo(hx, hy);
            }
            ctx.closePath();
            ctx.stroke();
          }
        }

        // ─────────────────────────────────────────────
        // 2. DATA FLOW PARTICLES
        // ─────────────────────────────────────────────
        dataParticles.forEach((p, idx) => {
          p.x += p.vx;
          p.y += p.vy;
          p.life++;
          
          // Wrap or reset
          if (p.x < -10 || p.x > width + 10 || p.y < -10 || p.y > height + 10 || p.life > p.maxLife) {
            Object.assign(dataParticles[idx], createParticle());
            dataParticles[idx].y = -10;
          }
          
          const fadeRatio = Math.min(1, (p.maxLife - p.life) / 50);
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          // Convert hex color to rgba
          const hexMatch = p.color.match(/^#([0-9a-f]{6})$/i);
          if (hexMatch) {
            const r = parseInt(hexMatch[1].slice(0, 2), 16);
            const g = parseInt(hexMatch[1].slice(2, 4), 16);
            const b = parseInt(hexMatch[1].slice(4, 6), 16);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha * fadeRatio})`;
          } else {
            ctx.fillStyle = `rgba(139, 92, 246, ${p.alpha * fadeRatio})`;
          }
          ctx.fill();
        });

        // ─────────────────────────────────────────────
        // 3. ATTENTION BEAMS (Cross-attention visualization)
        // ─────────────────────────────────────────────
        if (now - lastBeamTime > beamInterval && attentionBeams.length < 8) {
          lastBeamTime = now;
          const startNeuron = neurons[Math.floor(Math.random() * neurons.length)];
          const endNeuron = neurons[Math.floor(Math.random() * neurons.length)];
          if (startNeuron.layer !== endNeuron.layer) {
            attentionBeams.push({
              startX: startNeuron.x,
              startY: startNeuron.y,
              endX: endNeuron.x,
              endY: endNeuron.y,
              progress: 0,
              color: layerColors[startNeuron.layer],
              width: 1 + Math.random() * 2,
            });
          }
        }
        
        for (let i = attentionBeams.length - 1; i >= 0; i--) {
          const beam = attentionBeams[i];
          beam.progress += 0.02;
          
          if (beam.progress > 1) {
            attentionBeams.splice(i, 1);
            continue;
          }
          
          const alpha = Math.sin(beam.progress * Math.PI) * 0.3;
          const currentX = beam.startX + (beam.endX - beam.startX) * beam.progress;
          const currentY = beam.startY + (beam.endY - beam.startY) * beam.progress;
          
          // Helper to convert hex to rgba
          const hexToRgba = (hex: string, a: number) => {
            const match = hex.match(/^#([0-9a-f]{6})$/i);
            if (match) {
              const r = parseInt(match[1].slice(0, 2), 16);
              const g = parseInt(match[1].slice(2, 4), 16);
              const b = parseInt(match[1].slice(4, 6), 16);
              return `rgba(${r}, ${g}, ${b}, ${a})`;
            }
            return `rgba(139, 92, 246, ${a})`;
          };
          
          // Draw beam trail
          const gradient = ctx.createLinearGradient(beam.startX, beam.startY, currentX, currentY);
          gradient.addColorStop(0, "transparent");
          gradient.addColorStop(0.8, hexToRgba(beam.color, alpha));
          gradient.addColorStop(1, beam.color);
          
          ctx.beginPath();
          ctx.moveTo(beam.startX, beam.startY);
          ctx.lineTo(currentX, currentY);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = beam.width;
          ctx.stroke();
          
          // Beam head glow
          ctx.beginPath();
          ctx.arc(currentX, currentY, 3, 0, Math.PI * 2);
          ctx.fillStyle = beam.color;
          ctx.shadowBlur = 8;
          ctx.shadowColor = beam.color;
          ctx.fill();
          ctx.shadowBlur = 0;
        }

        // ─────────────────────────────────────────────
        // 4. EXPERT MODULES (8 Experts in MoE Ring)
        // ─────────────────────────────────────────────
        
        // Draw center hub
        const hubPulse = (Math.sin(time * 2) + 1) * 0.5;
        const hubGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 60);
        hubGradient.addColorStop(0, `rgba(139, 92, 246, ${0.15 + hubPulse * 0.1})`);
        hubGradient.addColorStop(0.5, `rgba(139, 92, 246, ${0.05 + hubPulse * 0.05})`);
        hubGradient.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
        ctx.fillStyle = hubGradient;
        ctx.fill();
        
        // Router label
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.fillStyle = `rgba(139, 92, 246, ${0.5 + hubPulse * 0.3})`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("ROUTER", centerX, centerY - 8);
        ctx.font = "9px monospace";
        ctx.fillStyle = "rgba(217, 70, 239, 0.6)";
        ctx.fillText("TOP-K=2", centerX, centerY + 8);

        // Draw connections from center to experts
        experts.forEach((expert, idx) => {
          const lineAlpha = expert.active ? 0.4 : 0.1;
          const gradient = ctx.createLinearGradient(centerX, centerY, expert.x, expert.y);
          gradient.addColorStop(0, `rgba(139, 92, 246, ${lineAlpha})`);
          gradient.addColorStop(1, `${expert.color}${expert.active ? '60' : '20'}`);
          
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(expert.x, expert.y);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = expert.active ? 2 : 0.5;
          ctx.stroke();
        });

        // Toggle expert activation (simulate routing decisions)
        if (now - lastExpertToggle > expertToggleInterval) {
          lastExpertToggle = now;
          // Deactivate all
          experts.forEach(e => e.active = false);
          // Activate top-k=2 random experts
          const indices = Array.from({ length: 8 }, (_, i) => i);
          for (let k = 0; k < 2; k++) {
            const pick = Math.floor(Math.random() * indices.length);
            experts[indices[pick]].active = true;
            experts[indices[pick]].confidence = 0.7 + Math.random() * 0.3;
            indices.splice(pick, 1);
          }
        }

        // Draw expert modules
        experts.forEach((expert) => {
          expert.rotation += expert.active ? 0.008 : 0.002;
          expert.pulsePhase += 0.05;
          
          const pulse = expert.active ? (Math.sin(expert.pulsePhase) + 1) * 0.5 : 0;
          const glowRadius = expert.radius * (1.5 + pulse * 0.5);
          
          // Outer glow
          const glowGradient = ctx.createRadialGradient(
            expert.x, expert.y, expert.radius * 0.3,
            expert.x, expert.y, glowRadius
          );
          glowGradient.addColorStop(0, `${expert.color}${expert.active ? '40' : '15'}`);
          glowGradient.addColorStop(0.6, `${expert.color}${expert.active ? '15' : '08'}`);
          glowGradient.addColorStop(1, "transparent");
          
          ctx.beginPath();
          ctx.arc(expert.x, expert.y, glowRadius, 0, Math.PI * 2);
          ctx.fillStyle = glowGradient;
          ctx.fill();

          // Rotating octagon (8-sided for 8 experts)
          ctx.save();
          ctx.translate(expert.x, expert.y);
          ctx.rotate(expert.rotation);
          ctx.beginPath();
          for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2;
            const r = expert.radius * (1 + (i % 2) * 0.1);
            const x = Math.cos(angle) * r;
            const y = Math.sin(angle) * r;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.strokeStyle = expert.active ? expert.color : `${expert.color}50`;
          ctx.lineWidth = expert.active ? 2.5 : 1;
          ctx.stroke();
          ctx.fillStyle = expert.active ? `${expert.color}20` : `${expert.color}08`;
          ctx.fill();
          ctx.restore();

          // Expert ID badge
          if (expert.active) {
            ctx.beginPath();
            ctx.arc(expert.x, expert.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = expert.color;
            ctx.fill();
            ctx.font = "bold 9px sans-serif";
            ctx.fillStyle = "#fff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`${expert.expertId}`, expert.x, expert.y);
          }

          // Expert name and confidence
          ctx.font = "bold 9px 'JetBrains Mono', monospace";
          ctx.fillStyle = expert.active ? `${expert.color}ee` : `${expert.color}60`;
          ctx.textAlign = "center";
          ctx.fillText(expert.name.toUpperCase(), expert.x, expert.y + expert.radius + 14);
          
          if (expert.active) {
            ctx.font = "8px monospace";
            ctx.fillStyle = `${expert.color}aa`;
            ctx.fillText(`${(expert.confidence * 100).toFixed(0)}%`, expert.x, expert.y + expert.radius + 24);
          }
        });

        // ─────────────────────────────────────────────
        // 5. NEURAL SYNAPSES
        // ─────────────────────────────────────────────
        synapses.forEach(synapse => {
          if (!synapse.active) return;
          
          const from = neurons[synapse.from];
          const to = neurons[synapse.to];

          // Connection line
          const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
          gradient.addColorStop(0, `${from.color}25`);
          gradient.addColorStop(1, `${to.color}25`);
          
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = synapse.weight * 1.2;
          ctx.stroke();

          // Signal propagation
          if (synapse.signalProgress >= 0 && synapse.signalProgress <= 1) {
            const signalX = from.x + (to.x - from.x) * synapse.signalProgress;
            const signalY = from.y + (to.y - from.y) * synapse.signalProgress;
            
            // Signal trail
            for (let t = 0; t < 5; t++) {
              const trailProgress = Math.max(0, synapse.signalProgress - t * 0.03);
              const tx = from.x + (to.x - from.x) * trailProgress;
              const ty = from.y + (to.y - from.y) * trailProgress;
              const trailAlpha = (1 - t / 5) * 0.5;
              
              ctx.beginPath();
              ctx.arc(tx, ty, 2 - t * 0.3, 0, Math.PI * 2);
              ctx.fillStyle = `${from.color}${Math.floor(trailAlpha * 255).toString(16).padStart(2, '0')}`;
              ctx.fill();
            }
            
            // Main signal
            ctx.beginPath();
            ctx.arc(signalX, signalY, 3, 0, Math.PI * 2);
            ctx.fillStyle = from.color;
            ctx.shadowBlur = 12;
            ctx.shadowColor = from.color;
            ctx.fill();
            ctx.shadowBlur = 0;

            synapse.signalProgress += 0.03;
            if (synapse.signalProgress > 1) {
              synapse.signalProgress = -1;
              neurons[synapse.to].targetActivation = Math.min(1, neurons[synapse.to].targetActivation + 0.4);
            }
          }
        });

        // Trigger signals
        if (now - lastSignalTime > signalInterval) {
          lastSignalTime = now;
          const inputNeurons = neurons.filter(n => n.layer === 0);
          const selected = inputNeurons[Math.floor(Math.random() * inputNeurons.length)];
          if (selected) {
            selected.targetActivation = 1;
            synapses
              .filter(s => neurons[s.from] === selected && s.active)
              .forEach(s => { if (Math.random() > 0.3) s.signalProgress = 0; });
          }
        }

        // ─────────────────────────────────────────────
        // 6. NEURAL NODES
        // ─────────────────────────────────────────────
        neurons.forEach((neuron, idx) => {
          neuron.activation += (neuron.targetActivation - neuron.activation) * 0.08;
          neuron.targetActivation *= 0.97;
          neuron.pulsePhase += 0.03;

          const basePulse = (Math.sin(neuron.pulsePhase) + 1) * 0.15;
          const glowSize = neuron.radius + neuron.activation * 10 + basePulse * 3;
          
          // Glow
          const gradient = ctx.createRadialGradient(
            neuron.x, neuron.y, 0,
            neuron.x, neuron.y, glowSize
          );
          gradient.addColorStop(0, neuron.color);
          gradient.addColorStop(0.3, `${neuron.color}80`);
          gradient.addColorStop(1, "transparent");

          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, glowSize, 0, Math.PI * 2);
          ctx.fillStyle = gradient;
          ctx.fill();

          // Core
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, neuron.radius, 0, Math.PI * 2);
          const coreAlpha = Math.floor(80 + neuron.activation * 175);
          ctx.fillStyle = `${neuron.color}${coreAlpha.toString(16).padStart(2, '0')}`;
          ctx.fill();
          ctx.strokeStyle = `${neuron.color}aa`;
          ctx.lineWidth = 1;
          ctx.stroke();

          // Propagate
          if (neuron.activation > 0.6) {
            synapses
              .filter(s => s.from === idx && s.signalProgress < 0 && s.active)
              .forEach(s => { if (Math.random() > 0.6) s.signalProgress = 0; });
          }
        });

        // ─────────────────────────────────────────────
        // 7. QWEN3 MoE BRANDING (Subtle watermark)
        // ─────────────────────────────────────────────
        const brandPulse = (Math.sin(time * 1.5) + 1) * 0.5;
        ctx.font = "bold 48px 'Inter', sans-serif";
        ctx.fillStyle = `rgba(139, 92, 246, ${0.03 + brandPulse * 0.02})`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("QWEN3", centerX, centerY + expertRadius + 80);
        
        ctx.font = "bold 18px 'JetBrains Mono', monospace";
        ctx.fillStyle = `rgba(217, 70, 239, ${0.06 + brandPulse * 0.03})`;
        ctx.fillText("MIXTURE OF EXPERTS", centerX, centerY + expertRadius + 105);

        animationId = requestAnimationFrame(animate);
      };

      animate();

      return () => {
        window.removeEventListener("resize", resize);
        cancelAnimationFrame(animationId);
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
      <div className="absolute inset-0 bg-[#020617]" />

      {/* Canvas for dashboard/oracle/fraud themes */}
      {(theme === "dashboard" || theme === "oracle" || theme === "fraud") && (
        <canvas ref={canvasRef} className="absolute inset-0 opacity-100" />
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

      {/* Fraud theme: Neural Network / Qwen3 MoE aesthetic - canvas-based animation visible */}
      {theme === "fraud" && (
        <>
          {/* Ambient glows for neural network feel - Enhanced visibility */}
          <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-violet-600/15 rounded-full blur-[150px] pointer-events-none animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-cyan-500/12 rounded-full blur-[120px] pointer-events-none animate-pulse" style={{ animationDelay: "1s" }} />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-purple-500/10 rounded-full blur-[180px] pointer-events-none animate-pulse" style={{ animationDelay: "2s" }} />
          <div className="absolute top-1/3 right-1/3 w-[300px] h-[300px] bg-fuchsia-500/10 rounded-full blur-[100px] pointer-events-none animate-pulse" style={{ animationDelay: "0.5s" }} />
          
          {/* Grid overlay for AI/tech feel */}
          <div 
            className="absolute inset-0 opacity-[0.03] pointer-events-none"
            style={{
              backgroundImage: `
                linear-gradient(rgba(139, 92, 246, 0.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139, 92, 246, 0.5) 1px, transparent 1px)
              `,
              backgroundSize: "50px 50px",
            }}
          />
          
          {/* Scanning effect */}
          <div
            className="absolute left-0 right-0 h-[3px] opacity-40 pointer-events-none"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.9), rgba(6, 182, 212, 0.9), rgba(217, 70, 239, 0.9), transparent)",
              animation: "neural-scan 4s ease-in-out infinite",
            }}
          />
          
          {/* Corner accents - More prominent */}
          <div className="absolute top-0 left-0 w-40 h-40 border-l-2 border-t-2 border-violet-500/40 rounded-tl-3xl pointer-events-none" />
          <div className="absolute top-0 right-0 w-40 h-40 border-r-2 border-t-2 border-cyan-500/40 rounded-tr-3xl pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-40 h-40 border-l-2 border-b-2 border-fuchsia-500/40 rounded-bl-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-0 w-40 h-40 border-r-2 border-b-2 border-emerald-500/40 rounded-br-3xl pointer-events-none" />
          
          {/* MoE indicator dots at corners */}
          <div className="absolute top-4 left-4 flex gap-2 pointer-events-none">
            <div className="w-2 h-2 bg-violet-500 rounded-full animate-ping" />
            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-ping" style={{ animationDelay: "0.2s" }} />
            <div className="w-2 h-2 bg-fuchsia-500 rounded-full animate-ping" style={{ animationDelay: "0.4s" }} />
          </div>
          <div className="absolute top-4 right-4 flex gap-2 pointer-events-none">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" style={{ animationDelay: "0.6s" }} />
            <div className="w-2 h-2 bg-amber-500 rounded-full animate-ping" style={{ animationDelay: "0.8s" }} />
          </div>
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

        @keyframes neural-scan {
          0%, 100% {
            top: 0;
            opacity: 0;
          }
          10% {
            opacity: 0.3;
          }
          50% {
            opacity: 0.5;
          }
          90% {
            opacity: 0.3;
          }
          100% {
            top: 100%;
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
