"use client";

import { useEffect, useRef } from "react";

interface Neuron {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  pulsePhase: number;
  connections: number[];
  activity: number;
}

interface Signal {
  fromIndex: number;
  toIndex: number;
  progress: number;
  speed: number;
  hue: number;
}

/**
 * NeuralNetworkBackground - AI-era animated background
 * Creates a living neural network that responds to mouse movement
 */
export function NeuralNetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const neuronsRef = useRef<Neuron[]>([]);
  const signalsRef = useRef<Signal[]>([]);
  const frameRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let width = window.innerWidth;
    let height = window.innerHeight;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
      initNeurons();
    };

    const initNeurons = () => {
      const neuronCount = Math.floor((width * height) / 25000);
      neuronsRef.current = [];

      for (let i = 0; i < neuronCount; i++) {
        neuronsRef.current.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          radius: Math.random() * 3 + 2,
          pulsePhase: Math.random() * Math.PI * 2,
          connections: [],
          activity: Math.random(),
        });
      }

      // Create connections
      neuronsRef.current.forEach((neuron, i) => {
        const connectionCount = Math.floor(Math.random() * 3) + 1;
        for (let c = 0; c < connectionCount; c++) {
          const targetIndex = Math.floor(Math.random() * neuronsRef.current.length);
          if (targetIndex !== i && !neuron.connections.includes(targetIndex)) {
            neuron.connections.push(targetIndex);
          }
        }
      });
    };

    const createSignal = () => {
      if (neuronsRef.current.length === 0) return;

      const fromIndex = Math.floor(Math.random() * neuronsRef.current.length);
      const neuron = neuronsRef.current[fromIndex];

      if (neuron.connections.length > 0) {
        const toIndex = neuron.connections[Math.floor(Math.random() * neuron.connections.length)];
        signalsRef.current.push({
          fromIndex,
          toIndex,
          progress: 0,
          speed: 0.01 + Math.random() * 0.02,
          hue: 260 + Math.random() * 60, // Purple to cyan
        });
      }
    };

    const update = () => {
      frameRef.current++;

      // Randomly create signals
      if (frameRef.current % 30 === 0) {
        createSignal();
      }

      // Update neurons
      neuronsRef.current.forEach(neuron => {
        // Mouse attraction
        const dx = mouseRef.current.x - neuron.x;
        const dy = mouseRef.current.y - neuron.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 200) {
          const force = (200 - dist) / 200;
          neuron.vx += (dx / dist) * force * 0.02;
          neuron.vy += (dy / dist) * force * 0.02;
          neuron.activity = Math.min(1, neuron.activity + 0.1);
        } else {
          neuron.activity *= 0.99;
        }

        // Apply velocity
        neuron.x += neuron.vx;
        neuron.y += neuron.vy;

        // Friction
        neuron.vx *= 0.99;
        neuron.vy *= 0.99;

        // Wrap around edges
        if (neuron.x < 0) neuron.x = width;
        if (neuron.x > width) neuron.x = 0;
        if (neuron.y < 0) neuron.y = height;
        if (neuron.y > height) neuron.y = 0;

        // Update pulse
        neuron.pulsePhase += 0.02;
      });

      // Update signals
      signalsRef.current = signalsRef.current.filter(signal => {
        signal.progress += signal.speed;
        return signal.progress < 1;
      });
    };

    const draw = () => {
      ctx.fillStyle = "rgba(10, 10, 15, 0.1)";
      ctx.fillRect(0, 0, width, height);

      const neurons = neuronsRef.current;

      // Draw connections
      neurons.forEach(neuron => {
        neuron.connections.forEach(targetIndex => {
          const target = neurons[targetIndex];
          if (!target) return;

          const dx = target.x - neuron.x;
          const dy = target.y - neuron.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 300) {
            const opacity = (1 - dist / 300) * 0.15 * (0.5 + neuron.activity * 0.5);

            ctx.beginPath();
            ctx.moveTo(neuron.x, neuron.y);
            ctx.lineTo(target.x, target.y);
            ctx.strokeStyle = `rgba(139, 92, 246, ${opacity})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      });

      // Draw signals
      signalsRef.current.forEach(signal => {
        const from = neurons[signal.fromIndex];
        const to = neurons[signal.toIndex];
        if (!from || !to) return;

        const x = from.x + (to.x - from.x) * signal.progress;
        const y = from.y + (to.y - from.y) * signal.progress;

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
        gradient.addColorStop(0, `hsla(${signal.hue}, 80%, 60%, 0.8)`);
        gradient.addColorStop(0.5, `hsla(${signal.hue}, 80%, 60%, 0.3)`);
        gradient.addColorStop(1, `hsla(${signal.hue}, 80%, 60%, 0)`);

        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${signal.hue}, 80%, 70%, 1)`;
        ctx.fill();
      });

      // Draw neurons
      neurons.forEach(neuron => {
        const pulse = Math.sin(neuron.pulsePhase) * 0.3 + 0.7;
        const radius = neuron.radius * pulse * (1 + neuron.activity * 0.5);

        // Glow
        const gradient = ctx.createRadialGradient(neuron.x, neuron.y, 0, neuron.x, neuron.y, radius * 4);
        gradient.addColorStop(0, `rgba(139, 92, 246, ${0.3 * pulse * (0.5 + neuron.activity * 0.5)})`);
        gradient.addColorStop(1, "rgba(139, 92, 246, 0)");

        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, radius * 4, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 180, 255, ${0.4 + neuron.activity * 0.6})`;
        ctx.fill();
      });
    };

    const animate = () => {
      update();
      draw();
      animationId = requestAnimationFrame(animate);
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", handleMouseMove);
    resize();
    animate();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" style={{ opacity: 0.6 }} />;
}
