// src/render/particles.js
// CPU-style particle overlay renderer.
// Draws GPU-advected particle state on a 2D canvas with fading trails,
// line streaks for moving particles, and dots for near-stationary particles.

import { PARTICLE_DEFAULTS } from "../config.js";

export function createParticleRenderer(sim, options = {}) {
  const state = {
    enabled: false,

    canvas: null,
    ctx: null,

    count:
      options.count ??
      PARTICLE_DEFAULTS.countSingle,

    pointRadius:
      options.pointRadius ??
      PARTICLE_DEFAULTS.pointRadius,

    alphaMin:
      options.alphaMin ??
      PARTICLE_DEFAULTS.alphaMin,

    alphaMax:
      options.alphaMax ??
      PARTICLE_DEFAULTS.alphaMax,

    particleState: null, // Float32Array [x, y, vx, vy, ...]
  };

  function attachCanvas(canvas) {
    state.canvas = canvas;
    state.ctx = canvas?.getContext("2d") ?? null;
    resize();
  }

  function attachGpuContext(_gpuCtx) {
    // No-op: retained for compatibility with existing call sites.
  }

 function resize() {
    if (!state.canvas) return;

    const flowCanvas = sim.render?.flowCanvas;
    if (!flowCanvas) return;

    if (state.canvas.width !== flowCanvas.width) {
      state.canvas.width = flowCanvas.width;
    }

    if (state.canvas.height !== flowCanvas.height) {
      state.canvas.height = flowCanvas.height;
    }

    state.canvas.style.width = flowCanvas.style.width || "100%";
    state.canvas.style.height = flowCanvas.style.height || "100%";
  }

  function hardClear() {
    if (!state.ctx || !state.canvas) return;
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  }

  function clear() {
    if (!state.ctx || !state.canvas) return;

    const ctx = state.ctx;
    const W = state.canvas.width;
    const H = state.canvas.height;

    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0, 0, 0, 0.14)";
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function setEnabled(flag) {
    state.enabled = !!flag;
    sim.display.smokeVisible = state.enabled;
    if (!state.enabled) hardClear();
  }

  function isEnabled() {
    return state.enabled;
  }

  function getCount() {
    return state.count;
  }

  function setCount(count) {
    state.count = Math.max(0, count | 0);
  }

  function setParticleState(particleState) {
    state.particleState = particleState ?? null;

    if (particleState instanceof Float32Array) {
      state.count = Math.floor(particleState.length / 4);
    }
  }

  function render(_gpuCtx, _pipelines) {
    if (!state.enabled) return;
    if (!state.canvas || !state.ctx) return;
    if (!state.particleState) return;

    resize();
    clear();

    const ctx = state.ctx;
    const W = state.canvas.width;
    const H = state.canvas.height;
    const NX = sim.dims?.NX ?? 1;
    const NY = sim.dims?.NY ?? 1;

    const sx = W / NX;
    const sy = H / NY;

    const uRef = Math.max(1e-6, sim.params?.u0 ?? 0.06);
    const data = state.particleState;
    const count = Math.floor(data.length / 4);

    ctx.save();

    for (let i = 0; i < count; i++) {
      const x = data[i * 4 + 0];
      const y = data[i * 4 + 1];
      const vx = data[i * 4 + 2];
      const vy = data[i * 4 + 3];

      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

      const px = x * sx;
      const py = y * sy;

      const spd = Math.sqrt(vx * vx + vy * vy);
      const t = clamp(spd / uRef, 0, 1);
      const alpha = state.alphaMin + (state.alphaMax - state.alphaMin) * t;
      const color = `rgba(210,235,255,${alpha.toFixed(3)})`;
      const r = state.pointRadius + 1.2 * t;

      if (spd > 1e-8) {
        const dirx = vx / spd;
        const diry = vy / spd;

        const len = 6 + 18 * t;
        const x0 = px - 0.5 * len * dirx;
        const y0 = py - 0.5 * len * diry;
        const x1 = px + 0.5 * len * dirx;
        const y1 = py + 0.5 * len * diry;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2 + 1.8 * t;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      } else {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();
  }

  return {
    state,
    attachCanvas,
    attachGpuContext,
    resize,
    clear,
    render,
    setEnabled,
    isEnabled,
    getCount,
    setCount,
    setParticleState,
    hardClear,
  };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}


