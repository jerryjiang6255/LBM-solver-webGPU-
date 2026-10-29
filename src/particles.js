// src/particles.js
// ============================================================
// Passive smoke tracer particles for the LBM visualizer.
//
// Design goals:
// - One particle system per sim
// - Minimal coupling: only needs sim.ux, sim.uy, sim.solid, NX, NY
// - Lightweight update + draw
// - Compare-mode friendly
//
// Usage:
//   const particles = createParticles(sim, particleCanvas, options?)
//   particles.resizeCanvas()
//   particles.reset()
//   particles.step(dt)
//   particles.draw()
//
// Notes:
// - Particles are passive tracers; they do NOT affect the flow
// - They are respawned when leaving the domain, entering solids,
//   or stalling in near-dead flow
// - WebGPU: advection and respawn run on the GPU (particle_main in
//   gpu/solver.wgsl, same rules as below). This module seeds the initial
//   cloud and draws the latest GPU positions with the CPU drawing code.
// ============================================================

const DEFAULTS = {
  count: 1400,
  seed: 1337,

  // Advection scaling in lattice-cells per rendered frame
  dt: 1.0,

  // Respawn / stall thresholds
  minSpeedSq: 1e-8,
  respawnXMax: 2.0,
  respawnAttempts: 8,

  // Visuals
  pointRadius: 5,
  alphaMin: 0.16,
  alphaMax: 1,

  // Center-band respawn bias
  centerBiasProb: 0.35,
};

// ---------------------------------------------------------
// Public factory
// ---------------------------------------------------------
export function createParticles(sim, canvas, options = {}) {
  const cfg = { ...DEFAULTS, ...options };

  const state = {
    sim,
    canvas,
    ctx: canvas ? canvas.getContext("2d") : null,
    dpr: 1,

    count: cfg.count,
    x: new Float32Array(cfg.count),
    y: new Float32Array(cfg.count),
    spd: new Float32Array(cfg.count),
    vx: new Float32Array(cfg.count),
    vy: new Float32Array(cfg.count),

    pointRadius: cfg.pointRadius,
    alphaMin: cfg.alphaMin,
    alphaMax: cfg.alphaMax,

    dt: cfg.dt,
    minSpeedSq: cfg.minSpeedSq,
    respawnXMax: cfg.respawnXMax,
    respawnAttempts: cfg.respawnAttempts,
    centerBiasProb: cfg.centerBiasProb,

    rng: mulberry32(hashSeed(cfg.seed)),
    enabled: true,
  };

  function attachCanvas(nextCanvas) {
    state.canvas = nextCanvas;
    state.ctx = nextCanvas ? nextCanvas.getContext("2d") : null;
    resizeCanvas();
  }

  function resizeCanvas() {
    if (!state.canvas || !state.sim?.flowCanvas) return;

    const flowCanvas = state.sim.flowCanvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.dpr = dpr;

    state.canvas.width = flowCanvas.width;
    state.canvas.height = flowCanvas.height;

    state.canvas.style.width = flowCanvas.style.width || "100%";
    state.canvas.style.height = flowCanvas.style.height || "100%";
  }

  function reset() {
    const { NX, NY } = state.sim;

    for (let i = 0; i < state.count; i++) {
      // Scatter initial cloud across domain to avoid inlet pulsing waves
      state.x[i] = rand(state) * Math.max(1, NX - 1);
      state.y[i] = 1 + rand(state) * Math.max(1, NY - 2);
      state.spd[i] = 0.0;
      state.vx[i] = 0.0;
      state.vy[i] = 0.0;
    }

    state.sim.gpu?.uploadParticles(state.x, state.y);
  }

  // Advection happens on the GPU; pull the most recent positions.
  function syncFromGpu() {
    const data = state.sim.gpu?.latest?.particles;
    if (!data) return;
    const n = Math.min(state.count, Math.floor(data.length / 4));
    for (let i = 0; i < n; i++) {
      const vx = data[i * 4 + 2];
      const vy = data[i * 4 + 3];
      state.x[i] = data[i * 4];
      state.y[i] = data[i * 4 + 1];
      state.vx[i] = vx;
      state.vy[i] = vy;
      state.spd[i] = Math.sqrt(vx * vx + vy * vy);
    }
  }

  function step() {
    // GPU-side (see gpu/gpuSim.js encodeFrame, advectParticles).
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

    // Fade old trails instead of fully erasing them
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0, 0, 0, 0.14)";
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  function draw() {
    if (!state.enabled || !state.ctx || !state.canvas) return;
    const sim = state.sim;
    if (!sim) return;

    syncFromGpu();

    const ctx = state.ctx;
    const W = state.canvas.width;
    const H = state.canvas.height;
    const NX = sim.NX;
    const NY = sim.NY;

    clear();

    const sx = W / NX;
    const sy = H / NY;

    ctx.save();

    // Estimate a speed normalization from inlet velocity
    const uRef = Math.max(1e-6, sim.params?.u0 ?? 0.06);

    for (let i = 0; i < state.count; i++) {
      const px = state.x[i] * sx;
      const py = state.y[i] * sy;

      const ux = state.vx[i];
      const uy = state.vy[i];
      const spd = state.spd[i];

      const t = clamp(spd / uRef, 0, 1);
      const alpha = state.alphaMin + (state.alphaMax - state.alphaMin) * t;

      // White/cyan smoke look
      ctx.fillStyle = `rgba(210,235,255,${alpha.toFixed(3)})`;

      const r = state.pointRadius + 1.2 * t;

      if (spd > 1e-8) {
        const dirx = ux / spd;
        const diry = uy / spd;

        const len = 6 + 18 * t;
        const x0 = px - 0.5 * len * dirx;
        const y0 = py - 0.5 * len * diry;
        const x1 = px + 0.5 * len * dirx;
        const y1 = py + 0.5 * len * diry;

        ctx.strokeStyle = `rgba(210,235,255,${alpha.toFixed(3)})`;
        ctx.lineWidth = 1.2 + 1.8 * t;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function setEnabled(flag) {
    state.enabled = !!flag;
    if (!state.enabled) hardClear();
  }

  function isEnabled() {
    return state.enabled;
  }

  // Initialize
  if (state.canvas) resizeCanvas();
  reset();

  return {
    state,
    attachCanvas,
    resizeCanvas,
    reset,
    step,
    draw,
    clear,
    setEnabled,
    isEnabled,
  };
}

// ---------------------------------------------------------
// RNG utilities
// ---------------------------------------------------------
function hashSeed(seed) {
  let h = 2166136261 >>> 0;
  const s = String(seed);
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rand(state) {
  return state.rng();
}

// ---------------------------------------------------------
// Small helpers
// ---------------------------------------------------------
function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}



