// src/render/monitorRenderer.js
// CPU-side diagnostic monitor renderer.
// Consumes reduced sim diagnostics only.

import { MONITOR_MODES, BODY_TYPES, GPU_TUNING } from "../config.js";

const DEFAULT_SAMPLE_EVERY_FRAMES = GPU_TUNING.sampleDiagnosticsEveryFrames;
const DEFAULT_MAX_HISTORY = GPU_TUNING.maxMonitorHistory;
const DEFAULT_FORCE_STATS_WINDOW = GPU_TUNING.forceStatsWindow;
const DEFAULT_ST_CROSSING_HISTORY = GPU_TUNING.stCrossingHistory;

const METRICS = {
  meanSpeed: {
    label: "Mean Speed",
    color: "#38c8ff",
    read(sim) {
      return sim.diagnostics?.reductions?.meanSpeed ?? 0.0;
    },
  },

  rmsDeltaU: {
    label: "RMS ΔU",
    color: "#ffb366",
    read(sim) {
      return sim.diagnostics?.reductions?.rmsDeltaU ?? 0.0;
    },
  },

  meanRhoDev: {
    label: "Mean |ρ-1|",
    color: "#c78cff",
    read(sim) {
      return sim.diagnostics?.reductions?.meanRhoDev ?? 0.0;
    },
  },

  cd: {
    label: "Cd",
    color: "#62d8ff",
    read(sim) {
      return sim.diagnostics?.coeff?.cd ?? 0.0;
    },
  },

  cl: {
    label: "Cl",
    color: "#ffc27a",
    read(sim) {
      return sim.diagnostics?.coeff?.cl ?? 0.0;
    },
  },

  cm: {
    label: "Cm",
    color: "#d8a2ff",
    read(sim) {
      return sim.diagnostics?.coeff?.cm ?? 0.0;
    },
  },

  st: {
    label: "St",
    color: "#7fffd4",
    read(sim, state) {
      const type = sim?.params?.bodyType;
      if (type !== BODY_TYPES.CYLINDER && type !== BODY_TYPES.SQUARE_CYLINDER) return 0.0;

      const steps = state.stTracker.crossingSteps;
      if (steps.length < 2) return 0.0;

      const first = steps[0];
      const last = steps[steps.length - 1];
      if (last <= first) return 0.0;

      const cycles = steps.length - 1;
      const freq = cycles / (last - first);

      const U = Math.max(sim?.params?.u0 ?? 0.0, 1e-9);
      const L = Math.max(sim?.derived?.charLength ?? 0.0, 1e-9);

      return (freq * L) / U;
    },
  },

  uCenter: {
    label: "Ucenter",
    color: "#62d8ff",
    read(sim) {
      return sim.diagnostics?.probes?.uCenter ?? 0.0;
    },
  },

  vCenter: {
    label: "Vcenter",
    color: "#ffc27a",
    read(sim) {
      return sim.diagnostics?.probes?.vCenter ?? 0.0;
    },
  },
};

export function createMonitorRenderer(sim, options = {}) {
  const sampleEveryFrames =
    Number.isInteger(options.sampleEveryFrames) && options.sampleEveryFrames > 0
      ? options.sampleEveryFrames
      : DEFAULT_SAMPLE_EVERY_FRAMES;

  const maxHistory =
    Number.isInteger(options.maxHistory) && options.maxHistory > 0
      ? options.maxHistory
      : DEFAULT_MAX_HISTORY;

  const state = {
    visible: false,
    mode: MONITOR_MODES.CONVERGENCE,
    canvas: null,
    ctx: null,
    dpr: 1,
    lastSampledFrame: -1,

    stTracker: {
      clMeanEma: 0.0,
      hasPrev: false,
      prevCenteredCl: 0.0,
      crossingSteps: [],
    },
  };

  const history = {
    frame: [],
    step: [],
    meanSpeed: [],
    rmsDeltaU: [],
    meanRhoDev: [],
    cd: [],
    cl: [],
    cm: [],
    st: [],
    uCenter: [],
    vCenter: [],
  };

  function attachCanvas(canvas) {
    state.canvas = canvas;
    state.ctx = canvas?.getContext("2d") ?? null;
    resize();
  }

  function setVisible(flag) {
    state.visible = !!flag;
    if (!state.visible) clearCanvas();
  }

  function isVisible() {
    return state.visible;
  }

  function setMode(mode) {
    if (mode === MONITOR_MODES.CONVERGENCE || mode === MONITOR_MODES.FORCES) {
      state.mode = mode;
    }
  }

  function getMode() {
    return state.mode;
  }

  function resize() {
    if (!state.canvas) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    state.dpr = dpr;

    const cssW = state.canvas.clientWidth || 260;
    const cssH = state.canvas.clientHeight || 150;

    state.canvas.width = Math.round(cssW * dpr);
    state.canvas.height = Math.round(cssH * dpr);
  }

  function clearCanvas() {
    if (!state.ctx || !state.canvas) return;
    state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
  }

 function maybeSample() {
    const frame = sim.runtime?.frameCount ?? 0;
    if (frame === state.lastSampledFrame) return false;
    if (frame % sampleEveryFrames !== 0) return false;

    sampleNow();
    state.lastSampledFrame = frame;
    return true;
  }

  function sampleNow() {
    history.frame.push(sim.runtime?.frameCount ?? 0);
    history.step.push(sim.runtime?.stepCount ?? 0);

    history.meanSpeed.push(METRICS.meanSpeed.read(sim));
    history.rmsDeltaU.push(METRICS.rmsDeltaU.read(sim));
    history.meanRhoDev.push(METRICS.meanRhoDev.read(sim));

    history.cd.push(METRICS.cd.read(sim));
    history.cl.push(METRICS.cl.read(sim));
    history.cm.push(METRICS.cm.read(sim));

    updateStTracker(sim, state);
    history.st.push(METRICS.st.read(sim, state));

    history.uCenter.push(METRICS.uCenter.read(sim));
    history.vCenter.push(METRICS.vCenter.read(sim));

    trimHistory(history, maxHistory);
  }

  function reset() {
    state.lastSampledFrame = -1;

    state.stTracker.clMeanEma = 0.0;
    state.stTracker.hasPrev = false;
    state.stTracker.prevCenteredCl = 0.0;
    state.stTracker.crossingSteps.length = 0;

    for (const key in history) {
      history[key].length = 0;
    }

    clearCanvas();
  }

  function activeMetricKeys() {
    if (state.mode === MONITOR_MODES.FORCES) {
      const type = sim?.params?.bodyType;

      if (type === BODY_TYPES.LID_DRIVEN_CAVITY) {
        return ["uCenter", "vCenter"];
      }

      const keys = ["cd", "cl"];
      if (shouldShowMoment(sim)) keys.push("cm");
      if (type === BODY_TYPES.CYLINDER || type === BODY_TYPES.SQUARE_CYLINDER) keys.push("st");
      return keys;
    }

    return ["meanSpeed", "rmsDeltaU", "meanRhoDev"];
  }

  function draw() {
    if (!state.visible || !state.ctx || !state.canvas) return;

    const ctx = state.ctx;
    const W = state.canvas.width;
    const H = state.canvas.height;
    const dpr = state.dpr || 1;

    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.scale(dpr, dpr);

    const w = W / dpr;
    const h = H / dpr;

    roundRect(ctx, 0, 0, w, h, 10);
    ctx.fillStyle = "rgba(10,16,26,0.72)";
    ctx.fill();
    ctx.strokeStyle = "rgba(80,160,255,0.18)";
    ctx.lineWidth = 1;
    ctx.stroke();

    const left = 34;
    const right = w - 10;
    const top = 18;
    const bottom = h - 30;
    const plotW = right - left;
    const plotH = bottom - top;

    drawGrid(ctx, left, right, top, bottom, plotW, plotH);
    drawAxes(ctx, left, right, top, bottom);

    const keys = activeMetricKeys();

    let globalYMax = 1.0;
    if (state.mode === MONITOR_MODES.FORCES) {
      globalYMax = computeGlobalAbsMax(history, keys);
    }

    drawYLabels(ctx, state.mode, left, top, plotH, globalYMax);

    if (state.mode === MONITOR_MODES.FORCES) {
      drawZeroLine(ctx, left, right, top, plotH);
    }

    for (const key of keys) {
      drawMetricLine(ctx, history[key], METRICS[key].color, state.mode, left, top, bottom, plotW, plotH, globalYMax);
    }

    drawLegend(ctx, w, h, keys);
    drawStatsBlock(ctx, sim, history, keys, right, top);

    ctx.restore();
  }

  if (!sim.diagnostics) sim.diagnostics = {};
  sim.diagnostics.history = history;

  return {
    state,
    history,
    sampleEveryFrames,
    maxHistory,
    attachCanvas,
    setVisible,
    isVisible,
    setMode,
    getMode,
    resize,
    clearCanvas,
    maybeSample,
    sampleNow,
    reset,
    activeMetricKeys,
    draw,
  };
}

// ---------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------
function drawGrid(ctx, left, right, top, bottom, plotW, plotH) {
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();

  for (let i = 0; i <= 4; i++) {
    const y = top + (plotH * i) / 4;
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
  }

  for (let i = 0; i <= 4; i++) {
    const x = left + (plotW * i) / 4;
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
  }

  ctx.stroke();
}

function drawAxes(ctx, left, right, top, bottom) {
  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();
}

function drawYLabels(ctx, mode, left, top, plotH, globalYMax) {
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = '500 8px "JetBrains Mono", monospace';

  for (let i = 0; i <= 4; i++) {
    const y = top + (plotH * i) / 4;

    let label;
    if (mode === MONITOR_MODES.CONVERGENCE) {
      label = (1.0 - i / 4).toFixed(2);
    } else {
      const frac = 1.0 - i / 2.0;
      const value = frac * globalYMax;

      if (Math.abs(globalYMax) >= 1) label = value.toFixed(1);
      else if (Math.abs(globalYMax) >= 0.1) label = value.toFixed(2);
      else if (Math.abs(globalYMax) >= 0.01) label = value.toFixed(3);
      else label = value.toFixed(4);
    }

    ctx.fillText(label, 6, y + 3);
  }
}

function drawZeroLine(ctx, left, right, top, plotH) {
  const yZero = top + 0.5 * plotH;
  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, yZero);
  ctx.lineTo(right, yZero);
  ctx.stroke();
}

function drawMetricLine(ctx, arr, color, mode, left, top, bottom, plotW, plotH, globalYMax) {
  if (!arr || arr.length < 2) return;

  let localMax = 1e-12;
  if (mode === MONITOR_MODES.CONVERGENCE) {
    for (let i = 0; i < arr.length; i++) {
      const a = Math.abs(arr[i]);
      if (a > localMax) localMax = a;
    }
    if (localMax <= 0) localMax = 1.0;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();

  for (let j = 0; j < arr.length; j++) {
    const x = left + (plotW * j) / Math.max(arr.length - 1, 1);

    let y;
    if (mode === MONITOR_MODES.CONVERGENCE) {
      const yn = localMax > 0 ? Math.abs(arr[j]) / localMax : 0.0;
      y = bottom - plotH * yn;
    } else {
      const v = globalYMax > 0 ? arr[j] / globalYMax : 0.0;
      const yn = 0.5 - 0.5 * v;
      y = top + plotH * yn;
    }

    if (j === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  ctx.stroke();
}

function drawLegend(ctx, w, h, keys) {
  ctx.font = '500 10px "JetBrains Mono", monospace';
  const ly = h - 10;

  const widths = keys.map(key => ctx.measureText(METRICS[key].label).width + 28);
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  let lx = (w - totalWidth) * 0.5;

  for (const key of keys) {
    ctx.fillStyle = METRICS[key].color;
    ctx.fillRect(lx, ly - 7, 8, 2);

    ctx.fillStyle = "#9fb4d3";
    ctx.fillText(METRICS[key].label, lx + 12, ly);

    lx += ctx.measureText(METRICS[key].label).width + 28;
  }
}

function drawStatsBlock(ctx, sim, history, keys, right, top) {
  if (keys.length === 0) return;

  ctx.font = '500 9px "JetBrains Mono", monospace';
  ctx.textAlign = "right";

  const type = sim?.params?.bodyType;
  let ry = top + 10;

  if (type === BODY_TYPES.LID_DRIVEN_CAVITY && history.uCenter.length > 0) {
    const uStats = waveStats(history.uCenter);
    const vStats = waveStats(history.vCenter);

    const lines = [
      { label: "<Ucenter>", mean: uStats.mean, amp: uStats.amp, color: METRICS.uCenter.color },
      { label: "<Vcenter>", mean: vStats.mean, amp: vStats.amp, color: METRICS.vCenter.color },
    ];

    for (const line of lines) {
      ctx.fillStyle = line.color;
      ctx.fillText(
        `${line.label}: ${line.mean.toFixed(4)} ± ${line.amp.toFixed(4)}`,
        right - 2,
        ry
      );
      ry += 12;
    }

    ctx.textAlign = "left";
    return;
  }

  if (history.cd.length > 0 && history.cl.length > 0) {
    const cdStats = waveStats(history.cd);
    const clStats = waveStats(history.cl);
    const cmStats = waveStats(history.cm);
    const stStats = waveStats(history.st);

    const ratio = Math.abs(cdStats.mean) > 1e-6 ? clStats.mean / cdStats.mean : 0.0;

    const lines = [
      { label: "<Cd>", mean: cdStats.mean, amp: cdStats.amp, color: METRICS.cd.color },
      { label: "<Cl>", mean: clStats.mean, amp: clStats.amp, color: METRICS.cl.color },
    ];

    if (shouldShowMoment(sim)) {
      lines.push({ label: "<Cm>", mean: cmStats.mean, amp: cmStats.amp, color: METRICS.cm.color });
    }

    if (type === BODY_TYPES.CYLINDER || type === BODY_TYPES.SQUARE_CYLINDER) {
      lines.push({ label: "St", mean: stStats.mean, amp: null, color: METRICS.st.color });
    }

    lines.push({ label: "Cl/Cd", mean: ratio, amp: null, color: "#ffffff" });

    for (const line of lines) {
      ctx.fillStyle = line.color;
      const text = line.amp === null
        ? `${line.label}: ${line.mean.toFixed(4)}`
        : `${line.label}: ${line.mean.toFixed(4)} ± ${line.amp.toFixed(4)}`;

      ctx.fillText(text, right - 2, ry);
      ry += 12;
    }
  }

  ctx.textAlign = "left";
} 

// ---------------------------------------------------------
// Logic helpers
// ---------------------------------------------------------
function shouldShowMoment(sim) {
  const type = sim?.params?.bodyType;
  return (
    type === BODY_TYPES.AIRFOIL ||
    type === BODY_TYPES.NACA2412 ||
    type === BODY_TYPES.CYLINDER ||
    type === BODY_TYPES.SQUARE_CYLINDER
  );
}

function computeGlobalAbsMax(history, keys) {
  let globalYMax = 1e-12;

  for (const key of keys) {
    const arr = history[key];
    if (!arr || arr.length === 0) continue;

    for (let i = 0; i < arr.length; i++) {
      const a = Math.abs(arr[i]);
      if (a > globalYMax) globalYMax = a;
    }
  }

  return globalYMax <= 0 ? 1.0 : globalYMax;
}

function trimHistory(history, maxHistory) {
  const over = history.frame.length - maxHistory;
  if (over <= 0) return;

  for (const key in history) {
    history[key].splice(0, over);
  }
}

function updateStTracker(sim, state) {
  const type = sim?.params?.bodyType;
  if (type !== BODY_TYPES.CYLINDER && type !== BODY_TYPES.SQUARE_CYLINDER) return;

  const tracker = state.stTracker;
  const cl = sim.diagnostics?.coeff?.cl ?? 0.0;
  const step = sim.runtime?.stepCount ?? 0;

  tracker.clMeanEma = tracker.hasPrev
    ? 0.98 * tracker.clMeanEma + 0.02 * cl
    : cl;

  const centeredCl = cl - tracker.clMeanEma;

  if (tracker.hasPrev) {
    if (tracker.prevCenteredCl <= 0.0 && centeredCl > 0.0) {
      tracker.crossingSteps.push(step);

      if (tracker.crossingSteps.length > DEFAULT_ST_CROSSING_HISTORY) {
        tracker.crossingSteps.shift();
      }
    }
  }

  tracker.prevCenteredCl = centeredCl;
  tracker.hasPrev = true;
}

function recentWindow(arr, n) {
  if (!arr || arr.length === 0) return [];
  return arr.slice(Math.max(0, arr.length - n));
}

function meanOf(arr) {
  if (!arr || arr.length === 0) return 0.0;
  let s = 0.0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function minOf(arr) {
  if (!arr || arr.length === 0) return 0.0;
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] < m) m = arr[i];
  return m;
}

function maxOf(arr) {
  if (!arr || arr.length === 0) return 0.0;
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

function waveStats(arr, windowSize = DEFAULT_FORCE_STATS_WINDOW) {
  const win = recentWindow(arr, windowSize);
  if (win.length === 0) {
    return { mean: 0.0, min: 0.0, max: 0.0, amp: 0.0 };
  }

  const mean = meanOf(win);
  const min = minOf(win);
  const max = maxOf(win);
  const amp = Math.max(Math.abs(max - mean), Math.abs(mean - min));

  return { mean, min, max, amp };
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, 0.5 * w, 0.5 * h);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
} 
