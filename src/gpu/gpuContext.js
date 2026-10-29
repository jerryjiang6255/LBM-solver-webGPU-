// src/gpu/gpuContext.js
// ============================================================
// WebGPU device bootstrap + shared pipelines.
//
// Every pipeline uses layout "auto", so each bind group only
// carries the bindings its entry point actually touches. No pass
// uses more than 7 storage buffers, so the app runs on any
// adapter that meets the WebGPU baseline (8 per stage).
// ============================================================

export const MAX_STEPS_PER_FRAME = 32;   // per-step uniform slots
export const STEP_SLOT_BYTES = 256;      // minUniformBufferOffsetAlignment

// Bindings used by each solver entry point (see solver.wgsl).
export const SOLVER_BINDINGS = {
  macro_main:    [0, 1, 2, 4, 5],
  collide_main:  [0, 1, 2, 3, 4, 5, 6],
  stream_main:   [0, 1, 2, 3, 4, 5, 7],
  outlet_main:   [0, 2],
  rest_solids:   [0, 2, 5],
  force_main:    [0, 3, 7, 8, 9, 10],
  reduce_main:   [0, 4, 5, 10, 11],
  particle_main: [0, 4, 5, 12],
  gather_vec:    [0, 4, 5, 13],
  gather_dist:   [0, 2, 5, 13],
};

// Bindings used by each display entry point (see render.wgsl).
export const RENDER_BINDINGS = {
  smooth_init: [0, 1, 3],
  smooth_h:    [0, 2, 3, 4],
  smooth_v:    [0, 2, 3, 4],
  scalar_main: [0, 1, 2, 3, 5],
  fill_main:   [0, 2, 5],
  blur_h:      [0, 5, 6],
  blur_v:      [0, 5, 6],
  colorize:    [0, 2, 5, 7, 8],
  present:     [0, 9],
};

async function loadText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status} ${res.statusText}`);
  return res.text();
}

async function compile(device, code, label) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === "error");
  if (errors.length > 0) {
    const text = errors.map(e => `${label}:${e.lineNum}:${e.linePos} ${e.message}`).join("\n");
    throw new Error(`WGSL compilation failed:\n${text}`);
  }
  return module;
}

export async function initGpu() {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari.");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found.");

  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  device.lost.then(info => {
    const el = document.getElementById("app-error");
    if (el) el.textContent = `WebGPU device lost: ${info.message || info.reason}`;
    console.error("WebGPU device lost:", info);
  });
  device.addEventListener("uncapturederror", e => {
    console.error("WebGPU error:", e.error?.message ?? e.error);
    const el = document.getElementById("app-error");
    if (el && !el.textContent) el.textContent = `WebGPU error: ${e.error?.message ?? e.error}`;
  });

  const base = new URL("./", import.meta.url);
  const [solverCode, renderCode] = await Promise.all([
    loadText(new URL("solver.wgsl", base)),
    loadText(new URL("render.wgsl", base)),
  ]);

  const solverModule = await compile(device, solverCode, "solver.wgsl");
  const renderModule = await compile(device, renderCode, "render.wgsl");

  const compute = {};
  for (const entryPoint of Object.keys(SOLVER_BINDINGS)) {
    compute[entryPoint] = device.createComputePipeline({
      label: entryPoint,
      layout: "auto",
      compute: { module: solverModule, entryPoint },
    });
  }

  const display = {};
  for (const entryPoint of Object.keys(RENDER_BINDINGS)) {
    if (entryPoint === "present") continue;
    display[entryPoint] = device.createComputePipeline({
      label: entryPoint,
      layout: "auto",
      compute: { module: renderModule, entryPoint },
    });
  }

  display.present = device.createRenderPipeline({
    label: "field-present",
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "field_vs" },
    fragment: { module: renderModule, entryPoint: "field_fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  return { adapter, device, queue: device.queue, format, compute, display };
}

export function createBindGroup(device, pipeline, bindings, resolve, label) {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map(binding => ({ binding, resource: resolve(binding) })),
  });
}
