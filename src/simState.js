// src/simState.js
// Per-simulation state factory and schema authority.

import {
  GRID_PRESETS,
  DEFAULT_GRID_KEY,
  FLOW_DEFAULTS,
  BODY_DEFAULTS,
  PHYSICS_DEFAULTS,
  DISPLAY_DEFAULTS,
  RUNTIME_DEFAULTS,
  APP_DEFAULTS,
} from "./config.js";

import {
  Q,
  RHO0,
  U0,
} from "./physics/lattice.js";

import {
  buildForceModelState,
  resetForceDiagnostics,
} from "./physics/forceModel.js";

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------
function resolveDims(gridKey = DEFAULT_GRID_KEY) {
  return GRID_PRESETS[gridKey] ?? GRID_PRESETS[DEFAULT_GRID_KEY];
}

function makeEmptyGeometryState() {
  return {
    arrays: {
      solid: null,
      sdf: null,
      wallDist: null,
      wallAdjacent: null,
      linkMask: null,
      linkQ: null,
      solidList: null,
      fluidSurfaceList: null,
    },
    meta: null,
    paintedNodes: new Set(),
    dirtyFlags: {
      geometryDirty: true,
      gpuUploadRequired: true,
      bodyOverlayDirty: true,
      wallOverlayDirty: true,
    },
  };
}

function makeEmptyGpuState() {
  return {
    buffers: {
      fIn: null,
      fOut: null,
      macro: null,
      solid: null,
      sdf: null,
      wallDist: null,
      wallAdjacent: null,
      linkMask: null,
      linkQ: null,
      solidList: null,
      fluidSurfaceList: null,
      uniforms: null,
      force: null,
      reduction: null,
      particles: null,
      particleRender: null,
      readback: null,
    },
    bindGroups: {
      solverA: null,
      solverB: null,
      force: null,
      particles: null,
      fieldRender: null,
    },
    state: {
      initialized: false,
      activePopulationIsIn: true,
      uniformsDirty: true,
      geometryDirty: true,
      resizeDirty: true,
    },
    readback: {
      pending: false,
      lastFrameRead: -1,
      mapped: false,
    },
  };
}

function makeEmptyRenderState() {
  return {
    flowCanvas: null,
    flowContext: null,
    bodyCanvas: null,
    wallCanvas: null,
    particleCanvas: null,
    monitorCanvas: null,

    bodyRenderer: null,
    monitorRenderer: null,
    particleRenderer: null,

    canvasFormat: null,
    sizeDirty: true,
  };
}

function makeEmptyDiagnostics() {
  return {
    force: { fx: 0.0, fy: 0.0, mz: 0.0 },
    coeff: { cd: 0.0, cl: 0.0, cm: 0.0, st: 0.0 },
    probes: { uCenter: 0.0, vCenter: 0.0 },
    reductions: {
      meanSpeed: 0.0,
      rmsDeltaU: 0.0,
      meanRhoDev: 0.0,
    },
    history: null,
  };
}

function makeDerivedState(sim) {
  const charLength =
    sim?.geometry?.meta?.charLength ??
    Math.max(4, Math.floor(sim.dims.NY / 8)) * 2;

  const reynolds = sim.params.reynolds;
  const u0 = sim.params.u0;
  const nu = (u0 * charLength) / reynolds;
  const tau = 3.0 * nu + 0.5;
  const omega = 1.0 / tau;

  const forceModel = buildForceModelState({
    ...sim,
    derived: { charLength },
  });

  return {
    charLength,
    nu,
    tau,
    omega,
    forceRef: forceModel.forceRef,
    forceScale: forceModel.forceScale,
    momentScale: forceModel.momentScale,
  };
}

// ---------------------------------------------------------
// Public API
// ---------------------------------------------------------
export function createSimState(options = {}) {
  const {
    id = "sim",
    gridKey = DEFAULT_GRID_KEY,

    bodyType = BODY_DEFAULTS.bodyType,
    aoaDeg = BODY_DEFAULTS.aoaDeg,
    nozzleDiameter = BODY_DEFAULTS.nozzleDiameter,
    cylinderDiameter = BODY_DEFAULTS.cylinderDiameter,
    cavityAspect = BODY_DEFAULTS.cavityAspect,

    u0 = FLOW_DEFAULTS.u0,
    reynolds = FLOW_DEFAULTS.reynolds,
    uCoflow = FLOW_DEFAULTS.uCoflow,

    pulsationOn = FLOW_DEFAULTS.pulsationOn,
    pulsationAmp = FLOW_DEFAULTS.pulsationAmp,
    pulsationFreq = FLOW_DEFAULTS.pulsationFreq,

    collisionModel = PHYSICS_DEFAULTS.collisionModel,
    lesModel = PHYSICS_DEFAULTS.lesModel,
    wallModel = PHYSICS_DEFAULTS.wallModel,

    displayMode = DISPLAY_DEFAULTS.mode,
    colorScale = DISPLAY_DEFAULTS.colorScale,
    showVectors = DISPLAY_DEFAULTS.showVectors,
    showGrid = DISPLAY_DEFAULTS.showGrid,
    gridFine = DISPLAY_DEFAULTS.gridFine,
    showDist = DISPLAY_DEFAULTS.showDist,
    smokeVisible = DISPLAY_DEFAULTS.smokeVisible,

    running = RUNTIME_DEFAULTS.running,
    stepsPerFrame = RUNTIME_DEFAULTS.stepsPerFrame,
  } = options;

  const { NX, NY } = resolveDims(gridKey);
  const N = NX * NY;

  const sim = {
    id,

    constants: {
      Q,
      rho0: RHO0,
      u0Ref: U0,
    },

    dims: {
      gridKey,
      NX,
      NY,
      N,
      Q,
    },

    params: {
      bodyType,
      aoaDeg,
      nozzleDiameter,
      cylinderDiameter,
      cavityAspect,
      u0,
      reynolds,
      uCoflow,
    },

    physics: {
      collisionModel,
      lesModel,
      wallModel,
    },

    flow: {
      pulsationOn,
      pulsationAmp,
      pulsationFreq,
      uCoflow,
    },

    display: {
      mode: displayMode,
      colorScale,
      showVectors,
      showGrid,
      gridFine,
      showDist,
      smokeVisible,
    },

    runtime: {
      running,
      stepsPerFrame,
      frameCount: 0,
      stepCount: 0,
      lastDtMs: 0,
    },

    app: {
      sandboxActive: APP_DEFAULTS.sandboxActive,
      sandboxMode: APP_DEFAULTS.sandboxMode,
      convergenceVisible: APP_DEFAULTS.convergenceVisible,
    },

    geometry: makeEmptyGeometryState(),
    gpu: makeEmptyGpuState(),
    render: makeEmptyRenderState(),
    diagnostics: makeEmptyDiagnostics(),
    derived: null,
  };

  sim.derived = makeDerivedState(sim);
  resetForceDiagnostics(sim);

  return sim;
}

export function resetSimRuntimeState(sim) {
  sim.runtime.frameCount = 0;
  sim.runtime.stepCount = 0;
  sim.runtime.lastDtMs = 0;
}

export function updateDerivedSimParams(sim) {
  sim.derived = makeDerivedState(sim);
  return sim.derived;
}

export function setGridPreset(sim, gridKey) {
  const { NX, NY } = resolveDims(gridKey);

  sim.dims.gridKey = gridKey;
  sim.dims.NX = NX;
  sim.dims.NY = NY;
  sim.dims.N = NX * NY;
  sim.dims.Q = Q;

  sim.geometry = makeEmptyGeometryState();
  sim.gpu = makeEmptyGpuState();
  sim.render.sizeDirty = true;

  updateDerivedSimParams(sim);
  resetSimRuntimeState(sim);
  resetForceDiagnostics(sim);
}

export function attachSimCanvases(sim, canvases = {}) {
  sim.render.flowCanvas = canvases.flowCanvas ?? sim.render.flowCanvas;
  sim.render.bodyCanvas = canvases.bodyCanvas ?? sim.render.bodyCanvas;
  sim.render.wallCanvas = canvases.wallCanvas ?? sim.render.wallCanvas;
  sim.render.particleCanvas = canvases.particleCanvas ?? sim.render.particleCanvas;
  sim.render.monitorCanvas = canvases.monitorCanvas ?? sim.render.monitorCanvas;
}

export function markGeometryDirty(sim) {
  sim.geometry.dirtyFlags.geometryDirty = true;
  sim.geometry.dirtyFlags.gpuUploadRequired = true;
  sim.geometry.dirtyFlags.bodyOverlayDirty = true;
  sim.geometry.dirtyFlags.wallOverlayDirty = true;

  sim.gpu.state.geometryDirty = true;
  sim.gpu.state.uniformsDirty = true;
}

export function markUniformsDirty(sim) {
  sim.gpu.state.uniformsDirty = true;
}

export function setGeometryPackage(sim, geometryPackage) {
  sim.geometry.arrays = geometryPackage.arrays;
  sim.geometry.meta = geometryPackage.meta;

  sim.geometry.dirtyFlags.geometryDirty = false;
  sim.geometry.dirtyFlags.gpuUploadRequired = true;
  sim.geometry.dirtyFlags.bodyOverlayDirty = true;
  sim.geometry.dirtyFlags.wallOverlayDirty = true;

  updateDerivedSimParams(sim);
  markUniformsDirty(sim);
}

export function clearPaintedNodes(sim) {
  sim.geometry.paintedNodes.clear();
  sim.geometry.dirtyFlags.wallOverlayDirty = true;
}

export function setRunning(sim, running) {
  sim.runtime.running = !!running;
}

export function setStepsPerFrame(sim, stepsPerFrame) {
  sim.runtime.stepsPerFrame = stepsPerFrame | 0;
}

export function stepFrameCounter(sim) {
  sim.runtime.frameCount += 1;
}

export function stepStepCounter(sim, count = 1) {
  sim.runtime.stepCount += count;
}


