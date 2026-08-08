// src/main.js
// Top-level app orchestrator.

import {
  DEFAULT_GRID_KEY,
  GRID_PRESETS,
  APP_DEFAULTS,
  BODY_TYPES,
  MONITOR_MODES,
  PARTICLE_DEFAULTS,
} from "./config.js";

import {
  createSimState,
  setGeometryPackage,
  markUniformsDirty,
  setRunning,
  setStepsPerFrame,
} from "./simState.js";

import { buildGeometryPackage } from "./geometry.js";
import { createUserDrawController } from "./userDraw.js";
import { computeReBounds } from "./physics/boundaryModel.js";


import { initWebGPUContext } from "./gpu/gpuContext.js";
import { createGpuPipelines } from "./gpu/gpuPipelines.js";
import {
  initGpuSim,
  resetGpuSim,
  applyGeometryUpload,
  dispatchSimFrame,
  dispatchParticleStep,
  dispatchReopenNodes,
  readbackSimDiagnostics,
} from "./gpu/gpuSim.js";

import {
  destroySimGpuBuffers,
  uploadSimSolidMask,
} from "./gpu/gpuBuffers.js";


import { BodyRenderer } from "./render/bodyRenderer.js";
import { createMonitorRenderer } from "./render/monitorRenderer.js";
import { createFieldRenderer } from "./render/fieldRenderer.js";
import { createParticleRenderer } from "./render/particles.js";

import { createUIController } from "./uiController.js";


import { packSimUniforms } from "./gpu/gpuBuffers.js";



async function loadShaderCode() {
  const res = await fetch("./src/gpu/shaders.wgsl");
  if (!res.ok) {
    throw new Error(`Failed to load shader: ${res.status} ${res.statusText}`);
  }
  return await res.text();
}

// ---------------------------------------------------------
// App state
// ---------------------------------------------------------
const app = {
  gpuCtx: null,
  pipelines: null,
  fieldRenderer: null,

  sim1: null,
  sim2: null,

  compareMode: APP_DEFAULTS.compareMode,
  convergenceVisible: APP_DEFAULTS.convergenceVisible,
  sandboxActive: APP_DEFAULTS.sandboxActive,
  sandboxMode: APP_DEFAULTS.sandboxMode,
  smokeVisible: APP_DEFAULTS.smokeVisible,

  target1Active: APP_DEFAULTS.target1Active,
  target2Active: APP_DEFAULTS.target2Active,
  lastTargetedFlow: APP_DEFAULTS.lastTargetedFlow,

  currentGridKey: DEFAULT_GRID_KEY,
  monitorMode: MONITOR_MODES.CONVERGENCE,

  ui: null,

  fps: 0,
  lastFrameTime: performance.now(),
}; 

// ---------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------
async function main() {
  const shaderCode = await loadShaderCode();


  app.gpuCtx = await initWebGPUContext();
app.pipelines = createGpuPipelines(app.gpuCtx, shaderCode);

// TEMP: verify shader compiled
app.gpuCtx.device.pushErrorScope("validation");
app.gpuCtx.device.pushErrorScope("internal");
// small dummy dispatch to force pipeline compilation
const testEncoder = app.gpuCtx.device.createCommandEncoder();
const testPass = testEncoder.beginComputePass();
testPass.setPipeline(app.pipelines.compute.solverA);
// don't dispatch, just end
testPass.end();
app.gpuCtx.queue.submit([testEncoder.finish()]);
const internalErr = await app.gpuCtx.device.popErrorScope();
const validationErr = await app.gpuCtx.device.popErrorScope();
console.log("Pipeline internal error:", internalErr);
console.log("Pipeline validation error:", validationErr);

app.gpuCtx.device.addEventListener("uncapturederror", (event) => {
  console.error("UNCAPTURED GPU ERROR:", event.error.message);
});





app.fieldRenderer = createFieldRenderer(app.gpuCtx, app.pipelines);

  app.sim1 = createBaseSim("sim1", BODY_TYPES.AIRFOIL);
  buildAndAttachGeometry(app.sim1);
  setupRenderControllers(app.sim1, 1);
  initSimGpu(app.sim1);

// In main.js, after initGpuSim, add once:
const uniformData = packSimUniforms(app.sim1);
const uview = new DataView(uniformData);
console.log("JS-side uniforms:");
console.log("  NX=", uview.getUint32(0, true), "NY=", uview.getUint32(4, true));
console.log("  u0=", uview.getFloat32(16, true), "Re=", uview.getFloat32(20, true));
console.log("  charLen=", uview.getFloat32(24, true), "nu=", uview.getFloat32(28, true));
console.log("  omega=", uview.getFloat32(32, true), "tau=", uview.getFloat32(36, true));
console.log("  isLidDriven=", uview.getUint32(72, true));
console.log("  inletU=", uview.getFloat32(116, true));
  
  app.ui = createUIController(buildUiCallbacks());
  app.ui.sync(buildUiSnapshot());

  requestAnimationFrame(tick);
}

main().catch(err => {
  console.error(err);
  const errorEl = document.getElementById("app-error");
  if (errorEl) errorEl.textContent = err?.message ?? String(err);
});

// ---------------------------------------------------------
// Sim creation / setup
// ---------------------------------------------------------
function createBaseSim(id, bodyType) {
  return createSimState({
    id,
    gridKey: app.currentGridKey,
    bodyType,
  });
}

function buildAndAttachGeometry(sim) {
  const geometryPackage = buildGeometryPackage(sim);
  setGeometryPackage(sim, geometryPackage);
}

function setupRenderControllers(sim, index) {
  const flowCanvas = byId(`flow-canvas-${index}`);
  const bodyCanvas = byId(`body-canvas-${index}`);
  const wallCanvas = byId(`wall-canvas-${index}`);
  const particlesCanvas = byId(`particles-canvas-${index}`);
  const monitorCanvas = byId(`conv-canvas-${index}`);

  sim.render.flowCanvas = flowCanvas;
  sim.render.bodyCanvas = bodyCanvas;
  sim.render.wallCanvas = wallCanvas;
  sim.render.particleCanvas = particlesCanvas;
  sim.render.monitorCanvas = monitorCanvas;

  app.fieldRenderer.attachCanvas(sim, flowCanvas);

  sim.render.bodyRenderer = new BodyRenderer(bodyCanvas, sim.dims.NX, sim.dims.NY);

  sim.render.monitorRenderer = createMonitorRenderer(sim);
  sim.render.monitorRenderer.attachCanvas(monitorCanvas);
  sim.render.monitorRenderer.setVisible(app.convergenceVisible);
  sim.render.monitorRenderer.setMode(app.monitorMode);

  sim.render.particleRenderer = createParticleRenderer(sim, {
    count: app.compareMode ? PARTICLE_DEFAULTS.countCompare : PARTICLE_DEFAULTS.countSingle,
  });
sim.render.particleRenderer.attachCanvas(particlesCanvas, app.gpuCtx);
  sim.render.particleRenderer.setEnabled(app.smokeVisible);

  sim.render.bodyRenderer.draw(sim.geometry.meta);

 sim.userDraw = createUserDrawController(sim, wallCanvas, {
    isEnabled: () => app.sandboxActive,

    onGeometryEdited: (sim, { dirtyNodes = [] } = {}) => {
      sim.userDraw.redrawWalls();

      const solid = sim.geometry?.arrays?.solid;
      if (!solid || dirtyNodes.length === 0) return;

      for (const n of dirtyNodes) {
        solid[n] = 1;
      }

      uploadSimSolidMask(app.gpuCtx, sim);
    },

    onGeometryCommit: (sim, { mode, clearedNodes = [] } = {}) => {
      if (mode === "clear") {
        const solid = sim.geometry?.arrays?.solid;

        if (solid && clearedNodes.length > 0) {
          for (const n of clearedNodes) {
            solid[n] = 0;
          }

          uploadSimSolidMask(app.gpuCtx, sim);
          dispatchReopenNodes(app.gpuCtx, app.pipelines, sim, clearedNodes);
        }

        sim.userDraw.redrawWalls();

        sim.geometry.dirtyFlags.geometryDirty = false;
        sim.geometry.dirtyFlags.gpuUploadRequired = false;
        sim.gpu.state.geometryDirty = false;

        return;
      }

      // Normal stroke end:
      // live solid updates already happened during drag, so do not rebuild/reset.
      sim.geometry.dirtyFlags.geometryDirty = false;
      sim.geometry.dirtyFlags.gpuUploadRequired = false;
      sim.gpu.state.geometryDirty = false;
    },
  });

  sim.userDraw.attachOverlayCanvas(wallCanvas);
  sim.userDraw.attachInputCanvas(flowCanvas);
  sim.userDraw.bindPointerEvents(flowCanvas);
  updateCanvasCursors();
}

function initSimGpu(sim) {
  initGpuSim(app.gpuCtx, app.pipelines, sim, {
    particleCount: sim.render.particleRenderer?.getCount?.() ?? 0,
  });
}

function rebuildSimGeometry(sim) {
  const oldSolid = sim.geometry?.arrays?.solid
    ? new Uint8Array(sim.geometry.arrays.solid)
    : null;

  buildAndAttachGeometry(sim);
  clampSimReynoldsToBounds(sim);


  sim.render.bodyRenderer?.clear();
  sim.render.bodyRenderer?.draw(sim.geometry.meta);
  if (sim.userDraw) sim.userDraw.redrawWalls();

  applyGeometryUpload(app.gpuCtx, app.pipelines, sim);

  if (oldSolid && sim.geometry?.arrays?.solid) {
    const newSolid = sim.geometry.arrays.solid;
    const reopen = [];
    const closed = [];

    for (let n = 0; n < newSolid.length; n++) {
      if (oldSolid[n] !== 0 && newSolid[n] === 0) reopen.push(n);
      if (oldSolid[n] === 0 && newSolid[n] !== 0) closed.push(n);
    }

    // Zero out fIn and fOut for newly closed nodes so neighbors
    // don't pull stale fluid distributions during streaming
    if (closed.length > 0) {
    const { queue } = app.gpuCtx;
    // Must match what uploadInitialFields writes for solid nodes:
    // feq(i, rho=1, ux=0, uy=0) = W[i] * 1.0 = W[i]
    const restEq = new Float32Array([
        4/9,
        1/9, 1/9, 1/9, 1/9,
        1/36, 1/36, 1/36, 1/36,
    ]);
    for (const n of closed) {
        const byteOffset = n * 9 * 4;
        queue.writeBuffer(sim.gpu.buffers.fIn,  byteOffset, restEq);
        queue.writeBuffer(sim.gpu.buffers.fOut, byteOffset, restEq);
    }
    }

    if (reopen.length > 0) {
      dispatchReopenNodes(app.gpuCtx, app.pipelines, sim, reopen);
    }
  }

  sim.gpu.state.geometryDirty = false;
  sim.geometry.dirtyFlags.geometryDirty = false;
  sim.geometry.dirtyFlags.gpuUploadRequired = false;
}

function destroySim(sim) {
  if (!sim) return;
  if (sim.gpu?.readback) {
    sim.gpu.readback.pending = false;
    sim.gpu.readback.mapped = false;
  }
  destroySimGpuBuffers(sim);
}


// ---------------------------------------------------------
// Tick
// ---------------------------------------------------------
async function tick(now) {
  const dt = now - app.lastFrameTime;
  app.lastFrameTime = now;
  app.fps = dt > 0 ? 1000 / dt : 60;

  const sims = activeSims();

  for (const sim of sims) {
    if (sim.runtime.running) {
         dispatchSimFrame(app.gpuCtx, app.pipelines, sim, {
        stepCount: sim.runtime.stepsPerFrame,
      });

      sim.runtime.frameCount += 1;

      await readbackSimDiagnostics(app.gpuCtx, sim);
    }

    sim.render.monitorRenderer?.maybeSample();
    sim.render.monitorRenderer?.draw();

    app.fieldRenderer.render(sim);
    sim.render.particleRenderer?.render(app.gpuCtx, app.pipelines);
  }

  app.ui?.sync(buildUiSnapshot());
  requestAnimationFrame(tick);
}


// ---------------------------------------------------------
// UI callbacks
// ---------------------------------------------------------
function buildUiCallbacks() {
  return {
    setCompareMode(enabled) {
      if (!!enabled === app.compareMode) return;
      app.compareMode = !!enabled;

      if (app.compareMode) {
        if (!app.sim2) {
          app.sim2 = createBaseSim("sim2", BODY_TYPES.NACA2412);
          buildAndAttachGeometry(app.sim2);
          setupRenderControllers(app.sim2, 2);
          initSimGpu(app.sim2);
        }
      } else {
        destroySim(app.sim2);
        app.sim2 = null;
      }

      app.ui.sync(buildUiSnapshot());
    },

    setTargetSelection(state) {
      app.target1Active = state.target1Active;
      app.target2Active = state.target2Active;
      app.lastTargetedFlow = state.lastTargetedFlow;
      app.ui.sync(buildUiSnapshot());
    },

    setBodyType(targetMode, bodyType) {
      forEachTarget(targetMode, sim => {
        sim.params.bodyType = bodyType;
        rebuildSimGeometry(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setAoA(targetMode, aoaDeg) {
      forEachTarget(targetMode, sim => {
        sim.params.aoaDeg = aoaDeg;
        rebuildSimGeometry(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setNozzleDiameter(targetMode, diameter) {
      forEachTarget(targetMode, sim => {
        sim.params.nozzleDiameter = diameter;
        if (sim.params.bodyType === BODY_TYPES.NOZZLE) rebuildSimGeometry(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setCylinderDiameter(targetMode, diameter) {
      forEachTarget(targetMode, sim => {
        sim.params.cylinderDiameter = diameter;
        if (sim.params.bodyType === BODY_TYPES.CYLINDER) rebuildSimGeometry(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setCavityAspect(targetMode, aspect) {
      forEachTarget(targetMode, sim => {
        sim.params.cavityAspect = aspect;
        if (sim.params.bodyType === BODY_TYPES.LID_DRIVEN_CAVITY) rebuildSimGeometry(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

   setInletVelocity(targetMode, u0) {
      forEachTarget(targetMode, sim => {
        sim.params.u0 = u0;
        clampSimReynoldsToBounds(sim);
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setReynolds(targetMode, reynolds) {
      forEachTarget(targetMode, sim => {
        sim.params.reynolds = reynolds;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setPulseEnabled(targetMode, enabled) {
      forEachTarget(targetMode, sim => {
        sim.flow.pulsationOn = !!enabled;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setPulseAmp(targetMode, amp) {
      forEachTarget(targetMode, sim => {
        sim.flow.pulsationAmp = amp;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setPulseFreq(targetMode, freq) {
      forEachTarget(targetMode, sim => {
        sim.flow.pulsationFreq = freq;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setCollisionModel(targetMode, model) {
      forEachTarget(targetMode, sim => {
        sim.physics.collisionModel = model;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setLesModel(targetMode, model) {
      forEachTarget(targetMode, sim => {
        sim.physics.lesModel = sim.physics.lesModel === model ? "off" : model;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    },

    setWallModel(targetMode, enabled) {
      forEachTarget(targetMode, sim => {
        sim.physics.wallModel = !!enabled;
        markUniformsDirty(sim);
      });
      app.ui.sync(buildUiSnapshot());
    }, 

   setDisplayMode(mode) {
      for (const sim of activeSims()) {
        sim.display.mode = mode;
        markUniformsDirty(sim);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setColorScale(scale) {
      for (const sim of activeSims()) {
        sim.display.colorScale = scale;
        markUniformsDirty(sim);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setVectorsVisible(enabled) {
      for (const sim of activeSims()) {
        sim.display.showVectors = !!enabled;
        markUniformsDirty(sim);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setGridVisible(enabled) {
    for (const sim of activeSims()) {
        sim.display.showGrid = !!enabled;
        markUniformsDirty(sim);
    }
    app.ui.sync(buildUiSnapshot());
    },

    setRuntimeRunning(running) {
      for (const sim of activeSims()) {
        setRunning(sim, running);
      }
      app.ui.sync(buildUiSnapshot());
    },

   resetTarget(targetMode) {
      forEachTarget(targetMode, sim => {
        sim.geometry.paintedNodes.clear();

        if (sim.userDraw) {
          sim.userDraw.state.strokes.length = 0;
          sim.userDraw.state.currentStroke = null;
          sim.userDraw.state.lastLX = null;
          sim.userDraw.state.lastLY = null;
          sim.userDraw.state.isDrawing = false;
          sim.userDraw.redrawWalls();
        }

        rebuildSimGeometry(sim);

        resetGpuSim(app.gpuCtx, app.pipelines, sim, {
          particleCount: sim.render.particleRenderer?.getCount?.() ?? 0,
        });

        sim.render.monitorRenderer?.reset();
      });
      app.ui.sync(buildUiSnapshot());
    },

    clearSandbox(targetMode) {
      forEachTarget(targetMode, sim => {
        sim.userDraw?.clearDrawing();
      });
      app.ui.sync(buildUiSnapshot());
    },

    setFidelity(gridKey) {
      if (!GRID_PRESETS[gridKey]) return;
      if (gridKey === app.currentGridKey) return;

      app.currentGridKey = gridKey;

      const oldSim1 = app.sim1;
      const oldSim2 = app.sim2;

      app.sim1 = recreateSimFromExisting(oldSim1, "sim1");
      buildAndAttachGeometry(app.sim1);
      setupRenderControllers(app.sim1, 1);
      initSimGpu(app.sim1);

      if (app.compareMode && oldSim2) {
        app.sim2 = recreateSimFromExisting(oldSim2, "sim2");
        buildAndAttachGeometry(app.sim2);
        setupRenderControllers(app.sim2, 2);
        initSimGpu(app.sim2);
      } else {
        app.sim2 = null;
      }

      destroySim(oldSim1);
      destroySim(oldSim2);

      app.ui.sync(buildUiSnapshot());
    },

    setSandboxEnabled(enabled) {
      app.sandboxActive = !!enabled;
      updateCanvasCursors();
      app.ui.sync(buildUiSnapshot());
    },

    setSmokeVisible(enabled) {
      app.smokeVisible = !!enabled;
      for (const sim of activeSims()) {
        sim.display.smokeVisible = !!enabled;
        sim.render.particleRenderer?.setEnabled(!!enabled);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setMonitorVisible(enabled) {
      app.convergenceVisible = !!enabled;
      for (const sim of activeSims()) {
        sim.render.monitorRenderer?.setVisible(!!enabled);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setMonitorMode(mode) {
      app.monitorMode = mode;
      for (const sim of activeSims()) {
        sim.render.monitorRenderer?.setMode(mode);
      }
      app.ui.sync(buildUiSnapshot());
    },

    setStepsPerFrame(steps) {
      for (const sim of activeSims()) {
        setStepsPerFrame(sim, steps);
      }
      app.ui.sync(buildUiSnapshot());
    },
  };
}

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------

function clampSimReynoldsToBounds(sim) {
  const L =
    sim.geometry?.meta?.charLength ??
    sim.derived?.charLength ??
    1.0;

  const { reMin, reMax } = computeReBounds(sim.params.u0, L);
  sim.params.reynolds = Math.min(reMax, Math.max(reMin, sim.params.reynolds));
}

function activeSims() {
  const sims = [];
  if (app.sim1) sims.push(app.sim1);
  if (app.compareMode && app.sim2) sims.push(app.sim2);
  return sims;
}

function forEachTarget(targetMode, fn) {
  if (!app.compareMode) {
    if (app.sim1) fn(app.sim1);
    return;
  }

  if (targetMode === "flow1") {
    if (app.sim1) fn(app.sim1);
    return;
  }

  if (targetMode === "flow2") {
    if (app.sim2) fn(app.sim2);
    return;
  }

  if (app.sim1) fn(app.sim1);
  if (app.sim2) fn(app.sim2);
}

function recreateSimFromExisting(oldSim, id) {
 const sim = createSimState({
    id,
    gridKey: app.currentGridKey,

    bodyType: oldSim.params.bodyType,
    aoaDeg: oldSim.params.aoaDeg,
    nozzleDiameter: oldSim.params.nozzleDiameter,
    cylinderDiameter: oldSim.params.cylinderDiameter,
    cavityAspect: oldSim.params.cavityAspect,

    u0: oldSim.params.u0,
    reynolds: oldSim.params.reynolds,
    uCoflow: oldSim.params.uCoflow,

    pulsationOn: oldSim.flow.pulsationOn,
    pulsationAmp: oldSim.flow.pulsationAmp,
    pulsationFreq: oldSim.flow.pulsationFreq,

    collisionModel: oldSim.physics.collisionModel,
    lesModel: oldSim.physics.lesModel,
    wallModel: oldSim.physics.wallModel,

    displayMode: oldSim.display.mode,
    colorScale: oldSim.display.colorScale,
    showVectors: oldSim.display.showVectors,
    showGrid: oldSim.display.showGrid,
    gridFine: oldSim.display.gridFine,
    showDist: oldSim.display.showDist,
    smokeVisible: oldSim.display.smokeVisible,

    running: oldSim.runtime.running,
    stepsPerFrame: oldSim.runtime.stepsPerFrame,
  });

  clampSimReynoldsToBounds(sim);

  sim.geometry.paintedNodes = new Set(oldSim.geometry.paintedNodes);
  return sim;
}


function buildUiSnapshot() {
  return {
    compareMode: app.compareMode,
    convergenceVisible: app.convergenceVisible,
    sandboxActive: app.sandboxActive,
    target1Active: app.target1Active,
    target2Active: app.target2Active,
    lastTargetedFlow: app.lastTargetedFlow,
    monitorMode: app.monitorMode,
    fps: app.fps,
    sim1: app.sim1,
    sim2: app.sim2,
  };
}

function updateCanvasCursors() {
  for (const sim of activeSims()) {
    if (sim.render.flowCanvas) {
      sim.render.flowCanvas.style.cursor = app.sandboxActive ? "crosshair" : "default";
    }
  }
}

function byId(id) {
  return document.getElementById(id);
} 
