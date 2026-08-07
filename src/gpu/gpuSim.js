// src/gpu/gpuSim.js
// Per-sim GPU orchestration layer.
// Owns initialization, dispatch order, ping-pong state, and diagnostic readback.
// No DOM logic here.

import { createCommandEncoder, submitCommands } from "./gpuContext.js";
import {
  createSimGpuBuffers,
  uploadInitialFields,
  uploadSimGeometry,
  uploadSimParams,
  uploadParticleState,
  uploadReopenNodeList,
  zeroForceBuffer,
  zeroReductionBuffer,
  copyCurrentMacroToPrevious,
  readForceReadback,
  readReductionReadback,
} from "./gpuBuffers.js";
import { rebuildSimBindGroups, getActiveComputeBindGroup } from "./gpuPipelines.js";
import { applyForceCoefficients } from "../physics/forceModel.js";
import { PARTICLE_DEFAULTS, GPU_TUNING } from "../config.js";

// ---------------------------------------------------------
// Public API
// ---------------------------------------------------------
export function initGpuSim(gpuCtx, pipelines, sim, options = {}) {
  const particleCount = options.particleCount ?? 0;

  createSimGpuBuffers(gpuCtx, sim, { particleCount });

  if (sim.geometry?.arrays?.solid) {
    uploadSimGeometry(gpuCtx, sim, sim.geometry);
  }

  uploadSimParams(gpuCtx, sim);
  uploadInitialFields(gpuCtx, sim);

  if (particleCount > 0) {
    const particleData = buildInitialParticleData(sim, particleCount);
    uploadParticleState(gpuCtx, sim, particleData);
  }

  rebuildSimBindGroups(gpuCtx, pipelines, sim);

  sim.gpu.state.initialized = true;
  sim.gpu.state.geometryDirty = false;
  sim.gpu.state.uniformsDirty = false;

  return sim;
}

export function resetGpuSim(gpuCtx, pipelines, sim, options = {}) {
  if (sim.geometry?.arrays?.solid) {
    uploadSimGeometry(gpuCtx, sim, sim.geometry);
  }

  uploadSimParams(gpuCtx, sim);
  uploadInitialFields(gpuCtx, sim);

  const particleCount =
    options.particleCount ??
    getParticleCountFromBuffer(sim);

  if (particleCount > 0 && sim.gpu.buffers.particles) {
    const particleData = buildInitialParticleData(sim, particleCount);
    uploadParticleState(gpuCtx, sim, particleData);
  }

  rebuildSimBindGroups(gpuCtx, pipelines, sim);

  sim.gpu.state.activePopulationIsIn = true;
  sim.gpu.state.initialized = true;
}

export function applyGeometryUpload(gpuCtx, pipelines, sim) {
  if (!sim.geometry?.arrays?.solid) {
    throw new Error("applyGeometryUpload: sim geometry is missing.");
  }

  uploadSimGeometry(gpuCtx, sim, sim.geometry);
  uploadSimParams(gpuCtx, sim);
  rebuildSimBindGroups(gpuCtx, pipelines, sim);

  sim.gpu.state.geometryDirty = false;
  sim.gpu.state.uniformsDirty = false;
}

/**
 * Dispatch one full simulation frame for a sim:
 * - encodes all solver substeps into one command encoder
 * - optionally encodes diagnostics on cadence
 * - submits exactly once
 *
 * Returns:
 * {
 *   diagnosticsDispatched: boolean
 * }
 */
export function dispatchSimFrame(gpuCtx, pipelines, sim, options = {}) {
  if (!sim.gpu.state.initialized) {
    throw new Error(`dispatchSimFrame: sim '${sim.id}' is not initialized.`);
  }

  if (sim.gpu.state.geometryDirty || sim.geometry?.dirtyFlags?.gpuUploadRequired) {
    applyGeometryUpload(gpuCtx, pipelines, sim);
  }

  const stepCount = options.stepCount ?? sim.runtime.stepsPerFrame ?? 1;
  const diagnosticsEveryFrames =
    options.diagnosticsEveryFrames ?? GPU_TUNING.diagnosticsEveryFrames ?? 4;

  const currentFrame = sim.runtime.frameCount ?? 0;
  const shouldRunDiagnostics =
    diagnosticsEveryFrames > 0 &&
    (currentFrame % diagnosticsEveryFrames === 0);

    const encoder = createCommandEncoder(gpuCtx, `${sim.id}-frame-encoder`);
    const computeBindGroup = getActiveComputeBindGroup(sim);

    for (let s = 0; s < stepCount; s++) {
    sim.runtime.stepCount += 1;
    sim.gpu.state.uniformsDirty = true;
    uploadSimParams(gpuCtx, sim);
    sim.gpu.state.uniformsDirty = false;

    {
      const pass = encoder.beginComputePass({ label: `${sim.id}-solverA-pass-${s}` });
      pass.setPipeline(pipelines.compute.solverA);
      pass.setBindGroup(0, computeBindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(sim.dims.NX / GPU_TUNING.workgroupX),
        Math.ceil(sim.dims.NY / GPU_TUNING.workgroupY),
        1
      );
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({ label: `${sim.id}-solverB-pass-${s}` });
      pass.setPipeline(pipelines.compute.solverB);
      pass.setBindGroup(0, computeBindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(sim.dims.NX / GPU_TUNING.workgroupX),
        Math.ceil(sim.dims.NY / GPU_TUNING.workgroupY),
        1
      );
      pass.end();
    }
  }
 const particleCount = getParticleCountFromBuffer(sim);
  const shouldRunParticles =
    !!sim.display?.smokeVisible &&
    !!sim.gpu.buffers?.particles &&
    particleCount > 0;

  if (shouldRunParticles) {
    if (sim.gpu.state.uniformsDirty) {
      uploadSimParams(gpuCtx, sim);
      sim.gpu.state.uniformsDirty = false;
    }


    const pass = encoder.beginComputePass({ label: `${sim.id}-particle-pass` });
    pass.setPipeline(pipelines.compute.particle);
    pass.setBindGroup(0, computeBindGroup);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(particleCount / 128)), 1, 1);
    pass.end();
  }


  if (shouldRunDiagnostics) {
    if (sim.gpu.state.uniformsDirty) {
      uploadSimParams(gpuCtx, sim);
      sim.gpu.state.uniformsDirty = false;
    }

    zeroForceBuffer(gpuCtx, sim);
    zeroReductionBuffer(gpuCtx, sim);


    {
      const pass = encoder.beginComputePass({ label: `${sim.id}-force-pass` });
      pass.setPipeline(pipelines.compute.force);
      pass.setBindGroup(0, computeBindGroup);

      const count = sim.geometry?.arrays?.fluidSurfaceList?.length ?? 0;
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / 64)), 1, 1);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({ label: `${sim.id}-reduction-pass` });
      pass.setPipeline(pipelines.compute.reduction);
      pass.setBindGroup(0, computeBindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(sim.dims.N / 64)), 1, 1);
      pass.end();
    }

    encoder.copyBufferToBuffer(
      sim.gpu.buffers.force,
      0,
      sim.gpu.buffers.readback.force,
      0,
      16
    );

    encoder.copyBufferToBuffer(
      sim.gpu.buffers.reduction,
      0,
      sim.gpu.buffers.readback.reduction,
      0,
      16
    );

    // Snapshot current macro only after diagnostics have consumed prevMacro.
    copyCurrentMacroToPrevious(gpuCtx, sim, encoder);
  }

  submitCommands(gpuCtx, encoder.finish());

  if (shouldRunDiagnostics) {
    sim.gpu.readback.pending = true;
    sim.gpu.readback.lastFrameRead = currentFrame;
  }

  return {
    diagnosticsDispatched: shouldRunDiagnostics,
  };
}

export function dispatchParticleStep(gpuCtx, pipelines, sim) {
  if (!sim.gpu.state.initialized) return;
  if (!sim.gpu.buffers.particles) return;

  const count = getParticleCountFromBuffer(sim);
  if (count <= 0) return;

  if (sim.gpu.state.uniformsDirty) {
    uploadSimParams(gpuCtx, sim);
    sim.gpu.state.uniformsDirty = false;
  }

  const encoder = createCommandEncoder(gpuCtx, `${sim.id}-particle-encoder`);
  const computeBindGroup = getActiveComputeBindGroup(sim);

  const pass = encoder.beginComputePass({ label: `${sim.id}-particle-pass` });
  pass.setPipeline(pipelines.compute.particle);
  pass.setBindGroup(0, computeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / 128)), 1, 1);
  pass.end();

  submitCommands(gpuCtx, encoder.finish());
}

export function dispatchReopenNodes(gpuCtx, pipelines, sim, nodeIndices = []) {
  if (!sim.gpu.state.initialized) return;
  if (!nodeIndices || nodeIndices.length === 0) return;

  uploadReopenNodeList(gpuCtx, sim, nodeIndices);
  rebuildSimBindGroups(gpuCtx, pipelines, sim);

  if (sim.gpu.state.uniformsDirty) {
    uploadSimParams(gpuCtx, sim);
    sim.gpu.state.uniformsDirty = false;
  }

  const encoder = createCommandEncoder(gpuCtx, `${sim.id}-reopen-encoder`);
  const computeBindGroup = getActiveComputeBindGroup(sim);

  const pass = encoder.beginComputePass({ label: `${sim.id}-reopen-pass` });
  pass.setPipeline(pipelines.compute.reopen);
  pass.setBindGroup(0, computeBindGroup);
  pass.dispatchWorkgroups(Math.max(1, Math.ceil(nodeIndices.length / 64)), 1, 1);
  pass.end();

  submitCommands(gpuCtx, encoder.finish());
}

export async function readbackSimDiagnostics(gpuCtx, sim) {
  if (!sim.gpu.readback.pending) return sim.diagnostics;

  const [forceRaw, reduction] = await Promise.all([
    readForceReadback(gpuCtx, sim),
    readReductionReadback(gpuCtx, sim),
  ]);

  if (!forceRaw || !reduction) return sim.diagnostics;

  applyForceCoefficients(sim, forceRaw);

  sim.diagnostics.reductions.meanSpeed = reduction.meanSpeed;
  sim.diagnostics.reductions.rmsDeltaU = reduction.rmsDeltaU;
  sim.diagnostics.reductions.meanRhoDev = reduction.meanRhoDev;

  sim.gpu.readback.pending = false;
  return sim.diagnostics;
}

// ---------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------
function buildInitialParticleData(sim, count) {
  const data = new Float32Array(count * 4);
  const { NX, NY } = sim.dims;
  const isLidDriven = sim.params.bodyType === "lid-driven-cavity";

  const rng = mulberry32(PARTICLE_DEFAULTS.seed ^ hashString(sim.id));

  for (let i = 0; i < count; i++) {
    let x;
    let y;

    if (isLidDriven) {
      x = 2 + rng() * Math.max(1, NX - 4);
      y = 2 + rng() * Math.max(1, NY - 4);
    } else {
      x = rng() * Math.max(1, NX - 1);
      y = 1 + rng() * Math.max(1, NY - 2);
    }

    data[i * 4 + 0] = x;
    data[i * 4 + 1] = y;
    data[i * 4 + 2] = 0.0;
    data[i * 4 + 3] = 0.0;
  }

  return data;
}

function getParticleCountFromBuffer(sim) {
  const buf = sim.gpu?.buffers?.particles;
  if (!buf) return 0;
  return Math.floor(buf.size / (4 * 4));
}

function hashString(s) {
  let h = 2166136261 >>> 0;
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


