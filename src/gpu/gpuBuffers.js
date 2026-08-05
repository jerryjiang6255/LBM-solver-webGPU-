// src/gpu/gpuBuffers.js
// GPU buffer creation, upload, readback, and layout authority.
// No pipelines or command submission orchestration here.

import { alignTo, createReadbackBuffer, destroyBufferSafe } from "./gpuContext.js";
import { Q, RHO0, feqAt } from "../physics/lattice.js";
import { buildBoundaryUniforms } from "../physics/boundaryModel.js";
import { updateDerivedSimParams, resetSimRuntimeState } from "../simState.js";

const FLOAT_SIZE = 4;
const UINT_SIZE = 4;
const MACRO_STRIDE_FLOATS = 4; // rho, ux, uy, aux
const PARTICLE_STRIDE_FLOATS = 4; // x, y, vx, vy

// ---------------------------------------------------------
// Public API
// ---------------------------------------------------------
export function createSimGpuBuffers(gpuCtx, sim, options = {}) {
  const { device } = gpuCtx;
  const { N } = sim.dims;

  destroySimGpuBuffers(sim);

  const fByteSize = N * Q * FLOAT_SIZE;
  const macroByteSize = N * MACRO_STRIDE_FLOATS * FLOAT_SIZE;

  const particleCount = options.particleCount ?? 0;
  const particleByteSize = particleCount * PARTICLE_STRIDE_FLOATS * FLOAT_SIZE;

  sim.gpu.buffers = {
    fIn: createStorageBuffer(device, fByteSize, "sim-fIn"),
    fOut: createStorageBuffer(device, fByteSize, "sim-fOut"),
    macro: createStorageBuffer(device, macroByteSize, "sim-macro"),
    prevMacro: createStorageBuffer(device, macroByteSize, "sim-prevMacro"),

    solid: null,
    sdf: null,
    wallDist: null,
    wallAdjacent: null,
    linkMask: null,
    linkQ: null,
    solidList: null,
    fluidSurfaceList: null,

    uniforms: createUniformBuffer(device, getUniformByteSize(), "sim-uniforms"),

       force: createStorageBuffer(device, 16, "sim-forceAccum"),
    reduction: createStorageBuffer(device, 16, "sim-reduction"),

    particles: particleCount > 0
      ? createStorageBuffer(device, particleByteSize, "sim-particles")
      : null,

    reopenNodes: createStorageBuffer(device, UINT_SIZE, "sim-reopenNodes"),

    particleRender: null,

    readback: {
      force: createReadbackBuffer(gpuCtx, 16, "sim-force-readback"),
      reduction: createReadbackBuffer(gpuCtx, 16, "sim-reduction-readback"),
    },
  };

  sim.gpu.state.initialized = false;
  sim.gpu.state.activePopulationIsIn = true;
  sim.gpu.state.uniformsDirty = true;
  sim.gpu.state.geometryDirty = true;
  sim.gpu.state.resizeDirty = true;

  return sim.gpu.buffers;
}

export function destroySimGpuBuffers(sim) {
  const buffers = sim?.gpu?.buffers;
  if (!buffers) return;

  for (const key of Object.keys(buffers)) {
    const value = buffers[key];

    if (value && typeof value === "object" && !("destroy" in value)) {
      for (const nestedKey of Object.keys(value)) {
        destroyBufferSafe(value[nestedKey]);
      }
      continue;
    }

    destroyBufferSafe(value);
  }

  sim.gpu.buffers = {
    fIn: null,
    fOut: null,
    macro: null,
    prevMacro: null,
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
    reopenNodes: null,
    particleRender: null,
    readback: {
      force: null,
      reduction: null,
    },
  };
}

export function uploadInitialFields(gpuCtx, sim) {
  const { queue } = gpuCtx;
  const { N } = sim.dims;

  updateDerivedSimParams(sim);
  resetSimRuntimeState(sim);

  const fInit = new Float32Array(N * Q);
  const macroInit = new Float32Array(N * MACRO_STRIDE_FLOATS);

  const bodyType = sim.params.bodyType;
  const isLidDriven = bodyType === "lid-driven-cavity";
  const solid = sim.geometry.arrays?.solid;

  for (let n = 0; n < N; n++) {
    const isSolid = solid ? solid[n] !== 0 : false;

    let rho = RHO0;
    let ux = 0.0;
    let uy = 0.0;

    if (!isSolid && !isLidDriven) {
      ux = sim.params.u0;
    }

    macroInit[n * 4 + 0] = rho;
    macroInit[n * 4 + 1] = ux;
    macroInit[n * 4 + 2] = uy;
    macroInit[n * 4 + 3] = 0.0;

    const base = n * Q;
    for (let i = 0; i < Q; i++) {
      fInit[base + i] = feqAt(i, rho, ux, uy);
    }
  }

  queue.writeBuffer(sim.gpu.buffers.fIn, 0, fInit);
  queue.writeBuffer(sim.gpu.buffers.fOut, 0, fInit);
  queue.writeBuffer(sim.gpu.buffers.macro, 0, macroInit);
  queue.writeBuffer(sim.gpu.buffers.prevMacro, 0, macroInit);

  zeroForceBuffer(gpuCtx, sim);
  zeroReductionBuffer(gpuCtx, sim);

  sim.gpu.state.activePopulationIsIn = true;
  sim.gpu.state.initialized = true;
}

export function uploadSimGeometry(gpuCtx, sim, geometryPackage = sim.geometry) {
  const { device, queue } = gpuCtx;
  const arrays = geometryPackage?.arrays;

  if (!arrays) {
    throw new Error("uploadSimGeometry: missing geometry arrays.");
  }

  replaceGeometryBuffer(device, queue, sim, "solid",            toUint32Array(arrays.solid),            "geom-solid");
  replaceGeometryBuffer(device, queue, sim, "sdf",              arrays.sdf,                             "geom-sdf");
  replaceGeometryBuffer(device, queue, sim, "wallDist",         arrays.wallDist,                        "geom-wallDist");
  replaceGeometryBuffer(device, queue, sim, "wallAdjacent",     toUint32Array(arrays.wallAdjacent),     "geom-wallAdjacent");
  replaceGeometryBuffer(device, queue, sim, "linkMask",         toUint32Array(arrays.linkMask),         "geom-linkMask");
  replaceGeometryBuffer(device, queue, sim, "linkQ",            arrays.linkQ,                           "geom-linkQ");
  replaceGeometryBuffer(device, queue, sim, "solidList",        arrays.solidList ?? new Uint32Array(1), "geom-solidList");
  replaceGeometryBuffer(device, queue, sim, "fluidSurfaceList", arrays.fluidSurfaceList ?? new Uint32Array(1), "geom-fluidSurfaceList");

  sim.gpu.state.geometryDirty = false;
  sim.geometry.dirtyFlags.gpuUploadRequired = false;
}

export function uploadSimSolidMask(gpuCtx, sim) {
  const solid = sim.geometry?.arrays?.solid;
  const solidBuffer = sim.gpu?.buffers?.solid;

  if (!solid) {
    throw new Error("uploadSimSolidMask: missing sim.geometry.arrays.solid.");
  }

  if (!solidBuffer) {
    throw new Error("uploadSimSolidMask: missing sim.gpu.buffers.solid.");
  }

  gpuCtx.queue.writeBuffer(solidBuffer, 0, toUint32Array(solid));
}

export function uploadReopenNodeList(gpuCtx, sim, nodeIndices) {
  const { device, queue } = gpuCtx;

  const nodes = nodeIndices instanceof Uint32Array
    ? nodeIndices
    : new Uint32Array(nodeIndices ?? []);

  const byteSize = Math.max(UINT_SIZE, nodes.byteLength);

  if (!sim.gpu.buffers.reopenNodes || sim.gpu.buffers.reopenNodes.size < byteSize) {
    destroyBufferSafe(sim.gpu.buffers.reopenNodes);
    sim.gpu.buffers.reopenNodes = device.createBuffer({
      label: "sim-reopenNodes",
      size: alignTo(byteSize, 4),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
  }

  if (nodes.length > 0) {
    queue.writeBuffer(sim.gpu.buffers.reopenNodes, 0, nodes);
  }
}


export function uploadSimParams(gpuCtx, sim) {
  const uniformData = packSimUniforms(sim);
  gpuCtx.queue.writeBuffer(sim.gpu.buffers.uniforms, 0, uniformData);
  sim.gpu.state.uniformsDirty = false;
}

export function uploadParticleState(gpuCtx, sim, particleData) {
  if (!sim.gpu.buffers.particles) return;
  gpuCtx.queue.writeBuffer(sim.gpu.buffers.particles, 0, particleData);
}

export function zeroForceBuffer(gpuCtx, sim) {
  const zeros = new Int32Array(4);
  gpuCtx.queue.writeBuffer(sim.gpu.buffers.force, 0, zeros);
}

export function zeroReductionBuffer(gpuCtx, sim) {
  const zeros = new Int32Array(4);
  gpuCtx.queue.writeBuffer(sim.gpu.buffers.reduction, 0, zeros);
}

export function copyCurrentMacroToPrevious(gpuCtx, sim, encoder) {
  encoder.copyBufferToBuffer(
    sim.gpu.buffers.macro,
    0,
    sim.gpu.buffers.prevMacro,
    0,
    sim.dims.N * MACRO_STRIDE_FLOATS * FLOAT_SIZE
  );
}

export function getActivePopulationBuffers(sim) {
  const { fIn, fOut } = sim.gpu.buffers;
  return sim.gpu.state.activePopulationIsIn
    ? { src: fIn, dst: fOut }
    : { src: fOut, dst: fIn };
}

export function swapPopulationState(sim) {
  sim.gpu.state.activePopulationIsIn = !sim.gpu.state.activePopulationIsIn;
}

export function packSimUniforms(sim) {
  updateDerivedSimParams(sim);

  const b = buildBoundaryUniforms(sim);
  const d = sim.derived;
  const { NX, NY, N } = sim.dims;

const renderWidth = sim.render?.flowCanvas?.width ?? sim.dims.NX;
const renderHeight = sim.render?.flowCanvas?.height ?? sim.dims.NY;

  const collisionModel = sim.physics.collisionModel === "bgk" ? 1 : 0;
  const lesModel =
    sim.physics.lesModel === "off" ? 0 :
    sim.physics.lesModel === "smagorinsky" ? 1 : 2;

  const displayMode = sim.display.mode === "qcrit" ? 1 : 0;

  const buffer = new ArrayBuffer(getUniformByteSize());
  const view = new DataView(buffer);
  let o = 0;

  // slot 0
  view.setUint32(o, NX, true); o += UINT_SIZE;
  view.setUint32(o, NY, true); o += UINT_SIZE;
  view.setUint32(o, N, true); o += UINT_SIZE;
  view.setUint32(o, sim.runtime.stepCount, true); o += UINT_SIZE;

  // slot 1
  view.setFloat32(o, sim.params.u0, true); o += FLOAT_SIZE;
  view.setFloat32(o, sim.params.reynolds, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.charLength, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.nu, true); o += FLOAT_SIZE;

  // slot 2
  view.setFloat32(o, d.omega, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.tau, true); o += FLOAT_SIZE;
  view.setFloat32(o, sim.display.colorScale, true); o += FLOAT_SIZE;
  view.setUint32(o, displayMode, true); o += UINT_SIZE;

 // slot 3
  view.setUint32(o, collisionModel, true); o += UINT_SIZE;
  view.setUint32(o, lesModel, true); o += UINT_SIZE;
  view.setUint32(o, sim.physics.wallModel ? 1 : 0, true); o += UINT_SIZE;

  const displayFlags =
    (sim.display.showVectors ? 1 : 0) |
    (sim.display.showGrid ? 2 : 0) |
    (sim.display.showDist ? 4 : 0);

  view.setUint32(o, displayFlags, true); o += UINT_SIZE;



  // slot 4
  view.setUint32(o, sim.flow.pulsationOn ? 1 : 0, true); o += UINT_SIZE;
  view.setUint32(o, b.nozzleActive, true); o += UINT_SIZE;
  view.setUint32(o, b.isLidDriven, true); o += UINT_SIZE;
  view.setUint32(o, 0, true); o += UINT_SIZE;

  // slot 5
  view.setFloat32(o, sim.flow.pulsationAmp, true); o += FLOAT_SIZE;
  view.setFloat32(o, sim.flow.pulsationFreq, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.uCoflow, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.nozzleCy, true); o += FLOAT_SIZE;

  // slot 6
  view.setFloat32(o, b.nozzleHalfD, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.spongeStartX, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.spongeWidth, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.spongeStrength, true); o += FLOAT_SIZE;

  // slot 7
  view.setFloat32(o, b.lidU, true); o += FLOAT_SIZE;
  view.setFloat32(o, b.inletU, true); o += FLOAT_SIZE;
  view.setFloat32(o, sim.constants.rho0, true); o += FLOAT_SIZE;
  view.setFloat32(o, 0, true); o += FLOAT_SIZE;

  // slot 8
  view.setFloat32(o, d.forceRef.x, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.forceRef.y, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.forceScale, true); o += FLOAT_SIZE;
  view.setFloat32(o, d.momentScale, true); o += FLOAT_SIZE;

 // slot 9
  view.setFloat32(o, renderWidth, true); o += FLOAT_SIZE;
  view.setFloat32(o, renderHeight, true); o += FLOAT_SIZE;
  view.setFloat32(o, 0, true); o += FLOAT_SIZE;
  view.setFloat32(o, 0, true); o += FLOAT_SIZE;



  return buffer;
}

export async function readForceReadback(gpuCtx, sim) {
  const data = await mapReadbackInt32(sim.gpu.buffers.readback.force);
  if (!data) return null;
  return {
    fx: data[0] / 1e6,
    fy: data[1] / 1e6,
    mz: data[2] / 1e6,
  };
}

export async function readReductionReadback(gpuCtx, sim) {
  const data = await mapReadbackInt32(sim.gpu.buffers.readback.reduction);
  if (!data) return null;
  const fluidCount = data[3] >>> 0;
  return {
    meanSpeed: fluidCount > 0 ? (data[0] / 1e6) / fluidCount : 0,
    rmsDeltaU: fluidCount > 0 ? (data[1] / 1e6) / fluidCount : 0,
    meanRhoDev: fluidCount > 0 ? (data[2] / 1e6) / fluidCount : 0,
    fluidCount,
  };
}

// ---------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------
function createStorageBuffer(device, size, label) {
  return device.createBuffer({
    label,
    size: alignTo(size, 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
}

function createUniformBuffer(device, size, label) {
  return device.createBuffer({
    label,
    size: alignTo(size, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function replaceGeometryBuffer(device, queue, sim, key, typedArray, label) {
  destroyBufferSafe(sim.gpu.buffers[key]);

  const buf = device.createBuffer({
    label,
    size: alignTo(Math.max(typedArray.byteLength, 4), 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  queue.writeBuffer(buf, 0, typedArray);
  sim.gpu.buffers[key] = buf;
}

function toUint32Array(arrayLike) {
  if (arrayLike instanceof Uint32Array) return arrayLike;
  const out = new Uint32Array(arrayLike.length);
  for (let i = 0; i < arrayLike.length; i++) out[i] = arrayLike[i];
  return out;
}

function getUniformFloatCount() {
  return 11 * 4;
}

function getUniformByteSize() {
  return getUniformFloatCount() * FLOAT_SIZE;
}

async function mapReadbackInt32(buffer) {
  try {
    await buffer.mapAsync(GPUMapMode.READ);
  } catch (e) {
    if (e.name === "AbortError") return null;
    throw e;
  }
  const copy = new Int32Array(buffer.getMappedRange().slice(0));
  buffer.unmap();
  return copy;
}



export async function debugReadbackMacro(gpuCtx, sim) {
  console.log("readback: macro label=", sim.gpu.buffers.macro?.label, "size=", sim.gpu.buffers.macro?.size);
  const { device, queue } = gpuCtx;
  const N = sim.dims.N;
  const byteSize = N * 4 * 4; // N * 4 floats * 4 bytes

  const readBuf = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sim.gpu.buffers.macro, 0, readBuf, 0, byteSize);
  queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();

  // Sample a few interior fluid nodes
  const NX = sim.dims.NX;
  const NY = sim.dims.NY;
  const samples = [
    [Math.floor(NX * 0.1), Math.floor(NY * 0.5)],
    [Math.floor(NX * 0.5), Math.floor(NY * 0.5)],
    [Math.floor(NX * 0.8), Math.floor(NY * 0.5)],
  ];

  console.log("=== MACRO READBACK ===");
  for (const [x, y] of samples) {
    const n = y * NX + x;
    const rho = data[n * 4 + 0];
    const ux  = data[n * 4 + 1];
    const uy  = data[n * 4 + 2];
    console.log(`  node(${x},${y}): rho=${rho.toFixed(4)}, ux=${ux.toFixed(4)}, uy=${uy.toFixed(4)}`);
  }

  // Also check solid buffer
  const solidByteSize = N * 4;
  const solidReadBuf = device.createBuffer({
    size: solidByteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder2 = device.createCommandEncoder();
  encoder2.copyBufferToBuffer(sim.gpu.buffers.solid, 0, solidReadBuf, 0, solidByteSize);
  queue.submit([encoder2.finish()]);

  await solidReadBuf.mapAsync(GPUMapMode.READ);
  const solidData = new Uint32Array(solidReadBuf.getMappedRange().slice(0));
  solidReadBuf.unmap();
  solidReadBuf.destroy();

  let solidCount = 0;
  for (let i = 0; i < N; i++) if (solidData[i]) solidCount++;
  console.log(`  solid nodes on GPU: ${solidCount} / ${N}`);
  console.log("======================");
}

export async function debugReadbackUniforms(gpuCtx, sim) {
  const { device, queue } = gpuCtx;
  const byteSize = 176;

  const readBuf = device.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sim.gpu.buffers.uniforms, 0, readBuf, 0, byteSize);
  queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(readBuf.getMappedRange().slice(0));
  const fdata = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();

  console.log("=== UNIFORM READBACK ===");
  console.log("NX=", data[0], "NY=", data[1], "N=", data[2], "step=", data[3]);
  console.log("u0=", fdata[4], "Re=", fdata[5], "charLen=", fdata[6], "nu=", fdata[7]);
  console.log("omega=", fdata[8], "tau=", fdata[9]);
  console.log("collisionModel=", data[11], "lesModel=", data[12], "wallModel=", data[13]);
  console.log("isLidDriven=", data[18], "nozzleActive=", data[17]);
  console.log("inletU=", fdata[29], "lidU=", fdata[28]);
  console.log("========================");
}