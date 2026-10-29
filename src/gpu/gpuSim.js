// src/gpu/gpuSim.js
// ============================================================
// Per-simulation GPU state: buffers, bind groups, stepping,
// diagnostics, particles, overlay gathers and async readback.
//
// The CPU-side `sim` object (simInstance.js) stays the single source
// of truth for parameters and geometry; this class mirrors it to the
// GPU and runs the solver there.
// ============================================================

import { Q, W } from "../lbm.js";
import {
  SOLVER_BINDINGS,
  MAX_STEPS_PER_FRAME,
  STEP_SLOT_BYTES,
  createBindGroup,
} from "./gpuContext.js";

export const NODE_SOLID = 1;
export const NODE_PAINTED = 2;
export const NODE_WALLADJ = 4;

const PARAMS_BYTES = 112;
const SPONGE_WIDTH = 28;
const SPONGE_STRENGTH = 0.6;
const READBACK_RING = 3;

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export class GpuSim {
  constructor(gpu, sim, { particleCount = 0 } = {}) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.queue = gpu.queue;
    this.sim = sim;

    this.NX = sim.NX;
    this.NY = sim.NY;
    this.N = sim.NX * sim.NY;
    this.particleCount = particleCount;

    this.vecCount = Math.ceil(this.NX / 8) * Math.ceil(this.NY / 8);
    this.distCount = Math.ceil(this.NX / 16) * Math.ceil(this.NY / 16);
    this.gatherFloats = this.vecCount * 4 + this.distCount * 10;

    this.reduceGroups = Math.ceil(this.N / 256);
    this.forceGroups = 1;
    this.surfCount = 0;

    this.nodeType = new Uint32Array(this.N);
    this.frameSeed = 1;
    this.generation = 0;       // bumped on reset so stale readbacks are dropped
    this.destroyed = false;

    this.latest = { particles: null, gather: null };
    this.onDiagnostics = null;

    this.canvasW = 1;
    this.canvasH = 1;

    this.#allocate();
    this.#buildBindGroups();
  }

  // -------------------------------------------------------
  // Allocation
  // -------------------------------------------------------
  #allocate() {
    const d = this.device;
    const N = this.N;
    const mk = (size, label, usage = STORAGE) =>
      d.createBuffer({ label: `${this.sim.id}-${label}`, size: Math.max(16, Math.ceil(size / 4) * 4), usage });

    this.buffers = {
      params: mk(PARAMS_BYTES, "params", GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      steps: mk(MAX_STEPS_PER_FRAME * STEP_SLOT_BYTES, "steps", GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      fA: mk(N * Q * 4, "fA"),
      fB: mk(N * Q * 4, "fB"),
      mac: mk(N * 16, "mac"),
      prevMac: mk(N * 16, "prevMac"),
      nodeType: mk(N * 4, "nodeType"),
      wallDist: mk(N * 4, "wallDist"),
      linkQ: mk(N * Q * 4, "linkQ"),
      sdf: mk(N * 4, "sdf"),
      surfList: mk(N * 4, "surfList"),
      forcePartials: mk(Math.ceil(N / 64) * 16, "forcePartials"),
      reducePartials: mk(this.reduceGroups * 16, "reducePartials"),
      particles: mk(Math.max(1, this.particleCount) * 16, "particles"),
      gather: mk(this.gatherFloats * 4, "gather"),
    };

    this.readbackBytes =
      Math.ceil(N / 64) * 16 +       // force partials (upper bound)
      this.reduceGroups * 16 +       // reduce partials
      16 +                           // centre probe
      Math.max(1, this.particleCount) * 16 +
      this.gatherFloats * 4;

    this.ring = [];
    for (let i = 0; i < READBACK_RING; i++) {
      this.ring.push({
        buffer: d.createBuffer({
          label: `${this.sim.id}-readback-${i}`,
          size: Math.ceil(this.readbackBytes / 4) * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        busy: false,
      });
    }

    this.stepData = new ArrayBuffer(MAX_STEPS_PER_FRAME * STEP_SLOT_BYTES);
  }

  #buildBindGroups() {
    const b = this.buffers;
    const { compute } = this.gpu;

    const resolver = (entry, slot = 0) => binding => {
      switch (binding) {
        case 0: return { buffer: b.params };
        case 1: return { buffer: b.steps, offset: slot * STEP_SLOT_BYTES, size: 16 };
        case 2: return { buffer: b.fA };
        case 3: return { buffer: b.fB };
        case 4: return { buffer: b.mac };
        case 5: return { buffer: b.nodeType };
        case 6: return { buffer: b.wallDist };
        case 7: return { buffer: b.linkQ };
        case 8: return { buffer: b.sdf };
        case 9: return { buffer: b.surfList };
        case 10: return { buffer: entry === "force_main" ? b.forcePartials : b.reducePartials };
        case 11: return { buffer: b.prevMac };
        case 12: return { buffer: b.particles };
        case 13: return { buffer: b.gather };
        default: throw new Error(`unknown binding ${binding}`);
      }
    };

    const make = (entry, slot = 0) =>
      createBindGroup(this.device, compute[entry], SOLVER_BINDINGS[entry], resolver(entry, slot), `${this.sim.id}-${entry}-${slot}`);

    this.bind = { perStep: [] };
    for (let k = 0; k < MAX_STEPS_PER_FRAME; k++) {
      this.bind.perStep.push({
        macro_main: make("macro_main", k),
        collide_main: make("collide_main", k),
        stream_main: make("stream_main", k),
      });
    }
    for (const entry of ["outlet_main", "rest_solids", "force_main", "reduce_main", "particle_main", "gather_vec", "gather_dist"]) {
      this.bind[entry] = make(entry);
    }
  }

  destroy() {
    this.destroyed = true;
    for (const buf of Object.values(this.buffers)) buf.destroy();
    for (const r of this.ring) {
      if (!r.busy) r.buffer.destroy();
      else r.destroyWhenDone = true;
    }
  }

  // -------------------------------------------------------
  // Uploads
  // -------------------------------------------------------
  computeNodeType(n) {
    const s = this.sim;
    return (s.solid[n] ? NODE_SOLID : 0) |
      (s.paintedNodes.has(n) ? NODE_PAINTED : 0) |
      (s.wallAdjacent && s.wallAdjacent[n] ? NODE_WALLADJ : 0);
  }

  uploadNodeTypes() {
    for (let n = 0; n < this.N; n++) this.nodeType[n] = this.computeNodeType(n);
    this.queue.writeBuffer(this.buffers.nodeType, 0, this.nodeType);
  }

  writeNodeType(n) {
    this.nodeType[n] = this.computeNodeType(n);
    this.queue.writeBuffer(this.buffers.nodeType, n * 4, this.nodeType, n, 1);
  }

  // Geometry arrays from geometry.js (AoS link layout -> SoA).
  uploadGeometry() {
    const s = this.sim;
    const N = this.N;

    const lq = new Float32Array(N * Q);
    for (let n = 0; n < N; n++) {
      const base = n * Q;
      for (let i = 1; i < Q; i++) {
        if (s.linkMask[base + i]) lq[i * N + n] = s.linkQ[base + i];
      }
    }

    this.queue.writeBuffer(this.buffers.linkQ, 0, lq);
    this.queue.writeBuffer(this.buffers.wallDist, 0, s.wallDist);
    this.queue.writeBuffer(this.buffers.sdf, 0, s.sdf);

    const list = s.fluidSurfaceList;
    this.surfCount = list.length;
    this.forceGroups = Math.max(1, Math.ceil(this.surfCount / 64));
    if (list.length > 0) this.queue.writeBuffer(this.buffers.surfList, 0, list);

    this.uploadNodeTypes();
  }

  // Full field upload (AoS f from the CPU init path -> SoA).
  uploadField(f, rho, ux, uy) {
    const N = this.N;
    const soa = new Float32Array(N * Q);
    for (let n = 0; n < N; n++) {
      const base = n * Q;
      for (let i = 0; i < Q; i++) soa[i * N + n] = f[base + i];
    }
    this.queue.writeBuffer(this.buffers.fA, 0, soa);
    this.queue.writeBuffer(this.buffers.fB, 0, soa);

    const mac = new Float32Array(N * 4);
    for (let n = 0; n < N; n++) {
      mac[n * 4] = rho[n];
      mac[n * 4 + 1] = ux[n];
      mac[n * 4 + 2] = uy[n];
    }
    this.queue.writeBuffer(this.buffers.mac, 0, mac);
    this.queue.writeBuffer(this.buffers.prevMac, 0, mac);

    this.generation++;
  }

  // A freshly painted wall node: rest equilibrium, zero velocity.
  paintNode(n) {
    this.writeNodeType(n);
    const N = this.N;
    const one = new Float32Array(1);
    for (let i = 0; i < Q; i++) {
      one[0] = W[i];
      this.queue.writeBuffer(this.buffers.fA, (i * N + n) * 4, one);
    }
    this.queue.writeBuffer(this.buffers.mac, n * 16, new Float32Array([1, 0, 0, 0]));
  }

  uploadParticles(x, y) {
    const data = new Float32Array(this.particleCount * 4);
    for (let i = 0; i < this.particleCount; i++) {
      data[i * 4] = x[i];
      data[i * 4 + 1] = y[i];
    }
    this.queue.writeBuffer(this.buffers.particles, 0, data);
    this.latest.particles = data;
  }

  setCanvasSize(w, h) {
    this.canvasW = w;
    this.canvasH = h;
  }

  writeParams(extra = {}) {
    const s = this.sim;
    const NX = this.NX;
    const width = Math.min(SPONGE_WIDTH, Math.max(1, NX - 2));

    const buf = new ArrayBuffer(PARAMS_BYTES);
    const u = new Uint32Array(buf);
    const i32 = new Int32Array(buf);
    const f = new Float32Array(buf);

    u[0] = NX;
    u[1] = this.NY;
    u[2] = this.N;
    u[3] = s.params.bodyType === "lid-driven-cavity" ? 1 : 0;

    u[4] = s.physics.collisionModel === "bgk" ? 1 : 0;
    u[5] = s.physics.lesModel === "smagorinsky" ? 1 : s.physics.lesModel === "wale" ? 2 : 0;
    u[6] = s.physics.wallModel ? 1 : 0;
    u[7] = s.nozzle.active ? 1 : 0;

    f[8] = s.NU0;
    f[9] = s.nozzle.uCoflow;
    f[10] = s.nozzle.cy;
    f[11] = s.nozzle.halfD;

    i32[12] = NX - 1 - width;
    f[13] = width;
    f[14] = SPONGE_STRENGTH;
    f[15] = s.RHO0 ?? 1.0;

    f[16] = s.forceRef.x;
    f[17] = s.forceRef.y;
    f[18] = s.display.colorScale;
    u[19] = s.display.mode === "qcrit" ? 1 : 0;

    f[20] = extra.particleDt ?? s.runtime.stepsPerFrame;
    u[21] = this.particleCount;
    u[22] = this.frameSeed >>> 0;
    u[23] = this.surfCount;

    f[24] = this.canvasW;
    f[25] = this.canvasH;

    this.queue.writeBuffer(this.buffers.params, 0, buf);
  }

  // -------------------------------------------------------
  // Commands
  // -------------------------------------------------------
  #dispatch2D(pass, pipeline, bindGroup) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.NX / 64), this.NY, 1);
  }

  restSolids() {
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.#dispatch2D(pass, this.gpu.compute.rest_solids, this.bind.rest_solids);
    pass.end();
    this.queue.submit([enc.finish()]);
  }

  // Encode one frame: `inletVelocities[k]` is the CPU-computed
  // instantaneous inlet velocity for sub-step k (getInletVelocity).
  encodeFrame(encoder, {
    inletVelocities = [],
    firstStep = 0,
    advectParticles = false,
    particleDt = 1,
    diagnostics = false,
    gatherVec = false,
    gatherDist = false,
    readParticles = false,
  } = {}) {
    const { compute } = this.gpu;
    const steps = Math.min(inletVelocities.length, MAX_STEPS_PER_FRAME);
    const isCavity = this.sim.params.bodyType === "lid-driven-cavity";

    this.frameSeed = (this.frameSeed + 1) >>> 0;
    this.writeParams({ particleDt });

    if (steps > 0) {
      const f32 = new Float32Array(this.stepData);
      const u32 = new Uint32Array(this.stepData);
      for (let k = 0; k < steps; k++) {
        const o = (k * STEP_SLOT_BYTES) / 4;
        f32[o] = inletVelocities[k];
        u32[o + 1] = firstStep + k;
      }
      this.queue.writeBuffer(this.buffers.steps, 0, this.stepData, 0, steps * STEP_SLOT_BYTES);
    }

    const pass = encoder.beginComputePass({ label: `${this.sim.id}-frame` });

    for (let k = 0; k < steps; k++) {
      const bg = this.bind.perStep[k];
      this.#dispatch2D(pass, compute.macro_main, bg.macro_main);
      this.#dispatch2D(pass, compute.collide_main, bg.collide_main);
      this.#dispatch2D(pass, compute.stream_main, bg.stream_main);
      if (!isCavity) {
        pass.setPipeline(compute.outlet_main);
        pass.setBindGroup(0, this.bind.outlet_main);
        pass.dispatchWorkgroups(Math.ceil(this.NY / 64));
      }
    }

    if (advectParticles && this.particleCount > 0) {
      pass.setPipeline(compute.particle_main);
      pass.setBindGroup(0, this.bind.particle_main);
      pass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
    }

    if (diagnostics) {
      pass.setPipeline(compute.force_main);
      pass.setBindGroup(0, this.bind.force_main);
      pass.dispatchWorkgroups(this.forceGroups);

      pass.setPipeline(compute.reduce_main);
      pass.setBindGroup(0, this.bind.reduce_main);
      pass.dispatchWorkgroups(this.reduceGroups);
    }

    if (gatherVec) {
      pass.setPipeline(compute.gather_vec);
      pass.setBindGroup(0, this.bind.gather_vec);
      pass.dispatchWorkgroups(Math.ceil(this.vecCount / 64));
    }
    if (gatherDist) {
      pass.setPipeline(compute.gather_dist);
      pass.setBindGroup(0, this.bind.gather_dist);
      pass.dispatchWorkgroups(Math.ceil(this.distCount / 64));
    }

    pass.end();

    // ---- readback (non-blocking) ----
    const wantRead = diagnostics || readParticles || gatherVec || gatherDist;
    let pending = null;

    if (wantRead) {
      const slot = this.ring.find(r => !r.busy);
      if (slot) {
        const layout = {};
        let off = 0;
        const copy = (src, srcOffset, bytes, key) => {
          encoder.copyBufferToBuffer(src, srcOffset, slot.buffer, off, bytes);
          layout[key] = { off, bytes };
          off += bytes;
        };

        if (diagnostics) {
          copy(this.buffers.forcePartials, 0, this.forceGroups * 16, "force");
          copy(this.buffers.reducePartials, 0, this.reduceGroups * 16, "reduce");
          const cx = Math.floor(this.NX * 0.5);
          const cy = Math.floor(this.NY * 0.5);
          copy(this.buffers.mac, (cy * this.NX + cx) * 16, 16, "center");
        }
        if (readParticles && this.particleCount > 0) {
          copy(this.buffers.particles, 0, this.particleCount * 16, "particles");
        }
        if (gatherVec || gatherDist) {
          copy(this.buffers.gather, 0, this.gatherFloats * 4, "gather");
        }

        slot.busy = true;
        pending = { slot, layout, bytes: off, generation: this.generation, gatherVec, gatherDist };
      }
    }

    if (diagnostics) {
      // Previous-sample macros for RMS dU (after the reduction consumed them).
      encoder.copyBufferToBuffer(this.buffers.mac, 0, this.buffers.prevMac, 0, this.N * 16);
    }

    return pending;
  }

  // Call after queue.submit() for the encoder used in encodeFrame().
  resolveReadback(pending, meta) {
    if (!pending) return;
    const { slot, layout, bytes, generation } = pending;

    slot.buffer.mapAsync(GPUMapMode.READ, 0, Math.ceil(bytes / 4) * 4).then(() => {
      const mapped = slot.buffer.getMappedRange(0, Math.ceil(bytes / 4) * 4);
      const data = new Float32Array(mapped.slice(0));
      slot.buffer.unmap();
      slot.busy = false;
      if (slot.destroyWhenDone) slot.buffer.destroy();

      if (this.destroyed || generation !== this.generation) return;
      this.#consume(data, layout, pending, meta);
    }).catch(() => {
      slot.busy = false;
    });
  }

  #consume(data, layout, pending, meta) {
    const view = (key) => {
      const l = layout[key];
      return l ? data.subarray(l.off / 4, (l.off + l.bytes) / 4) : null;
    };

    const force = view("force");
    const reduce = view("reduce");
    if (force && reduce) {
      let fx = 0, fy = 0, mz = 0;
      for (let k = 0; k < force.length; k += 4) {
        fx += force[k];
        fy += force[k + 1];
        mz += force[k + 2];
      }

      let sumSpeed = 0, sumRhoDev = 0, sumDeltaUSq = 0, fluidCount = 0;
      for (let k = 0; k < reduce.length; k += 4) {
        sumSpeed += reduce[k];
        sumRhoDev += reduce[k + 1];
        sumDeltaUSq += reduce[k + 2];
        fluidCount += reduce[k + 3];
      }

      const center = view("center");
      this.onDiagnostics?.({
        fx, fy, mz,
        sumSpeed, sumRhoDev, sumDeltaUSq,
        fluidCount: Math.round(fluidCount),
        uCenter: center[1],
        vCenter: center[2],
        frame: meta?.frameCount ?? 0,
        step: meta?.stepCount ?? 0,
      });
    }

    const particles = view("particles");
    if (particles) this.latest.particles = particles;

    const gather = view("gather");
    if (gather) {
      this.latest.gather = {
        vec: pending.gatherVec ? gather.subarray(0, this.vecCount * 4) : null,
        dist: pending.gatherDist ? gather.subarray(this.vecCount * 4) : null,
      };
    }
  }
}
