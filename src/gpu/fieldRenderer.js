// src/gpu/fieldRenderer.js
// ============================================================
// GPU field renderer: the scalar-field part of render.js
// (speed / Q-criterion, solid fill, blur, Hermite colormap)
// drawn into a WebGPU canvas. Vector / grid / distribution
// overlays are drawn by overlayRenderer.js on a 2D canvas.
// ============================================================

import { RENDER_BINDINGS, createBindGroup } from "./gpuContext.js";

// ---------------------------------------------------------
// COLORMAP — cubic Hermite, 512 steps (identical to render.js)
// ---------------------------------------------------------
const COLORMAP_STEPS = 512;
const STOPS = [
  [0,   0,   180],
  [0,   210, 255],
  [0,   255, 80 ],
  [255, 220, 0  ],
  [255, 20,  0  ],
];

function cubicHermite(a, b, c, d, t) {
  const t2 = t * t, t3 = t2 * t;
  return (
    (-0.5 * a + 1.5 * b - 1.5 * c + 0.5 * d) * t3 +
    (a - 2.5 * b + 2 * c - 0.5 * d) * t2 +
    (-0.5 * a + 0.5 * c) * t +
    b
  );
}

function buildPackedColormap(steps) {
  const table = new Uint8ClampedArray(steps * 3);
  const nStops = STOPS.length;

  for (let s = 0; s < steps; s++) {
    const t      = s / (steps - 1);
    const scaled = t * (nStops - 1);
    const i1     = Math.min(Math.floor(scaled), nStops - 2);
    const local  = scaled - i1;
    const i0     = Math.max(i1 - 1, 0);
    const i2     = Math.min(i1 + 1, nStops - 1);
    const i3     = Math.min(i1 + 2, nStops - 1);

    for (let c = 0; c < 3; c++) {
      table[s * 3 + c] = Math.round(
        cubicHermite(STOPS[i0][c], STOPS[i1][c], STOPS[i2][c], STOPS[i3][c], local)
      );
    }
  }

  const packed = new Uint32Array(steps);
  for (let s = 0; s < steps; s++) {
    packed[s] = (table[s * 3] | (table[s * 3 + 1] << 8) | (table[s * 3 + 2] << 16) | (255 << 24)) >>> 0;
  }
  return packed;
}

const PACKED_COLORMAP = buildPackedColormap(COLORMAP_STEPS);

export class FieldRenderer {
  constructor(gpu, gpuSim, canvas) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.gpuSim = gpuSim;
    this.canvas = canvas;

    this.context = canvas.getContext("webgpu");
    this.context.configure({ device: this.device, format: gpu.format, alphaMode: "opaque" });

    const N = gpuSim.N;
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (size, label) => this.device.createBuffer({ label, size, usage: STORAGE });

    this.buffers = {
      velA: mk(N * 8, "velA"),
      velB: mk(N * 8, "velB"),
      scalar: mk(N * 4, "scalar"),
      scalarTmp: mk(N * 4, "scalarTmp"),
      color: mk(N * 4, "color"),
      cmap: mk(COLORMAP_STEPS * 4, "cmap"),
    };
    this.device.queue.writeBuffer(this.buffers.cmap, 0, PACKED_COLORMAP);

    const sb = gpuSim.buffers;
    const b = this.buffers;
    const resolve = binding => {
      switch (binding) {
        case 0: return { buffer: sb.params };
        case 1: return { buffer: sb.mac };
        case 2: return { buffer: sb.nodeType };
        case 3: return { buffer: b.velA };
        case 4: return { buffer: b.velB };
        case 5: return { buffer: b.scalar };
        case 6: return { buffer: b.scalarTmp };
        case 7: return { buffer: b.cmap };
        case 8: return { buffer: b.color };
        case 9: return { buffer: b.color };
        default: throw new Error(`unknown render binding ${binding}`);
      }
    };

    this.bind = {};
    for (const [entry, bindings] of Object.entries(RENDER_BINDINGS)) {
      this.bind[entry] = createBindGroup(this.device, gpu.display[entry], bindings, resolve, `${gpuSim.sim.id}-${entry}`);
    }
  }

  destroy() {
    for (const buf of Object.values(this.buffers)) buf.destroy();
    try { this.context.unconfigure(); } catch { /* ignore */ }
  }

  encode(encoder, mode) {
    const { display } = this.gpu;
    const NX = this.gpuSim.NX;
    const NY = this.gpuSim.NY;
    const gx = Math.ceil(NX / 64);

    const pass = encoder.beginComputePass({ label: "field-display" });
    const run = entry => {
      pass.setPipeline(display[entry]);
      pass.setBindGroup(0, this.bind[entry]);
      pass.dispatchWorkgroups(gx, NY, 1);
    };

    if (mode === "qcrit") {
      run("smooth_init");
      for (let p = 0; p < 3; p++) {
        run("smooth_h");
        run("smooth_v");
      }
    }
    run("scalar_main");
    run("fill_main");
    for (let p = 0; p < 2; p++) {
      run("blur_h");
      run("blur_v");
    }
    run("colorize");
    pass.end();

    const rp = encoder.beginRenderPass({
      label: "field-present",
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    rp.setPipeline(display.present);
    rp.setBindGroup(0, this.bind.present);
    rp.draw(3);
    rp.end();
  }
}
