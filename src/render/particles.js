// src/render/particles.js
// GPU particle renderer/controller.
// Assumes particle state is already GPU-resident and updated by compute.
// This file owns particle render orchestration only.

import {
  ensureCanvasContext,
  resizeCanvasToDisplaySize,
  createCommandEncoder,
  submitCommands,
} from "../gpu/gpuContext.js";

import { PARTICLE_DEFAULTS } from "../config.js";

export function createParticleRenderer(sim, options = {}) {
  const state = {
    enabled: false,

    canvas: null,
    gpuContextInfo: null,

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
  };

  function attachCanvas(canvas, gpuCtx = null) {
    state.canvas = canvas;

    if (gpuCtx && canvas) {
      state.gpuContextInfo = ensureCanvasContext(gpuCtx, canvas);
    }

    resize();
  }

  function attachGpuContext(gpuCtx) {
    if (!state.canvas || !gpuCtx) return;
    state.gpuContextInfo = ensureCanvasContext(gpuCtx, state.canvas);
  }

  function resize() {
    if (!state.canvas) return;

    resizeCanvasToDisplaySize(state.canvas, 2);

    if (sim.render?.flowCanvas) {
      state.canvas.style.width = sim.render.flowCanvas.style.width || "100%";
      state.canvas.style.height = sim.render.flowCanvas.style.height || "100%";
    }
  }

  function clear() {
    // For GPU canvas rendering we do not manually clear via 2D context.
    // The render pass should clear when needed.
  }

  function setEnabled(flag) {
    state.enabled = !!flag;
    sim.display.smokeVisible = state.enabled;
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

  function render(gpuCtx, pipelines) {
    if (!state.enabled) return;
    if (!state.canvas) return;

    const particlePipeline = pipelines?.render?.particles;
    if (!particlePipeline) return;

    const particleBindGroup = sim?.gpu?.bindGroups?.particleRender;
    if (!particleBindGroup) return;

    if (!state.gpuContextInfo) {
      state.gpuContextInfo = ensureCanvasContext(gpuCtx, state.canvas);
    }

    resize();

    const currentTexture = state.gpuContextInfo.context.getCurrentTexture();
    const view = currentTexture.createView();

    const encoder = createCommandEncoder(gpuCtx, `${sim.id}-particle-render-encoder`);

    const pass = encoder.beginRenderPass({
      label: `${sim.id}-particle-render-pass`,
      colorAttachments: [
        {
          view,
          loadOp: "load",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(particlePipeline);
    pass.setBindGroup(0, particleBindGroup);
    pass.draw(6, state.count, 0, 0);
    pass.end();

    submitCommands(gpuCtx, encoder.finish());
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
  };
}



