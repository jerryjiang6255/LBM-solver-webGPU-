// src/render/fieldRenderer.js
// GPU-backed primary field renderer.
// No solver stepping, no DOM control logic.

import {
  ensureCanvasContext,
  resizeCanvasToDisplaySize,
  createCommandEncoder,
  submitCommands,
} from "../gpu/gpuContext.js";

import { uploadSimParams } from "../gpu/gpuBuffers.js";

export function createFieldRenderer(gpuCtx, pipelines) {
  function attachCanvas(sim, canvas) {
    sim.render.flowCanvas = canvas;
    sim.render.flowContext = ensureCanvasContext(gpuCtx, canvas);
    sim.render.canvasFormat = sim.render.flowContext.format;
    sim.render.sizeDirty = true;
  }

  function resize(sim) {
    const canvas = sim.render.flowCanvas;
    if (!canvas) return false;

    const changed = resizeCanvasToDisplaySize(canvas, 2);
    if (changed) {
      sim.render.sizeDirty = false;
    }
    return changed;
  }

  function render(sim, options = {}) {
    if (!sim.render.flowCanvas) {
      throw new Error(`render(field): sim '${sim.id}' has no flow canvas attached.`);
    }

    if (!sim.gpu?.bindGroups?.fieldRender) {
      throw new Error(`render(field): sim '${sim.id}' field render bind group is missing.`);
    }

    resize(sim);

    if (sim.gpu.state.uniformsDirty || options.forceUniformUpload) {
      uploadSimParams(gpuCtx, sim);
    }

    const contextInfo = ensureCanvasContext(gpuCtx, sim.render.flowCanvas);
    sim.render.flowContext = contextInfo;

    const currentTexture = contextInfo.context.getCurrentTexture();
    const view = currentTexture.createView();

    const encoder = createCommandEncoder(gpuCtx, `${sim.id}-field-render-encoder`);

    const pass = encoder.beginRenderPass({
      label: `${sim.id}-field-render-pass`,
      colorAttachments: [
        {
          view,
          clearValue: options.clearValue ?? { r: 0.023, g: 0.039, b: 0.063, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipelines.render.field);
    pass.setBindGroup(0, sim.gpu.bindGroups.fieldRender);
    pass.draw(3, 1, 0, 0);
    pass.end();

    submitCommands(gpuCtx, encoder.finish());
  }

  return {
    attachCanvas,
    resize,
    render,
  };
}


