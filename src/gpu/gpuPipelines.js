// src/gpu/gpuPipelines.js
// Shared pipeline and bind-group-layout creation.
// No per-sim runtime stepping here.

import { createShaderModule } from "./gpuContext.js";

export function createGpuPipelines(gpuCtx, shaderCode) {
  const { device, presentationFormat } = gpuCtx;
  const shaderModule = createShaderModule(gpuCtx, shaderCode, "lbm-shaders");

  const bindGroupLayouts = {
    simCompute: createSimComputeBindGroupLayout(device),
    fieldRender: createFieldRenderBindGroupLayout(device),
    particleRender: createParticleRenderBindGroupLayout(device),
  };

  const pipelineLayouts = {
    solver: device.createPipelineLayout({
      label: "solver-pipeline-layout",
      bindGroupLayouts: [bindGroupLayouts.simCompute],
    }),

    fieldRender: device.createPipelineLayout({
      label: "field-render-pipeline-layout",
      bindGroupLayouts: [bindGroupLayouts.fieldRender],
    }),

    particleRender: device.createPipelineLayout({
      label: "particle-render-pipeline-layout",
      bindGroupLayouts: [bindGroupLayouts.particleRender],
    }),
  };

 const compute = {
    solverA: device.createComputePipeline({
      label: "solverA-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "solverA_main",
      },
    }),

    solverB: device.createComputePipeline({
      label: "solverB-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "solverB_main",
      },
    }),

    force: device.createComputePipeline({
      label: "force-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "force_main",
      },
    }),

    particle: device.createComputePipeline({
      label: "particle-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "particle_main",
      },
    }),

    reduction: device.createComputePipeline({
      label: "reduction-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "reduction_main",
      },
    }),

    reopen: device.createComputePipeline({
      label: "reopen-pipeline",
      layout: pipelineLayouts.solver,
      compute: {
        module: shaderModule,
        entryPoint: "reopen_main",
      },
    }),
  };

  const render = {
    field: device.createRenderPipeline({
      label: "field-render-pipeline",
      layout: pipelineLayouts.fieldRender,

      vertex: {
        module: shaderModule,
        entryPoint: "field_vs",
      },

      fragment: {
        module: shaderModule,
        entryPoint: "field_fs",
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
            writeMask: GPUColorWrite.ALL,
          },
        ],
      },

      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
    }),

    particles: device.createRenderPipeline({
      label: "particle-render-pipeline",
      layout: pipelineLayouts.particleRender,

      vertex: {
        module: shaderModule,
        entryPoint: "particle_vs",
      },

      fragment: {
        module: shaderModule,
        entryPoint: "particle_fs",
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
            writeMask: GPUColorWrite.ALL,
          },
        ],
      },

      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
    }),
  };

  return {
    shaderModule,
    bindGroupLayouts,
    pipelineLayouts,
    compute,
    render,
  };
}

export function createSimComputeBindGroup(device, pipelines, sim, srcBuffer, dstBuffer, labelSuffix = "compute") {
  const b = sim.gpu.buffers;
  const particleBuffer = b.particles ?? createFallbackParticleBuffer(device);
  console.log("createSimComputeBindGroup: macro label=", b.macro?.label, "size=", b.macro?.size);

  return device.createBindGroup({
    label: `${sim.id}-${labelSuffix}-bind-group`,
    layout: pipelines.bindGroupLayouts.simCompute,
    entries: [
      { binding: 0, resource: { buffer: b.uniforms } },

      { binding: 1, resource: { buffer: srcBuffer } },
      { binding: 2, resource: { buffer: dstBuffer } },
      { binding: 3, resource: { buffer: b.macro } },

      { binding: 4, resource: { buffer: b.solid } },
      { binding: 5, resource: { buffer: b.sdf } },
      { binding: 6, resource: { buffer: b.wallDist } },
      { binding: 7, resource: { buffer: b.wallAdjacent } },

      { binding: 8, resource: { buffer: b.linkMask } },
      { binding: 9, resource: { buffer: b.linkQ } },
      { binding: 10, resource: { buffer: b.fluidSurfaceList } },

      { binding: 11, resource: { buffer: b.force } },
      { binding: 12, resource: { buffer: b.reduction } },

     { binding: 13, resource: { buffer: particleBuffer } },
      { binding: 14, resource: { buffer: b.prevMacro } },
      { binding: 15, resource: { buffer: b.reopenNodes } },
    ],
  });
}

export function createFieldRenderBindGroup(device, pipelines, sim) {
  const b = sim.gpu.buffers;

  return device.createBindGroup({
    label: `${sim.id}-field-render-bind-group`,
    layout: pipelines.bindGroupLayouts.fieldRender,
    entries: [
      { binding: 0, resource: { buffer: b.uniforms } },
      { binding: 1, resource: { buffer: b.macro } },
      { binding: 2, resource: { buffer: b.solid } },
    ],
  });
}

export function createParticleRenderBindGroup(device, pipelines, sim) {
  const b = sim.gpu.buffers;
  const particleBuffer = b.particles ?? createFallbackParticleBuffer(device);

  return device.createBindGroup({
    label: `${sim.id}-particle-render-bind-group`,
    layout: pipelines.bindGroupLayouts.particleRender,
    entries: [
      { binding: 0, resource: { buffer: b.uniforms } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: b.macro } },
    ],
  });
}

export function rebuildSimBindGroups(gpuCtx, pipelines, sim) {
  const { device } = gpuCtx;
  const b = sim.gpu.buffers;

  sim.gpu.bindGroups.computeForward = createSimComputeBindGroup(
    device,
    pipelines,
    sim,
    b.fIn,
    b.fOut,
    "compute-forward"
  );

  sim.gpu.bindGroups.computeBackward = createSimComputeBindGroup(
    device,
    pipelines,
    sim,
    b.fOut,
    b.fIn,
    "compute-backward"
  );

  sim.gpu.bindGroups.fieldRender = createFieldRenderBindGroup(device, pipelines, sim);
  sim.gpu.bindGroups.particleRender = createParticleRenderBindGroup(device, pipelines, sim);

  return sim.gpu.bindGroups;
}

export function getActiveComputeBindGroup(sim) {
  return sim.gpu.state.activePopulationIsIn
    ? sim.gpu.bindGroups.computeForward
    : sim.gpu.bindGroups.computeBackward;
}

// ---------------------------------------------------------
// Internal layout builders
// Must match shaders.wgsl exactly.
// ---------------------------------------------------------
function createSimComputeBindGroupLayout(device) {
  return device.createBindGroupLayout({
    label: "sim-compute-bgl",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },

      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },

      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },

      {
        binding: 8,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 9,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 10,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },

      {
        binding: 11,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 12,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },

          {
        binding: 13,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 14,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 15,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
}

function createFieldRenderBindGroupLayout(device) {
  return device.createBindGroupLayout({
    label: "field-render-bgl",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
}

function createParticleRenderBindGroupLayout(device) {
  return device.createBindGroupLayout({
    label: "particle-render-bgl",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
}

let _fallbackParticleBuffer = null;

function createFallbackParticleBuffer(device) {
  if (_fallbackParticleBuffer) return _fallbackParticleBuffer;

  _fallbackParticleBuffer = device.createBuffer({
    label: "fallback-particle-buffer",
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return _fallbackParticleBuffer;
}


