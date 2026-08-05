// src/gpu/gpuContext.js
// WebGPU bootstrap and shared context creation.
// No sim-specific buffers, pipelines, or solver logic here.

export async function initWebGPUContext() {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not supported in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new Error("Failed to acquire a WebGPU adapter.");
  }

 const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(
        16,
        adapter.limits.maxStorageBuffersPerShaderStage
      ),
    },
  });

  const queue = device.queue;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  const gpuCtx = {
    adapter,
    device,
    queue,
    presentationFormat,

    features: {
      adapter: new Set(adapter.features),
      device: new Set(device.features),
    },

    limits: {
      adapter: adapter.limits,
      device: device.limits,
    },

    contexts: new WeakMap(),
  };

  installDeviceErrorHooks(gpuCtx);
  return gpuCtx;
}

export function configureCanvasContext(gpuCtx, canvas, options = {}) {
  if (!gpuCtx?.device) {
    throw new Error("configureCanvasContext: invalid gpuCtx.");
  }
  if (!canvas) {
    throw new Error("configureCanvasContext: canvas is required.");
  }

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Failed to get WebGPU canvas context.");
  }

  const alphaMode = options.alphaMode ?? "premultiplied";
  const usage =
    options.usage ??
    (GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC);

  context.configure({
    device: gpuCtx.device,
    format: gpuCtx.presentationFormat,
    alphaMode,
    usage,
  });

  const info = {
    canvas,
    context,
    format: gpuCtx.presentationFormat,
    alphaMode,
    usage,
  };

  gpuCtx.contexts.set(canvas, info);
  return info;
}

export function getConfiguredCanvasContext(gpuCtx, canvas) {
  return gpuCtx?.contexts?.get(canvas) ?? null;
}

export function ensureCanvasContext(gpuCtx, canvas, options = {}) {
  const existing = getConfiguredCanvasContext(gpuCtx, canvas);
  if (existing) return existing;
  return configureCanvasContext(gpuCtx, canvas, options);
}

export function resizeCanvasToDisplaySize(canvas, maxDevicePixelRatio = 2) {
  if (!canvas) return false;

  const dpr = Math.min(window.devicePixelRatio || 1, maxDevicePixelRatio);
  const width = Math.max(1, Math.round((canvas.clientWidth || 1) * dpr));
  const height = Math.max(1, Math.round((canvas.clientHeight || 1) * dpr));

  if (canvas.width === width && canvas.height === height) {
    return false;
  }

  canvas.width = width;
  canvas.height = height;
  return true;
}

export function createCommandEncoder(gpuCtx, label = "main-command-encoder") {
  return gpuCtx.device.createCommandEncoder({ label });
}

export function submitCommands(gpuCtx, commandBuffers) {
  const list = Array.isArray(commandBuffers) ? commandBuffers : [commandBuffers];
  gpuCtx.queue.submit(list.filter(Boolean));
}

export function createShaderModule(gpuCtx, code, label = "shader-module") {
  if (typeof code !== "string" || code.length === 0) {
    throw new Error("createShaderModule: WGSL code must be a non-empty string.");
  }

  return gpuCtx.device.createShaderModule({
    label,
    code,
  });
}

export async function validateShaderModule(gpuCtx, code, label = "shader-module") {
  const module = createShaderModule(gpuCtx, code, label);

  if ("getCompilationInfo" in module) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === "error");

    if (errors.length > 0) {
      const text = errors
        .map(e => `[${e.lineNum}:${e.linePos}] ${e.message}`)
        .join("\n");
      throw new Error(`WGSL compilation failed for ${label}:\n${text}`);
    }
  }

  return module;
}

export function createReadbackBuffer(gpuCtx, byteSize, label = "readback-buffer") {
  return gpuCtx.device.createBuffer({
    label,
    size: alignTo(byteSize, 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

export function destroyBufferSafe(buffer) {
  if (!buffer) return;
  try {
    buffer.destroy();
  } catch {
    // no-op
  }
}


export function destroyTextureSafe(texture) {
  if (!texture) return;
  try {
    texture.destroy();
  } catch {
    // no-op
  }
}

export function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function installDeviceErrorHooks(gpuCtx) {
  const { device } = gpuCtx;

  device.lost.then(info => {
    console.error("WebGPU device lost:", info);
  });

  device.addEventListener?.("uncapturederror", event => {
    console.error("WebGPU uncaptured error:", event.error);
  });
}


