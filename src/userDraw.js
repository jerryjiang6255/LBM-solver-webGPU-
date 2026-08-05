// src/userDraw.js
// CPU-side sandbox wall editing controller.
// Owns stroke state and painted-node edits only.
// Supports live dirty-node reporting during drag so GPU-side solid masks
// can be updated without a full geometry rebuild on every mouse move.

export function createUserDrawController(sim, wallCanvas, callbacks = {}) {
  const {
    onGeometryEdited = () => {},
    onGeometryCommit = () => {},
    isEnabled = () => true,
  } = callbacks;

  const state = {
    brushRadius: 1,
    strokes: [],
    currentStroke: null,
    lastLX: null,
    lastLY: null,
    isDrawing: false,
    mode: "draw", // "draw" | "erase"

    overlayCanvas: wallCanvas,
    inputCanvas: null,
  };

  function setMode(mode) {
    if (mode === "draw" || mode === "erase") {
      state.mode = mode;
    }
  }

  function setBrushRadius(r) {
    state.brushRadius = Math.max(1, r | 0);
  }

  function attachOverlayCanvas(canvas) {
    state.overlayCanvas = canvas;
  }

  function attachInputCanvas(canvas) {
    state.inputCanvas = canvas;
  }

  function beginStroke(lx, ly, mode = state.mode) {
    state.isDrawing = true;
    state.lastLX = Math.round(lx);
    state.lastLY = Math.round(ly);
    state.currentStroke = [{ x: lx, y: ly, mode }];
  }

  function paintStroke(lx, ly, mode = state.mode) {
    const x1 = Math.round(lx);
    const y1 = Math.round(ly);
    const dirtySet = new Set();

    if (state.lastLX === null || state.lastLY === null) {
      paintBrush(x1, y1, mode, dirtySet);
    } else {
      rasterLine(state.lastLX, state.lastLY, x1, y1, mode, dirtySet);
    }

    state.lastLX = x1;
    state.lastLY = y1;

    if (state.currentStroke) {
      state.currentStroke.push({ x: lx, y: ly, mode });
    }

    if (dirtySet.size > 0) {
      markEditedOnly();
      onGeometryEdited(sim, { dirtyNodes: Array.from(dirtySet) });
    }
  }

  function endStroke(mode = state.mode) {
    if (state.currentStroke && state.currentStroke.length > 0) {
      state.strokes.push(state.currentStroke);
    }

    state.isDrawing = false;
    state.currentStroke = null;
    state.lastLX = null;
    state.lastLY = null;

    markNeedsCommit();
    onGeometryCommit(sim, { mode });
  }

 function clearDrawing() {
    const clearedNodes = Array.from(sim.geometry.paintedNodes);

    sim.geometry.paintedNodes.clear();

    state.strokes.length = 0;
    state.currentStroke = null;
    state.lastLX = null;
    state.lastLY = null;
    state.isDrawing = false;

    markNeedsCommit();
    onGeometryCommit(sim, { mode: "clear", clearedNodes });
  }

  function redrawWalls() {
    const canvas = state.overlayCanvas;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const NX = sim.dims.NX;
    const NY = sim.dims.NY;
    const W = canvas.width;
    const H = canvas.height;
    const sx = W / NX;
    const sy = H / NY;

    ctx.clearRect(0, 0, W, H);

    const allStrokes = state.currentStroke
      ? [...state.strokes, state.currentStroke]
      : state.strokes;

    if (allStrokes.length === 0) {
      sim.geometry.dirtyFlags.wallOverlayDirty = false;
      return;
    }

    const lineW = Math.max(sx, sy) * (state.brushRadius + 0.5);

    for (const pts of allStrokes) {
      if (!pts || pts.length === 0) continue;

      ctx.save();
      ctx.strokeStyle = "rgba(175, 198, 230, 0.75)";
      ctx.lineWidth = lineW * 1.25;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      drawSmoothStroke(ctx, pts, sx, sy);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "#292a2f";
      ctx.lineWidth = lineW;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      drawSmoothStroke(ctx, pts, sx, sy);
      ctx.stroke();
      ctx.restore();
    }

    sim.geometry.dirtyFlags.wallOverlayDirty = false;
  }

  function canvasToLattice(clientX, clientY) {
    const canvas = state.inputCanvas;
    if (!canvas) return { lx: 0, ly: 0 };

    const rect = canvas.getBoundingClientRect();

    return {
      lx: (clientX - rect.left) * (sim.dims.NX / rect.width),
      ly: (clientY - rect.top) * (sim.dims.NY / rect.height),
    };
  }

  function bindPointerEvents(targetCanvas = state.inputCanvas) {
    if (!targetCanvas) return;
    attachInputCanvas(targetCanvas);

    targetCanvas.addEventListener("mousedown", e => {
      if (!isEnabled()) return;
      if (e.button !== 0) return;

      const { lx, ly } = canvasToLattice(e.clientX, e.clientY);
      beginStroke(lx, ly);
      paintStroke(lx, ly);
      redrawWalls();
    });

    targetCanvas.addEventListener("mousemove", e => {
      if (!isEnabled()) return;
      if (!state.isDrawing) return;

      const { lx, ly } = canvasToLattice(e.clientX, e.clientY);
      paintStroke(lx, ly);
      redrawWalls();
    });

    targetCanvas.addEventListener("mouseup", () => {
      if (!isEnabled()) return;
      if (!state.isDrawing) return;

      endStroke();
      redrawWalls();
    });

    targetCanvas.addEventListener("mouseleave", () => {
      if (!isEnabled()) return;
      if (!state.isDrawing) return;

      endStroke();
      redrawWalls();
    });

    targetCanvas.addEventListener("touchstart", e => {
      if (!isEnabled()) return;

      e.preventDefault();
      const t = e.touches[0];
      const { lx, ly } = canvasToLattice(t.clientX, t.clientY);
      beginStroke(lx, ly);
      paintStroke(lx, ly);
      redrawWalls();
    }, { passive: false });

    targetCanvas.addEventListener("touchmove", e => {
      if (!isEnabled()) return;
      if (!state.isDrawing) return;

      e.preventDefault();
      const t = e.touches[0];
      const { lx, ly } = canvasToLattice(t.clientX, t.clientY);
      paintStroke(lx, ly);
      redrawWalls();
    }, { passive: false });

    targetCanvas.addEventListener("touchend", () => {
      if (!isEnabled()) return;
      if (!state.isDrawing) return;

      endStroke();
      redrawWalls();
    });
  }

  function paintBrush(lx, ly, mode = state.mode, dirtySet = null) {
    const NX = sim.dims.NX;
    const NY = sim.dims.NY;
    const paintedNodes = sim.geometry.paintedNodes;

    const x0 = Math.round(lx);
    const y0 = Math.round(ly);
    const r = state.brushRadius;

    let changed = false;

    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy > r * r) continue;

        const x = x0 + dx;
        const y = y0 + dy;

        if (x < 1 || x >= NX - 1 || y < 1 || y >= NY - 1) continue;

        const n = y * NX + x;

        if (mode === "draw") {
          if (paintedNodes.has(n)) continue;
          paintedNodes.add(n);
          changed = true;
          if (dirtySet) dirtySet.add(n);
        } else {
          if (!paintedNodes.has(n)) continue;
          paintedNodes.delete(n);
          changed = true;
          if (dirtySet) dirtySet.add(n);
        }
      }
    }

    if (changed) {
      markEditedOnly();
    }
  }

  function rasterLine(x0, y0, x1, y1, mode, dirtySet = null) {
    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;

    let err = dx + dy;
    let x = x0;
    let y = y0;

    while (true) {
      paintBrush(x, y, mode, dirtySet);
      if (x === x1 && y === y1) break;

      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y += sy;
      }
    }
  }

  function markEditedOnly() {
    sim.geometry.dirtyFlags.wallOverlayDirty = true;
  }

  function markNeedsCommit() {
    sim.geometry.dirtyFlags.geometryDirty = true;
    sim.geometry.dirtyFlags.gpuUploadRequired = true;
    sim.geometry.dirtyFlags.wallOverlayDirty = true;

    sim.gpu.state.geometryDirty = true;
    sim.gpu.state.uniformsDirty = true;
  }

  return {
    state,
    setMode,
    setBrushRadius,
    attachOverlayCanvas,
    attachInputCanvas,
    bindPointerEvents,
    canvasToLattice,
    beginStroke,
    paintStroke,
    endStroke,
    clearDrawing,
    redrawWalls,
    paintBrush,
    get strokes() {
      return state.strokes;
    },
  };
}

function drawSmoothStroke(ctx, pts, sx, sy) {
  if (pts.length === 1) {
    const x = pts[0].x * sx;
    const y = pts[0].y * sy;
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth * 0.5, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(pts[0].x * sx, pts[0].y * sy);

  for (let i = 1; i < pts.length - 1; i++) {
    const mx = ((pts[i].x + pts[i + 1].x) * 0.5) * sx;
    const my = ((pts[i].y + pts[i + 1].y) * 0.5) * sy;
    ctx.quadraticCurveTo(pts[i].x * sx, pts[i].y * sy, mx, my);
  }

  const last = pts[pts.length - 1];
  ctx.lineTo(last.x * sx, last.y * sy);
}



