// src/userDraw.js
// ============================================================
// Compare-mode ready wall drawing.
// Instance-based: each sim/canvas gets its own drawer state.
// Updated for global fidelity presets: dimensions come from sim.
// WebGPU: painted nodes are mirrored to the GPU through sim.gpu
// (node flags + rest-equilibrium populations).
// ============================================================

const BRUSH_R = 1;

export function createUserDraw(sim, wallCanvas) {
  const strokes = [];
  let currentStroke = null;
  let lastLX = null;
  let lastLY = null;

  function beginStroke(lx, ly) {
    lastLX = Math.round(lx);
    lastLY = Math.round(ly);
    currentStroke = [{ x: lx, y: ly }];
  }

  function paintStroke(lx, ly, mode = "draw") {
    const x1 = Math.round(lx);
    const y1 = Math.round(ly);

    if (lastLX === null) {
      paintBrush(x1, y1, mode);
    } else {
      bresenham(lastLX, lastLY, x1, y1, mode);
    }

    lastLX = x1;
    lastLY = y1;

    if (currentStroke && mode === "draw") {
      currentStroke.push({ x: lx, y: ly });
    }
  }

  function endStroke(mode = "draw") {
    lastLX = null;
    lastLY = null;

    if (currentStroke && currentStroke.length > 0 && mode === "draw") {
      strokes.push(currentStroke);
    }
    currentStroke = null;
  }

  function bresenham(x0, y0, x1, y1, mode) {
    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;

    let x = x0, y = y0;
    while (true) {
      paintBrush(x, y, mode);
      if (x === x1 && y === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x += sx; }
      if (e2 <= dx) { err += dx; y += sy; }
    }
  }

  function paintBrush(lx, ly, mode) {
    const { NX, NY, solid, solidBase, paintedNodes } = sim;

    const x0 = Math.round(lx);
    const y0 = Math.round(ly);

    for (let dy = -BRUSH_R; dy <= BRUSH_R; dy++) {
      for (let dx = -BRUSH_R; dx <= BRUSH_R; dx++) {
        if (dx * dx + dy * dy > BRUSH_R * BRUSH_R) continue;

        const x = x0 + dx;
        const y = y0 + dy;
        if (x < 1 || x >= NX - 1 || y < 1 || y >= NY - 1) continue;

        const n = y * NX + x;

        if (mode === "draw") {
          if (solid[n]) continue;
          solid[n] = 1;
          paintedNodes.add(n);
          sim.gpu?.paintNode(n);   // rest equilibrium, u = 0 (CPU reequilibrate)

        } else {
          if (!paintedNodes.has(n)) continue;
          solid[n] = solidBase[n];
          paintedNodes.delete(n);
          sim.gpu?.writeNodeType(n);
        }
      }
    }
  }

  function clearDrawing() {
    const { solid, solidBase, paintedNodes } = sim;

    const cleared = Array.from(paintedNodes);
    for (const n of cleared) {
      solid[n] = solidBase[n];
    }

    paintedNodes.clear();
    if (sim.gpu) {
      for (const n of cleared) sim.gpu.writeNodeType(n);
    }

    strokes.length = 0;
    currentStroke = null;
    lastLX = null;
    lastLY = null;
  }

  function redrawWalls() {
    if (!wallCanvas) return;

    const { NX, NY } = sim;

    const ctx = wallCanvas.getContext("2d");
    const cW = wallCanvas.width;
    const cH = wallCanvas.height;
    const sx = cW / NX;
    const sy = cH / NY;

    ctx.clearRect(0, 0, cW, cH);

    const allStrokes = currentStroke
      ? [...strokes, currentStroke]
      : strokes;

    if (allStrokes.length === 0) return;

    const lineW = sx * (BRUSH_R + 0.5);

    for (const pts of allStrokes) {
      if (pts.length === 0) continue;

      ctx.save();
      ctx.strokeStyle = "rgba(175, 198, 230, 0.75)";
      ctx.lineWidth   = lineW * 1.25;
      ctx.lineCap     = "round";
      ctx.lineJoin    = "round";
      drawCatmullRom(ctx, pts, sx, sy);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "#292a2f";
      ctx.lineWidth   = lineW;
      ctx.lineCap     = "round";
      ctx.lineJoin    = "round";
      drawCatmullRom(ctx, pts, sx, sy);
      ctx.stroke();
      ctx.restore();
    }
  }

  return {
    beginStroke,
    paintStroke,
    endStroke,
    clearDrawing,
    redrawWalls,
    paintBrush,
    get strokes() {
      return strokes;
    }
    
  };
}

// ---------------------------------------------------------
// drawCatmullRom(ctx, pts, sx, sy)
// ---------------------------------------------------------
function drawCatmullRom(ctx, pts, sx, sy) {
  if (pts.length === 1) {
    const x = pts[0].x * sx;
    const y = pts[0].y * sy;
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(pts[0].x * sx, pts[0].y * sy);

  for (let i = 1; i < pts.length - 1; i++) {
    const mx = ((pts[i].x + pts[i + 1].x) / 2) * sx;
    const my = ((pts[i].y + pts[i + 1].y) / 2) * sy;
    ctx.quadraticCurveTo(
      pts[i].x * sx,
      pts[i].y * sy,
      mx, my
    );
  }

  const last = pts[pts.length - 1];
  ctx.lineTo(last.x * sx, last.y * sy);
}


