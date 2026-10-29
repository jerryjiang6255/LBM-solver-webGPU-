// src/overlayRenderer.js
// ============================================================
// Vector / grid / distribution overlays, drawn with the CPU
// renderer's Canvas2D code (render.js) on a transparent canvas
// stacked directly above the WebGPU field canvas.
//
// Inputs are sparse GPU samples (gather_vec / gather_dist in
// gpu/solver.wgsl):
//   vector samples : every 8th node,  (ux, uy, solid, 0)
//   dist samples   : every 16th node, (f0..f8, solid)
// ============================================================

export function OverlayRenderer(canvas, NX, NY) {
  this.NX = NX;
  this.NY = NY;
  this.canvas = canvas;
  this.ctx = canvas.getContext("2d");
  this.W = canvas.width;
  this.H = canvas.height;
}

OverlayRenderer.prototype.resize = function (W, H) {
  this.canvas.width = W;
  this.canvas.height = H;
  this.W = W;
  this.H = H;
};

// Same conditions as Renderer.prototype.draw() in render.js.
OverlayRenderer.prototype.draw = function (gather, options) {
  const ctx = this.ctx;
  ctx.clearRect(0, 0, this.W, this.H);

  const mode = (options && options.mode) || "speed";
  const showVectors = (options && options.showVectors) || false;
  const showGrid = (options && options.showGrid) || false;
  const gridFine = (options && options.gridFine !== undefined) ? options.gridFine : true;
  const showDist = (options && options.showDist) || false;
  const scale = (options && options.scale) || 0.2;

  if (showVectors && mode === "speed" && gather && gather.vec) {
    this.drawVectorField(gather.vec, {
      spacing: 8,
      scale,
      maxLen: 1.2,
    });
  }

  if (showGrid) {
    this.drawGrid(gather ? gather.dist : null, {
      showFine:    gridFine,
      showCoarse:  true,
      coarseStep:  8,
      showDist:    showDist,
      distSpacing: 16,
      distScale:   48,
    });
  }
};

// ---------------------------------------------------------
// Vector overlay (render.js)
// ---------------------------------------------------------
OverlayRenderer.prototype.drawVectorField = function (samples, options) {
  const NX = this.NX;
  const NY = this.NY;

  const spacing = (options && options.spacing) || 8;
  const scale   = (options && options.scale) || 0.15;
  const maxLen  = (options && options.maxLen) || 0.85;

  const ctx = this.ctx;
  const Wc  = this.W;
  const Hc  = this.H;

  const cellW = Wc / NX;
  const cellH = Hc / NY;
  const maxPx = Math.min(cellW, cellH) * spacing * maxLen;

  const cols = Math.ceil(NX / spacing);

  ctx.save();
  ctx.lineCap = "round";

  for (let gy = 0; gy < NY; gy += spacing) {
    for (let gx = 0; gx < NX; gx += spacing) {
      const k = ((gy / spacing) * cols + gx / spacing) * 4;
      if (samples[k + 2]) continue;

      const u = samples[k];
      const v = samples[k + 1];
      const spd = Math.sqrt(u * u + v * v);
      if (spd < 1e-6) continue;

      const t = Math.min(spd / scale, 1.0);
      const len = t * maxPx;

      const dx = (u / spd) * len;
      const dy = (v / spd) * len;

      const px = gx * cellW;
      const py = gy * cellH;
      const tx = px + dx;
      const ty = py + dy;

      const alpha = 0.25 + 0.75 * t;
      ctx.strokeStyle = `rgba(255,255,255,${alpha.toFixed(2)})`;
      ctx.lineWidth = Math.max(1.5, cellW * 0.25);

      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(tx, ty);
      ctx.stroke();

      if (len > 3) {
        const headLen   = Math.min(len * 0.45, cellW * 1.5);
        const headAngle = Math.PI / 6;
        const angle     = Math.atan2(dy, dx);

        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(
          tx - headLen * Math.cos(angle - headAngle),
          ty - headLen * Math.sin(angle - headAngle)
        );
        ctx.moveTo(tx, ty);
        ctx.lineTo(
          tx - headLen * Math.cos(angle + headAngle),
          ty - headLen * Math.sin(angle + headAngle)
        );
        ctx.stroke();
      }
    }
  }

  ctx.restore();
};

// ---------------------------------------------------------
// Grid / distribution overlay (render.js)
// ---------------------------------------------------------
OverlayRenderer.prototype.drawGrid = function (distSamples, options) {
  const NX = this.NX;
  const NY = this.NY;

  const showFine    = (options && options.showFine !== undefined) ? options.showFine : true;
  const showCoarse  = (options && options.showCoarse !== undefined) ? options.showCoarse : true;
  const coarseStep  = (options && options.coarseStep) || 8;
  const showDist    = (options && options.showDist) || false;
  const distSpacing = (options && options.distSpacing) || 16;
  const distScale   = (options && options.distScale) || 80;

  const ctx   = this.ctx;
  const Wc    = this.W;
  const Hc    = this.H;
  const cellW = Wc / NX;
  const cellH = Hc / NY;

  ctx.save();

  if (showFine) {
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();

    for (let x = 0; x <= NX; x++) {
      const px = x * cellW;
      ctx.moveTo(px, 0);
      ctx.lineTo(px, Hc);
    }
    for (let y = 0; y <= NY; y++) {
      const py = y * cellH;
      ctx.moveTo(0, py);
      ctx.lineTo(Wc, py);
    }
    ctx.stroke();
  }

  if (showCoarse) {
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 0.75;
    ctx.beginPath();

    for (let x = 0; x <= NX; x += coarseStep) {
      const px = x * cellW;
      ctx.moveTo(px, 0);
      ctx.lineTo(px, Hc);
    }
    for (let y = 0; y <= NY; y += coarseStep) {
      const py = y * cellH;
      ctx.moveTo(0, py);
      ctx.lineTo(Wc, py);
    }
    ctx.stroke();
  }

  if (showDist && distSamples) {
    const distCols = Math.ceil(NX / distSpacing);
    const CX_D = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    const CY_D = [0, 0, -1, 0, 1, -1, -1, 1, 1];

    const DIR_COLORS = [
      "rgba(255,255,255,0.9)",
      "rgba(100,220,255,0.85)",
      "rgba(100,220,255,0.85)",
      "rgba(100,220,255,0.85)",
      "rgba(100,220,255,0.85)",
      "rgba(255,220,80,0.8)",
      "rgba(255,220,80,0.8)",
      "rgba(255,220,80,0.8)",
      "rgba(255,220,80,0.8)",
    ];

    ctx.lineWidth = Math.max(0.6, cellW * 0.12);
    ctx.lineCap = "round";

    for (let gy = 0; gy < NY; gy += distSpacing) {
      for (let gx = 0; gx < NX; gx += distSpacing) {
        const k = ((gy / distSpacing) * distCols + gx / distSpacing) * 10;
        if (distSamples[k + 9]) continue;

        const px = gx * cellW;
        const py = gy * cellH;

        ctx.fillStyle = "rgba(255,255,255,0.6)";
        ctx.beginPath();
        ctx.arc(px, py, Math.max(1, cellW * 0.15), 0, Math.PI * 2);
        ctx.fill();

        for (let i = 0; i < 9; i++) {
          const fi = distSamples[k + i];
          const armLen = fi * distScale * Math.min(cellW, cellH);

          if (i === 0) {
            const r = Math.min(armLen * 0.5, cellW * 0.4);
            if (r > 0.5) {
              ctx.strokeStyle = DIR_COLORS[0];
              ctx.beginPath();
              ctx.arc(px, py, r, 0, Math.PI * 2);
              ctx.stroke();
            }
            continue;
          }

          const ex = px + CX_D[i] * armLen;
          const ey = py + CY_D[i] * armLen;

          ctx.strokeStyle = DIR_COLORS[0];
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(ex, ey);
          ctx.stroke();

          if (armLen > cellW * 0.4) {
            const angle   = Math.atan2(CY_D[i], CX_D[i]);
            const headLen = armLen * 0.3;
            const headAng = Math.PI / 5;

            ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(
              ex - headLen * Math.cos(angle - headAng),
              ey - headLen * Math.sin(angle - headAng)
            );
            ctx.moveTo(ex, ey);
            ctx.lineTo(
              ex - headLen * Math.cos(angle + headAng),
              ey - headLen * Math.sin(angle + headAng)
            );
            ctx.stroke();
          }
        }
      }
    }
  }

  ctx.restore();
}; 
