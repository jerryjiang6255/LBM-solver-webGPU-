// src/render/bodyRenderer.js
// CPU-side body overlay renderer.
// Renders only base body geometry metadata, not user-painted walls.

import { BODY_TYPES } from "../config.js";

export class BodyRenderer {
  constructor(canvas, NX, NY) {
    this.canvas = canvas;
    this.ctx = canvas?.getContext("2d") ?? null;
    this.NX = NX;
    this.NY = NY;
  }

  attachCanvas(canvas) {
    this.canvas = canvas;
    this.ctx = canvas?.getContext("2d") ?? null;
  }

  setDims(NX, NY) {
    this.NX = NX;
    this.NY = NY;
  }

  clear() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(meta) {
    const ctx = this.ctx;
    if (!ctx || !meta) return;

    const Wc = ctx.canvas.width;
    const Hc = ctx.canvas.height;

    const sx = Wc / this.NX;
    const sy = Hc / this.NY;

    ctx.clearRect(0, 0, Wc, Hc);
    ctx.save();

    ctx.shadowColor = "rgba(130,190,255,0.5)";
    ctx.shadowBlur = 14 * Math.min(sx, sy);

    switch (meta.type) {
      case BODY_TYPES.CYLINDER:
        drawCylinder(ctx, meta.visualParams ?? meta, sx, sy);
        break;

      case BODY_TYPES.LID_DRIVEN_CAVITY:
        drawLidDrivenCavity(ctx, meta.visualParams ?? meta, this.NX, this.NY, sx, sy);
        break;

      case BODY_TYPES.AIRFOIL:
      case BODY_TYPES.NACA2412:
        drawAirfoil(ctx, meta, sx, sy);
        break;

      case BODY_TYPES.SQUARE_CYLINDER:
        drawSquareCylinder(ctx, meta.visualParams ?? meta, sx, sy);
        break;

      case BODY_TYPES.BLOCKS:
        drawBlocks(ctx, meta.visualParams ?? meta, sx, sy);
        break;

      case BODY_TYPES.NOZZLE:
        drawNozzle(ctx, meta.visualParams ?? meta, sx, sy);
        break;

      case BODY_TYPES.EMPTY:
      default:
        break;
    }

    ctx.restore();
  }
}

// ---------------------------------------------------------
// Shape renderers
// ---------------------------------------------------------
function drawCylinder(ctx, params, sx, sy, pad = 0.5) {
  const rx = (params.radius + pad) * sx;
  const ry = (params.radius + pad) * sy;

  const grad = ctx.createRadialGradient(
    params.cx * sx, params.cy * sy, 0,
    params.cx * sx, params.cy * sy, rx
  );
  grad.addColorStop(0, "#48506a");
  grad.addColorStop(0.5, "#1a1f2e");
  grad.addColorStop(1, "#0a0d16");

  ctx.beginPath();
  ctx.ellipse(params.cx * sx, params.cy * sy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawLidDrivenCavity(ctx, params, NX, NY, sx, sy) {
  const wallThickness = params.wallThickness ?? 1;
  const cavityAspect = params.cavityAspect ?? 1.0;

  const outerLeft = wallThickness;
  const outerRight = NX - 1 - wallThickness;
  const outerBottom = wallThickness;
  const outerTop = NY - 1 - wallThickness;

  const cavityHeight = (outerTop - outerBottom) + 1 - 2 * wallThickness;
  const cavityWidth = Math.max(4, Math.round(cavityHeight * cavityAspect));

  const midX = 0.5 * (outerLeft + outerRight);
  const cavityLeftInner = Math.round(midX - cavityWidth / 2);
  const cavityRightInner = Math.round(midX + cavityWidth / 2);

  const tX = Math.max(wallThickness * sx, 14);
  const tY = Math.max(wallThickness * sy, 14);

  const outerX = outerLeft * sx;
  const outerY = outerBottom * sy;
  const outerW = (outerRight - outerLeft + 1) * sx;
  const outerH = (outerTop - outerBottom + 1) * sy;

  const innerX = outerX + tX;
  const innerY = outerY + tY;
  const innerW = outerW - 2 * tX;
  const innerH = outerH - 2 * tY;

  const wall1X = cavityLeftInner * sx;
  const wall2X = cavityRightInner * sx;
  const wallW = Math.max(wallThickness * sx, 12);

  const rOuter = Math.max(8, Math.min(outerW, outerH) * 0.025);
  const rInner = Math.max(6, Math.min(innerW, innerH) * 0.02);

  ctx.beginPath();
  roundRectPath(ctx, outerX, outerY, outerW, outerH, rOuter);
  roundRectPath(ctx, innerX, innerY, innerW, innerH, rInner);
  ctx.rect(wall1X - 0.5 * wallW, innerY, wallW, innerH);
  ctx.rect(wall2X - 0.5 * wallW, innerY, wallW, innerH);

  const grad = ctx.createLinearGradient(0, outerY, 0, outerY + outerH);
  grad.addColorStop(0.0, "#48506a");
  grad.addColorStop(0.45, "#1a1f2e");
  grad.addColorStop(1.0, "#0a0d16");

  ctx.fillStyle = grad;
  ctx.fill("evenodd");

  ctx.beginPath();
  roundRectPath(ctx, outerX + 0.5, outerY + 0.5, outerW - 1, outerH - 1, rOuter);
  ctx.strokeStyle = "rgba(175,198,230,0.78)";
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.beginPath();
  roundRectPath(ctx, innerX + 0.5, innerY + 0.5, innerW - 1, innerH - 1, rInner);
  ctx.strokeStyle = "rgba(175,198,230,0.40)";
  ctx.lineWidth = 1.1;
  ctx.stroke();
}

function drawAirfoil(ctx, meta, sx, sy) {
  const pts = nacaContourExpanded(meta, 0.5);

  if (!pts.length) return;

  ctx.beginPath();
  ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
  for (let k = 1; k < pts.length; k++) {
    ctx.lineTo(pts[k][0] * sx, pts[k][1] * sy);
  }
  ctx.closePath();

  const cy = meta.cy ?? meta.visualParams?.cy ?? 0;
  const chord = meta.chord ?? meta.visualParams?.chord ?? 1;

  const grad = ctx.createLinearGradient(
    0, (cy - chord * 0.25) * sy,
    0, (cy + chord * 0.25) * sy
  );
  grad.addColorStop(0, "#48506a");
  grad.addColorStop(0.45, "#1a1f2e");
  grad.addColorStop(1, "#0a0d16");

  ctx.fillStyle = grad;
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
}

function drawSquareCylinder(ctx, params, sx, sy) {
  const { cx, cy, size, alpha = 0 } = params;
  const half = size / 2;

  const cosA = Math.cos(alpha);
  const sinA = Math.sin(alpha);

  const corners = [
    [-half, -half],
    [ half, -half],
    [ half,  half],
    [-half,  half],
  ].map(([x, y]) => [
    cx + x * cosA - y * sinA,
    cy + x * sinA + y * cosA,
  ]);

  ctx.beginPath();
  ctx.moveTo(corners[0][0] * sx, corners[0][1] * sy);
  for (let k = 1; k < corners.length; k++) {
    ctx.lineTo(corners[k][0] * sx, corners[k][1] * sy);
  }
  ctx.closePath();

  const grad = ctx.createLinearGradient(
    (cx - half) * sx, (cy - half) * sy,
    (cx + half) * sx, (cy + half) * sy
  );
  grad.addColorStop(0, "#48506a");
  grad.addColorStop(0.5, "#1a1f2e");
  grad.addColorStop(1, "#0a0d16");

  ctx.fillStyle = grad;
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
}

function drawBlocks(ctx, params, sx, sy) {
  const { blocks, blockSize } = params;
  const half = blockSize / 2 + 1.5;

  for (const [bx, by] of blocks) {
    const pw = half * 2 * sx;
    const ph = half * 2 * sy;
    const px = bx * sx - half * sx;
    const py = by * sy - half * sy;
    const r = Math.min(pw, ph) * 0.2;

    const grad = ctx.createRadialGradient(
      bx * sx, by * sy, 0,
      bx * sx, by * sy, Math.max(pw, ph) * 0.7
    );
    grad.addColorStop(0, "#48506a");
    grad.addColorStop(0.5, "#1a1f2e");
    grad.addColorStop(1, "#0a0d16");

    ctx.beginPath();
    roundRectPath(ctx, px, py, pw, ph, r);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = "rgba(175,198,230,0.8)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
}

function drawNozzle(ctx, params, sx, sy) {
  const { cy, diameter, length } = params;
  const halfD = diameter / 2;
  const wallThickness = 4;

  const topY1 = (cy - halfD - wallThickness) * sy;
  const topY2 = (cy - halfD) * sy;
  const botY1 = (cy + halfD) * sy;
  const botY2 = (cy + halfD + wallThickness) * sy;
  const endX = (length + 0.9) * sx;

  const gradTop = ctx.createLinearGradient(0, topY1, 0, topY2);
  gradTop.addColorStop(0, "#48506a");
  gradTop.addColorStop(0.5, "#1a1f2e");
  gradTop.addColorStop(1, "#0a0d16");

  const gradBot = ctx.createLinearGradient(0, botY1, 0, botY2);
  gradBot.addColorStop(0, "#0a0d16");
  gradBot.addColorStop(0.5, "#1a1f2e");
  gradBot.addColorStop(1, "#48506a");

  const startX = -10;

  ctx.beginPath();
  ctx.rect(startX, topY1, endX - startX, topY2 - topY1);
  ctx.fillStyle = gradTop;
  ctx.fill();
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.2;
  ctx.stroke();

  ctx.beginPath();
  ctx.rect(startX, botY1, endX - startX, botY2 - botY1);
  ctx.fillStyle = gradBot;
  ctx.fill();
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.2;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(endX, topY1);
  ctx.lineTo(endX, topY2);
  ctx.moveTo(endX, botY1);
  ctx.lineTo(endX, botY2);
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ---------------------------------------------------------
// Airfoil contour helpers
// ---------------------------------------------------------
function nacaContourExpanded(meta, padLattice) {
  const raw = nacaContour(meta);
  const n = raw.length;
  const out = [];

  for (let i = 0; i < n; i++) {
    const prev = raw[(i - 1 + n) % n];
    const next = raw[(i + 1) % n];

    const tx = next[0] - prev[0];
    const ty = next[1] - prev[1];
    const len = Math.sqrt(tx * tx + ty * ty) || 1.0;

    const nx = -ty / len;
    const ny = tx / len;

    out.push([
      raw[i][0] + nx * padLattice,
      raw[i][1] + ny * padLattice,
    ]);
  }

  return out;
}

function nacaContour(meta) {
  const cx = meta.cx ?? meta.visualParams?.cx ?? 0;
  const cy = meta.cy ?? meta.visualParams?.cy ?? 0;
  const chord = meta.chord ?? meta.visualParams?.chord ?? 1;
  const alpha = meta.alpha ?? meta.visualParams?.alpha ?? 0;

  const profile = meta.profile ?? meta.visualParams?.profile ?? "0012";

  const cosA = Math.cos(alpha);
  const sinA = Math.sin(alpha);
  const nPts = 120;

  const upper = [];
  const lower = [];

  let m = 0.0;
  let p = 0.0;
  let t = 0.12;

  if (profile === "2412") {
    m = 0.02;
    p = 0.4;
    t = 0.12;
  }

  for (let s = 0; s <= nPts; s++) {
    const xn = 0.5 * (1 - Math.cos(Math.PI * s / nPts));

    let yt = (t / 0.2) * chord * (
       0.2969 * Math.sqrt(xn)
      - 0.1260 * xn
      - 0.3516 * xn * xn
      + 0.2843 * xn * xn * xn
      - 0.1015 * xn * xn * xn * xn
    );
    if (xn >= 1.0) yt = 0.0;

    let yc = 0.0;
    let dycDx = 0.0;

    if (m > 0 && p > 0 && p < 1) {
      if (xn < p) {
        yc = m * chord * (2 * p * xn - xn * xn) / (p * p);
        dycDx = (2 * m / (p * p)) * (p - xn);
      } else {
        const op = 1 - p;
        yc = m * chord * (1 - 2 * p + 2 * p * xn - xn * xn) / (op * op);
        dycDx = (2 * m / (op * op)) * (p - xn);
      }
    }

    const theta = Math.atan(dycDx);

    const xBase = (xn - 0.5) * chord;
    const xCam = xBase;
    const yCam = -yc;

    const xu = xCam - yt * Math.sin(theta);
    const yu = yCam + yt * Math.cos(theta);
    const xl = xCam + yt * Math.sin(theta);
    const yl = yCam - yt * Math.cos(theta);

    upper.push([
      cx + xu * cosA - yu * sinA,
      cy + xu * sinA + yu * cosA,
    ]);

    lower.push([
      cx + xl * cosA - yl * sinA,
      cy + xl * sinA + yl * cosA,
    ]);
  }

  return [...upper, ...[...lower].reverse()];
}

// ---------------------------------------------------------
// Canvas path helper
// ---------------------------------------------------------
function roundRectPath(ctx, x, y, w, h, r) {
  const rr = Math.min(r, 0.5 * w, 0.5 * h);
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}


