// src/renderBody.js
// ============================================================
// Body overlay renderer only.
// Compare-mode ready: pass a canvas per sim, no module-global ctx.
// Updated for global fidelity presets: renderer owns NX/NY.
//
// Change 2 additions:
// - NACA2412 rendering
// - tandem cylinders rendering
// - square cylinder rendering
// ============================================================

export class BodyRenderer {
  constructor(canvas, NX, NY) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.NX = NX;
    this.NY = NY;
  }

  draw(params) {
    const ctx = this.ctx;
    if (!ctx) return;

    const Wc = ctx.canvas.width;
    const Hc = ctx.canvas.height;

    const sx = Wc / this.NX;
    const sy = Hc / this.NY;

    ctx.clearRect(0, 0, Wc, Hc);
    ctx.save();
    ctx.shadowColor = "rgba(130,190,255,0.5)";
    ctx.shadowBlur  = 14 * Math.min(sx, sy);

    // -----------------------------------------------------
    // Cylinder
    // -----------------------------------------------------
    if (params.type === "cylinder") {
      drawCylinder(ctx, params, sx, sy);


    // -----------------------------------------------------
    // Airfoil families
    // -----------------------------------------------------
    } else if (params.type === "airfoil" || params.type === "naca2412") {
      const pts = nacaContourExpanded(params, 0.5);

      ctx.beginPath();
      ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
      for (let k = 1; k < pts.length; k++) {
        ctx.lineTo(pts[k][0] * sx, pts[k][1] * sy);
      }
      ctx.closePath();

      const grad = ctx.createLinearGradient(
        0, (params.cy - params.chord * 0.25) * sy,
        0, (params.cy + params.chord * 0.25) * sy
      );
      grad.addColorStop(0,    "#48506a");
      grad.addColorStop(0.45, "#1a1f2e");
      grad.addColorStop(1,    "#0a0d16");

      ctx.fillStyle   = grad;
      ctx.strokeStyle = "rgba(175,198,230,0.8)";
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();

    // -----------------------------------------------------
    // Square cylinder
    // -----------------------------------------------------
    } else if (params.type === "square-cylinder") {
      const { cx, cy, size, alpha = 0 } = params;
      const pad  = 0;
      const half = size / 2 + pad;  // ← one half value used for everything
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

      // Single path — fill AND stroke use same corners
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
      grad.addColorStop(0,   "#48506a");
      grad.addColorStop(0.5, "#1a1f2e");
      grad.addColorStop(1,   "#0a0d16");

      ctx.fillStyle   = grad;
      ctx.strokeStyle = "rgba(175,198,230,0.8)";
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();  

    // -----------------------------------------------------
    // Blocks
    // -----------------------------------------------------
    } else if (params.type === "blocks") {
      const { blocks, blockSize } = params;
      const pad = 1.5;
      const half = blockSize / 2 + pad;

      for (const [bx, by] of blocks) {
        const pw = half * 2 * sx;
        const ph = half * 2 * sy;
        const px = bx * sx - half * sx;
        const py = by * sy - half * sy;
        const r  = Math.min(pw, ph) * 0.2;

        const grad = ctx.createRadialGradient(
          bx * sx, by * sy, 0,
          bx * sx, by * sy, Math.max(pw, ph) * 0.7
        );
        grad.addColorStop(0,   "#48506a");
        grad.addColorStop(0.5, "#1a1f2e");
        grad.addColorStop(1,   "#0a0d16");

        ctx.beginPath();
        ctx.roundRect(px, py, pw, ph, r);
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.strokeStyle = "rgba(175,198,230,0.8)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

    }

    ctx.restore();
  }

  clear() {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }
}

// ---------------------------------------------------------
// drawCylinder(ctx, params, sx, sy)
// ---------------------------------------------------------
function drawCylinder(ctx, params, sx, sy, pad = 0.5) {
  const rx = (params.radius + pad) * sx;
  const ry = (params.radius + pad) * sy;

  const grad = ctx.createRadialGradient(
    params.cx * sx, params.cy * sy, 0,
    params.cx * sx, params.cy * sy, rx
  );
  grad.addColorStop(0,   "#48506a");
  grad.addColorStop(0.5, "#1a1f2e");
  grad.addColorStop(1,   "#0a0d16");

  ctx.beginPath();
  ctx.ellipse(params.cx * sx, params.cy * sy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = "rgba(175,198,230,0.8)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ---------------------------------------------------------
// nacaContourExpanded(params, padLattice)
// ---------------------------------------------------------
function nacaContourExpanded(params, padLattice) {
  const raw = nacaContour(params);
  const n   = raw.length;
  const out = [];

  for (let i = 0; i < n; i++) {
    const prev = raw[(i - 1 + n) % n];
    const next = raw[(i + 1) % n];
    const tx = next[0] - prev[0];
    const ty = next[1] - prev[1];
    const len = Math.sqrt(tx * tx + ty * ty) || 1;
    const nx = -ty / len;
    const ny =  tx / len;

    out.push([
      raw[i][0] + nx * padLattice,
      raw[i][1] + ny * padLattice,
    ]);
  }

  return out;
}

// ---------------------------------------------------------
// nacaContour(params)
// ---------------------------------------------------------
// Supports both:
// - NACA0012-style symmetric params
// - NACA2412-style cambered params via params.profile = "2412"
function nacaContour(params) {
  const { cx, cy, chord, alpha } = params;
  const profile = params.profile || "0012";

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
    if (xn >= 1.0) yt = 0;

    let yc = 0.0;
    let dyc_dx = 0.0;

    if (m > 0 && p > 0 && p < 1) {
      if (xn < p) {
        yc      = m * chord * (2 * p * xn - xn * xn) / (p * p);
        dyc_dx  = (2 * m / (p * p)) * (p - xn);  // per unit xn
      } else {
        const op = 1 - p;
        yc      = m * chord * (1 - 2*p + 2*p*xn - xn*xn) / (op * op);
        dyc_dx  = (2 * m / (op * op)) * (p - xn);  // per unit xn
      }
    }

    const theta = Math.atan(dyc_dx);  

    const xBase = (xn - 0.5) * chord;
    const xCam  = xBase;
    const yCam  = -yc;

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


