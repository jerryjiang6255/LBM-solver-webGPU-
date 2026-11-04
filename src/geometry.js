// src/geometry.js
// ============================================================
// Compare-mode ready geometry builders. Body placement and sizing
// (fractions of the domain) live in caseSpec.js.
// Updated for global fidelity presets: geometry no longer imports
// fixed NX/NY/N from lbm.js. Dimensions are passed in explicitly.
// ============================================================

import { Q, CX, CY } from "./lbm.js";

// BFL q clamping
const Q_MIN = 0.01;
const Q_MAX = 1.0;

// ---------------------------------------------------------
// SDF FUNCTIONS
// ---------------------------------------------------------
function sdfCircle(cx, cy, radius) {
  return (x, y) => Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius;
}

function sdfEmpty() {
  return () => 1.0;
}

function sdfNACA0012(cx, cy, chord, alpha) {
  return sdfNACA4(cx, cy, chord, alpha, 0.0, 0.0, 0.12);
}

// ---------------------------------------------------------
// sdfNACA2412(cx, cy, chord, alpha)
// ---------------------------------------------------------
function sdfNACA2412(cx, cy, chord, alpha) {
  return sdfNACA4(cx, cy, chord, alpha, 0.02, 0.4, 0.12);
}

// ---------------------------------------------------------
// sdfNACA4(...)
// ---------------------------------------------------------
// Approximate SDF-style evaluator for 4-digit NACA airfoils.
// m = max camber, p = camber position, t = thickness ratio
function sdfNACA4(cx, cy, chord, alpha, m, p, t) {
  const cos_a = Math.cos(alpha);
  const sin_a = Math.sin(alpha);

  return (x, y) => {
    const dx =  (x - cx) * cos_a + (y - cy) * sin_a;
    const dy = -(x - cx) * sin_a + (y - cy) * cos_a;

    const xn = dx / chord + 0.5;
    if (xn < 0 || xn > 1) return Math.abs(dx) + Math.abs(dy);

    // Thickness distribution (in lattice units)
    let yt = (t / 0.2) * chord * (
       0.2969 * Math.sqrt(xn)
      - 0.1260 * xn
      - 0.3516 * xn * xn
      + 0.2843 * xn * xn * xn
      - 0.1015 * xn * xn * xn * xn
    );
    if (xn >= 1.0) yt = 0;

    // Camber line — in lattice units (multiply by chord, NOT chord²)
    let yc      = 0.0;
    let dyc_dxn = 0.0;

    if (m > 0 && p > 0 && p < 1) {
      if (xn < p) {
        yc      =  m * chord * (2 * p * xn - xn * xn) / (p * p);
        dyc_dxn = (2 * m * chord / (p * p)) * (p - xn);
      } else {
        const op = 1 - p;
        yc      =  m * chord * (1 - 2*p + 2*p*xn - xn*xn) / (op * op);
        dyc_dxn = (2 * m * chord / (op * op)) * (p - xn);
      }
    }

    // theta is the camber slope angle
    // dyc_dxn is in lattice units per unit xn, so divide by chord to get per-lattice
    const theta = Math.atan(dyc_dxn / chord);

    // dy is measured from chord line — subtract camber to get distance from camber line
    // Negate yc because screen y increases downward but lift is upward
    const yRel  = dy - (-yc);   // ← negated yc fixes upside-down
    const ySurf = yt * Math.cos(theta);

    return Math.abs(yRel) - ySurf;
  };
}

// ---------------------------------------------------------
// sdfExplicitBlocks(blocks, blockSize)
// ---------------------------------------------------------
function sdfExplicitBlocks(blocks, blockSize) {
  const half = blockSize / 2;

  return (x, y) => {
    let minDist = 1.0;
    for (let k = 0; k < blocks.length; k++) {
      const lx = Math.abs(x - blocks[k][0]) - half;
      const ly = Math.abs(y - blocks[k][1]) - half;
      const dist = Math.max(lx, ly);
      if (dist < minDist) minDist = dist;
    }
    return minDist;
  };
}

// ---------------------------------------------------------
// sdfRotatedBox(cx, cy, halfW, halfH, alpha)
// ---------------------------------------------------------
function sdfRotatedBox(cx, cy, halfW, halfH, alpha) {
  const cos_a = Math.cos(alpha);
  const sin_a = Math.sin(alpha);

  return (x, y) => {
    const dx =  (x - cx) * cos_a + (y - cy) * sin_a;
    const dy = -(x - cx) * sin_a + (y - cy) * cos_a;

    const qx = Math.abs(dx) - halfW;
    const qy = Math.abs(dy) - halfH;

    const ox = Math.max(qx, 0);
    const oy = Math.max(qy, 0);
    const outside = Math.sqrt(ox * ox + oy * oy);
    const inside = Math.min(Math.max(qx, qy), 0);

    return outside + inside;
  };
}

// ---------------------------------------------------------
// sdfUnion(...sdfs)
// ---------------------------------------------------------
function sdfUnion(...sdfs) {
  return (x, y) => {
    let d = Infinity;
    for (let k = 0; k < sdfs.length; k++) {
      const dk = sdfs[k](x, y);
      if (dk < d) d = dk;
    }
    return d;
  };
}

// ---------------------------------------------------------
// buildGeometryFromSDF(dims, sdfFunc)
// ---------------------------------------------------------
function buildGeometryFromSDF(dims, sdfFunc) {
  const { NX, NY } = dims;
  const N = NX * NY;

  const sdf          = new Float32Array(N);
  const solid        = new Uint8Array(N);
  const wallDist     = new Float32Array(N);
  const wallAdjacent = new Uint8Array(N);
  const linkMask     = new Uint8Array(Q * N);
  const linkQ        = new Float32Array(Q * N);
  const solidTmp     = [];
  const fluidTmp     = [];

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const n     = y * NX + x;
      const d     = sdfFunc(x, y);
      sdf[n]      = d;
      solid[n]    = d <= 0 ? 1 : 0;
      wallDist[n] = d > 0 ? d : 0.0;
    }
  }

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const n       = y * NX + x;
      const isSolid = solid[n];

      if (isSolid) {
        if (
          x > 0 && x < NX - 1 &&
          y > 0 && y < NY - 1 &&
          (!solid[n - 1] || !solid[n + 1] || !solid[n - NX] || !solid[n + NX])
        ) {
          solidTmp.push(n);
        }
        continue;
      }

      const sdfF = sdf[n];
      let isAdj  = 0;

      for (let i = 1; i < Q; i++) {
        const nx = x + CX[i];
        const ny = y + CY[i];
        if (nx < 0 || nx >= NX || ny < 0 || ny >= NY) continue;

        const ns = ny * NX + nx;
        if (!solid[ns]) continue;

        isAdj = 1;

        const sdfS  = sdf[ns];
        const denom = sdfF - sdfS;
        let q = denom > 1e-10 ? sdfF / denom : 0.5;
        if (q < Q_MIN) q = Q_MIN;
        if (q > Q_MAX) q = Q_MAX;

        const off     = n * Q + i;
        linkMask[off] = 1;
        linkQ[off]    = q;
      }

      wallAdjacent[n] = isAdj;
      if (isAdj) fluidTmp.push(n);
    }
  }

  return {
    solid,
    sdf,
    wallDist,
    wallAdjacent,
    linkMask,
    linkQ,
    solidList:        new Uint32Array(solidTmp),
    fluidSurfaceList: new Uint32Array(fluidTmp),
  };
}

// ---------------------------------------------------------
// Public builders
// ---------------------------------------------------------
export function buildCylinder(dims, cx, cy, radius) {
  return {
    geometry: buildGeometryFromSDF(dims, sdfCircle(cx, cy, radius)),
    charLength: 2 * radius,
  };
}

export function buildAirfoil(dims, cx, cy, chord, alpha) {
  return {
    geometry: buildGeometryFromSDF(dims, sdfNACA0012(cx, cy, chord, alpha)),
    charLength: chord,
  };
}

// ---------------------------------------------------------
// buildNACA2412(dims, cx, cy, chord, alpha)
// ---------------------------------------------------------
export function buildNACA2412(dims, cx, cy, chord, alpha) {
  return {
    geometry: buildGeometryFromSDF(dims, sdfNACA2412(cx, cy, chord, alpha)),
    charLength: chord,
  };
}

// ---------------------------------------------------------
// buildSquareCylinder(dims, cx, cy, size, alpha)
// ---------------------------------------------------------
export function buildSquareCylinder(dims, cx, cy, size, alpha) {
  const half = size / 2;
  return {
    geometry: buildGeometryFromSDF(dims, sdfRotatedBox(cx, cy, half, half, alpha)),
    charLength: size,
  };
}

export function buildEmpty(dims) {
  return {
    geometry: buildGeometryFromSDF(dims, sdfEmpty()),
    charLength: dims.NY,
  };
}

// ---------------------------------------------------------
// buildBlocks(dims, blocks, blockSize)
// blocks: array of [cx, cy] node coordinates (may be fractional).
// ---------------------------------------------------------
export function buildBlocks(dims, blocks, blockSize) {
  return {
    geometry: buildGeometryFromSDF(dims, sdfExplicitBlocks(blocks, blockSize)),
    blocks,
    blockSize,
    charLength: blockSize * 4,
  };
}
