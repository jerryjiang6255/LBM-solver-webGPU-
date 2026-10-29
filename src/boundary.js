// src/boundary.js
// ============================================================
// Compare-mode ready boundary conditions.
// Updated for global fidelity presets: no fixed NX/NY imports.
// ============================================================

import { feqAt } from "./simInstance.js";

// ---------------------------------------------------------
// Sponge config
// ---------------------------------------------------------
const SPONGE_WIDTH    = 28;
const SPONGE_STRENGTH = 0.6;

// Cache sponge weights by NX so they are built once per grid width
const spongeCache = new Map();

// ---------------------------------------------------------
// getSpongeWeights(NX)
// ---------------------------------------------------------
function getSpongeWeights(NX) {
  let spongeW = spongeCache.get(NX);
  if (spongeW) return spongeW;

  spongeW = new Float32Array(NX);

  const width = Math.min(SPONGE_WIDTH, Math.max(1, NX - 2));
  const start = NX - 1 - width;

  for (let x = start; x < NX; x++) {
    const t = (x - start) / width;
    spongeW[x] = SPONGE_STRENGTH * t * t;
  }

  spongeCache.set(NX, spongeW);
  return spongeW;
}

// ---------------------------------------------------------
// setNozzleInlet(sim, cy, diameter, uCoflow)
// ---------------------------------------------------------
export function setNozzleInlet(sim, cy, diameter, uCoflow) {
  sim.nozzle.active  = true;
  sim.nozzle.cy      = cy;
  sim.nozzle.halfD   = diameter / 2;
  sim.nozzle.uCoflow = uCoflow;
}

// ---------------------------------------------------------
// clearNozzleInlet(sim)
// ---------------------------------------------------------
export function clearNozzleInlet(sim) {
  sim.nozzle.active = false;
}

// ---------------------------------------------------------
// applyBoundaryConditions(sim, u0)
// ---------------------------------------------------------
export function applyBoundaryConditions(sim, u0) {
  const { NX, NY, Q, f, rho, ux, uy, solid, nozzle } = sim;
  const idx = (x, y) => y * NX + x;

  // -------------------------------------------------------
  // Lid-driven cavity mode
  // -------------------------------------------------------
  if (sim.params?.bodyType === "lid-driven-cavity") {
    // Moving top lid on the first interior fluid row
    const yTop = 2;
    for (let x = 1; x < NX - 1; x++) {
      const n = idx(x, yTop);
      if (solid[n]) continue;

      const base = n * Q;
      for (let i = 0; i < Q; i++) {
        f[base + i] = feqAt(i, 1.0, u0, 0.0);
      }
    }

    const yBot = NY - 3;
    for (let x = 1; x < NX - 1; x++) {
      const n = idx(x, yBot);
      if (solid[n]) continue;

      const base = n * Q;
      for (let i = 0; i < Q; i++) {
        f[base + i] = feqAt(i, 1.0, 0.0, 0.0);
      }
    }


    // Optional corner-adjacent stabilization for left/right interior columns
    // for (let y = 1; y < NY - 1; y++) {
    //   const nL = idx(1, y);
    //   const nR = idx(NX - 2, y);

    //   if (!solid[nL]) {
    //     const baseL = nL * Q;
    //     for (let i = 0; i < Q; i++) {
    //       f[baseL + i] = feqAt(i, 1.0, 0.0, 0.0);
    //     }
    //   }

    //   if (!solid[nR]) {
    //     const baseR = nR * Q;
    //     for (let i = 0; i < Q; i++) {
    //       f[baseR + i] = feqAt(i, 1.0, 0.0, 0.0);
    //     }
    //   }
    // }

    return;
  }

  const spongeW = getSpongeWeights(NX);

  // ---- left column: split inlet for nozzle, uniform for everything else ----
  for (let y = 0; y < NY; y++) {
    const n = idx(0, y);
    if (solid[n]) continue;

    let inletU;
    if (nozzle.active) {
      const distFromCenter = Math.abs(y - nozzle.cy);
      inletU = distFromCenter <= nozzle.halfD ? u0 : nozzle.uCoflow;
    } else {
      inletU = u0;
    }

    const base = n * Q;
    for (let i = 0; i < Q; i++) {
      f[base + i] = feqAt(i, 1.0, inletU, 0.0);
    }
  }

  // ---- right columns: sponge ----
  const width = Math.min(SPONGE_WIDTH, Math.max(1, NX - 2));
  const xSpongeStart = NX - 1 - width;

  for (let x = xSpongeStart; x < NX - 1; x++) {
    const s = spongeW[x];

    for (let y = 0; y < NY; y++) {
      const n = idx(x, y);
      if (solid[n]) continue;

      const r = rho[n];
      const u = ux[n];
      const v = uy[n];
      const base = n * Q;

      for (let i = 0; i < Q; i++) {
        const off = base + i;
        f[off] = (1 - s) * f[off] + s * feqAt(i, r, u, v);
      }
    }
  }

  // ---- outlet zero-gradient copy ----
  if (NX >= 2) {
    for (let y = 0; y < NY; y++) {
      const nSrc = idx(NX - 2, y);
      const nDst = idx(NX - 1, y);

      for (let i = 0; i < Q; i++) {
        f[nDst * Q + i] = f[nSrc * Q + i];
      }
    }
  }

  // ---- top and bottom: coflow or freestream depending on mode ----
  const wallU = nozzle.active ? nozzle.uCoflow : u0;

  for (let x = 0; x < NX; x++) {
    const nb = idx(x, 0);
    const nt = idx(x, NY - 1);

    const baseB = nb * Q;
    const baseT = nt * Q;

    for (let i = 0; i < Q; i++) {
      const eq = feqAt(i, 1.0, wallU, 0.0);
      f[baseB + i] = eq;
      f[baseT + i] = eq;
    }
  }
}


