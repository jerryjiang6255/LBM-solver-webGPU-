// src/boundary.js
// ============================================================
// CPU boundary conditions, used once on the initial field before it is
// uploaded to the GPU (the GPU applies the same conditions every step
// in solver.wgsl stream_main / outlet_main).
// ============================================================

import { feqAt } from "./simInstance.js";
import { spongeLayout } from "./caseSpec.js";

// ---------------------------------------------------------
// applyBoundaryConditions(sim, u0)
// ---------------------------------------------------------
export function applyBoundaryConditions(sim, u0) {
  const { NX, NY, Q, f, rho, ux, uy, solid } = sim;
  const idx = (x, y) => y * NX + x;

  // ---- left column: uniform inlet ----
  for (let y = 0; y < NY; y++) {
    const n = idx(0, y);
    if (solid[n]) continue;

    const base = n * Q;
    for (let i = 0; i < Q; i++) {
      f[base + i] = feqAt(i, 1.0, u0, 0.0);
    }
  }

  // ---- right columns: sponge (same layout as the GPU) ----
  const sponge = spongeLayout(NX);

  for (let x = sponge.start; x < NX - 1; x++) {
    const t = (x - sponge.start) / sponge.width;
    const s = sponge.strength * t * t;

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

  // ---- top and bottom: freestream ----
  for (let x = 0; x < NX; x++) {
    const nb = idx(x, 0);
    const nt = idx(x, NY - 1);

    const baseB = nb * Q;
    const baseT = nt * Q;

    for (let i = 0; i < Q; i++) {
      const eq = feqAt(i, 1.0, u0, 0.0);
      f[baseB + i] = eq;
      f[baseT + i] = eq;
    }
  }
}
