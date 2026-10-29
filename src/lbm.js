// src/lbm.js
// ============================================================
// Immutable lattice constants + pure helper functions only.
// All mutable simulation state now lives in simInstance.js
// Grid dimensions are no longer defined here so global fidelity
// presets can create sims at different resolutions.
// ============================================================

// ---------------------------------------------------------
// D2Q9 LATTICE CONSTANTS
// ---------------------------------------------------------
export const Q = 9;

// ---------------------------------------------------------
// REFERENCE CONSTANTS
// ---------------------------------------------------------
export const RHO0 = 1.0;
export const U0   = 0.06; // lattice-unit reference velocity

// ---------------------------------------------------------
// D2Q9 LATTICE VECTORS
// ---------------------------------------------------------
export const CX = new Int8Array([0, 1, 0, -1, 0, 1, -1, -1, 1]);
export const CY = new Int8Array([0, 0, 1, 0, -1, 1, 1, -1, -1]);

export const W = new Float32Array([
  4 / 9,
  1 / 9, 1 / 9, 1 / 9, 1 / 9,
  1 / 36, 1 / 36, 1 / 36, 1 / 36,
]);

export const OPP = new Uint8Array([0, 3, 4, 1, 2, 7, 8, 5, 6]);

// ---------------------------------------------------------
// idx(x, y, NX)
// ---------------------------------------------------------
// Grid-aware indexing helper.
// NX must be passed explicitly now that grid size is dynamic.
export function idx(x, y, NX) {
  return y * NX + x;
}

// ---------------------------------------------------------
// nodeCount(NX, NY)
// ---------------------------------------------------------
export function nodeCount(NX, NY) {
  return NX * NY;
}

// ---------------------------------------------------------
// feqAt(i, r, u, v)
// ---------------------------------------------------------
export function feqAt(i, r, u, v) {
  const cu = CX[i] * u + CY[i] * v;
  const uu = u * u + v * v;
  return W[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
}


