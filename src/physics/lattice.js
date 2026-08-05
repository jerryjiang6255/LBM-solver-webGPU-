// src/physics/lattice.js
// Immutable D2Q9 lattice constants and pure helpers only.

export const Q = 9;

export const RHO0 = 1.0;
export const U0 = 0.06;

export const CX = new Int8Array([0, 1, 0, -1, 0, 1, -1, -1, 1]);
export const CY = new Int8Array([0, 0, 1, 0, -1, 1, 1, -1, -1]);

export const W = new Float32Array([
  4 / 9,
  1 / 9, 1 / 9, 1 / 9, 1 / 9,
  1 / 36, 1 / 36, 1 / 36, 1 / 36,
]);

export const OPP = new Uint8Array([0, 3, 4, 1, 2, 7, 8, 5, 6]);

export const TRT_PAIRS = new Uint8Array([1, 3, 2, 4, 5, 7, 6, 8]);

export const CS_SQ = 1 / 3;
export const CS = Math.sqrt(CS_SQ);
export const INV_CS_SQ = 3.0;
export const INV_2CS_SQ = 1.5;

export function idx(x, y, NX) {
  return y * NX + x;
}

export function nodeCount(NX, NY) {
  return NX * NY;
}

export function pdfIndex(n, i) {
  return n * Q + i;
}

export function clampNodeX(x, NX) {
  return Math.max(0, Math.min(NX - 1, x));
}

export function clampNodeY(y, NY) {
  return Math.max(0, Math.min(NY - 1, y));
}

export function inBounds(x, y, NX, NY) {
  return x >= 0 && x < NX && y >= 0 && y < NY;
}

export function latticeSpeedSq(u, v) {
  return u * u + v * v;
}

export function latticeSpeed(u, v) {
  return Math.sqrt(latticeSpeedSq(u, v));
}

export function feqAt(i, rho, ux, uy) {
  const cu = CX[i] * ux + CY[i] * uy;
  const uu = ux * ux + uy * uy;
  return W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
}

export function fillEquilibriumAt(out, base, rho, ux, uy) {
  const uu = ux * ux + uy * uy;
  const v215 = 1.5 * uu;

  for (let i = 0; i < Q; i++) {
    const cu = CX[i] * ux + CY[i] * uy;
    out[base + i] = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - v215);
  }
}

export function computeMacroscopicFromF(f, base) {
  let rho = 0.0;
  let jx = 0.0;
  let jy = 0.0;

  for (let i = 0; i < Q; i++) {
    const fi = f[base + i];
    rho += fi;
    jx += fi * CX[i];
    jy += fi * CY[i];
  }

  const invRho = rho !== 0 ? 1.0 / rho : 0.0;

  return {
    rho,
    ux: jx * invRho,
    uy: jy * invRho,
  };
}

export function tauFromNu(nu) {
  return 3.0 * nu + 0.5;
}

export function omegaFromTau(tau) {
  return 1.0 / tau;
}

export function nuFromTau(tau) {
  return (tau - 0.5) / 3.0;
}

export function tauFromRe(uRef, LRef, re) {
  const nu = (uRef * LRef) / re;
  return tauFromNu(nu);
}

export function nuFromRe(uRef, LRef, re) {
  return (uRef * LRef) / re;
}

export function omegaFromRe(uRef, LRef, re) {
  return omegaFromTau(tauFromRe(uRef, LRef, re));
}


