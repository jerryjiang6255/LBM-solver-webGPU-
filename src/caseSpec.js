// src/caseSpec.js
// ============================================================
// Resolution-independent case definition.
//
// A case is described by dimensionless numbers (body size as a fraction
// of the domain, position as fractions, Re, u0 = Ma * cs, pulse Strouhal
// number). Everything in lattice cells is derived here from the grid
// size, so the same case resolves consistently at every fidelity:
// geometry, characteristic length, force reference, sponge layout and
// the pulse frequency per step.
//
// Placement convention: node (i, j) sits at the continuous position
// (i + 0.5, j + 0.5) in cell units. A body at domain fraction (fx, fy)
// has its centre at continuous (fx * NX, fy * NY), i.e. node coordinate
// fx * NX - 0.5. Averaging an r x r block of high-res nodes then lands
// exactly on the low-res node centre, so bodies line up across
// fidelities (and a body at fy = 0.5 is exactly symmetric about the
// domain's mid-line).
// ============================================================

import {
  buildAirfoil,
  buildNACA2412,
  buildCylinder,
  buildSquareCylinder,
  buildBlocks,
  buildEmpty,
} from "./geometry.js";

export const CS_LATTICE = 1 / Math.sqrt(3);

// Outlet sponge: width as a fraction of NX (14 cells at NX = 256,
// 28 at 512, 56 at 1024, 112 at 2048) and peak relaxation strength.
export const SPONGE_FRACTION = 14 / 256;
export const SPONGE_STRENGTH = 0.6;

// Body layout (fractions of the domain).
export const AIRFOIL = { x: 1 / 3, y: 0.5, chordOverW: 1 / 3 };
export const CYLINDER = { x: 0.25, y: 0.5, defaultDOverH: 0.125, minDOverH: 0.04, maxDOverH: 0.4 };
export const SQUARE = { x: 0.25, y: 0.5, sizeOverH: 0.25 };
export const BLOCKS = {
  sizeOverH: 1 / 32,
  cols: [0.30, 0.43, 0.56, 0.69],
  rows4: [0.18, 0.38, 0.62, 0.82],
  rows3: [0.28, 0.50, 0.72],
};

export const BODY_TYPES = ["airfoil", "naca2412", "cylinder", "square-cylinder", "blocks", "empty"];
export const AOA_BODIES = new Set(["airfoil", "naca2412", "square-cylinder"]);

// Continuous domain fraction -> node coordinate.
export function nodeX(fx, NX) { return fx * NX - 0.5; }
export function nodeY(fy, NY) { return fy * NY - 0.5; }

// ---------------------------------------------------------
// characteristicLength(params, dims)
// Reynolds-number length in cells, without building geometry.
// ---------------------------------------------------------
export function characteristicLength(params, { NX, NY }) {
  switch (params?.bodyType ?? "airfoil") {
    case "airfoil":
    case "naca2412":
      return AIRFOIL.chordOverW * NX;
    case "cylinder":
      return (params.cylinderDH ?? CYLINDER.defaultDOverH) * NY;
    case "square-cylinder":
      return SQUARE.sizeOverH * NY;
    case "blocks":
      return 4 * BLOCKS.sizeOverH * NY;
    case "empty":
    default:
      return NY;
  }
}

// ---------------------------------------------------------
// resolveGeometry(params, dims)
// Builds the lattice geometry for a case at this grid size.
// Returns { g, p, charLength, refLength, forceRef }:
//   g          geometry arrays (geometry.js)
//   p          body description for the renderer (node coordinates)
//   charLength Reynolds length in cells
//   refLength  force-coefficient reference length in cells
//   forceRef   moment reference point (node coordinates)
// ---------------------------------------------------------
export function resolveGeometry(params, dims) {
  const { NX, NY } = dims;
  const bodyType = params.bodyType;
  const alpha = (params.aoaDeg ?? 0) * Math.PI / 180;

  if (bodyType === "airfoil" || bodyType === "naca2412") {
    const chord = AIRFOIL.chordOverW * NX;
    const cx = nodeX(AIRFOIL.x, NX);
    const cy = nodeY(AIRFOIL.y, NY);
    const build = bodyType === "airfoil" ? buildAirfoil : buildNACA2412;
    const result = build(dims, cx, cy, chord, alpha);
    return {
      g: result.geometry,
      p: { type: bodyType, cx, cy, chord, alpha, profile: bodyType === "airfoil" ? "0012" : "2412" },
      charLength: chord,
      refLength: chord,
      forceRef: { x: cx - chord / 4, y: cy },
    };
  }

  if (bodyType === "cylinder") {
    const diameter = (params.cylinderDH ?? CYLINDER.defaultDOverH) * NY;
    const radius = 0.5 * diameter;
    const cx = nodeX(CYLINDER.x, NX);
    const cy = nodeY(CYLINDER.y, NY);
    const result = buildCylinder(dims, cx, cy, radius);
    return {
      g: result.geometry,
      p: { type: "cylinder", cx, cy, radius },
      charLength: diameter,
      refLength: diameter,
      forceRef: { x: cx, y: cy },
    };
  }

  if (bodyType === "square-cylinder") {
    const size = SQUARE.sizeOverH * NY;
    const cx = nodeX(SQUARE.x, NX);
    const cy = nodeY(SQUARE.y, NY);
    const result = buildSquareCylinder(dims, cx, cy, size, alpha);
    return {
      g: result.geometry,
      p: { type: "square-cylinder", cx, cy, size, alpha },
      charLength: size,
      refLength: size,
      forceRef: { x: cx, y: cy },
    };
  }

  if (bodyType === "blocks") {
    const blockSize = BLOCKS.sizeOverH * NY;
    const blocks = [];
    for (const fx of [BLOCKS.cols[0], BLOCKS.cols[2]]) {
      for (const fy of BLOCKS.rows4) blocks.push([nodeX(fx, NX), nodeY(fy, NY)]);
    }
    for (const fx of [BLOCKS.cols[1], BLOCKS.cols[3]]) {
      for (const fy of BLOCKS.rows3) blocks.push([nodeX(fx, NX), nodeY(fy, NY)]);
    }
    const result = buildBlocks(dims, blocks, blockSize);
    const L = 4 * blockSize;
    return {
      g: result.geometry,
      p: { type: "blocks", blocks, blockSize },
      charLength: L,
      refLength: L,
      forceRef: { x: 0.5 * NX, y: 0.5 * NY },
    };
  }

  // empty
  const result = buildEmpty(dims);
  return {
    g: result.geometry,
    p: { type: "empty" },
    charLength: NY,
    refLength: NY,
    forceRef: { x: 0, y: 0 },
  };
}

// ---------------------------------------------------------
// spongeLayout(NX)
// Outlet sponge columns: x in [start, NX - 2] with weight
// strength * ((x - start) / width)^2.
// ---------------------------------------------------------
export function spongeLayout(NX) {
  const width = Math.min(Math.max(1, NX - 2), Math.max(1, Math.round(SPONGE_FRACTION * NX)));
  return { width, start: NX - 1 - width, strength: SPONGE_STRENGTH };
}

// ---------------------------------------------------------
// pulseFreqPerStep(st, u0, L)
// Inlet pulsation frequency in cycles per lattice step from the
// forcing Strouhal number St = f L / u0.
// ---------------------------------------------------------
export function pulseFreqPerStep(st, u0, L) {
  return L > 0 ? (st * u0) / L : 0;
}
