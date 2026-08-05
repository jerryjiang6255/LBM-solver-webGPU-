// src/geometry.js
// CPU-side geometry construction for WebGPU solver.
// Produces stable geometry arrays + metadata package.
// Supports user-painted wall merging via sim.geometry.paintedNodes.

import { Q, CX, CY } from "./physics/lattice.js";
import { BODY_TYPES } from "./config.js";
import { buildForceModelState } from "./physics/forceModel.js";

const Q_MIN = 0.01;
const Q_MAX = 1.0;

// ---------------------------------------------------------
// Public API
// ---------------------------------------------------------
export function buildGeometryPackage(sim) {
  const dims = sim.dims;
  const params = sim.params;
  const alpha = (params.aoaDeg * Math.PI) / 180.0;

  let geometryPackage;

  switch (params.bodyType) {
    case BODY_TYPES.AIRFOIL:
      geometryPackage = buildAirfoilPackage(dims, alpha);
      break;

    case BODY_TYPES.NACA2412:
      geometryPackage = buildNACA2412Package(dims, alpha);
      break;

    case BODY_TYPES.CYLINDER:
      geometryPackage = buildCylinderPackage(dims, params.cylinderDiameter);
      break;

    case BODY_TYPES.SQUARE_CYLINDER:
      geometryPackage = buildSquareCylinderPackage(dims, alpha);
      break;

    case BODY_TYPES.LID_DRIVEN_CAVITY:
      geometryPackage = buildLidDrivenCavityPackage(dims, params.cavityAspect);
      break;

    case BODY_TYPES.NOZZLE:
      geometryPackage = buildNozzlePackage(dims, params.nozzleDiameter);
      break;

    case BODY_TYPES.BLOCKS:
      geometryPackage = buildBlocksPackage(dims);
      break;

    case BODY_TYPES.EMPTY:
    default:
      geometryPackage = buildEmptyPackage(dims);
      break;
  }

  applyPaintedWallsToGeometryPackage(sim, geometryPackage);

  const forceState = buildForceModelState(
    { ...sim, geometry: { meta: geometryPackage.meta } },
    geometryPackage.meta
  );

  geometryPackage.meta = {
    ...geometryPackage.meta,
    forceRef: forceState.forceRef,
    forceScale: forceState.forceScale,
    momentScale: forceState.momentScale,
  };

  return geometryPackage;
}

export function applyPaintedWallsToGeometryPackage(sim, geometryPackage) {
  const paintedNodes = sim?.geometry?.paintedNodes;
  if (!paintedNodes || paintedNodes.size === 0) return geometryPackage;

  const { NX, NY } = sim.dims;
  const arrays = geometryPackage.arrays;

  for (const n of paintedNodes) {
    const x = n % NX;
    const y = (n / NX) | 0;

    if (x < 1 || x >= NX - 1 || y < 1 || y >= NY - 1) continue;

    arrays.solid[n] = 1;
    arrays.sdf[n] = -1.0;
    arrays.wallDist[n] = 0.0;
  }

  rebuildDerivedGeometryFields(arrays, NX, NY);
  return geometryPackage;
}

export function buildAirfoilPackage(dims, alpha = 0) {
  const { NX, NY } = dims;
  const chord = Math.floor(NX / 3);
  const cx = Math.floor(NX / 3);
  const cy = Math.floor(NY / 2);

  return buildFromSdf(
    dims,
    sdfNACA0012(cx, cy, chord, alpha),
    {
      type: BODY_TYPES.AIRFOIL,
      cx,
      cy,
      chord,
      alpha,
      profile: "0012",
      charLength: chord,
      visualParams: { cx, cy, chord, alpha, profile: "0012" },
    }
  );
}

export function buildNACA2412Package(dims, alpha = 0) {
  const { NX, NY } = dims;
  const chord = Math.floor(NX / 3);
  const cx = Math.floor(NX / 3);
  const cy = Math.floor(NY / 2);

  return buildFromSdf(
    dims,
    sdfNACA2412(cx, cy, chord, alpha),
    {
      type: BODY_TYPES.NACA2412,
      cx,
      cy,
      chord,
      alpha,
      profile: "2412",
      charLength: chord,
      visualParams: { cx, cy, chord, alpha, profile: "2412" },
    }
  );
}

export function buildCylinderPackage(dims, diameter = 32) {
  const { NX, NY } = dims;
  const cx = Math.floor(NX / 4);
  const cy = Math.floor(NY / 2);
  const radius = diameter * 0.5;

  return buildFromSdf(
    dims,
    sdfCircle(cx, cy, radius),
    {
      type: BODY_TYPES.CYLINDER,
      cx,
      cy,
      radius,
      diameter,
      charLength: diameter,
      visualParams: { cx, cy, radius },
    }
  );
} 

export function buildSquareCylinderPackage(dims, alpha = 0) {
  const { NX, NY } = dims;
  const cx = Math.floor(NX / 4);
  const cy = Math.floor(NY / 2);
  const size = Math.max(8, Math.floor(NY / 4));

  return buildFromSdf(
    dims,
    sdfRotatedBox(cx, cy, size * 0.5, size * 0.5, alpha),
    {
      type: BODY_TYPES.SQUARE_CYLINDER,
      cx,
      cy,
      size,
      alpha,
      charLength: size,
      visualParams: { cx, cy, size, alpha },
    }
  );
}

export function buildLidDrivenCavityPackage(dims, cavityAspect = 1.0) {
  const { NX, NY } = dims;
  const wallThickness = 1;

  const outerLeft = wallThickness;
  const outerRight = NX - 1 - wallThickness;
  const outerBottom = wallThickness;
  const outerTop = NY - 1 - wallThickness;

  const cavityHeight = (outerTop - outerBottom) + 1 - 2 * wallThickness;
  const cavityWidth = Math.max(4, Math.round(cavityHeight * cavityAspect));

  const midX = 0.5 * (outerLeft + outerRight);
  const cavityLeftInner = Math.round(midX - cavityWidth / 2);
  const cavityRightInner = Math.round(midX + cavityWidth / 2);
  const charLength = (cavityRightInner - cavityLeftInner) - 1;

  return buildFromSdf(
    dims,
    sdfLidDrivenCavity(NX, NY, wallThickness, cavityAspect),
    {
      type: BODY_TYPES.LID_DRIVEN_CAVITY,
      wallThickness,
      cavityAspect,
      cavityWidth,
      cavityHeight,
      cavityLeftInner,
      cavityRightInner,
      charLength,
      visualParams: {
        wallThickness,
        cavityAspect,
      },
    }
  );
}

export function buildNozzlePackage(dims, diameter = 16) {
  const { NX, NY } = dims;
  const cy = Math.floor(NY / 2);
  const length = Math.floor(NX * 0.15);

  return buildFromSdf(
    dims,
    sdfNozzle(cy, diameter, length),
    {
      type: BODY_TYPES.NOZZLE,
      cy,
      diameter,
      length,
      charLength: diameter,
      visualParams: { cy, diameter, length },
    }
  );
}

export function buildBlocksPackage(dims) {
  const { NX, NY } = dims;
  const blockSize = 4;

  const colXs = [
    Math.floor(NX * 0.30),
    Math.floor(NX * 0.43),
    Math.floor(NX * 0.56),
    Math.floor(NX * 0.69),
  ];

  const rows4 = [
    Math.floor(NY * 0.18),
    Math.floor(NY * 0.38),
    Math.floor(NY * 0.62),
    Math.floor(NY * 0.82),
  ];

  const rows3 = [
    Math.floor(NY * 0.28),
    Math.floor(NY * 0.50),
    Math.floor(NY * 0.72),
  ];

  const blocks = [];
  for (const x of [colXs[0], colXs[2]]) {
    for (const y of rows4) blocks.push([x, y]);
  }
  for (const x of [colXs[1], colXs[3]]) {
    for (const y of rows3) blocks.push([x, y]);
  }

  return buildFromSdf(
    dims,
    sdfExplicitBlocks(blocks, blockSize),
    {
      type: BODY_TYPES.BLOCKS,
      blocks,
      blockSize,
      charLength: blockSize * 4,
      visualParams: { blocks, blockSize },
    }
  );
}

export function buildEmptyPackage(dims) {
  return buildFromSdf(
    dims,
    sdfEmpty(),
    {
      type: BODY_TYPES.EMPTY,
      charLength: dims.NY,
      visualParams: { type: BODY_TYPES.EMPTY },
    }
  );
} 

// ---------------------------------------------------------
// Core builder
// ---------------------------------------------------------
function buildFromSdf(dims, sdfFunc, meta) {
  const arrays = buildGeometryArraysFromSdf(dims, sdfFunc);
  return { arrays, meta };
}

function buildGeometryArraysFromSdf(dims, sdfFunc) {
  const { NX, NY } = dims;
  const N = NX * NY;

  const sdf = new Float32Array(N);
  const solid = new Uint8Array(N);
  const wallDist = new Float32Array(N);
  const wallAdjacent = new Uint8Array(N);
  const linkMask = new Uint8Array(Q * N);
  const linkQ = new Float32Array(Q * N);

  const solidTmp = [];
  const fluidTmp = [];

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const n = y * NX + x;
      const d = sdfFunc(x, y);

      sdf[n] = d;
      solid[n] = d <= 0 ? 1 : 0;
      wallDist[n] = d > 0 ? d : 0.0;
    }
  }

  finalizeDerivedGeometryFields({
    NX,
    NY,
    solid,
    sdf,
    wallDist,
    wallAdjacent,
    linkMask,
    linkQ,
    solidTmp,
    fluidTmp,
  });

  return {
    solid,
    sdf,
    wallDist,
    wallAdjacent,
    linkMask,
    linkQ,
    solidList: new Uint32Array(solidTmp),
    fluidSurfaceList: new Uint32Array(fluidTmp),
  };
}

function rebuildDerivedGeometryFields(arrays, NX, NY) {
  arrays.wallAdjacent.fill(0);
  arrays.linkMask.fill(0);
  arrays.linkQ.fill(0);

  const solidTmp = [];
  const fluidTmp = [];

  for (let n = 0; n < arrays.wallDist.length; n++) {
    arrays.wallDist[n] = arrays.solid[n] ? 0.0 : Math.max(arrays.sdf[n], 0.0);
  }

  finalizeDerivedGeometryFields({
    NX,
    NY,
    solid: arrays.solid,
    sdf: arrays.sdf,
    wallDist: arrays.wallDist,
    wallAdjacent: arrays.wallAdjacent,
    linkMask: arrays.linkMask,
    linkQ: arrays.linkQ,
    solidTmp,
    fluidTmp,
  });

  arrays.solidList = new Uint32Array(solidTmp);
  arrays.fluidSurfaceList = new Uint32Array(fluidTmp);
}

function finalizeDerivedGeometryFields(ctx) {
  const {
    NX, NY,
    solid, sdf, wallAdjacent, linkMask, linkQ,
    solidTmp, fluidTmp,
  } = ctx;

  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const n = y * NX + x;

      if (solid[n]) {
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
      let isAdjacent = 0;

      for (let i = 1; i < Q; i++) {
        const nx = x + CX[i];
        const ny = y + CY[i];

        if (nx < 0 || nx >= NX || ny < 0 || ny >= NY) continue;

        const ns = ny * NX + nx;
        if (!solid[ns]) continue;

        isAdjacent = 1;

        const sdfS = sdf[ns];
        const denom = sdfF - sdfS;
        let q = denom > 1e-10 ? sdfF / denom : 0.5;

        if (q < Q_MIN) q = Q_MIN;
        if (q > Q_MAX) q = Q_MAX;

        const off = n * Q + i;
        linkMask[off] = 1;
        linkQ[off] = q;
      }

      wallAdjacent[n] = isAdjacent;
      if (isAdjacent) fluidTmp.push(n);
    }
  }
} 

// ---------------------------------------------------------
// SDFs
// ---------------------------------------------------------
function sdfEmpty() {
  return () => 1.0;
}

function sdfCircle(cx, cy, radius) {
  return (x, y) => Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius;
}

function sdfNACA0012(cx, cy, chord, alpha) {
  return sdfNACA4(cx, cy, chord, alpha, 0.0, 0.0, 0.12);
}

function sdfNACA2412(cx, cy, chord, alpha) {
  return sdfNACA4(cx, cy, chord, alpha, 0.02, 0.4, 0.12);
}

function sdfNACA4(cx, cy, chord, alpha, m, p, t) {
  const cosA = Math.cos(alpha);
  const sinA = Math.sin(alpha);

  return (x, y) => {
    const dx = (x - cx) * cosA + (y - cy) * sinA;
    const dy = -(x - cx) * sinA + (y - cy) * cosA;

    const xn = dx / chord + 0.5;
    if (xn < 0 || xn > 1) return Math.abs(dx) + Math.abs(dy);

    let yt = (t / 0.2) * chord * (
       0.2969 * Math.sqrt(xn)
      - 0.1260 * xn
      - 0.3516 * xn * xn
      + 0.2843 * xn * xn * xn
      - 0.1015 * xn * xn * xn * xn
    );
    if (xn >= 1.0) yt = 0.0;

    let yc = 0.0;
    let dycDxn = 0.0;

    if (m > 0 && p > 0 && p < 1) {
      if (xn < p) {
        yc = m * chord * (2 * p * xn - xn * xn) / (p * p);
        dycDxn = (2 * m * chord / (p * p)) * (p - xn);
      } else {
        const op = 1 - p;
        yc = m * chord * (1 - 2 * p + 2 * p * xn - xn * xn) / (op * op);
        dycDxn = (2 * m * chord / (op * op)) * (p - xn);
      }
    }

    const theta = Math.atan(dycDxn / chord);
    const yRel = dy - (-yc);
    const ySurf = yt * Math.cos(theta);

    return Math.abs(yRel) - ySurf;
  };
}

function sdfRotatedBox(cx, cy, halfW, halfH, alpha) {
  const cosA = Math.cos(alpha);
  const sinA = Math.sin(alpha);

  return (x, y) => {
    const dx = (x - cx) * cosA + (y - cy) * sinA;
    const dy = -(x - cx) * sinA + (y - cy) * cosA;

    const qx = Math.abs(dx) - halfW;
    const qy = Math.abs(dy) - halfH;

    const ox = Math.max(qx, 0);
    const oy = Math.max(qy, 0);
    const outside = Math.sqrt(ox * ox + oy * oy);
    const inside = Math.min(Math.max(qx, qy), 0);

    return outside + inside;
  };
}

function sdfLidDrivenCavity(NX, NY, wallThickness = 1, cavityAspect = 1.0) {
  const outerLeft = wallThickness;
  const outerRight = NX - 1 - wallThickness;
  const outerBottom = wallThickness;
  const outerTop = NY - 1 - wallThickness;

  const cavityHeight = (outerTop - outerBottom) + 1 - 2 * wallThickness;
  const cavityWidth = Math.max(4, Math.round(cavityHeight * cavityAspect));

  const midX = 0.5 * (outerLeft + outerRight);
  const cavityLeftInner = Math.round(midX - cavityWidth / 2);
  const cavityRightInner = Math.round(midX + cavityWidth / 2);

  return (x, y) => {
    const inOuterFluid =
      x >= outerLeft && x <= outerRight &&
      y >= outerBottom && y <= outerTop;

    if (!inOuterFluid) return -1.0;
    if (y <= outerBottom || y >= outerTop) return -1.0;
    if (x <= outerLeft || x >= outerRight) return -1.0;
    if (x === cavityLeftInner || x === cavityRightInner) return -1.0;

    return 1.0;
  };
}

function sdfNozzle(cy, diameter, length) {
  const halfD = diameter / 2;
  const wallThickness = 3;

  return (x, y) => {
    if (x > length) return 1.0;

    const dist = Math.abs(y - cy);
    if (dist <= halfD) return 1.0;
    if (dist <= halfD + wallThickness) return -1.0;

    return 1.0;
  };
}

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
