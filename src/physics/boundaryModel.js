// src/physics/boundaryModel.js
// CPU-side boundary semantics and parameter packing only.
// No per-node timestep loops belong here.

import {
  BOUNDARY_DEFAULTS,
  BODY_TYPES,
  UI_LIMITS,
} from "../config.js";

import {
  CS,
} from "./lattice.js";

// ---------------------------------------------------------
// Mode helpers
// ---------------------------------------------------------
export function isLidDrivenMode(sim) {
  return sim?.params?.bodyType === BODY_TYPES.LID_DRIVEN_CAVITY;
}

export function isNozzleMode(sim) {
  return sim?.params?.bodyType === BODY_TYPES.NOZZLE;
}

export function hasPulsation(sim) {
  return !!sim?.flow?.pulsationOn;
}

// ---------------------------------------------------------
// Inlet / lid velocity semantics
// ---------------------------------------------------------
export function getInstantaneousDriveVelocity(sim, step = 0) {
  const base = sim?.params?.u0 ?? 0.0;
  const flow = sim?.flow;

  if (!flow?.pulsationOn) return base;

  const amp = flow.pulsationAmp ?? 0.0;
  const freq = flow.pulsationFreq ?? 0.0;
  const u = base * (1 + amp * Math.sin(2 * Math.PI * freq * step));
  return Math.max(0.0, u);
}

export function getBoundaryDriveLabel(sim) {
  return isLidDrivenMode(sim) ? "lid" : "inlet";
}

export function getReferenceDriveVelocity(sim, step = null) {
  if (step === null) {
    step = sim?.runtime?.stepCount ?? 0;
  }
  return getInstantaneousDriveVelocity(sim, step);
}

export function getMachFromVelocity(u) {
  return u / CS;
}

// ---------------------------------------------------------
// Sponge semantics
// ---------------------------------------------------------
export function getSpongeConfig(sim) {
  const NX = sim?.dims?.NX ?? 0;
  const width = Math.min(BOUNDARY_DEFAULTS.spongeWidth, Math.max(1, NX - 2));
  const startX = Math.max(0, NX - 1 - width);

  return {
    width,
    strength: BOUNDARY_DEFAULTS.spongeStrength,
    startX,
  };
}

export function buildSpongeWeights(NX) {
  const out = new Float32Array(NX);

  const width = Math.min(BOUNDARY_DEFAULTS.spongeWidth, Math.max(1, NX - 2));
  const start = NX - 1 - width;

  for (let x = start; x < NX; x++) {
    const t = (x - start) / width;
    out[x] = BOUNDARY_DEFAULTS.spongeStrength * t * t;
  }

  return out;
}

// ---------------------------------------------------------
// Boundary flags / semantics
// ---------------------------------------------------------
export function getBoundaryFlags(sim) {
  return {
    isLidDriven: isLidDrivenMode(sim),
    isNozzle: isNozzleMode(sim),
    pulsationOn: hasPulsation(sim),
    hasTopBottomFreestream: !isLidDrivenMode(sim),
    hasLeftInlet: !isLidDrivenMode(sim),
    hasRightOutlet: !isLidDrivenMode(sim),
    hasSpongeOutlet: !isLidDrivenMode(sim),
  };
}

export function getNozzleConfig(sim) {
  const NY = sim?.dims?.NY ?? 0;
  const nozzleDiameter = sim?.params?.nozzleDiameter ?? 16;

  return {
    active: isNozzleMode(sim),
    cy: Math.floor(NY / 2),
    diameter: nozzleDiameter,
    halfD: nozzleDiameter * 0.5,
    uCoflow: sim?.flow?.uCoflow ?? sim?.params?.uCoflow ?? 0.005,
  };
}

// ---------------------------------------------------------
// UI / Reynolds helpers
// ---------------------------------------------------------
export function getCharacteristicLengthForBoundaryUse(sim) {
  return sim?.geometry?.meta?.charLength ?? sim?.derived?.charLength ?? 1.0;
}

export function computeReBounds(u0, L) {
  const nuMin = (UI_LIMITS.tauMinUI - 0.5) / 3.0;
  const nuMax = (UI_LIMITS.tauMaxUI - 0.5) / 3.0;

  const reMaxRaw = (u0 * L) / nuMin;
  const reMinRaw = (u0 * L) / nuMax;

  const step = UI_LIMITS.reStep;
  const reMin = Math.max(step, Math.ceil(reMinRaw / step) * step);
  const reMax = Math.max(reMin, Math.floor(reMaxRaw / step) * step);

  return { reMin, reMax };
}

// ---------------------------------------------------------
// Packed uniform builder
// This is semantic packing only; gpuBuffers.js decides
// final byte layout.
// ---------------------------------------------------------
export function buildBoundaryUniforms(sim) {
  const step = sim?.runtime?.stepCount ?? 0;
  const uDrive = getInstantaneousDriveVelocity(sim, step);
  const flags = getBoundaryFlags(sim);
  const sponge = getSpongeConfig(sim);
  const nozzle = getNozzleConfig(sim);

  return {
    step,
    uDrive,
    inletU: flags.isLidDriven ? 0.0 : uDrive,
    lidU: flags.isLidDriven ? uDrive : 0.0,

    isLidDriven: flags.isLidDriven ? 1 : 0,
    isNozzle: flags.isNozzle ? 1 : 0,
    pulsationOn: flags.pulsationOn ? 1 : 0,

    spongeStartX: sponge.startX,
    spongeWidth: sponge.width,
    spongeStrength: sponge.strength,

    nozzleActive: nozzle.active ? 1 : 0,
    nozzleCy: nozzle.cy,
    nozzleHalfD: nozzle.halfD,
    uCoflow: nozzle.uCoflow,
  };
}


