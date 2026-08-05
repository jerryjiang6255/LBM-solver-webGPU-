// src/physics/forceModel.js
// CPU-side force reference, scaling, and coefficient interpretation only.
// No per-node force accumulation belongs here.

import { BODY_TYPES } from "../config.js";

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------
export function getDynamicPressure(rhoRef, uRef) {
  return 0.5 * rhoRef * uRef * uRef;
}

export function getReferenceDensity(sim) {
  return sim?.constants?.rho0 ?? sim?.physicsRef?.rho0 ?? 1.0;
}

export function getReferenceVelocity(sim) {
  return sim?.params?.u0 ?? sim?.physicsRef?.u0 ?? 1.0;
}

export function getCharacteristicLength(sim) {
  return (
    sim?.geometry?.meta?.charLength ??
    sim?.derived?.charLength ??
    sim?.dims?.NY ??
    1.0
  );
}

// ---------------------------------------------------------
// Force reference point by body type
// ---------------------------------------------------------
export function computeForceReference(sim, geometryMeta = sim?.geometry?.meta) {
  const type = geometryMeta?.type ?? sim?.params?.bodyType ?? BODY_TYPES.EMPTY;
  const charLength = geometryMeta?.charLength ?? getCharacteristicLength(sim);

  if (type === BODY_TYPES.AIRFOIL || type === BODY_TYPES.NACA2412) {
    const cx = geometryMeta?.cx ?? 0.0;
    const cy = geometryMeta?.cy ?? 0.0;
    const chord = geometryMeta?.chord ?? charLength;
    return { x: cx - chord / 4, y: cy };
  }

  if (type === BODY_TYPES.CYLINDER) {
    return {
      x: geometryMeta?.cx ?? 0.0,
      y: geometryMeta?.cy ?? 0.0,
    };
  }

  if (type === BODY_TYPES.SQUARE_CYLINDER) {
    return {
      x: geometryMeta?.cx ?? 0.0,
      y: geometryMeta?.cy ?? 0.0,
    };
  }

  if (type === BODY_TYPES.LID_DRIVEN_CAVITY) {
    return {
      x: (sim?.dims?.NX ?? 0) * 0.5,
      y: (sim?.dims?.NY ?? 0) * 0.5,
    };
  }

  if (type === BODY_TYPES.NOZZLE) {
    const length = geometryMeta?.length ?? charLength;
    return {
      x: length * 0.5,
      y: geometryMeta?.cy ?? (sim?.dims?.NY ?? 0) * 0.5,
    };
  }

  if (type === BODY_TYPES.BLOCKS) {
    return {
      x: (sim?.dims?.NX ?? 0) * 0.5,
      y: (sim?.dims?.NY ?? 0) * 0.5,
    };
  }

  return { x: 0.0, y: 0.0 };
}

// ---------------------------------------------------------
// Scaling by body type
// ---------------------------------------------------------
export function computeForceScales(sim, geometryMeta = sim?.geometry?.meta) {
  const rhoRef = getReferenceDensity(sim);
  const uRef = getReferenceVelocity(sim);
  const qDyn = getDynamicPressure(rhoRef, uRef);

  const type = geometryMeta?.type ?? sim?.params?.bodyType ?? BODY_TYPES.EMPTY;
  const charLength = geometryMeta?.charLength ?? getCharacteristicLength(sim);

  if (type === BODY_TYPES.AIRFOIL || type === BODY_TYPES.NACA2412) {
    const chord = geometryMeta?.chord ?? charLength;
    return {
      forceScale: qDyn * chord,
      momentScale: qDyn * chord * chord,
    };
  }

  if (type === BODY_TYPES.CYLINDER) {
    const radius = geometryMeta?.radius ?? (charLength * 0.5);
    const diameter = 2 * radius;
    return {
      forceScale: qDyn * diameter,
      momentScale: qDyn * diameter * diameter,
    };
  }

  if (type === BODY_TYPES.SQUARE_CYLINDER) {
    const size = geometryMeta?.size ?? charLength;
    return {
      forceScale: qDyn * size,
      momentScale: qDyn * size * size,
    };
  }

  if (type === BODY_TYPES.NOZZLE) {
    const diameter = geometryMeta?.diameter ?? charLength;
    return {
      forceScale: qDyn * diameter,
      momentScale: qDyn * diameter * diameter,
    };
  }

  if (type === BODY_TYPES.LID_DRIVEN_CAVITY || type === BODY_TYPES.BLOCKS) {
    const L = Math.max(charLength, 1.0);
    return {
      forceScale: qDyn * L,
      momentScale: qDyn * L * L,
    };
  }

  {
    const L = Math.max(charLength, 1.0);
    return {
      forceScale: qDyn * L,
      momentScale: qDyn * L * L,
    };
  }
}

// ---------------------------------------------------------
// Full derived package for one geometry state
// ---------------------------------------------------------
export function buildForceModelState(sim, geometryMeta = sim?.geometry?.meta) {
  const forceRef = computeForceReference(sim, geometryMeta);
  const { forceScale, momentScale } = computeForceScales(sim, geometryMeta);

  return {
    forceRef,
    forceScale,
    momentScale,
  };
}

// ---------------------------------------------------------
// Apply raw reduced force outputs to sim diagnostics
// rawForce = { fx, fy, mz }
// Convention:
// - +fx => drag-positive streamwise
// - +fy => screen-down positive, so aerodynamic lift = -fy
// ---------------------------------------------------------
export function applyForceCoefficients(sim, rawForce) {
  const fx = rawForce?.fx ?? 0.0;
  const fy = rawForce?.fy ?? 0.0;
  const mz = rawForce?.mz ?? 0.0;

  const forceScale =
    sim?.derived?.forceScale ??
    sim?.geometry?.meta?.forceScale ??
    1.0;

  const momentScale =
    sim?.derived?.momentScale ??
    sim?.geometry?.meta?.momentScale ??
    1.0;

  const cd = forceScale !== 0 ? fx / forceScale : 0.0;
  const cl = forceScale !== 0 ? -fy / forceScale : 0.0;
  const cm = momentScale !== 0 ? mz / momentScale : 0.0;

  if (!sim.diagnostics) sim.diagnostics = {};
  if (!sim.diagnostics.force) sim.diagnostics.force = { fx: 0.0, fy: 0.0, mz: 0.0 };
  if (!sim.diagnostics.coeff) sim.diagnostics.coeff = { cd: 0.0, cl: 0.0, cm: 0.0, st: 0.0 };

  sim.diagnostics.force.fx = fx;
  sim.diagnostics.force.fy = fy;
  sim.diagnostics.force.mz = mz;

  sim.diagnostics.coeff.cd = cd;
  sim.diagnostics.coeff.cl = cl;
  sim.diagnostics.coeff.cm = cm;

  return sim.diagnostics.coeff;
}

// ---------------------------------------------------------
// Geometry-derived values should also be mirrored into sim
// for easy access by renderers / GPU upload code.
// ---------------------------------------------------------
export function applyForceModelStateToSim(sim, geometryMeta = sim?.geometry?.meta) {
  const { forceRef, forceScale, momentScale } = buildForceModelState(sim, geometryMeta);

  if (!sim.derived) sim.derived = {};
  if (!sim.diagnostics) sim.diagnostics = {};
  if (!sim.diagnostics.force) sim.diagnostics.force = { fx: 0.0, fy: 0.0, mz: 0.0 };
  if (!sim.diagnostics.coeff) sim.diagnostics.coeff = { cd: 0.0, cl: 0.0, cm: 0.0, st: 0.0 };

  sim.derived.forceRef = forceRef;
  sim.derived.forceScale = forceScale;
  sim.derived.momentScale = momentScale;

  return {
    forceRef,
    forceScale,
    momentScale,
  };
}

// ---------------------------------------------------------
// Reset outputs only
// ---------------------------------------------------------
export function resetForceDiagnostics(sim) {
  if (!sim.diagnostics) sim.diagnostics = {};
  sim.diagnostics.force = { fx: 0.0, fy: 0.0, mz: 0.0 };
  sim.diagnostics.coeff = { cd: 0.0, cl: 0.0, cm: 0.0, st: 0.0 };
}




