// src/gpu/shaders.wgsl
// Central WGSL source for the WebGPU LBM solver.

const Q : u32 = 9u;

const CX : array<i32, 9> = array<i32, 9>(
  0, 1, 0, -1, 0, 1, -1, -1, 1
);

const CY : array<i32, 9> = array<i32, 9>(
  0, 0, 1, 0, -1, 1, 1, -1, -1
);

const OPP : array<u32, 9> = array<u32, 9>(
  0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u
);

const W0 : f32 = 4.0 / 9.0;
const W1 : f32 = 1.0 / 9.0;
const W5 : f32 = 1.0 / 36.0;

const W : array<f32, 9> = array<f32, 9>(
  W0,
  W1, W1, W1, W1,
  W5, W5, W5, W5
);

const CS_SQ : f32 = 1.0 / 3.0;
const PI : f32 = 3.141592653589793;

const U_CAP   : f32 = 0.28;
const RHO_MIN : f32 = 0.05;
const RHO_MAX : f32 = 5.0;
const TAU_MIN : f32 = 0.545;
const TAU_MAX : f32 = 5.0;

const CS_LES       : f32 = 0.17;
const CW_LES       : f32 = 0.325;
const DELTA_LES    : f32 = 1.0;
const CS_DELTA_SQ  : f32 = CS_LES * CS_LES * DELTA_LES * DELTA_LES;
const CW_DELTA_SQ  : f32 = CW_LES * CW_LES * DELTA_LES * DELTA_LES;
const WALE_EPS     : f32 = 1e-20;

const KAPPA      : f32 = 0.41;
const B_LOG      : f32 = 5.2;
const A_PLUS     : f32 = 25.0;
const WALL_ITERS : u32 = 6u;
const WALL_TOL   : f32 = 1e-4;
const Y_MIN      : f32 = 0.5;

const TRT_LAMBDA : f32 = 3.0 / 16.0;
const FORCE_COS_THRESH : f32 = 0.1;

// Particle visual defaults
const PARTICLE_RADIUS_PX : f32 = 5.0;
const PARTICLE_ALPHA_MIN : f32 = 0.16;
const PARTICLE_ALPHA_MAX : f32 = 1.0;

struct SimUniforms {
  NX : u32,
  NY : u32,
  N  : u32,
  stepCount : u32,

  u0 : f32,
  reynolds : f32,
  charLength : f32,
  nu0 : f32,

  omega0 : f32,
  tau0 : f32,
  colorScale : f32,
  displayMode : u32,

 collisionModel : u32,
  lesModel : u32,
  wallModel : u32,
  displayFlags : u32,

  pulsationOn : u32,
  nozzleActive : u32,
  isLidDriven : u32,
  _pad0 : u32,

  pulsationAmp : f32,
  pulsationFreq : f32,
  uCoflow : f32,
  nozzleCy : f32,

  nozzleHalfD : f32,
  spongeStartX : f32,
  spongeWidth : f32,
  spongeStrength : f32,

  lidU : f32,
  inletU : f32,
  rho0 : f32,
  _pad1 : f32,

  forceRefX : f32,
  forceRefY : f32,
  forceScale : f32,
  momentScale : f32,

 renderWidth : f32,
  renderHeight : f32,
  _pad2 : f32,
  _pad3 : f32,

};

struct MacroCell {
  rho : f32,
  ux  : f32,
  uy  : f32,
  aux : f32,
};

struct ForceAccum {
  fx : atomic<i32>,
  fy : atomic<i32>,
  mz : atomic<i32>,
  _pad : atomic<i32>,
};

struct ReductionOut {
  meanSpeed  : atomic<i32>,
  rmsDeltaU  : atomic<i32>,
  meanRhoDev : atomic<i32>,
  fluidCount : atomic<u32>,
};

struct Particle {
  x : f32,
  y : f32,
  vx : f32,
  vy : f32,
};

struct VsOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

struct ParticleVsOut {
  @builtin(position) position : vec4<f32>,
  @location(0) alpha : f32,
  @location(1) localUv : vec2<f32>,
};
@group(0) @binding(0) var<uniform> U : SimUniforms;
@group(0) @binding(1) var<storage, read_write> fIn : array<f32>;
@group(0) @binding(2) var<storage, read_write> fOut : array<f32>;
@group(0) @binding(3) var<storage, read_write> macroField : array<MacroCell>;
@group(0) @binding(4) var<storage, read> solid : array<u32>;
@group(0) @binding(5) var<storage, read> sdf : array<f32>;
@group(0) @binding(6) var<storage, read> wallDist : array<f32>;
@group(0) @binding(7) var<storage, read> wallAdjacent : array<u32>;
@group(0) @binding(8) var<storage, read> linkMask : array<u32>;
@group(0) @binding(9) var<storage, read> linkQ : array<f32>;
@group(0) @binding(10) var<storage, read> fluidSurfaceList : array<u32>;
@group(0) @binding(11) var<storage, read_write> forceAccum : ForceAccum;
@group(0) @binding(12) var<storage, read_write> reductionOut : ReductionOut;
@group(0) @binding(13) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(14) var<storage, read_write> prevMacroField : array<MacroCell>;
@group(0) @binding(15) var<storage, read> reopenNodeList : array<u32>;

fn idx(x : u32, y : u32) -> u32 {
  return y * U.NX + x;
}

fn pdfIndex(n : u32, i : u32) -> u32 {
  return n * Q + i;
}

fn inBounds(x : i32, y : i32) -> bool {
  return x >= 0 && x < i32(U.NX) && y >= 0 && y < i32(U.NY);
}

fn clampTau(tau : f32) -> f32 {
  return clamp(tau, TAU_MIN, TAU_MAX);
}

fn isFinite_f32(x : f32) -> bool {
  return x == x && abs(x) < 3.4e38;
}


fn feq(i : u32, rho : f32, ux : f32, uy : f32) -> f32 {
  let cu = f32(CX[i]) * ux + f32(CY[i]) * uy;
  let uu = ux * ux + uy * uy;
  return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

fn getDriveVelocity(step : u32) -> f32 {
  if (U.pulsationOn == 0u) {
    return U.u0;
  }
  let s = sin(2.0 * PI * U.pulsationFreq * f32(step));
  let u = U.u0 * (1.0 + U.pulsationAmp * s);
  return max(0.0, u);
}

fn atomicAddF32Scaled(dst : ptr<storage, atomic<i32>, read_write>, value : f32) {
  atomicAdd(dst, i32(round(value * 1e6)));
}

fn macroAt(n : u32) -> MacroCell {
  return macroField[n];
}

fn sampleMacroClamped(x : i32, y : i32) -> MacroCell {
  let xc = clamp(x, 0, i32(U.NX) - 1);
  let yc = clamp(y, 0, i32(U.NY) - 1);
  return macroField[idx(u32(xc), u32(yc))];
}

fn computeSmagNuT(rho : f32, fneqPxx : f32, fneqPyy : f32, fneqPxy : f32) -> f32 {
  let strainFactor = -1.5 / rho;
  let sxx = strainFactor * fneqPxx;
  let syy = strainFactor * fneqPyy;
  let sxy = strainFactor * fneqPxy;
  let smag = sqrt(2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy));
  return CS_DELTA_SQ * smag;
}

fn computeWaleNuT(x : i32, y : i32) -> f32 {
  if (x <= 0 || x >= i32(U.NX) - 1 || y <= 0 || y >= i32(U.NY) - 1) {
    return 0.0;
  }

  let nL = idx(u32(x - 1), u32(y));
  let nR = idx(u32(x + 1), u32(y));
  let nD = idx(u32(x), u32(y - 1));
  let nU = idx(u32(x), u32(y + 1));

  if (solid[nL] != 0u || solid[nR] != 0u || solid[nD] != 0u || solid[nU] != 0u) {
    return 0.0;
  }

  let mL = macroField[nL];
  let mR = macroField[nR];
  let mD = macroField[nD];
  let mU = macroField[nU];

  let dudx = 0.5 * (mR.ux - mL.ux);
  let dudy = 0.5 * (mU.ux - mD.ux);
  let dvdx = 0.5 * (mR.uy - mL.uy);
  let dvdy = 0.5 * (mU.uy - mD.uy);

  let sxx = dudx;
  let syy = dvdy;
  let sxy = 0.5 * (dudy + dvdx);

  let IS = sxx * sxx + syy * syy + 2.0 * sxy * sxy;

  let Gxx = dudx * dudx + dudy * dvdx;
  let Gxy = dudy * (dudx + dvdy);
  let Gyx = dvdx * (dudx + dvdy);
  let Gyy = dvdx * dudy + dvdy * dvdy;

  let trG = Gxx + Gyy;
  let Sd_xx = Gxx - (1.0 / 3.0) * trG;
  let Sd_yy = Gyy - (1.0 / 3.0) * trG;
  let Sd_xy = 0.5 * (Gxy + Gyx);

  let ID = Sd_xx * Sd_xx + Sd_yy * Sd_yy + 2.0 * Sd_xy * Sd_xy;
  if (ID <= WALE_EPS) {
    return 0.0;
  }

  let num = pow(ID, 1.5);
  let den = pow(IS, 2.5) + pow(ID, 1.25) + WALE_EPS;
  return CW_DELTA_SQ * (num / den);
}

fn applyWallModel(n : u32, ux : f32, uy : f32, nuT : f32) -> f32 {
  if (U.wallModel == 0u || U.lesModel == 0u || wallAdjacent[n] == 0u) {
    return nuT;
  }

  let y = max(wallDist[n], Y_MIN);
  let uMag = sqrt(ux * ux + uy * uy);

  var uStar = max(0.1 * uMag, 1e-6);

  for (var k : u32 = 0u; k < WALL_ITERS; k = k + 1u) {
    let yuOvNu = max((y * uStar) / U.nu0, 1e-3);
    let lhs = (1.0 / KAPPA) * log(yuOvNu) + B_LOG;
    let R = uStar * lhs - uMag;
    let Rp = lhs + 1.0 / KAPPA;

    if (abs(Rp) < 1e-10) {
      break;
    }

    let next = max(uStar - R / Rp, 1e-6);
    if (abs(next - uStar) < WALL_TOL) {
      uStar = next;
      break;
    }

    uStar = next;
  }

  let yPlus = (y * uStar) / U.nu0;
  let fDamp = 1.0 - exp(-yPlus / A_PLUS);
  return nuT * fDamp;
}
@compute @workgroup_size(16, 8, 1)
fn solverA_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  if (x >= U.NX || y >= U.NY) {
    return;
  }

  let n = idx(x, y);

  if (solid[n] != 0u) {
    macroField[n] = MacroCell(1.0, 0.0, 0.0, 0.0);
    return;
  }

  var rho = 0.0;
  var jx = 0.0;
  var jy = 0.0;

  for (var i : u32 = 0u; i < Q; i = i + 1u) {
    let fi = fIn[pdfIndex(n, i)];
    rho += fi;
    jx += fi * f32(CX[i]);
    jy += fi * f32(CY[i]);
  }

var ux = select(0.0, jx / rho, rho > 1e-10);
  var uy = select(0.0, jy / rho, rho > 1e-10);

  let badRho = rho != rho || !(rho > RHO_MIN && rho < RHO_MAX);
  let badVel = ux != ux || uy != uy;

  if (badRho || badVel) {
    let ux0 = U.u0;
    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      fOut[pdfIndex(n, i)] = feq(i, 1.0, ux0, 0.0);
    }
    macroField[n] = MacroCell(1.0, ux0, 0.0, 0.0);
    return;
  }

  let spd2 = ux * ux + uy * uy;
  if (spd2 > U_CAP * U_CAP) {
    let sc = U_CAP / sqrt(spd2);
    ux *= sc;
    uy *= sc;
  }

  macroField[n] = MacroCell(rho, ux, uy, 0.0);

  var fneqPxx = 0.0;
  var fneqPyy = 0.0;
  var fneqPxy = 0.0;

  var eqVals : array<f32, 9>;

  for (var i : u32 = 0u; i < Q; i = i + 1u) {
    let eq = feq(i, rho, ux, uy);
    eqVals[i] = eq;

    let fneq = fIn[pdfIndex(n, i)] - eq;
    fneqPxx += fneq * f32(CX[i] * CX[i]);
    fneqPyy += fneq * f32(CY[i] * CY[i]);
    fneqPxy += fneq * f32(CX[i] * CY[i]);
  }

  var nuT = 0.0;

  if (U.lesModel == 1u) {
    nuT = computeSmagNuT(rho, fneqPxx, fneqPyy, fneqPxy);
  } else if (U.lesModel == 2u) {
    nuT = computeWaleNuT(i32(x), i32(y));
  }

  nuT = applyWallModel(n, ux, uy, nuT);

  let nuTot = U.nu0 + nuT;

  if (U.collisionModel == 1u) {
    let tau = clampTau(3.0 * nuTot + 0.5);
    let omega = 1.0 / tau;

    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      let fi = fIn[pdfIndex(n, i)];
      fOut[pdfIndex(n, i)] = fi - omega * (fi - eqVals[i]);
    }
  } else {
    let tauPlus = clampTau(3.0 * nuTot + 0.5);
    let omegaPlus = 1.0 / tauPlus;
    let tauMinus = TRT_LAMBDA / (tauPlus - 0.5) + 0.5;
    let omegaMinus = 1.0 / tauMinus;

    fOut[pdfIndex(n, 0u)] =
      fIn[pdfIndex(n, 0u)] - omegaPlus * (fIn[pdfIndex(n, 0u)] - eqVals[0u]);

    let pairs = array<vec2<u32>, 4>(
      vec2<u32>(1u, 3u),
      vec2<u32>(2u, 4u),
      vec2<u32>(5u, 7u),
      vec2<u32>(6u, 8u),
    );

    for (var p : u32 = 0u; p < 4u; p = p + 1u) {
      let i = pairs[p].x;
      let ib = pairs[p].y;

      let fi = fIn[pdfIndex(n, i)];
      let fib = fIn[pdfIndex(n, ib)];

      let eqi = eqVals[i];
      let eqib = eqVals[ib];

      let fPlus = 0.5 * (fi + fib);
      let fMinus = 0.5 * (fi - fib);
      let eqPlus = 0.5 * (eqi + eqib);
      let eqMinus = 0.5 * (eqi - eqib);

      let fPlusPost = fPlus - omegaPlus * (fPlus - eqPlus);
      let fMinusPost = fMinus - omegaMinus * (fMinus - eqMinus);

      fOut[pdfIndex(n, i)] = fPlusPost + fMinusPost;
      fOut[pdfIndex(n, ib)] = fPlusPost - fMinusPost;
    }
  }
}

@compute @workgroup_size(16, 8, 1)
fn solverB_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  if (x >= U.NX || y >= U.NY) {
    return;
  }

  let n = idx(x, y);

  if (solid[n] != 0u) {
    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      fIn[pdfIndex(n, i)] = fOut[pdfIndex(n, i)];
    }
    return;
  }

  for (var i : u32 = 0u; i < Q; i = i + 1u) {
    let sx = i32(x) - CX[i];
    let sy = i32(y) - CY[i];
    let dst = pdfIndex(n, i);

    if (inBounds(sx, sy)) {
      let ns = idx(u32(sx), u32(sy));
      fIn[dst] = fOut[pdfIndex(ns, i)];
    } else {
      fIn[dst] = fOut[dst];
    }
  }

 for (var i : u32 = 1u; i < Q; i = i + 1u) {
    let off = pdfIndex(n, i);
    let opp = OPP[i];

    let nxs = i32(x) + CX[i];
    let nys = i32(y) + CY[i];

    // Live-painted / solid-mask fallback:
    // if the forward neighbor in direction i is solid, do simple bounceback now.
    if (inBounds(nxs, nys)) {
      let ns = idx(u32(nxs), u32(nys));
      if (solid[ns] != 0u) {
        fIn[off] = fOut[pdfIndex(n, opp)];
        continue;
      }
    }

    // Existing BFL path for precomputed curved-wall links.
    if (linkMask[off] == 0u) {
      continue;
    }

    let q = linkQ[off];
    let fOppHere = fOut[pdfIndex(n, opp)];

    if (abs(q - 0.5) < 1e-6) {
      fIn[off] = fOppHere;
    } else if (q < 0.5) {
      let xb = i32(x) - CX[i];
      let yb = i32(y) - CY[i];

      var fOppBehind = fOppHere;
      if (inBounds(xb, yb)) {
        let nb = idx(u32(xb), u32(yb));
        fOppBehind = fOut[pdfIndex(nb, opp)];
      }

      let inv1p2q = 1.0 / (1.0 + 2.0 * q);
      fIn[off] = inv1p2q * fOppHere + (2.0 * q * inv1p2q) * fOppBehind;
    } else {
      let xa = i32(x) + CX[i];
      let ya = i32(y) + CY[i];

      var fOppAhead = fOppHere;
      if (inBounds(xa, ya)) {
        let na = idx(u32(xa), u32(ya));
        fOppAhead = fOut[pdfIndex(na, opp)];
      }

      let inv2q = 1.0 / (2.0 * q);
      fIn[off] = inv2q * fOppHere + (1.0 - inv2q) * fOppAhead;
    }
  }


  let uDrive = getDriveVelocity(U.stepCount);

  if (U.isLidDriven != 0u) {
    if (y == 2u && x > 0u && x < U.NX - 1u) {
      for (var i : u32 = 0u; i < Q; i = i + 1u) {
        fIn[pdfIndex(n, i)] = feq(i, 1.0, uDrive, 0.0);
      }
    }

    if (y == U.NY - 3u && x > 0u && x < U.NX - 1u) {
      for (var i : u32 = 0u; i < Q; i = i + 1u) {
        fIn[pdfIndex(n, i)] = feq(i, 1.0, 0.0, 0.0);
      }
    }
   return;
  }

 if (x == 0u && solid[n] == 0u) {
    var inletU = uDrive;

    if (U.nozzleActive != 0u) {
      let dy = abs(f32(y) - U.nozzleCy);
      inletU = select(U.uCoflow, uDrive, dy <= U.nozzleHalfD);
    }

    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      fIn[pdfIndex(n, i)] = feq(i, 1.0, inletU, 0.0);
    }
  }


  if (y == 0u || y == U.NY - 1u) {
    let wallU = select(uDrive, U.uCoflow, U.nozzleActive != 0u);
    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      fIn[pdfIndex(n, i)] = feq(i, 1.0, wallU, 0.0);
    }
  }

if (x == U.NX - 1u && U.NX >= 2u) {
    let src = idx(U.NX - 2u, y);
    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      fIn[pdfIndex(n, i)] = fOut[pdfIndex(src, i)];
    }
  }

 if (f32(x) >= U.spongeStartX && x < U.NX - 1u && solid[n] == 0u) {
    var rhoS = 0.0;
    var jxS = 0.0;
    var jyS = 0.0;

    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      let fi = fIn[pdfIndex(n, i)];
      rhoS += fi;
      jxS += fi * f32(CX[i]);
      jyS += fi * f32(CY[i]);
    }

    let invR = select(1.0, 1.0 / rhoS, rhoS > 0.0);
    let uxS = jxS * invR;
    let uyS = jyS * invR;

    let sLocal = max(0.0, min(1.0, (f32(x) - U.spongeStartX) / max(U.spongeWidth, 1.0)));
    let s = U.spongeStrength * sLocal * sLocal;

    for (var i : u32 = 0u; i < Q; i = i + 1u) {
      let eq = feq(i, rhoS, uxS, uyS);
      let off = pdfIndex(n, i);
      fIn[off] = (1.0 - s) * fIn[off] + s * eq;
    }
  }

}

@compute @workgroup_size(64, 1, 1)
fn force_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  let listLength = arrayLength(&fluidSurfaceList);

  if (k >= listLength) {
    return;
  }

  let n = fluidSurfaceList[k];
  let x = i32(n % U.NX);
  let y = i32(n / U.NX);

  var nxBody = 0.0;
  var nyBody = 0.0;

  let xl = max(x - 1, 0);
  let xr = min(x + 1, i32(U.NX) - 1);
  let yd = max(y - 1, 0);
  let yu = min(y + 1, i32(U.NY) - 1);

  let gx = (sdf[idx(u32(xr), u32(y))] - sdf[idx(u32(xl), u32(y))]) / max(f32(xr - xl), 1.0);
  let gy = (sdf[idx(u32(x), u32(yu))] - sdf[idx(u32(x), u32(yd))]) / max(f32(yu - yd), 1.0);

  let glen = sqrt(gx * gx + gy * gy);
  if (glen > 1e-10) {
    nxBody = gx / glen;
    nyBody = gy / glen;
  }

  for (var i : u32 = 1u; i < Q; i = i + 1u) {
    let off = pdfIndex(n, i);
    if (linkMask[off] == 0u) {
      continue;
    }

    let cx = f32(CX[i]);
    let cy = f32(CY[i]);
    let cMag = sqrt(cx * cx + cy * cy);
    let cosA = -(cx * nxBody + cy * nyBody) / cMag;

    if (cosA <= FORCE_COS_THRESH) {
      continue;
    }

   let fi = fOut[off] - W[i] * U.rho0;

    let dFx = 2.0 * cx * fi;
    let dFy = 2.0 * cy * fi;

    let ax = f32(x) + 0.5 * cx - U.forceRefX;
    let ay = f32(y) + 0.5 * cy - U.forceRefY;
    let dMz = ax * dFy - ay * dFx;

    atomicAddF32Scaled(&forceAccum.fx, dFx);
    atomicAddF32Scaled(&forceAccum.fy, dFy);
    atomicAddF32Scaled(&forceAccum.mz, dMz);
  }
}

fn flagShowVectors() -> bool {
  return (U.displayFlags & 1u) != 0u;
}

fn flagShowGrid() -> bool {
  return (U.displayFlags & 2u) != 0u;
}

fn flagShowDist() -> bool {
  return (U.displayFlags & 4u) != 0u;
}


fn sampleVelocityBilinear(xf : f32, yf : f32) -> vec2<f32> {
  let xc = clamp(xf, 0.0, f32(U.NX) - 1.001);
  let yc = clamp(yf, 0.0, f32(U.NY) - 1.001);

  let x0 = u32(floor(xc));
  let y0 = u32(floor(yc));
  let x1 = min(x0 + 1u, U.NX - 1u);
  let y1 = min(y0 + 1u, U.NY - 1u);

  let tx = xc - f32(x0);
  let ty = yc - f32(y0);

  let m00 = macroField[idx(x0, y0)];
  let m10 = macroField[idx(x1, y0)];
  let m01 = macroField[idx(x0, y1)];
  let m11 = macroField[idx(x1, y1)];

  let ux0 = mix(m00.ux, m10.ux, tx);
  let ux1 = mix(m01.ux, m11.ux, tx);
  let uy0 = mix(m00.uy, m10.uy, tx);
  let uy1 = mix(m01.uy, m11.uy, tx);

  return vec2<f32>(
    mix(ux0, ux1, ty),
    mix(uy0, uy1, ty)
  );
}
@compute @workgroup_size(128, 1, 1)
fn particle_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let count = arrayLength(&particles);

  if (i >= count) {
    return;
  }

  var p = particles[i];
  let vel = sampleVelocityBilinear(p.x, p.y);

  p.vx = vel.x;
  p.vy = vel.y;
  p.x = p.x + vel.x;
  p.y = p.y + vel.y;

  let xi = u32(clamp(i32(p.x), 0, i32(U.NX) - 1));
  let yi = u32(clamp(i32(p.y), 0, i32(U.NY) - 1));
  let n = idx(xi, yi);

  let bad =
    p.x < 0.0 ||
    p.x >= f32(U.NX - 1u) ||
    p.y < 1.0 ||
    p.y >= f32(U.NY - 1u) ||
    solid[n] != 0u ||
    (vel.x * vel.x + vel.y * vel.y) < 1e-8;

  if (bad) {
    p.x = 0.5;
    p.y = f32(U.NY) * 0.5;
    p.vx = 0.0;
    p.vy = 0.0;
  }

  particles[i] = p;
}

@compute @workgroup_size(64, 1, 1)
fn reduction_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = gid.x;
  if (n >= U.N) {
    return;
  }

  if (solid[n] != 0u) {
    return;
  }

  let m = macroField[n];
  let p = prevMacroField[n];

  let speed = sqrt(m.ux * m.ux + m.uy * m.uy);
  let du = m.ux - p.ux;
  let dv = m.uy - p.uy;
  let rms = sqrt(du * du + dv * dv);
  let rhoDev = abs(m.rho - 1.0);

  atomicAddF32Scaled(&reductionOut.meanSpeed, speed);
  atomicAddF32Scaled(&reductionOut.rmsDeltaU, rms);
  atomicAddF32Scaled(&reductionOut.meanRhoDev, rhoDev);
  atomicAdd(&reductionOut.fluidCount, 1u);
}

@compute @workgroup_size(64, 1, 1)
fn reopen_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  let count = arrayLength(&reopenNodeList);

  if (k >= count) {
    return;
  }

  let n = reopenNodeList[k];
  if (n >= U.N) {
    return;
  }

  if (solid[n] != 0u) {
    return;
  }

  let x = i32(n % U.NX);
  let y = i32(n / U.NX);

  var rhoSum = 0.0;
  var uxSum = 0.0;
  var uySum = 0.0;
  var cnt = 0u;

  for (var j : u32 = 1u; j < Q; j = j + 1u) {
    let xn = x + CX[j];
    let yn = y + CY[j];

    if (!inBounds(xn, yn)) {
      continue;
    }

    let nn = idx(u32(xn), u32(yn));
    if (solid[nn] != 0u) {
      continue;
    }

    let m = macroField[nn];
    rhoSum += m.rho;
    uxSum += m.ux;
    uySum += m.uy;
    cnt += 1u;
  }

  var rho = 1.0;
  var ux = 0.0;
  var uy = 0.0;

  if (cnt > 0u) {
    let inv = 1.0 / f32(cnt);
    rho = rhoSum * inv;
    ux = uxSum * inv;
    uy = uySum * inv;
  }

  macroField[n] = MacroCell(rho, ux, uy, 0.0);
  prevMacroField[n] = MacroCell(rho, ux, uy, 0.0);

  for (var i : u32 = 0u; i < Q; i = i + 1u) {
    let eq = feq(i, rho, ux, uy);
    fIn[pdfIndex(n, i)] = eq;
    fOut[pdfIndex(n, i)] = eq;
  }
}

// ---------------------------------------------------------
// Field render
// ---------------------------------------------------------
@group(0) @binding(1) var<storage, read> renderMacroField : array<MacroCell>;
@group(0) @binding(2) var<storage, read> renderSolid : array<u32>;

@vertex
fn field_vs(@builtin(vertex_index) vid : u32) -> VsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0)
  );

  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 2.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(2.0, 0.0)
  );

  var out : VsOut;
  out.position = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

fn sampleRenderMacroClamped(x : i32, y : i32) -> MacroCell {
  let xc = clamp(x, 0, i32(U.NX) - 1);
  let yc = clamp(y, 0, i32(U.NY) - 1);
  return renderMacroField[idx(u32(xc), u32(yc))];
}

fn colormap(t : f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);

  if (x < 0.25) {
    let a = x / 0.25;
    return mix(vec3<f32>(0.0, 0.0, 0.7059), vec3<f32>(0.0, 0.8235, 1.0), a);
  }
  if (x < 0.5) {
    let a = (x - 0.25) / 0.25;
    return mix(vec3<f32>(0.0, 0.8235, 1.0), vec3<f32>(0.0, 1.0, 0.3137), a);
  }
  if (x < 0.75) {
    let a = (x - 0.5) / 0.25;
    return mix(vec3<f32>(0.0, 1.0, 0.3137), vec3<f32>(1.0, 0.8627, 0.0), a);
  }

  let a = (x - 0.75) / 0.25;
  return mix(vec3<f32>(1.0, 0.8627, 0.0), vec3<f32>(1.0, 0.0784, 0.0), a);
}

fn qcritAt(x : i32, y : i32) -> f32 {
  let mL = sampleRenderMacroClamped(x - 1, y);
  let mR = sampleRenderMacroClamped(x + 1, y);
  let mD = sampleRenderMacroClamped(x, y - 1);
  let mU = sampleRenderMacroClamped(x, y + 1);


  let dudx = 0.5 * (mR.ux - mL.ux);
  let dudy = 0.5 * (mU.ux - mD.ux);
  let dvdx = 0.5 * (mR.uy - mL.uy);
  let dvdy = 0.5 * (mU.uy - mD.uy);

  let omega = dvdx - dudy;
  let strain2 = dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx) * (dudy + dvdx);

  return 0.5 * (omega * omega - 2.0 * strain2);
}

@fragment
fn field_fs(in : VsOut) -> @location(0) vec4<f32> {
  let xf = clamp(in.uv.x, 0.0, 0.999999) * f32(U.NX);
  let yf = clamp(in.uv.y, 0.0, 0.999999) * f32(U.NY);

  let x0 = u32(floor(xf));
  let y0 = u32(floor(yf));
  let x1 = min(x0 + 1u, U.NX - 1u);
  let y1 = min(y0 + 1u, U.NY - 1u);

  let tx = fract(xf);
  let ty = fract(yf);

  let n = idx(x0, y0);

  if (renderSolid[idx(x0, y0)] != 0u) {
    return vec4<f32>(40.0 / 255.0, 40.0 / 255.0, 45.0 / 255.0, 1.0);
  }

  let m00 = renderMacroField[idx(x0, y0)];
  let m10 = renderMacroField[idx(x1, y0)];
  let m01 = renderMacroField[idx(x0, y1)];
  let m11 = renderMacroField[idx(x1, y1)];

  let s00 = select(1.0, 0.0, renderSolid[idx(x0, y0)] != 0u);
  let s10 = select(1.0, 0.0, renderSolid[idx(x1, y0)] != 0u);
  let s01 = select(1.0, 0.0, renderSolid[idx(x0, y1)] != 0u);
  let s11 = select(1.0, 0.0, renderSolid[idx(x1, y1)] != 0u);

  let ux = (mix(m00.ux * s00, m10.ux * s10, tx) * (1.0 - ty)
          + mix(m01.ux * s01, m11.ux * s11, tx) * ty);
  let uy = (mix(m00.uy * s00, m10.uy * s10, tx) * (1.0 - ty)
          + mix(m01.uy * s01, m11.uy * s11, tx) * ty);
  let rho = (mix(m00.rho, m10.rho, tx) * (1.0 - ty)
           + mix(m01.rho, m11.rho, tx) * ty);

  var t = 0.0;

  if (U.displayMode == 1u) {
    let q = qcritAt(i32(x0), i32(y0));
    let qScaled = clamp(q * 500.0, -1.0, 1.0);
    t = 0.5 * (qScaled + 1.0);
  } else {
    let speed = sqrt(ux * ux + uy * uy);
    t = clamp(speed / max(U.colorScale, 1e-6), 0.0, 1.0);
  }

  var rgb = colormap(t);

  if (flagShowDist()) {
    let rhoDev = abs(rho - 1.0);
    let d = clamp(rhoDev * 30.0, 0.0, 1.0);
    rgb = mix(rgb, vec3<f32>(1.0, 1.0, 1.0), 0.35 * d);
  }

  if (flagShowGrid()) {
    let gx = fract(xf);
    let gy = fract(yf);
    let lineThresh = clamp(1.5 / max(f32(U.NX) / 32.0, 1.0), 0.018, 0.10);
    let line = select(
      0.0,
      1.0,
      gx < lineThresh || gx > (1.0 - lineThresh) ||
      gy < lineThresh || gy > (1.0 - lineThresh)
    );
    rgb = mix(rgb, vec3<f32>(0.92, 0.96, 1.0), 0.18 * line);
  }

  if (flagShowVectors()) {
    let spacing = max(8.0, round(f32(U.NX) / 22.0));

    let cellX = floor(xf / spacing) * spacing + 0.5 * spacing;
    let cellY = floor(yf / spacing) * spacing + 0.5 * spacing;

    let sx = u32(clamp(i32(cellX), 0, i32(U.NX) - 1));
    let sy = u32(clamp(i32(cellY), 0, i32(U.NY) - 1));
    let sn = idx(sx, sy);

    if (renderSolid[sn] == 0u) {
      let vm = renderMacroField[sn];
      let vx = vm.ux;
      let vy = vm.uy;
      let vmag = sqrt(vx * vx + vy * vy);

      if (vmag > 1e-6) {
        let dir = vec2<f32>(vx, vy) / vmag;

        let local = vec2<f32>(xf - cellX, yf - cellY);
        let along = dot(local, dir);
        let perp = dot(local, vec2<f32>(-dir.y, dir.x));

        let halfLen = clamp(1.5 + 8.0 * (vmag / max(U.u0, 1e-6)), 1.5, 9.0);
        let shaft = select(
          0.0,
          1.0,
          abs(perp) < 0.22 && along > -halfLen && along < halfLen
        );

        let tipBase = dir * halfLen;
        let tip1 = vec2<f32>(
          local.x - tipBase.x + 0.9 * dir.x - 0.7 * (-dir.y),
          local.y - tipBase.y + 0.9 * dir.y - 0.7 * dir.x
        );
        let tip2 = vec2<f32>(
          local.x - tipBase.x + 0.9 * dir.x + 0.7 * (-dir.y),
          local.y - tipBase.y + 0.9 * dir.y + 0.7 * dir.x
        );

        let head1 = select(0.0, 1.0, length(tip1) < 0.28);
        let head2 = select(0.0, 1.0, length(tip2) < 0.28);

        let vecMask = max(shaft, max(head1, head2));
        rgb = mix(rgb, vec3<f32>(0.95, 0.98, 1.0), 0.75 * vecMask);
      }
    }
  }

  return vec4<f32>(rgb, 1.0);
}




// ---------------------------------------------------------
// Particle render
// ---------------------------------------------------------
@group(0) @binding(1) var<storage, read> renderParticles : array<Particle>;
@group(0) @binding(2) var<storage, read> particleMacroField : array<MacroCell>;

@vertex
fn particle_vs(
  @builtin(vertex_index) vertexIndex : u32,
  @builtin(instance_index) instanceIndex : u32
) -> ParticleVsOut {
  let p = renderParticles[instanceIndex];

  let xi = u32(clamp(i32(p.x), 0, i32(U.NX) - 1));
  let yi = u32(clamp(i32(p.y), 0, i32(U.NY) - 1));
  let m = particleMacroField[idx(xi, yi)];

  let speed = sqrt(p.vx * p.vx + p.vy * p.vy);
  let uRef = max(U.u0, 1e-6);
  let t = clamp(speed / uRef, 0.0, 1.0);
  let alpha = PARTICLE_ALPHA_MIN + (PARTICLE_ALPHA_MAX - PARTICLE_ALPHA_MIN) * t;

  let radiusPx = PARTICLE_RADIUS_PX + 1.2 * t;

  let quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0)
  );

  let local = quad[vertexIndex];

  let px = ((p.x + 0.5) / f32(U.NX)) * 2.0 - 1.0;
  let py = 1.0 - ((p.y + 0.5) / f32(U.NY)) * 2.0;

 let radiusNdcX = (radiusPx / max(U.renderWidth, 1.0)) * 2.0;
  let radiusNdcY = (radiusPx / max(U.renderHeight, 1.0)) * 2.0;

  var out : ParticleVsOut;
  out.position = vec4<f32>(
    px + local.x * radiusNdcX,
    py + local.y * radiusNdcY,
    0.0,
    1.0
  );
  out.alpha = alpha;
  out.localUv = local;
  _ = m; // reserved if future color-by-local-flow is desired
  return out;
}

@fragment
fn particle_fs(in : ParticleVsOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.localUv, in.localUv);
  if (r2 > 1.0) {
    discard;
  }

  let edge = smoothstep(1.0, 0.65, r2);
  let alpha = in.alpha * edge;

  let rgb = vec3<f32>(210.0 / 255.0, 235.0 / 255.0, 1.0);
  return vec4<f32>(rgb, alpha);
}
