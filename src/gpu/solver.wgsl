// src/gpu/solver.wgsl
// ============================================================
// WebGPU port of the CPU reference solver (LBM-solver- V7).
//
// One CPU time step  (collideFused -> stream -> applyBoundaryConditions)
// maps onto these passes, in order:
//
//   macro_main   moments, velocity cap, bad-node re-equilibration flag
//   collide_main TRT / BGK + Smagorinsky / WALE + wall model   fA -> fB
//   stream_main  pull streaming of interior nodes (solid nodes included,
//                exactly like the CPU in-place sweep), BFL wall
//                reconstruction from the post-stream snapshot, full-way
//                bounce-back on painted nodes, inlet / sponge / top-bottom
//                or lid boundary conditions                       fB -> fA
//   outlet_main  zero-gradient outlet copy (x = NX-1 <- x = NX-2)
//
// Populations are stored structure-of-arrays: f[i * N + n].
// After a full step the current populations are always in fA.
// ============================================================

const Q : u32 = 9u;

const NODE_SOLID   : u32 = 1u;
const NODE_PAINTED : u32 = 2u;
const NODE_WALLADJ : u32 = 4u;

// Stability caps (collisionFused.js)
const U_CAP   : f32 = 0.28;
const RHO_MIN : f32 = 0.35;
const RHO_MAX : f32 = 3.0;
const TAU_MIN : f32 = 0.545;
const TAU_MAX : f32 = 5.0;

// LES
const CS_DELTA_SQ : f32 = 0.17 * 0.17;
const CW_DELTA_SQ : f32 = 0.325 * 0.325;
const WALE_EPS    : f32 = 1e-20;

// Wall model
const KAPPA      : f32 = 0.41;
const B_LOG      : f32 = 5.2;
const A_PLUS     : f32 = 25.0;
const WALL_ITERS : u32 = 6u;
const WALL_TOL   : f32 = 1e-4;
const Y_MIN      : f32 = 0.5;

const TRT_LAMBDA : f32 = 3.0 / 16.0;
const PAIR_A = array<u32, 4>(1u, 2u, 5u, 6u);
const PAIR_B = array<u32, 4>(3u, 4u, 7u, 8u);

// Force (forceSurface.js)
const COS_THRESH : f32 = 0.1;

struct Params {
  NX : u32,
  NY : u32,
  N : u32,
  isCavity : u32,

  collisionModel : u32,   // 0 = TRT, 1 = BGK
  lesModel : u32,         // 0 = off, 1 = Smagorinsky, 2 = WALE
  wallModel : u32,
  nozzleActive : u32,

  nu0 : f32,
  uCoflow : f32,
  nozzleCy : f32,
  nozzleHalfD : f32,

  spongeStart : i32,
  spongeWidth : f32,
  spongeStrength : f32,
  rho0 : f32,

  forceRefX : f32,
  forceRefY : f32,
  colorScale : f32,
  displayMode : u32,      // 0 = speed, 1 = qcrit

  particleDt : f32,
  particleCount : u32,
  frameSeed : u32,
  surfCount : u32,

  canvasW : f32,
  canvasH : f32,
  _pad0 : u32,
  _pad1 : u32,
};

struct StepParams {
  uInst : f32,   // instantaneous inlet / lid velocity for this step
  step : u32,
  _a : u32,
  _b : u32,
};

@group(0) @binding(0)  var<uniform> P : Params;
@group(0) @binding(1)  var<uniform> S : StepParams;
@group(0) @binding(2)  var<storage, read_write> fA : array<f32>;
@group(0) @binding(3)  var<storage, read_write> fB : array<f32>;
@group(0) @binding(4)  var<storage, read_write> mac : array<vec4<f32>>;   // rho, ux, uy, reequilFlag
@group(0) @binding(5)  var<storage, read> nodeType : array<u32>;
@group(0) @binding(6)  var<storage, read> wallDist : array<f32>;
@group(0) @binding(7)  var<storage, read> linkQ : array<f32>;            // SoA, 0 = no link
@group(0) @binding(8)  var<storage, read> sdf : array<f32>;
@group(0) @binding(9)  var<storage, read> surfList : array<u32>;
@group(0) @binding(10) var<storage, read_write> partials : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> prevMac : array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> particles : array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> gatherOut : array<f32>;

// ---------------------------------------------------------
// Lattice helpers
// ---------------------------------------------------------
const CXA = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CYA = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const OPPA = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
const WA = array<f32, 9>(
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);

fn cx(i : u32) -> i32 { return CXA[i]; }
fn cy(i : u32) -> i32 { return CYA[i]; }
fn opp(i : u32) -> u32 { return OPPA[i]; }
fn wq(i : u32) -> f32 { return WA[i]; }

fn feq(i : u32, r : f32, u : f32, v : f32) -> f32 {
  let cu = f32(cx(i)) * u + f32(cy(i)) * v;
  let uu = u * u + v * v;
  return wq(i) * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

// Bit-level finiteness test (immune to fast-math NaN folding).
fn finite(x : f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn isSolid(n : u32) -> bool {
  return (nodeType[n] & NODE_SOLID) != 0u;
}

fn inBounds(x : i32, y : i32) -> bool {
  return x >= 0 && x < i32(P.NX) && y >= 0 && y < i32(P.NY);
}

fn nodeIndex(x : i32, y : i32) -> u32 {
  return u32(y) * P.NX + u32(x);
}

// Post-streaming population j at (x, y), read from the post-collision
// buffer fB. Interior nodes pull from their upstream neighbour; edge nodes
// are not streamed by the CPU sweep and keep their own value.
fn postStream(x : i32, y : i32, j : u32) -> f32 {
  let NX = i32(P.NX);
  let NY = i32(P.NY);
  if (x >= 1 && x <= NX - 2 && y >= 1 && y <= NY - 2) {
    return fB[j * P.N + nodeIndex(x - cx(j), y - cy(j))];
  }
  return fB[j * P.N + nodeIndex(x, y)];
}

// ---------------------------------------------------------
// Pass 1: macroscopic moments (collideFused.js, first half)
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn macro_main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  if (isSolid(n)) { return; }

  var r = 0.0;
  var jx = 0.0;
  var jy = 0.0;
  for (var i = 0u; i < Q; i++) {
    let fi = fA[i * P.N + n];
    r += fi;
    jx += fi * f32(cx(i));
    jy += fi * f32(cy(i));
  }

  if (!finite(r) || r < RHO_MIN || r > RHO_MAX) {
    mac[n] = vec4<f32>(1.0, S.uInst, 0.0, 1.0);
    return;
  }

  var u = jx / r;
  var v = jy / r;
  if (!finite(u) || !finite(v)) {
    mac[n] = vec4<f32>(1.0, S.uInst, 0.0, 1.0);
    return;
  }

  let spd2 = u * u + v * v;
  if (spd2 > U_CAP * U_CAP) {
    let sc = U_CAP / sqrt(spd2);
    u *= sc;
    v *= sc;
  }

  mac[n] = vec4<f32>(r, u, v, 0.0);
}

// ---------------------------------------------------------
// Pass 2: collision (collideFused.js, second half)
// ---------------------------------------------------------
fn waleNuT(x : i32, y : i32) -> f32 {
  if (x <= 0 || x >= i32(P.NX) - 1 || y <= 0 || y >= i32(P.NY) - 1) { return 0.0; }

  let nL = nodeIndex(x - 1, y);
  let nR = nodeIndex(x + 1, y);
  let nD = nodeIndex(x, y - 1);
  let nU = nodeIndex(x, y + 1);
  if (isSolid(nL) || isSolid(nR) || isSolid(nD) || isSolid(nU)) { return 0.0; }

  let dudx = 0.5 * (mac[nR].y - mac[nL].y);
  let dudy = 0.5 * (mac[nU].y - mac[nD].y);
  let dvdx = 0.5 * (mac[nR].z - mac[nL].z);
  let dvdy = 0.5 * (mac[nU].z - mac[nD].z);

  let Sxx = dudx;
  let Syy = dvdy;
  let Sxy = 0.5 * (dudy + dvdx);
  let IS = Sxx * Sxx + Syy * Syy + 2.0 * Sxy * Sxy;

  let Gxx = dudx * dudx + dudy * dvdx;
  let Gxy = dudy * (dudx + dvdy);
  let Gyx = dvdx * (dudx + dvdy);
  let Gyy = dvdx * dudy + dvdy * dvdy;

  let trG = Gxx + Gyy;
  let Sdxx = Gxx - (1.0 / 3.0) * trG;
  let Sdyy = Gyy - (1.0 / 3.0) * trG;
  let Sdxy = 0.5 * (Gxy + Gyx);
  let ID = Sdxx * Sdxx + Sdyy * Sdyy + 2.0 * Sdxy * Sdxy;
  if (ID <= WALE_EPS) { return 0.0; }

  let num = pow(ID, 1.5);
  let den = pow(IS, 2.5) + pow(ID, 1.25) + WALE_EPS;
  return CW_DELTA_SQ * (num / den);
}

fn wallModelDamp(n : u32, u : f32, v : f32, nuT : f32) -> f32 {
  if (P.wallModel == 0u || P.lesModel == 0u) { return nuT; }
  if ((nodeType[n] & NODE_WALLADJ) == 0u) { return nuT; }

  let y = max(wallDist[n], Y_MIN);
  let uMag = sqrt(u * u + v * v);

  var uStar = max(0.1 * uMag, 1e-6);
  for (var k = 0u; k < WALL_ITERS; k++) {
    let yuOvNu = max((y * uStar) / P.nu0, 1e-3);
    let lhs = (1.0 / KAPPA) * log(yuOvNu) + B_LOG;
    let R = uStar * lhs - uMag;
    let Rp = lhs + 1.0 / KAPPA;
    if (abs(Rp) < 1e-10) { break; }

    var next = uStar - R / Rp;
    if (!finite(next) || next <= 0.0) { next = 1e-6; }
    if (abs(next - uStar) < WALL_TOL) {
      uStar = next;
      break;
    }
    uStar = next;
  }

  let yPlus = (y * uStar) / P.nu0;
  let fDamp = 1.0 - exp(-yPlus / A_PLUS);
  return nuT * fDamp;
}

@compute @workgroup_size(64, 1, 1)
fn collide_main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  let N = P.N;

  // Solid nodes are not collided: carry their populations through.
  if (isSolid(n)) {
    for (var i = 0u; i < Q; i++) { fB[i * N + n] = fA[i * N + n]; }
    return;
  }

  let m = mac[n];
  if (m.w > 0.5) {
    for (var i = 0u; i < Q; i++) { fB[i * N + n] = feq(i, 1.0, S.uInst, 0.0); }
    return;
  }

  let r = m.x;
  let u = m.y;
  let v = m.z;

  var fl : array<f32, 9>;
  var eq : array<f32, 9>;
  var Pxx = 0.0;
  var Pyy = 0.0;
  var Pxy = 0.0;
  for (var i = 0u; i < Q; i++) {
    let c_x = f32(cx(i));
    let c_y = f32(cy(i));
    fl[i] = fA[i * N + n];
    eq[i] = feq(i, r, u, v);
    let fneq = fl[i] - eq[i];
    Pxx += c_x * c_x * fneq;
    Pyy += c_y * c_y * fneq;
    Pxy += c_x * c_y * fneq;
  }

  var nuT = 0.0;
  if (P.lesModel == 1u) {
    let sf = -1.5 / r;
    let Sxx = sf * Pxx;
    let Syy = sf * Pyy;
    let Sxy = sf * Pxy;
    nuT = CS_DELTA_SQ * sqrt(2.0 * (Sxx * Sxx + Syy * Syy + 2.0 * Sxy * Sxy));
  } else if (P.lesModel == 2u) {
    nuT = waleNuT(i32(g.x), i32(g.y));
  }
  nuT = wallModelDamp(n, u, v, nuT);

  let nuTot = P.nu0 + nuT;

  if (P.collisionModel == 1u) {
    let tau = clamp(3.0 * nuTot + 0.5, TAU_MIN, TAU_MAX);
    let omega = 1.0 / tau;
    for (var i = 0u; i < Q; i++) {
      fB[i * N + n] = fl[i] - omega * (fl[i] - eq[i]);
    }
    return;
  }

  let tauPlus = clamp(3.0 * nuTot + 0.5, TAU_MIN, TAU_MAX);
  let omegaPlus = 1.0 / tauPlus;
  let tauMinus = TRT_LAMBDA / (tauPlus - 0.5) + 0.5;
  let omegaMinus = 1.0 / tauMinus;

  fB[n] = fl[0] - omegaPlus * (fl[0] - eq[0]);

  for (var p = 0u; p < 4u; p++) {
    let i = PAIR_A[p];
    let ib = PAIR_B[p];

    let fPlus = 0.5 * (fl[i] + fl[ib]);
    let fMinus = 0.5 * (fl[i] - fl[ib]);
    let eqPlus = 0.5 * (eq[i] + eq[ib]);
    let eqMinus = 0.5 * (eq[i] - eq[ib]);

    let fPlusPost = fPlus - omegaPlus * (fPlus - eqPlus);
    let fMinusPost = fMinus - omegaMinus * (fMinus - eqMinus);

    fB[i * N + n] = fPlusPost + fMinusPost;
    fB[ib * N + n] = fPlusPost - fMinusPost;
  }
}

// ---------------------------------------------------------
// Pass 3: streaming + BFL + painted bounce-back + boundaries
// (streaming.js stream() followed by boundary.js)
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn stream_main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let NX = i32(P.NX);
  let NY = i32(P.NY);
  let n = g.y * P.NX + g.x;
  let N = P.N;

  var r : array<f32, 9>;
  for (var j = 0u; j < Q; j++) { r[j] = postStream(x, y, j); }

  // BFL interpolated bounce-back on links into solid geometry.
  for (var i = 1u; i < Q; i++) {
    let q = linkQ[i * N + n];
    if (q <= 0.0) { continue; }

    let o = opp(i);
    let fOppHere = postStream(x, y, o);

    if (abs(q - 0.5) < 1e-6) {
      r[i] = fOppHere;
    } else if (q < 0.5) {
      let xb = x - cx(i);
      let yb = y - cy(i);
      var fOppBehind = fOppHere;
      if (inBounds(xb, yb)) { fOppBehind = postStream(xb, yb, o); }
      let inv1p2q = 1.0 / (1.0 + 2.0 * q);
      r[i] = inv1p2q * fOppHere + (2.0 * q * inv1p2q) * fOppBehind;
    } else {
      let xa = x + cx(i);
      let ya = y + cy(i);
      var fOppAhead = fOppHere;
      if (inBounds(xa, ya)) { fOppAhead = postStream(xa, ya, o); }
      let inv2q = 1.0 / (2.0 * q);
      r[i] = inv2q * fOppHere + (1.0 - inv2q) * fOppAhead;
    }
  }

  // Full-way bounce-back on user-painted wall nodes.
  if ((nodeType[n] & NODE_PAINTED) != 0u) {
    var t = r[1]; r[1] = r[3]; r[3] = t;
    t = r[2]; r[2] = r[4]; r[4] = t;
    t = r[5]; r[5] = r[7]; r[7] = t;
    t = r[6]; r[6] = r[8]; r[8] = t;
  }

  let solid = isSolid(n);

  if (P.isCavity != 0u) {
    // Moving lid on the first interior fluid row, resting row at the bottom.
    if (x >= 1 && x <= NX - 2 && !solid) {
      if (y == 2) {
        for (var i = 0u; i < Q; i++) { r[i] = feq(i, 1.0, S.uInst, 0.0); }
      }
      if (y == NY - 3) {
        for (var i = 0u; i < Q; i++) { r[i] = feq(i, 1.0, 0.0, 0.0); }
      }
    }
  } else {
    // Left inlet (split velocity profile for the nozzle).
    if (x == 0 && !solid) {
      var inletU = S.uInst;
      if (P.nozzleActive != 0u) {
        inletU = select(P.uCoflow, S.uInst, abs(f32(y) - P.nozzleCy) <= P.nozzleHalfD);
      }
      for (var i = 0u; i < Q; i++) { r[i] = feq(i, 1.0, inletU, 0.0); }
    }

    // Outlet sponge, relaxed toward equilibrium at this step's macros.
    if (x >= P.spongeStart && x <= NX - 2 && !solid) {
      let t = f32(x - P.spongeStart) / P.spongeWidth;
      let s = P.spongeStrength * t * t;
      let m = mac[n];
      for (var i = 0u; i < Q; i++) {
        r[i] = (1.0 - s) * r[i] + s * feq(i, m.x, m.y, m.z);
      }
    }

    // Top and bottom: freestream (or co-flow for the nozzle).
    if (y == 0 || y == NY - 1) {
      let wallU = select(S.uInst, P.uCoflow, P.nozzleActive != 0u);
      for (var i = 0u; i < Q; i++) { r[i] = feq(i, 1.0, wallU, 0.0); }
    }
  }

  for (var i = 0u; i < Q; i++) { fA[i * N + n] = r[i]; }
}

// ---------------------------------------------------------
// Pass 4: zero-gradient outlet copy (non-cavity only)
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn outlet_main(@builtin(global_invocation_id) g : vec3<u32>) {
  let y = g.x;
  if (y < 1u || y + 1u >= P.NY || P.NX < 2u) { return; }
  let dst = y * P.NX + (P.NX - 1u);
  let src = dst - 1u;
  for (var i = 0u; i < Q; i++) {
    fA[i * P.N + dst] = fA[i * P.N + src];
  }
}

// ---------------------------------------------------------
// Rest equilibrium for every solid node (after a geometry rebuild)
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn rest_solids(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  if (!isSolid(n)) { return; }
  for (var i = 0u; i < Q; i++) { fA[i * P.N + n] = wq(i); }
}

// ---------------------------------------------------------
// Diagnostics: normal-filtered fluctuation momentum exchange
// (forceSurface.js). Uses the post-stream / pre-BFL snapshot of
// the last step, reconstructed from fB.
// ---------------------------------------------------------
var<workgroup> wsum : array<vec4<f32>, 256>;

fn reduceWorkgroup(li : u32, size : u32) {
  var stride = size / 2u;
  loop {
    if (stride == 0u) { break; }
    workgroupBarrier();
    if (li < stride) { wsum[li] = wsum[li] + wsum[li + stride]; }
    stride = stride / 2u;
  }
}

@compute @workgroup_size(64, 1, 1)
fn force_main(
  @builtin(global_invocation_id) g : vec3<u32>,
  @builtin(local_invocation_index) li : u32,
  @builtin(workgroup_id) wg : vec3<u32>
) {
  var acc = vec4<f32>(0.0);
  let k = g.x;

  if (k < P.surfCount) {
    let n = surfList[k];
    let NX = i32(P.NX);
    let NY = i32(P.NY);
    let xn = i32(n % P.NX);
    let yn = i32(n / P.NX);

    let xl = select(xn, xn - 1, xn > 0);
    let xr = select(xn, xn + 1, xn < NX - 1);
    let yd = select(yn, yn - 1, yn > 0);
    let yu = select(yn, yn + 1, yn < NY - 1);
    let gx = (sdf[nodeIndex(xr, yn)] - sdf[nodeIndex(xl, yn)]) / f32(xr - xl);
    let gy = (sdf[nodeIndex(xn, yu)] - sdf[nodeIndex(xn, yd)]) / f32(yu - yd);
    let glen = sqrt(gx * gx + gy * gy);
    var nxb = 0.0;
    var nyb = 0.0;
    if (glen > 1e-10) {
      nxb = gx / glen;
      nyb = gy / glen;
    }

    for (var i = 1u; i < Q; i++) {
      if (linkQ[i * P.N + n] <= 0.0) { continue; }
      let c_x = f32(cx(i));
      let c_y = f32(cy(i));
      let cMag = sqrt(c_x * c_x + c_y * c_y);
      let cosA = -(c_x * nxb + c_y * nyb) / cMag;
      if (cosA <= COS_THRESH) { continue; }

      let fi = postStream(xn, yn, i) - wq(i) * P.rho0;
      let dFx = 2.0 * c_x * fi;
      let dFy = 2.0 * c_y * fi;
      let ax = f32(xn) + 0.5 * c_x - P.forceRefX;
      let ay = f32(yn) + 0.5 * c_y - P.forceRefY;
      acc += vec4<f32>(dFx, dFy, ax * dFy - ay * dFx, 0.0);
    }
  }

  wsum[li] = acc;
  reduceWorkgroup(li, 64u);
  if (li == 0u) { partials[wg.x] = wsum[0]; }
}

// ---------------------------------------------------------
// Diagnostics: convergence sums (convergenceMonitor.js)
// partial = (sum speed, sum |rho-1|, sum |u - uPrev|^2, fluid count)
// ---------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn reduce_main(
  @builtin(global_invocation_id) g : vec3<u32>,
  @builtin(local_invocation_index) li : u32,
  @builtin(workgroup_id) wg : vec3<u32>
) {
  var acc = vec4<f32>(0.0);
  let n = g.x;
  if (n < P.N && !isSolid(n)) {
    let m = mac[n];
    let p = prevMac[n];
    let du = m.y - p.y;
    let dv = m.z - p.z;
    acc = vec4<f32>(sqrt(m.y * m.y + m.z * m.z), abs(m.x - 1.0), du * du + dv * dv, 1.0);
  }
  wsum[li] = acc;
  reduceWorkgroup(li, 256u);
  if (li == 0u) { partials[wg.x] = wsum[0]; }
}

// ---------------------------------------------------------
// Smoke tracer particles (particles.js step/respawn)
// particle = (x, y, vx, vy)
// ---------------------------------------------------------
fn hash32(a : u32) -> u32 {
  var x = a;
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

fn rand01(seed : ptr<function, u32>) -> f32 {
  *seed = hash32(*seed + 0x9e3779b9u);
  return f32(*seed >> 8u) / 16777216.0;
}

fn sampleVel(xp : f32, yp : f32) -> vec2<f32> {
  let xc = clamp(xp, 0.0, f32(P.NX) - 1.001);
  let yc = clamp(yp, 0.0, f32(P.NY) - 1.001);
  let x0 = u32(floor(xc));
  let y0 = u32(floor(yc));
  let x1 = min(x0 + 1u, P.NX - 1u);
  let y1 = min(y0 + 1u, P.NY - 1u);
  let tx = xc - f32(x0);
  let ty = yc - f32(y0);
  let v00 = mac[y0 * P.NX + x0].yz;
  let v10 = mac[y0 * P.NX + x1].yz;
  let v01 = mac[y1 * P.NX + x0].yz;
  let v11 = mac[y1 * P.NX + x1].yz;
  let a = v00 * (1.0 - tx) + v10 * tx;
  let b = v01 * (1.0 - tx) + v11 * tx;
  return a * (1.0 - ty) + b * ty;
}

fn solidAt(xp : f32, yp : f32) -> bool {
  let xi = clamp(i32(xp), 0, i32(P.NX) - 1);
  let yi = clamp(i32(yp), 0, i32(P.NY) - 1);
  return isSolid(nodeIndex(xi, yi));
}

@compute @workgroup_size(64, 1, 1)
fn particle_main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= P.particleCount) { return; }

  let p = particles[i];
  let vel = sampleVel(p.x, p.y);
  var x = p.x + vel.x * P.particleDt;
  var y = p.y + vel.y * P.particleDt;
  var out = vec4<f32>(x, y, vel.x, vel.y);

  let NXf = f32(P.NX);
  let NYf = f32(P.NY);
  var recycle = !finite(x) || !finite(y);
  if (!recycle) {
    recycle = x < 0.0 || x >= NXf - 1.0 || y < 1.0 || y >= NYf - 1.0;
  }
  if (!recycle) {
    recycle = solidAt(x, y) || (vel.x * vel.x + vel.y * vel.y) < 1e-8;
  }

  if (recycle) {
    var seed = hash32(i * 747796405u + P.frameSeed * 2891336453u + 1u);
    var placed = false;
    for (var k = 0u; k < 8u; k++) {
      var rx = 0.0;
      var ry = 0.0;
      if (P.isCavity != 0u) {
        rx = 2.0 + rand01(&seed) * max(1.0, NXf - 4.0);
        ry = 2.0 + rand01(&seed) * max(1.0, NYf - 4.0);
      } else {
        rx = rand01(&seed) * 2.0;
        if (rand01(&seed) < 0.35) {
          ry = clamp(NYf * 0.5 + (rand01(&seed) - 0.5) * (NYf / 3.0), 1.0, NYf - 2.0);
        } else {
          ry = 1.0 + rand01(&seed) * max(1.0, NYf - 2.0);
        }
      }
      if (!solidAt(rx, ry)) {
        out = vec4<f32>(rx, ry, 0.0, 0.0);
        placed = true;
        break;
      }
    }
    if (!placed) {
      if (P.isCavity != 0u) {
        out = vec4<f32>(NXf * 0.5, NYf * 0.5, 0.0, 0.0);
      } else {
        out = vec4<f32>(0.5, NYf * 0.5, 0.0, 0.0);
      }
    }
  }

  particles[i] = out;
}

// ---------------------------------------------------------
// Overlay gathers: sparse samples for the Canvas2D vector and
// distribution overlays (render.js drawVectorField / drawGrid).
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn gather_vec(@builtin(global_invocation_id) g : vec3<u32>) {
  let cols = (P.NX + 7u) / 8u;
  let rows = (P.NY + 7u) / 8u;
  let k = g.x;
  if (k >= cols * rows) { return; }
  let n = (k / cols) * 8u * P.NX + (k % cols) * 8u;
  let m = mac[n];
  gatherOut[k * 4u + 0u] = m.y;
  gatherOut[k * 4u + 1u] = m.z;
  gatherOut[k * 4u + 2u] = select(0.0, 1.0, isSolid(n));
  gatherOut[k * 4u + 3u] = 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn gather_dist(@builtin(global_invocation_id) g : vec3<u32>) {
  let vecCount = ((P.NX + 7u) / 8u) * ((P.NY + 7u) / 8u);
  let cols = (P.NX + 15u) / 16u;
  let rows = (P.NY + 15u) / 16u;
  let k = g.x;
  if (k >= cols * rows) { return; }
  let n = (k / cols) * 16u * P.NX + (k % cols) * 16u;
  let base = vecCount * 4u + k * 10u;
  for (var i = 0u; i < Q; i++) { gatherOut[base + i] = fA[i * P.N + n]; }
  gatherOut[base + 9u] = select(0.0, 1.0, isSolid(n));
}
