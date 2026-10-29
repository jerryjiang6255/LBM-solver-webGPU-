// src/gpu/render.wgsl
// ============================================================
// GPU port of render.js Renderer.draw() scalar-field path:
//
//   speed : |u|                                  per node
//   qcrit : 3 passes of [1 2 1] velocity smoothing, then Q
//   solids: -1, then filled from fluid neighbours (fillSolids)
//   blur  : 2 passes of separable [1 2 1] / 4  (blurField)
//   color : 512-step cubic-Hermite colormap, solid grey for
//           geometry (painted walls keep the flow colour)
//
// The colour image (one RGBA8 per lattice node) is then drawn
// to the canvas with bilinear filtering, like drawImage() with
// image smoothing on the CPU.
// ============================================================

const NODE_SOLID   : u32 = 1u;
const NODE_PAINTED : u32 = 2u;
const COLORMAP_STEPS : u32 = 512u;

struct Params {
  NX : u32,
  NY : u32,
  N : u32,
  isCavity : u32,
  collisionModel : u32,
  lesModel : u32,
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
  displayMode : u32,
  particleDt : f32,
  particleCount : u32,
  frameSeed : u32,
  surfCount : u32,
  canvasW : f32,
  canvasH : f32,
  _pad0 : u32,
  _pad1 : u32,
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> mac : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> nodeType : array<u32>;
@group(0) @binding(3) var<storage, read_write> velA : array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> velB : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> scalar : array<f32>;
@group(0) @binding(6) var<storage, read_write> scalarTmp : array<f32>;
@group(0) @binding(7) var<storage, read> cmap : array<u32>;
@group(0) @binding(8) var<storage, read_write> colorOut : array<u32>;
@group(0) @binding(9) var<storage, read> colorIn : array<u32>;

fn solidAt(n : u32) -> bool {
  return (nodeType[n] & NODE_SOLID) != 0u;
}

fn interior(x : u32, y : u32) -> bool {
  return x > 0u && x + 1u < P.NX && y > 0u && y + 1u < P.NY;
}

// ---------------------------------------------------------
// Velocity smoothing for Q-criterion (Renderer.smoothVelocity)
// velA plays the role of uxS/uyS, velB of uxT/uyT.
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn smooth_init(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  velA[n] = mac[n].yz;
}

@compute @workgroup_size(64, 1, 1)
fn smooth_h(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  if (!interior(g.x, g.y)) {
    // The CPU scratch buffer is never written on edge nodes (stays 0).
    velB[n] = vec2<f32>(0.0);
    return;
  }
  if (solidAt(n)) {
    velB[n] = velA[n];
    return;
  }
  velB[n] = (velA[n - 1u] + 2.0 * velA[n] + velA[n + 1u]) * 0.25;
}

@compute @workgroup_size(64, 1, 1)
fn smooth_v(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  if (!interior(g.x, g.y)) { return; }
  let n = g.y * P.NX + g.x;
  if (solidAt(n)) {
    velA[n] = velB[n];
    return;
  }
  velA[n] = (velB[n - P.NX] + 2.0 * velB[n] + velB[n + P.NX]) * 0.25;
}

// ---------------------------------------------------------
// Scalar field
// ---------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn scalar_main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;

  if (solidAt(n)) {
    scalar[n] = -1.0;
    return;
  }

  if (P.displayMode == 1u) {
    if (interior(g.x, g.y)) {
      let dudx = (velA[n + 1u].x - velA[n - 1u].x) * 0.5;
      let dudy = (velA[n + P.NX].x - velA[n - P.NX].x) * 0.5;
      let dvdx = (velA[n + 1u].y - velA[n - 1u].y) * 0.5;
      let dvdy = (velA[n + P.NX].y - velA[n - P.NX].y) * 0.5;
      let omega = dvdx - dudy;
      let strain2 = dudx * dudx + dvdy * dvdy + 0.5 * (dudy + dvdx) * (dudy + dvdx);
      scalar[n] = 0.5 * (omega * omega - 2.0 * strain2);
    } else {
      scalar[n] = 0.0;
    }
    return;
  }

  let m = mac[n];
  scalar[n] = sqrt(m.y * m.y + m.z * m.z);
}

// fillSolids: interior solid nodes take the mean of fluid 8-neighbours.
// Only fluid values are read and only solid values are written, so this
// is race-free in place, exactly as on the CPU.
@compute @workgroup_size(64, 1, 1)
fn fill_main(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  if (!interior(g.x, g.y)) { return; }
  let n = g.y * P.NX + g.x;
  if (!solidAt(n)) { return; }

  let NX = P.NX;
  var nb = array<u32, 8>(
    n - 1u, n + 1u, n - NX, n + NX,
    n - NX - 1u, n - NX + 1u, n + NX - 1u, n + NX + 1u
  );
  var sum = 0.0;
  var cnt = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let m = nb[k];
    if (!solidAt(m)) {
      sum += scalar[m];
      cnt += 1.0;
    }
  }
  scalar[n] = select(0.0, sum / cnt, cnt > 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn blur_h(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  if (g.x == 0u || g.x + 1u == P.NX) {
    scalarTmp[n] = scalar[n];
    return;
  }
  scalarTmp[n] = (scalar[n - 1u] + 2.0 * scalar[n] + scalar[n + 1u]) * 0.25;
}

@compute @workgroup_size(64, 1, 1)
fn blur_v(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  if (g.y == 0u || g.y + 1u == P.NY) {
    scalar[n] = scalarTmp[n];
    return;
  }
  scalar[n] = (scalarTmp[n - P.NX] + 2.0 * scalarTmp[n] + scalarTmp[n + P.NX]) * 0.25;
}

@compute @workgroup_size(64, 1, 1)
fn colorize(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= P.NX || g.y >= P.NY) { return; }
  let n = g.y * P.NX + g.x;
  let nt = nodeType[n];

  if ((nt & NODE_SOLID) != 0u && (nt & NODE_PAINTED) == 0u) {
    colorOut[n] = 40u | (40u << 8u) | (45u << 16u) | (255u << 24u);
    return;
  }

  let s = scalar[n];
  var t = 0.0;
  if (P.displayMode == 1u) {
    t = s * (P.colorScale * 10000.0);
    t = (clamp(t, -1.0, 1.0) + 1.0) * 0.5;
  } else {
    t = clamp(s * (1.0 / P.colorScale), 0.0, 1.0);
  }
  if (!(t >= 0.0)) { t = 0.0; }   // NaN guard (JS "| 0" maps NaN to 0)

  let step = min(u32(t * f32(COLORMAP_STEPS - 1u)), COLORMAP_STEPS - 1u);
  colorOut[n] = cmap[step];
}

// ---------------------------------------------------------
// Present: full-screen triangle, bilinear sample of the node image
// ---------------------------------------------------------
struct VsOut {
  @builtin(position) pos : vec4<f32>,
};

@vertex
fn field_vs(@builtin(vertex_index) vid : u32) -> VsOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );
  var o : VsOut;
  o.pos = vec4<f32>(p[vid], 0.0, 1.0);
  return o;
}

fn unpack(c : u32) -> vec3<f32> {
  return vec3<f32>(
    f32(c & 255u),
    f32((c >> 8u) & 255u),
    f32((c >> 16u) & 255u)
  ) / 255.0;
}

fn texel(x : i32, y : i32) -> vec3<f32> {
  let xc = u32(clamp(x, 0, i32(P.NX) - 1));
  let yc = u32(clamp(y, 0, i32(P.NY) - 1));
  return unpack(colorIn[yc * P.NX + xc]);
}

@fragment
fn field_fs(in : VsOut) -> @location(0) vec4<f32> {
  // Pixel centre -> source texel space (texel centres at k + 0.5).
  let sx = in.pos.x * f32(P.NX) / P.canvasW - 0.5;
  let sy = in.pos.y * f32(P.NY) / P.canvasH - 0.5;
  let x0 = i32(floor(sx));
  let y0 = i32(floor(sy));
  let tx = sx - f32(x0);
  let ty = sy - f32(y0);

  let c00 = texel(x0, y0);
  let c10 = texel(x0 + 1, y0);
  let c01 = texel(x0, y0 + 1);
  let c11 = texel(x0 + 1, y0 + 1);
  let rgb = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
  return vec4<f32>(rgb, 1.0);
}
