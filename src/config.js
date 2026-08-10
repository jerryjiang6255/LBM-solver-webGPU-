// src/config.js
// Central immutable config for the WebGPU LBM app.

export const GRID_PRESETS = {
  low:  { NX: 160, NY: 80 },
  mid:  { NX: 256, NY: 128 },
  high: { NX: 1024, NY: 512 },
};

export const DEFAULT_GRID_KEY = "mid";

export const FLOW_DEFAULTS = {
  u0: 0.1732,
  reynolds: 500,
  uCoflow: 0.005,

  pulsationOn: false,
  pulsationAmp: 0.25,
  pulsationFreq: 0.01,
};

export const BODY_DEFAULTS = {
  bodyType: "airfoil",
  aoaDeg: 10,
  nozzleDiameter: 16,
  cylinderDiameter: 32,
  cavityAspect: 1.0,
};

export const PHYSICS_DEFAULTS = {
  collisionModel: "trt",      // "trt" | "bgk"
  lesModel: "smagorinsky",    // "off" | "smagorinsky" | "wale"
  wallModel: true,
};

export const DISPLAY_DEFAULTS = {
  mode: "speed",              // "speed" | "qcrit"
  colorScale: 0.5,
  showVectors: false,
  showGrid: false,
  gridFine: true,
  showDist: false,
  smokeVisible: false,
};

export const RUNTIME_DEFAULTS = {
  running: true,
  stepsPerFrame: 10,
};

export const UI_LIMITS = {
  tauMinUI: 0.545,
  tauMaxUI: 5.0,
  reStep: 10,

  aoa: { min: -45, max: 45, step: 0.5 },
  u0: { min: 0.01, max: 0.23, step: 0.005 },
  reynolds: { min: 20, max: 10000, step: 10 },
  nozzleDiameter: { min: 8, max: 80, step: 2 },
  cylinderDiameter: { min: 8, max: 96, step: 2 },
  cavityAspect: { min: 0.5, max: 2.0, step: 0.05 },
  pulsationAmp: { min: 0.0, max: 0.5, step: 0.01 },
  pulsationFreq: { min: 0.001, max: 0.05, step: 0.001 },
  colorScale: { min: 0.02, max: 1.0, step: 0.005 },
  stepsPerFrame: { min: 1, max: 20, step: 1 },
};

export const GPU_TUNING = {
  workgroupX: 16,
  workgroupY: 8,

  sampleDiagnosticsEveryFrames: 1,
  maxMonitorHistory: 600,
  forceStatsWindow: 240,
  stCrossingHistory: 12,

  // Performance tuning:
  // Run GPU diagnostics + CPU readback only every N rendered frames.
  // This does not change solver physics; it only reduces measurement overhead.
  diagnosticsEveryFrames: 1,
};

export const PARTICLE_DEFAULTS = {
  countSingle: 800,
  countCompare: 600,
  pointRadius: 5,
  alphaMin: 0.16,
  alphaMax: 1.0,
  dt: 1.0,
  minSpeedSq: 1e-8,
  respawnXMax: 2.0,
  respawnAttempts: 8,
  centerBiasProb: 0.35,
  seed: 1337,
};

export const BOUNDARY_DEFAULTS = {
  spongeWidth: 28,
  spongeStrength: 0.6,
};

export const LBM_LIMITS = {
  uCap: 0.28,
  rhoMin: 0.35,
  rhoMax: 3.0,
  tauMin: 0.545,
  tauMax: 5.0,
};

export const LES_DEFAULTS = {
  cs: 0.17,
  cw: 0.325,
  delta: 1.0,
  waleEps: 1e-20,
};

export const WALL_MODEL_DEFAULTS = {
  kappa: 0.41,
  bLog: 5.2,
  aPlus: 25.0,
  wallIters: 6,
  wallTol: 1e-4,
  yMin: 0.5,
};

export const TRT_DEFAULTS = {
  lambda: 3.0 / 16.0,
};

export const APP_DEFAULTS = {
  compareMode: false,
  convergenceVisible: false,
  sandboxActive: false,
  sandboxMode: "draw",
  smokeVisible: false,
  target1Active: true,
  target2Active: true,
  lastTargetedFlow: 1,
};

export const MONITOR_MODES = {
  CONVERGENCE: "convergence",
  FORCES: "forces",
};

export const DISPLAY_MODES = {
  SPEED: "speed",
  QCRIT: "qcrit",
};

export const BODY_TYPES = {
  AIRFOIL: "airfoil",
  NACA2412: "naca2412",
  CYLINDER: "cylinder",
  SQUARE_CYLINDER: "square-cylinder",
  LID_DRIVEN_CAVITY: "lid-driven-cavity",
  NOZZLE: "nozzle",
  BLOCKS: "blocks",
  EMPTY: "empty",
};

export const COLLISION_MODELS = {
  TRT: "trt",
  BGK: "bgk",
};

export const LES_MODELS = {
  OFF: "off",
  SMAGORINSKY: "smagorinsky",
  WALE: "wale",
};

export const TARGET_MODES = {
  FLOW1: "flow1",
  FLOW2: "flow2",
  BOTH: "both",
};

export const CS_LATTICE = 1 / Math.sqrt(3);


