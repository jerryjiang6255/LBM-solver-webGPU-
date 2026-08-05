// src/uiController.js
// DOM ownership and UI event translation only.
// No GPU calls, no solver stepping, no geometry building here.

import {
  TARGET_MODES,
  BODY_TYPES,
  DISPLAY_MODES,
  MONITOR_MODES,
  COLLISION_MODELS,
  LES_MODELS,
  GRID_PRESETS,
  CS_LATTICE,
} from "./config.js";

export function createUIController(callbacks = {}) {
  const cb = {
    setCompareMode: callbacks.setCompareMode ?? (() => {}),
    setTargetSelection: callbacks.setTargetSelection ?? (() => {}),
    setBodyType: callbacks.setBodyType ?? (() => {}),
    setAoA: callbacks.setAoA ?? (() => {}),
    setNozzleDiameter: callbacks.setNozzleDiameter ?? (() => {}),
    setCylinderDiameter: callbacks.setCylinderDiameter ?? (() => {}),
    setCavityAspect: callbacks.setCavityAspect ?? (() => {}),
    setInletVelocity: callbacks.setInletVelocity ?? (() => {}),
    setReynolds: callbacks.setReynolds ?? (() => {}),
    setPulseEnabled: callbacks.setPulseEnabled ?? (() => {}),
    setPulseAmp: callbacks.setPulseAmp ?? (() => {}),
    setPulseFreq: callbacks.setPulseFreq ?? (() => {}),
    setCollisionModel: callbacks.setCollisionModel ?? (() => {}),
    setLesModel: callbacks.setLesModel ?? (() => {}),
    setWallModel: callbacks.setWallModel ?? (() => {}),
    setDisplayMode: callbacks.setDisplayMode ?? (() => {}),
    setColorScale: callbacks.setColorScale ?? (() => {}),
    setVectorsVisible: callbacks.setVectorsVisible ?? (() => {}),
    setGridVisible: callbacks.setGridVisible ?? (() => {}),
    setRuntimeRunning: callbacks.setRuntimeRunning ?? (() => {}),
    resetTarget: callbacks.resetTarget ?? (() => {}),
    clearSandbox: callbacks.clearSandbox ?? (() => {}),
    setFidelity: callbacks.setFidelity ?? (() => {}),
    setSandboxEnabled: callbacks.setSandboxEnabled ?? (() => {}),
    setSmokeVisible: callbacks.setSmokeVisible ?? (() => {}),
    setMonitorVisible: callbacks.setMonitorVisible ?? (() => {}),
    setMonitorMode: callbacks.setMonitorMode ?? (() => {}),
    setStepsPerFrame: callbacks.setStepsPerFrame ?? (() => {}),
  };

  const el = getDomRefs();
  let uiState = {
    compareMode: false,
    target1Active: true,
    target2Active: true,
    lastTargetedFlow: 1,
  };

  bindStaticEvents();

  function bindStaticEvents() {
    el.compareBtn?.addEventListener("click", () => {
      cb.setCompareMode(!uiState.compareMode);
    });

    el.target1Btn?.addEventListener("click", () => {
      if (uiState.target1Active && !uiState.target2Active) return;

      const nextTarget1 = !uiState.target1Active;
      const nextTarget2 = uiState.target2Active;

      cb.setTargetSelection({
        target1Active: nextTarget1,
        target2Active: nextTarget2,
        lastTargetedFlow:
          nextTarget1 && !nextTarget2 ? 1 :
          nextTarget1 && nextTarget2 ? 1 :
          uiState.lastTargetedFlow,
      });
    });

    el.target2Btn?.addEventListener("click", () => {
      if (uiState.target2Active && !uiState.target1Active) return;

      const nextTarget1 = uiState.target1Active;
      const nextTarget2 = !uiState.target2Active;

      cb.setTargetSelection({
        target1Active: nextTarget1,
        target2Active: nextTarget2,
        lastTargetedFlow:
          nextTarget2 && !nextTarget1 ? 2 :
          nextTarget1 && nextTarget2 ? 2 :
          uiState.lastTargetedFlow,
      });
    });

    bindBodyButtons();
    bindPhysicsButtons();
    bindDisplayButtons();
    bindRuntimeButtons();
    bindSliderEvents();
    bindMonitorButtons();
  } 

 function bindBodyButtons() {
    byId("body-airfoil")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.AIRFOIL));
    byId("body-naca2412")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.NACA2412));
    byId("body-cylinder")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.CYLINDER));
    byId("body-square-cylinder")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.SQUARE_CYLINDER));
    byId("body-lid-driven-cavity")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.LID_DRIVEN_CAVITY));
    byId("body-nozzle")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.NOZZLE));
    byId("body-blocks")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.BLOCKS));
    byId("body-empty")?.addEventListener("click", () => cb.setBodyType(activeTargetMode(), BODY_TYPES.EMPTY));
  }

  function bindPhysicsButtons() {
    byId("collision-trt")?.addEventListener("click", () => cb.setCollisionModel(activeTargetMode(), COLLISION_MODELS.TRT));
    byId("collision-bgk")?.addEventListener("click", () => cb.setCollisionModel(activeTargetMode(), COLLISION_MODELS.BGK));

    byId("les-smag")?.addEventListener("click", () => cb.setLesModel(activeTargetMode(), LES_MODELS.SMAGORINSKY));
    byId("les-wale")?.addEventListener("click", () => cb.setLesModel(activeTargetMode(), LES_MODELS.WALE));

    byId("wall-on")?.addEventListener("click", () => cb.setWallModel(activeTargetMode(), true));
    byId("wall-off")?.addEventListener("click", () => cb.setWallModel(activeTargetMode(), false));

    byId("pulse-on")?.addEventListener("click", () => cb.setPulseEnabled(activeTargetMode(), true));
    byId("pulse-off")?.addEventListener("click", () => cb.setPulseEnabled(activeTargetMode(), false));
  }

  function bindDisplayButtons() {
    byId("mode-speed")?.addEventListener("click", () => cb.setDisplayMode(DISPLAY_MODES.SPEED));
    byId("mode-qcrit")?.addEventListener("click", () => cb.setDisplayMode(DISPLAY_MODES.QCRIT));

    byId("vector-btn")?.addEventListener("click", () => {
      const next = !byId("vector-btn")?.classList.contains("active");
      cb.setVectorsVisible(next);
    });

    byId("dist-btn")?.addEventListener("click", () => {
      const next = !byId("dist-btn")?.classList.contains("active");
      cb.setGridVisible(next);
    });

    el.smokeBtn?.addEventListener("click", () => {
      const next = !el.smokeBtn.classList.contains("active");
      cb.setSmokeVisible(next);
    });
  }

  function bindRuntimeButtons() {
    byId("pause-btn")?.addEventListener("click", () => {
      const runningNow = byId("pause-btn")?.textContent !== "Resume";
      cb.setRuntimeRunning(!runningNow);
    });

    byId("reset-btn")?.addEventListener("click", () => {
      cb.resetTarget(activeTargetMode());
    });

    byId("clear-btn")?.addEventListener("click", () => {
      cb.clearSandbox(activeTargetMode());
    });

    byId("sandbox-btn")?.addEventListener("click", () => {
      const next = !byId("sandbox-btn")?.classList.contains("active");
      cb.setSandboxEnabled(next);
    });

    byId("grid-low")?.addEventListener("click", () => cb.setFidelity("low"));
    byId("grid-mid")?.addEventListener("click", () => cb.setFidelity("mid"));
    byId("grid-high")?.addEventListener("click", () => cb.setFidelity("high"));
  }

  function bindSliderEvents() {
    el.aoaSlider?.addEventListener("input", e => cb.setAoA(activeTargetMode(), parseFloat(e.target.value)));
    el.nozzleSlider?.addEventListener("input", e => cb.setNozzleDiameter(activeTargetMode(), parseInt(e.target.value, 10)));
    el.cylinderSlider?.addEventListener("input", e => cb.setCylinderDiameter(activeTargetMode(), parseInt(e.target.value, 10)));
    el.cavityAspectSlider?.addEventListener("input", e => cb.setCavityAspect(activeTargetMode(), parseFloat(e.target.value)));
    el.u0Slider?.addEventListener("input", e => cb.setInletVelocity(activeTargetMode(), parseFloat(e.target.value)));
    el.reSlider?.addEventListener("input", e => cb.setReynolds(activeTargetMode(), parseFloat(e.target.value)));
    el.pulseAmpSlider?.addEventListener("input", e => cb.setPulseAmp(activeTargetMode(), parseFloat(e.target.value)));
    el.pulseFreqSlider?.addEventListener("input", e => cb.setPulseFreq(activeTargetMode(), parseFloat(e.target.value)));
    el.scaleSlider?.addEventListener("input", e => cb.setColorScale(parseFloat(e.target.value)));
    el.stepsSlider?.addEventListener("input", e => cb.setStepsPerFrame(parseInt(e.target.value, 10)));
  }

  function bindMonitorButtons() {
    el.convergenceBtn?.addEventListener("click", () => {
      const next = !el.convergenceBtn.classList.contains("active");
      cb.setMonitorVisible(next);
    });

    byId("monitor-conv-btn-1")?.addEventListener("click", () => cb.setMonitorMode(MONITOR_MODES.CONVERGENCE));
    byId("monitor-conv-btn-2")?.addEventListener("click", () => cb.setMonitorMode(MONITOR_MODES.CONVERGENCE));
    byId("monitor-force-btn-1")?.addEventListener("click", () => cb.setMonitorMode(MONITOR_MODES.FORCES));
    byId("monitor-force-btn-2")?.addEventListener("click", () => cb.setMonitorMode(MONITOR_MODES.FORCES));
  }

 function sync(appState) {
    uiState = {
      compareMode: !!appState.compareMode,
      target1Active: !!appState.target1Active,
      target2Active: !!appState.target2Active,
      lastTargetedFlow: appState.lastTargetedFlow ?? 1,
    };

    _uiCompareMode = uiState.compareMode;
    _uiTarget1Active = uiState.target1Active;
    _uiTarget2Active = uiState.target2Active;



    const sim1 = appState.sim1 ?? null;
    const sim2 = appState.sim2 ?? null;
    const refSim = getReferenceSim(sim1, sim2);

    syncCompareUi(appState);
    syncBodyButtons(sim1, sim2);
    syncPhysicsButtons(sim1, sim2);
    syncDisplayButtons(refSim, appState.compareMode);
    syncRows(refSim);
    syncSliders(sim1, sim2, refSim);
    syncStats(appState, refSim);
    syncMonitorUi(appState);
  }

  function syncCompareUi(appState) {
    el.compareBtn?.classList.toggle("active", !!appState.compareMode);
    el.canvasWrap2 && (el.canvasWrap2.style.display = appState.compareMode ? "" : "none");
    el.flowTargetBar && (el.flowTargetBar.style.display = appState.compareMode ? "" : "none");

    el.target1Btn?.classList.toggle("active", !!appState.target1Active);
    el.target2Btn?.classList.toggle("active", !!appState.target2Active);
  }

  function syncBodyButtons(sim1, sim2) {
    clearGroup(bodyButtonIds);

    const v1 = sim1?.params?.bodyType ?? null;
    const v2 = uiState.compareMode ? (sim2?.params?.bodyType ?? null) : null;

   for (const [id, value] of Object.entries(bodyButtonMap)) {
      const btn = byId(id);
      if (!btn) continue;

      if (!uiState.compareMode) {
        if (value === v1) btn.classList.add("active");
        continue;
      }

      const m1 = value === v1;
      const m2 = value === v2;

      if (m1 || m2) btn.classList.add("active");
    }

    applyTargetClassesToButtons(bodyButtonIds);
  }

  function syncPhysicsButtons(sim1, sim2) {
    clearGroup(physicsButtonIds);

    const ref = getReferenceSim(sim1, sim2);
    if (!ref) return;

    byId(ref.physics.collisionModel === COLLISION_MODELS.BGK ? "collision-bgk" : "collision-trt")?.classList.add("active");

    if (ref.physics.lesModel === LES_MODELS.SMAGORINSKY) {
      byId("les-smag")?.classList.add("active");
    } else if (ref.physics.lesModel === LES_MODELS.WALE) {
      byId("les-wale")?.classList.add("active");
    }

   byId(ref.physics.wallModel ? "wall-on" : "wall-off")?.classList.add("active");
    byId(ref.flow.pulsationOn ? "pulse-on" : "pulse-off")?.classList.add("active");

    applyTargetClassesToButtons(physicsButtonIds);
  }

  function syncDisplayButtons(refSim, compareMode) {
    if (!refSim) return;

    clearGroup(["mode-speed", "mode-qcrit"]);
    byId(refSim.display.mode === DISPLAY_MODES.QCRIT ? "mode-qcrit" : "mode-speed")?.classList.add("active");

    byId("vector-btn")?.classList.toggle("active", !!refSim.display.showVectors);
    byId("dist-btn")?.classList.toggle("active", !!refSim.display.showGrid);
    el.smokeBtn?.classList.toggle("active", !!refSim.display.smokeVisible);

    for (const id of ["vector-btn", "dist-btn", "mode-speed", "mode-qcrit"]) {
      byId(id)?.classList.toggle("global-compare", !!compareMode);
    }
  }

 function syncRows(refSim) {
    const type = refSim?.params?.bodyType;

    const aoaBodies = new Set([
      BODY_TYPES.AIRFOIL,
      BODY_TYPES.NACA2412,
      BODY_TYPES.SQUARE_CYLINDER,
    ]);

    show(el.aoaRow, aoaBodies.has(type));
    show(el.nozzleRow, type === BODY_TYPES.NOZZLE);
    show(el.cylinderRow, type === BODY_TYPES.CYLINDER);
    show(el.cavityAspectRow, type === BODY_TYPES.LID_DRIVEN_CAVITY);
    show(el.pulseAmpRow, !!refSim?.flow?.pulsationOn);
    show(el.pulseFreqRow, !!refSim?.flow?.pulsationOn);

    if (el.u0Label) {
      el.u0Label.textContent =
        type === BODY_TYPES.LID_DRIVEN_CAVITY
          ? "Lid velocity"
          : "Inlet velocity";
    }
  }

  function syncSliders(sim1, sim2, refSim) {
    if (!refSim) return;

    if (el.aoaSlider) {
      el.aoaSlider.value = String(refSim.params.aoaDeg);
      setText(el.aoaValue, `${refSim.params.aoaDeg.toFixed(1)}°`);
    }

    if (el.nozzleSlider) {
      el.nozzleSlider.value = String(refSim.params.nozzleDiameter);
      setText(el.nozzleValue, String(refSim.params.nozzleDiameter));
    }

    if (el.cylinderSlider) {
      el.cylinderSlider.value = String(refSim.params.cylinderDiameter);
      setText(el.cylinderValue, String(refSim.params.cylinderDiameter));
    }

    if (el.cavityAspectSlider) {
      el.cavityAspectSlider.value = String(refSim.params.cavityAspect);
      setText(el.cavityAspectValue, Number(refSim.params.cavityAspect).toFixed(2));
    }

    if (el.u0Slider) {
      el.u0Slider.value = String(refSim.params.u0);
      setText(el.u0Value, `Ma ${(refSim.params.u0 / CS_LATTICE).toFixed(2)}`);
    }

    if (el.reSlider) {
      el.reSlider.value = String(refSim.params.reynolds);
      setText(el.reValue, refSim.params.reynolds.toFixed(0));
    }

    if (el.pulseAmpSlider) {
      el.pulseAmpSlider.value = String(refSim.flow.pulsationAmp);
      setText(el.pulseAmpValue, refSim.flow.pulsationAmp.toFixed(2));
    }

    if (el.pulseFreqSlider) {
      el.pulseFreqSlider.value = String(refSim.flow.pulsationFreq);
      setText(el.pulseFreqValue, refSim.flow.pulsationFreq.toFixed(3));
    }

    if (el.scaleSlider) {
      el.scaleSlider.value = String(refSim.display.colorScale);
      setText(el.scaleValue, refSim.display.colorScale.toFixed(3));
    }

    if (el.stepsSlider) {
      el.stepsSlider.value = String(refSim.runtime.stepsPerFrame);
      setText(el.stepsValue, String(refSim.runtime.stepsPerFrame));
    }

    applyTargetClassesToSliders([
      el.aoaSlider,
      el.nozzleSlider,
      el.cylinderSlider,
      el.cavityAspectSlider,
      el.u0Slider,
      el.reSlider,
      el.pulseAmpSlider,
      el.pulseFreqSlider,
      el.stepsSlider,
    ]);

    syncGridStats(refSim);
  }

  function syncGridStats(refSim) {
    const NX = refSim?.dims?.NX ?? GRID_PRESETS.mid.NX;
    const NY = refSim?.dims?.NY ?? GRID_PRESETS.mid.NY;

    setText(el.gridNx, String(NX));
    setText(el.gridNy, String(NY));
    setText(el.nodesValue, (NX * NY).toLocaleString());

    clearGroup(["grid-low", "grid-mid", "grid-high"]);
    byId(`grid-${refSim.dims.gridKey}`)?.classList.add("active");
  } 

 function syncStats(appState, refSim) {
    setText(el.fpsValue, appState.fps != null ? appState.fps.toFixed(0) : "--");

    const f1 = appState.sim1?.runtime?.frameCount ?? 0;
    const f2 = appState.sim2?.runtime?.frameCount ?? 0;
    setText(el.frameValue, appState.compareMode ? `${f1} / ${f2}` : `${f1}`);

    const anyRunning =
      (appState.sim1?.runtime?.running ?? false) ||
      (appState.sim2?.runtime?.running ?? false);

    if (el.pauseBtn) {
      el.pauseBtn.textContent = anyRunning ? "Pause" : "Resume";
      el.pauseBtn.classList.toggle("active", !anyRunning);
    }

    if (el.sandboxBtn) {
      el.sandboxBtn.classList.toggle("active", !!appState.sandboxActive);
    }
  }

  function syncMonitorUi(appState) {
    el.convergenceBtn?.classList.toggle("active", !!appState.convergenceVisible);

    const mode = appState.monitorMode ?? MONITOR_MODES.CONVERGENCE;

    for (const id of ["monitor-conv-btn-1", "monitor-conv-btn-2"]) {
      byId(id)?.classList.toggle("active", mode === MONITOR_MODES.CONVERGENCE);
    }

    for (const id of ["monitor-force-btn-1", "monitor-force-btn-2"]) {
      byId(id)?.classList.toggle("active", mode === MONITOR_MODES.FORCES);
    }

    show(byId("monitor-mode-bar-1"), !!appState.convergenceVisible && !!appState.sim1);
    show(byId("monitor-mode-bar-2"), !!appState.convergenceVisible && !!appState.compareMode && !!appState.sim2);
  }

  function activeTargetMode() {
    if (!uiState.compareMode) return TARGET_MODES.FLOW1;
    if (uiState.target1Active && uiState.target2Active) return TARGET_MODES.BOTH;
    if (uiState.target1Active) return TARGET_MODES.FLOW1;
    if (uiState.target2Active) return TARGET_MODES.FLOW2;
    return TARGET_MODES.FLOW1;
  }

  function getReferenceSim(sim1, sim2) {
    const mode = activeTargetMode();

    if (!uiState.compareMode) return sim1;
    if (mode === TARGET_MODES.FLOW2) return sim2 ?? sim1;
    if (mode === TARGET_MODES.BOTH) {
      return uiState.lastTargetedFlow === 2 ? (sim2 ?? sim1) : sim1;
    }
    return sim1;
  }

  return {
    el,
    sync,
    activeTargetMode,
  };
} 

// ---------------------------------------------------------
// DOM refs
// ---------------------------------------------------------
function getDomRefs() {
  return {
    compareBtn: byId("compare-btn"),
    convergenceBtn: byId("convergence-btn"),
    sandboxBtn: byId("sandbox-btn"),
    smokeBtn: byId("smoke-btn"),
    pauseBtn: byId("pause-btn"),

    flowTargetBar: byId("flow-target-bar"),
    target1Btn: byId("flow-label-1"),
    target2Btn: byId("flow-label-2"),

    canvasWrap2: byId("canvas-wrap-2"),

    aoaRow: byId("aoa-row"),
    nozzleRow: byId("nozzle-row"),
    cylinderRow: byId("cylinder-row"),
    cavityAspectRow: byId("cavity-aspect-row"),
    pulseAmpRow: byId("pulse-amp-row"),
    pulseFreqRow: byId("pulse-freq-row"),

    aoaSlider: byId("aoa-slider"),
    nozzleSlider: byId("nozzle-slider"),
    cylinderSlider: byId("cylinder-slider"),
    cavityAspectSlider: byId("cavity-aspect-slider"),
    u0Slider: byId("u0-slider"),
    reSlider: byId("re-slider"),
    pulseAmpSlider: byId("pulse-amp-slider"),
    pulseFreqSlider: byId("pulse-freq-slider"),
    scaleSlider: byId("scale-slider"),
    stepsSlider: byId("steps-slider"),

    aoaValue: byId("aoa-value"),
    nozzleValue: byId("nozzle-value"),
    cylinderValue: byId("cylinder-value"),
    cavityAspectValue: byId("cavity-aspect-value"),
    u0Value: byId("u0-value"),
    reValue: byId("re-value"),
    pulseAmpValue: byId("pulse-amp-value"),
    pulseFreqValue: byId("pulse-freq-value"),
    scaleValue: byId("scale-value"),
    stepsValue: byId("steps-value"),

    u0Label: byId("u0-label"),

    fpsValue: byId("fps-value"),
    frameValue: byId("frame-value"),

    gridNx: byId("grid-nx"),
    gridNy: byId("grid-ny"),
    nodesValue: byId("nodes-value"),
  };
}

// ---------------------------------------------------------
// Static maps
// ---------------------------------------------------------
const bodyButtonMap = {
  "body-airfoil": BODY_TYPES.AIRFOIL,
  "body-naca2412": BODY_TYPES.NACA2412,
  "body-cylinder": BODY_TYPES.CYLINDER,
  "body-square-cylinder": BODY_TYPES.SQUARE_CYLINDER,
  "body-lid-driven-cavity": BODY_TYPES.LID_DRIVEN_CAVITY,
  "body-nozzle": BODY_TYPES.NOZZLE,
  "body-blocks": BODY_TYPES.BLOCKS,
  "body-empty": BODY_TYPES.EMPTY,
};

const bodyButtonIds = Object.keys(bodyButtonMap);

const physicsButtonIds = [
  "collision-trt",
  "collision-bgk",
  "les-smag",
  "les-wale",
  "wall-on",
  "wall-off",
  "pulse-on",
  "pulse-off",
];

// ---------------------------------------------------------
// Compare target class helpers
// ---------------------------------------------------------
// function getTargetClass(compareMode, target1Active, target2Active) {
//   if (!compareMode) return "";
//   if (target1Active && target2Active) return "target-both";
//   if (target1Active) return "target-flow1";
//   if (target2Active) return "target-flow2";
//   return "";
// }

// function getSliderClass(compareMode, target1Active, target2Active) {
//   if (!compareMode) return "";
//   if (target1Active && target2Active) return "slider-both";
//   if (target1Active) return "slider-flow1";
//   if (target2Active) return "slider-flow2";
//   return "";
// }

// function applyTargetClassesToButtons(ids) {
//   const cls = getTargetClass(
//     currentUiCompareMode(),
//     currentUiTarget1Active(),
//     currentUiTarget2Active()
//   );

//   for (const id of ids) {
//     const node = byId(id);
//     if (!node) continue;
//     node.classList.remove("target-flow1", "target-flow2", "target-both");
//     if (cls) node.classList.add(cls);
//   }
// }

// function applyTargetClassesToSliders(nodes) {
//   const cls = getSliderClass(
//     currentUiCompareMode(),
//     currentUiTarget1Active(),
//     currentUiTarget2Active()
//   );

//   for (const node of nodes) {
//     if (!node) continue;
//     node.classList.remove("slider-flow1", "slider-flow2", "slider-both");
//     if (cls) node.classList.add(cls);
//   }
// }

// let _uiCompareMode = false;
// let _uiTarget1Active = true;
// let _uiTarget2Active = true;

// function currentUiCompareMode() {
//   return _uiCompareMode;
// }
// function currentUiTarget1Active() {
//   return _uiTarget1Active;
// }
// function currentUiTarget2Active() {
//   return _uiTarget2Active;
// }
// ---------------------------------------------------------
// Compare target class helpers
// ---------------------------------------------------------
let _uiCompareMode = false;
let _uiTarget1Active = true;
let _uiTarget2Active = true;

function getTargetClass(compareMode, target1Active, target2Active) {
  if (!compareMode) return "";
  if (target1Active && target2Active) return "target-both";
  if (target1Active) return "target-flow1";
  if (target2Active) return "target-flow2";
  return "";
}

function getSliderClass(compareMode, target1Active, target2Active) {
  if (!compareMode) return "";
  if (target1Active && target2Active) return "slider-both";
  if (target1Active) return "slider-flow1";
  if (target2Active) return "slider-flow2";
  return "";
}

function applyTargetClassesToButtons(ids) {
  const cls = getTargetClass(_uiCompareMode, _uiTarget1Active, _uiTarget2Active);

  for (const id of ids) {
    const node = byId(id);
    if (!node) continue;
    node.classList.remove("target-flow1", "target-flow2", "target-both");
    if (cls) node.classList.add(cls);
  }
}

function applyTargetClassesToSliders(nodes) {
  const cls = getSliderClass(_uiCompareMode, _uiTarget1Active, _uiTarget2Active);

  for (const node of nodes) {
    if (!node) continue;
    node.classList.remove("slider-flow1", "slider-flow2", "slider-both");
    if (cls) node.classList.add(cls);
  }
}


// ---------------------------------------------------------
// Small helpers
// ---------------------------------------------------------
function byId(id) {
  return document.getElementById(id);
}

function show(node, visible) {
  if (!node) return;
  node.style.display = visible ? "" : "none";
}

function setText(node, text) {
  if (!node) return;
  node.textContent = text;
}

function clearGroup(ids) {
  for (const id of ids) {
    byId(id)?.classList.remove("active");
  }
}

// Keep compare target class helper state updated
const _originalCreateUIController = createUIController;
createUIController = function patchedCreateUIController(callbacks = {}) {
  const controller = _originalCreateUIController(callbacks);
  const originalSync = controller.sync;

  controller.sync = function patchedSync(appState) {
    _uiCompareMode = !!appState.compareMode;
    _uiTarget1Active = !!appState.target1Active;
    _uiTarget2Active = !!appState.target2Active;
    originalSync(appState);
  };

  return controller;
}; 
