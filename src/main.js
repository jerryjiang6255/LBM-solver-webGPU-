
// src/main.js
// ============================================================
// WebGPU LBM sandbox — application layer.
//
// This is a port of the CPU solver's index.html script (LBM-solver-
// V7) with the CPU solver calls replaced by their GPU counterparts:
//   collideFused / stream / applyBoundaryConditions -> gpu/solver.wgsl
//   computeSurfaceForces + convergence sums         -> GPU reductions
//   Renderer (scalar field)                         -> gpu/fieldRenderer.js
//   Renderer (vectors / grid / distribution)        -> overlayRenderer.js
// Everything else (UI behaviour, defaults, compare / fidelity logic,
// geometry, sandbox drawing, monitor) follows the CPU code.
// ============================================================

    import {
      createSimInstance,
      buildInitialField,
      setReynolds,
      setCharacteristicLength,
      getInletVelocity
    } from "./simInstance.js";
    import {
      applyBoundaryConditions,
      setNozzleInlet,
      clearNozzleInlet
    } from "./boundary.js";
    import { BodyRenderer } from "./renderBody.js";
    import { createUserDraw } from "./userDraw.js";
    import { createConvergenceMonitor } from "./convergenceMonitor.js";
    import { createParticles } from "./particles.js";
    import { OverlayRenderer } from "./overlayRenderer.js";
    import { initGpu, MAX_STEPS_PER_FRAME } from "./gpu/gpuContext.js";
    import { GpuSim } from "./gpu/gpuSim.js";
    import { FieldRenderer } from "./gpu/fieldRenderer.js";
    import {
      buildAirfoil,
      buildNACA2412,
      buildEmpty,
      defaultBlockMatrix,
      buildNozzle,
      buildCylinder,
      buildSquareCylinder,
      buildLidDrivenCavity
    } from "./geometry.js";

    const gpu = await initGpu();

    // =====================================================
    // Constants / global state
    // =====================================================
    // The GPU version runs every fidelity level at higher resolution
    // than the CPU version (CPU: 160x80 / 256x128 / 512x256).
    const GRID_PRESETS = {
      low:  { NX: 256,  NY: 128 },
      mid:  { NX: 512,  NY: 256 },
      high: { NX: 1024, NY: 512 },
    };

    const U_COFLOW = 0.005;
    const CS_LATTICE = 1 / Math.sqrt(3);

    let currentGridKey = "mid";
    let sandboxActive = false;
    let sandboxMode = "draw";
    let smokeVisible = false;


    let sim1 = null;
    let sim2 = null;

    let compareMode = false;
    let target1Active = true;
    let target2Active = true;
    let convergenceVisible = false;
    let lastTargetedFlow = 1;

    const TAU_MIN_UI = 0.545;
    const TAU_MAX_UI = 5.0;
    const RE_STEP = 10;


    // =====================================================
    // Cached DOM refs
    // =====================================================
    const compareBtn = document.getElementById("compare-btn");
    const convergenceBtn = document.getElementById("convergence-btn");
    const canvasWrap2 = document.getElementById("canvas-wrap-2");
    const flowTargetBar = document.getElementById("flow-target-bar");
    const target1Btn = document.getElementById("flow-label-1");
    const target2Btn = document.getElementById("flow-label-2");
    const smokeBtn = document.getElementById("smoke-btn");


    const monitorBar1 = document.getElementById("monitor-mode-bar-1");
    const monitorBar2 = document.getElementById("monitor-mode-bar-2");
    const monitorConvBtn1 = document.getElementById("monitor-conv-btn-1");
    const monitorForceBtn1 = document.getElementById("monitor-force-btn-1");
    const monitorConvBtn2 = document.getElementById("monitor-conv-btn-2");
    const monitorForceBtn2 = document.getElementById("monitor-force-btn-2");

    const aoaRow = document.getElementById("aoa-row");
    const nozzleRow = document.getElementById("nozzle-row");
    const cavityRow = document.getElementById("cavity-row");
    const nozzleSlider = document.getElementById("nozzle-slider");
    const nozzleOutput = document.getElementById("nozzle-value");
    const cylinderRow = document.getElementById("cylinder-row");
    const cylinderSlider = document.getElementById("cylinder-slider");
    const cylinderOutput = document.getElementById("cylinder-value");


    const pulseAmpRow = document.getElementById("pulse-amp-row");
    const pulseFreqRow = document.getElementById("pulse-freq-row");
    const pulseAmpSlider = document.getElementById("pulse-amp-slider");
    const pulseFreqSlider = document.getElementById("pulse-freq-slider");
    const pulseAmpValue = document.getElementById("pulse-amp-value");
    const pulseFreqValue = document.getElementById("pulse-freq-value");

    const sandboxBtn = document.getElementById("sandbox-btn");
    const clearBtn = document.getElementById("clear-btn");
    const pauseBtn = document.getElementById("pause-btn");
    const tooltip = document.getElementById("tooltip");

    const cavityAspectRow = document.getElementById("cavity-aspect-row");
    const cavityAspectSlider = document.getElementById("cavity-aspect-slider");
    const cavityAspectValue = document.getElementById("cavity-aspect-value");


    // =====================================================
    // UI groups / mappings
    // =====================================================
    const gridGroup = ["grid-low", "grid-mid", "grid-high"];
    const modeGroup = ["mode-speed", "mode-qcrit"];

    const bodyGroup = [
      "body-airfoil",
      "body-naca2412",
      "body-cylinder",
      "body-square-cylinder",
      "body-lid-driven-cavity",
      "body-nozzle",
      "body-blocks",
      "body-empty"
    ]; 
   const collisionGroup = ["collision-trt", "collision-bgk"];
    const lesGroup = ["les-smag", "les-wale"];
    const wallGroup = ["wall-on", "wall-off"];
    const pulseGroup = ["pulse-on", "pulse-off"];

    const COMPARE_GROUP_CONFIG = {
      body: {
        ids: bodyGroup,
        getValue(sim) {
          return sim?.params?.bodyType ?? null;
        },
        idToValue: {
          "body-airfoil": "airfoil",
          "body-naca2412": "naca2412",
          "body-cylinder": "cylinder",
          "body-square-cylinder": "square-cylinder",
          "body-lid-driven-cavity": "lid-driven-cavity",
          "body-nozzle": "nozzle",
          "body-blocks": "blocks",
          "body-empty": "empty",
        },
      },

      collision: {
        ids: collisionGroup,
        getValue(sim) {
          return sim?.physics?.collisionModel ?? null;
        },
        idToValue: {
          "collision-trt": "trt",
          "collision-bgk": "bgk",
        },
      },

      les: {
        ids: lesGroup,
        getValue(sim) {
          return sim?.physics?.lesModel ?? "off";
        },
        idToValue: {
          "les-smag": "smagorinsky",
          "les-wale": "wale",
        },
      },


      wall: {
        ids: wallGroup,
        getValue(sim) {
          return sim?.physics?.wallModel ? "on" : "off";
        },
        idToValue: {
          "wall-on": "on",
          "wall-off": "off",
        },
      },

      pulse: {
        ids: pulseGroup,
        getValue(sim) {
          return sim?.flow?.pulsationOn ? "on" : "off";
        },
        idToValue: {
          "pulse-on": "on",
          "pulse-off": "off",
        },
      },
    };

    const TARGET_CONTROL_SLIDERS = [
      "aoa-slider",
      "nozzle-slider",
      "cylinder-slider",
      "cavity-aspect-slider",
      "u0-slider",
      "pulse-amp-slider",
      "pulse-freq-slider",
      "re-slider"
    ];



    const DISPLAY_GLOBAL_BUTTONS = [
      "vector-btn",
      "dist-btn",
      "mode-speed",
      "mode-qcrit"
    ];

   // =====================================================
    // UI state helpers
    // =====================================================
    function getPhysicsFromUI() {
      let lesModel = "off";

      if (document.getElementById("les-wale").classList.contains("active")) {
        lesModel = "wale";
      } else if (document.getElementById("les-smag").classList.contains("active")) {
        lesModel = "smagorinsky";
      }

      return {
        collisionModel: document.getElementById("collision-bgk").classList.contains("active")
          ? "bgk"
          : "trt",
        lesModel,
        wallModel: !document.getElementById("wall-off").classList.contains("active"),
      };
    }


    function getFlowFromUI() {
      return {
        pulsationOn: document.getElementById("pulse-on").classList.contains("active"),
        pulsationAmp: parseFloat(pulseAmpSlider.value),
        pulsationFreq: parseFloat(pulseFreqSlider.value),
      };
    }

    function currentDims() {
      return GRID_PRESETS[currentGridKey];
    }

    function getCharacteristicLengthForBody(simLike) {
      const { NX, NY } = currentDims();
      const bodyType = simLike?.params?.bodyType ?? "airfoil";

      if (bodyType === "airfoil" || bodyType === "naca2412") {
        return Math.floor(NX / 3);
      }

      if (bodyType === "cylinder") {
        const diameter = simLike?.params?.cylinderDiameter ?? parseInt(document.getElementById("cylinder-slider")?.value || "32", 10);
        return diameter;
      }

      if (bodyType === "square-cylinder") {
        return Math.max(8, Math.floor(NY / 4));
      }

    if (bodyType === "lid-driven-cavity") {
        const wallThickness = 1;
        const cavityAspect = simLike?.params?.cavityAspect ?? parseFloat(cavityAspectSlider.value);

        const outerLeft   = wallThickness;
        const outerRight  = NX - 1 - wallThickness;
        const outerBottom = wallThickness;
        const outerTop    = NY - 1 - wallThickness;

        const cavityHeight = (outerTop - outerBottom) + 1 - 2 * wallThickness;
        const cavityWidth = Math.max(4, Math.round(cavityHeight * cavityAspect));

        const midX = 0.5 * (outerLeft + outerRight);
        const cavityLeftInner  = Math.round(midX - cavityWidth / 2);
        const cavityRightInner = Math.round(midX + cavityWidth / 2);

        return (cavityRightInner - cavityLeftInner) - 1;
      }

      if (bodyType === "nozzle") {
        return simLike?.params?.nozzleDiameter ?? parseInt(nozzleSlider.value, 10);
      }

      if (bodyType === "blocks") {
        return 16;
      }

      return Math.max(4, Math.floor(NY / 8)) * 2;
    }

    function computeReBounds(u0, L) {
      const nuMin = (TAU_MIN_UI - 0.5) / 3.0;
      const nuMax = (TAU_MAX_UI - 0.5) / 3.0;

      const reMaxRaw = (u0 * L) / nuMin;
      const reMinRaw = (u0 * L) / nuMax;

      const reMin = Math.max(RE_STEP, Math.ceil(reMinRaw / RE_STEP) * RE_STEP);
      const reMax = Math.max(reMin, Math.floor(reMaxRaw / RE_STEP) * RE_STEP);

      return { reMin, reMax };
    }

    function activeReferenceSim() {
      const mode = activeTargetMode();
      let refSim = sim1;
      if (compareMode && mode === "flow2" && sim2) refSim = sim2;
      if (compareMode && mode === "both" && lastTargetedFlow === 2 && sim2) refSim = sim2;
      return refSim;
    }

    function updateReSliderBounds() {
      const reSlider = document.getElementById("re-slider");
      const refSim = activeReferenceSim();

      const u0 = refSim?.params?.u0 ?? parseFloat(document.getElementById("u0-slider").value);
      const L = getCharacteristicLengthForBody(refSim);

      const { reMin, reMax } = computeReBounds(u0, L);

      reSlider.min = String(reMin);
      reSlider.max = String(reMax);
      reSlider.step = String(RE_STEP);

      let current = parseFloat(reSlider.value);
      if (!Number.isFinite(current)) current = reMin;
      current = Math.min(reMax, Math.max(reMin, current));
      current = Math.round(current / RE_STEP) * RE_STEP;
      current = Math.min(reMax, Math.max(reMin, current));

      reSlider.value = String(current);
      document.getElementById("re-value").textContent = current.toFixed(0);

      targetedSims().forEach(sim => {
        sim.params.reynolds = current;
        setReynolds(sim, current);
      });
    }


    function updateGridStats() {
      const { NX, NY } = currentDims();
      document.getElementById("grid-nx").textContent = NX;
      document.getElementById("grid-ny").textContent = NY;
      document.getElementById("nodes-value").textContent = (NX * NY).toLocaleString();
    }

    function dropFidelity(currentKey) {
      if (currentKey === "high") return "mid";
      return "low";
    }

    function raiseFidelity(currentKey) {
      if (currentKey === "low") return "mid";
      return "high";
    }

    function setActiveBtn(ids, activeId) {
      ids.forEach(id => document.getElementById(id).classList.remove("active"));
      document.getElementById(activeId).classList.add("active");
    }

    function activeTargetMode() {
      if (!compareMode) return "flow1";
      if (target1Active && target2Active) return "both";
      if (target1Active) return "flow1";
      if (target2Active) return "flow2";
      return "flow1";
    }

    function targetedSims() {
      if (!compareMode) return sim1 ? [sim1] : [];
      const out = [];
      if (target1Active && sim1) out.push(sim1);
      if (target2Active && sim2) out.push(sim2);
      return out;
    }

    function forAllSims(fn) {
      if (sim1 && sim1.solid) fn(sim1);
      if (compareMode && sim2 && sim2.solid) fn(sim2);
    }

    function updateFlowLabels() {
      flowTargetBar.style.display = compareMode ? "" : "none";
      target1Btn.classList.toggle("active", target1Active);
      target2Btn.classList.toggle("active", target2Active);
    }

    function clearCompareVisualState(el) {
      el.classList.remove(
        "active",
        "target-flow1",
        "target-flow2",
        "target-both",
        "target-both-same",
        "target-different",
        "global-compare",
        "slider-flow1",
        "slider-flow2",
        "slider-both",
        "slider-both-same",
        "slider-both-diff"
      );
    }

    function groupValuesEqual(v1, v2) {
      return v1 === v2;
    }
    function updateVelocityLabel() {
      const label = document.getElementById("u0-label");
      if (!label) return;

      const mode = activeTargetMode();
      let refSim = sim1;
      if (compareMode && mode === "flow2" && sim2) refSim = sim2;
      if (compareMode && mode === "both" && lastTargetedFlow === 2 && sim2) refSim = sim2;

      label.textContent =
        refSim?.params?.bodyType === "lid-driven-cavity"
          ? "Lid velocity"
          : "Inlet velocity";
    }

    function renderCompareGroup(groupConfig) {
      const { ids, getValue, idToValue } = groupConfig;

      const v1 = sim1 ? getValue(sim1) : null;
      const v2 = compareMode && sim2 ? getValue(sim2) : null;

      ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;

        clearCompareVisualState(el);

        const thisValue = idToValue[id];

        if (!compareMode || !sim2) {
          if (thisValue === v1) {
            el.classList.add("active", "target-flow1");
          }
          return;
        }

        const match1 = thisValue === v1;
        const match2 = thisValue === v2;

        if (match1 && match2) {
          el.classList.add("active", "target-both", "target-both-same");
        } else if (match1) {
          el.classList.add("active", "target-flow1");
        } else if (match2) {
          el.classList.add("active", "target-flow2");
        }
      });
    }



    function renderAllCompareGroups() {
      renderCompareGroup(COMPARE_GROUP_CONFIG.body);
      renderCompareGroup(COMPARE_GROUP_CONFIG.collision);
      renderCompareGroup(COMPARE_GROUP_CONFIG.les);
      renderCompareGroup(COMPARE_GROUP_CONFIG.wall);
      renderCompareGroup(COMPARE_GROUP_CONFIG.pulse);
    }

    function renderDisplayButtons() {
      DISPLAY_GLOBAL_BUTTONS.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.remove("global-compare");
        if (compareMode) el.classList.add("global-compare");
      });
    }

    function getSliderState(id) {
      switch (id) {
        case "aoa-slider":
          return {
            getValue(sim) { return sim?.params?.aoaDeg ?? 0; },
            setValueText(v) { document.getElementById("aoa-value").textContent = `${v.toFixed(1)}°`; },
          };

        case "nozzle-slider":
          return {
            getValue(sim) { return sim?.params?.nozzleDiameter ?? 16; },
            setValueText(v) { nozzleOutput.textContent = `${Math.round(v)}`; },
          };
        
        case "cavity-aspect-slider":
          return {
            getValue(sim) { return sim?.params?.cavityAspect ?? 1.0; },
            setValueText(v) { cavityAspectValue.textContent = Number(v).toFixed(2); },
          };

        case "u0-slider":
          return {
            getValue(sim) { return sim?.params?.u0 ?? 0.1732; },
            setValueText(v) {
              document.getElementById("u0-value").textContent =
                `Ma ${(v / CS_LATTICE).toFixed(2)}`;
            },
          };

        case "cylinder-slider":
          return {
            getValue(sim) { return sim?.params?.cylinderDiameter ?? 32; },
            setValueText(v) { cylinderOutput.textContent = `${Math.round(v)}`; },
          };

        case "pulse-amp-slider":
          return {
            getValue(sim) { return sim?.flow?.pulsationAmp ?? 0.25; },
            setValueText(v) { pulseAmpValue.textContent = v.toFixed(2); },
          };

        case "pulse-freq-slider":
          return {
            getValue(sim) { return sim?.flow?.pulsationFreq ?? 0.010; },
            setValueText(v) { pulseFreqValue.textContent = v.toFixed(3); },
          };

        case "re-slider":
          return {
            getValue(sim) { return sim?.params?.reynolds ?? 5000; },
            setValueText(v) { document.getElementById("re-value").textContent = v.toFixed(0); },
          };

        default:
          return null;
      }
    }

    function renderSingleSlider(id) {
      const slider = document.getElementById(id);
      if (!slider) return;

      clearCompareVisualState(slider);

      const cfg = getSliderState(id);
      if (!cfg) return;

      const mode = activeTargetMode();

      const v1 = sim1 ? cfg.getValue(sim1) : parseFloat(slider.value);
      const v2 = compareMode && sim2 ? cfg.getValue(sim2) : v1;
      const same = groupValuesEqual(v1, v2);

      if (!compareMode || !sim2) {
        slider.classList.add("slider-flow1");
        if (document.activeElement !== slider) {
          slider.value = v1;
          cfg.setValueText(Number(v1));
        }
        slider.disabled = false;
        return;
      }

      if (mode === "flow1") {
        slider.classList.add("slider-flow1");
        if (document.activeElement !== slider) {
          slider.value = v1;
          cfg.setValueText(Number(v1));
        }
        slider.disabled = false;
        return;
      }

      if (mode === "flow2") {
        slider.classList.add("slider-flow2");
        if (document.activeElement !== slider) {
          slider.value = v2;
          cfg.setValueText(Number(v2));
        }
        slider.disabled = false;
        return;
      }

      if (same) {
        slider.classList.add("slider-both", "slider-both-same");
        if (document.activeElement !== slider) {
          slider.value = v1;
          cfg.setValueText(Number(v1));
        }
      } else {
        slider.classList.add("slider-both", "slider-both-diff");
        const refValue = lastTargetedFlow === 2 ? v2 : v1;
        if (document.activeElement !== slider) {
          slider.value = refValue;
          cfg.setValueText(Number(refValue));
        }
      }

      slider.disabled = false;
    } 



    function renderAllSliders() {
      TARGET_CONTROL_SLIDERS.forEach(renderSingleSlider);
    }

    function updatePulseRows() {
      const mode = activeTargetMode();

      let refSim = sim1;
      if (compareMode && mode === "flow2" && sim2) refSim = sim2;
      if (compareMode && mode === "both" && lastTargetedFlow === 2 && sim2) refSim = sim2;

      const on = !!refSim?.flow?.pulsationOn;
      pulseAmpRow.style.display = on ? "" : "none";
      pulseFreqRow.style.display = on ? "" : "none";
    }

    function updateBodyRows(type) {
      const aoaBodies = new Set(["airfoil", "naca2412", "square-cylinder"]);

      if (aoaRow) aoaRow.style.display = aoaBodies.has(type) ? "" : "none";
      if (nozzleRow) nozzleRow.style.display = type === "nozzle" ? "" : "none";
      if (typeof cylinderRow !== "undefined" && cylinderRow) {
        cylinderRow.style.display = type === "cylinder" ? "" : "none";
      }
      if (cavityRow) cavityRow.style.display = type === "lid-driven-cavity" ? "" : "none";
      if (typeof cavityAspectRow !== "undefined" && cavityAspectRow) {
        cavityAspectRow.style.display = type === "lid-driven-cavity" ? "" : "none";
      }
    }





    function syncTargetUI() {
      updateFlowLabels();
      renderAllCompareGroups();
      renderAllSliders();
      renderDisplayButtons();
      updatePulseRows();
      updateVelocityLabel();


      const mode = activeTargetMode();
      let refSim = sim1;
      if (compareMode && mode === "flow2" && sim2) refSim = sim2;
      if (compareMode && mode === "both" && lastTargetedFlow === 2 && sim2) refSim = sim2;
      if (refSim) updateBodyRows(refSim.params.bodyType);
      if (refSim) updateBodyRows(refSim.params.bodyType);
      updateReSliderBounds();

    }

    // =====================================================
    // Monitor mode helpers
    // =====================================================
    function currentMonitorMode() {
      if (sim1?.convergence) return sim1.convergence.getMode();
      return "convergence";
    }

    function setMonitorMode(mode) {
      if (sim1?.convergence) sim1.convergence.setMode(mode);
      if (sim2?.convergence) sim2.convergence.setMode(mode);
      updateMonitorModeButtons();
    }

    function updateMonitorModeButtons() {
      const mode = currentMonitorMode();

      [monitorConvBtn1, monitorConvBtn2].forEach(btn => {
        if (!btn) return;
        btn.classList.toggle("active", mode === "convergence");
      });

      [monitorForceBtn1, monitorForceBtn2].forEach(btn => {
        if (!btn) return;
        btn.classList.toggle("active", mode === "forces");
      });
    }

    function updateMonitorBars() {
      const show1 = convergenceVisible && !!sim1;
      const show2 = convergenceVisible && compareMode && !!sim2;

      if (monitorBar1) monitorBar1.style.display = show1 ? "" : "none";
      if (monitorBar2) monitorBar2.style.display = show2 ? "" : "none";

      updateMonitorModeButtons();
    }

    // =====================================================
    // Sim setup / recreation
    // =====================================================
    function cloneCanvasById(id) {
      const oldCanvas = document.getElementById(id);
      const newCanvas = oldCanvas.cloneNode(false);
      oldCanvas.parentNode.replaceChild(newCanvas, oldCanvas);
      return newCanvas;
    }

    function destroySimGpu(sim) {
      if (!sim) return;
      sim.fieldRenderer?.destroy();
      sim.gpu?.destroy();
      sim.fieldRenderer = null;
      sim.gpu = null;
      sim.ready = false;
    }

    function retireSim(sim) {
      if (!sim) return;
      sim.disposed = true;
      destroySimGpu(sim);
    }

    function setupSim(sim, flowId, bodyId, wallId, convId) {
      sim.flowCanvas = cloneCanvasById(flowId);

      const overlayId = flowId === "flow-canvas-1" ? "overlay-canvas-1" : "overlay-canvas-2";
      sim.overlayCanvas = cloneCanvasById(overlayId);

      const particlesId = flowId === "flow-canvas-1" ? "particles-canvas-1" : "particles-canvas-2";
      sim.particlesCanvas = cloneCanvasById(particlesId);

      sim.bodyCanvas = cloneCanvasById(bodyId);
      sim.wallCanvas = cloneCanvasById(wallId);
      sim.convCanvas = cloneCanvasById(convId);

      sim.flowCanvas.style.cursor = sandboxActive ? "crosshair" : "default";

      requestAnimationFrame(() => {
        if (sim.disposed) return;

        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const W = Math.round((sim.flowCanvas.clientWidth || sim.NX * 4) * dpr);
        const H = Math.round(W * sim.NY / sim.NX);

        sim.flowCanvas.width = W;
        sim.flowCanvas.height = H;

        sim.overlay = new OverlayRenderer(sim.overlayCanvas, sim.NX, sim.NY);
        sim.overlay.resize(W, H);

        sim.particlesCanvas.width = W;
        sim.particlesCanvas.height = H;
        sim.particlesCanvas.style.width = sim.flowCanvas.style.width || "100%";
        sim.particlesCanvas.style.height = sim.flowCanvas.style.height || "100%";

        sim.bodyCanvas.width = W;
        sim.bodyCanvas.height = H;
        sim.wallCanvas.width = W;
        sim.wallCanvas.height = H;

        sim.convCanvas.style.display = convergenceVisible ? "" : "none";

        const particleCount = compareMode ? 900 : 1400;
        sim.gpu = new GpuSim(gpu, sim, { particleCount });
        sim.gpu.setCanvasSize(W, H);
        sim.gpu.onDiagnostics = diag => handleDiagnostics(sim, diag);
        sim.fieldRenderer = new FieldRenderer(gpu, sim.gpu, sim.flowCanvas);

        sim.bodyRenderer = new BodyRenderer(sim.bodyCanvas, sim.NX, sim.NY);
        sim.userDraw = createUserDraw(sim, sim.wallCanvas);

        sim.particles = createParticles(sim, sim.particlesCanvas, {
          count: particleCount,
          dt: 1.0,
        });
        sim.particles.setEnabled(smokeVisible);
        sim.particles.resizeCanvas();

        sim.convergence = createConvergenceMonitor(sim);
        sim.convergence.attachCanvas(sim.convCanvas);
        sim.convergence.setVisible(convergenceVisible);
        sim.convergence.setMode("convergence");

        updateMonitorBars();

        bindCanvasInput(sim);
        resetSim(sim);
        sim.ready = true;
      });
    }


   function createSim(id, bodyType, aoaDeg, overrides = {}) {
      const { NX, NY } = currentDims();

      const physicsUI = getPhysicsFromUI();
      const flowUI = getFlowFromUI();

      return createSimInstance({
        id,
        NX,
        NY,
        bodyType,
        aoaDeg,
        nozzleDiameter: overrides.nozzleDiameter ?? parseInt(nozzleSlider.value, 10),
        cylinderDiameter: overrides.cylinderDiameter ?? parseInt(document.getElementById("cylinder-slider")?.value || "32", 10),
        cavityAspect: overrides.cavityAspect ?? parseFloat(cavityAspectSlider.value),
        u0: overrides.u0 ?? parseFloat(document.getElementById("u0-slider").value),
        reynolds: overrides.reynolds ?? parseFloat(document.getElementById("re-slider").value),
        displayMode: document.getElementById("mode-qcrit").classList.contains("active") ? "qcrit" : "speed",
        colorScale: parseFloat(document.getElementById("scale-slider").value),
        showVectors: document.getElementById("vector-btn").classList.contains("active"),
        showGrid: document.getElementById("dist-btn").classList.contains("active"),
        gridFine: document.getElementById("dist-btn").classList.contains("active"),
        showDist: document.getElementById("dist-btn").classList.contains("active"),
        stepsPerFrame: parseInt(document.getElementById("steps-slider").value, 10),
        running: document.getElementById("pause-btn").textContent !== "Resume",
        nozzleCy: Math.floor(NY / 2),
        uCoflow: U_COFLOW,

        collisionModel: overrides.collisionModel ?? physicsUI.collisionModel,
        lesModel: overrides.lesModel ?? physicsUI.lesModel,
        wallModel: overrides.wallModel ?? physicsUI.wallModel,

        pulsationOn: overrides.pulsationOn ?? flowUI.pulsationOn,
        pulsationAmp: overrides.pulsationAmp ?? flowUI.pulsationAmp,
        pulsationFreq: overrides.pulsationFreq ?? flowUI.pulsationFreq,
      });
    }

    function recreateSims() {
      const prevSim1 = sim1;
      const prevSim2 = sim2;
      retireSim(prevSim1);
      retireSim(prevSim2);

      sim1 = createSim(
        "sim1",
        prevSim1 ? prevSim1.params.bodyType : "airfoil",
        prevSim1 ? prevSim1.params.aoaDeg : 10,
        prevSim1 ? {
          nozzleDiameter: prevSim1.params.nozzleDiameter,
          u0: prevSim1.params.u0,
          reynolds: prevSim1.params.reynolds,
          collisionModel: prevSim1.physics.collisionModel,
          lesModel: prevSim1.physics.lesModel,
          wallModel: prevSim1.physics.wallModel,
          pulsationOn: prevSim1.flow.pulsationOn,
          pulsationAmp: prevSim1.flow.pulsationAmp,
          pulsationFreq: prevSim1.flow.pulsationFreq,
          cylinderDiameter: prevSim1.params.cylinderDiameter,
          cavityAspect: prevSim1.params.cavityAspect,

        } : {}
      );



      setupSim(sim1, "flow-canvas-1", "body-canvas-1", "wall-canvas-1", "conv-canvas-1");

      if (compareMode) {
        sim2 = createSim(
          "sim2",
          prevSim2 ? prevSim2.params.bodyType : "naca2412",
          prevSim2 ? prevSim2.params.aoaDeg : 10,
          prevSim2 ? {
            nozzleDiameter: prevSim2.params.nozzleDiameter,
            u0: prevSim2.params.u0,
            reynolds: prevSim2.params.reynolds,
            collisionModel: prevSim2.physics.collisionModel,
            lesModel: prevSim2.physics.lesModel,
            wallModel: prevSim2.physics.wallModel,
            pulsationOn: prevSim2.flow.pulsationOn,
            pulsationAmp: prevSim2.flow.pulsationAmp,
            pulsationFreq: prevSim2.flow.pulsationFreq,
            cylinderDiameter: prevSim2.params.cylinderDiameter,
            cavityAspect: prevSim2.params.cavityAspect,

          } : {}
        );



        setupSim(sim2, "flow-canvas-2", "body-canvas-2", "wall-canvas-2", "conv-canvas-2");
      } else {
        sim2 = null;
      }

      updateGridStats();
      requestAnimationFrame(() => {
        resizeOverlayCanvases();
        syncTargetUI();
        updateMonitorBars();
      });
    } 

   // =====================================================
    // Geometry
    // =====================================================
    function resolveGeometry(sim) {
      const dims = { NX: sim.NX, NY: sim.NY };
      const bodyType = sim.params.bodyType;
      const aoaDeg = sim.params.aoaDeg;
      const nozzleDiameter = sim.params.nozzleDiameter;
      const alpha = aoaDeg * Math.PI / 180;

      if (bodyType === "airfoil") {
        const chord = Math.floor(sim.NX / 3);
        const acx = Math.floor(sim.NX / 3);
        const acy = Math.floor(sim.NY / 2);
        const result = buildAirfoil(dims, acx, acy, chord, alpha);

        return {
          g: result.geometry,
          p: { type: "airfoil", cx: acx, cy: acy, chord, alpha, profile: "0012" },
          charLength: result.charLength
        };
      }

      if (bodyType === "naca2412") {
        const chord = Math.floor(sim.NX / 3);
        const acx = Math.floor(sim.NX / 3);
        const acy = Math.floor(sim.NY / 2);
        const result = buildNACA2412(dims, acx, acy, chord, alpha);

        return {
          g: result.geometry,
          p: { type: "naca2412", cx: acx, cy: acy, chord, alpha, profile: "2412" },
          charLength: result.charLength
        };
      }

      if (bodyType === "square-cylinder") {
        const cx = Math.floor(sim.NX / 4);
        const cy = Math.floor(sim.NY / 2);
        const size = Math.max(8, Math.floor(sim.NY / 4));
        const result = buildSquareCylinder(dims, cx, cy, size, alpha);

        return {
          g: result.geometry,
          p: { type: "square-cylinder", cx, cy, size, alpha },
          charLength: result.charLength
        };
      }

     if (bodyType === "lid-driven-cavity") {
        const cavityAspect = sim.params.cavityAspect ?? parseFloat(cavityAspectSlider.value);
        const result = buildLidDrivenCavity(dims, cavityAspect);

        return {
          g: result.geometry,
          p: {
            type: "lid-driven-cavity",
            wallThickness: result.wallThickness,
            cavityAspect: result.cavityAspect
          },
          charLength: result.charLength
        };
      }

      if (bodyType === "empty") {
        const result = buildEmpty(dims);
        return {
          g: result.geometry,
          p: { type: "empty" },
          charLength: result.charLength
        };
      }

      if (bodyType === "blocks") {
        const r = defaultBlockMatrix(dims);
        return {
          g: r.geometry,
          p: { type: "blocks", blocks: r.blocks, blockSize: r.blockSize },
          charLength: r.charLength
        };
      }

      if (bodyType === "nozzle") {
        const result = buildNozzle(dims, Math.floor(sim.NY / 2), nozzleDiameter);
        return {
          g: result.geometry,
          p: {
            type: "nozzle",
            cy: result.cy,
            diameter: result.diameter,
            length: result.length
          },
          charLength: result.charLength
        };
      }

     if (bodyType === "cylinder") {
        const cx = Math.floor(sim.NX / 4);
        const cy = Math.floor(sim.NY / 2);
        const diameter = sim.params.cylinderDiameter ?? Math.max(8, Math.floor(sim.NY / 4));
        const radius = diameter * 0.5;
        const result = buildCylinder(dims, cx, cy, radius);

        return {
          g: result.geometry,
          p: {
            type: "cylinder",
            cx,
            cy,
            radius
          },
          charLength: result.charLength
        };
      }

    }

    function applyGeometry(sim, { g, p, charLength }) {
      setCharacteristicLength(sim, charLength);

      sim.solidBase = g.solid;
      sim.solid = new Uint8Array(g.solid);
      sim.wallDist = g.wallDist;
      sim.wallAdjacent = g.wallAdjacent;
      sim.linkMask = g.linkMask;
      sim.linkQ = g.linkQ;
      sim.solidList = g.solidList;
      sim.fluidSurfaceList = g.fluidSurfaceList;

      // Signed distance field: needed by forceSurface.js to estimate the
      // local body normal used by the normal-filtered momentum exchange.
      sim.sdf = g.sdf;

      // ---------------------------------------------------
      // Force reference point / coefficient reference length
      // ---------------------------------------------------
      // The dynamic pressure 0.5*rho*u0^2 is evaluated live in
      // forceSurface.js from the current inlet velocity, so only the
      // reference length is fixed here.
      let refLength;

      if (p.type === "airfoil" || p.type === "naca2412") {
        sim.forceRef.x = p.cx - p.chord / 4;
        sim.forceRef.y = p.cy;
        refLength = p.chord;

      } else if (p.type === "cylinder") {
        sim.forceRef.x = p.cx;
        sim.forceRef.y = p.cy;
        refLength = 2 * p.radius;

      } else if (p.type === "square-cylinder") {
        sim.forceRef.x = p.cx;
        sim.forceRef.y = p.cy;
        refLength = p.size;

      } else if (p.type === "lid-driven-cavity") {
        sim.forceRef.x = sim.NX * 0.5;
        sim.forceRef.y = sim.NY * 0.5;
        refLength = charLength || sim.L || 1.0;

      } else if (p.type === "nozzle") {
        sim.forceRef.x = p.length * 0.5;
        sim.forceRef.y = p.cy;
        refLength = p.diameter;

      } else if (p.type === "blocks") {
        sim.forceRef.x = sim.NX * 0.5;
        sim.forceRef.y = sim.NY * 0.5;
        refLength = charLength || sim.L || 1.0;

      } else {
        sim.forceRef.x = 0.0;
        sim.forceRef.y = 0.0;
        refLength = Math.max(charLength || sim.L || 1.0, 1.0);
      }

      sim.forceRefLength = refLength;

      if (sim.bodyRenderer) {
        sim.bodyRenderer.clear();
        if (sim.params.bodyType !== "empty") {
          sim.bodyRenderer.draw(p);
        }
      }


      if (sim.nozzle) {
        if (sim.params.bodyType === "nozzle") {
          setNozzleInlet(sim, Math.floor(sim.NY / 2), sim.params.nozzleDiameter, U_COFLOW);
        } else {
          clearNozzleInlet(sim);
        }
      }

      if (sim.userDraw) sim.userDraw.redrawWalls();

      sim.gpu?.uploadGeometry();
    }

    function buildGeometry(sim) {
      applyGeometry(sim, resolveGeometry(sim));
    }

    function rebuildGeometryOnly(sim) {
      // Not set up yet: resetSim() will build from the new params.
      if (!sim.ready) return;
      sim.userDraw.clearDrawing();
      const res = resolveGeometry(sim);
      applyGeometry(sim, res);

      // Every solid node goes back to rest equilibrium f_i = w_i.
      sim.gpu.restSolids();
    }

    function resetSim(sim) {
      if (sim.userDraw) sim.userDraw.clearDrawing();
      buildGeometry(sim);
      setReynolds(sim, sim.params.reynolds);

      const init = buildInitialField(sim, sim.params.u0, sim.solid);
      sim.runtime.frameCount = 0;
      sim.runtime.stepCount = 0;
      if (sim.convergence) sim.convergence.reset();

      const inletU = getInletVelocity(sim);
      applyBoundaryConditions({ ...sim, ...init }, inletU);

      sim.gpu.uploadField(init.f, init.rho, init.ux, init.uy);

      if (sim.particles) sim.particles.reset();
    }



   // =====================================================
    // Canvas input
    // =====================================================
    function canvasToLattice(canvas, clientX, clientY, sim) {
      const rect = canvas.getBoundingClientRect();
      return {
        lx: (clientX - rect.left) * (sim.NX / rect.width),
        ly: (clientY - rect.top) * (sim.NY / rect.height),
      };
    }

    function bindCanvasInput(sim) {
      let isDrawing = false;
      const canvas = sim.flowCanvas;
      const draw = sim.userDraw;

      canvas.addEventListener("mousedown", e => {
        if (!sandboxActive || e.button !== 0) return;
        isDrawing = true;
        const { lx, ly } = canvasToLattice(canvas, e.clientX, e.clientY, sim);
        draw.beginStroke(lx, ly);
        draw.paintStroke(lx, ly, sandboxMode);
        draw.redrawWalls();
      });

      canvas.addEventListener("mousemove", e => {
        if (!sandboxActive || !isDrawing) return;
        const { lx, ly } = canvasToLattice(canvas, e.clientX, e.clientY, sim);
        draw.paintStroke(lx, ly, sandboxMode);
        draw.redrawWalls();
      });

      canvas.addEventListener("mouseup", () => {
        isDrawing = false;
        draw.endStroke(sandboxMode);
        draw.redrawWalls();
      });

      canvas.addEventListener("mouseleave", () => {
        isDrawing = false;
        draw.endStroke(sandboxMode);
        draw.redrawWalls();
      });

      canvas.addEventListener("touchstart", e => {
        if (!sandboxActive) return;
        e.preventDefault();
        isDrawing = true;
        const { lx, ly } = canvasToLattice(canvas, e.touches[0].clientX, e.touches[0].clientY, sim);
        draw.beginStroke(lx, ly);
        draw.paintStroke(lx, ly, sandboxMode);
        draw.redrawWalls();
      }, { passive: false });

      canvas.addEventListener("touchmove", e => {
        if (!sandboxActive || !isDrawing) return;
        e.preventDefault();
        const { lx, ly } = canvasToLattice(canvas, e.touches[0].clientX, e.touches[0].clientY, sim);
        draw.paintStroke(lx, ly, sandboxMode);
        draw.redrawWalls();
      }, { passive: false });

      canvas.addEventListener("touchend", () => {
        isDrawing = false;
        draw.endStroke(sandboxMode);
        draw.redrawWalls();
      });
    } 

   // =====================================================
    // Solver loop (GPU)
    // =====================================================
    // Force coefficients from the GPU momentum-exchange sums, scaled
    // with the live inlet velocity (forceSurface.js getScaling).
    function handleDiagnostics(sim, diag) {
      sim.force.fx = diag.fx;
      sim.force.fy = diag.fy;
      sim.force.mz = diag.mz;

      const rho0 = sim.RHO0 ?? 1.0;
      const U = sim.params?.u0 ?? sim.U0 ?? 0.1;
      const L = sim.forceRefLength ?? sim.L ?? 1.0;
      const qDyn = 0.5 * rho0 * U * U;
      sim.forceScale = qDyn * L;
      sim.momentScale = qDyn * L * L;

      // Cl negated: y=0 is at top of domain so upward lift = negative Fy.
      sim.coeff.cd = sim.forceScale !== 0 ? diag.fx / sim.forceScale : 0.0;
      sim.coeff.cl = sim.forceScale !== 0 ? -diag.fy / sim.forceScale : 0.0;
      sim.coeff.cm = sim.momentScale !== 0 ? diag.mz / sim.momentScale : 0.0;

      sim.convergence?.sampleFromDiagnostics(diag);
    }

    function encodeSimFrame(sim, encoder) {
      const running = sim.runtime.running;
      const steps = running ? Math.min(sim.runtime.stepsPerFrame, MAX_STEPS_PER_FRAME) : 0;

      // Instantaneous inlet velocity per sub-step, exactly as stepOnce().
      const inletVelocities = [];
      const firstStep = sim.runtime.stepCount;
      for (let s = 0; s < steps; s++) {
        inletVelocities.push(getInletVelocity(sim));
        sim.runtime.frameCount++;
        sim.runtime.stepCount++;
      }

      const showVectors = sim.display.showVectors && sim.display.mode === "speed";
      const showDist = sim.display.showGrid && sim.display.showDist;

      const pending = sim.gpu.encodeFrame(encoder, {
        inletVelocities,
        firstStep,
        advectParticles: running && smokeVisible,
        particleDt: sim.runtime.stepsPerFrame,
        diagnostics: steps > 0,
        gatherVec: showVectors,
        gatherDist: showDist,
        readParticles: smokeVisible && (running || !sim.gpu.latest.particles),
      });

      sim.fieldRenderer.encode(encoder, sim.display.mode);

      return {
        pending,
        meta: { frameCount: sim.runtime.frameCount, stepCount: sim.runtime.stepCount },
      };
    }

    function resizeOverlayCanvases() {
      forAllSims(sim => {
        if (!sim || !sim.convCanvas || !sim.flowCanvas) return;

        const cssW = Math.max(220, sim.flowCanvas.clientWidth * 0.42);
        const cssH = Math.max(130, sim.flowCanvas.clientHeight * 0.26);
        sim.convCanvas.style.width = `${cssW}px`;
        sim.convCanvas.style.height = `${cssH}px`;
        if (sim.convergence) sim.convergence.resize();
      });
    }



    let framesInFlight = 0;
    let lastTime = performance.now();
    let smoothedFps = 60;

    function tick(now) {
      // Back-pressure: never queue more than two frames of GPU work.
      if (framesInFlight >= 2) {
        requestAnimationFrame(tick);
        return;
      }

      const dt = now - lastTime;
      lastTime = now;
      smoothedFps = 0.9 * smoothedFps + 0.1 * (dt > 0 ? 1000 / dt : 60);

      const encoder = gpu.device.createCommandEncoder({ label: "frame" });
      const readbacks = [];

      forAllSims(sim => {
        if (!sim.ready || !sim.gpu) return;
        readbacks.push({ sim, ...encodeSimFrame(sim, encoder) });
      });

      gpu.queue.submit([encoder.finish()]);
      framesInFlight++;
      gpu.queue.onSubmittedWorkDone().then(() => { framesInFlight--; });

      for (const r of readbacks) r.sim.gpu.resolveReadback(r.pending, r.meta);

      forAllSims(sim => {
        if (!sim.ready) return;

        sim.overlay.draw(sim.gpu.latest.gather, {
          mode: sim.display.mode,
          scale: sim.display.colorScale,
          showVectors: sim.display.showVectors,
          showGrid: sim.display.showGrid,
          gridFine: sim.display.gridFine,
          showDist: sim.display.showDist,
        });

        if (sim.particles) sim.particles.draw();

        if (sim.convergence) sim.convergence.draw();
      });

      document.getElementById("fps-value").textContent = smoothedFps.toFixed(0);
      document.getElementById("frame-value").textContent =
        `${sim1 ? sim1.runtime.frameCount : 0}${sim2 ? " / " + sim2.runtime.frameCount : ""}`;

      requestAnimationFrame(tick);
    }

    // =====================================================
    // Compare target controls
    // =====================================================
    target1Btn.addEventListener("click", () => {
      if (target1Active && !target2Active) return;
      target1Active = !target1Active;
      if (target1Active && !target2Active) lastTargetedFlow = 1;
      if (target1Active && target2Active) lastTargetedFlow = 1;
      syncTargetUI();
    });

    target2Btn.addEventListener("click", () => {
      if (target2Active && !target1Active) return;
      target2Active = !target2Active;
      if (target2Active && !target1Active) lastTargetedFlow = 2;
      if (target1Active && target2Active) lastTargetedFlow = 2;
      syncTargetUI();
    }); 

   // =====================================================
    // Compare toggle
    // =====================================================
    compareBtn.addEventListener("click", () => {
      compareMode = !compareMode;
      compareBtn.classList.toggle("active", compareMode);
      document.querySelector(".compare-wrap").classList.toggle("compare-active", compareMode);
      canvasWrap2.style.display = compareMode ? "" : "none";

      if (compareMode) {
        canvasWrap2.offsetWidth;

        const droppedKey = dropFidelity(currentGridKey);
        if (droppedKey !== currentGridKey) {
          currentGridKey = droppedKey;
          setActiveBtn(gridGroup, `grid-${droppedKey}`);
          updateGridStats();
        }

        const currentBodyType = sim1 ? sim1.params.bodyType : "airfoil";
        const currentAoaDeg = sim1 ? sim1.params.aoaDeg : 10;

        retireSim(sim1);
        retireSim(sim2);

        sim1 = createSim("sim1", currentBodyType, currentAoaDeg);
        setupSim(sim1, "flow-canvas-1", "body-canvas-1", "wall-canvas-1", "conv-canvas-1");

        sim2 = createSim("sim2", "naca2412", currentAoaDeg);
        setupSim(sim2, "flow-canvas-2", "body-canvas-2", "wall-canvas-2", "conv-canvas-2");

        target1Active = true;
        target2Active = true;
        lastTargetedFlow = 1;
      } else {
        const raisedKey = raiseFidelity(currentGridKey);
        if (raisedKey !== currentGridKey) {
          currentGridKey = raisedKey;
          setActiveBtn(gridGroup, `grid-${raisedKey}`);
          updateGridStats();
        }

        const currentBodyType = sim1 ? sim1.params.bodyType : "airfoil";
        const currentAoaDeg = sim1 ? sim1.params.aoaDeg : 10;

        retireSim(sim1);
        retireSim(sim2);

        sim1 = createSim("sim1", currentBodyType, currentAoaDeg);
        setupSim(sim1, "flow-canvas-1", "body-canvas-1", "wall-canvas-1", "conv-canvas-1");

        sim2 = null;
        target1Active = true;
        target2Active = true;
        lastTargetedFlow = 1;
      }

      requestAnimationFrame(() => {
        resizeOverlayCanvases();
        syncTargetUI();
        updateMonitorBars();
      });
    }); 

   // =====================================================
    // Monitor button + mode buttons
    // =====================================================
    convergenceBtn.addEventListener("click", () => {
      convergenceVisible = !convergenceVisible;
      convergenceBtn.classList.toggle("active", convergenceVisible);

      if (sim1?.convCanvas) sim1.convCanvas.style.display = convergenceVisible ? "" : "none";
      if (sim2?.convCanvas) sim2.convCanvas.style.display = convergenceVisible ? "" : "none";

      if (sim1?.convergence) sim1.convergence.setVisible(convergenceVisible);
      if (sim2?.convergence) sim2.convergence.setVisible(convergenceVisible);

      updateMonitorBars();
      requestAnimationFrame(resizeOverlayCanvases);
    });

    monitorConvBtn1?.addEventListener("click", () => setMonitorMode("convergence"));
    monitorConvBtn2?.addEventListener("click", () => setMonitorMode("convergence"));
    monitorForceBtn1?.addEventListener("click", () => setMonitorMode("forces"));
    monitorForceBtn2?.addEventListener("click", () => setMonitorMode("forces"));

    // =====================================================
    // Body controls
    // =====================================================
   function switchBody(type) {
      targetedSims().forEach(sim => {
        sim.params.bodyType = type;
        rebuildGeometryOnly(sim);
      });

      syncTargetUI();
    }


    document.getElementById("body-cylinder").addEventListener("click", () => switchBody("cylinder"));
    document.getElementById("body-airfoil").addEventListener("click", () => switchBody("airfoil"));
    document.getElementById("body-naca2412").addEventListener("click", () => switchBody("naca2412"));
    document.getElementById("body-square-cylinder").addEventListener("click", () => switchBody("square-cylinder"));
    document.getElementById("body-lid-driven-cavity").addEventListener("click", () => switchBody("lid-driven-cavity"));
    document.getElementById("body-blocks").addEventListener("click", () => switchBody("blocks"));
    document.getElementById("body-empty").addEventListener("click", () => switchBody("empty"));
    document.getElementById("body-nozzle").addEventListener("click", () => switchBody("nozzle"));


    document.getElementById("aoa-slider").addEventListener("input", e => {
      const aoaDeg = parseFloat(e.target.value);
      document.getElementById("aoa-value").textContent = aoaDeg.toFixed(1) + "°";

      targetedSims().forEach(sim => {
        sim.params.aoaDeg = aoaDeg;
        if (["airfoil", "naca2412", "square-cylinder"].includes(sim.params.bodyType)) {
          rebuildGeometryOnly(sim);
        }
      });

      syncTargetUI();
    });

    nozzleSlider.addEventListener("input", () => {
      const nozzleDiameter = parseInt(nozzleSlider.value, 10);
      nozzleOutput.textContent = nozzleDiameter;

      targetedSims().forEach(sim => {
        sim.params.nozzleDiameter = nozzleDiameter;
        if (sim.params.bodyType === "nozzle") {
          setNozzleInlet(sim, Math.floor(sim.NY / 2), nozzleDiameter, U_COFLOW);
          rebuildGeometryOnly(sim);
        }
      });

      syncTargetUI();
    });

    cylinderSlider.addEventListener("input", () => {
      const diameter = parseInt(cylinderSlider.value, 10);
      cylinderOutput.textContent = diameter;

      targetedSims().forEach(sim => {
        sim.params.cylinderDiameter = diameter;
        if (sim.params.bodyType === "cylinder") {
          rebuildGeometryOnly(sim);
        }
      });

      syncTargetUI();
    });

    cavityAspectSlider.addEventListener("input", () => {
      const aspect = parseFloat(cavityAspectSlider.value);
      cavityAspectValue.textContent = aspect.toFixed(2);

      targetedSims().forEach(sim => {
        sim.params.cavityAspect = aspect;
        if (sim.params.bodyType === "lid-driven-cavity") {
          rebuildGeometryOnly(sim);
        }
      });

      syncTargetUI();
    });




   // =====================================================
    // Flow controls
    // =====================================================
    document.getElementById("u0-slider").addEventListener("input", e => {
      const u0 = parseFloat(e.target.value);

      targetedSims().forEach(sim => {
        sim.params.u0 = u0;
      });

      document.getElementById("u0-value").textContent =
        `Ma ${(u0 / CS_LATTICE).toFixed(2)}`;

      syncTargetUI();
    });




    document.getElementById("re-slider").addEventListener("input", e => {
      let reynolds = parseFloat(e.target.value);

      const reSlider = document.getElementById("re-slider");
      const reMin = parseFloat(reSlider.min);
      const reMax = parseFloat(reSlider.max);

      reynolds = Math.min(reMax, Math.max(reMin, reynolds));
      reynolds = Math.round(reynolds / RE_STEP) * RE_STEP;
      reynolds = Math.min(reMax, Math.max(reMin, reynolds));

      reSlider.value = String(reynolds);
      document.getElementById("re-value").textContent = reynolds.toFixed(0);

      targetedSims().forEach(sim => {
        sim.params.reynolds = reynolds;
        setReynolds(sim, reynolds);
      });

      syncTargetUI();
    });


    document.getElementById("pulse-on").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.flow.pulsationOn = true;
      });
      syncTargetUI();
    });

    document.getElementById("pulse-off").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.flow.pulsationOn = false;
      });
      syncTargetUI();
    });

    pulseAmpSlider.addEventListener("input", e => {
      const amp = parseFloat(e.target.value);
      pulseAmpValue.textContent = amp.toFixed(2);

      targetedSims().forEach(sim => {
        sim.flow.pulsationAmp = amp;
      });

      syncTargetUI();
    });

    pulseFreqSlider.addEventListener("input", e => {
      const freq = parseFloat(e.target.value);
      pulseFreqValue.textContent = freq.toFixed(3);

      targetedSims().forEach(sim => {
        sim.flow.pulsationFreq = freq;
      });

      syncTargetUI();
    });

    
    smokeBtn.addEventListener("click", () => {
      smokeVisible = !smokeVisible;
      smokeBtn.classList.toggle("active", smokeVisible);

      forAllSims(sim => {
        if (sim.particles) sim.particles.setEnabled(smokeVisible);
      });
    });


    // =====================================================
    // Physics controls
    // =====================================================
    document.getElementById("collision-trt").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.collisionModel = "trt";
      });
      syncTargetUI();
    });

    document.getElementById("collision-bgk").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.collisionModel = "bgk";
      });
      syncTargetUI();
    });

    document.getElementById("les-smag").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.lesModel =
          sim.physics.lesModel === "smagorinsky" ? "off" : "smagorinsky";
      });
      syncTargetUI();
    });

    document.getElementById("les-wale").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.lesModel =
          sim.physics.lesModel === "wale" ? "off" : "wale";
      });
      syncTargetUI();
    });


    document.getElementById("wall-on").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.wallModel = true;
      });
      syncTargetUI();
    });

    document.getElementById("wall-off").addEventListener("click", () => {
      targetedSims().forEach(sim => {
        sim.physics.wallModel = false;
      });
      syncTargetUI();
    }); 

   // =====================================================
    // Display controls
    // =====================================================
    document.getElementById("mode-speed").addEventListener("click", () => {
      setActiveBtn(modeGroup, "mode-speed");
      forAllSims(sim => sim.display.mode = "speed");
      renderDisplayButtons();
    });

    document.getElementById("mode-qcrit").addEventListener("click", () => {
      setActiveBtn(modeGroup, "mode-qcrit");
      forAllSims(sim => sim.display.mode = "qcrit");
      renderDisplayButtons();
    });

    document.getElementById("vector-btn").addEventListener("click", () => {
      const next = !sim1?.display.showVectors;
      document.getElementById("vector-btn").classList.toggle("active", next);
      forAllSims(sim => sim.display.showVectors = next);
      renderDisplayButtons();
    });

    document.getElementById("dist-btn").addEventListener("click", () => {
      const on = !sim1?.display.showGrid;
      document.getElementById("dist-btn").classList.toggle("active", on);
      forAllSims(sim => {
        sim.display.showGrid = on;
        sim.display.gridFine = on;
        sim.display.showDist = on;
      });
      renderDisplayButtons();
    });

    document.getElementById("scale-slider").addEventListener("input", e => {
      const scale = parseFloat(e.target.value);
      document.getElementById("scale-value").textContent = scale.toFixed(3);
      forAllSims(sim => sim.display.colorScale = scale);
    });

    // =====================================================
    // Runtime controls
    // =====================================================
    document.getElementById("steps-slider").addEventListener("input", e => {
      const steps = parseInt(e.target.value, 10);
      document.getElementById("steps-value").textContent = steps;
      forAllSims(sim => {
        sim.runtime.stepsPerFrame = steps;
      });
    });

    sandboxBtn.addEventListener("click", () => {
      sandboxActive = !sandboxActive;
      sandboxBtn.classList.toggle("active", sandboxActive);

      if (sim1) sim1.flowCanvas.style.cursor = sandboxActive ? "crosshair" : "default";
      if (sim2) sim2.flowCanvas.style.cursor = sandboxActive ? "crosshair" : "default";
    });

    clearBtn.addEventListener("click", () => {
      targetedSims().forEach(sim => {
        if (!sim.userDraw) return;
        sim.userDraw.clearDrawing();
        sim.userDraw.redrawWalls();
      });
    });

    pauseBtn.addEventListener("click", () => {
      const anyRunning =
        (sim1 && sim1.runtime.running) ||
        (sim2 && sim2.runtime.running);

      const next = !anyRunning;

      pauseBtn.textContent = next ? "Pause" : "Resume";
      pauseBtn.classList.toggle("active", !next);

      if (sim1) sim1.runtime.running = next;
      if (sim2) sim2.runtime.running = next;
    });

    document.getElementById("reset-btn").addEventListener("click", () => {
      if (sim1?.ready) resetSim(sim1);
      if (sim2?.ready) resetSim(sim2);
      requestAnimationFrame(syncTargetUI);
    }); 

   // =====================================================
    // Fidelity
    // =====================================================
    document.getElementById("grid-low").addEventListener("click", () => {
      currentGridKey = "low";
      setActiveBtn(gridGroup, "grid-low");
      recreateSims();
    });

    document.getElementById("grid-mid").addEventListener("click", () => {
      currentGridKey = "mid";
      setActiveBtn(gridGroup, "grid-mid");
      recreateSims();
    });

    document.getElementById("grid-high").addEventListener("click", () => {
      currentGridKey = "high";
      setActiveBtn(gridGroup, "grid-high");
      recreateSims();
    });

    // =====================================================
    // Tooltip
    // =====================================================
    document.querySelectorAll(".tooltip-trigger").forEach(el => {
      el.addEventListener("mouseenter", e => {
        tooltip.textContent = el.dataset.tip;
        tooltip.classList.add("visible");
        positionTooltip(e);
      });

      el.addEventListener("mousemove", positionTooltip);

      el.addEventListener("mouseleave", () => {
        tooltip.classList.remove("visible");
      });
    });

    function positionTooltip(e) {
      const pad = 14;
      const tw = tooltip.offsetWidth || 280;
      const th = tooltip.offsetHeight || 80;
      let x = e.clientX + pad;
      let y = e.clientY + pad;
      if (x + tw > window.innerWidth - pad) x = e.clientX - tw - pad;
      if (y + th > window.innerHeight - pad) y = e.clientY - th - pad;
      tooltip.style.left = x + "px";
      tooltip.style.top = y + "px";
    }

    window.addEventListener("resize", () => {
      resizeOverlayCanvases();

      if (sim1?.convergence) sim1.convergence.resize();
      if (sim2?.convergence) sim2.convergence.resize();

      if (sim1?.particles) sim1.particles.resizeCanvas();
      if (sim2?.particles) sim2.particles.resizeCanvas();
    });


   // =====================================================
    // Initial UI sync / bootstrap
    // =====================================================
document.getElementById("u0-value").textContent =
  `Ma ${(parseFloat(document.getElementById("u0-slider").value) / CS_LATTICE).toFixed(2)}`;


    document.getElementById("re-value").textContent =
      parseFloat(document.getElementById("re-slider").value).toFixed(0);

    document.getElementById("steps-value").textContent =
      parseInt(document.getElementById("steps-slider").value, 10);

    document.getElementById("scale-value").textContent =
      parseFloat(document.getElementById("scale-slider").value).toFixed(3);

    document.getElementById("aoa-value").textContent =
      parseFloat(document.getElementById("aoa-slider").value).toFixed(1) + "°";

    nozzleOutput.textContent = parseInt(nozzleSlider.value, 10);
    cylinderOutput.textContent = parseInt(cylinderSlider.value, 10);
    pulseAmpValue.textContent = parseFloat(pulseAmpSlider.value).toFixed(2);
    pulseFreqValue.textContent = parseFloat(pulseFreqSlider.value).toFixed(3);
    cavityAspectValue.textContent = parseFloat(cavityAspectSlider.value).toFixed(2);


    updateGridStats();
    setActiveBtn(modeGroup, "mode-speed");
    setActiveBtn(gridGroup, "grid-mid");

    recreateSims();

    requestAnimationFrame(() => {
      resizeOverlayCanvases();
      syncTargetUI();
      renderDisplayButtons();
      updateMonitorBars();
    });

    requestAnimationFrame(tick);
