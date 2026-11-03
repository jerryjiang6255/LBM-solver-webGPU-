# ML Plan: Physics-Informed CNN Correction for the WebGPU LBM Sandbox

**Status:** draft for review (UI layout finalised) · **Repo baseline:** `jerryjiang6255/LBM-solver-webGPU-` @ `bc0b7c9` · **Date:** 2026-11-03

Items marked **[DECIDE]** still need a decision. Items marked **[VERIFY]** need checking against the current code before implementation.

**UI mockups:** design canvas "CFD page control bar with ML updates" (https://claude.ai/artifact/4b3ka8gFMs2S8hnxvqXQRy): CFD page bar, training page bar with monitor/dataset windows, Classic view, Feature maps view. Section 8 describes them.

---

## 0. Goal

Make a **low-resolution** run (256×128) behave like a **high-resolution** run (e.g. 1024×512) while staying much faster than the high-res run.

A small CNN looks at the low-res flow every step and predicts a **stress correction** Π. Π is added to the populations through a fixed D2Q9 formula that **conserves mass and momentum exactly**. The CNN is trained on snapshot pairs captured from high-res runs and shrunk to low res. Training happens in the browser, on a dedicated training page that visualises the network as it learns.

Success means:
- on held-out cases, corrected low-res Cd, Cl amplitude and Strouhal number are clearly closer to high-res than plain low-res is;
- long runs stay stable;
- the corrected run is much cheaper than high-res (target: at least 5× faster).

---

## 1. How it works (summary)

One corrected low-res step:

```
macro (ρ, u) → ML inputs → CNN → Π → collide → f* + Δf(Π) → stream + BCs → outlet
```

- The **LBM still does the physics**. The CNN only adds a stress correction.
- **Physics layer (no learned numbers):** $\Delta f_i = \chi\,\frac{w_i}{2c_s^4}\,(Q_i:\Pi)$. For any Π: $\sum_i \Delta f_i = 0$, $\sum_i \mathbf{c}_i \Delta f_i = 0$, $\sum_i \mathbf{c}_i\mathbf{c}_i \Delta f_i = \Pi$.
- After streaming, the correction acts as $\Delta(\rho\mathbf{u}) \approx -\nabla\cdot\Pi$, i.e. a learned subgrid stress.

**Training on one snapshot pair:**
- **A** = high-res frame at step N, shrunk to low res (input).
- **C** = high-res frame at step N + r, shrunk (truth).
- **B** = one low-res step from A, with the CNN correction.
- The loss compares B with C. Backpropagation computes a gradient for every weight; Adam updates the weights.

---

## 2. Fixed decisions (the spec)

### 2.1 Resolution pairs

**New fidelity presets** (Fidelity tab: Ultra / High / Mid / Low):

| Preset | Grid | Nodes |
|---|---|---|
| Ultra (new) | 2048×1024 | 2,097,152 |
| High | 1024×512 | 524,288 |
| Mid | 512×256 | 131,072 |
| Low (new size) | 256×128 | 32,768 |

**This is a remap, not just an added button.** The current code has `low: 512×256, mid: 1024×512, high: 2048×1024` (`GRID_PRESETS` in `main.js`). Every level shifts down one name and a new 256×128 Low is added. **[VERIFY]** everything keyed on the old names (compare-mode fidelity drop, default fidelity, tooltips). **[DECIDE]** the default fidelity (currently Mid, which becomes 512×256).

- **Ultra:** check the adapter's buffer limits at startup and hide Ultra if it doesn't fit; disable it in compare mode.
- Models always **run at Low (256×128)**.
- The capture's high resolution is whatever fidelity the CFD page is on when capturing. **Capture and Sweep are disabled at Low and in compare mode.**

| Capture fidelity | Ratio r | Model label |
|---|---|---|
| Mid 512×256 | 2 | `256←512` |
| High 1024×512 | 4 | `256←1024` (**recommended default**) |
| Ultra 2048×1024 | 8 | `256←2048` |

- **One model = one resolution pair.** Collections and models from different pairs must never be mixed. Enforce this everywhere (capture, combine, training checks, import).

### 2.2 Time pairing and scaling

With ratio r, the same inlet velocity in lattice units, and geometry scaled by r:
- one low-res step = **r high-res steps**: capture a frame every r high-res steps;
- viscosity matched for the same Re: $\tau_{hi} = 0.5 + r(\tau_{lo} - 0.5)$;
- body size in cells: r times larger at high res.

**[VERIFY]** how the current presets set u0, τ (ν0) and body size per fidelity, and that the low-res and high-res runs of a case really have the same Re and physically aligned geometry. Check by overlaying the shrunk high-res solid mask on the low-res mask.

### 2.3 Physics settings during capture, training and runtime

The current default has **Smagorinsky LES on** (`les-smag` active). An LES model competes with the learned stress, so this must be fixed per model.

**[DECIDE]**, recommended option A:
- **A (recommended):** capture high-res with **LES off** where stable; run the low-res training step and the runtime with a **fixed** setup (e.g. TRT, LES off, wall model off). The CNN learns the full subgrid correction.
- **B:** keep Smagorinsky on at low res; the CNN learns the residual on top of it.

Whichever is chosen: **training and runtime must use identical collision, LES, wall-model and BC settings**. Record them in the model card.

**UI rule:** selecting a model on the CFD page **applies that model's physics settings and locks** the TRT/BGK, Smag/WALE and Wall buttons (tooltip: "set by ML2"). Clicking the model again turns it off and unlocks them.

**Re and inlet velocity** are both sliders (Flow tab). τ is computed from Re, u0, body size and fidelity. Clamp the Re slider's maximum so τ stays above the stability limit for the current u0 and fidelity (stricter at Low), and explain the limit in a tooltip.

The pulsing inlet depends on the step count, so record step numbers with every frame.

### 2.4 Inputs (9 channels, 256×128 each)

| # | Channel | Formula | Notes |
|---|---|---|---|
| 0 | density deviation | ρ − 1 | |
| 1 | x-velocity | u_x / u0 | |
| 2 | y-velocity | u_y / u0 | |
| 3 | strain xx | ∂u_x/∂x · D/u0 | central differences |
| 4 | strain xy | ½(∂u_x/∂y + ∂u_y/∂x) · D/u0 | |
| 5 | strain yy | ∂u_y/∂y · D/u0 | |
| 6 | vorticity | (∂u_y/∂x − ∂u_x/∂y) · D/u0 | |
| 7 | wall distance | min(d, 10)/10 | from the SDF, in low-res cells |
| 8 | Reynolds number | Re / Re_ref | constant over the grid |

- All channels except 7 and 8 are 0 inside solids.
- D = body size in low-res cells. **[DECIDE]** Re_ref (e.g. 200).
- Padding at domain edges: replicate the edge value.

### 2.5 Architecture

- **Standard:** 9 → 16 → 16 → 16 → 3, all 3×3 convolutions, ReLU after the hidden layers, **no** ReLU on the output. 6,387 parameters.
- **Sizes offered:** teaching (8 channels, ≈2.0k params), standard (16, ≈6.4k), large (4 hidden layers × 32, ≈31k). The size is fixed when a model is created.
- **Initialisation:** He for hidden layers (σ = sqrt(2 / (C_in·9))), biases 0; **output layer all zeros** (an untrained model = the plain solver).
- **Field of view:** 9×9 cells (11×11 counting the input derivatives).
- **Option:** leaky ReLU (slope 0.01) if dead channels become a problem.

### 2.6 Output and physics layer

- Raw output $o$ (3 channels) → $\Pi_{\alpha\beta} = s\,\rho\,u_0^2\,o_{\alpha\beta}$. **[DECIDE]** s (default 0.05).
- $\Delta f$ per direction (fluid cells only, χ = 1; also χ = 0 in the sponge zone):

| i | c_i | Δf_i |
|---|---|---|
| 0 | (0,0) | −(2/3)(Πxx + Πyy) |
| 1, 3 | (±1,0) | (1/3)Πxx − (1/6)Πyy |
| 2, 4 | (0,±1) | (1/3)Πyy − (1/6)Πxx |
| 5, 7 | (1,1), (−1,−1) | (1/12)(Πxx + Πyy) + (1/4)Πxy |
| 6, 8 | (−1,1), (1,−1) | (1/12)(Πxx + Πyy) − (1/4)Πxy |

**[VERIFY]** the direction numbering against `solver.wgsl` (CXA/CYA tables) and remap the table to match.

- Applied **after collide, before stream**.
- **Runtime strength α** (Physics tab slider, 0–100%, default 100%): Π_used = α·Π. Training always uses α = 1.

### 2.7 Loss

$$\mathcal{L} = \frac{1}{N_{fluid}}\sum M(x,y)\,\frac{|\mathbf{u}_B - \mathbf{u}_C|^2}{u_0^2}$$

**Loss mask M = 0** for: solid cells; cells whose r×r high-res block was partly solid; inlet columns; outlet columns; the sponge zone.

Optional later terms (off by default):
- density: λ_ρ ≈ 0.1, scaled by $(3u_0^2)^2$;
- force: λ_F ≈ 0.01–0.1 on Cd and Cl.

### 2.8 Optimiser and training defaults

| Setting | Default | Where in the UI |
|---|---|---|
| Batch size | 8 | Settings slider |
| Max epochs | 100 | Settings slider |
| Learning rate η | 1e-3 (new), 3e-4 (continue) | Settings slider |
| Unroll steps K | 1 | Settings → Advanced (1, 2, 4, 8, 16) |
| Model size | standard | chosen when **New** is selected in Mode; locked afterwards |
| Adam β1, β2, ε | 0.9, 0.999, 1e-8 | advanced |
| Warm-up | ~200 steps, linear | advanced |
| Reduce on plateau | ×0.3 after 5 epochs without improvement | advanced |
| Early stopping | patience 15, minimum improvement 0.5%, minimum η 1e-5 | advanced |
| Gradient clipping | on (global norm cap) | advanced |
| Pairs per epoch | random subset (e.g. 2,000) if the dataset is larger | advanced |

**Keep the best epoch** (lowest validation loss), not the last one.

---

## 3. Runtime integration (CFD page)

Per-step dispatch when a model is active (in `gpuSim.js`):

```
macro_main → ml_inputs → conv L1 → conv L2 → conv L3 → conv out (Π)
          → collide_main → ml_apply (Δf into fB) → stream_main → outlet_main
```

- **Separate pipelines and bind groups for the ML passes**, each with ≤ 4–5 storage buffers. Do not add bindings to the solver's bind group (see parity audit A1).
- **Buffers (Low, float32):** inputs 9 maps (1.2 MB), h1/h2/h3 16 maps each (2.1 MB each), Π 3 maps (0.4 MB), weights (~25 KB), uniforms (strength, s, u0, Re_ref, D).
- **Expected cost:** standard model ≈ 6,300 multiply-adds per cell per step, roughly 3–10× a plain low-res step depending on shader quality. Optional: apply every N steps (the model must then be trained the same way).
- **Model buttons** (ML1–ML4) are greyed out unless the sim is at Low (the model's low resolution). Selecting one applies and locks its physics settings (2.3); clicking it again turns the model off. Empty slots are shown dashed and disabled.

---

## 4. Data capture (CFD page)

### 4.1 Shrinking (momentum-conserving, on the GPU)

For each r×r block:
- $\rho_{lo} = \text{mean}(\rho)$
- $(\rho u)_{lo} = \text{mean}(\rho u)$
- $u_{lo} = (\rho u)_{lo} / \rho_{lo}$

Also compute a per-block **fluid fraction**, used for the loss mask.

### 4.2 Step-aligned capture

- The capture hooks into the **solver step counter**, not the render loop (sub-steps per frame are not multiples of r).
- Save a frame when `(step − startStep) % r == 0`.
- Read back asynchronously (`mapAsync`) and write to OPFS immediately. Never accumulate frames in RAM. If the write queue grows, pause capture briefly rather than block the solver.

### 4.3 Automatic capture (Capture button)

1. **Wait for convergence:** running mean of Cd and Cl amplitude change < ~1% over the last few periods (reuse the history in `convergenceMonitor.js`).
2. **Detect the flow type** from the lift signal: shedding (Cl oscillates; period from zero crossings) or steady.
3. **Capture:**
   - shedding: 2 cycles × 16 bursts per cycle, spaced evenly in **phase**, 9 frames per burst (8 pairs) → 256 pairs;
   - steady: 3 bursts → 24 pairs.
4. **Automatic checks:** converged, no NaN, ρ within limits, frame spacing exactly r steps, consistent period, shrunk totals match high-res totals.
5. The result goes to **pending**. Approve / Discard.

Also store the case's **high-res reference time series** (Cd, Cl, St; small) for later comparison on the CFD page.

**Optional advanced:** capture the transient (from start); custom cycles and bursts.

### 4.4 Sweep

- **Runs on the CFD page** (Dataset tab → Sweep). **[DECIDE]** where its parameters are set: the final training-page bar has no sweep panel, so the simplest option is a **popup panel below the CFD bar** (like Monitor) opened by the Sweep button. A "fill gaps" action on the training page's dataset window could pre-fill it.
- Parameter lists (Re, body, angle of attack, …); full grid or random spread; estimated time and storage shown before starting.
- Per case: set parameters at high res → converge → automatic capture → checks → save to pending.
- Start each case from the previous converged flow where only Re changed.
- Skip and log failed cases; save progress after each case (resumable); Pause / Resume / Cancel.
- Automatic held-out marking (see 6.3).

### 4.5 Pending / approve / discard

- Pending data lives in OPFS `/pending/`, so it survives reloads. On load, prompt: "Unapproved capture found (N pairs): Approve / Discard".
- **Approve** turns the whole pending set into a new **collection** (a relabel, not a copy). **Discard** deletes the whole pending set (with confirmation).
- A pending review list allows deleting individual recordings before approving (needed for sweeps).

---

## 5. Storage (OPFS) and file formats

### 5.1 Layout

```
/pending/<recordingId>/            recording.json, geometry.bin, frames.bin
/collections/<collectionId>/       collection.json + recordings/<recordingId>/...
/models/<modelId>/                 card.json, weights.bin, adam.bin, coreset/, validation/, versions/
```

- Call `navigator.storage.persist()` on first use; show `navigator.storage.estimate()` in a storage meter (warn at ~80%).
- **Origin note:** OPFS is per origin (scheme + host + **port**). Keep a fixed dev port. Use export/import to move data between dev and the deployed site, or between machines.

### 5.2 Recording format

- **`recording.json`:** id, resolution pair (lo NX×NY, hi NX×NY, r), solver version/commit, physics settings (collision, LES, wall model, pulse), case (body, Re, u0, τ_lo, τ_hi, AoA, size), start step, frame spacing, frame count, step number per frame, held-out flag, auto-check results, reference Cd/Cl/St summary.
- **`geometry.bin`:** low-res solid mask, SDF, BFL link fractions (q), fluid-fraction / loss mask.
- **`frames.bin`:** N × [ρ−1, u_x, u_y] at 256×128.

**[DECIDE] precision: float32 recommended.** Velocity errors between low and high res are ~1e-4–1e-3, and float16 spacing near u ≈ 0.1 is ~6e-5, which adds noticeable noise to the training targets. Storing ρ−1 (not ρ) is required in any 16-bit format. Option: ρ−1 in float16 and velocities in float32.

**Size (float32):** ~393 KB per frame; ~440 KB per pair with 9-frame bursts; ~113 MB per automatic shedding capture.

### 5.3 Model file

- **`card.json`:** name, description, version history, resolution pair, solver version, physics settings, architecture, constants (s, Re_ref, wall clip, input scaling), training history (datasets, epochs, best validation loss, R, rollout metrics), training range (Re, bodies, angles).
- **`weights.bin`:** all weights and biases.
- **`adam.bin`:** m, v, step count.
- **`coreset/`** and **`validation/`:** see 6.2 and 6.3.

**Export options:** "model only" (weights + card, ~25 KB) or "model + core set" (for continuing training elsewhere).

**Import** checks resolution pair, solver version and architecture. Export/import of large files is **streamed in chunks** (never one giant in-memory buffer). Every collection and recording has a unique ID, to detect duplicate imports.

---

## 6. Dataset management

### 6.1 Collections

- Every approval creates a collection. Collections form a **shared pool** that only grows; **nothing is deleted automatically**.
- **Dataset window** (training page → Session → Dataset; layout in 8.4): one card per collection (geometry, Re range, U0, grid pair, pairs, held-out count, size, badges), with per-case details on expand.
- **Checkboxes select; buttons act on the selection:**
  - **Train** (use for training)
  - **Combine:** groups recordings and keeps per-recording metadata; same resolution pair only
  - **Edit:** remove cases, rename, toggle held-out
  - **Delete**
- Badges: "in training", "held out", "used by".
- Sort, filter, collapse; storage meter; optional storage cap.
- **Import dataset** (Session tab) adds a collection from a file.

### 6.2 Core set (per model)

- Created automatically when a model is saved; stored with the model; **locked** in the dataset window (shown only for the selected model; deleted only with the model).
- **Content:** a fixed number of pairs **per case** (e.g. 20–30), spread across shedding phases, for **every case the model has ever trained on**. Resample stratified by case after each round, so old cases never fade out.
- **Continue training:** core set selected automatically. Unticking it shows a warning ("may forget earlier cases").
- Because every model carries its own core set, **any collection can be deleted without breaking a model**.

### 6.3 Validation set (and optional test set)

**Automatic selection (default), editable:**
1. Group by **case** (unique geometry, Re, u0, AoA), merging duplicates across collections so they land on the same side.
2. Per geometry, hold out ~15% of cases (at least 1), picked from the **interior** of the range (not the extremes).
3. At least 2 validation cases overall; fixed random seed.
4. Warn if there are too few cases. A geometry with one case stays in training (no validation coverage, noted).

- **Locked per model once saved.** Continue-training may only **add** held-out cases from new data.
- **Optional test set:** a few cases used only for final evaluation (paper numbers), never for decisions.

---

## 7. Training engine (`trainer.js` + `backward.wgsl`)

### 7.1 Per pair, forward

1. Load A and C (and the case geometry); optionally mirror (see 7.3).
2. **Rebuild populations from A:** $f_i = f_i^{eq}(\rho,\mathbf{u}) + f_i^{neq}$, with $f_i^{neq} = -3\,w_i\,\rho\,\tau\,(Q_i:S)$ (S from central differences; use τ+ for TRT).
3. Compute inputs; run the CNN forward, **storing** in, z/h for each layer, and Π.
4. Run the **same** macro/collide as the solver (same WGSL) → f*; apply Δf → f**; stream + BCs → f_new; compute ρ_B, u_B.
5. Compute the loss against C on the mask.

### 7.2 Per pair, backward

| Stage | Operation | Blame shape (Low) |
|---|---|---|
| 1 | loss → δu_B = 2M(u_B − u_C)/(N u0²) | 2×256×128 |
| 2 | δf_new_i = δu·(c_i − u_B)/ρ_B | 9×256×128 |
| 3 | reverse stream: δf**_i(x) = δf_new_i(x + c_i); reverse BCs (bounce-back: same route; BFL: transpose of its linear mix; inlet: stop; outlet copy: add to source) | 9×256×128 |
| 4 | δΔf = δf** (identity; the f* branch is dropped for K = 1) | — |
| 5 | δΠ = transposed Δf table (·χ) | 3×256×128 |
| 6 | δo = s ρ u0² δΠ | 3×256×128 |
| 7 | output layer: ∂L/∂V = Σ δo·h3; ∂L/∂b = Σ δo; δh3 = flipped-filter gather | grads 3×16×3×3+3; δh3 16×256×128 |
| 8 | ReLU gate: δz3 = δh3 where h3 > 0 | 16×256×128 |
| 9 | layer 3: grads Σ δz3·h2; δh2 gather | 16×16×3×3+16 |
| 10–11 | gate, then layer 2 | 16×16×3×3+16 |
| 12–13 | gate, then layer 1 (grads Σ δz1·in; stop; optional saliency δin) | 16×9×3×3+16 |

- Stages 2–6 run in one kernel (gather from neighbours, no intermediate arrays).
- **Start with BFL reversed as plain bounce-back**, then refine.
- **Weight gradients:** one workgroup per weight, tree reduction over cells (no float atomics in WGSL; avoid integer-atomic overflow, cf. audit A8).
- **Batch:** zero the gradient buffers, accumulate over B pairs, average, NaN check, clip, then one Adam step.

### 7.3 Batches and data handling

- **Diverse batches:** random pairs across cases, Re values, phases and bodies. Balance sampling so each case is drawn about equally. Batches must contain **several Re values** for the Re input to be learnable.
- **Mirror augmentation (y-flip):** u_y → −u_y, ω → −ω, S_xy → −S_xy, Π_xy → −Π_xy, and the **geometry buffers must be mirrored too** (mask, SDF, BFL q). Only valid if the flipped domain and BCs are physically equivalent. **[VERIFY]** for each body/BC.
- **Prefetch** the next batch from OPFS while the GPU trains.
- Training never loads the whole dataset to the GPU.

### 7.4 Per epoch

- **Validation loss** (forward only) on the validation set; R = L(model) / L(plain) on the same pairs.
- Early stopping, reduce-on-plateau, save the best; checkpoint every epoch (weights + Adam) for crash recovery.
- **Rollout test** every ~5 epochs: a few thousand low-res steps with the model on 1–2 validation cases → Cd mean, Cl amplitude, St vs high-res reference; stability flag.

### 7.5 Multi-step training (K > 1, later phase)

- Run K corrected steps from A; compare with C at A + K·r high-res steps.
- The backward pass chains through all K steps; the f* branch is now **kept** (it depends on earlier corrections).
- Memory ≈ K × single-step. Keep K·Δt a small fraction of the shedding period.

---

## 8. User interface

Both pages share one shell: a header line, then a single control bar of grouped buttons (same colours, fonts and button style as the current app), with popup windows that open **below the bar**. Target bar width ≈ 1,860 px, so it fits a 1920-wide screen; groups use stacked 2-column grids to stay narrow. Every new control gets a tooltip. Mockups: see the design canvas linked at the top.

### 8.1 CFD page bar (9 groups)

| Group | Controls | Notes |
|---|---|---|
| **Session** | Sandbox · **Training** (new) · Monitor · Compare | Training switches to the training view (same page, same OPFS) |
| **Body** | NACA 0012 · NACA 2412 · Blocks · Cylinder · Empty; body parameter slider (e.g. Cylinder diameter) | Nozzle, Lid Cavity and Square removed (lid-cavity code paths can go) |
| **Flow** | Pulse Off / Pulse On · Smoke Tracer Particles · Inlet velocity slider · **Reynolds number slider** (moved here from Physics) | Re slider clamped by stability (2.3) |
| **Physics** | 2×3 grid: TRT \| BGK · Smag \| WALE · Wall On \| Wall Off; **Model strength** slider (new) | Buttons lock to the model's settings while a model is active; strength slider disabled ("off") with no model |
| **Dataset** (new) | Capture · Sweep · Approve · Discard; counter **pending / total** pairs | Capture/Sweep only at Mid/High/Ultra, not in compare mode. Approve/Discard act on the whole pending set |
| **Model** (new) | ML1 · ML2 · ML3 · ML4, stacked; buttons show the slot's model name | Greyed unless at Low; click again = off; empty slot dashed |
| **Display** | Vectors · Distribution · Speed · Q-Criterion; Color scale slider | Optional later: "ML correction" view mode (Π or −∇·Π) |
| **Stats** | Pause · Clear · Reset; Sim speed slider; FPS and Frame readouts | unchanged |
| **Fidelity** | **Ultra** (new) · High · Mid · Low, stacked; grid and node readout | Remapped presets (2.1) |

**Monitor overlay (CFD page):** add an **ML** mode next to Convergence and Forces, showing: capture status + phase-coverage ring, sweep progress (Pause / Resume / Cancel), out-of-range warning (current case outside the model's training range), reference comparison (plain / ML / stored high-res Cd, Cl amplitude, St), and a conservation meter. A small badge appears on the Monitor button when something needs attention.

**Other CFD-page rules:**
- Pending data survives reloads; on load, prompt "Unapproved capture found (N pairs): Approve / Discard".
- Optional: compare mode with ML vs no-ML side by side.
- Switching to the training view must destroy or pause the CFD sim's GPU resources and listeners (cf. audit A11).

### 8.2 Training page bar (9 groups)

The header gains a **TRAINING** tag and a summary line (architecture and resolution pair). Groups 2–8 are the training path, numbered 1–7 in their headings; later steps unlock as earlier ones are completed (name and description are editable at any time).

| Group | Controls | Notes |
|---|---|---|
| **Session** | Solver · Monitor · Dataset · Import Dataset | Monitor and Dataset open popup windows below the bar (8.3, 8.4); both can be open together |
| **1 Model** | ML1 · ML2 · ML3 · ML4, stacked | Empty slots dashed |
| **2 Mode** | New · Continue · Import Model; readout: model size, parameter count, resolution pair | New in an occupied slot asks for confirmation. Model size is chosen on New and locked afterwards. Import checks resolution pair, solver version and architecture |
| **3 Settings** (optional) | Batch size · Max epochs · Learning rate sliders; **Advanced ›** | Advanced holds unroll steps K, Adam, warm-up, plateau, patience, clipping, pairs per epoch (2.8) |
| **4 Checks** | Run checks; list of results tagged **OK / WARN / ERROR** | Errors block Start (no data, no validation set, resolution/physics/architecture mismatch). Warnings allow Start (little data, low variety, continuing without core set, single Re). Also runs automatically when settings change |
| **5 Train** | Start · Pause · **Save model**; status line | Save **glows green** when early stopping triggers (training pauses). Save stores the **best** checkpoint as a new version and updates the model's core set |
| **6 Name** | Name field; Description text box | Default name e.g. "Model 2 · 256←1024 · v3" |
| **7 Export** | Model · Dataset; **Delete** (model) | Export model: "model only" or "model + core set". Delete asks for confirmation |
| **View** | Classic; Feature maps: Measurements · Situations · Ingredients | See 8.5 |

### 8.3 Monitor window (training page)

Popup below the bar, left side (~380 px wide):
- **Readouts:** epoch / max, validation loss, best validation loss @ epoch, training loss, improvement ratio R, current learning rate (plus ETA and pairs/s).
- **Loss chart:** training and validation curves per epoch, with a marker at the best epoch.
- **Diagnostics** (expandable): gradient and update size per layer (|Δw|/|w| ≈ 1e-3 healthy), NaN/clip counts, channel activity and dead channels, weight histograms, mean |Π|/(ρu0²), conservation residual, Π vs true subgrid stress correlation, per-case validation loss, error by region, rollout Cd/St/stability, storage.

### 8.4 Dataset window (training page)

Popup below the bar, right side:
- **Header:** total size (pairs, cases), memory used / available (storage meter, warn at ~80%), and what is currently selected (collections + core set).
- **Action buttons** on the left, acting on the checked collections: **Train** (use for training) · **Combine** · **Edit** · **Delete**.
- **Collection cards** (one per approved batch), each with: name, geometry list, Re range [min, max, count], U0, angle range if any, grid pair, pairs + held-out count, size, badges ("in training", "held out", "used by"), and a **select** checkbox. More than four cards scroll horizontally.
- **Core set card** for the selected model: shown only for that model, **locked**, lists its cases, pairs per case, fixed validation set and size, "updated each time the model is saved", and is **included automatically** when continuing (unticking it raises a check warning).
- Nothing is ever deleted automatically; Import Dataset (Session) adds a collection from a file.

### 8.5 Main area: views

Shared toolbar above both views:
- **Probe cell** picker: small speed-field thumbnail of the selected sample; click to move the probe.
- **Sample** selector (which snapshot pair is shown).
- **Show:** Activation / Blame toggle.
- **View** buttons (mirror the bar's View group).
- **Pass:** Step / Watch / Fast mode, plus a status line ("forward ▸ L2 · epoch 47 · batch 112").

**Classic view (default):**
- Columns: 9 inputs (labelled ρ−1, uₓ, u_y, Sxx, Sxy, Syy, ω, d, Re) → layer 1 (16) → layer 2 (16) → layer 3 (16) → output Π (3), then a physics-layer box (Π → Δf → stream → B) and the current pair's loss.
- **Circles:** fill brightness = activation **at the probe cell** (dark = off); channel number inside; amber **ring** = % of cells where the channel is active; **dead channels** outlined in dashed coral.
- **Lines:** teal = support (w > 0), coral = veto (w < 0); width = filter strength Σ|w|; weak lines faded by default ("show all" option).
- **Pass animation:** the layer being computed is highlighted; lines into it are bright, later layers faded until reached. Backward pass sweeps right → left in coral; lines flash on update.
- **Hover a line:** popover with the 3×3 filter (numbers, coloured by sign), Σ|w| and its contribution at the probe.
- **Hover a circle:** popover with that channel's feature map (probe marked), % active and value at the probe.
- Legend along the bottom.

**Feature-map views (Measurements = layer 1, Situations = layer 2, Ingredients = layer 3):**
- **4×4 grid** of the 16 channel maps for the selected sample (same colour map as the flow field), probe marked on every tile. Caption: channel number, an editable "looks like: …" hint (channels have no built-in names), % active, value at the probe. Dead channels dimmed with a coral border.
- **Selected channel panel:** enlarged map, stats (% active, value at probe, bias, hint), and its **incoming 3×3 filters** (one per input channel; strongest outlined).
- Blame toggle shows the blame maps instead of activations.
- At smaller window heights the grid scrolls or tiles shrink.

**Readbacks for visuals:** non-blocking (`mapAsync`), at most ~1–2 per second in Watch mode; full detail in Step mode; minimal in Fast mode.

---

## 9. Tests (`tests/ml-tests.html`, run after every ML shader change)

| Test | Pass condition |
|---|---|
| Conservation (random Π, thousands of steps) | total mass and momentum unchanged to float rounding; per cell Σ Δf = 0, Σ c Δf = 0, Σ cc Δf = Π |
| Uniform Π | interior velocity unchanged |
| Smagorinsky equivalence | Π = −2ρν_t S matches a run with τ + 3ν_t |
| Strength 0 / zero-init model | bit-identical to the plain solver |
| Shrink | shrunk totals of ρ and ρu equal the high-res totals |
| Capture alignment | frame step numbers exactly r apart |
| Physics backward | uniform δu → δΠ = 0 in the interior; nudge test on Π (incl. near walls) |
| Gradient checks | central difference vs backward for sample weights/biases in every layer (rel. error < 1e-2, ε ≈ 1e-3; avoid z ≈ 0 cells) |
| ReLU gates | δz = 0 exactly where h = 0 |
| Re/bias check | single-Re batch: Σ_pq ∂L/∂W_{k,Re} = (Re/Re_ref)·∂L/∂b_k |
| Zero-start | V = 0 → δh3 = 0 exactly and ∂L/∂V ≠ 0 |
| Adam | one step on known gradients matches a hand calculation |
| OPFS persistence | pending and collections survive a reload; import/export round-trip is identical |

**Acceptance (per model):**
- R < 1 on validation;
- rollout Cd/St error clearly below plain low-res on validation cases;
- stable for ≥ 10 shedding periods;
- runtime cost vs plain low-res and vs high-res reported.

---

## 10. Build phases

| Phase | Build | Done when |
|---|---|---|
| 0 | Measure the gap: cylinder Re 100/200 at Low vs High (Cd, Cl amp, St) | the gap is known (decides how much there is to gain) |
| 1 | Physics layer: Π buffer, `ml_apply`, strength | conservation, uniform-Π, Smagorinsky and strength-0 tests pass |
| 2 | CNN forward: `ml_inputs`, `conv`, weights, He init, step hook | random model runs every step; feature maps readable; cost measured |
| 3 | Capture: shrink, step hook, OPFS, pending/approve/discard, automatic capture | shrink/alignment/persistence tests pass |
| 4 | Training engine (no UI) | all gradient checks pass; one case trains to R < 1; rollout moves Cd/St toward high-res |
| 5 | Training page UI | train and watch from the UI |
| 6 | Model management: slots, versions, core set, validation, import/export | continue-training keeps old cases; cross-machine round-trip works |
| 7 | Sweep | unattended multi-case sweep completes and resumes |
| 8 | Multi-step training (K > 1) | long rollouts more stable than with K = 1 |

**Phase 4 is the main risk.** Prove it before building the full UI.

---

## 11. Code layout and integration points

```
src/ml/
  shaders/  mlInputs.wgsl  conv.wgsl  mlApply.wgsl  shrink.wgsl  reconstruct.wgsl
            backward.wgsl  reduce.wgsl  adam.wgsl
  cnnModel.js   mlRuntime.js   capture.js   datasetStore.js   trainer.js
  trainingPage.js   networkView.js
tests/ml-tests.html
docs/ML_PLAN.md
```

**Touch points in existing code:**
- `src/gpu/gpuSim.js`: ML dispatches in the step loop; capture hook on the step counter.
- `src/main.js`, `index.html`: new Dataset and Model groups, strength slider, Re slider moved to Flow, Training button, view switching, removed bodies, and the `GRID_PRESETS` remap plus Ultra. Move tab/UI logic into modules rather than growing `main.js`.
- `src/convergenceMonitor.js`: reuse its Cd/Cl history for convergence and period detection.
- **Reuse** the solver's macro/collide/stream WGSL in training, so training and runtime are guaranteed identical.

---

## 12. Development setup

- VS Code + Live Server (or `npx serve`) on a **fixed port**; a WGSL syntax extension; Chrome.
- Label all buffers/pipelines; use `device.pushErrorScope("validation")` around new passes during development; inspect OPFS in DevTools → Application.
- `git tag v1-cfd` before starting. One branch per phase (`ml/phase1-physics-layer`, …).
- Hide ML UI behind `?ml=1` until ready.
- Run `tests/ml-tests.html` after every ML shader change.

---

## 13. Open decisions and checks

- [ ] **[DECIDE]** LES/collision setup for capture, training and runtime (2.3)
- [ ] **[DECIDE]** Re_ref, stress scale s (2.4, 2.6)
- [ ] **[DECIDE]** frame precision: float32, or ρ−1 f16 + u f32 (5.2)
- [ ] **[DECIDE]** default fidelity after the preset remap (2.1)
- [ ] **[DECIDE]** where sweep parameters are set: popup below the CFD bar recommended (4.4)
- [x] Training steps: 7 numbered steps in the training bar (8.2)
- [x] Add a 2048×1024 fidelity: yes, as Ultra (2.1)
- [ ] **[VERIFY]** fidelity preset remap: everything keyed on the old Low/Mid/High names (2.1)
- [ ] **[VERIFY]** fidelity presets: u0, τ and body size per fidelity; geometry alignment across fidelities (2.2)
- [ ] **[VERIFY]** Ultra fits the adapter's buffer limits; hide it otherwise (2.1)
- [ ] **[VERIFY]** D2Q9 direction numbering vs the Δf table (2.6)
- [ ] **[VERIFY]** which bodies/BCs allow mirror augmentation (7.3)
- [ ] **[VERIFY]** BFL formula in `stream_main`, for the backward pass (7.2)

---

## 14. References

- Um et al., "Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers," NeurIPS 2020.
- Kochkov et al., "Machine learning–accelerated computational fluid dynamics," PNAS 118(21), 2021.
- Corbetta et al., "Toward learning Lattice Boltzmann collision operators," Eur. Phys. J. E, 2023.
- Bedrunka et al., "Machine learning enhanced collision operator for the lattice Boltzmann method based on invariant networks," 2024/2025.
- Gonzalez-Sieiro et al., "Reducing spatial discretization error on coarse CFD simulations using an OpenFOAM-embedded deep learning framework," 2024.
- Smagorinsky, "General circulation experiments with the primitive equations," Monthly Weather Review, 1963.
- He et al., "Delving deep into rectifiers," ICCV 2015.
- Kingma & Ba, "Adam: A method for stochastic optimization," 2014.
