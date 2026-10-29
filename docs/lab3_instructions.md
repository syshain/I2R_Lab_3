# LAB 3 — Vision-Guided Pick & Place of Two Coloured Blocks

**Module:** Robotics for Mechanical Engineers (Undergraduate / MSc)
**Robot Platform:** UFACTORY Lite 6 (6-DOF articulated arm, wrist-mounted camera, vacuum gripper)
**Prerequisites:** Lab 1 (Forward Kinematics, Camera & Hand-Eye Calibration), Lab 2 (Inverse Kinematics & Trajectory Planning)
**Duration:** 3 hours
**Software:** Python 3.12+, OpenCV 4.x, NumPy, SciPy, xArm Python SDK

---

## Learning Objectives

By the end of this lab you will be able to:

1. Segment a known object in a live monocular image using **HSV colour thresholding**, and tune the thresholds interactively against real lighting.
2. Infer depth from a single image when the object's physical size is known, using the pinhole camera model.
3. Convert an image-plane error into a base-frame motion command, and drive a robot **iteratively from live visual feedback** (eye-in-hand servoing).
4. Solve inverse kinematics locally in Python (reusing the Lab 2 method) and command the servos safely through the xArm SDK.
5. Integrate calibrated DH parameters and the hand-eye transform (from Lab 1) into a multi-stage, human-supervised pick-and-place task that includes a **vacuum gripper**.
6. Read and extend a **modularised** control stack — recognising which module owns detection, planning, display, and hardware, and why those boundaries are drawn where they are.

The central idea of the lab is that a wrist-mounted camera sees the block *through* the geometry between the camera, the flange, and the suction cup. Getting the cup onto the block therefore requires reasoning about **three linked frames at once** — not just "move toward where the pixel is."

---

## Code Architecture

Following the convention established in Labs 1–2, you work in `Lab_3/skeletons/`, a flat directory of single-responsibility modules with descriptive docstrings and plain sibling imports. Each file owns exactly one concern and arrives with its function bodies stubbed for you to implement; the entry point (`pick_and_place.py`) is a thin orchestrator that sequences them but owns no geometry or hardware detail of its own. The table below describes what each module is responsible for — the *signatures* are given, the *bodies* are yours to write.

| Module | Responsibility | Key contents |
|--------|----------------|--------------|
| `constants.py` | Every tunable knob + HSV defaults | `BLOCK_SIDE_MM`, `TOOL_OFFSET_MM`, `DOWN_RPY_DEG`, phase tolerances/steps, safety-taper limits, `COLOR_THRESHOLDS` (live-mutable HSV bands) |
| `vision.py` | Colour detection → base-frame point | `_hsv_color_mask`, `_best_blob`, `detect_color_block`, `unproject_to_base`, `compute_detection` |
| `motion.py` | Geometry planners + robot-facing moves | `flange_pose_base`, `plan_centre_step`, `plan_descent_step`, `cup_shift_flange`, `cup_axis_base`, `compute_topface_target`, `move_to_pose`, `nudge_z`, `nudge_xy`, `orient_downward`, `shift_cup_over_block`, `descend_along_cup_axis` |
| `ui_display.py` | Camera, OpenCV UI, calibration loaders, key input | `load_camera_intrinsics`, `load_calibration_results`, `open_camera`, `size_windows`, `show_frame`, `wait_next`, `tune_masks`, trackbar helpers |
| `fk_functions.py` | Forward kinematics (shared from Lab 1) | `fk_from_zeta(q, zeta)` |
| `ik_solver.py` | Numerical inverse kinematics (port of Lab 2) | `solve_ik(T_target, q0, zeta)`, `pose_to_matrix`, joint limits |
| `robot_io.py` | Thin xArm SDK wrapper (all hardware calls) | `connect_arm`, `read_joints_rad`, `move_to_joints`, `go_home`, `open_bio_gripper`, `close_bio_gripper`, `stop_bio_gripper` |
| `pick_and_place.py` | **Orchestrator only** — interactive phase loops + top-level `run()` | `vertical_nudges`, `fine_position`, `approach_and_pick`, `safe_hold`, `run` |

Two design rules worth internalising before you touch anything:

- **Internal maths runs in metres + radians; the SDK speaks millimetres + degrees.** The unit conversion happens once, inside `robot_io.py`. Everything upstream of it works in SI.
- **No cycles.** `pick_and_place` imports from `motion`, `vision`, and `ui_display`; those three import from `constants` (and `vision`); none of them import back up to the orchestrator. You can read any leaf module in isolation without loading the whole program.

To run: `cd Lab_3/src && python3 pick_and_place.py`

---

## Problem Statement

The robot's end-effector carries both a wrist camera and a **vacuum gripper** (a flat suction cup). On the work surface sit two wooden cubes, each with side length **s = 25 mm**: one **red**, one **green**. Wooden blocks are used deliberately — they give the vacuum cup a sealed surface to grip and present a matte, low-reflection face that segments cleanly.

Your task has two stages:

**Stage 1 — Pick the red block.** Locate it by colour, approach it with a five-phase vision-guided manoeuvre, engage the vacuum, confirm the grip, and lift back to a safe overview pose.

**Stage 2 — Track & place the green block.** From that overview pose, run the *same* five-phase navigation on the green block, then set it down in place and retract.

Every stage is **keygated**: the live camera feed is shown the whole time and the arm only advances when you press a key. A human stays in the safety loop for every autonomous motion. Press **[q]** at any point to abort — the script releases the suction and holds position so you can recover by hand.

**The Challenge:** A monocular camera gives you only 2D pixels, but the cup must land on a specific 3D point on the table. Worse, the camera and the cup are physically separated — the optical axis does *not* pass through the suction pad. You must use the calibrated hand-eye transform to translate "the block is here in the image" into "the cup is here in space," accounting for that offset explicitly.

---

## Part A — Monocular Depth Estimation Theory

### Pinhole Camera Model

A calibrated pinhole camera projects a 3D point **P** = [X, Y, Z]ᵀ in the camera frame onto the image plane at pixel coordinates (u, v):

$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

where $f_x, f_y$ are the focal lengths in pixels and $(c_x, c_y)$ is the principal point. These form the intrinsic matrix $\mathbf{K}$ you calibrated in Lab 2:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### Recovering Depth from Known Size

If an object of known physical width $S$ (metres) subtends $w$ pixels in the image, and its face is approximately perpendicular to the optical axis, then by similar triangles:

$$Z \approx \frac{f_x \cdot S}{w}$$

This is exactly what `detect_color_block()` computes: it takes the larger bounding-box dimension as $w$ and plugs in `BLOCK_SIDE_MM`. The approximation degrades when the cube is tilted relative to the camera (foreshortening makes it look narrower, so depth is **over**-estimated) — which is why the code applies a correction bias later.

**Task A1:** Using your intrinsics (`fx ≈ fy`, read them from `data/camera_matrix.npy`) and a **25 mm** cube, calculate the expected pixel width of the cube at distances of 150 mm, 250 mm, and 400 mm. At what distance does the cube drop below the detector's minimum reliable size? (Compare with `MIN_AREA_PX = 300` and the aspect-ratio guard in `_best_blob`.)

**Task A2:** Derive the depth uncertainty $\sigma_z$ as a function of pixel-measurement uncertainty $\sigma_w$. If $\sigma_w \approx 2$ px (one-pixel edge error on each side), what is $\sigma_z$ at $Z = 250$ mm? Propagate through $z = f \cdot S / w$ with the chain rule. Express your answer in millimetres.

---

## Part B — Colour Detection Pipeline (HSV Segmentation)

### Why HSV, not RGB?

A raw BGR channel box ("is R high and G,B low?") couples hue to brightness: the same red cube reads differently under warm vs. cool room light because its value channel shifts. **Hue–Saturation–Value** separates the *colour* (hue) from how much colour there is (saturation) from how bright it is (value), so a band chosen on hue survives lighting changes far better than a fixed RGB rectangle.

Two details make this pipeline robust:

- **Red wraps around the hue wheel.** In OpenCV, hue is quantised to `[0, 179]`. Red sits at *both* ends of that range (near 0 and near 179), so a single contiguous band cannot capture it. The code therefore defines **two** red bands — `H[0,10]` and `H[160,179]` — and unions them. Green is a single mid-hue band `H[40,85]`.
- **Live-tunable trackbars.** Because no default perfectly matches your filament + lamps, the program opens two mask windows with sliders for every bound before any motion happens. Dragging a slider mutates the shared `COLOR_THRESHOLDS` dict, so the very next frame re-masks with the new values.

### Code Overview

Detection lives in `src/vision.py`; the interactive tuning UI lives in `src/ui_display.py`:

**`vision.py` — segmentation & base-frame resolution:**

| Function | Purpose |
|----------|---------|
| `detect_color_block(frame, color, K, dist)` | Undistorts, masks by colour, finds the best blob, estimates depth. Returns a dict or `{'mask': ...}` if nothing found. |
| `_hsv_color_mask(frame_bgr, color)` | Converts BGR→HSV once, then ORs together every band in `COLOR_THRESHOLDS[color]`. This is where red's two wraparound bands get unioned. |
| `_best_blob(mask)` | Picks the single best contour. Guards: area within `[MIN_AREA_PX, MAX_AREA_FRAC·img_area]`, aspect ratio within `[0.3, 3.0]`, confidence ≥ `MIN_CONFIDENCE`. |
| `unproject_to_base(u, v, depth_m, K, T_cam)` | Unprojects a pixel at a given depth into a BASE-frame 3-D point (metres). |
| `compute_detection(frame, color, K, dist, T_flange, T_ee_cam)` | **Single source of truth** for every downstream consumer: detects the block *and* resolves it into the base frame (`p_block_base`, `dist_flange_mm`, `xy_gap_mm`). Display, approach, and descent all consume this so they agree on the same numbers. |

**`ui_display.py` — live threshold tuning (no robot involved):**

| Function | Purpose |
|----------|---------|
| `tune_masks(cap, K, dist)` | Interactive calibration step. Runs its own tight display loop showing the feed + both mask windows with their HSV bound-sliders; **[Enter]** accepts, **[r]** resets, **[q]** quits. |
| `create_threshold_trackbars()` / `sync_threshold_trackbars()` / `read_threshold_trackbars()` / `reset_thresholds()` | Build, keep in sync, read, and restore the HSV sliders. Dragging one mutates the shared `COLOR_THRESHOLDS` dict immediately. |
| `show_frame(...)` | Reads a frame, draws the overlay + detection box + optional extra marks (e.g. the projected cup tip), displays, returns the pressed key. |

Each successful detection returns:
- `center_px` : (u, v) centroid
- `bbox_px`   : (x, y, w, h) bounding box
- `depth_m`   : camera-to-block distance from known size (metres)
- `confidence`: quality score [0, 1]
- `mask`      : (H, W) uint8 debug mask

**Confidence scoring:** a solid cube viewed head-on fills its bounding box almost completely, so `_best_blob` *rewards* high fill ratio (≥ 0.7 → full credit) rather than penalising it, and adds a circularity bonus for compact shapes. An earlier draft rejected the highest-fill blobs outright — the inversion is worth studying.

**Task B1 — Tune the Masks (hands-on).** Run the program and stop at the TUNE MASKS step:

```bash
cd Lab_3/src
python3 pick_and_place.py
```

Drag the sliders in the RED and GREEN windows until each highlights *only* its cube — no background bleed, no holes in the block. Record the final HSV bounds the console prints. Then:

1. Why does red need **two** hue bands while green needs one? What would happen to red detection if you kept only the `H[0,10]` band?
2. Which bound (H/S/V lo or hi) did you have to move most to reject the table/shadows? Explain what that bound is doing.
3. What is the purpose of the `MAX_AREA_FRAC = 0.5` guard? What would the detector return if you removed it and pointed the camera at a mostly-uniform wall?

**Task B2 — Analyse the Blob Selection.** Read `_best_blob` and answer:

1. Why binarise (threshold) before `cv2.findContours`? A solid cube on a flat background has no internal edge — what would contouring the raw grayscale find instead?
2. The aspect-ratio guard rejects blobs wider than 3× or taller than 3×. Give a realistic false-positive (e.g. a strip of coloured tape on the table) that this guard filters out.
3. Write a short paragraph (3–4 sentences) on the trade-off between **sensitivity** (catching the real block under varied lighting) and **specificity** (rejecting the table, shadows, clutter) in this pipeline, and which knob trades off against which.

---

## Part C — Geometry: Three Linked Frames

This is the heart of the lab. The camera, the flange, and the suction cup do **not** share a common origin, and getting the cup onto the block means transforming across all three.

### The Coordinate Chain

All internal maths is in **metres + radians**; the SDK speaks **millimetres + degrees**. At any instant:

$$T_{flange} = FK(q,\ \zeta) \qquad (\text{base} \to \text{flange})$$

$$T_{cam} = T_{flange} \cdot T_{ee\_cam} \qquad (\text{base} \to \text{camera})$$

where $\zeta$ is the nominal DH parameter set (from `fk_functions.fk_from_zeta`) and $T_{ee\_cam}$ is the joint-based hand-eye transform loaded by `ui_display.load_calibration_results()`. A detected pixel centre is unprojected out to its measured depth to give a 3-D point in the **base** frame (`vision.unproject_to_base`); every planned move is expressed against that base-frame point, so all commands live in the robot's own coordinates.

### The Cup Offset (why "centre in frame" ≠ "cup over block")

In the flange frame, the point where the camera's optical axis crosses the flange plane (the **view-axis footprint**) sits roughly **~57 mm off to one side**, whereas the suction-cup tip is at the flange origin projected down along flange Z. `motion.cup_shift_flange(T_ee_cam)` computes this as a pure **XY-only** translation — `footprint[:2] - tip[:2]` — deliberately leaving height untouched. That ~57 mm lateral separation is the crux of the task:

- Servoing the **camera** onto the block keeps vision alive, but leaves the **cup** ~57 mm away in XY.
- So the approach splits into "**lower the camera** onto the block" (Phases B/C) followed by a one-shot horizontal "**shift the cup**" (Phase D) computed directly from the hand-eye matrix.

### Orientation: making the cup point straight down

With the wrist flipped to `(roll, pitch, yaw) = (0°, 180°, 0°)` (`constants.DOWN_RPY_DEG`), the flange **+Z points down** toward the table and the pad hangs out the +Z side. This anti-parallel flip is what Phase A performs via `motion.orient_downward`. Note the consequence: **"down" for the cup is the flange +Z direction carried into the base frame** — *not* a hardcoded base −Z. The helper `motion.cup_axis_base(T_flange, T_ee_cam)` derives the true descent direction from the camera optical axis so it stays correct even if the wrist isn't perfectly level.

**Task C1 — Trace the Transform.** Given a detection whose base-frame point is $p_{block}$ and the current $T_{flange}$, sketch (in words or equations) the sequence of transforms that yields the flange-frame XY offset between the view-axis footprint and the cup tip. Where in the code is this computed? (Look at `cup_shift_flange`.)

**Task C2 — Why derive, don't hardcode.** Suppose the operator jogs the wrist slightly off-level during Phase A. Why does computing the descent direction from the optical axis (`cup_axis_base`) beat a fixed `[0, 0, -1]` base vector? What goes wrong with the fixed version?

---

## Part D — The Five-Phase Approach

`pick_and_place.approach_and_pick(...)` runs these phases for whichever block is targeted. Each is keygated. The pure geometry/planning lives in `motion.py`; the interactive keystroke loops that call it live in `pick_and_place.py`.

| Phase | Name | What it does | Keys |
|-------|------|--------------|------|
| **A** | Orient Down | Flip the wrist to `pitch 180°` so the cup faces the table. | `[Enter]` to execute |
| **B** | Centre in Frame | Keystroke-driven **rotational** re-aiming that drives the block's pixel centroid to the principal point. No advance — pure aiming. Up to `LATERAL_STEPS` presses. | `[Enter]` per step; auto-exits when within `CENTRE_TOL_PX` |
| **C** | Lower Camera | Each press issues **ONE fused IK-solved move** that re-aims *and* advances `DESCENT_STEP_MM` (15 mm) along the optical axis. Keeps the camera locked on the block (vision stays alive). | `[w]`/`[s]`/↑/↓ advance; `[Enter]` done; `[a]` abort |
| **D** | Shift Cup Over Block | One-shot flange-frame **XY-only** translation (~57 mm, height unchanged) carrying the cup from the view-axis footprint onto the block. Also snapshots the top-face seat pose and the lift-back pose. | `[Enter]` to shift |
| **E** | Fine Positioning | Hover-and-settle loop: creep the cup onto the top face, manage suction, confirm the grip. | see table below |

### Why fuse aim + advance into one motion (Phase C)?

An earlier design did "lateral move, then vertical move" as two separate solves. Between the two, the block drifted, so each solve corrected the other's error — producing visible **fishing/hunting**. Fusing aim and advance into a single target pose (`motion.plan_descent_step`) means there is only one motion to solve, so nothing fights itself and no inter-step error accumulates. This logic was ported from the proven white-block navigator in `reference/python_cube_nav/navigator.py::plan_step`; Phase B uses its rotation-only sibling `motion.plan_centre_step`.

### Phase E — Fine Positioning Controls

Once the cup is hovering over the block, this interactive loop settles it onto the top face and manages the vacuum:

| Key | Action |
|-----|--------|
| `[s]` / ↓ | Creep **down** `FINE_Z_STEP_MM` (3 mm, base Z) |
| `[w]` / ↑ | Creep **up** `FINE_Z_STEP_MM` |
| `[z]` / `[c]` | Creep **−X** / **+X** `FINE_XY_STEP_MM` (3 mm) |
| `[a]` / `[d]` | Creep **−Y** / **+Y** `FINE_XY_STEP_MM` |
| `[g]` | Drive to the frozen **top-face seat pose** `T_target` (position *and* orientation captured at the Phase D gate). Falls back to a live descent along the cup axis if no target was captured. |
| `[o]` | **Engage** suction (close the vacuum cup onto the block) |
| `[r]` | **Release** suction (open the cup) |
| `[y]` | **Confirm** the grip — accepted only once suction is engaged; this is the phase exit |
| `[x]` | Abort (note: `[a]` is taken by −Y here, so abort moved to `[x]`) |

The phase does **not** end on its own after engaging suction: you must press `[y]` to confirm the grip looks good. That prevents a weak or failed suction from being lifted straight away.

**Gripper control.** `[o]` and `[r]` drive the vacuum cup via the `robot_io.py` wrappers (`open_bio_gripper` / `close_bio_gripper`), which route to the SDK's `open_lite6_gripper` / `close_lite6_gripper` and surface the returned status code rather than assuming success. Treat `[o]` as "make the cup grasp" and `[r]` as "let go"; verify on your rig which physical state each produces before relying on it. (See the polarity warning under Task D3.)

**Task D1 — Study the Fusion.** Read `motion.plan_centre_step` and `motion.plan_descent_step` side by side. They share an AIM block but differ only in the ADVANCE section. Explain, in terms of the residual the IK solver minimises, why issuing aim+advance as *one* target pose removes the fishing that the two-step version exhibited.

**Task D2 — The Seat Pose.** `motion.compute_topface_target` freezes a full flange pose (position + orientation) at the Phase D gate, and `[g]` drives straight to it. Why snapshot the pose *once* at that moment rather than re-estimating the contact height on every `[g]` press? What would re-estimating on each press risk introducing?

**Task D3 — Live Pick (hardware required).** With both blocks placed on a stable surface, run the full program and complete Stage 1 (pick the red block):

```bash
cd Lab_3/src
python3 pick_and_place.py
```

Work through: TUNE MASKS → FREE MOVE (jog until both blocks are visible) → STAGE 0 → Phases A–E for the red block. Observe and record:
- Does the block stay centred during Phase B, or drift off-centre (a sign of a hand-eye or sign error)?
- How many Phase C advances did the descent take? Was there any visible hunting?
- At the Phase D gate, what cup-offset (mm) did the console report? Does it match your Part C estimate of ~57 mm?
- What was the flange Z at touchdown, and did `[g]` land the cup on the top face?
- Did the vacuum hold on the first `[o]`, or did you need to re-seat with `[s]`?

> **⚠ Verify gripper polarity on your rig before trusting `[o]`/`[r]`.** The `robot_io.py` wrappers document `open_bio_gripper` as *release* and `close_bio_gripper` as *engage*, and every other release site in the program (green set-down, `safe_hold`) uses `open_bio_gripper()` to let go. But inside `fine_position()`, `[o]` ("ENGAGE") currently calls `open_bio_gripper()` and `[r]` ("RELEASE") calls `close_bio_gripper()` — the reverse. Confirm which key actually makes the cup grasp; if it's inverted, the fix is a one-line swap of those two calls. Do not rely on `[y]` "confirming" a grip until you've watched the pad physically seat.

---

## Part E — Multi-Block Sequencing & Safe Teardown

### Lifting back to an overview pose

After confirming the red grip, the arm does **not** home blindly. During the free-move prelude and Stage 0, the orchestrator (`pick_and_place.py`) continuously remembers the last flange pose from which **both** blocks were simultaneously visible (`capture_both_blocks_pose` / `maybe_update_overview`). That pose becomes `T_liftback`: after picking, the arm drives back to it, so the remaining (green) block is still in view and findable by vision. If no such pose was ever captured, it falls back to raising `constants.LIFT_BACK_UP_MM` (80 mm) along the cup axis from the pre-shift hover.

### Terminal behaviours

`approach_and_pick` takes a `terminal` argument:

- **`'lift'`** (default, red): after `[y]` confirm, drive to `T_liftback` and return holding the block.
- **`'release'`** (green): after `[y]` confirm, set the block **down in place**, open the cup, then retract to `T_liftback`.

### Distance-based safety taper

`motion._approach_schedule(dist_mm)` scales **both** the commanded step length and the servo speed linearly with distance: far away → fast, big steps (`FAR_SPEED_PCT`, `ADVANCE_STEP_MAX_MM`); inside `constants.CLOSE_ZONE_MM` (60 mm) → creeping, tiny steps floored at `ADVANCE_STEP_MIN_MM` (1 mm). This keeps the final approach gentle without slowing the initial travel.

### Safe teardown

On completion the arm homes (`robot_io.go_home`). On **any** failure or abort, `pick_and_place.safe_hold` releases the suction and **holds position** (it does *not* home), so you can inspect and recover by hand without the arm sweeping somewhere unexpected. Aborting after a confirmed grip always releases suction first so a held block is never dragged.

**Task E1 — Why return to the overview pose?** Explain why lifting back to a remembered both-blocks-visible viewpoint is safer and more useful than either (a) a blind +Z clearance bump or (b) homing immediately. What does it guarantee for the second block?

**Task E2 — Failure injection.** Mentally walk through what happens if you press `[x]` (abort) in Phase E *after* engaging suction but *before* confirming. Trace the exact sequence of calls that ensures the block is not dragged. Which line releases the suction, and why does it run before the arm moves anywhere?

**Task E3 — Improvements.** Propose two concrete improvements to this pipeline (e.g. force/torque sensing for touchdown instead of vision-only depth, a grasp-quality metric from suction pressure, automatic re-grasp on slip). For each, state which error source it addresses, the rough order-of-magnitude benefit, and implementation complexity.

---

## Deliverables — Lab Report

Submit a report (PDF or Word) containing:

1. **Title page** — name, student number, date, lab partner(s).
2. **Part A** — Tasks A1 and A2 with worked calculations (use your actual `fx`).
3. **Part B** — Your tuned HSV bounds; answers to B1 (why red needs two bands, which bound mattered most, the max-area guard) and B2 (binarisation, aspect guard, sensitivity/specificity paragraph). Include a screenshot of the two mask windows at acceptance.
4. **Part C** — Task C1 (transform trace to the cup offset) and C2 (why derive the descent direction).
5. **Part D** — Task D1 (fusion vs. fishing), D2 (snapshot rationale), and D3 live results: Phase B centring behaviour, Phase C step count and any hunting, the reported Phase D offset vs. your ~57 mm estimate, touchdown Z, and whether the vacuum held first try.
6. **Part E** — Tasks E1, E2, and E3.
7. **Synthesis** (½ page) — Compare this eye-in-hand pick-and-place with the open-loop calibrated approach of Labs 1–2. When is closed-loop vision guidance essential, and when is calibrated kinematics alone sufficient? What fundamental limitation of monocular vision persists even with perfect calibration?

---

## File Structure

```
Lab_3/
├── src/                           # ACTIVE code for this lab (flat siblings)
│   ├── pick_and_place.py          # Orchestrator only: interactive phase loops + run()
│   ├── constants.py               # Tunable knobs + HSV defaults (_clone_defaults)
│   ├── vision.py                  # Colour detection → base-frame resolution
│   ├── motion.py                  # Geometry planners + robot-facing move primitives
│   ├── ui_display.py              # Camera / OpenCV UI / calibration loaders / keys
│   ├── fk_functions.py            # Forward kinematics (shared from Lab 1)
│   ├── ik_solver.py               # Numerical inverse kinematics (port of Lab 2)
│   └── robot_io.py                # Thin xArm SDK wrapper (connect, move, gripper)
├── data/                          # Calibration artefacts (.npy) loaded by ui_display
│   ├── camera_matrix.npy          #   K (intrinsics)
│   ├── dist_coeffs.npy            #   distortion coefficients
│   ├── calibration_results.npy    #   zeta_nominal + T_ee_cam_joint (hand-eye)
│   └── board_config.json          #   ArUco board spec used during Lab 1 calibration
├── skeletons/                     # Student starting point — same modules, bodies stubbed
│   └── *.py                       #   each function raises NotImplementedError("TODO: …")
├── reference/                     # REFERENCE ONLY — not imported by the active code
│   └── python_cube_nav/           #   proven white-block navigator (source of Phase C port)
│       ├── run_lab3.py            #     the original monocular-nav demo
│       ├── navigator.py           #     source of the fused plan_step (Phase C port)
│       ├── cube_detector.py       #     luminance-extreme detector (prior approach)
│       ├── ik_solver.py           #     duplicate of src/ik_solver.py (kept for reference)
│       └── fk_functions.py        #     duplicate of src/fk_functions.py (kept for reference)
├── docs/
│   └── lab3_instructions.md       # This file
└── commit-log.md                  # Version history
```

To run: `cd Lab_3/src && python3 pick_and_place.py`

---

## Safety Notes

- You jog the arm manually during FREE MOVE — keep bystanders clear.
- Every autonomous motion is one small, keygated step; keep your hand near the emergency stop.
- Use the phase-abort key (`[a]`, or `[x]` inside Phase E) to stop stepping at any moment, and `[q]` to quit.
- Ensure both blocks sit on a stable, flat surface and cannot be knocked over.
- The approach brings the cup close to the block — stand ready to E-stop if collision risk appears.
- On any failure the arm **holds position** (it does not home) so you can recover by hand; only a fully successful run homes automatically.
- To slow the approach further, reduce `DESCENT_STEP_MM` and/or `FINE_Z_STEP_MM` in `constants.py`.
