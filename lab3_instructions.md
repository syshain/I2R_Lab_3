# LAB 3 — Monocular Vision-Guided Navigation

**Module:** Robotics for Mechanical Engineers (Undergraduate / MSc)
**Robot Platform:** UFACTORY Lite 6 (6-DOF articulated arm, wrist-mounted camera)
**Prerequisites:** Lab 1 (Forward Kinematics), Lab 2 (IK & Calibration)
**Duration:** 3 hours
**Software:** Python 3.12+, OpenCV, NumPy, SciPy, xArm Python SDK

---

## Learning Objectives

By the end of this lab you will be able to:

1. Detect known objects in monocular images using colour-space segmentation and contour analysis.
2. Infer depth from a single image when object physical dimensions are known, using the pinhole camera model.
3. Drive a robot interactively from live visual feedback, converting an image-plane error into a base-frame motion command.
4. Solve inverse kinematics for a desired end-effector pose (reusing the Lab 2 method) and command the servos safely through the xArm SDK.
5. Integrate calibrated DH parameters and hand-eye transform (from Lab 2) into a vision-guided task.

---

## Problem Statement

The robot's end-effector carries both a camera and a gripper (the gripper is not needed for this task). On the work surface sit two cubical blocks, each with side length **s = 50 mm**. Both are 3D-printed, so neither is pure white nor pure black; one prints in a light colour and the other in a dark colour. We call them the **light block** and the **dark block**.

Your task has two phases:

**Phase 1 — Manual positioning:** You position the arm by hand (teach/jog mode) so that *both* blocks are visible in the wrist-camera view. There is no automatic scanning — placing the blocks in view is your job, which keeps the number of moving parts small and puts a human in the safety loop before any autonomous motion begins.

**Phase 2 — Interactive navigation:** From the keyboard, step the end-effector a fixed amount at a time toward the **light** block, guided entirely by real-time monocular vision. Each keypress computes where the light block is in the image, converts that into a physical displacement, solves IK for the new end-effector pose, and commands the servos. The **dark** block is detected and shown as an operator warning but does not steer the path automatically. Stop once the end-effector is within a contact distance of the light block.

**The Challenge:** A monocular camera provides only 2D pixel coordinates. Depth is lost. However, because the cube side length is known (50 mm), you can recover approximate depth from the apparent size of the cube in the image using the pinhole camera model. This is the core insight of the lab.

---

## Part A — Monocular Depth Estimation Theory

### Pinhole Camera Model

A calibrated pinhole camera projects a 3D point **P** = [X, Y, Z]ᵀ in the camera frame onto the image plane at pixel coordinates (u, v):

$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

where fx, fy are the focal lengths in pixels and (cx, cy) is the principal point. These five parameters form the camera intrinsic matrix K, which you calibrated in Lab 2:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### Recovering Depth from Known Size

If an object of known physical width S (in metres) subtends w pixels in the image, and the object face is approximately perpendicular to the optical axis, then:

$$Z \approx \frac{f_x \cdot S}{w}$$

This follows directly from similar triangles in the pinhole model. The approximation degrades when the object is tilted relative to the camera (perspective foreshortening makes it appear narrower, overestimating depth), but it is sufficient for coarse navigation.

**Task A1:** Given your camera intrinsics (fx ≈ 1649 px) and a 50 mm cube, calculate the expected pixel width of the cube at distances of 300 mm, 500 mm, and 800 mm. At what distance does the cube become too small to detect reliably (say, fewer than 20 pixels wide)?

**Task A2:** Derive the depth uncertainty σz as a function of pixel measurement uncertainty σw. If σw ≈ 2 pixels (one-pixel edge localization error on each side), what is σz at Z = 500 mm? Express your answer in millimetres. Hint: propagate through z = f·S/w using the chain rule.

---

## Part B — Cube Detection Pipeline

### Why not just look for "white" and "black"?

Because the blocks are 3D-printed, their surface colours are muted and shift with the room lighting — the "light" block may read as off-white or pale grey, the "dark" one as charcoal rather than pure black. A fixed HSV rule like *low saturation AND high value* assumes a specific colour and breaks the moment the print filament or the lamp changes. So instead of hunting for two named colours, we work on the **grayscale luminance** channel and simply pick out the two extremes that are present:

- The **light block** is the brightest compact, roughly-square region in view.
- The **dark block** is the darkest such region.

This sidesteps the classic failure of a single global threshold. If you draw one dividing line between "block" and "background", it can only separate *one* block from the table at a time; whenever the table's brightness happens to fall *between* the two blocks, one of them gets absorbed into the background and vanishes. Selecting the intensity extremes independently avoids that — the light block wins the "brightest" contest and the dark block wins the "darkest" contest regardless of what the background measures as.

### Code Overview

The detection pipeline is implemented in `python_cube_nav/cube_detector.py`:

| Function | Purpose |
|----------|---------|
| `detect_cubes(frame, K, dist)` | Main entry: undistorts the frame, finds the light and dark candidates, deduplicates, returns up to two detections |
| `_candidate_regions(gray)` | Sweeps a set of grayscale thresholds to enumerate every distinct intensity region (a solid block has no internal edge, so we must binarise before contourcing), recording each region's mean luminance |
| `_find_extreme_block(gray, K, want, label)` | Picks the single brightest (`want='bright'`) or darkest (`want='dark')` valid region among the candidates |
| `_estimate_depth_from_size(w_px, h_px, fx)` | Applies Z = f·S/w to estimate camera-to-cube distance |
| `_compute_confidence(contour, bbox)` | Heuristic score based on fill ratio and circularity |
| `_dedupe_same_object(light_det, dark_det)` | If the brightest and darkest winners overlap heavily they are the same physical block seen twice — keep the stronger |

Two guards keep it sane: blobs covering more than `MAX_AREA_FRAC` (= 0.5) of the frame are ignored (that is the table/background, not a block), and each class keeps only its single best candidate, so exactly one light and one dark block are reported.

Each detection is returned as a dictionary containing:
- `colour`: 'light' or 'dark'
- `center_px`: (u, v) pixel centroid
- `bbox_px`: (x, y, w, h) bounding box
- `depth_m`: estimated distance in metres
- `confidence`: quality score [0, 1]
- `mean_lum`: mean grayscale value inside the region

### Task B1 — Run the Demo Mode

Test the detection pipeline without hardware:

```bash
cd Lab_3/python_cube_nav
python run_lab3.py --demo
```

This creates a synthetic 640×480 grey image with two painted rectangles (a light one and a dark one, simulating the printed blocks) and runs the full detector. Open the saved visualization at `Lab_3/Data/demo_detection.png` and verify:
- Both blocks are detected and labelled 'light' / 'dark'
- Depth estimates are reasonable (compare with your Task A1 calculations)
- Confidence scores are above the 0.2 threshold

### Task B2 — Analyse Detection Robustness

Open `cube_detector.py` and examine the design choices:

1. **Extreme selection vs. global threshold:** `_find_extreme_block` picks the single brightest (or darkest) region rather than drawing one threshold between "block" and "background". Construct a mental picture where the table brightness lies *between* the two blocks. Explain why a single Otsu-style cut would lose one block in that case, while extreme selection keeps both.

2. **Maximum-area guard:** Blobs covering more than `MAX_AREA_FRAC = 0.5` of the frame are discarded. What would happen to the detector if you removed this guard and pointed the camera at a mostly-uniform wall?

3. **Why binarise before contourcing?** A solid block sitting on a flat background has no internal edge, so running `cv2.findContours` directly on the grayscale finds only the frame border. `_candidate_regions` sweeps a set of thresholds to create binary regions first. What would the detector return if you skipped that step and contoured the raw image?

4. **Same-object deduplication:** `_dedupe_same_object` drops the weaker of the two winners if their bounding boxes overlap heavily (IoU > 0.3). Describe a lighting situation where the brightest and darkest candidates could actually be the *same* physical block (hint: a shaded corner versus a lit face), and why merging them matters.

Write a short paragraph (3–4 sentences) summarising the trade-off between detection sensitivity (finding real blocks under varied lighting) and specificity (rejecting the table, shadows, and other clutter) in this pipeline.

---

## Part C — Manual Positioning & Interactive Control

### Why no automatic scanning?

Earlier drafts had the robot auto-sweep its base joint to search for the blocks. That adds several failure modes (cable strain, unattended motion, an extra control loop to debug) for little benefit in a teaching setting. Instead, **you** place the arm so both blocks are visible, and then drive the approach yourself one fixed step at a time. A human stays in the safety loop for every single motion.

### Running the program

```bash
cd Lab_3/python_cube_nav
python run_lab3.py
```

On startup the program connects to the robot, enables motion, and opens the wrist-camera window (wrapped in a try/except, so a disconnected camera prints an error and exits cleanly instead of crashing). It then runs two phases:

**Phase 1 — Manual positioning.** Jog the arm with the teach pendant or the UFactory web interface (`http://<IP>:18333`) until both blocks are clearly in the camera view. When they are, click the window and press **`g`** to begin. Press **`q`** to cancel.

**Phase 2 — Interactive navigation.** Drive the approach with the keyboard:

| Key | Action |
|-----|--------|
| `w` | Take one fixed step (~10 mm) toward the light block |
| `a` | Abort navigation (stop stepping) |
| `r` | Reset the arm to home position |
| `q` | Quit (goes home and disconnects) |

Each `w` press: detects the light block, computes a fixed-size displacement toward it, solves IK for the resulting end-effector pose, checks the joint limits, and commands the servos (blocking until the move finishes). If the light block drops out of view, the step is refused and the banner tells you to reposition. The loop also halts automatically once the light block is within the contact distance.

### Task C1 — Understand the Interaction Model

1. Why does putting a human in the loop for every step make this safer than a fully autonomous approach, even though it is slower? What can the operator catch that the software cannot?

 2. The program refuses to step when the light block is not detected. Why is "no detection → do nothing" a safer default than "no detection → keep going straight"?

3. What is the purpose of the `r` (reset to home) key, and why is it useful mid-task?

---

## Part D — Vision-Guided Navigation

### Approach Strategy

For each `w` keypress, the navigator performs one closed-loop step toward the light block:

1. **Detect** the light (and dark) block(s) in the current frame.
2. **Centre** the light block by computing its lateral offset from the image principal point.
3. **Convert** that pixel offset into a physical displacement using the current depth estimate and the camera intrinsics, and add a fixed forward component along the optical axis.
4. **Rotate** the camera-frame displacement into the base frame using FK and the calibrated T_ee_cam.
5. **Solve IK** for the new end-effector pose (orientation preserved) using the Python solver from `ik_solver.py` — the Lab 2 method.
6. **Check joint limits**, then **command the servos** and wait for the move to complete.

The dark block is drawn on screen as a warning but does not divert the path — avoiding it is the operator's call.

### Inverse Kinematics — the same routine as Lab 2

Inverse kinematics here is solved **in Python**, and it is a direct port of the very routine you implemented in Lab 2 (`MATLAB_IK/ik_solve.m` + `ik_objective.m`). Nothing about the method has changed — only the language. Concretely, `python_cube_nav/ik_solver.py` reuses three things you already met:

| Piece | Lab 2 (where you saw it) | Lab 3 (`ik_solver.py`) |
|-------|--------------------------|------------------------|
| Error vector `[position; orientation]` | `ik_objective.m` | `ik_objective()` — identical formula |
| Orientation error = axis-angle of `R_cur·R_tgtᵀ` | `ik_objective.m` (the `acos((trace−1)/2)` block) | `rotation_log_map()` — the *same* log map used in Lab 2's calibration (`so3.py`) |
| Levenberg–Marquardt solve with joint limits | `ik_solve.m` → `lsqnonlin` | `solve_ik()` → `scipy.optimize.least_squares` (bounded LM) |

So the connection is deliberate: **the inverse kinematics that drives the robot in Lab 3 is the one you derived and tested by hand in Lab 2.** If your Lab 2 MATLAB solver converged to a given target from a given initial guess, the Python solver will converge to the same solution branch from the same seed.

The solver minimises the 6-vector residual `e(q) = [T(q)·p − p_target ; log(R(q)·R_targetᵀ)]` over the six joint angles, clamped to the UFACTORY Lite 6 limits, using Levenberg–Marquardt. It is seeded from the arm's current joint angles each step, so it stays on the same solution branch rather than jumping to a different elbow/wrist configuration.

Two practical notes:

- Because IK is solved locally, the program does **not** depend on the controller exposing an IK service. (The xArm SDK v1.17.3 does have `get_inverse_kinematics(pose) -> (code, angles)`, but there is no `wait_until_ready()` / `inverse_kinematics_command`; motion completion is always handled by `set_servo_angle(..., wait=True)`, which blocks until the move finishes.)
- Setting `USE_CONTROLLER_IK = True` in `navigator.py` turns on a cross-check: after each Python-IK solution it also queries the controller's own IK and prints the maximum joint difference. A small number confirms your Python solver agrees with the hardware; a large number flags a sign/unit/convention mismatch worth investigating.

### Coordinate Transform Chain

At each step, the desired EE motion δp_camera (from image analysis) must be expressed in the base frame for IK:

$$\delta\mathbf{p}_{base} = \mathbf{R}_{base}^{cam} \cdot \delta\mathbf{p}_{cam}$$

where R_base^cam = FK(q) · T_ee_cam gives the camera orientation in the base frame. The target base-frame position is p_target = p_current + δp_base.

### Task D1 — Study the Navigator and the IK Port

Read `python_cube_nav/navigator.py` and `python_cube_nav/ik_solver.py` and answer:

1. **Pixel-to-lateral conversion:** the lines computing `angle_x`, `angle_y`, `lateral_x`, `lateral_y`. Verify the formula lateral ≈ Z · tan(pixel_offset / focal_length). Why is `np.tan()` used instead of the small-angle approximation?

2. **Fixed step size:** `APPROACH_STEP_MM = 10` caps how far the arm advances per keypress. Why is a bounded, operator-paced step safer than a velocity-controlled continuous approach?

3. **Contact condition:** the loop stops when `depth_mm ≤ CONTACT_DISTANCE_MM + CUBE_SIDE_M * 1000`. Why is the cube side length added to the contact threshold?

4. **The Lab 2 connection:** Open `ik_objective()` in `ik_solver.py` beside `MATLAB_IK/ik_objective.m` from Lab 2. Confirm they compute the same 6-vector error. Which line in each corresponds to the orientation error, and why is that term a log-map (axis-angle) rather than a subtraction of Euler angles?

5. **Seeding and multiplicity:** `solve_ik` is called with the arm's *current* joint angles as the initial guess. Given what you learned about solution multiplicity in Lab 2 (Task A2), why does seeding from the current pose keep the arm on one solution branch instead of flipping to a different elbow/wrist configuration mid-approach?

### Task D2 — Simulate One Step

Without running on hardware, manually trace one navigation step for this scenario:
- Light block 200 px right of the image centre, depth = 400 mm, fx = fy = 1649 px
- End-effector currently at home, orientation unchanged

Compute:
- The lateral displacement (show your arithmetic)
- The forward displacement (given APPROACH_STEP_MM = 10 mm)
- The resulting target base-frame position (qualitatively: which way does it move?)

### Task D3 — Live Navigation (Hardware Required)

With both blocks placed on the work surface, run the full program and drive the approach to the light block:

```bash
python run_lab3.py
```

Position the arm (Phase 1), then step toward the light block with `w` (Phase 2). Observe:
- Does the light block stay centred as you step, or does it drift off-centre (indicating a hand-eye or sign error)?
- How many steps did the approach take?
- Did the dark block ever come close to the path? How did you respond?
- What was the final depth reading at contact?

Examine the saved navigation log at `Lab_3/Data/navigation_log.npy`:

```python
import numpy as np
log = np.load('Lab_3/Data/navigation_log.npy', allow_pickle=True).tolist()
for entry in log[-5:]:
    print(entry)
```

---

## Part E — Error Analysis and Discussion

### Sources of Navigation Error

Monocular navigation accumulates errors from multiple sources:

1. **Depth estimation error:** From Task A2, σz grows quadratically with distance. At 600 mm, a 2-pixel edge error translates to roughly ___ mm depth uncertainty.

2. **Perspective foreshortening:** As the robot approaches at an angle, the cube face tilts relative to the camera. The apparent width shrinks, causing the depth estimator to overestimate distance. The robot keeps approaching further than necessary.

3. **Hand-eye calibration residual:** Even after Lab 2 calibration, T_ee_cam has finite error (~4 mm position, ~1° orientation). This propagates into every coordinate transform.

4. **Detection jitter:** Frame-to-frame variation in centroid and bounding box due to lighting changes, compression artefacts, and motion blur.

### Task E1 — Quantify Total Uncertainty

Using your calibration results from Lab 2 and the depth uncertainty from Task A2, compute the total expected positioning uncertainty at the contact point. Combine:
- Depth estimation σz at the average approach distance
- Lateral estimation σlat (similar derivation to σz, using image-centre offset)
- Hand-eye calibration residual (use the final spread from Stage 2)

Report the combined uncertainty as a 3D ellipsoid (σx, σy, σz).

### Task E2 — Improvements

Propose three concrete improvements to this pipeline that would reduce navigation error. For each, explain:
- What error source it addresses
- Roughly how much improvement you expect (order of magnitude)
- Implementation complexity (low/medium/high)

Examples to consider: stereo camera, structured light, multi-view triangulation during approach, active pose adjustment to face the cube head-on before measuring depth, Kalman filtering of depth estimates across frames.

---

## Deliverables — Lab Report

Submit a report (PDF or Word) containing:

1. **Title page** — Name, student number, date, lab partner(s).
2. **Part A** — Answers to Tasks A1 and A2 with worked calculations.
3. **Part B** — Detection robustness analysis (Task B2 paragraph). Include the demo visualization screenshot.
4. **Part C** — Answers to the interaction-model questions (Task C1): why human-in-the-loop is safer, and why "no detection → do nothing" is the right default.
5. **Part D** — Answers to Task D1 (including the Lab 2 IK-port comparison and the seeding/multiplicity question). Manual trace from Task D2. Live navigation results from Task D3: step count, final depth, trajectory quality assessment. Last 5 entries from the navigation log.
6. **Part E** — Combined uncertainty calculation (Task E1). Three proposed improvements with justification (Task E2).
7. **Synthesis** (½ page) — Compare monocular navigation (Lab 3) with the calibrated open-loop approach from Labs 1–2. When is vision guidance essential, and when is calibrated kinematics sufficient? What fundamental limitation of monocular vision persists even with perfect calibration?

---

## File Structure

```
Lab_3/
├── python_cube_nav/
│   ├── run_lab3.py              # Main entry point (--demo or live), camera window + key loop
│   ├── cube_detector.py         # Colour segmentation + depth inference
│   ├── navigator.py             # Interactive single-step planner + servo helpers
│   ├── ik_solver.py             # Python IK — port of Lab 2's MATLAB ik_solve/ik_objective
│   └── fk_functions.py          # Forward kinematics (shared from Lab 2)
├── Data/
│   ├── demo_detection.png       # Generated by --demo mode
│   └── navigation_log.npy       # Generated by live run
└── lab3_instructions.md         # This file

---

## Safety Notes

- You jog the arm manually in Phase 1 — keep bystanders clear while doing so.
- Every autonomous motion in Phase 2 is one small step at a time; keep your hand near the emergency stop and be ready to use it.
- Use the `a` key to abort stepping at any moment, and `r` to return to home.
- Ensure the blocks are placed on a stable, flat surface and cannot be knocked over by the robot.
- The approach brings the EE close to the light block — stand ready to E-stop if collision risk appears.
- Reduce `APPROACH_STEP_MM` in `navigator.py` to 5 mm if you want slower, safer approach motions.
