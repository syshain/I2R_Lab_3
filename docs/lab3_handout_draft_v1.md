# ME4 Robotic Manipulators and Automation

# LABORATORY 3

## The Hand and the Eye: Tool Calibration and Vision-Guided Pick and Place

*Finish the eye-to-hand calibration with a touch measurement, find two coloured blocks with the wrist camera, plan a path to them through via points, and stack one block on the other — with you in the loop for every move that could touch something.*

**STUDENT HANDOUT**

| | |
|---|---|
| Academic year | 2026–27 |
| Version | Draft 1 — [date] |
| Author | Prof Ferdinando Rodriguez y Baena |
| Contact | Microsoft Teams or 020 7594 7046 |
| Module leader | Nicolas Cinosi |
| Runs in | Weeks 17 and 18, Spring 2027, MTX Laboratory |

---

## 1 The laboratory at a glance

### 1.1 What you will do

In Laboratory 2 you calibrated the kinematics of the arm and the transform from flange to camera. That left one transform missing: the camera and the suction cup ride on the same flange, but nothing yet tells the gripper where the camera is, or the camera where the gripper is. This laboratory closes that last loop and then uses the whole calibrated chain to manipulate real objects.

The session has four tasks:

1. **Task 1** — measure the flange-to-cup offset with one camera capture and several touches: the camera looks at the ArUco artifact once, putting its top face in the base frame through the calibrated chain; you then move the arm to touch that face several times by hand, and the agreement of the per-touch solutions recovers the one unknown tool translation. From it you form T_G_C and see the whole chain report the artifact relative to the tool.
2. **Task 2** — segment the red and green blocks by HSV colour, tuning thresholds live against the actual bench lighting.
3. **Task 3** — estimate a block's position and yaw in one shot from its known 25 mm size, then plan and fly a via-point transit that stops the cup 50 mm back along the cup axis from the estimate. You try linear, cubic and quintic interpolation and compare them.
4. **Task 4** — stack the blocks: green to the stack base, red on top of green, every final approach jogged by hand under key control, with the vacuum engaging only when you press SPACE.

**You work in pairs, at one bench, for one three-hour session.** Tasks 1 and 2 build on each other, and Task 4 uses everything before it, so the order is fixed.

### 1.2 Software

| Software | Used for |
|---|---|
| Python 3 (station PC) | All scripts run in the `Lab_3/src` folder of your downloaded repository |
| OpenCV 4 | ArUco detection, colour segmentation, camera windows |
| NumPy / SciPy | Linear algebra, least squares, the inverse kinematics solver from Laboratory 2 |
| UFactory Studio (browser, port 18333) | Robot dashboard: enabling, Manual Mode, parking |
| xArm Python SDK | Every hardware call, wrapped in `robot_io.py` |

You need the same virtual environment you built for Laboratory 2 (Section 1.3). Nothing in this laboratory installs anything new.

### 1.3 Setting up your station

Work through these steps in order at the start of the session.

(a) Log in to the station PC and open UFactory Studio at `localhost:18333` in the browser. Confirm the robot is *enabled* and put it into Manual Mode, exactly as in Laboratory 2. Manual Mode is what lets your key presses move the arm in Task 4.

(b) Check the robot address in `src/robot_io.py` (`ROBOT_IP`) matches the bench you are sitting at. It is the same address you used in Laboratory 2.

(c) Open a terminal in your extracted `Lab_3` folder and activate the environment:

```
python -m venv .venv          (only if you have not already, in the Lab 2 folder — reuse that one)
.venv\Scripts\activate        (Windows)
source .venv/bin/activate     (Linux/macOS)
pip install -r requirements.txt   (only on first use)
```

(d) Confirm the calibration files from Laboratory 2 are present in `data/`: `camera_matrix.npy`, `dist_coeffs.npy` and `calibration_results.npy`. Task 1 loads `calibration_results.npy`, which holds the calibrated DH parameters (zeta_stage2) and the hand-eye transform T₆C fitted together with them in Laboratory 2's final stage — the two were solved against each other, so they are used as a pair. **If your pair calibrated at a different bench, ask your GTA for a copy of that station's files** — a T₆C from another camera mount is worse than none.

(e) Confirm the ArUco artifact and the two wooden blocks (one red, one green, 25 mm cubes) are at your bench. The artifact stays where Laboratory 2 left it for now; Task 1 will tell you what to do with it.

(f) Open the `src/` folder in Visual Studio Code. You will edit the files there.

> **If the robot stops and reports a joint out of range** — a planned motion has been refused, or a key-gated jog hit a limit. The scripts check the joint limits of Appendix B before every move, so this almost always means an IK solution was attempted from a pose near a limit. Clear the error in UFactory Studio, then re-run; the scripts never move before solving IK for the whole path.

> **If the robot will not connect** — check the address in step (b), check the arm is enabled in UFactory Studio, and check nobody else is holding a connection. `connect_arm()` never resets the arm, so a failed connect leaves the pose untouched; just fix the cause and try again.

---

## 2 Learning outcomes

This laboratory develops the following learning outcomes, which are assessed through the Möbius questions and your answers in this handout.

| Code | Outcome, in brief |
|---|---|
| LO1 | Close an eye-to-hand calibration loop with a physical touch measurement, and reason about which transforms the data can and cannot identify |
| LO2 | Segment real objects by colour under uncontrolled lighting, and recover metric depth from a single image using known size |
| LO3 | Plan and execute task-space paths through via points, and compare interpolation profiles for smoothness, safety and speed |
| LO4 | Integrate perception, calibration and planning into a complete pick-and-place task with a human in the safety loop |

---

## 3 Safety

> **The arm moves under your keys in this laboratory.** In Tasks 3 and 4 every approach to a block is *key-gated*: each key press is one small step, and nothing moves unless you press. Keep a hand near the emergency stop whenever the arm is moving, planned or jogged. The one-shot estimate of Section 5.3 is expected to be wrong by centimetres; that is why the arm never descends on it — it stops 50 mm back along the cup axis from the estimate (with a level wrist, 50 mm above it) and hands control to you.

- The vacuum gripper is powerful enough to lift a 25 mm wooden block and gentle enough to be safe to touch, but never let it close on a finger: engage it only with the cup flat on a block, which is exactly what the SPACE confirmation in Task 4 asks you to verify first.
- The 50 mm stand-off is not a suggestion. `transit_to_hover()` refuses to plan closer to the block than that along the cup axis, and every planned move is flown at the conservative transit speed regardless.
- If anything looks wrong — a block shifts, the arm heads somewhere you did not intend, a jog does not stop — press **[q]**. The scripts release the vacuum, stop moving, and leave the arm where it is so you can recover. Tell your GTA.
- At the end of the session, park the robot with UFactory Studio before powering off (Section 7.3). This is compulsory.

---

## 4 Before the lab

### 4.1 Preparation

You cannot complete this laboratory without the following, all from Laboratory 2. Check them before the session; there is no time to redo them during it.

- A working inverse-kinematics solver. Laboratory 3 imports the one you wrote (`ik_solver.py`, unchanged from Laboratory 2).
- The camera intrinsics (`camera_matrix.npy`, `dist_coeffs.npy`) from Laboratory 1.
- The joint-based hand-eye transform `T_ee_cam` and the calibrated DH parameters fitted together with it (Stage 2), saved together in `data/calibration_results.npy` by Laboratory 2's `run_calibration.py`.

Read Sections 5.1 to 5.4 before you start. They contain the four ideas the tasks apply, and the Möbius questions assume you have.

### 4.2 Syllabus references

The tasks end with "Look up" lines. These codes are what they mean; each is a short note in `docs/` of the repository.

| Code | Reference note |
|---|---|
| S1 | Frames and homogeneous transformations in the manipulation chain (refresher on Laboratory 1) |
| S2 | The pinhole camera model, intrinsics and distortion |
| S3 | Least squares: normal equations, the pseudo-inverse, and when a solution is not unique |
| S4 | Colour spaces and HSV thresholding, including the hue wraparound |
| S5 | Depth from known size: similar triangles and the pixel-width formula |
| S6 | Trajectory interpolation: blending functions and orientation interpolation on SO(3) |
| S7 | Observability and gauge freedoms in calibration problems |
| S8 | End-effectors: suction, sealing and why the cup's pose matters |

---

## 5 Background

### 5.1 The hand and the eye

Figure 1 shows the chain of transforms this laboratory works with. The base frame {0} is fixed in the robot. Joint readings give the flange pose T₀₆ through forward kinematics. Laboratory 2's calibration gave the flange-to-camera transform T₆C. This laboratory adds the flange-to-cup transform T₆G — one number, the tool offset d along the flange axis — and the camera-to-gripper transform T_G_C = T₆G⁻¹ T₆C that ties the two wrist tools together.

The wrist camera is *eye-in-hand*: it moves with the arm, so everything it measures is known only relative to the flange. When the camera looks at the artifact, the measurement chain is

  T₀_W = T₀₆ · T₆C · T_C_W  (camera route),  or  T₀_W = T₀₆ · T₆G · T_G_W (gripper route)

and here is the difficulty: the camera never sees the cup, and the cup never touches anything the camera can independently locate *in the same measurement*. If you multiply the camera route by any rigid motion that moves the *whole world* — the artifact, the table, everything — both routes change by the same amount and no camera observation can tell the difference. This is a *gauge freedom* (S7): the pair (T₆G, T_G_C) is observable from camera data only through their product T₆C. Laboratory 2 already fixed T₆C, so Task 1 does not re-solve it. Instead it pins T₆G with a measurement of a different kind entirely: *touch*.

### 5.2 Touch calibration of the tool offset

The measurement takes one camera capture and several touches, and no stage involves a least-squares solve: each touch is one linear equation in the single unknown d.

**Capture 1: the camera look.** With the artifact stationary, one frame in which the camera sees it gives the artifact-to-camera transform T_C_W. The chain calibrated in Laboratory 2 turns that into the artifact pose in the base frame, using the joint readings at that moment for T₀₆:

  T₀_A = T₀₆ · T₆C · T_C_W

The artifact model declares one face the *top face* — the flat face carrying marker 0 — and its centre is simply the average of that marker's four corners, p_top in the artifact frame. Before the arm moves at all, the camera has already located, in the base frame, the very point the cup is about to touch.

**Capture 2: the touches.** You now move the arm by hand and bring the cup down onto the top face several times — at least three, and about six is a good number. No touch has to be vertical or at the face centre: a hand-guided arm cannot reliably achieve either, and the formulation below does not ask for it. The top face is a *plane*. From the CAPTURE 1 pose T₀_A, the top-face marker's four corners give the face's outward unit normal n and a point p_face on it, both in the base frame. At touch i the joint readings give the flange pose (R_i, p_i) through forward kinematics, and the cup tip sits a fixed distance d along the flange +z axis (Figure 1) — the camera's optical axis, and so the cup, lies almost exactly along flange +z. Wherever on the face the tip lands, and at whatever wrist tilt, two things are true at that instant: the contact point lies in the face plane, and the tip position is t_i + d·z_i with t_i = p_i and z_i the flange +z axis in the base frame. Each touch is therefore one linear equation in the single unknown d:

  n · (t_i + d·z_i) = n · p_face,

solved per touch for its own d_i. The check is *consistency*, the same mean/spread reporting the other labs use: the script records every touch, shows the running mean, standard deviation and spread of the d_i, and warns above 5 mm. A touch nearly parallel to the face (|n · z_i| small) divides by a near-zero number, so its d_i is unreliable: the script flags such a touch as a *graze*, and 'r' removes the last touch ('c' clears all) so a bad touch can be dropped before ESC saves the set.

With d known, T₆G follows immediately (a pure translation of d along the flange +z axis), and T_G_C = T₆G⁻¹ T₆C completes the chain. The report prints a per-touch table — each touch's d, wrist tilt, distance from the face centre and flags — then d's mean, standard deviation and spread next to the nominal 75 mm for reference. It also prints T_G_A, the artifact pose as the gripper frame sees it at the moment of a touch: the transform that says where the object sits relative to the tool, which is what a grasp actually consumes. T_G_A is pose-dependent — each touch lands at its own in-plane point — so the report keeps the touch that landed closest to the face centre as the representative one. The consistency check is the check on all of this: if the per-touch d_i agree tightly, the touches really did place the tip on the flange axis in the plane the camera named, and d can be trusted.

### 5.3 Seeing the blocks: colour and known size

The blocks are found in three steps, all in `vision.py`.

**Colour.** Each frame is converted from BGR to HSV, and a pixel belongs to a colour if its H, S and V fall inside any of that colour's *bands* — ranges of each channel set in `lab_config.py` and adjustable live with the trackbars of Task 2. Two details matter. First, OpenCV hue runs 0 to 179, so red straddles the wraparound: it is *two* bands, one near 0 and one near 179, and a pixel is red if it is in either (S4). Second, the saturation floor is what rejects the bench: wood, shadows and the grey table all have low saturation, so a floor around 50–70 keeps them out of the mask.

The largest connected blob in the mask above a minimum area becomes the detection, reported as a bounding box and centroid. Blobs filling more than half the frame are rejected — that is usually the mask flooding.

**Depth.** One image, one camera, one number: a monocular camera cannot measure depth by itself. But the blocks are 25 mm cubes, and size is depth in disguise. If the block's top face is S = 25 mm wide and appears w pixels wide, the pinhole model (S2, S5) gives the distance along the optical axis directly by similar triangles:

  Z ≈ f_x · S / w.

This formula has a hidden assumption: it treats the silhouette as a fronto-parallel face of width w. Your camera views the block from the wrist, tilted by 15–35° in practice, and a tilted cube's silhouette is *longer* than its top face — the top face foreshortens to S·cos α while a side face of height S·sin α leans into view. Feeding that elongated width into Z ≈ f_x·S/w reads the block *closer* than it is: tens of millimetres at these angles, which is most of the 50 mm hover budget.

`vision.py` therefore does not patch the formula; it fits the pose. A block on the table has its top face at a *known* height (table surface plus one block side), so the only unknowns are the block's (x, y) and yaw. `fit_block_pose()` projects a hypothesised cube into the image, measures its silhouette exactly the way the detector measures the real one (minAreaRect centre, both extents, angle), and adjusts (x, y, yaw) until the two silhouettes agree — a perspective-correct inversion of the forward model, seeded at the naive Z ≈ f_x·S/w estimate. When the fit cannot explain the contour (a cluttered or broken mask) it falls back to the naive estimate and says so on the console.

**Yaw.** `estimate_block_yaw()` reads the block's rotation about the vertical from the minimum-area rectangle fitted to the top-face blob: OpenCV's `minAreaRect` returns an angle α for a rectangle of width w and height h, and the block's yaw is ψ = α − θ, where θ picks the rectangle edge that corresponds to the block's front edge. This reading is exact looking straight down and drifts as the view tilts, which is why `fit_block_pose()` refines it: the one-shot estimate reports the fitted yaw. Either way, yaw from a square face is only defined modulo 90° — the detector cannot tell an edge from the perpendicular edge. The cup is rotationally symmetric, so the yaw governs nothing in the motion — it is a deliverable, because in Task 4 you compare it against where the block actually ended up.

### 5.4 Planning through via points

A planned transit in this laboratory is a sequence of *via points* you choose, joined by an interpolation profile you choose, ending at a hover 50 mm back along the cup axis from the one-shot estimate. The three profiles in `planners.py` blend the position linearly in a parameter s ∈ [0, 1] with three different blending functions (S6):

| Profile | Blend b(s) | Character |
|---|---|---|
| linear | s | constant velocity; corners at every via point |
| cubic | 3s² − 2s³ | zero velocity at both ends of each segment; still corner-ish at vias |
| quintic | 10s³ − 15s⁴ + 6s⁵ | zero velocity *and* zero acceleration at both ends; smoothest, slowest |

Orientation is interpolated on SO(3), not on angles: the relative rotation R₁R₀ᵀ is mapped to its rotation vector ω (S1), a fraction s of ω is mapped back with the exponential map, and the result premultiplies R₀. Interpolating roll-pitch-yaw numbers directly does not follow a rotation and can twist the long way round.

For each pose on the path the solver computes inverse kinematics *before the arm moves* — the whole path is solved first, and if any pose fails, nothing has moved. This is deliberate: an arm that discovers an unreachable pose halfway through a transit is an arm hanging over your blocks.

Why via points at all? Because the straight line from where you are to above the block may pass through the artifact, the other block, or the stack. Choosing vias that route around obstacles and come down onto the hover point from above is the planning skill this task teaches, and Task 4 asks you to justify your choices.

---

## 6 Tasks

### 6.1 Task 1: The touch calibration

**CORE | 45 min | LO1 | Module outcome 1, 3 | ILO 1.1, 6.1, 7.1 | Lecture 13 (week 17) | Chapter 5 | Look up S1, S7**

**Purpose.** To measure the flange-to-cup offset d with one camera capture and several touches, form T_G_C from it, and see the completed eye-to-hand chain report where the artifact sits relative to the tool.

You will complete four functions in `src/handeye_gripper.py`: `top_face_centre()`, `artifact_pose_in_base()`, `top_face_plane()` and `touch_d_estimate()`. The two capture loops (`capture_artifact_pose`, `capture_touch`) and the report (`report_and_save`) are provided — read them, because they show exactly how your functions are used. Everything the script prints is for your tables; the transforms are yours.

(a) **Open the calibration script and read it.** `handeye_gripper.py` runs one camera capture and one touch stage: CAPTURE 1 takes one camera look at the stationary artifact and forms T₀_A through the calibrated chain; CAPTURE 2 records a set of touches on the artifact's top face. Note the constants at the top of the file: which marker marks the top face, the minimum number of touches, the consistency warning and the graze threshold. What each is for is Section 5.2 and Question 1.1.

(b) **Complete `top_face_centre()`.** It returns the centre of the top face as the average of the top-face marker's four corners. The corner coordinates come from `board_config.json`, which — unlike Laboratory 2's re-centred corner model — uses the raw board frame, so **do not copy the corner handling from your Laboratory 2 detector**.

(c) **Complete `artifact_pose_in_base()`** and run CAPTURE 1. The function is one chain: forward kinematics from the joint readings, Laboratory 2's T₆C, and the detection's T_C_W. Lay the artifact flat with its marker-0 face up, keep it still, and press SPACE once the live window reports the artifact in view. While the artifact is in view, the live window also draws the artifact frame W's own axes from its origin — the centroid of all the artifact's marker corners, the same convention as Laboratory 2 (length `AXES_LENGTH_MM` in `lab_config.py`): check that x, y and z point where the handout's frame definition says they should before you trust T_C_W. SPACE freezes the frame: the pose is then checked on a static image that also marks, with a cross, the point the cup tip must later touch (the axes stay drawn there too). SPACE on the frozen frame accepts the capture ('r' retakes, 'q' aborts); the camera feed is not needed while you check. Record the reprojection error and T₀_A's position in Table 1.

> **If the detection is weak or the reprojection error is large** — the T₀_A you are saving here is what turns the touch into a measurement, so it is worth being sure of. Check that all five markers were seen, that the artifact did not move between the detection and the capture, and that the 15 px reprojection gate (as Laboratory 2) is not silently rejecting views. Re-run rather than continuing: CAPTURE 2 measures the touch against this pose.

(d) **Move the arm by hand for CAPTURE 2.** Put the arm into hand-guiding mode in UFactory Studio so you can move it freely. CAPTURE 2 shows no camera feed at all: no detection, no colour masks, just a status panel reporting how many touches are recorded and the running statistics of Section 5.2. The artifact will usually not be in view under the cup, and nothing needs to be. No touch has to be vertical or at the face centre — Section 5.2 explains why. Do not record yet.

(e) **Complete `top_face_plane()` and `touch_d_estimate()`, then record your touches.** Bring the cup down onto the top face — any point on the face, any wrist tilt — and press SPACE to record the touch. The script immediately prints the touch's d, wrist tilt and distance from the face centre, and flags a *graze* if the touch was too parallel to the face. Move somewhere else on the face, vary the wrist tilt, and touch again: aim for about six touches from a spread of positions and orientations ('r' removes the last touch, 'c' clears all). The panel shows the running mean, standard deviation and spread of the per-touch d values; if the spread is large, look for the odd touch out and remove it. ESC saves the set once at least the minimum number of touches are recorded. The arm is moved by hand throughout: this is a measurement of where the *cup* is, not a robot-motion task.

(f) **Read the report.** It prints a per-touch table — each touch's d, wrist tilt, distance from the face centre and flags — then d's mean, standard deviation and spread next to the nominal 75 mm for reference, and the translations of T_G_C and T_G_A — the artifact pose as the tool frame sees it at the moment of the representative touch (the one closest to the face centre). The result is saved to `data/gripper_calibration.npy` automatically: the same file every later task loads the tool offset from. Complete Table 1.

**Table 1: The touch calibration**

| Quantity | Value |
|---|---|
| Reprojection error of CAPTURE 1 (px) | ............ |
| Artifact position T₀_A (x, y, z mm) | ............ |
| Per-touch d values (mm, one per touch) | ............ |
| Nominal TOOL_OFFSET_MM, for reference (mm) | ............ |
| d standard deviation / spread (mm) | ............ |
| T_G_C translation (mm) | ............ |
| T_G_A translation (mm) | ............ |

**Question 1.1** Section 5.2 solves each touch separately for its own d_i and then asks the per-touch values to agree. Explain why a single touch's d_i can be badly wrong even when the code is correct, and what the agreement of several touches is testing that a single touch cannot test.

Response:

.....................................................................................................................................................................

.....................................................................................................................................................................

   **Möbius M1.1: answer now, on your own**
   Answer the question on what the touch agreement tests. Look up: S1.

**Question 1.2** Section 5.1 argues that camera data alone can never pin T₆G. Your touch measurement did. State precisely what new information the touch provides that a camera observation cannot, in one sentence.

Response:

.....................................................................................................................................................................

   **Möbius M1.2: answer now, on your own**
   Answer the question on gauge freedoms. Look up: S7.

**Question 1.3** Suppose the suction cup is knocked and its tip lengthens by 2 mm after calibration. Which numbers in Table 1 change, and which stay exactly the same? What does that tell you about what the touch residuals can and cannot validate?

Response:

.....................................................................................................................................................................

   **Möbius M1.3: answer now, on your own**
   Answer the question on what the residuals validate. Look up: S1.

> **✔ Checkpoint 1**
> Show your GTA Table 1 and your answers to Questions 1.1 to 1.3. **GTA initials: ____________**

### 6.2 Task 2: Find the blocks by colour

**CORE | 30 min | LO2 | Module outcome 3 | ILO 6.1, 6.2 | Lecture 13 (week 17) | Chapter 5 | Look up S2, S4**

**Purpose.** To segment the red and green blocks reliably under your bench's actual lighting, and to understand which channel is doing the work, before any motion depends on the detections.

You will complete `_hsv_color_mask()` and `_best_blob()` in `src/vision.py`. The live tuning interface (`tune_masks()` in `ui_display.py`) and the defaults in `lab_config.py` are provided.

(a) **Complete `_hsv_color_mask()`.** Convert the frame to HSV, build one binary mask per band of the colour (red has two, green one), and OR them together. Morphological opening and closing are applied for you after your function returns; what your function must get right is the per-band comparison and the combination.

(b) **Complete `_best_blob()`.** From the mask, find the connected components, discard those below the minimum area or above the maximum frame fraction, and return the bounding box, centroid, minimum-area rectangle *and raw contour* of the largest survivor. The pixel width is what Section 5.3's depth formula consumes, and the raw contour is what `fit_block_pose()` measures — think about which width you return for a block seen at an angle, because Task 3's accuracy depends on it.

(c) **Run the tuning session.** `python pick_and_place.py` opens with it: the main window shows the live feed with both blocks boxed, and the two mask windows show what each colour's mask currently sees, with a trackbar per threshold bound. Drag until **each mask highlights only its own block** — not the table, not shadows, not the other block, not the wood edge of the bench. `[r]` resets to defaults; `[Enter]` accepts and prints your final bands.

(d) **Stress the masks before accepting.** Slide the blocks around the bench, swap their positions, and shade the camera with your hand. A mask that works in one corner of the bench and fails in another will fail during Task 4, at the worst moment. Record your accepted bands in Table 2 and note where the masks are fragile.

**Table 2: Your accepted HSV bands (H in 0–179)**

| Colour | Band | H lo | H hi | S lo | S hi | V lo | V hi |
|---|---|---|---|---|---|---|---|
| red | 1 | ...... | ...... | ...... | ...... | ...... | ...... |
| red | 2 | ...... | ...... | ...... | ...... | ...... | ...... |
| green | 1 | ...... | ...... | ...... | ...... | ...... | ...... |

**Question 2.1** The saturation floor rejects most of the table and bench. Give one lighting change in the room that would make a correct-looking floor of 70 start failing, and say in which direction the floor should move to survive it.

Response:

.....................................................................................................................................................................

   **Möbius M2.1: answer now, on your own**
   Answer the question on HSV robustness. Look up: S4.

**Question 2.2** Red needs two hue bands and green needs one. Explain the wraparound that makes this necessary, and state the H value a third colour would need to straddle if you added a magenta block.

Response:

.....................................................................................................................................................................

   **Möbius M2.2: answer now, on your own**
   Answer the question on the hue wraparound. Look up: S4.

> **✔ Checkpoint 2**
> Show your GTA Table 2 and your tuned masks on the live feed. **GTA initials: ____________**

### 6.3 Task 3: The one-shot estimate and the planned transit

**CORE | 50 min | LO2, LO3 | Module outcome 1, 3, 4 | ILO 5.1, 5.2, 6.1, 7.1 | Lectures 13 and 14 (weeks 17, 18) | Chapter 5 | Look up S1, S2, S5, S6**

**Purpose.** To produce a metric estimate of a block's pose from a single frame using known size, then plan and fly a via-point transit to a hover 50 mm back along the cup axis from it, comparing all three interpolation profiles on the same path.

You will complete `one_shot_estimate()`, `fit_block_pose()` and `estimate_block_yaw()` in `src/vision.py`, and the blending and composition functions in `src/planners.py` (`_blend_linear`, `_blend_cubic`, `_blend_quintic`, `interpolate_pose`, `plan_via_path`). The interactive pipeline in `pick_and_place.py` — profile choice, via prompts, IK-before-motion execution — is provided; read `execute_path()` before flying anything, because its fail-before-move discipline is the safety argument of Question 3.2.

(a) **Complete the pose estimate.** In `one_shot_estimate()`, first unproject the blob centroid to the base frame at the known-size depth Z = f_x·S/w of Section 5.3, using the camera pose chain T_cam = T_flange · T_ee_cam. The result estimates the block's **top-face centre** — think about why the top face and not the centroid of the whole cube. Then pass that baseline into `fit_block_pose()`, which corrects it for the camera's tilt onto the block: complete it following its docstring, projecting a hypothesised cube into the image and matching its silhouette against the measured contour. Record both estimates and compare them — the gap between the two is the tilt error of Section 5.3, measured by you.

(b) **Complete the yaw estimate** in `estimate_block_yaw()` from the minimum-area rectangle of the top-face blob, following Section 5.3. Handle the angle ambiguity: a square block has 90° symmetry, so report the yaw modulo 90°. `one_shot_estimate()` reports the yaw from the silhouette fit; keep `estimate_block_yaw()` as the baseline reading and compare the two in part (c).

(c) **Take one-shot estimates.** Position the camera over the red block by key-jogging, press `[Enter]` in the live view, and record the printed estimate: position, yaw, depth, image centroid. Repeat five times from different camera poses around the block — the spread of the estimates is data, not noise to be hidden.

(d) **Complete `interpolate_pose()` and `plan_via_path()`.** Position blends with the three functions of Section 5.4's table; orientation blended on the rotation vector as described there. Verify with the self-test: `python planners.py` runs a synthetic path through two via points and prints a report for each profile — check that all three end at the same final pose and differ only in how they travel.

**Table 3: The three profiles on your via path**

| Quantity | linear | cubic | quintic |
|---|---|---|---|
| Segment peak velocity (from `path_report`) | ...... | ...... | ...... |
| Velocity at each via point (zero or not) | ...... | ...... | ...... |
| Poses rejected by IK (if any) | ...... | ...... | ...... |
| Your impression of smoothness (1–5) | ...... | ...... | ...... |

(e) **Choose vias and fly the transit.** The script prompts for via points as cup-tip positions in mm in the base frame, then for a profile. The transit stops 50 mm back along the cup axis from the estimate — with the wrist level that is 50 mm above the block. Verify on the live feed that the cup hovers over the block, and judge the miss by eye against the block below. Record in Table 4 where the hover actually ended up relative to the block.

**Table 4: One-shot estimate versus reality, red block**

| Quantity | Estimate | Observed at hover |
|---|---|---|
| x (mm, base frame) | ...... | ...... |
| y (mm, base frame) | ...... | ...... |
| z of top face (mm) | ...... | ...... |
| yaw (°, mod 90) | ...... | ...... |

> **Why the arm stops at 50 mm** — the estimate carries centimetres of error by design of this task: a single frame, a quantised pixel width, a face at a small angle. A planned descent onto a wrong estimate is a planned collision. The hover turns the error from a crash into an observation, and Task 4 turns the observation into a human decision.

**Question 3.1** Your five estimates differ. Name the two error sources you can quantify from your own recorded data (without new experiments), and for each, the table or quantity that quantifies it.

Response:

.....................................................................................................................................................................

.....................................................................................................................................................................

   **Möbius M3.1: answer now, on your own**
   Answer the question on one-shot accuracy. Look up: S2, S5.

**Question 3.2** `execute_path()` solves IK for every pose before the arm moves. State what would go wrong, physically, if instead each pose were solved and executed in turn, and why "the IK failed at pose 17, abort" is not an acceptable outcome at that point.

Response:

.....................................................................................................................................................................

   **Möbius M3.2: answer now, on your own**
   Answer the question on plan-then-execute. Look up: S6.

**Question 3.3** From Table 3 and what you saw flying each profile, which profile would you specify for (a) a transit passing 30 mm from the stack, and (b) a long empty-table crossing? Justify each from the properties of its blending function, not from preference.

Response:

.....................................................................................................................................................................

   **Möbius M3.3: answer now, on your own**
   Answer the question on choosing a profile. Look up: S6.

> **✔ Checkpoint 3**
> Show your GTA Tables 3 and 4 and your answers to Questions 3.1 to 3.3. **GTA initials: ____________**

**Extension 3.1: The same path, wrongly calibrated**

**EXTENSION | 15 min | LO2 | ILO 8.2 | Look up S5, S7**

(a) Temporarily load an identity T₆C (rename `gripper_calibration.npy` and let the script fall back to its default) and repeat steps (c) and (e) of Task 3.

(b) Record how the estimate and the hover move, and explain the direction of the error from the geometry of Section 5.1: which transform in the chain is the identity standing in for, and what does its absence do to the lever arm between camera and cup?

(c) Restore the file. The extension overwrites nothing if you renamed rather than edited — leave it that way.

**Extension 3.2: A fourth profile**

**EXTENSION | 20 min | LO3 | ILO 5.2, 8.2 | Look up S6**

(a) Add a trapezoidal velocity profile to `planners.py` as a fourth blend, keeping the same signature so the profile picker picks it up unchanged.

(b) Fly it on your Task 3 path and fill a fourth column of Table 3. Where does it beat quintic, and where does it lose?

### 6.4 Task 4: The stack

**CORE | 40 min | LO1, LO3, LO4 | Module outcome 1, 3, 4, 5 | ILO 5.1, 7.1, 7.2 | Lecture 14 (week 18) | Chapter 5 | Look up S1, S6, S8**

**Purpose.** To complete an integrated pick-and-place task — green block to the stack base, red block on top of the green — using everything calibrated and written so far, with you supplying the judgement that the estimates cannot.

You will complete `keygate_approach()` in `src/key_control.py` and wire the two transfers in `do_transfer()` / `run()` in `src/pick_and_place.py`. The vacuum wrappers in `robot_io.py` are provided.

(a) **Complete `keygate_approach()`.** Starting from the hover, each key press is one `KEY_STEP_MM` step of the cup tip in the base frame, at `KEY_SPEED_PCT` (the full legend is Appendix B.2). The flange pose for a given tip position comes from `flange_for_tip()`, which holds the wrist orientation fixed while the tip moves — note that the tip is a point on the flange's z axis, so a base-frame tip step is *not* a base-frame flange step; that translation is what your function composes. Clamp the tip to no lower than 2 mm above the table — the final press to contact is yours to make deliberately, not a step size's.

(b) **The green transfer.** Run `pick_and_place.py`. The script opens with mask tuning (your Task 2 bands), the profile choice (your Task 3 answer), then runs the pipeline for green: one-shot estimate, via-point transit to 50 mm back along the cup axis from the estimate, key-gated jog to contact, SPACE to engage the vacuum. Lift, transit to above the stack base, key-gate down until the block seats on the table, SPACE to release. **Record the printed seat height** — the red transfer builds on it.

(c) **The red transfer.** The same pipeline runs for red, with the destination computed one block height *above the green seat height you just recorded* — not above the nominal table height. Note what the script does with your measured seat height and why the nominal value would embed the green placement's error into the red approach (Question 4.2).

(d) **Judge the stack.** Photograph the finished stack beside a ruler if one is at the bench, and complete Table 5.

**Table 5: The stack**

| Quantity | Green transfer | Red transfer |
|---|---|---|
| One-shot estimate of block position (mm) | ...... | ...... |
| Number of jog steps to contact | ...... | ...... |
| Tip height at SPACE (mm) | ...... | ...... |
| Profile used | ...... | ...... |
| Vias used (count, and why those) | ...... | ...... |
| Result (stacked cleanly / adjusted / dropped) | ...... | ...... |

**Question 4.1** The key-gated approach exists because the estimate is imperfect. But the arm could measure contact itself (joint current, vacuum pressure). Argue for or against replacing the human in this loop once the estimates are known to be good to a few millimetres, naming one capability the human still has that such sensing does not.

Response:

.....................................................................................................................................................................

.....................................................................................................................................................................

   **Möbius M4.1: answer now, on your own**
   Answer the question on the human in the loop. Look up: S8.

**Question 4.2** The red destination was set from the *measured* green seat height, not from the nominal chain. Say what error would have propagated into the red approach if the nominal chain had been used, and estimate its size from your Table 5 numbers.

Response:

.....................................................................................................................................................................

   **Möbius M4.2: answer now, on your own**
   Answer the question on error propagation. Look up: S1, S5.

> **✔ Checkpoint 4**
> Show your GTA the stack and Table 5, and your answers to Questions 4.1 and 4.2. **GTA initials: ____________**

---

## 7 Wrap up

### 7.1 Save your work

Save your completed `handeye_gripper.py`, `vision.py`, `planners.py`, `key_control.py` and `pick_and_place.py`, your tables and your answers somewhere of your own, then delete your working files from the station PC. Nothing is carried forward, and **Laboratory 4 starts from a fresh repository** — but keep your `ik_solver.py`: the coursework needs it, and it is the same file you kept from Laboratory 2.

### 7.2 Submit your Möbius answers

Check that you have answered all 8 Möbius questions for this lab, then submit. The assignment closes at the end of your session. It is worth 5% of the module and is answered individually, even though you work in pairs: your partner will see different versions of the questions. You may use this handout, your own results and the lecture notes.

### 7.3 Reset your station

> **⚠ Park the robot before you switch anything off**
> This is compulsory and it is the last thing you do with the robot. In UFactory Studio, turn off Manual Mode and press Initial Position. The arm drives itself to the park position stored for your bench: Figure 6 for a bench against a window, Figure 7 for every other bench.
> Check the path is clear first and watch the arm the whole way, with a hand near the emergency stop. Do not type joint angles in to fold the arm yourself.
> If the arm does not reach the shape in the photograph, stop and tell your GTA. Ask your GTA to look at it before you power off.

*(Figures 6 and 7 are the same bench photographs as Laboratory 2's handout and are not reprinted here.)*

- Park the robot with Initial Position, check it against the photograph for your bench, then power it off.
- The vacuum pump stays connected to the arm; leave it attached. Return the ArUco artifact, the red block and the green block to the tray.
- Log out of the PC and delete local copies of your data. **Leave `data/gripper_calibration.npy` in place** — it is station data, like the Laboratory 2 calibration files, and the next pair at this bench uses it.
- Ask your GTA to sign off the reset before you leave.

---

## 8 Further reading

[1] Cinosi, N. *Introduction to Robotics*: course notes for ME4-ROB. Chapter 2 (kinematics of manipulators), Chapter 3 (inverse kinematics) and Chapter 5 (robotic vision). On Canvas.

[2] Cinosi, N. *Introduction to Robotics*: lecture slides. Decks 11 to 14 (robotic vision, weeks 15 to 18). On Canvas.

[3] Craig, J. J. (2017) *Introduction to Robotics: Mechanics and Control*. 4th edn. Pearson. Chapter 4 (inverse manipulator kinematics) — the convention your `ik_solver.py` uses — and the sections on sensing and end-effectors.

[4] Siciliano, B., Sciavicco, L., Villani, L. and Oriolo, G. (2009) *Robotics: Modelling, Planning and Control*. Springer. Chapter 3 (the pinhole model and calibration) and the trajectory-planning treatment of blending functions behind Section 5.4.

[5] Lynch, K. M. and Park, F. C. (2017) *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press. Chapter 3 for the rotation vector and the exponential map used in orientation interpolation. Free at modernrobotics.org.

[6] Hartley, R. and Zisserman, A. (2004) *Multiple View Geometry in Computer Vision*. 2nd edn. Cambridge University Press. The camera-model chapter is the reference for the depth-from-size formula's assumptions.

[7] SciPy. `scipy.optimize.least_squares`. docs.scipy.org — the solver behind both the touch calibration and the inverse kinematics.

[8] UFACTORY. *Lite 6 user manual* and *Kinematic and dynamic parameters of UFACTORY Lite 6*. On Canvas. Appendix B reprints the joint limits the scripts enforce.

---

## Appendix A Where this lab sits in the module

Use these codes in task tags, learning outcomes and Möbius questions.

### A.1 Module outcomes (module specification)

| Code | Module outcome |
|---|---|
| 1 | Model robotic manipulators and their kinematics |
| 2 | Develop and use dynamic models of manipulators |
| 3 | Apply core computer vision and perception techniques to support robotic manipulation |
| 4 | Design and implement an integrated control and perception solution for an industrial robotic manipulation task, demonstrating originality and innovation in their chosen approach |
| 5 | Organise and gather information and solve problems using information technology within individual and group projects |

### A.2 Intended learning outcomes (module prospectus)

| Code | Group | Outcome, in brief |
|---|---|---|
| 1.1 | Geometric modelling | Formulate and implement spatial coordinate mappings using homogeneous transformations and Denavit–Hartenberg parameters |
| 2.1 | Forward and inverse kinematics | Derive and solve forward and inverse kinematic models for 6-DOF manipulators |
| 2.2 | Forward and inverse kinematics | Analyse and justify solution choices given multiple or no solutions, joint limits, singularities and workspace constraints |
| 3.1 | Jacobian analysis | Derive, interpret and use Jacobians to relate joint and task space velocities and propagate uncertainty |
| 3.2 | Jacobian analysis | Identify singularities and evaluate their effect on control, safety and feasibility |
| 5.1 | Trajectory planning | Design and implement feasible joint and task space trajectories within kinematic and dynamic limits |
| 5.2 | Trajectory planning | Critically appraise alternative planning strategies for accuracy, smoothness, safety, cycle time and energy |
| 6.1 | Computer vision | Apply and evaluate calibration, feature extraction, pose estimation, recognition and tracking for manipulation |
| 6.2 | Computer vision | Integrate visual feedback into control and decision making, and assess noise, occlusion and lighting |
| 7.1 | Integrated automation | Specify, design and implement an integrated control and perception system for a 6-DOF industrial manipulator performing grasping |
| 7.2 | Integrated automation | Demonstrate originality and innovation in the solution approach |
| 8.1 | Validation and professional skills | Design and conduct experiments or simulations with appropriate metrics, test plans and analysis |
| 8.2 | Validation and professional skills | Critically evaluate performance and limitations, and propose prioritised improvements |
| 8.3 | Validation and professional skills | Communicate complex engineering work through reports, portfolios, presentations and teamwork |

### A.3 Lecture and laboratory schedule

| Week | Lecture | Laboratory |
|---|---|---|
| 15 | Lecture 11: Robotic vision I | Laboratory 2, group A |
| 16 | Lecture 12: Robotic vision II | Laboratory 2, group B |
| 17 | Lecture 13: Robotic vision III | **Laboratory 3, group A** |
| 18 | Lecture 14: Robotic vision IV | **Laboratory 3, group B** |
| 19–24 | Lectures 15–24: Mobile robots, advanced robotic vision | Laboratory 4 build |
| 25 | — | Laboratory 4 demonstrations |

Lectures from Dr Cinosi's timetable for 2026–27. Laboratories run twice, a week apart, once for each half of the class. Days, times and rooms are on Celcat and on Canvas.

---

## Appendix B Parameters, the key legend, and the files

### B.1 Fixed parameters of this laboratory

| Parameter | Value | Where it lives |
|---|---|---|
| Block side S | 25 mm | `BLOCK_SIDE_MM`, `lab_config.py` |
| Nominal tool offset (pre-calibration guess) | 75 mm | `TOOL_OFFSET_MM`, `lab_config.py` — Task 1 replaces this with your measured d |
| Hover stand-off | 50 mm back along the cup axis from the estimate / destination | `HOVER_ABOVE_MM`, `lab_config.py` |
| Table surface height (base frame) | 12.5 mm | `TABLE_Z_MM`, `lab_config.py` |
| Stack base (base frame xy) | (250, 0) mm | `PLACE_XY_MM`, `lab_config.py` |
| Key-gated step | 3 mm per press | `KEY_STEP_MM`, `lab_config.py` |
| Calibration captures (one look + several touches) | 6+ touches | `handeye_gripper.py` |
| Reprojection rejection | 15 px | `aruco_artifact_detector.py` (as Laboratory 2) |

### B.2 The key legend

| Key | Action (cup tip, base frame) |
|---|---|
| `[w]` / `[s]` | +x / −x |
| `[a]` / `[d]` | +y / −y |
| `[z]` / `[c]` | down / up |
| `[SPACE]` | confirm: engage vacuum (pick) or release it (place) |
| `[q]` | abort: release vacuum, stop moving, leave the arm where it is |

Joint limits enforced before every move are the "at your bench" row of Laboratory 2's Appendix B: J1 80° to 270°, J2 −150° to 150°, J3 −3.5° to 300°, J4 −180° to 180°, J5 −124° to 124°, J6 −180° to 180°.

### B.3 Getting the files, and what they are

Everything you need is in the student repository on GitHub, at the tag [lab3-2026-v1]. The repository is public, so you do not need a GitHub account. The steps are the ones you followed for Laboratory 2: open the repository page, switch to the tagged version, Download ZIP, extract into a working folder of your own named for your pair, and open the folder in Visual Studio Code.

What is in the download.

| File | Purpose | Provided or you write |
|---|---|---|
| `src/lab_config.py` | Every tunable in one place: sizes, hover heights, HSV defaults, speeds | Provided (defaults may need tuning in Task 2) |
| `src/fk_lite6.py` | Forward kinematics in the modified DH convention, as you finished it in Laboratory 1 | Provided |
| `src/ik_solver.py` | The inverse kinematics you completed in Laboratory 2, unchanged | You bring yours |
| `src/robot_io.py` | Every hardware call, wrapped: connect, joints, servo moves, vacuum | Provided |
| `src/aruco_artifact_detector.py` | ArUco detection and the artifact geometry (`board_config.json`), reworked for this lab's raw board frame | Provided |
| `src/handeye_gripper.py` | Task 1: the two capture loops and the report are provided; you complete the four maths functions — the top-face centre, the artifact pose in the base, the top-face plane and the per-touch solve | You complete |
| `src/vision.py` | Tasks 2 and 3: colour masks, blob selection, known-size depth, yaw | You complete |
| `src/planners.py` | Task 3: the three blends, pose interpolation, via-point path, path report | You complete |
| `src/key_control.py` | Task 4: the key-gated approach to contact | You complete |
| `src/pick_and_place.py` | Tasks 3 and 4: the interactive pipeline that ties everything together | You wire the last pieces |
| `src/motion.py`, `src/ui_display.py` | Move primitives, camera windows, live overlays, mask tuning | Provided |
| `data/board_config.json` | The artifact: five markers, IDs 0, 2, 3, 4 and 5; the raw board frame this lab uses | Provided |
| `data/camera_matrix.npy`, `data/dist_coeffs.npy`, `data/calibration_results.npy` | From Laboratories 1 and 2 at your bench | Station data |
| `requirements.txt` | Every Python library the scripts need, with OpenCV pinned at 4.13.0.92 | Provided |

The printed ArUco artifact and the two wooden blocks are at your bench. `docs/` holds the reference notes S1 to S8, and the GTAs' notes on how the solvers work.

---

## Appendix C Fallback datasets

If your calibration or detection fails, or you fall more than 15 minutes behind, your GTA will direct you to one of these. Use it to complete the analysis and state in your results that you used fallback data. Using it does not affect your Möbius mark.

They are on the local disk of every station PC, not on the network and not on GitHub.

| Fallback | What it contains | Use it for |
|---|---|---|
| `fallback/gripper_calib_A.npy` | A tool-offset calibration from a healthy session: one touch set with spread under the consistency warning | Task 1 onward if your calibration captures cannot be completed |
| `fallback/hsv_bands_bench.txt` | Accepted band sets recorded at this bench under both lighting states | Task 2 onward if your masks cannot be stabilised |
