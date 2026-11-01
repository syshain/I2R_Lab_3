# Laboratory 3: hand–eye calibration and autonomous pick-and-place

**UFactory Lite 6, wrist camera, vacuum gripper**

*Version: Draft 2 — 1 November 2026*

*In this laboratory you finish the eye-to-hand calibration with a touch measurement,
find the two blocks with the wrist camera, plan a path to them through via points,
and stack one block on the other — with you in the loop for every move that could
touch something.*

---

## 1. At a glance

This laboratory connects the pieces from Laboratories 1 and 2 into an autonomous task.
Your camera is rigidly mounted on the arm's wrist; the arm stands on the bench; two
wooden blocks sit somewhere on it. You will calibrate the chain from camera to arm base,
use it to locate the blocks, and fly a pick-and-place task through it.

### 1.1 The four tasks

**Task 1** — finish the hand–eye calibration by touching the ArUco artifact from three
poses and re-solving the full chain, now with all four unknowns pinned.

**Task 2** — segment the two blocks (block 1 and block 2) with per-channel RGB threshold
bands, tuning them live against the actual bench lighting.

**Task 3** — estimate a block's position and yaw in one shot from its known 25 mm size,
then plan and fly a via-point transit that stops the cup 50 mm back along the cup axis
from the estimate, comparing linear, cubic and quintic interpolation. The flight is under
the dead-man switch: the arm moves only while you hold [g].

**Task 4** — the stack: the script picks block 1 and places it on top of block 2. Every
motion phase runs under the dead-man switch, the vacuum engages only when you press
[SPACE] with the cup resting on the block, and a successful [SPACE] ends its phase: the
lift-off, the retreat and the final arm-disable then run by themselves.

### 1.2 The software

Everything lives in `src/` of this repository. The pieces you met in Laboratory 2
(`lab_config.py`, `calibrate_handeye.py`, `capture_data.py`, `reporting.py`, `ui_display.py`,
`checker.py`, `solve_ik.py`, `ik_lite6.py`, `motion.py`, `run_calibration.py`) carry over
with a few additions; the new files are:

| Module | Purpose |
|---|---|
| `robot.py` | The only module that touches the xArm SDK: connect, joints, servo moves, control-mode switching and the vacuum gripper wrappers |
| `vision.py` | RGB masks, detections, depth and the one-shot fit — you complete three of its functions (Task 2 and Task 3) |
| `planners.py` | Pose interpolation profiles: linear, cubic and quintic — you complete the two profiles that are not linear (Task 3) |
| `key_control.py` | Task 4: the dead-man-gated path executors and the tip↔flange composition. You complete `flange_for_tip()` and read the executors your phases run on |
| `pick_and_place.py` | Tasks 3 and 4: the interactive pipeline that ties everything together — you wire the transfer in `do_transfer()` / `run()` |
| `path_recorder.py` | Records the executed cup-tip path of every motion phase and saves 3D plots and a CSV after the run |

Dependencies: Python 3, NumPy, SciPy, OpenCV, Matplotlib, and the UFactory SDK.
`requirements.txt` pins the versions used to develop the lab.

### 1.3 The station

Before the session:

- (a) **Connect to the arm** and open UFactory Studio (same procedure as Laboratory 2):
  the robot's IP is `lab_config.ROBOT_IP` if Studio does not show it.
- (b) **Put the controller in Manual Mode** (Studio, "Freedrive"/Manual toggle). Task 1's
  touches and the camera aiming of Tasks 3–4 are done by hand-guiding the arm — the
  scripts never drive it in those phases.
- (c) **The wrist camera** is mounted on the tool flange, looking down the flange axis.
  Same camera as Laboratory 2; the same calibration constants carry over.
- (d) **Confirm the vacuum pump** is connected to the arm's end effector port (the
  scripts' gripper wrappers fail loudly if it is not).
- (e) **Confirm the ArUco artifact and the two wooden blocks** (block 1 and block 2,
  25 mm cubes — they may be any two colours; the masks of Task 2 are tuned to whatever
  is at your bench) are at your bench. Put block 2 where you want the stack to stand,
  clear of the artifact and of block 1. The artifact stays where Laboratory 2 left it
  for now; Task 1 will tell you what to do with it.

---

## 2. Learning outcomes

| # | Outcome | Where |
|---|---|---|
| LO1 | Complete an eye-to-hand calibration whose chain has a gauge freedom, and close it with a tool-offset measurement | Task 1 |
| LO2 | Build a colour segmentation robust to real lighting, tuned live at the bench | Task 2 |
| LO3 | Fit a pose in one shot from a known object size, and plan a through-via path to it | Task 3 |
| LO4 | Integrate calibration, vision, planning and control into a complete robot task, and understand why a human stays in the loop | Task 4 |

---

## 3. Safety

**The arm moves under your keys in this laboratory.** In Tasks 3 and 4 every motion phase
runs under a *dead-man switch*: the arm advances only while you hold down **[g]**, and
releasing it halts the arm at once, mid-phase, wherever it is. Keep a hand near the
emergency stop whenever the arm is moving, planned or keyed. The one-shot estimate of
Section 5.3 is expected to be wrong by centimetres; that is why the arm never descends on
it — the planned transit stops 50 mm back along the cup axis from the estimate (with a
level wrist, 50 mm above it), and the last stretch of the descent is yours to fly, step
by step, under the switch.

- **The vacuum gripper picks up what it seals onto.** Engage it only with the cup flat on
  a block — which is exactly what the [SPACE] confirmation asks you to verify first.
  **[SPACE] does more than toggle the pump**: a successful toggle *ends the phase it
  belongs to* (the descent or the seating) and starts the automatic lift-off or retreat.
  Press [SPACE] only when the cup is where it should be: pressed early, it engages the
  vacuum in mid-air, and the arm lifts with nothing held — abort with [q] and re-run.
- **The arm never plans a move through your hand.** Every planned move is solved against
  the Lite 6 joint limits before anything moves, and flown at the conservative transit
  speed regardless of the profile or executor you chose. The 50 mm stand-off between the
  hover and the block is enforced by the pipeline, not by your vigilance.
- **[q] aborts** whatever phase is open: it releases the vacuum, stops moving, and leaves
  the arm where it is so you can recover. Use it the moment anything looks wrong.
- **Two short phases run without the switch, by design:** the lift after the pick and the
  retreat after the seating. Both start from a vacuum toggle you just confirmed, both are
  short and vertical, and the arm is disabled (controller stopped) as soon as the retreat
  is clear.
- **Park the arm** (Studio, back to Manual Mode and rest pose) before leaving the bench.
  Never leave the arm powered with its controller live and nobody at the keys.

---

## 4. Before the laboratory

### 4.1 Preparation

- Revise the eye-to-hand formulation from Laboratory 2: the chain
  $^{B}T_{E}(q)\,{}^{E}T_{G}\,{}^{G}T_{A}\,{}^{A}T_{C}$ and why the pair of rigid motions
  you solve cannot pin all of it in one shot.
- Revise RGB thresholding (the per-channel lo/hi bands of `vision.py`).
- Read Section 5 end to end before touching the arm. The lab session is short; the maths
  is not.
- Have your Laboratory 2 calibration constants (`data/calib/*.npy`) available on the
  machine you will run from — `pick_and_place.py` loads them.

### 4.2 Syllabus references

S1 rotations and SE(3), S2 homogeneous transforms, S3 camera models and calibration,
S4 colour segmentation, S5 kinematics and inverse kinematics, S6 trajectory generation,
S7 state machines in interactive robotics, S8 human-in-the-loop safety.

---

## 5. Background

### 5.1 The eye-to-hand chain, and its one gauge freedom

Laboratory 2 left you with $^{B}T_{G}$ and $^{G}T_{A}$ from hand–eye AX=XB and
$^{A}T_{C}$ from calibration of a camera over known points — but asked you to *pretend*
the tool offset d (flange-to-cup-tip distance) was known, because with the cup always at
a fixed height above the artifact, d and the artifact height trade off exactly: the chain
$^{B}T_{E}(q)\,{}^{E}T_{G}\,{}^{G}T_{A}\,{}^{A}T_{C}$ has a rigid-motion freedom you
cannot see from touch data alone. Task 1 breaks the freedom by touching the artifact at
three *different heights*, which pins d against everything else.

Read `hand_eye_calibration_math.md` for the full derivation; here we only restate what
you implement:

- $^{B}T_{E}(q)$ comes from forward kinematics (`motion.flange_pose_base`).
- $^{E}T_{G}$ and $^{G}T_{A}$ come from the Laboratory 2 hand–eye solve, re-run on the
  CAPTURE 1 dataset.
- $^{A}T_{C}$ comes from the Laboratory 2 artifact calibration, re-run on the same
  captures.
- d comes from the CAPTURE 2 touches of Task 1.

The gauge freedom is why the touch-based d is *not* identifiable in Laboratory 2 and
*is* in Task 1: same maths, different data.

### 5.2 Touching: how d is measured

You lower the cup until it rests on the artifact's pad, and the script reads the joints.
From $q$, forward kinematics gives $^{B}T_{E}$; the *tip* position is
$^{B}T_{E}\,[0,0,d,1]^\top$. Touching the same world point from several orientations
makes the tip positions agree at one value of d — the residual of a least-squares fit
over the touches. `handeye_gripper.py` implements this; you supply the touch poses
(Studio, Manual Mode) and the script does the algebra.

### 5.3 One-shot block estimation

The blocks are found in three steps, all inside `vision.py`:

1. **Segment.** The masks threshold the raw BGR frame directly: a pixel belongs to a
   colour if its red, green and blue values each fall inside that colour's *bands* — one
   lo/hi range per channel, set in `lab_config.py` and adjustable live with the trackbars
   of Task 2. What rejects the bench is the capping: the bench's wood, shadows and grey
   table are greyish, and a grey pixel cannot be bright in one channel and dark in the
   other two at the same time. The band for block 1's colour demands a high value in that
   colour's channel and caps the other two; the band for block 2's colour is its mirror
   image with its own channel. If several bands were defined for a colour they would be
   OR-ed together — a pixel joins the mask if it passes any of them — though each block
   ships with a single band by default.
2. **Depth.** The block is a cube of known side (25 mm). A ground plane looks like a
   trapezoid in the image; its projected top edge's width in pixels pins the depth once
   the intrinsics are known. `estimate_block_depth()` does this from the mask's top row.
3. **Yaw.** The same projected geometry: the block's four corners in the image, matched
   to the known square, give the rotation about the camera's z axis up to 90° symmetry —
   `fit_block_pose()` resolves it.

`one_shot_estimate()` composes the three into a single $^{B}T_{A}$ estimate with yaw,
and reports its residual and the image centroid so you can judge it in Table 4.

### 5.4 Via points and the two executors

A planned transit is a straight line from the current pose to a hover 50 mm back along
the cup axis from the one-shot estimate, then through any via points you enter, then to
the hover. The path is *interpolated*: for each pair of consecutive poses, a profile
fills in the in-between poses. Three profiles exist in `planners.py`:

| Profile | Character |
|---|---|
| Linear | Constant velocity between poses. The default. |
| Cubic | Zero velocity at the segment endpoints; smooth in position, slow near the vias. |
| Quintic | Zero velocity *and* zero acceleration at the endpoints; the smoothest, the slowest through vias. |

For each pose on the path the solver computes inverse kinematics *before* the arm moves
(`key_control._fine_plan`): if any step cannot be solved within the joint limits, the
error is printed and nothing has moved. This is the discipline that makes the
dead-man-key flight safe to abort: a plan that would have jammed the arm is rejected
while the arm is still still.

**Flying it: the two executors.** The planned path is executed by one of two executors,
chosen once when `pick_and_place.py` starts. The *gated* executor (the default) runs in
the controller's position mode: it pre-solves the whole path into 3 mm steps and advances
one step per frame you hold [g]. It is deliberately plain — the controller stops at each
commanded step, so the motion can feel slightly jolted, but every step is a full, checked
position command, which is the most conservative thing the arm can be asked to do. The
*streamed* executor runs in servo motion mode (mode 1): it pre-solves the same path and
then streams pose ticks to the controller every 20 ms while [g] is held, letting the
controller blend between them. The motion is noticeably smoother — jolt-free, with no
stopping at each step — at the price of stricter timing: the stream must keep its tick
rate, so nothing that could stall (a vision decision, a prompt) is allowed mid-motion.
Both executors share the dead-man switch and the solve-everything-before-moving
discipline; descent and seating stream at half the transit speed so the cup's approach
stays gentle. Task 4 asks you to compare them on the same path.

**Why via points at all?** The camera looks *down*; the cup approaches *along its own
axis*. Between them lies a change of posture the solver would happily take through a
configuration that swings the arm across the bench. A via point — entered as a cup-tip
position in mm in the base frame — pins the path where you know it is safe, and Task 4
asks you to justify your choices.

---

## 6. The tasks

### 6.1 Task 1 — closing the gauge: touch, then re-solve

**Purpose.** To pin the tool offset d against the chain, and to see the gauge freedom
close in the reprojection numbers.

**You will complete** the four `NotImplementedError` stubs in `src/handeye_gripper.py`:
`tip_positions()`, `fit_tool_offset()`, and the two wrapper functions that feed them.
Everything else is provided.

(a) **Touch the artifact three times, at three heights.** In UFactory Studio (Manual
Mode), lower the cup onto the artifact's pad with the arm in three different postures:
one approach nearly vertical, one noticeably tilted, one in between. Press `[Enter]` in
`capture_data.py`'s live view at each touch — the script saves the joints. **The three
touches must be at different heights on the pad** — that is what breaks the gauge
freedom; three coplanar touches reproduce it.

(b) **Re-run the full calibration chain** (`run_calibration.py`). The script now solves
hand–eye and artifact on CAPTURE 1 and d on CAPTURE 2, and reports the reprojection
residuals before and after the touch-based d.

(c) **Record in Table 1** and compare with your Laboratory 2 numbers.

| Quantity | Laboratory 2 (d guessed) | Laboratory 3 (d measured) |
|---|---|---|
| Reprojection RMS over CAPTURE 1 | | |
| d (tool offset, mm) | | |
| Spread of the three touch residuals (mm) | | |

(d) **Question 1.1.** Why did the three touches need different heights to pin d? Say
what is special about a set of coplanar touches, in terms of the gauge freedom, in two
or three sentences.

(e) **Question 1.2.** Your measured d differs from the value Laboratory 2 asked you to
assume (75 mm). Which of the chain's transforms must have been wrong before, and by how
much, if the re-solved chain now reproduces the artifact's points better?

**Checkpoint 1.** `run_calibration.py` reports a reprojection RMS under 2 mm on CAPTURE 1
with the measured d. Show the printout.

---

### 6.2 Task 2 — segmenting the blocks under real light

**Purpose.** To build a colour segmentation that works at *your* bench, in *your*
lighting — not in the developer's.

**You will complete** `bgr_bands_to_mask()` and `count_pixels_per_block()` in
`src/vision.py`.

(a) **Read the band definitions** in `lab_config.py` (`BLOCK1_BANDS`, `BLOCK2_BANDS`).
Each block has one band: three (lo, hi) pairs, one per BGR channel. A pixel joins the
block's mask if all three of its channels pass.

(b) **Complete the two stubs.** `bgr_bands_to_mask()` turns bands into a boolean mask
(one band, or the OR of several); `count_pixels_per_block()` reports, per block, how many
pixels of the live frame pass its mask.

(c) **Run the tuning session.** `python pick_and_place.py` opens with it: the main window
shows the live feed with both blocks boxed, and the two mask windows show what each
block's mask currently sees, with a trackbar per threshold bound. Drag until **each mask
highlights only its own block** — not the table, not shadows, not the other block, not
the wood edge of the bench.

(d) **Stress the masks before accepting.** Slide the blocks around the bench, swap them,
shade the camera with your hand, and switch on the bench lamp. The masks do not need to
survive *any* lighting — only the lighting of the bench you will fly Task 4 at — but
they must survive *changes within that session*. Record in Table 2.

| Trial | block 1 pixels | block 2 pixels | Notes |
|---|---|---|---|
| Both blocks lit, bench empty | | | |
| Hand shading half the bench | | | |
| Blocks swapped (does each mask follow its colour?) | | | |

(e) **Question 2.1.** The band for block 1 caps the other two channels, and that capping
is what rejects most of the table and bench: a grey surface cannot be bright in one
channel and dark in the other two at once. Give one lighting change in the room that
would make the default band for block 1 start failing — either flooding onto the bench
or losing the block — and say which bound you would retune first to survive it.

(f) **Question 2.2.** The masks allow more than one band per block, though the defaults
ship exactly one per block. Explain what a second band is for — for example, block 1
partly in shadow — and what distinguishes the pixels a second band admits from those the
first already covers.

**Checkpoint 2.** Both masks isolate their block on the live feed under at least two
lighting changes. Show the two mask windows.

---

### 6.3 Task 3 — one-shot estimation and the via-point transit

**Purpose.** To estimate a block's pose from its known size and the camera chain, then
plan and fly a via-point transit to a hover 50 mm back along the cup axis from it,
comparing all three interpolation profiles on the same path. The flight is under the
dead-man switch: the arm moves only while [g] is held.

**You will complete** `estimate_block_depth()`, `fit_block_pose()` and
`one_shot_estimate()` in `src/vision.py`, and `cubic_profile()` and `quintic_profile()`
in `src/planners.py`.

(a) **In `one_shot_estimate()`, compose the three steps** of Section 5.3: mask, depth,
yaw. The function receives the calibrated chain
$^{B}T_{E}(q)\,{}^{E}T_{G}\,{}^{G}T_{A}\,{}^{A}T_{C}$ and returns a base-frame block
pose with yaw.

(b) **Complete the two planner profiles** in `planners.py`: cubic (zero velocity at the
segment endpoints) and quintic (zero velocity and zero acceleration there). The linear
profile is provided and is the default at the prompt.

(c) **Take one-shot estimates.** Position the camera over block 1 by hand-guiding the
arm in UFactory Studio (Manual Mode), press `[Enter]` in the live view, and record the
printed estimate: position, yaw, depth, image centroid. Repeat five times from different
camera poses around the block — the spread of the estimates is data, not noise to be
hidden.

(d) **Complete Table 3** with the estimates, and compare their spread to the block's own
25 mm side. An estimate good to a few millimetres is excellent; a biased cluster is a
calibration problem (Task 1's residuals tell you how good the chain was).

(e) **Choose vias and fly the transit.** The script prompts for via points as cup-tip
positions in mm in the base frame, then for a profile (Enter accepts the default,
linear). The transit then flies under the dead-man switch: hold [g] to move, release to
pause — the plan stays solved, and holding again resumes. The transit stops 50 mm back
along the cup axis from the estimate — with the wrist level that is 50 mm above the
block. Verify on the live feed that the cup hovers over the block, and judge the miss by
eye against the block below. Record in Table 4 where the hover actually ended up
relative to the block. **The arm now waits at the hover, with the descent phase already
on screen: record Table 4 before touching [g] again** — the descent that follows is
Task 4's business.

| | Est. 1 | Est. 2 | Est. 3 | Est. 4 | Est. 5 |
|---|---|---|---|---|---|
| x, y, z (mm, base frame) | | | | | |
| yaw (°) | | | | | |

Table 3: Five one-shot estimates of block 1 from different camera poses.

| | Estimate (mm) | Reality (mm, judged on the live feed) | Miss |
|---|---|---|---|
| x, y | | | |
| z (height) | | | |

Table 4: One-shot estimate versus reality, block 1.

(f) **Question 3.1.** The depth estimate uses the block's known 25 mm side. Say what
happens to it when the block is tilted 5° out of level — does the bias grow, shrink, or
stay? Justify from the projected-geometry argument of Section 5.3.

(g) **Question 3.2.** `gated_path()` (and `streamed_path()`) solve IK for every step of
the path before the arm moves. Explain why that discipline matters specifically because
the path's *destination* came from a vision estimate rather than a taught point.

(h) **Extension** (if time allows). Fly the same transit twice, once with each executor
(re-run the script and change the executor choice), and note the difference in smoothness
beside the profile you flew. Which one would you trust closer to a human's hand, and why?

**Checkpoint 3.** A completed Table 3 and Table 4, and the hover's miss judged on the
live feed.

---

### 6.4 Task 4 — the stack

**Purpose.** To complete an integrated pick-and-place task — block 1 picked and placed
on top of block 2 — using everything calibrated and written so far, with you supplying
the judgement that the estimates cannot, and the dead-man switch keeping every motion
under your hand.

**You will complete** `flange_for_tip()` in `src/key_control.py` and wire the transfer
in `do_transfer()` and `run()` in `src/pick_and_place.py`. The two path executors,
`gated_path()` and `streamed_path()`, are provided — read them first, because they are
the safety machinery every phase you wire runs on, and Section 5.4 describes what each
does.

(a) **Complete `flange_for_tip()`.** Every phase you wire commands the *cup tip* in the
base frame, but the arm is a flange: the flange pose that puts the tip at a given
base-frame point is the tip point pulled back along the flange's z axis by the tool
offset d. Note that the tip is a point on the flange's z axis, so a base-frame tip step
is *not* a base-frame flange step; that translation is what your function composes. The
same function places the 2 mm-above-table safety floor on the descent and seating
targets.

(b) **The transfer.** Position block 2 where the stack should stand, clear of the
artifact, and block 1 within reach. Run `pick_and_place.py`. The script opens with mask
tuning (your Task 2 bands), then the profile choice (Enter accepts linear — your Task 3
answer may differ) and the executor choice (Enter accepts gated; fly it once with
streamed if time allows and note the difference). Then it runs the whole transfer under
the dead-man switch:

- **Estimate and transit.** Aim the camera at block 1 (hand-guiding), press `[Enter]`
  for the one-shot estimate — record it for Table 5 — choose your vias, and fly the
  transit to the hover under [g].
- **Descent and pick.** Hold [g] and bring the cup down onto the block. When the cup
  rests flat on it, press [SPACE]: the vacuum engages, the script waits 2 s for the
  suction to build, and the **lift runs by itself**, back to the pose the estimate was
  taken at. The dead-man gate resumes at the next transit.
- **Second estimate, transit and seating.** Aim the camera at block 2, press `[Enter]`
  for its one-shot estimate, and the destination is computed one block height above
  block 2's *estimated* top face — the destination rides on the vision estimate, not on
  a nominal table height. Choose vias and fly the second transit, then hold [g] to seat
  the block until it rests on block 2, and press [SPACE]: the vacuum releases and the
  **retreat runs by itself**, lifting the cup 80 mm clear. The arm is then disabled
  (controller stopped); re-enable it in UFactory Studio if you want to move it again.
- **Record the printed seat height** — the tip height at the moment of release. It is
  your stack's measured reality.

(c) **Judge the stack.** Photograph the finished stack beside a ruler if one is at the
bench, and complete Table 5. The executed tip paths of every phase were recorded as they
flew: after the script exits, `data/trajectories/` holds a 3D plot per phase (transits,
descent, lift, seating, retreat) with your vias marked as larger dots, and a CSV of the
whole executed path — open the plots and check they show what you felt: the gated
executor's stepped lift against the streamed executor's continuous glide, if you flew
both.

| | Pick (block 1) | Place (on top of block 2) |
|---|---|---|
| One-shot estimate of block position (mm) | | |
| Tip height at [SPACE] (mm) | | |
| Profile used | | |
| Executor used (gated / streamed) | | |
| Vias used (count, and why those) | | |
| Result (stacked cleanly / adjusted / dropped) | — | |

Table 5: The stack.

(d) **Question 4.1.** The dead-man descent exists because the estimate is imperfect:
you fly the last stretch and your eyes confirm contact before [SPACE]. But the arm could
measure contact itself (joint current, vacuum pressure) and lift the moment suction
seals. Argue for or against replacing the human in this loop once the estimates are
known to be good to a few millimetres, naming one capability the human still has that
such sensing does not.

(e) **Question 4.2.** The destination was computed one block height above block 2's
*estimated* top face, not from a nominal table height. Say what error would have
propagated into the placement if the nominal height had been used instead of the
estimate, and estimate its size from your Table 5 numbers.

**Checkpoint 4.** A photographed stack, a completed Table 5, and the `data/trajectories/`
plots from your run.

---

## 7. Wrapping up

### 7.1 What to save

Save your completed `handeye_gripper.py`, `vision.py`, `planners.py`, `key_control.py`
and `pick_and_place.py`, the 3D path plots and CSV that `pick_and_place.py` left in
`data/trajectories/`, your tables and your answers somewhere of your own. The lab
machines are wiped between sessions; nothing on them survives.

### 7.2 Möbius questions

1. Your Table 4 miss was a few centimetres. List the three largest contributions to it
   in order, and say which single measurement you would redo to shrink it most.
2. The streamed executor is smoother but demands timing discipline; the gated one is
   safer but jolts. Which would you fly with a *novice* at the keys, and which with an
   *empty* bench? Justify both from the safety argument, not from preference.

### 7.3 Leave the bench as you found it

Return the ArUco artifact and the two blocks to the tray, and park the arm (Studio,
Manual Mode off, rest pose). The scripts disable the arm at the end of the task; re-enable
it in UFactory Studio if you want to move it again, and leave it parked and disabled.

---

## 8. Further reading

- Furgale et al., *Unified temporal and spatial calibration for multi-sensor systems*
  (IROS 2013) — the gauge-freedom argument in its general form.
- Quigley et al., *Introduction to Robotics*, ch. 3–4 — SE(3), Jacobians, and the
  interpolation profiles you implemented.
- The UFactory Lite 6 user manual, sections on manual mode, servo motion (mode 1) and
  the vacuum gripper — what the SDK wrappers in `robot.py` are calling.
- Corke, *Robotics, Vision and Control*, ch. 10–13 — hand–eye calibration and
  colour-based object localisation, both in one framework.

---

## Appendix A — module and schedule

This laboratory is assessed as part of the Robotics module's laboratory component
(15% of the module mark, of which Laboratory 3 is one third). The module's intended
learning outcomes ILO3 (design and implement a complete robot perception–action loop)
and ILO4 (evaluate safety-critical interactive systems) are assessed here through the
four checkpoints, the tables and the Möbius questions.

| Week | Activity |
|---|---|
| 5 | Laboratory 2 calibration data carried over; Tasks 1–2 in session |
| 6 | Tasks 3–4 in session; report due one week after |

---

## Appendix B — parameters, keys, files

### B.1 Numbers you will quote

| Quantity | Value | Where |
|---|---|---|
| Camera → artifact, hover → block | 50 mm back along the cup axis | `HOVER_ABOVE_MM`, `lab_config.py` |
| Table surface height | 12.5 mm above the base frame's z = 0 | `TABLE_Z_MM`, `lab_config.py` |
| Dead-man fine step | 3 mm per step; the arm advances one step per frame [g] is held | `KEY_STEP_MM`, `lab_config.py` |
| Transit speed (gated executor) | 10% of the controller's maximum | `KEY_SPEED_PCT`, `lab_config.py` |
| Vacuum settle | 2 s between engaging the vacuum and any motion | `GRIPPER_SETTLE_S`, `lab_config.py` |
| Stream tick / streamed speeds | 20 ms ticks; 40 mm/s transits, 20 mm/s descent and seating | `STREAM_TICK_S`, `STREAM_SPEED_MM_S`, `STREAM_DESCENT_MM_S`, `lab_config.py` |
| Retreat clearance | 80 mm above the seat before the arm is disabled | `LIFT_CLEAR_MM`, `lab_config.py` |
| Block side | 25 mm (used by the depth fit) | `BLOCK_SIDE_MM`, `lab_config.py` |

The table surface height is a safety floor, not an input to the pose fit: the descent
and seating targets are clamped 2 mm above it, and CAPTURE 1's plausibility check uses
it to reject nonsense.

### B.2 The key legend

During Tasks 3 and 4 the arm moves only while you hold **[g]**. The phases that can
touch the vacuum show the full legend on screen; the others show it without [SPACE].

| Key | Action |
|---|---|
| `[g]` (hold) | move: the arm advances one fine step (3 mm) per frame while the key is down; **release it and the arm stops at once** |
| `[SPACE]` | gripper on/off: engage the vacuum (cup on the block) or release it (block seated). A successful toggle *ends the phase*: the automatic lift-off or retreat starts right after, without [g] |
| `[q]` | abort: release the vacuum, stop moving, leave the arm where it is |

A failed [SPACE] (the pump does not respond) keeps the phase open so it can be retried.
The automatic phases (lift, retreat) run without [g] and ignore [SPACE]; [q] aborts
them too.

The camera aiming of Tasks 3–4 and CAPTURE 2's touches in Task 1 are done by
hand-guiding in UFactory Studio (Manual Mode), not by keys.

The Lite 6's joint limits (degrees, from `lab_config.JOINT_LIMITS_DEG`) are enforced by
every planned move: a path that would leave them is rejected before the arm moves.

### B.3 The files

| File | Purpose | You complete |
|---|---|---|
| `lab_config.py` | Every tunable in one place; hardware values readable from the environment | nothing |
| `robot.py` | The only module that touches the xArm SDK: connect, joints, servo moves, control-mode switching, gripper wrappers | nothing |
| `reporting.py` | One reporting grammar for all scripts (banner / field / progress) | nothing |
| `ui_display.py` | OpenCV windows, overlays, the trackbar tuning session | nothing |
| `checker.py` | Laboratory 2's calibration checks, carried over | nothing |
| `capture_data.py` | CAPTURE 1 and CAPTURE 2 recording | nothing |
| `solve_ik.py`, `ik_lite6.py` | The Lite 6 inverse kinematics, with joint limits | nothing |
| `motion.py` | Forward kinematics and single-pose moves | nothing |
| `calibrate_handeye.py` | Laboratory 2's hand–eye solve, carried over | nothing |
| `run_calibration.py` | The full chain: hand–eye + artifact + d | nothing |
| `handeye_gripper.py` | The touch-based tool offset d | Task 1 (4 functions) |
| `vision.py` | Masks, detections, depth, one-shot estimate | Tasks 2–3 (4 functions) |
| `planners.py` | Pose interpolation profiles | Task 3 (2 profiles) |
| `key_control.py` | Task 4: the dead-man-gated executors and the tip↔flange composition. You complete `flange_for_tip()` and read the executors your phases run on | Task 4 (1 function) |
| `pick_and_place.py` | Tasks 3–4: the interactive pipeline that ties everything together | Task 4 (wire the transfer) |
| `path_recorder.py` | Records the executed cup-tip path of every phase; saves 3D plots and a CSV to `data/trajectories/` | nothing |

**Getting the files.** Everything is in the laboratory repository, tagged
`[lab3-2026-v1]`; clone or download it before the session. The `data/` directory in the
repository is regenerated by hardware runs — if you are working from a fresh clone,
`run_calibration.py` will tell you which datasets it expects.

---

## Appendix C — fallback datasets

If the bench hardware is unavailable on the day, `data/fallback/` contains pre-recorded
CAPTURE 1 and CAPTURE 2 datasets, and pre-recorded wrist-camera frames of the two
blocks. Tasks 1–3 run unchanged on them. Task 4 needs the arm; on the fallback data it
becomes a paper exercise: plan the transfer on the recorded frames, and hand in the via
choices with their justification instead of Table 5's last row.
