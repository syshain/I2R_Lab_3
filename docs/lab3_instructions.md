# Lab 3 — Instructions: vision-guided pick and place

This is the bench run-sheet. The handout has the theory, the derivation of each
calibration chain, and what your report must assess; this file tells you what to run,
in what order, what the scripts print, and what to write down as you go.

The Lab 3 code is **provided and complete** — you are not filling in skeletons. Your job
is to verify each chain against something independent, operate the keygated workflow on
the bench, and judge the numbers the scripts print. The one habit from Lab 1 that
matters most here: a solver is only converged when its **pose residual** is small, never
because its step size went quiet.

## 0. Before you start

Work from the `I2R_lab_development` root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install numpy scipy opencv-python xarm-python-sdk pillow
```

Then check `Labs/Lab_3/src/lab_config.py`: the robot IP and camera index (or export
`ROBOT_IP` / `CAMERA_INDEX`), and the geometry constants (`TOOL_OFFSET_MM` is only a
nominal reference — Stage 1 measures the real one).

Have on the bench: the Lite 6 with the wrist camera and vacuum cup attached, the ArUco
artifact on the table, one red and one green 25 mm block. Read `docs/STRUCTURE.md` once
so you know which module does what.

## 1. Verify the inverse kinematics (no hardware)

`ik_lite6.py` is the Lab 2 solver, running here on the **calibrated** DH parameters from
`calibration_results.npy` (it falls back to the nominal table only if that file is
missing). You do not write it; you verify it.

```bash
cd Labs/Lab_3/src
python ik_lite6.py
```

The residual self-check builds a target pose from a known joint set with the FK and
stands the residual at those same readings: it must be zero, or your residual chain is
broken. The verification block then offsets the target and checks the residual is
non-zero but sane. Record the residual norms in both blocks.

Then solve one pose from the command line (mm, degrees):

```bash
python ik_lite6.py 150 0 300 0 180 0
```

and compare the solved joint angles with UFactory Studio's own IK readout for the same
pose. Repeat one solve with a starting guess appended — the solve then runs **inside the
joint box**, every step projected onto the limits. If you see the banner report a
restart ("converged from posture ..."), that is the solver retrying standard postures
after its first run stalled along a joint-limit face; note which posture it needed.

Record: the self-check residuals, one solved pose (target, solved joints, the printed
position/orientation error), and one bounded solve.

## 2. Stage 1 — gripper calibration (robot + camera)

```bash
python handeye_gripper.py
```

CAPTURE 1: hold the artifact still, let the camera see it once. SPACE captures and
freezes the frame so you can check the pose and the marked touch point before accepting
it (SPACE = accept, r = retake, q = abort). Heed the plausibility warnings: a face below
the table plane, or more than 1500 mm from the flange, means a bad capture — retake it,
the camera closer to the artifact and more frontal.

CAPTURE 2: move the arm **by hand** and touch the artifact's top face (the flat face
carrying the marker) with the cup tip. At least 3 touches; about 6 is a good number. Any
point on the face and any wrist tilt are fine — the touches do not have to be at the
centre and need not be vertical. SPACE records each touch, r removes the last, c clears
all, ESC saves when you have enough, q aborts.

The report prints per-touch d values, their spread, and warnings. Take the warnings
seriously: a touch flagged GRAZE is nearly parallel to the face and unreliable; a per-
touch spread above 5 mm means the touches disagree — remove the outliers and retake. A
fitted d more than 20 mm away from the nominal 75 mm usually means a bad CAPTURE 1.

Record: the fitted tool offset d (mm), the per-touch d values and their spread, the
T_G_C translation, and the T_G_A translation. The file `gripper_calibration.npy` is
written to `data/`; every later stage loads the tool offset from it.

## 3. The pick-and-place run (robot + camera)

```bash
python pick_and_place.py
```

The script walks a keygated sequence; a human confirms every gate. The on-screen legend
for the final approach is `[w/s]=+/-X [a/d]=+/-Y [z]=down [c]=up [SPACE]=confirm
[q]=abort`, with 3 mm steps at 10 % speed.

Phase by phase:

1. **Tune masks.** Live RGB threshold trackbars (not HSV — Lab 3 thresholds in RGB) for
   the red and green bands. Adjust until each block gives one clean, solid, square-ish
   mask with no background bleed, then confirm. Record the final thresholds; the
   defaults are in `lab_config.py`, your tuned bands are what the run uses.
2. **Choose the transit profile** (linear, cubic or quintic) and enter any via points
   you want the transit to pass through.
3. **One-shot estimate.** One camera frame measures the green block's full top-face pose
   (x, y, z, yaw) by a silhouette fit — no hardcoded plane heights. The block is printed
   under `ONE-SHOT ESTIMATE — record for the report`: write down the position, yaw,
   depth, view and confidence exactly as printed. At this point the script also prints
   the FK chain's tool-tip position next to the joints read; compare it with UFactory
   Studio's live readout as a diagnostic cross-check of the whole calibration chain.
4. **Planned transit to the hover point.** The hover point sits HOVER_ABOVE_MM above the
   measured block top, approached from LIFT_CLEAR_MM behind along the cup axis. The
   path report (per-segment length, peak joint speed, planner timing) is printed —
   record it for the profile comparison.
5. **Keygated descent.** You drive the last millimetres by hand. Watch the cup seal on
   the block face; SPACE confirms the contact when the cup is seated.
6. **Pick.** The vacuum engages (the SDK's gripper-open state is suction ON), the block
   lifts clear by LIFT_CLEAR_MM. Record the seated z height the script prints.
7. **Place.** The same one-shot → transit → hover → descent chain brings the green
   block onto the top of the red block, the vacuum releases, and the arm retreats.

Every phase can be aborted with q; on exit the script always releases the camera and
best-effort stops the gripper. If anything goes wrong mid-descent, q immediately and
move the arm clear by hand.

Record: both one-shot estimates, the tuned thresholds, the path reports, the seat
height, and how long the transfer took.

## 4. The via-point planners (no hardware)

```bash
python planners.py
```

This compares the linear, cubic and quintic interpolation profiles on the same
three-pose path, with orientation interpolated on SO(3). Record the comparison table it
prints; you will use it to argue which profile you would choose for a real transit and
why.

## Safety

- Keep a hand on the abort key (q works at every gate) and know where the e-stop is.
- The keygates exist so that no autonomous motion is larger than 3 mm without a human
  confirming. Do not raise `KEY_STEP_MM` or `KEY_SPEED_PCT` to make the lab go faster.
- Check the workspace before each run: blocks seated, cable slack clear of the arm,
  nobody's hands under the cup.
- The vacuum releases on script exit, but treat a stuck block as a manual task: stop the
  arm, then release by hand.

## Troubleshooting

- **Masks bleed into the background** — fix the lighting first (the thresholds are RGB
  bands; a white-balance shift breaks them), then retune the trackbars.
- **"artifact NOT in view" or a plausibility warning in CAPTURE 1** — move the camera
  closer and more frontal to the artifact; a bad PnP depth is the usual cause.
- **IK prints NOT CONVERGED** — the target is unreachable or outside the joint box; the
  printed settled position tells you how far off it is. Try the CLI with a starting
  guess near where the arm actually is.
- **A restart banner during a bounded solve** — not a failure; the solver resumed from
  a standard posture. Note it and carry on.
- **QT trackbar crash** — run with the QT platform plugin available
  (`export QT_QPA_PLATFORM=offscreen` is wrong for tuning; use a desktop session, or
  check that `opencv-python` (not `-headless`) is installed).
