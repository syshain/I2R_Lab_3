# Lab 3 — Repository Structure

Lab 3 is **vision-guided pick and place** in Python: the wrist camera locates a red and a
green 25 mm block, the Lite 6 navigates its vacuum gripper over the green block using the
Lab 1/2 calibration, picks it up under a keygate, and places it on the red block. Every
autonomous motion is small and keygated; a human stays in the safety loop throughout.

## Frames and units

| Symbol | Meaning |
|--------|---------|
| `0`    | Robot base frame |
| `6`    | End-effector (flange) frame |
| `C`    | Wrist camera frame |

- `T_0_6` from forward kinematics on joint angles; `T_6_C` is the hand-eye transform from
  Lab 1/2 (`calibration_results.npy`, fitted jointly with the calibrated DH `zeta_stage2`).
- Internal maths runs in **metres + radians**. The xArm SDK speaks **millimetres + degrees**;
  the conversion happens once, inside `robot_io.py` / at the FK boundary (`fk_from_zeta`
  takes `zeta` lengths in mm and divides by 1000).

## Top-level layout

```
Lab_3/
├── src/          # Active Python code. One flat directory of single-responsibility modules.
├── data/         # Calibration artifacts (.npy/.json) produced by Labs 1/2 + this lab's runs.
├── docs/         # This structure guide + lab instructions.
└── commit-log.md # Human-readable version history
```

`data/` is produced by earlier labs and by this lab's own scripts; paths are resolved
relative to the repo via `data_path()` in `lab_config.py`, never hardcoded absolute paths.

## Module roles

The orchestrator imports downward only — no cycles: `pick_and_place.py` imports from
`motion`, `vision` and `ui_display`; those import the leaf modules. Any leaf module can be
read in isolation.

| Module | Owns |
|--------|------|
| `lab_config.py` | Every tunable in one place: block size, tool offset, hover height, thresholds (`_RGB_DEFAULTS`, live-mutable), hardware (`ROBOT_IP`, `CAMERA_INDEX` — the only two values readable from environment variables), every `data/` filename, and `data_path()`. Imports nothing local. |
| `fk_lite6.py` | Modified (Craig) D-H forward kinematics for the Lite 6: `fk_from_zeta(q, zeta)` (zeta packed `[a|alpha|d|theta_off]`, lengths in mm, angles in rad) and `pose_from_xyz_rpy` for cross-checking the controller's own pose readout. |
| `ik_solver.py` | Numerical inverse kinematics (bounded Levenberg–Marquardt on the pose residual, seeded from the arm's current joints). Also the nominal DH table and joint limits. Runnable directly for a self-test. |
| `vision.py` | Colour detection → base-frame geometry: RGB masking, best-blob selection, monocular depth from the block's known 25 mm size, and `fit_block_pose` — a 4-parameter (x, y, z, yaw) silhouette fit that measures the block's full pose from the image with no plane/height assumption. |
| `motion.py` | Motion primitives: `flange_pose_base` (FK at the current joints) and `move_to_pose` (IK + via the SDK). |
| `planners.py` | Via-point interpolation between Cartesian poses (linear / cubic / quintic), with orientation interpolated on SO(3). Runnable directly for a self-test that compares the profiles. |
| `robot_io.py` | The only module that touches the xArm SDK: connect, read joints, servo moves, and the vacuum gripper wrappers. `open_gripper` engages suction, `close_gripper` releases, `stop_gripper` is a best-effort stop. |
| `ui_display.py` | Camera, OpenCV windows, calibration/data loaders, the RGB threshold trackbars, and the keygated display/wait loops. No geometry, no planning. |
| `aruco_artifact_detector.py` | One camera frame → `T_C_W` by monocular PnP on the Lab 1/2 ArUco artifact (used during calibration capture). |
| `handeye_gripper.py` | Gripper calibration: touches against the ArUco artifact's top face measure the flange→cup distance `d`, from which the cup-tip frame G is built (`T_6_G`, `T_G_C = inv(T_6_G) @ T_6_C`). Writes `gripper_calibration.npy`. |
| `key_control.py` | Key-gated final approach: the last millimetres from the hover point down to contact are driven by hand (`keygate_approach`), plus the cup-tip ↔ flange geometry helpers `flange_for_tip` / `tip_position`. |
| `pick_and_place.py` | **Orchestrator only** — the keygated phase sequence (tune masks → free move → hover → one-shot estimate → descent → pick → place) and top-level `run()`. Owns no geometry or hardware detail. |

## data/ contents

| File | Meaning |
|------|---------|
| `camera_matrix.npy` / `dist_coeffs.npy` | Camera intrinsics and distortion (Lab 1 chessboard calibration). K is valid at 1920×1080; the camera is pinned to that resolution. |
| `calibration_results.npy` | Calibrated DH `zeta_stage2` + hand-eye `T_6_C` fitted WITH it (Lab 2). The pair must go into FK together. |
| `gripper_calibration.npy` | Stage-1 tool offset (flange→cup distance) from `handeye_gripper.py`. |
| `aruco_config.json` | ArUco artifact definition for `aruco_artifact_detector.py`. |
| `calibration_data.npy`, `capture_log.json` | Stored calibration captures and their log. |

## How to run

Create and activate a virtual environment at the `I2R_lab_development` root, then install
the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy opencv-python xarm-python-sdk pillow
```

Open `src/lab_config.py` and check the hardware parameters; `ROBOT_IP` and `CAMERA_INDEX`
can also be exported as environment variables on the bench (they have safe defaults).

Entry points (run from `Lab_3/src`, or anywhere — `data/` is resolved by the scripts):

```bash
python pick_and_place.py        # the full keygated pick-and-place (robot + camera)
python handeye_gripper.py       # tool-offset + hand-eye refit calibration (robot + camera)
python ik_solver.py             # IK self-test (no hardware)
python planners.py              # interpolator comparison self-test (no hardware)
```

Only `pick_and_place.py` and `handeye_gripper.py` need the arm and camera connected; all
other modules import without hardware.

Requires: Python 3.12+, NumPy 2.4.4, SciPy 1.17.1, OpenCV 4.13.0.92, xArm Python SDK 1.17.3,
Pillow 12.2.0, and a reachable UFACTORY Lite 6 with wrist camera and vacuum gripper.
