# v1 - First commit
21 Aug 2026
Working lab 3 code
- Python - vision based navigation towards dark block

# v2 - Added grasping task program
24 Aug 2026
- Added a Python inverse kinematics solver for cube navigation (future versions use this instead of the SDK's IK)
- Added grasping task programs: teach a trajectory, then recreate it with IK to move the block to a new location

# v3 - Vision-guided HSV pick & place + full cleanup
11 Sep 2026
- Two-stage task as the sole active entry point: pick RED, then track & place GREEN
- Colour detection via live-tunable HSV trackbars; tune_masks() before any motion
- Five-phase keygated approach per block; multi-block sequencing via a both-blocks-visible overview pose
- Split the monolithic driver into modules

# v4 - Rebuild with 1/2 calibration chain 
29 Oct 2026
- Two-capture touch calibration (handeye_gripper.py) for the tool offset and T_G_C, loaded together with the calibrated DH zeta (mm)
- Full task: one-shot block pose via a 4-parameter silhouette fit (x, y, z, yaw - all measured from vision, no hardcoded plane heights), base-frame vertical hover, keygated descent, suction pick, place on the red block; linear/cubic/quintic via-point planners
- Rebuilt data/calibration_results.npy from Lab 2's validated results (rolled zeta fixed); corrupt original kept as a backup
- Colour thresholding moved to RGB with live sliders; QT trackbar crash fixed
- Modularised to 12 flat src modules; one-line docstrings; dead code and stray prints removed; units consistent throughout (m + rad internally, SDK boundary in robot_io)
- Added docs/STRUCTURE.md; py_compile clean across all 12 src modules; IK self-test converges to 0.0 mm

# v4.1 - Added results files
## 29 Oct 2026
- Removed .npy and .npz ignore from .gitignore
- Data files now exist in the committed repository

# v5 - Align with Lab 2's codebase
30 Oct 2026
- Ported Lab 2's units.py, so3.py and reporting.py into Lab_3; renamed ik_solver.py -> ik_lite6.py and robot_io.py -> robot.py, and rewired every script onto them
- ik_lite6.py runs on the calibrated zeta from calibration_results.npy (nominal only as fallback) and keeps the residual-based convergence rule; when the primary bounded LM run stalls along a joint-limit face, solve_ik restarts from a handful of standard postures (IK self-test converges to < 0.01 mm on all reachable cases, NOT CONVERGED on the unreachable one)
- All scripts now print through the shared reporting grammar (banner/heading/field/progress)
- docs/STRUCTURE.md and docs/lab3_instructions.md rewritten to the new module layout; scratch diagnostic files removed
- py_compile and import checks clean across all src modules

# v6 - Dead-man gate, streamed execution, automatic liftoff
1 Nov 2026
- All motion phases gated on holding [g]; releasing it halts the arm at once
- [SPACE] toggles the vacuum with a 2 s settle before motion; a successful toggle ends the descent/seating phase
- Post-pick lift and post-seating retreat run automatically after the toggle; arm disabled (controller state 4) at the end of the task
- red/green renamed block/mask 1 and 2 throughout, so any block colour works
- PathRecorder saves executed-path 3D plots + CSV; console vias overlayed as larger dots
- Linear interpolation made the default profile
- New streamed executor (servo mode 1, jolt-free) alongside the keygated position-mode one, selectable at start; both share the dead-man gate

# v7 - One run per task; IK rejection count
## 9 Nov 2026
- One run of pick_and_place.py covers the whole task: every pre-descent transit ends at an offer — [Enter] descends, [p] re-asks profile and executor and plans another path, [q] finishes the run; [q] mid-flight aborts only that flight and returns to the offer
- _fine_plan reports how many fine poses IK rejected on a failed plan (feeds the handout's "poses rejected by IK" table row)
- Commented-out execute_path and key-gated jog blocks deleted with their now-unused imports; ONE_SHOT_DIAG off in the GTA copy; disable_arm only at the end of a run, never on a phase abort

# v8 - Streamed mode fix and robot serial number recording
## 11 Nov 2026
- Streamed mode fix added to reduce jolting at the start. 
- Robot serial number recorded by all robot-interfacing programs for hardware uniqueness

# v9 - Loader guards, NotImplementedError stubs
## 12 Nov 2026
- Camera/calibration loads from previous labs guarded with a "copy your Lab N files into data/" message (both trees)

# v10 - Wrist flip guard, cup tip protection and STRUCTURE.md update
## 12 Nov 2026
- Wrist-flip guard: _fine_plan rejects the whole move when any joint jumps more than IK_JUMP_DEG (10 deg) between consecutive fine steps
- Descent/seating z-floor now protects the cup tip (flange floor raised by the tool offset), not the flange
- Seating readout measures against the destination block (dest_block), not the carried block
- PathRecorder buffers vias prompted before their segment opens, so they reach the saved plot
- CAMERA_INDEX default 4 to 0; JOINT_LIMITS_DEG default J1 set to the bench limits 80..270
- docs/STRUCTURE.md updated to current behaviour (executors, wrist-flip guard, data/trajectories, mask_bands.json)

## v10.1 - Minor fixes
## 13 Nov 2026
- PathRecorder: vias always held until the next transit starts, so later flights label their own vias (not "lift"/"retreat")
- Wrist-flip guard: first fine step compared against the arm's current joints; jump measured modulo 2pi so the +/-180 deg wrap is not read as a flip
- handeye_gripper: touch spread uses sample std (ddof=1), matching the reference copy

# v10.2 - Ferdinando's fix list 
## 19 Nov 2026
- gated_path: a held-[g] frame after the last step no longer indexes past the plan (crashed the gripper phases, whose loop stays open for [SPACE])
- Streamed executor streams joint targets (set_servo_angle_j) instead of Cartesian poses; priming sends the controller's own joint readout, which removes the start-of-run jolt (bench run still to confirm)
- Artefact frame renamed A -> W to match the handout: T_0_A/T_G_A are now T_0_W/T_G_W in handeye_gripper.py, its console output and gripper_calibration.npy keys
- one_shot_gate: warns which joints sit outside JOINT_LIMITS_DEG and what to do, instead of a cryptic "IK failed at transit step 1"
- Wrist-flip guard: jump now measured as the raw joint difference, NOT modulo 360 (reverses the v10.1 reading); a near-full turn such as J6 +179 -> -179 is no longer waved through as a 2-degree step
