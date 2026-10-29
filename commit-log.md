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
