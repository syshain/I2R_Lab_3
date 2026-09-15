# v1 - First commit
21 Aug 2026
Working lab 3 code 
- Python - vision based navigation towards dark block

# v2 - Added grasping task program
24 Aug 2026
## Python changes
- Added python based inverse kinematics solver to cube navigation
    - Future versions will use this solver instead of relying on the sdks inverse kinematics
- Added grasping task programs. Teach robot a trajectory then recreate the trajectory using inverse kinematics and move the block to a different, possibly pre-decided location

# v3 - Vision-guided HSV pick & place + full cleanup
11 Sep 2026
## Significant edits
- Sole active entry point. Replaces the older monocular white-block nav as the Lab 3 task.
- Two-stage task: pick RED (terminal='lift'), then track & place GREEN (terminal='release').
- Colour detection via HSV thresholding (replaces BGR channel boxes):
  - Live-tunable per-bound trackbars in two mask windows (RED_MASK_WINDOW / GREEN_MASK_WINDOW); tune_masks() calibration step runs before any motion.
- Five-phase keygated approach per block: A orient-down (pitch 180), B rotational centre-in-frame, C fused aim+advance optical-axis descent , D one-shot shit to orient gripper, E fine-positioning hover/settle with suction control.
- Multi-block sequencing: continuously captures a both-blocks-visible overview pose (T_liftback) during free-move + Stage 0; lift-back returns there so the remaining block stays findable by vision. Fallback = +Z along cup axis.
- Split the monolithic driver into modules 
