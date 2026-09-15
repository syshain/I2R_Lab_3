"""Shared tunable constants for Lab 3 pick-and-place.

Every magic number that the other modules (vision, motion, ui_display,
pick_and_place) read from a single place. Grouped by phase/concern so it is easy
to find what to tweak on the bench.

Units convention: lengths in MILLIMETRES unless suffixed _M/_MM makes it
ambiguous; angles in DEGREES unless suffixed _RAD. Internal maths elsewhere runs
in metres + radians; these are the operator-facing tuning knobs.
"""

import numpy as np


# --- Block geometry ---------------------------------------------------------
BLOCK_SIDE_MM = 25.0        # cube edge length (mm); used by the known-size depth
                            # estimator (depth = fx * side / pixel_size).

# --- Tool offset ------------------------------------------------------------
TOOL_OFFSET_MM = 75.0       # flange-face -> suction-cup-tip distance (MM).
                            # NOT published by UFACTORY — measure/tune on your rig.
                            # Every pick/place Z target subtracts this so the CUP
                            # (not the flange) lands on the block top.

# --- Orientation: cup pointing straight DOWN at the table -------------------
# The suction-cup tip lies along the camera optical axis, which in the flange
# frame is ~[0.12, 0.02, 0.99] i.e. almost flange +Z. So "cup facing the table"
# == flipping the wrist so flange +Z points base -Z (anti-parallel). A single
# pitch flip does it — no Rodrigues, no full rotation solve.
DOWN_RPY_DEG = (0.0, 180.0, 0.0)   # roll, pitch, yaw that make the cup point down

# --- Phase B: centre the block in the IMAGE (camera-view servoing) ----------
CENTRE_TOL_PX = 6.0         # |u-cx| and |v-cy| both below this -> "centred"
LATERAL_STEPS = 12          # keystrokes available to centre the block

# --- Phase C descent: single-motion optical-axis advance --------------------
DESCENT_STEP_MM = 15.0      # forward travel per [w]/[s] press along the view axis
MAX_AIM_DEG_DESCENT = 10.0  # max wrist steering folded into a single step

# --- Phase E: fine positioning after the cup shift --------------------------
FINE_Z_STEP_MM = 3.0        # base-Z travel per [s]/[w] press in fine positioning
FINE_XY_STEP_MM = 3.0       # base-X/Y travel per [z]/[c] (X) or [a]/[d] (Y) press
DEPTH_BIAS_MM = 5.0         # subtract from the over-optimistic depth estimate
AUTO_LOWER_MIN_GAP_MM = 2.0 # [g] stops if this would put us within this of contact

# --- Pick -> lift ------------------------------------------------------------
LIFT_BACK_UP_MM = 80.0      # clearance gained when lifting off a grasped block

# --- Distance-based safety taper ---------------------------------------------
ADVANCE_STEP_MAX_MM = 40.0  # max per-press step when FAR from the block
ADVANCE_STEP_MIN_MM = 1.0   # floor: never command a step smaller than 1 mm
CLOSE_ZONE_MM = 60.0        # within this range, step+speed scale down linearly
FAR_SPEED_PCT = 30          # servo speed while far away (fast approach)
NEAR_SPEED_PCT = 8          # servo speed at the near end (creeping in)

MIN_CONFIDENCE = 0.35       # reject detections weaker than this


# ---------------------------------------------------------------------------
# HSV colour thresholds (OpenCV convention: H in [0,179], S/V in [0,255]).
#
# Why HSV over plain RGB here: hue decouples colour from brightness, so a block
# lit brighter or dimmer as the wrist moves stays inside its hue band — that's
# the robustness we lost with BGR boxes. The catch is that RED wraps around the
# hue wheel, so it needs TWO bands (the low-hue side near 0 and the high-hue side
# near 179); GREEN is a single mid-range band. Every bound is live-tunable via
# the trackbars in the two mask windows (see ui_display.tune_masks()), so these
# defaults are just sensible starting points for wooden blocks under lab lighting.
#
# Structure: COLOR_THRESHOLDS[color] is a LIST of bands; each band is a dict with
# h_lo/h_hi/s_lo/s_hi/v_lo/v_hi. A pixel belongs to the colour if it falls inside
# ANY band. Each bound maps to one trackbar: <COLOR>_band<i>_<CH>_<lo|hi>.
# ---------------------------------------------------------------------------
_HSV_DEFAULTS = {
    # Red: two bands straddling the 0/179 wraparound. Saturation kept fairly low
    # on the floor so pale/shadowed wood still registers; value floored to drop
    # near-black shadow pixels.
    'red': [
        {'h_lo': 0,   'h_hi': 10,  's_lo': 70,  's_hi': 255, 'v_lo': 60, 'v_hi': 255},
        {'h_lo': 160, 'h_hi': 179, 's_lo': 70,  's_hi': 255, 'v_lo': 60, 'v_hi': 255},
    ],
    # Green: one mid-hue band. Wooden greens often sit lower-saturation than
    # plastic, hence the modest s_lo floor.
    'green': [
        {'h_lo': 40,  'h_hi': 85,  's_lo': 50,  's_hi': 255, 'v_lo': 40, 'v_hi': 255},
    ],
}


def _clone_defaults():
    """Deep-copy the default HSV bands (so reset restores an independent copy)."""
    return {c: [dict(b) for b in bands] for c, bands in _HSV_DEFAULTS.items()}


# Live-mutable state, seeded from the defaults. Trackbars read/write THIS dict.
COLOR_THRESHOLDS = _clone_defaults()

MIN_AREA_PX = 300           # reject specks / noise
MAX_AREA_FRAC = 0.5         # reject blobs that fill most of the frame
