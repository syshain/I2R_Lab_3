"""lab_config — every tunable for Lab 3 (colour-block stacking) in one place."""

import copy
import os

import numpy as np

# Hardware (env-overridable, same mechanism as Labs 1-2)
ROBOT_IP = os.environ.get('ROBOT_IP', '192.168.1.158')
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '4'))   # wrist USB camera

# Solver / joint box (Lite 6 mechanical limits, degrees; JOINT LIMITS in the
# UFACTORY handbook). ik_lite6.py and robot.py both read these.
JOINT_LIMITS_DEG = np.array([[-360.0, 360.0], [-150.0, 150.0], [-3.5, 300.0],
                             [-360.0, 360.0], [-124.0, 124.0], [-360.0, 360.0]])
IK_FD_EPS_RAD = 1e-5        # centred-difference step for the IK Jacobian, radians

# Block geometry / workflow / detection
BLOCK_SIDE_MM = 25.0
AXES_LENGTH_MM = 20.0       # axis arrows on the live preview when the artifact
                            # is detected (ui_display.py); frame W's own axes
TOOL_OFFSET_MM = 75.0       # flange -> cup-tip distance along flange +z (MM);
                            # measured by the two-capture touch calibration in
                            # handeye_gripper.py -> ../data/gripper_calibration.npy
DOWN_RPY_DEG = (0.0, 180.0, 0.0)
HOVER_ABOVE_MM = 50.0
LIFT_CLEAR_MM = 80.0
TABLE_Z_MM = 12.5
PLACE_XY_MM = (250.0, 0.0)
KEY_STEP_MM = 3.0
KEY_SPEED_PCT = 10
MIN_CONFIDENCE = 0.35
MIN_AREA_PX = 300
MAX_AREA_FRAC = 0.5
_RGB_DEFAULTS = {'red': [{'r_lo': 150, 'r_hi': 255, 'g_lo': 0, 'g_hi': 110, 'b_lo': 0, 'b_hi': 110}], 'green': [{'r_lo': 0, 'r_hi': 110, 'g_lo': 120, 'g_hi': 255, 'b_lo': 0, 'b_hi': 110}]}

def _clone_defaults():
    """Deep-copy the default RGB bands (so reset restores an independent copy)."""
    return copy.deepcopy(_RGB_DEFAULTS)
COLOR_THRESHOLDS = _clone_defaults()

# Data files (all under ../data/, reached via data_path())
DATA_FOLDER = 'data'
BOARD_CONFIG_NAME = 'aruco_config.json'   # the artifact config file on disk
CAMERA_MATRIX_NAME = 'camera_matrix.npy'
DIST_COEFFS_NAME = 'dist_coeffs.npy'
CALIBRATION_DATA_NAME = 'calibration_data.npy'
RESULTS_NPY_NAME = 'calibration_results.npy'
GRIPPER_CALIBRATION_NAME = 'gripper_calibration.npy'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_FOLDER)

def data_path(filename):
    """Absolute path to a file inside the data folder."""
    return os.path.join(DATA_DIR, filename)
