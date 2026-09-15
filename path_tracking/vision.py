"""Colour-block detection and base-frame resolution for Lab 3.

Single source of truth for turning a wrist-camera frame into a 3-D block
position in the BASE frame. Everything downstream (display, approach planning,
descent) consumes ``compute_detection`` so they all agree on the same numbers.

Pipeline per frame:
    undistort -> HSV mask (any live band for the colour) -> best blob ->
    known-size depth estimate -> unproject centroid to a base-frame point.

Units: internal maths in METRES; pixel quantities in pixels; depths reported
back in millimetres by ``compute_detection``.
"""

import numpy as np
import cv2

from constants import (
    BLOCK_SIDE_MM,
    MIN_AREA_PX,
    MAX_AREA_FRAC,
    MIN_CONFIDENCE,
    COLOR_THRESHOLDS,
)


# ===========================================================================
# Colour segmentation
# ===========================================================================
def _hsv_color_mask(frame_bgr, color):
    """Binary mask of pixels inside ANY live HSV band for ``color``.

    Converts the undistorted BGR frame to HSV once, then ORs together every band
    in COLOR_THRESHOLDS[color]. Because red uses two hue bands (the 0/179
    wraparound), both are unioned so the full red region is captured. Bounds are
    read straight from the live dict the trackbars mutate, so dragging a slider
    changes the very next frame's mask with no other wiring.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    acc = None
    for band in COLOR_THRESHOLDS[color]:
        lo = np.array([band['h_lo'], band['s_lo'], band['v_lo']], dtype=np.uint8)
        hi = np.array([band['h_hi'], band['s_hi'], band['v_hi']], dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    return acc


def _best_blob(mask):
    """Pick the single best contour in a binary mask.

    Returns (centroid_px, bbox, area_px, confidence) or None if nothing valid.
    Confidence blends fill ratio (square-ish) and circularity, mirroring the
    heuristic in cube_detector._compute_confidence.
    """
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(mask.shape[0] * mask.shape[1])

    best = None
    best_key = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_FRAC * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 3.0:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        # A solid block viewed head-on fills its bounding box almost completely
        # (fill ~1.0); a thin sliver or noisy crescent does not. So reward HIGH
        # fill rather than penalising it — the earlier version rejected the very
        # best detections. Circularity bonus rewards compact, blob-like shapes.
        fill_ratio = area / max(w * h, 1)
        fill_score = 1.0 if fill_ratio >= 0.7 else max(0.0, fill_ratio / 0.7)
        perimeter = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 1 else 0.0
        conf = float(min(1.0, 0.6 * fill_score + 0.4 * circ))
        if conf <= MIN_CONFIDENCE:
            continue

        key = conf * 1000.0 + area
        if key > best_key:
            best_key = key
            best = ((cx, cy), (x, y, w, h), float(area), conf)
    return best


def detect_color_block(frame, color, K, dist_coeffs):
    """Detect a red or green cube in one frame.

    Parameters
    ----------
    frame : BGR image from OpenCV
    color : 'red' or 'green'
    K, dist_coeffs : camera intrinsics

    Returns a dict (or None) with keys:
        'center_px'  : (u, v) centroid
        'bbox_px'    : (x, y, w, h)
        'depth_m'    : camera-to-block distance estimated from known size (m)
        'confidence' : [0, 1]
        'mask'       : (H, W) uint8 debug mask
    """
    undistorted = cv2.undistort(frame, K, dist_coeffs)   # still BGR

    if color not in ('red', 'green'):
        raise ValueError("color must be 'red' or 'green'")
    mask = _hsv_color_mask(undistorted, color)

    blob = _best_blob(mask)
    if blob is None:
        return {'mask': mask}

    (cx, cy), bbox, area, conf = blob
    fx = K[0, 0]
    S_px = max(bbox[2], bbox[3])
    depth_m = fx * (BLOCK_SIDE_MM / 1000.0) / max(S_px, 5)

    return {
        'center_px': (cx, cy),
        'bbox_px': bbox,
        'depth_m': float(depth_m),
        'confidence': conf,
        'mask': mask,
    }


# ===========================================================================
# Unprojection + base-frame resolution
# ===========================================================================
def unproject_to_base(u, v, depth_m, K, T_cam):
    """Unproject a pixel at the given depth into a BASE-frame 3-D point (metres)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    p_cam = depth_m * ray_cam
    return T_cam[:3, :3] @ p_cam + T_cam[:3, 3]


def compute_detection(frame, color, K, dist_coeffs, T_flange, T_ee_cam):
    """Detect a coloured block AND resolve it into the BASE frame.

    Single source of truth for every downstream consumer (display, approach,
    descend) so they all agree on the same numbers.

    Parameters
    ----------
    frame : BGR image
    color : 'red' or 'green'
    K, dist_coeffs : intrinsics
    T_flange : (4,4) current base->flange transform (metres/radians)
    T_ee_cam : (4,4) flange->camera hand-eye transform

    Returns None if no valid blob, else a dict with:
        'center_px'      : (u, v) pixel centroid
        'bbox_px'        : (x, y, w, h)
        'depth_mm'       : camera-to-block distance (mm), from known size
        'confidence'     : [0, 1]
        'mask'           : (H, W) uint8 debug mask
        'p_block_base'   : (3,) block centre in base frame (m)
        'p_flange_base'  : (3,) flange position in base frame (m)
        'dist_flange_mm' : |flange - block| straight-line distance (mm)
        'xy_gap_mm'      : horizontal (XY) offset flange->block (mm)
    """
    det = detect_color_block(frame, color, K, dist_coeffs)
    if 'bbox_px' not in det:
        return None
    T_cam = T_flange @ T_ee_cam
    u, v = det['center_px']
    depth_m = det['depth_m']
    p_block = unproject_to_base(u, v, depth_m, K, T_cam)
    p_flange = T_flange[:3, 3]
    return {
        'center_px': det['center_px'],
        'bbox_px': det['bbox_px'],
        'depth_mm': float(depth_m * 1000.0),
        'confidence': det['confidence'],
        'mask': det['mask'],
        'p_block_base': p_block,
        'p_flange_base': p_flange.copy(),
        'dist_flange_mm': float(np.linalg.norm(p_block - p_flange) * 1000.0),
        'xy_gap_mm': float(np.hypot(*(p_block[:2] - p_flange[:2])) * 1000.0),
    }
