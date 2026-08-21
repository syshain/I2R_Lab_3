"""Monocular Block Detector — finds the single dark 50 mm cube in a frame.

The target is one 3D-printed dark cube sitting on a lighter work surface. We
work on the grayscale luminance channel and look for the darkest compact,
roughly-square region that is big enough to be a real block (not a speck) and
small enough to be a block (not the whole background).

Depth is inferred from the known physical size (50 mm) via the pinhole model:
    z = f * S_real / S_pixels
"""

import numpy as np
import cv2


# Known cube side length in metres
CUBE_SIDE_M = 0.050  # 50 mm

DARK_THRESH = 50     # pixels darker than this are "block" candidates
MIN_AREA_PX = 400     # reject specks / noise
MAX_AREA_FRAC = 0.5   # reject blobs covering most of the frame (background)


def detect_block(frame, K, dist_coeffs):
    """Detect the dark cube in a single camera frame.

    Parameters
    ----------
    frame : ndarray (H, W, 3) — BGR image from OpenCV
    K : ndarray (3, 3) — camera intrinsic matrix
    dist_coeffs : ndarray — distortion coefficients

    Returns
    -------
    dict or None. If found, the dict has keys:
        'center_px'   : (u, v) pixel coordinates of centroid
        'bbox_px'     : (x, y, w, h) bounding box in pixels
        'contour'     : numpy array of contour points
        'depth_m'     : estimated distance from camera optical centre (metres)
        'confidence'  : heuristic quality score [0, 1]
        'mean_lum'    : mean grayscale value inside the region
        'mask'        : (H, W) uint8 binary image used for detection (debug view)

    When no block is found, returns a dict with only 'mask' set (so callers can
    still display the thresholded image) and all other keys absent.
    """
    undistorted = cv2.undistort(frame, K, dist_coeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    img_h, img_w = gray.shape[:2]
    frame_area = float(img_h * img_w)
    fx = K[0, 0]

    # Dark regions -> white mask so findContours can pick them up.
    _, mask = cv2.threshold(gray, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_key = None  # higher = better (darkest among valid candidates)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_FRAC * frame_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 3.0:
            continue

        depth = _estimate_depth_from_size(w, h, fx)
        conf = _compute_confidence(cnt, (x, y, w, h))
        if depth is None or conf <= 0.2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        roi = gray[y:y + h, x:x + w]
        mean_lum = float(np.mean(roi)) if roi.size else 0.0

        det = {
            'center_px': (cx, cy),
            'bbox_px': (x, y, w, h),
            'contour': cnt,
            'depth_m': depth,
            'confidence': conf,
            'mean_lum': mean_lum,
            'mask': mask,
        }

        # Among all valid dark blocks, keep the darkest (lowest mean luminance).
        key = -mean_lum * 1000.0 + conf
        if best_key is None or key > best_key:
            best, best_key = det, key

    if best is None:
        # No block found, but still hand back the mask so the debug window can
        # show what the thresholding saw this frame.
        return {'mask': mask}
    return best


def _estimate_depth_from_size(bbox_w_px, bbox_h_px, fx):
    """Estimate camera-to-object distance from apparent pixel extent.

    Assumes the object face is roughly perpendicular to the optical axis and
    fills the bounding box. Uses the larger of width/height for robustness.

    pinhole model:  z = f * S_real / S_pixels
    """
    S_px = max(bbox_w_px, bbox_h_px)
    if S_px < 5:
        return None
    depth = fx * CUBE_SIDE_M / S_px
    return float(depth)


def _compute_confidence(contour, bbox):
    """Heuristic confidence: compactness and area ratio."""
    x, y, w, h = bbox
    area_bbox = w * h
    area_contour = cv2.contourArea(contour)
    if area_bbox == 0:
        return 0.0
    fill_ratio = area_contour / area_bbox
    # A square viewed head-on fills ~78% of its bounding box (circle inscribed).
    # Allow generous range for perspective distortion.
    if 0.3 <= fill_ratio <= 0.95:
        shape_score = 1.0
    else:
        shape_score = max(0.0, 1.0 - abs(fill_ratio - 0.65) * 4)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return 0.0
    circularity = 4 * np.pi * area_contour / (perimeter ** 2)
    return float(min(1.0, 0.6 * shape_score + 0.4 * circularity))
