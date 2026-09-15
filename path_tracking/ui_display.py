"""Camera, OpenCV UI, calibration loading, and operator-input helpers for Lab 3.

Everything the interactive loop needs that is NOT motion planning or detection:
loading intrinsics / hand-eye from disk, opening the wrist camera, creating the
display windows and HSV trackbars, rendering a frame with overlays, and reading
keyboard input between gated stages.

Kept separate so pick_and_place.py stays a thin orchestrator and vision.py /
motion.py stay free of any GUI dependency.
"""

import time
from pathlib import Path

import numpy as np
import cv2

from constants import COLOR_THRESHOLDS, _clone_defaults
from vision import _hsv_color_mask, detect_color_block


_SCRIPT_DIR = Path(__file__).resolve().parent          # .../path_tracking
_CALIB_DIR = _SCRIPT_DIR.parent / 'Calibration'        # .../Lab_3/Calibration

CAMERA_INDEX = 5    # wrist USB camera device index

WINDOW = "Lab 3 : Pick & Place"
RED_MASK_WINDOW = "Lab 3 : RED mask"
GREEN_MASK_WINDOW = "Lab 3 : GREEN mask"
MASK_WINDOWS = [RED_MASK_WINDOW, GREEN_MASK_WINDOW]


# ===========================================================================
# Calibration loading (same convention as path_tracker.py / run_lab3.py)
# ===========================================================================
def load_camera_intrinsics():
    """Load camera matrix and distortion coefficients."""
    K = np.load(str(_CALIB_DIR / 'camera_matrix.npy'))
    dist = np.load(str(_CALIB_DIR / 'dist_coeffs.npy'))
    return K, dist


def load_calibration_results():
    """Load nominal DH (zeta) and the joint-based hand-eye T_ee_cam.

    We use the NOMINAL Lite 6 DH set deliberately: it matches the controller's
    own FK better than the kinematically-calibrated set (see run_lab3.py notes).
    """
    cal_file = str(_CALIB_DIR / 'calibration_results.npy')
    try:
        data = np.load(cal_file, allow_pickle=True).item()
        T_ee_cam = data['T_ee_cam_joint'] if 'T_ee_cam_joint' in data else data['T_ee_cam']
        zeta = data['zeta_nominal']
        print("Loaded NOMINAL DH + joint hand-eye from:", cal_file)
        return zeta, T_ee_cam
    except FileNotFoundError:
        print("WARNING: no calibration results found; using nominal DH + identity T_ee_cam.")
        from ik_solver import get_nominal_zeta
        return get_nominal_zeta(), np.eye(4)


# ===========================================================================
# Camera + windows
# ===========================================================================
def open_camera(cam_index=CAMERA_INDEX):
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera at index %d" % cam_index)
    for _ in range(5):
        cap.read()
    return cap


def size_windows(window_names):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for name in MASK_WINDOWS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    ww, wh = max(320, int(sw / 2)), max(240, int(sh / 2))
    for name in window_names:
        cv2.resizeWindow(name, ww, wh)


# --- Live HSV-threshold sliders ---------------------------------------------
# Per band, six trackbars (H/S/V lo+hi) live inside that colour's mask window.
# Red has two hue bands, so its window carries twelve sliders (band0 + band1);
# green has one band, six sliders. Dragging one mutates COLOR_THRESHOLDS
# immediately; the next show_frame() call re-syncs all sliders to the current
# values so programmatic resets stick too.
_TB_CHANNELS = ('h', 's', 'v')
_TB_SUFFIXES = ('lo', 'hi')


def _win_for(color):
    return RED_MASK_WINDOW if color == 'red' else GREEN_MASK_WINDOW


def _tb_name(color, band_idx, ch, suffix):
    return "%s_b%d_%s_%s" % (color.upper(), band_idx, ch.upper(), suffix)


def _iter_bounds():
    """Yield (color, band_idx, channel, suffix, value) for every live bound."""
    for color in ('red', 'green'):
        for bi, band in enumerate(COLOR_THRESHOLDS[color]):
            for ch in _TB_CHANNELS:
                for suf in _TB_SUFFIXES:
                    yield color, bi, ch, suf, band['%s_%s' % (ch, suf)]


def create_threshold_trackbars():
    """Create every HSV bound-trackbar, seeded from the current thresholds.

    Hue sliders range 0–179 (OpenCV's halved wheel); S and V range 0–255.
    Idempotent — safe to call once at startup.
    """
    for color, bi, ch, suf, val in _iter_bounds():
        name = _tb_name(color, bi, ch, suf)
        win = _win_for(color)
        rng = 179 if ch == 'h' else 255
        cv2.createTrackbar(name, win, 0, rng, lambda _: None)
        cv2.setTrackbarPos(name, win, int(val))


def sync_threshold_trackbars():
    """Push the current threshold values back into every slider.

    Called once per displayed frame so external changes (a reset, or code nudging
    a bound) are reflected even though OpenCV has no change callback on
    getTrackbarPos. Cheap: a handful of setTrackbarPos calls per bar.
    """
    for color, bi, ch, suf, val in _iter_bounds():
        cv2.setTrackbarPos(_tb_name(color, bi, ch, suf), _win_for(color), int(val))


def read_threshold_trackbars():
    """Pull whatever the operator dragged back into COLOR_THRESHOLDS.

    Must be called before each frame's mask is computed so a freshly-dragged
    slider affects THAT frame, not just the next one. Clamps hi >= lo per channel
    so a band never becomes inverted.
    """
    for color in ('red', 'green'):
        win = _win_for(color)
        for bi, band in enumerate(COLOR_THRESHOLDS[color]):
            for ch in _TB_CHANNELS:
                lo_n = _tb_name(color, bi, ch, 'lo')
                hi_n = _tb_name(color, bi, ch, 'hi')
                lo_v = cv2.getTrackbarPos(lo_n, win)
                hi_v = cv2.getTrackbarPos(hi_n, win)
                band['%s_lo' % ch] = int(min(lo_v, hi_v))
                band['%s_hi' % ch] = int(max(lo_v, hi_v))


def reset_thresholds():
    """Restore both colours to their default HSV bands."""
    fresh = _clone_defaults()
    for color in ('red', 'green'):
        COLOR_THRESHOLDS[color] = fresh[color]


def draw_debug(mask, label, accent_bgr=(0, 0, 255)):
    """Render a colour-segmentation mask as a labelled BGR debug image."""
    if mask is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "waiting for frame...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return blank
    coloured = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    coloured[mask > 0] = accent_bgr
    cv2.putText(coloured, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 255), 2)
    return coloured


def _both_masks(frame, K, dist_coeffs):
    """Compute BOTH colour masks from one BGR frame (shared undistort).

    Returns {'red': mask, 'green': mask}. Undistorting once and reusing it keeps
    the two windows perfectly in register with the main feed.
    """
    und = cv2.undistort(frame, K, dist_coeffs)
    return {'red': _hsv_color_mask(und, 'red'),
            'green': _hsv_color_mask(und, 'green')}


def show_frame(cap, status_line, info_lines=None, det=None, color='red', hint=None,
               extra_marks=None, K=None, dist_coeffs=None, show_both_masks=False):
    """Read one frame, draw overlay + detection box, display, return pressed key.

    The main window always shows the annotated feed. When ``K``/``dist_coeffs``
    are supplied (and/or ``show_both_masks`` is True) the two dedicated mask
    windows are refreshed too — one per colour — so you can watch red and green
    segmentation side by side and drag their threshold sliders live.

    ``extra_marks`` is an optional list of (u, v, label, bgr_color) tuples drawn as
    labelled markers on top of the frame — used to project the live cup-tip position
    into the image so the operator can SEE which way the cup is off-centre.
    """
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        return None

    # Pull any freshly-dragged slider values before we compute anything, so a
    # change affects THIS frame. No-op when the trackbars don't exist yet.
    read_threshold_trackbars()

    for u, v, lbl, colr in (extra_marks or []):
        cv2.drawMarker(frame, (int(u), int(v)), colr, cv2.MARKER_TILTED_CROSS, 18, 3)
        cv2.putText(frame, lbl, (int(u) + 10, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colr, 2)

    info_lines = info_lines or []
    if det is not None and det.get('bbox_px') is not None:
        x, y, w, h = det['bbox_px']
        col = (0, 0, 255) if color == 'red' else (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        # Tag near the box shows the base-frame flange->block distance — the
        # number that actually governs the next move.
        tag = "%.0f mm" % det['dist_flange_mm']
        cv2.putText(frame, tag, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        # Crosshair on the detected centroid.
        cu, cvv = int(det['center_px'][0]), int(det['center_px'][1])
        cv2.drawMarker(frame, (cu, cvv), col, cv2.MARKER_CROSS, 14, 2)

    bar_h = 90 + len(info_lines) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [status_line, hint or "[Enter]/[n] next     [q] quit"]
    for i, ln in enumerate(lines):
        cc = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, ln, (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cc, 2)
    for j, ln in enumerate(info_lines):
        cv2.putText(frame, ln, (10, 28 + 2 * 28 + j * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(WINDOW, frame)

    # Refresh the two per-colour mask windows whenever intrinsics are available.
    if K is not None and dist_coeffs is not None:
        masks = _both_masks(frame, K, dist_coeffs)
        cv2.imshow(RED_MASK_WINDOW,
                   draw_debug(masks['red'], "RED", accent_bgr=(0, 0, 255)))
        cv2.imshow(GREEN_MASK_WINDOW,
                   draw_debug(masks['green'], "GREEN", accent_bgr=(0, 200, 0)))
        sync_threshold_trackbars()

    return cv2.waitKey(1) & 0xFF


def tune_masks(cap, K, dist_coeffs):
    """Interactive threshold-calibration step: drag the sliders until BOTH masks
    cleanly isolate their block, then confirm.

    Runs its own tight display loop (no robot involved) showing the live feed plus
    both mask windows with their HSV bound-sliders (red carries two hue bands, so
    more sliders). The operator widens / narrows bounds until the red window lights
    up ONLY the red cube and the green window ONLY the green cube — no background
    bleed, no holes in the block.

    Keys:  [Enter]/[n] done & accept   [r] reset to defaults   [q] quit run.
    Returns True when accepted, False on quit.
    """
    print("\n" + "=" * 62)
    print("TUNE MASKS — drag the sliders in the RED and GREEN windows")
    print("=" * 62)
    print("Goal: each window's highlighted region should cover ONLY its block.")
    print("  [Enter]=accept   [r]=reset thresholds   [q]=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        read_threshold_trackbars()

        # Annotated main feed: draw each colour's box so you can correlate what
        # the mask sees with where it sits in the real image.
        disp = frame.copy()
        for color in ('red', 'green'):
            det = detect_color_block(frame, color, K, dist_coeffs)
            if 'bbox_px' in det:
                x, y, w, h = det['bbox_px']
                col = (0, 0, 255) if color == 'red' else (0, 200, 0)
                cv2.rectangle(disp, (x, y), (x + w, y + h), col, 2)
                cu, cvv = int(det['center_px'][0]), int(det['center_px'][1])
                cv2.drawMarker(disp, (cu, cvv), col, cv2.MARKER_CROSS, 14, 2)
        bar_h = 96
        ov = disp.copy()
        cv2.rectangle(ov, (0, 0), (disp.shape[1], bar_h), (40, 40, 40), -1)
        cv2.addWeighted(ov, 0.6, disp, 0.4, 0, disp)
        cv2.putText(disp, "TUNING — refine the two mask windows below",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(disp, "[Enter]=accept   [r]=reset   [q]=quit",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow(WINDOW, disp)

        masks = _both_masks(frame, K, dist_coeffs)
        cv2.imshow(RED_MASK_WINDOW,
                   draw_debug(masks['red'], "RED", accent_bgr=(0, 0, 255)))
        cv2.imshow(GREEN_MASK_WINDOW,
                   draw_debug(masks['green'], "GREEN", accent_bgr=(0, 200, 0)))
        sync_threshold_trackbars()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('\r'), ord('\n'), ord('n')):
            read_threshold_trackbars()
            print("Thresholds accepted (HSV, H in 0..179):")
            for color in ('red', 'green'):
                for bi, band in enumerate(COLOR_THRESHOLDS[color]):
                    print("  %-5s b%d  H[%d,%d]  S[%d,%d]  V[%d,%d]"
                          % (color.upper(), bi,
                             band['h_lo'], band['h_hi'],
                             band['s_lo'], band['s_hi'],
                             band['v_lo'], band['v_hi']))
            return True
        elif key == ord('r'):
            reset_thresholds()
            print("  thresholds reset to defaults.")
        elif key == ord('q'):
            print("Mask tuning aborted — quitting.")
            return False


def wait_next(status_line, cap, arm, zeta, K, dist, T_ee_cam, color,
              info_lines=None, abort_key='a'):
    """Show the feed until the operator advances or aborts.

    [Enter]/[n] -> advance;  [abort_key] (default 'a') -> abort this task.

    While waiting it continuously re-detects and overlays the LIVE base-frame
    readouts (flange->block distance, XY gap, depth) so you can judge alignment
    before committing.

    Returns (advanced: bool, aborted: bool).
    """
    from motion import flange_pose_base
    from vision import compute_detection

    while True:
        q_rad = _read_joints_safe(arm)
        det = None
        dyn_info = list(info_lines) if info_lines else []
        if q_rad is not None and color is not None:
            T_flange = flange_pose_base(q_rad, zeta)
            ret, frame = cap.read()
            if ret:
                det = compute_detection(frame, color, K, dist, T_flange, T_ee_cam)
            if det is not None:
                dyn_info.append("flange->block : %.0f mm" % det['dist_flange_mm'])
                dyn_info.append("  XY gap      : %.1f mm   (depth %.0f mm)"
                                % (det['xy_gap_mm'], det['depth_mm']))
            else:
                dyn_info.append("%s block NOT in view" % color.upper())
        elif color is not None:
            dyn_info.append("%s block NOT in view" % color.upper())
        key = show_frame(cap, status_line, dyn_info, det, color,
                         hint="[Enter]/[n] next   [%s]=ABORT  [q]=QUIT" % abort_key,
                         K=K, dist_coeffs=dist)
        if key in (ord('\r'), ord('\n'), ord('n')):
            return True, False
        # Both [a] (per-stage abort) and [q] (global quit) bail out of this task;
        # the caller's abort path releases suction safely before unwinding.
        if key in (ord(abort_key), ord('q')):
            return False, True


def _read_joints_safe(arm):
    import robot_io
    try:
        return robot_io.read_joints_rad(arm)
    except Exception:
        return None
