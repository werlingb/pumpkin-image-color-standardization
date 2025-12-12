# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:17:01 2025

@author: werlingb
"""

# create_reference_npz_fixed_window_v3_resizable.py
# Same as v2 but with resizable corner-selection window (no overflow cutoff)

import importlib, subprocess, sys, os
from tkinter import Tk, filedialog
import numpy as np
import cv2

# ---------- auto install ----------
def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

# Ensure numpy and cv2
np = _ensure("numpy")
cv2 = _ensure("cv2")

# ---------- helpers ----------
def ask_image():
    Tk().withdraw()
    return filedialog.askopenfilename(title="Select color checker reference image",
                                      filetypes=[("Image files","*.jpg *.jpeg *.png")])

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)

def get_card_corners(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    disp = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    clone = disp.copy()
    pts = []

    win = "Click 4 corners (any order), ENTER when done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1400, clone.shape[1]), min(900, clone.shape[0]))

    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow(win, clone)

    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, click)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(pts) >= 4:  # ENTER
            break
    cv2.destroyWindow(win)
    inv = 1.0 / scale
    pts_full = [(int(x * inv), int(y * inv)) for (x, y) in pts[:4]]
    return order_corners(pts_full)

def warp_card(img, corners, target_wh=(400, 600)):
    W, H = target_wh
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (W, H))

def sample_patches(warp, rows=6, cols=4):
    lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)
    H, W = lab.shape[:2]
    ch, cw = H/rows, W/cols
    vals, boxes = [], []
    frac = 0.4  # central 40%
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r*ch), int((r+1)*ch)
            x0, x1 = int(c*cw), int((c+1)*cw)
            h, w = y1-y0, x1-x0
            cy0 = y0 + int((1-frac)/2*h)
            cy1 = y1 - int((1-frac)/2*h)
            cx0 = x0 + int((1-frac)/2*w)
            cx1 = x1 - int((1-frac)/2*w)
            roi = lab[cy0:cy1, cx0:cx1]
            vals.append([np.median(roi[...,0]), np.median(roi[...,1]), np.median(roi[...,2])])
            boxes.append((x0,y0,x1,y1, cx0,cy0,cx1,cy1))
    return np.array(vals, np.float32), boxes

# ---------- main ----------
def main():
    print("=== Create Reference NPZ (Resizable window) ===")
    img_path = ask_image()
    if not img_path: return

    img = cv2.imread(img_path)
    corners = get_card_corners(img)
    warp = warp_card(img, corners, (400, 600))
    ref_colors, boxes = sample_patches(warp)

    # save npz next to image
    base, _ = os.path.splitext(img_path)
    npz_path = base + "_ref.npz"
    gray_idx = np.array([3,7,11,15,19,23])  # neutral patches
    np.savez(npz_path, ref_colors=ref_colors.reshape(6,4,3), gray_idx=gray_idx)
    print(f"Saved reference: {npz_path}")

    # diagnostic preview (outer + sampled region)
    prev = warp.copy()
    for (x0,y0,x1,y1,cx0,cy0,cx1,cy1) in boxes:
        cv2.rectangle(prev,(x0,y0),(x1,y1),(0,255,0),1)
        cv2.rectangle(prev,(cx0,cy0),(cx1,cy1),(0,0,255),1)
    out_png = base + "_grid_preview.png"
    cv2.imwrite(out_png, prev)
    print(f"Saved grid preview: {out_png}")

if __name__ == "__main__":
    main()
