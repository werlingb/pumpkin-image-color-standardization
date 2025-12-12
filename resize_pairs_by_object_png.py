# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:29:02 2025

@author: werlingb
"""

# -*- coding: utf-8 -*-
"""
Resize image pairs so a reference object (e.g., basketball) matches size.
Handles single-image varieties gracefully. Outputs PNGs + CSV log.
"""

import importlib, subprocess, sys, os, glob, csv
import numpy as np
import cv2
from tkinter import Tk, filedialog

# --- auto-install ---
def _ensure(pkg):
    try: return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

np = _ensure("numpy")
cv2 = _ensure("cv2")

# --- helpers ---
def ask_folder():
    Tk().withdraw()
    return filedialog.askdirectory(title="Select folder with pumpkin image(s)")

def click_two_points(img, title):
    clone = img.copy()
    pts = []
    def click(e,x,y,f,p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
            cv2.circle(clone,(x,y),4,(0,0,255),-1)
            cv2.imshow(title, clone)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(1400, img.shape[1]), min(900, img.shape[0]))
    cv2.imshow(title, clone)
    cv2.setMouseCallback(title, click)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(pts) >= 2:
            break
    cv2.destroyWindow(title)
    return np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))

def resize_to_match(img_ref, img_to_scale, d_ref, d_img):
    scale = d_ref / max(d_img, 1e-6)
    new_size = (int(img_to_scale.shape[1]*scale), int(img_to_scale.shape[0]*scale))
    scaled = cv2.resize(img_to_scale, new_size, interpolation=cv2.INTER_CUBIC)
    return scaled, scale

def stack_side_by_side(a,b):
    h = max(a.shape[0], b.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im,0,h-im.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    return np.hstack([pad(a), pad(b)])

# --- main ---
def main():
    print("===== Pairwise Resize by Variety Name - PNG Output =====")
    folder = ask_folder()
    if not folder: return

    images = sorted(sum([glob.glob(os.path.join(folder, ext))
                         for ext in ("*.jpg","*.jpeg","*.png")], []))
    if not images:
        print("No images found.")
        return

    # group by variety name (before first underscore)
    groups = {}
    for path in images:
        name = os.path.basename(path)
        key = name.split("_")[0]
        groups.setdefault(key, []).append(path)

    out_dir = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_resized")
    os.makedirs(out_dir, exist_ok=True)
    info_dir = os.path.join(out_dir, "resize_info")
    os.makedirs(info_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "resize_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Variety", "Ref_Image", "Resized_Image", "Scale", "Ref_W", "Ref_H", "New_W", "New_H", "Note"])

        for variety, paths in sorted(groups.items()):
            print(f"\nProcessing variety: {variety}")
            if len(paths) == 1:
                # only one image — copy it over
                src = paths[0]
                base = os.path.splitext(os.path.basename(src))[0] + "_resized.png"
                dst = os.path.join(out_dir, base)
                img = cv2.imread(src)
                cv2.imwrite(dst, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                writer.writerow([variety, os.path.basename(src), "-", "-", img.shape[1], img.shape[0], "-", "-", "single image (copied)"])
                print("  ✓ single image copied")
                continue

            # two or more — process first two only
            a_path, b_path = sorted(paths)[:2]
            img_a = cv2.imread(a_path)
            img_b = cv2.imread(b_path)
            if img_a is None or img_b is None:
                print("  !! Could not read one of the images.")
                continue

            dA = click_two_points(img_a, f"Click 2 points across basketball in {os.path.basename(a_path)} (ENTER when done)")
            dB = click_two_points(img_b, f"Click 2 points across basketball in {os.path.basename(b_path)} (ENTER when done)")
            scaled_b, scale = resize_to_match(img_a, img_b, dA, dB)

            base_a = os.path.splitext(os.path.basename(a_path))[0] + "_resized.png"
            base_b = os.path.splitext(os.path.basename(b_path))[0] + "_resized.png"
            out_a = os.path.join(out_dir, base_a)
            out_b = os.path.join(out_dir, base_b)

            cv2.imwrite(out_a, img_a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(out_b, scaled_b, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            preview = stack_side_by_side(img_a, scaled_b)
            prev_path = os.path.join(info_dir, f"{variety}_preview.png")
            cv2.imwrite(prev_path, preview)

            writer.writerow([variety, os.path.basename(a_path), os.path.basename(b_path),
                             f"{scale:.4f}", img_a.shape[1], img_a.shape[0],
                             scaled_b.shape[1], scaled_b.shape[0], "paired & resized"])
            print(f"  ✓ scaled {os.path.basename(b_path)} by {scale:.3f}")

    print(f"\nAll done! Resized images + CSV saved to:\n{out_dir}")

if __name__ == "__main__":
    main()
