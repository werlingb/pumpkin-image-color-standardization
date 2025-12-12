# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:40:12 2025

@author: werlingb
"""

# luminance_match_to_original.py
# Adjusts luminance of color-corrected images to match originals
# - Browse to folder with originals, then folder with corrected images
# - Outputs adjusted PNGs into a new "_lumadj" folder next to corrected folder
# - Creates CSV log with before/after L* and scale factor

import importlib, subprocess, sys, os, glob, csv
def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

np = _ensure("numpy")
cv2 = _ensure("cv2")
from tkinter import Tk, filedialog

# --- GUI ---
def ask_folder(title):
    Tk().withdraw()
    return filedialog.askdirectory(title=title)

def get_mean_L(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[...,0]))

def main():
    print("=== Luminance Match to Original ===")
    orig_folder = ask_folder("Select folder with ORIGINAL (uncorrected) images")
    if not orig_folder: return
    corr_folder = ask_folder("Select folder with COLOR-CORRECTED images")
    if not corr_folder: return

    # Create output folder alongside corrected folder
    parent_dir = os.path.dirname(corr_folder)
    base_name = os.path.basename(corr_folder.rstrip(os.sep))
    out_folder = os.path.join(parent_dir, base_name + "_lumadj")
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output folder: {out_folder}")

    results_log = []

    # Gather files
    originals = sorted(glob.glob(os.path.join(orig_folder, "*.*")))
    corrected = sorted(glob.glob(os.path.join(corr_folder, "*.*")))

    for b in corrected:
        name = os.path.basename(b)
        prefix = name.split("_corrected")[0]
        # find matching original
        match = [f for f in originals if os.path.basename(f).startswith(prefix)]
        if not match:
            print(f"⚠️ No original match for {name}")
            continue

        orig_img = cv2.imread(match[0])
        corr_img = cv2.imread(b)
        if orig_img is None or corr_img is None:
            print(f"⚠️ Skipped unreadable: {name}")
            continue

        # Compute mean L*
        L_orig = get_mean_L(orig_img)
        L_corr = get_mean_L(corr_img)

        scale = L_orig / max(L_corr, 1e-6)
        scl_min, scl_max = 0.8, 1.2
        scale = float(np.clip(scale, scl_min, scl_max))

        # Apply scale to corrected image
        lab = cv2.cvtColor(corr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[...,0] = np.clip(lab[...,0] * scale, 0, 255)
        adj = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Save adjusted PNG
        base, _ = os.path.splitext(os.path.basename(b))
        out_path = os.path.join(out_folder, f"{base}_lumadj.png")
        cv2.imwrite(out_path, adj, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        results_log.append([name, f"{L_orig:.2f}", f"{L_corr:.2f}", f"{scale:.3f}"])
        print(f"✓ {name}: scale={scale:.3f} (Lorig={L_orig:.2f}, Lcorr={L_corr:.2f})")

    # --- Write CSV summary ---
    csv_path = os.path.join(out_folder, "luminance_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "L_original", "L_corrected", "scale_factor"])
        for rec in results_log:
            writer.writerow(rec)
    print(f"\n✓ Luminance summary saved: {csv_path}")
    print("Done! All adjusted PNGs are in the new folder.")

if __name__ == "__main__":
    main()
