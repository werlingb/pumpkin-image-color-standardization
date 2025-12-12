# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:28:05 2025

@author: werlingb
"""

# -*- coding: utf-8 -*-
"""
Generate color swatches from pumpkin color CSV (adds RGB + HEX columns)
 - Reads color_summary.csv with L,a,b columns
 - Groups by variety, computes median of each Lab channel
 - Converts to RGB + Hex for easy reproducibility
 - Creates labeled PNG swatches and swatch_summary.csv
"""

import importlib, subprocess, sys, os, csv
import numpy as np

# --- auto install ---
def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

cv2 = _ensure("cv2")
pandas = _ensure("pandas")

# --- main ---
def main():
    print("===== Pumpkin Swatch Generator (RGB + HEX enhanced) =====")
    from tkinter import Tk, filedialog
    Tk().withdraw()
    csv_path = filedialog.askopenfilename(title="Select color_summary.csv",
                                          filetypes=[("CSV files","*.csv")])
    if not csv_path:
        print("No file selected.")
        return

    df = pandas.read_csv(csv_path)
    if not {"variety","L","a","b"}.issubset(df.columns):
        print("CSV missing expected columns (needs variety, L, a, b).")
        return

    # ensure numeric
    df["L"] = pandas.to_numeric(df["L"], errors="coerce")
    df["a"] = pandas.to_numeric(df["a"], errors="coerce")
    df["b"] = pandas.to_numeric(df["b"], errors="coerce")

    # output folder
    base_dir = os.path.dirname(csv_path)
    out_dir = os.path.join(os.path.dirname(base_dir),
                           os.path.basename(base_dir) + "_swatches")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    print("Generating swatches...")

    for variety, grp in df.groupby("variety"):
        med = grp[["L","a","b"]].median().to_numpy().astype(np.float32)

        # convert Lab→RGB (OpenCV expects 1×1×3 uint8)
        lab_patch = np.uint8([[med]])
        rgb_patch = cv2.cvtColor(lab_patch, cv2.COLOR_Lab2RGB)[0,0]
        r,g,b = [int(x) for x in rgb_patch]
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        n = len(grp)

        # make swatch
        sw = np.full((200,200,3), rgb_patch, np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{variety}_swatch.png"), cv2.cvtColor(sw, cv2.COLOR_RGB2BGR))

        summary_rows.append([variety, n,
                             f"{med[0]:.2f}", f"{med[1]:.2f}", f"{med[2]:.2f}",
                             r,g,b,hex_color])

    # write summary csv
    out_csv = os.path.join(out_dir, "swatch_summary.csv")
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variety","n_samples","L_median","a_median","b_median","R","G","B","Hex"])
        w.writerows(summary_rows)

    print(f"\n✓ Done! {len(summary_rows)} swatches created.")
    print(f"Output folder: {out_dir}")

if __name__ == "__main__":
    main()
