# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:35:34 2025

@author: werlingb
"""

# -*- coding: utf-8 -*-
"""
Extract median Lab colors from pumpkins (flex: polygon or full image)
- Option to draw polygons (multiple pumpkins per image) or analyze full image
- Saves each polygon as one row in CSV
- Optional preview image with outlines
- Auto-installs numpy & OpenCV if missing
"""

import importlib, subprocess, sys, os, csv, glob
from tkinter import Tk, filedialog, simpledialog

# --- auto install ---
def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

np = _ensure("numpy")
cv2 = _ensure("cv2")

# --- pick folder ---
def ask_folder():
    Tk().withdraw()
    return filedialog.askdirectory(title="Select folder with pumpkin images")

# --- choose analysis mode ---
def ask_mode():
    Tk().withdraw()
    ans = simpledialog.askstring("Analysis mode",
        "Type 'poly' to draw polygons around pumpkins\n"
        "or 'full' to analyze the entire image.\n(Default = full)")
    if ans is None:
        return "full"
    ans = ans.strip().lower()
    return "poly" if ans.startswith("p") else "full"

# --- color extraction ---
def get_lab_median(img_bgr, mask=None):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if mask is not None:
        L = np.median(lab[...,0][mask>0])
        a = np.median(lab[...,1][mask>0])
        b = np.median(lab[...,2][mask>0])
    else:
        L = np.median(lab[...,0])
        a = np.median(lab[...,1])
        b = np.median(lab[...,2])
    return float(L), float(a), float(b)

# --- polygon drawing ---
def get_polygons(img):
    """
    Lets user draw one or more polygons on the image.
    - Left click = add point
    - ENTER = finish current polygon
    - D = done with all polygons (proceeds to next image)
    - R = reset current polygon
    """
    clone = img.copy()
    current = []
    polys = []

    win = "Draw polygon(s): ENTER=finish polygon, D=done, R=reset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1400, clone.shape[1]), min(900, clone.shape[0]))

    def click(event, x, y, flags, param):
        nonlocal current, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            # orange points
            cv2.circle(clone, (x, y), 6, (0, 165, 255), -1)
            # yellow connecting lines
            if len(current) > 1:
                cv2.line(clone, current[-2], current[-1], (0, 255, 255), 2)
            cv2.imshow(win, clone)

    cv2.setMouseCallback(win, click)

    while True:
        cv2.imshow(win, clone)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(current) >= 3:  # ENTER = confirm polygon
            polys.append(np.array(current, np.int32))
            current = []
            clone = img.copy()
            # draw all completed polys
            for poly in polys:
                cv2.polylines(clone, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow(win, clone)
        elif k in [ord('r'), ord('R')]:  # reset current polygon
            current = []
            clone = img.copy()
            for poly in polys:
                cv2.polylines(clone, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow(win, clone)
        elif k in [ord('d'), ord('D')]:  # done
            break

    cv2.destroyWindow(win)
    return polys


# --- main ---
def main():
    print("===== Pumpkin Color Extractor v6 (Polygon or Full Image) =====")
    folder = ask_folder()
    if not folder:
        print("No folder selected.")
        return

    mode = ask_mode()
    print(f"Mode: {mode.upper()}")

    out_dir = os.path.join(os.path.dirname(folder),
                           os.path.basename(folder) + "_color_data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "color_summary.csv")

    images = sorted(sum([glob.glob(os.path.join(folder, ext))
                         for ext in ("*.jpg","*.jpeg","*.png")], []))
    if not images:
        print("No images found.")
        return

    print(f"Processing {len(images)} image(s)...\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variety","image_name","region_id","L","a","b","method"])

        for path in images:
            name = os.path.basename(path)
            variety = name.split("_")[0]
            img = cv2.imread(path)
            if img is None:
                print(f"  !! Skipped unreadable file: {name}")
                continue

            if mode == "poly":
                masks = get_polygons(img)
                preview = img.copy()
                for i, m in enumerate(masks):
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [m], 1)
                    L, a, b = get_lab_median(img, mask)
                    writer.writerow([variety, name, i+1, f"{L:.2f}", f"{a:.2f}", f"{b:.2f}", "median"])
                    cv2.polylines(preview, [m], isClosed=True, color=(0,255,0), thickness=3)
                prev_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_preview.jpg")
                cv2.imwrite(prev_path, preview)
                print(f"  {name}: {len(masks)} regions processed")
            else:
                L,a,b = get_lab_median(img)
                writer.writerow([variety, name, 1, f"{L:.2f}", f"{a:.2f}", f"{b:.2f}", "median"])
                print(f"  {name}: Full image analyzed")

    print(f"\n✓ Done! Results saved to: {out_csv}")
    print(f"Diagnostic previews (if any): {out_dir}")

if __name__ == "__main__":
    main()
