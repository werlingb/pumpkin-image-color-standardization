# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 21:05:16 2025

@author: werlingb
"""

# -*- coding: utf-8 -*-
"""
Batch color correction — FINAL rotation-robust version (v13d)
Works with any card rotation (0°, 90°, 180°, 270°) and keeps overlay aligned.
"""

import importlib, subprocess, sys, os, glob
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog

# --- Auto-install dependencies ---
def _ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(pkg)

np = _ensure("numpy")
cv2 = _ensure("cv2")

# ---------------------------------------------------
# GUI helpers
# ---------------------------------------------------
def ask_folder():
    Tk().withdraw()
    return filedialog.askdirectory(title="Select folder with pumpkin images")

def ask_npz():
    Tk().withdraw()
    return filedialog.askopenfilename(title="Select reference .npz",
                                      filetypes=[("NumPy Zip", "*.npz")])


# ---------------------------------------------------
# Corner selection
# ---------------------------------------------------
def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_card_corners(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    disp = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    clone = disp.copy()
    pts = []

    def click(e,x,y,f,p):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
            cv2.circle(clone,(x,y),3,(0,0,255),-1)
            cv2.imshow(win,clone)

    win = "Click 4 corners (any order), ENTER to confirm"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1400,clone.shape[1]), min(900,clone.shape[0]))
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, click)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(pts) >= 4:
            break
    cv2.destroyWindow(win)
    inv = 1.0/scale
    return order_corners([(int(x*inv),int(y*inv)) for (x,y) in pts[:4]])

# ---------------------------------------------------
# Core helpers
# ---------------------------------------------------
def warp_card(img, corners, target_wh):
    W,H = target_wh
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W,H))

def sample_patches_central40(warp_bgr, rows, cols):
    lab = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2LAB)
    H,W = lab.shape[:2]; ch,cw = H/rows, W/cols; frac=0.4
    vals,boxes=[],[]
    for r in range(rows):
        for c in range(cols):
            y0=int(r*ch); y1=int((r+1)*ch)
            x0=int(c*cw); x1=int((c+1)*cw)
            h=y1-y0; w=x1-x0
            cy0=y0+int((1-frac)/2*h); cy1=y1-int((1-frac)/2*h)
            cx0=x0+int((1-frac)/2*w); cx1=x1-int((1-frac)/2*w)
            roi=lab[cy0:cy1,cx0:cx1]
            vals.append([np.median(roi[...,0]),np.median(roi[...,1]),np.median(roi[...,2])])
            boxes.append((cx0,cy0,cx1,cy1))
    return np.array(vals,np.float32),boxes

def rotate_samples(samples, rows, cols, k):
    grid = samples.reshape(rows, cols, 3)
    rot = np.rot90(grid, k, axes=(0,1))
    r2,c2 = (rows,cols) if k%2==0 else (cols,rows)
    return rot.reshape(-1,3), r2, c2

def patch_lsq_map_Lab(src,dst):
    X = np.hstack([src,np.ones((src.shape[0],1),np.float32)])
    return np.linalg.lstsq(X,dst,rcond=None)[0]

def apply_Lab_map(img,M):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32)
    H,W=lab.shape[:2]
    X=np.hstack([lab.reshape(-1,3),np.ones((H*W,1),np.float32)])
    out=X@M
    lab_out=np.stack([
        np.clip(out[:,0],0,255).reshape(H,W),
        np.clip(out[:,1],0,255).reshape(H,W),
        np.clip(out[:,2],0,255).reshape(H,W)],axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab_out,cv2.COLOR_LAB2BGR)

def draw_boxes(img,boxes,color=(0,0,255)):
    vis=img.copy()
    for (x0,y0,x1,y1) in boxes:
        cv2.rectangle(vis,(x0,y0),(x1,y1),color,1)
    return vis

def side_by_side(a,b):
    h=max(a.shape[0],b.shape[0])
    pad=lambda im: cv2.copyMakeBorder(im,0,h-im.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    return np.hstack([pad(a),pad(b)])

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    print("===== Batch Color Correction v13d (Final rotation-robust) =====")
    folder = ask_folder()
    if not folder: return
    npz_path = ask_npz()
    if not npz_path: return

    ref = np.load(npz_path)
    ref_colors = ref["ref_colors"].astype(np.float32)  # (6,4,3)
    gray_idx = ref["gray_idx"]

    info_dir = os.path.join(folder, "correction_info")
    os.makedirs(info_dir, exist_ok=True)
    corrected_dir = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_corrected")
    os.makedirs(corrected_dir, exist_ok=True)
    images = sorted(sum([glob.glob(os.path.join(folder, ext))
                         for ext in ("*.jpg","*.jpeg","*.png")], []))
    if not images:
        print("No images found.")
        return

    log_path = os.path.join(info_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_path,"w",encoding="utf-8") as log:
        log.write("file,rotation,mirror,fit_rmse\n")

    for p in images:
        name = os.path.basename(p)
        print(f"\nProcessing: {name}")
        img = cv2.imread(p)
        if img is None:
            print("  !! could not read image"); continue

        corners = get_card_corners(img)
        (x0,y0),(x1,y1),(x2,y2),(x3,y3)=corners
        top_len  = np.hypot(x1-x0, y1-y0)
        left_len = np.hypot(x3-x0, y3-y0)
        if top_len > left_len:
            warp = warp_card(img, corners, (600, 400))
            base_rows, base_cols = 4, 6
        else:
            warp = warp_card(img, corners, (400, 600))
            base_rows, base_cols = 6, 4

        samples, rois = sample_patches_central40(warp, base_rows, base_cols)

        best = None
        # Try all 4 rotations
        for k in range(4):
            rotated, r, c = rotate_samples(samples, base_rows, base_cols, k)
            for mirror in [False, True]:
                grid = rotated.reshape(r,c,3)
                if mirror: grid = grid[:, ::-1, :]
                det = grid.reshape(-1,3)
                M = patch_lsq_map_Lab(det, ref_colors.reshape(-1,3))
                X = np.hstack([det, np.ones((det.shape[0],1),np.float32)])
                rmse = float(np.sqrt(np.mean((X@M - ref_colors.reshape(-1,3))**2)))
                if best is None or rmse < best["rmse"]:
                    best = {"k":k,"mirror":mirror,"M":M,"rmse":rmse}

        corrected = apply_Lab_map(img, best["M"])

        # Save corrected and preview
        base, _ = os.path.splitext(p)
        out_img = os.path.join(corrected_dir, os.path.basename(base) + "_corrected.png")
        cv2.imwrite(out_img, corrected, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        preview = side_by_side(draw_boxes(warp, rois), 
                               cv2.resize(corrected, (warp.shape[1], warp.shape[0])))
        cv2.imwrite(os.path.join(info_dir, name.replace(".jpg","_preview.jpg")), preview)

        with open(log_path,"a",encoding="utf-8") as log:
            log.write(f"{name},{best['k']},{best['mirror']},{best['rmse']:.3f}\n")

        print(f"  ✓ rotation={best['k']*90}°, mirror={'yes' if best['mirror'] else 'no'}, rmse={best['rmse']:.3f}")

    print("\nDone. Corrected images are next to originals.")
    print(f"Previews & log: {info_dir}")

if __name__ == "__main__":
    main()
