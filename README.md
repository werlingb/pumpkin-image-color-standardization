**pumpkin-image-color-standardization**

Python scripts to standardize color in pumpkin images using a photographed color checker, extract Lab color metrics, and analyze variety-level color differences. Designed for reproducible image-based color analysis in applied horticultural research.

While these scripts standardize image processing steps, users are responsible for ensuring appropriate image collection, evaluating data quality, and interpreting results within the context of their experimental design.

**Table of Contents**
## Table of Contents

1. Overview
2. Why use this workflow
3. Color checker and color model
   1. Color checker requirements
   2. Patch order and compatibility
   3. Color space and correction method
4. Repository contents
   1. Core scripts
   2. Helper scripts
5. File placement and launch options
   1. Recommended folder structure
   2. Windows `.bat` launchers
6. Step-by-step: running the core workflow
   1. Creating a reference color profile
   2. Taking a good reference photo
   3. Applying color correction (when needed)
   4. Extracting pumpkin color values
   5. Analyzing and summarizing color differences
7. Intended use and limitations
8. Code availability
9. Software and system requirements
   1. Software requirements
   2. Adding Python to PATH
   3. Required Python packages

**1.  Overview**

This repository contains Python scripts developed to help researchers make fair, repeatable color comparisons
among pumpkin varieties using digital photographs.

Color in images often varies due to lighting, camera settings, and shooting conditions.
These scripts allow users to:

-standardize color across images using a photographed color checker,

-extract consistent color measurements from pumpkins, and

-summarize and visualize color differences among varieties.

The workflow is modular: users may run only the steps that fit their needs.

**2. Why use this workflow**

You might use this pipeline if you want to:

-Compare pumpkin color across varieties without lighting bias

-Ensure images taken on different days or cameras are directly comparable

-Convert subjective color impressions into numeric values

-Create reproducible figures and summaries for reports or publications

This approach may be useful for variety trials, postharvest studies, and applied horticultural research.

**3. Color checker and color model**

***3.1 Color checker requirements***

This workflow assumes a 24-patch color checker arranged in a 6 × 4 grid.

Your card will work if it has 24 distinct color patches arranged in a regular 6 × 4 grid layout; uniform spacing (rows and columns align); neutral gray/black/white patches and saturated colors (reds, yellows, blues, greens).

What does not matter:

-Exact physical size (it does not need to be 4 × 6 inches)

-Brand name (Pixiss, X-Rite–style, or similar are acceptable)

Mixing different color checker brands within a dataset is not recommended.

***3.2 Patch order and compatibility***

The scripts assume:

-Each patch has a fixed position relative to the others

-Rotating or mirroring the card does not change internal patch order

Cards with irregular layouts, nonstandard patch ordering, or mixed patch sizes may not
be compatible without modifying the scripts.

Rule of thumb:
If your card looks like a “standard 24-patch color checker,” it will almost certainly work.

***3.3 Color space and correction method***

All color calculations are performed in CIELAB (L*a*b*) color space. CIELAB (L*a*b*) color space is a perceptually based color model designed so that numerical differences correspond more closely to how humans perceive color differences. L* represents lightness, a* represents the green–red axis, and b* represents the blue–yellow axis. Separating lightness from color makes CIELAB well suited for quantitative color comparison and analysis. In this workflow, Lab values are computed using OpenCV’s implementation, which follows the CIELAB color model but uses an 8-bit scaled representation. Using OpenCV’s scaled Lab representation simplifies image processing, avoids negative values, and allows consistent, efficient comparison of color differences across large image sets. Because OpenCV uses a scaled representation of CIELAB rather than true CIE units, this workflow is intended for relative color comparison among samples; it may be less appropriate for applications requiring absolute colorimetric calibration or direct comparison to published CIE L*a*b* values.

CIELAB is used because:
-it separates lightness (L*) from color (a*, b*)
-differences correspond more closely to human perception

Conceptually, color correction works as follows:
-Known colors are sampled from the photographed color checker card placed in the picture along with the pumpkins
-These values are compared to a reference image taken in even lighting
-A linear least-squares correction is calculated to map colors in the pumpkin image to the reference image

The correction is applied to the entire image. 

Median values and central patch regions from cards are used to reduce noise and glare effects.

**4. Repository contents**

***4.1 Core scripts***

create_reference_npz_fixed_window_v3.py — creates a reference color profile from a photographed color checker

batch_color_correct_v13c.py — applies rotation-robust color correction to image folders using the reference profile

extract_pumpkin_colors_v6_flex_poly_or_full.py — extracts median Lab color values from pumpkins

pumpkin_color_analysis_final.py — summarizes and analyzes variety-level color differences using PCA and clustering


***4.2 Helper scripts***

These scripts are not required for reproducibility but aid visualization:

create_color_swatches_from_csv.py — creates labeled color swatches

resize_pairs_by_object_png.py — resizes images for visual comparison

luminance_match_to_original.py — optional brightness matching


**5. File placement and launch options**
***5.1 Recommended folder structure***
A typical project folder should contain:

- **scripts/** — Python scripts and Windows `.bat` launchers
- **images_raw/** — Original images containing the color checker
- **images_corrected/** — Color-corrected images generated by the workflow
- **color_data/** — CSV outputs from color extraction
- **analysis_results/** — Plots, tables, and summary outputs

You only need to create a project folder with subfolders called "scripts" and "images_raw", the scripts will create others automatically within your project folder.

You can use whatever project folder name you want.

Scripts use file-selection dialogs; paths do not need to be hard-coded.

***5.2 Windows .bat launchers***

Batch files, which end with the extension .bat, allow scripts to be run by double-clicking, instead of through a console.

Placement:
Each .bat file should be placed in the same folder as its corresponding .py script. 


**6. Step-by-step: running the core workflow**

***6.1 Creating a reference color profile***

**-Script:** create_reference_npz_fixed_window_v3.py

**-Launcher:** create_reference_npz_launcher.bat

Used once per project to define what “correct color” looks like for a specific
color checker, camera, and lighting setup.

Outputs:

*_ref.npz — reference color profile

*_grid_preview.png — diagnostic preview of sampled regions

***6.2 Taking a good reference photo***

A good reference photo should have:
-the entire color checker visible
-flat placement (not warped)
-even, diffuse lighting
-minimal glare and no hard shadows
-sufficient resolution to clearly resolve patches
-Diffuse light (overcast daylight, shade, or diffused indoor lighting) is strongly recommended.

After creating the reference, review the _grid_preview.png file to confirm
that sampling regions avoid glare, shadows, and patch edges.

***6.3 Applying color correction (recommended, but optional)***

**-Script:** batch_color_correct_v13c.py

**-Launcher:** launch_color_correct_v13c.bat

Use when images were taken under variable lighting, or on different days. 
Color correction may be skipped if all images were taken under tightly controlled,
identical lighting.

Outputs:

*a folder of corrected images called: <input_folder name>_corrected/

*correction_info/ (previews and RMSE log)

***6.4 Extracting pumpkin color values***

**-Script:** extract_pumpkin_colors_v6_flex_poly_or_full.py

**-Launcher:** run_extract_pumpkin_colors_v6.bat

Converts images into numeric Lab color data.

Color-corrected images are recommended but not required.

Raw images may be used if lighting is consistent.

Supports:

-Full-image mode: Analyze color of an entire image, use if you take an image where the whole frame is filled by a single pumpkin with nothing else in the field of view; e.g., a zoomed in picture of a section of a pumpkin.

-Polygon mode: Multiple pumpkins per image, use if you have multiple pumpkins in an image. Allows you to outline each pumpkin and save color data for each as a separate row. By outlining each pumpkin you define the region you want color analyses for and omit the background.

Outputs:

*A csv file with CIELAB color values for each pumpkin: <image_folder>_color_data/color_summary.csv

***6.5 Analyzing and summarizing color differences***

**-Script:** pumpkin_color_analysis_final.py

**-Launcher:** run_pumpkin_color_analysis.bat

Generates PCA results, clustering summaries, plots, and tables, to help users describe and visualize differences in color.

***7. Intended use and limitations***

Designed for controlled photography with a visible color checker

Best suited for relative comparisons among varieties

Not intended for absolute color calibration or uncontrolled field imagery

**9. Software and system requirements**

***9.1 Software requirements***

-Python 3.9 or newer

-Python must be available on the system path (i.e., `python` runs from a command prompt)

***9.2 Adding Python to PATH***

For the provided `.bat` launchers to work, Python must be available on your system PATH
(i.e., the command `python` runs from a command prompt). You can do this a few ways.

A) Option 1 — During Python installation (recommended)
1. Download Python from https://www.python.org
2. Run the installer
3. On the **first screen**, check the box:
   **“Add Python to PATH”**
4. Click **Install Now**
5. Restart your computer after installation

This is the simplest and most reliable approach.

B) Option 2 — Verify that Python is on PATH
1. Open **Command Prompt**
2. Type: python --version

3. If a version number appears, Python is already on PATH and no action is needed.

If you see an error such as *“python is not recognized as an internal or external command”*,
Python is not yet on PATH.

C) Option 3 — Manually add Python to PATH (advanced)
If Python is installed but not on PATH:
1. Open **Start → Settings → System → About**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Under **System variables**, select **Path** → **Edit**
5. Add the folder containing `python.exe`  
(commonly something like `C:\Users\YourName\AppData\Local\Programs\Python\Python39\`)
6. Click **OK** to save and restart your computer

After this, `python --version` should work from a command prompt.

***9.3 Required Python packages***
The core scripts rely on commonly used scientific Python packages, including:
- `numpy`
- `pandas`
- `opencv-python` (cv2)
- `matplotlib`
- `scikit-learn`

Most scripts will attempt to install missing packages automatically when run.

### Notes
- No prior experience with Git or the command line is required when using the provided `.bat` launchers.
- Users running the scripts outside Windows may need to install dependencies manually using `pip`.

