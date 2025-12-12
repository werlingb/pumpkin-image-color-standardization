# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:22:21 2025

@author: werlingb
"""

# pumpkin_color_analysis_final_smallset.py
"""
Pumpkin Color Analysis Summary (robust version for small datasets)
------------------------------------------------------------------
Performs:
 1. Between/within-variety variance ratios for L,a,b
 2. PCA to identify main color variation axes
 3. K-means clustering with silhouette-based selection (k = 3–6)
    → deterministic seed, guardrail to avoid over-splitting small sets
 4. Lets user browse for color_summary.csv
 5. Saves results into a subfolder:  <CSV_folder>/_color_analysis_results/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, Tk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === 1. Select input file ===
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select your color_summary.csv file",
    filetypes=[("CSV files", "*.csv")]
)
if not file_path:
    print("No file selected. Exiting.")
    exit()

df = pd.read_csv(file_path)
for c in ["L", "a", "b"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# === 2. Prepare output folder ===
base_dir = os.path.dirname(file_path)
out_dir = os.path.join(base_dir, "_color_analysis_results")
os.makedirs(out_dir, exist_ok=True)

# === 3. Collapse to per-variety medians ===
var_medians = df.groupby("variety")[["L", "a", "b"]].median()

# === 4. Between/within variance ratios ===
grand_mean = df[["L", "a", "b"]].mean()
between_var = ((var_medians - grand_mean) ** 2).sum()
within_var = df.groupby("variety")[["L", "a", "b"]].var().mean()
var_ratio = (between_var / within_var).round(2)

# === 4b. 2D scatterplot of variety medians for most variable Lab axes, color-coded by clusters ===

# Determine which two color dimensions vary most across varieties
top_dims = var_ratio.sort_values(ascending=False).index[:2].tolist()

# Only plot clusters if they exist (i.e., after clustering step)
# We'll create the plot later, after KMeans is done, so just store the axes for now
median_plot_dims = top_dims

# === 5. PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(var_medians)
pca = PCA(n_components=3)
pca_coords = pca.fit_transform(X_scaled)

# === 5b. Save PCA loadings (feature contributions) ===
loadings = pd.DataFrame(
    pca.components_.T,
    index=["L", "a", "b"],
    columns=[f"PC{i+1}" for i in range(3)]
)
loadings = (loadings * 100).round(1)  # express as % contribution for readability
loadings.to_csv(os.path.join(out_dir, "pca_loadings.csv"))
# === 5c. Automatic data-driven interpretation of PCA loadings ===
interpretation_lines = []
for i, col in enumerate(loadings.columns):
    pc_name = col
    sorted_vars = loadings[col].abs().sort_values(ascending=False)
    top_vars = sorted_vars.index[:2].tolist()  # top 2 contributing variables
    signs = ["+" if loadings.loc[v, col] >= 0 else "−" for v in top_vars]
    # short directional phrase
    direction = " and ".join([f"{v}({s})" for v, s in zip(top_vars, signs)])
    main_driver = top_vars[0]
    interpretation_lines.append(
        f"{pc_name}: mainly driven by {direction}, indicating that this component reflects variation along the {main_driver} axis."
    )

pca_interpretation_text = (
    "Automatic PCA interpretation (based on loadings):\n"
    + "\n".join([f" - {l}" for l in interpretation_lines])
    + "\n\n"
)

# save PCA coordinates and variance
pca_df = pd.DataFrame(
    pca_coords, index=var_medians.index, columns=[f"PC{i+1}" for i in range(3)]
)
pca_df.to_csv(os.path.join(out_dir, "pca_results.csv"))
pca_var = (pca.explained_variance_ratio_ * 100).round(2)
pd.DataFrame(
    {"Component": [f"PC{i+1}" for i in range(3)], "ExplainedVariance(%)": pca_var}
).to_csv(os.path.join(out_dir, "pca_variance.csv"), index=False)

# === 6. Evaluate clustering (k = 3–6) ===
k_values = range(3, 7)
inertias, sil_scores = [], []
best_k = 3
best_score = -1

for k in k_values:
    # deterministic clustering
    km = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X_scaled)
    inertias.append(km.inertia_)
    score = silhouette_score(X_scaled, km.labels_)
    sil_scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k

# --- guardrail: keep 3 clusters unless silhouette gain > 0.02 over k=3 ---
if best_k > 3 and (best_score - sil_scores[0]) < 0.02:
    best_k = 3
    best_score = sil_scores[0]

# === 7. Plot elbow & silhouette (true values, not normalized) ===
fig, ax1 = plt.subplots(figsize=(7, 5))

# plot inertia (left axis)
color1 = "darkorange"
ax1.plot(k_values, inertias, "o-", color=color1, label="Inertia (Elbow)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (lower = tighter clusters)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

# plot silhouette on right axis
ax2 = ax1.twinx()
color2 = "teal"
ax2.plot(k_values, sil_scores, "s--", color=color2, label="Silhouette")
ax2.set_ylabel("Silhouette score (higher = better separation)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0, 1)  # silhouette is always between 0–1

plt.title(f"Elbow & Silhouette for Pumpkin Color Clustering (chosen k = {best_k})")

# combine legends cleanly
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "elbow_silhouette.png"), dpi=300)
plt.close()


# === 8. Final K-means ===
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50).fit(X_scaled)
var_medians["cluster"] = kmeans.labels_

# === 9. Summarize clusters ===
cluster_summary = var_medians.groupby("cluster")[["L", "a", "b"]].mean().round(1)
cluster_members = (
    var_medians.reset_index().groupby("cluster")["variety"].apply(list)
)

rows = []
for clust in cluster_summary.index:
    Lmean, amean, bmean = cluster_summary.loc[clust]
    varieties = ", ".join(cluster_members.loc[clust])
    rows.append(
        {
            "Cluster": clust,
            "Mean_L": Lmean,
            "Mean_a": amean,
            "Mean_b": bmean,
            "Varieties": varieties,
        }
    )
cluster_table = pd.DataFrame(rows)
cluster_table.to_csv(os.path.join(out_dir, "cluster_groups.csv"), index=False)
# === 9b. Separate lightness and hue descriptors per cluster ===
desc_rows = []
L_vals = cluster_summary["L"]

# Dynamic thresholds for lightness
Lmin, Lmax = L_vals.min(), L_vals.max()
Llow = Lmin + 0.33 * (Lmax - Lmin)
Lhigh = Lmin + 0.66 * (Lmax - Lmin)

def lightness_label(L):
    if L <= Llow:
        return "very dark"
    elif L <= Lhigh:
        return "medium"
    else:
        return "light"

# Hue descriptor logic (only call reddish/yellowish if visibly different)
def hue_label(a, b):
    diff = a - b
    if abs(diff) > 15:   # only label if the hue difference is big
        return "reddish" if diff > 0 else "yellowish"
    else:
        return "balanced"

for clust in cluster_summary.index:
    Lmean = cluster_summary.loc[clust, "L"]
    amean = cluster_summary.loc[clust, "a"]
    bmean = cluster_summary.loc[clust, "b"]
    varieties = ", ".join(cluster_members.loc[clust])

    lightness_desc = lightness_label(Lmean)
    hue_desc = hue_label(amean, bmean)

    desc_rows.append({
        "Cluster": clust,
        "Mean_L": round(Lmean, 1),
        "Mean_a": round(amean, 1),
        "Mean_b": round(bmean, 1),
        "Lightness_Descriptor": lightness_desc,
        "Hue_Descriptor": hue_desc,
        "Descriptor": f"{lightness_desc} orange (darker vs lighter tones mainly reflect both lightness and redness)",
        "Example_varieties": varieties
    })

desc_table = pd.DataFrame(desc_rows)
desc_table.to_csv(os.path.join(out_dir, "cluster_descriptors.csv"), index=False)

# --- ensure distinct wording ---
desc_table = pd.DataFrame(desc_rows).sort_values("Mean_L")
seen = set()
for i, row in desc_table.iterrows():
    desc = row["Descriptor"]
    if desc in seen:
        # Make it distinct based on relative lightness/hue
        if "light" in desc and "very light" not in desc:
            desc = desc.replace("light", "very light")
        elif "medium" in desc:
            desc = desc.replace("medium", "medium-dark")
        elif "dark" in desc and "very dark" not in desc:
            desc = desc.replace("dark", "very dark")
        else:
            desc += " tint"
        desc_table.at[i, "Descriptor"] = desc
    seen.add(desc)

desc_table.to_csv(os.path.join(out_dir, "cluster_descriptors.csv"), index=False)

# === 9c. Final polished color swatch visualization (sorted, readable, wrapped) ===
import matplotlib.patches as mpatches
import skimage.color as color
import numpy as np
import textwrap

swatch_path = os.path.join(out_dir, "cluster_color_swatches.png")

# Sort clusters darkest → lightest
desc_sorted = desc_table.sort_values("Mean_L")

# Configure figure height dynamically
row_height = 1.0
fig_height = len(desc_sorted) * row_height + 0.5
fig, ax = plt.subplots(figsize=(9, fig_height))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(desc_sorted))
ax.axis("off")

for i, (_, row) in enumerate(desc_sorted.iterrows()):
    L = row["Mean_L"]
    a = row["Mean_a"]
    b = row["Mean_b"]

    # Convert OpenCV → Lab if needed
    if L > 100:
        L_scaled = L / 2.55
        a_scaled = a - 128
        b_scaled = b - 128
    else:
        L_scaled = L
        a_scaled = a
        b_scaled = b

    rgb = color.lab2rgb(np.array([[[L_scaled, a_scaled, b_scaled]]]))[0, 0, :]
    rgb = np.clip(rgb, 0, 1)

    # Draw color swatch rectangle
    rect = mpatches.Rectangle((0.05, i), 0.25, 0.7, color=rgb, transform=ax.transData)
    ax.add_patch(rect)

    # Prepare text lines
    descriptor = f"Cluster {int(row['Cluster'])}: {row['Descriptor']}"
    varieties = row['Example_varieties']
    # truncate after ~60 chars, add linebreak if long
    if len(varieties) > 55:
        varieties_wrapped = textwrap.fill(varieties, width=55)
    else:
        varieties_wrapped = varieties

    ax.text(0.33, i + 0.45, descriptor, fontsize=9, va='center', ha='left')
    ax.text(0.33, i + 0.10, varieties_wrapped, fontsize=8, va='center', ha='left', color='dimgray')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(swatch_path, dpi=200, bbox_inches="tight")
plt.close()

# === 10. PCA scatterplot colored by clusters ===
colors = ["#e66101", "#fdb863", "#b2abd2", "#5e3c99", "#66c2a5", "#fc8d62"]
fig, ax = plt.subplots(figsize=(7, 5))
for i, (variety, row) in enumerate(var_medians.iterrows()):
    ax.scatter(
        pca_coords[i, 0],
        pca_coords[i, 1],
        color=colors[int(row["cluster"]) % len(colors)],
        s=100,
        edgecolor="k",
    )
    ax.text(pca_coords[i, 0] + 0.05, pca_coords[i, 1], variety, fontsize=8)
ax.set_xlabel(f"PC1 ({pca_var[0]:.1f}% var.)")
ax.set_ylabel(f"PC2 ({pca_var[1]:.1f}% var.)")
ax.set_title(f"PCA of Pumpkin Variety Color (k = {best_k})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pca_clusters.png"), dpi=300)
plt.close()

# === 12. 2D Lab scatterplot (medians) color-coded by cluster ===
colors = ["#e66101", "#fdb863", "#b2abd2", "#5e3c99", "#66c2a5", "#fc8d62"]

fig, ax = plt.subplots(figsize=(6, 5))
for c in sorted(var_medians["cluster"].unique()):
    subset = var_medians[var_medians["cluster"] == c]
    ax.scatter(
        subset[median_plot_dims[0]],
        subset[median_plot_dims[1]],
        color=colors[c % len(colors)],
        s=120,
        edgecolor="k",
        label=f"Cluster {c}"
    )
    for v in subset.index:
        ax.text(
            subset.loc[v, median_plot_dims[0]] + 0.15,
            subset.loc[v, median_plot_dims[1]],
            v,
            fontsize=8
        )

ax.set_xlabel(median_plot_dims[0])
ax.set_ylabel(median_plot_dims[1])
ax.set_title(f"Variety medians in color space ({median_plot_dims[0]} vs {median_plot_dims[1]})")
ax.legend(fontsize=8, frameon=False, loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, f"variety_medians_{median_plot_dims[0]}_{median_plot_dims[1]}_byCluster.png"),
    dpi=300
)
plt.close()

# === 13. Write interpretive summary ===
summary_path = os.path.join(out_dir, "color_analysis_summary.txt")
with open(summary_path, "w") as f:
    f.write("Pumpkin Color Analysis Summary\n")
    f.write("---------------------------------\n\n")
    f.write("Between/within-variety variance ratios:\n")
    for col in ["L", "a", "b"]:
        f.write(f"  {col}: {var_ratio[col]:.2f}\n")
    f.write("\nInterpretation:\n")
    f.write(" - Ratios >10 indicate strong, consistent variety-level color differences.\n")
    f.write(" - Higher ratios highlight which axis (L,a,b) drives most separation.\n")
    f.write(" - Typically, L (lightness) and a (orange/redness) dominate pumpkin color variation.\n\n")
    f.write("PCA results (see pca_results.csv and pca_variance.csv):\n")
    f.write(f" - PC1 explains {pca_var[0]:.1f}%, PC2 {pca_var[1]:.1f}% of variance.\n")
    f.write("   → These primarily describe a gradient from dark to light orange tones.\n\n")
    f.write(pca_interpretation_text)
    f.write(f"K-means clustering (robust, small-sample adjusted): chosen k = {best_k}\n")
    f.write(f" - Silhouette peak score = {best_score:.3f}\n")
    f.write(" - Normalized inertia and silhouette curves plotted in elbow_silhouette.png.\n")
    f.write(" - Cluster membership and mean colors saved to cluster_groups.csv.\n\n")
    f.write("PCA scatterplot colored by clusters saved as pca_clusters.png.\n\n")
    f.write("✓ Outputs generated in folder:\n")
    f.write(f"   {out_dir}\n")

print(f"\n✓ Analysis complete (chosen k = {best_k}, silhouette = {best_score:.3f}).")
print(f"Results saved in: {out_dir}")
