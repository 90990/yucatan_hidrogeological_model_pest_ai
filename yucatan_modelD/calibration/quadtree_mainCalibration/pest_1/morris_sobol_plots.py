import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# CONFIG
# =========================
base = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\calibration\quadtree_mainCalibration\pest_1")
outdir = base / "morris_sobol_figs_pretty_png"
outdir.mkdir(parents=True, exist_ok=True)

morris_head_path = base / "gsa_morris1_rmse_head.csv"
morris_sgd_path  = base / "gsa_morris1_rmse_sgd.csv"

assert morris_head_path.exists(), f"Missing {morris_head_path}"
assert morris_sgd_path.exists(), f"Missing {morris_sgd_path}"

DPI = 300  # solo PNG

# =========================
# DATA
# =========================
m_head = pd.read_csv(morris_head_path)
m_sgd  = pd.read_csv(morris_sgd_path)

s_head = pd.DataFrame([
    ("mk_rest_l2", 0.403406, 0.189239, 6.137928e-01, 3.493437e-01),
    ("mr_rest",    0.253983, 0.205829, 5.536033e-01, 4.068526e-01),
    ("mr_upland",  0.029401, 0.058685, 4.366395e-02, 3.073806e-02),
    ("mfgw_rest", -0.008528, 0.015321, 1.637040e-02, 2.762547e-02),
    ("mk_rest_l1",  0.019813, 0.028149, 6.983721e-03, 5.539844e-03),
    ("mk_upland_l2",-0.001794,0.006151, 1.670077e-03, 1.190282e-03),
    ("mk_ring_l2",  0.001713, 0.003233, 1.268231e-04, 8.332468e-05),
    ("mk_upland_l1",-0.000205,0.001235, 7.373562e-05, 6.133177e-05),
    ("mk_ring_l1", -0.000721,0.001337, 8.416120e-06, 4.354098e-06),
    ("mfgw_upland", 0.000002,0.000023, 3.813407e-08, 3.191595e-08),
], columns=["parnme","S1","S1_conf","ST","ST_conf"])

s_sgd = pd.DataFrame([
    ("mr_rest",     9.912649e-01, 0.180173, 9.400458e-01, 1.747495e-01),
    ("mr_upland",   6.449455e-02, 0.033472, 1.886264e-02, 4.647611e-03),
    ("mk_rest_l2",  8.761259e-03, 0.017926, 6.595739e-03, 5.668525e-03),
    ("mfgw_rest",   4.175643e-03, 0.015035, 3.935229e-03, 4.534858e-03),
    ("mk_rest_l1",  1.066295e-03, 0.002837, 1.434394e-04, 1.967621e-04),
    ("mk_ring_l2",  1.591510e-03, 0.003407, 1.332769e-04, 4.205846e-05),
    ("mk_ring_l1", -2.256582e-05, 0.000664, 5.040216e-06, 1.688714e-06),
    ("mk_upland_l2",-4.771006e-05, 0.000225, 1.014941e-06, 1.025541e-06),
    ("mk_upland_l1",-5.053255e-06, 0.000031, 4.273954e-08, 3.512426e-08),
    ("mfgw_upland", 6.073515e-07, 0.000013, 4.600483e-09, 3.757671e-09),
], columns=["parnme","S1","S1_conf","ST","ST_conf"])

# =========================
# STYLE HELPERS
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

ZONE_COLORS = {
    "rest":   "tab:blue",
    "upland": "tab:orange",
    "ring":   "tab:green",
    "other":  "tab:gray",
}
TYPE_MARKERS = {
    "mk": "o",
    "mr": "s",
    "mfgw": "^",
    "other": "D",
}

def safe_save_png(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            path = path.with_name(path.stem + "_new" + path.suffix)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

def parse_parnme(p: str):
    p = str(p).strip()
    parts = p.split("_")
    ptype = parts[0] if len(parts) >= 1 else "other"
    zone  = parts[1] if len(parts) >= 2 else "other"
    layer = parts[2] if len(parts) >= 3 else "all"
    if zone not in ("rest", "upland", "ring"):
        zone = "other"
    if ptype not in ("mk", "mr", "mfgw"):
        ptype = "other"
    return ptype, zone, layer

def add_style_columns(d: pd.DataFrame):
    tmp = d["parnme"].apply(lambda x: pd.Series(parse_parnme(x), index=["ptype","zone","layer"]))
    return pd.concat([d.reset_index(drop=True), tmp], axis=1)

def make_positive_for_log(arr: np.ndarray, eps: float = None):
    a = np.asarray(arr, dtype=float).copy()
    pos = a[a > 0]
    if eps is None:
        eps = (np.nanmin(pos) / 10.0) if pos.size else 1e-12
    a[a <= 0] = eps
    return a, eps

# =========================
# PLOTS (SIN ZOOM)
# =========================
def morris_plot_pretty_log(
    df: pd.DataFrame,
    title: str,
    out_png: Path,
    label_top: int = 6,
):
    d = df.copy()
    d["parnme"]  = d["parnme"].astype(str).str.strip()
    d["mu_star"] = pd.to_numeric(d["mu_star"], errors="coerce")
    d["sigma"]   = pd.to_numeric(d["sigma"], errors="coerce")
    d = d.dropna(subset=["mu_star","sigma"])
    d = add_style_columns(d)

    d["mu_star_logsafe"], epsx = make_positive_for_log(d["mu_star"].values)
    d["sigma_logsafe"],  epsy = make_positive_for_log(d["sigma"].values)

    fig, ax = plt.subplots(figsize=(8.7, 6.3), constrained_layout=True)

    for (zone, ptype), g in d.groupby(["zone","ptype"], sort=False):
        ax.scatter(
            g["mu_star_logsafe"], g["sigma_logsafe"],
            s=70,
            alpha=0.85,
            marker=TYPE_MARKERS.get(ptype, "o"),
            c=ZONE_COLORS.get(zone, "tab:gray"),
            edgecolors="k",
            linewidths=0.3
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Morris $\mu^\ast$ (influencia global) [log]")
    ax.set_ylabel(r"Morris $\sigma$ (no linealidad / interacciones) [log]")
    ax.set_title(title)

    ax.grid(True, which="both", alpha=0.25)
    ax.set_axisbelow(True)

    # Nota si hubo ajuste por log
    note = []
    if (d["mu_star"].values <= 0).any():
        note.append(rf"$\mu^\ast \le 0$ ajustado a $\epsilon={epsx:.1e}$")
    if (d["sigma"].values <= 0).any():
        note.append(rf"$\sigma \le 0$ ajustado a $\epsilon={epsy:.1e}$")
    if note:
        ax.text(0.02, 0.02, " | ".join(note), transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    # Etiquetar top por mu_star (original)
    top = d.sort_values("mu_star", ascending=False).head(label_top)
    xmax = float(d["mu_star_logsafe"].max()) if len(d) else 1.0
    ymax = float(d["sigma_logsafe"].max()) if len(d) else 1.0

    for _, r in top.iterrows():
        x, y = float(r["mu_star_logsafe"]), float(r["sigma_logsafe"])
        dx = -8 if x > 0.85*xmax else 8
        dy = -8 if y > 0.85*ymax else 8
        ha = "right" if dx < 0 else "left"
        va = "top" if dy < 0 else "bottom"
        ax.annotate(
            r["parnme"], (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha, va=va,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            clip_on=False
        )

    # Leyendas: color=zona, marker=tipo
    zone_handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=ZONE_COLORS[z], markeredgecolor="k",
               markersize=8, label=z)
        for z in ["rest","upland","ring","other"]
        if (d["zone"] == z).any()
    ]
    type_handles = [
        Line2D([0],[0], marker=TYPE_MARKERS[t], linestyle="None",
               color="k", markersize=8, label=t)
        for t in ["mk","mr","mfgw","other"]
        if (d["ptype"] == t).any()
    ]
    leg1 = ax.legend(handles=zone_handles, title="Zona", loc="lower right", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, title="Tipo", loc="upper left", frameon=True)

    out_png = safe_save_png(fig, out_png, dpi=DPI)
    plt.close(fig)
    return out_png

def sobol_st_barplot_pretty(
    df: pd.DataFrame,
    title: str,
    out_png: Path,
    top_n: int = 10,
):
    d = df.copy()
    d["parnme"]   = d["parnme"].astype(str).str.strip()
    d["ST"]       = pd.to_numeric(d["ST"], errors="coerce")
    d["ST_conf"]  = pd.to_numeric(d["ST_conf"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["ST"]).sort_values("ST", ascending=False).head(top_n)
    d = add_style_columns(d)

    fig, ax = plt.subplots(figsize=(9.3, 6.3), constrained_layout=True)

    y = np.arange(len(d))
    colors = [ZONE_COLORS.get(z, "tab:gray") for z in d["zone"].tolist()]

    ax.barh(
        y, d["ST"].values,
        xerr=d["ST_conf"].values,
        color=colors,
        edgecolor="k",
        linewidth=0.3,
        error_kw=dict(ecolor="0.25", lw=1.0, capsize=3, capthick=1.0),
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["parnme"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(r"Sobol $ST$ (efecto total)")
    ax.set_title(title)

    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = float((d["ST"] + d["ST_conf"]).max())
    ax.set_xlim(0, min(1.15, max(0.1, 1.05*xmax)))

    # números al final de cada barra
    xlim = ax.get_xlim()[1]
    for i, st in enumerate(d["ST"].values):
        ax.text(min(st + 0.01, xlim * 0.985), i, f"{st:.3f}", va="center", ha="left", fontsize=10)

    # leyenda por zona
    zone_handles = [
        Line2D([0],[0], marker="s", linestyle="None",
               markerfacecolor=ZONE_COLORS[z], markeredgecolor="k",
               markersize=9, label=z)
        for z in ["rest","upland","ring","other"]
        if (d["zone"] == z).any()
    ]
    ax.legend(handles=zone_handles, title="Zona", loc="lower right", frameon=True)

    out_png = safe_save_png(fig, out_png, dpi=DPI)
    plt.close(fig)
    return out_png

# =========================
# RUN
# =========================
paths = {}

paths["morris_head_png"] = morris_plot_pretty_log(
    m_head,
    "Diagrama de Morris (log-log): influencia global sobre RMSE grupo 'heads'",
    outdir / "fig_morris_head_loglog.png",
    label_top=6
)

paths["morris_sgd_png"] = morris_plot_pretty_log(
    m_sgd,
    "Diagrama de Morris (log-log): influencia global sobre RMSE grupo 'SGD'",
    outdir / "fig_morris_sgd_loglog.png",
    label_top=6
)

paths["sobol_head_png"] = sobol_st_barplot_pretty(
    s_head,
    "Sobol (ST): efecto total sobre RMSE grupo 'heads'",
    outdir / "fig_sobol_ST_head_pretty.png",
    top_n=10
)

paths["sobol_sgd_png"] = sobol_st_barplot_pretty(
    s_sgd,
    "Sobol (ST): efecto total sobre RMSE grupo 'SGD'",
    outdir / "fig_sobol_ST_sgd_pretty.png",
    top_n=10
)

for k, v in paths.items():
    print(f"  {k}: {v}")