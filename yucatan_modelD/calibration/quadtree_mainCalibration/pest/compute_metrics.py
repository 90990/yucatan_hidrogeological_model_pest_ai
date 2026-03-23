#!/usr/bin/env python
"""
compute_metrics_robust.py

Robust comparison of observed vs simulated heads for the current folder.

Expected files (in the same folder as this script, unless you pass --workdir):
  - obs_heads.csv   with columns: obs_id, head_obs (optional: weight)
  - sim_heads.dat   two columns per line: obs_id  head_sim

It is tolerant to:
  - whitespace OR comma separation in sim_heads.dat
  - extra debug lines in sim_heads.dat (it will skip non-numeric lines)
  - case / whitespace / quote differences in obs_id

Usage:
  python compute_metrics_robust.py
  python compute_metrics_robust.py --workdir "C:\path\to\pest"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

def norm_id(s: str) -> str:
    s = str(s)
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", "", s)  # remove internal whitespace
    return s.lower()

def read_obs(obs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(obs_path)
    # try to infer column names if user has different headers
    cols_lower = {c.lower(): c for c in df.columns}
    if "obs_id" in cols_lower:
        idcol = cols_lower["obs_id"]
    else:
        idcol = df.columns[0]
    if "head_obs" in cols_lower:
        hcol = cols_lower["head_obs"]
    elif "head" in cols_lower:
        hcol = cols_lower["head"]
    else:
        # assume second column
        hcol = df.columns[1]
    if "weight" in cols_lower:
        wcol = cols_lower["weight"]
    else:
        wcol = None

    out = pd.DataFrame({
        "obs_id": df[idcol].astype(str).map(norm_id),
        "head_obs": pd.to_numeric(df[hcol], errors="coerce"),
    })
    if wcol is not None:
        out["weight"] = pd.to_numeric(df[wcol], errors="coerce").fillna(1.0)
    else:
        out["weight"] = 1.0

    out = out.dropna(subset=["obs_id", "head_obs"])
    return out

def read_sim(sim_path: Path) -> pd.DataFrame:
    rows = []
    # line-by-line parse to survive any junk/debug lines
    for ln in sim_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # accept comma or whitespace separation
        if "," in ln:
            parts = [p.strip() for p in ln.split(",") if p.strip() != ""]
        else:
            parts = ln.split()
        if len(parts) < 2:
            continue
        oid = norm_id(parts[0])
        val = parts[1]
        # sometimes there are trailing tokens; only keep if second token is numeric
        if not _NUM_RE.match(val):
            continue
        rows.append((oid, float(val)))

    df = pd.DataFrame(rows, columns=["obs_id", "head_sim"])
    # keep last duplicate if any
    df = df.drop_duplicates(subset=["obs_id"], keep="last")
    return df

def wrmse(a:np.ndarray, b:np.ndarray, w:np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    res = b - a
    num = np.sum((w*res)**2)
    den = np.sum(w**2) if np.sum(w**2) > 0 else len(w)

    return float(np.sqrt(num / den))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, default=None, help="Folder containing obs_heads.csv and sim_heads.dat")
    args = ap.parse_args()

    wd = Path(args.workdir).resolve() if args.workdir else Path(__file__).resolve().parent
    obs_path = wd / "obs_heads.csv"
    sim_path = wd / "sim_heads.dat"

    if not obs_path.exists():
        raise FileNotFoundError(f"Missing {obs_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"Missing {sim_path}")

    obs = read_obs(obs_path)
    sim = read_sim(sim_path)

    # diagnostics
    inter = set(obs["obs_id"]) & set(sim["obs_id"])
    if len(inter) == 0:
        print("ERROR: 0 matching obs_id between obs_heads.csv and sim_heads.dat")
        print(f"obs rows: {len(obs)}  sim rows: {len(sim)}")
        print("sample obs ids:", obs["obs_id"].head(10).tolist())
        print("sample sim ids:", sim["obs_id"].head(10).tolist())
        # show a few differences
        obs_only = list(set(obs["obs_id"]) - set(sim["obs_id"]))[:10]
        sim_only = list(set(sim["obs_id"]) - set(obs["obs_id"]))[:10]
        print("in obs not in sim (sample):", obs_only)
        print("in sim not in obs (sample):", sim_only)
        return

    df = obs.merge(sim, on="obs_id", how="inner")
    df = df.dropna(subset=["head_obs", "head_sim"])

    n = len(df)
    if n == 0:
        print("ERROR: after merge, all head values are NaN. Check sim_heads.dat values.")
        return

    err = df["head_sim"].to_numpy() - df["head_obs"].to_numpy()
    w = df["weight"].to_numpy()

    rmse = float(np.sqrt(np.mean(err**2)))
    w_rmse = wrmse(df["head_obs"].to_numpy(), df["head_sim"].to_numpy(), w)
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    max_abs = float(np.max(np.abs(err)))
    med_abs = float(np.median(np.abs(err)))

    phi = float(np.sum((w * err) ** 2))
    wrmse = float(np.sqrt(phi / n))

    print(f"N: {n}")
    print(f"RMSE: {rmse:.3f} m")
    print(f"WRMSE: {w_rmse:.3f} m")
    print(f"MAE : {mae:.3f} m")
    print(f"Bias: {bias:.3f} m")
    print(f"Max |err|: {max_abs:.3f} m")
    print(f"Median |err|: {med_abs:.3f} m")
    print(f"PHI (sum (w*res)^2): {phi:.3f}")
    print(f"Weighted RMSE sqrt(phi/N): {wrmse:.6f}")

if __name__ == "__main__":
    main()


