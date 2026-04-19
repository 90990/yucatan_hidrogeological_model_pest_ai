"""
AI parameter sampler + runner for MODFLOW 6 (DISU) surrogate dataset generation.

What it does:
  - Creates run_00000, run_00001, ... directories
  - Copies BASE_WS into each run directory
  - Writes params.csv (absolute parameter values)
  - Executes forward_run_ai_abs.py
  - Reads sim_heads.dat (expects N_OBS heads)
  - Saves dataset as .npz (X, Y, par_names, status) + CSV summaries

Key improvements vs your current version:
  - Recharge bounds include 0.25 for BOTH zones (critical)
  - Mixed design sampling: includes theta0 exactly + local cloud + global LHS
  - Safer output folder naming to avoid overwriting previous experiments
"""

from __future__ import annotations
import json
import sys
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# USER SETTINGS
# -----------------------------
BASE_WS = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\quadtreeGrid\gridgen_disu")
OUT_ROOT = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\AI")

N_RUNS = 200
SEED = 200
N_OBS = 135

RUNNER_SCRIPT = "forward_run_ai_mult.py"  # must exist inside BASE_WS
# RUNNER_SCRIPT = "forward_run_ai_mult_with_SGD.py"  # must exist inside BASE_WS
PYTHON_EXE = sys.executable              # current python

INCLUDE_SGD_IN_DATASET = False

# -----------------------------
# PARAMETER DEFINITIONS
# -----------------------------
@dataclass(frozen=True)
class ParDef:
    name: str
    low: float
    high: float
    sampling: str = "log"  # "log" or "linear"


# Suggested bounds (moderately widened K, fixed RCH bounds to include 0.25, slightly widened EVT)

PARS: list[ParDef] = [
    # K 
    ParDef("mk_upland_l1", 1.0,        3.0,        "log"),      # (1e-3..3e-3)/1e-3
    ParDef("mk_ring_l1",   1/9,        1.0,        "log"),      # (1e-2..9e-2)/9e-2
    ParDef("mk_rest_l1",   4/9,        1.0,        "log"),      # (4e-3..9e-3)/9e-3

    ParDef("mk_upland_l2", 1.0,        3.0,        "log"),      # (1e-2..3e-2)/1e-2
    ParDef("mk_ring_l2",   1/9,        1.0,        "log"),      # (1e-1..9e-1)/9e-1
    ParDef("mk_rest_l2",   4/9,        1.0,        "log"),      # (4e-2..9e-2)/9e-2

    # Recharge multiplier about c_inf=0.25 base (linear is fine)
    # The intended absolute range is 0.05..0.50, then:

    ParDef("mr_upland",    0.05/0.25,  0.50/0.25,  "linear"),   # Multiplier go from 0.2..2
    ParDef("mr_rest",      0.05/0.25,  0.50/0.25,  "linear"),   # Multiplier go from 0.2..2

    # EVT multiplier about fgw=0.30 base value
    # So the intendedn absolute range is 0.05...
    ParDef("mFGW_upland",         0.05/0.30,  0.50/0.30,  "linear"),   # Multiplier go from 0.1667..1.6667
    ParDef("mFGW_rest",         0.05/0.30,  0.50/0.30,  "linear"),   # Multiplier go from 0.1667..1.6667
]

PAR_NAMES = [p.name for p in PARS]


# Known-good “anchor” vector theta0 (expressed for c_inf=1.0 base and fgw=1.0 base)
# proven manual solution:
#   K: (1e-3, 9e-2, 9e-3, 1e-2, 9e-1, 9e-2)
#   RCH: 0.25, 0.25  (since c_inf was 0.25 with multipliers 1.0)
#   EVT: 0.30
THETA0 = np.array([
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0,
], dtype=float)


# Mixed design settings
FRAC_LOCAL = 0.80            # 80% local around baseline
SIGMA_LOG10 = 0.10           # std dev in log10-space for K local sampling, ~25% multiplicative std (10^0.10 ~ 1.26)
SIGMA_LIN_FRAC = 0.15        # std dev as fraction of (hi-lo) for linear params, 15% of (hi-lo) fopr linear multipliers
ENFORCE_RCH_REST_GE_UPLAND = False  # keep False for now (do not bias away from equality)

# -----------------------------
# SAMPLING helpers
# -----------------------------
def local_cloud(theta0: np.ndarray, pars: list[ParDef], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(pars)), dtype=float)

    for j, p in enumerate(pars):
        lo, hi = float(p.low), float(p.high)

        if p.sampling.lower() == "log":
            mu = np.log10(theta0[j])
            z = rng.normal(mu, SIGMA_LOG10_K, size=n)
            x = 10.0 ** z
        else:
            sd = SIGMA_LIN * (hi - lo)
            x = rng.normal(theta0[j], sd, size=n)

        X[:, j] = np.clip(x, lo, hi)

    return X


# -----------------------------
# SAMPLING HELPERS
# -----------------------------
def lhs_unit(n: int, d: int, seed: int) -> np.ndarray:
    """Latin hypercube in [0,1]. Falls back to uniform if SciPy unavailable."""
    rng = np.random.default_rng(seed)
    if HAS_SCIPY:
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        return sampler.random(n)
    return rng.random((n, d))


def scale_samples(u01: np.ndarray, pars: list[ParDef]) -> np.ndarray:
    """Convert [0,1] samples to absolute values with per-parameter scaling."""
    X = np.zeros_like(u01, dtype=float)
    for j, p in enumerate(pars):
        if p.sampling.lower() == "linear":
            X[:, j] = p.low + u01[:, j] * (p.high - p.low)
        elif p.sampling.lower() == "log":
            if p.low <= 0 or p.high <= 0:
                raise ValueError(f"log sampling requires positive bounds for {p.name}")
            lo = np.log10(p.low)
            hi = np.log10(p.high)
            X[:, j] = 10.0 ** (lo + u01[:, j] * (hi - lo))
        else:
            raise ValueError(f"Unknown sampling mode: {p.sampling}")
    return X


def local_cloud(theta0: np.ndarray, pars: list[ParDef], n: int, seed: int) -> np.ndarray:
    """Local perturbations around theta0 with truncation to bounds."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(pars)), dtype=float)

    for j, p in enumerate(pars):
        lo, hi = p.low, p.high
        if p.sampling.lower() == "log":
            mu = np.log10(theta0[j])
            z = rng.normal(mu, SIGMA_LOG10, size=n)
            x = 10.0 ** z
        else:
            sd = SIGMA_LIN_FRAC * (hi - lo)
            x = rng.normal(theta0[j], sd, size=n)

        X[:, j] = np.clip(x, lo, hi)

    return X


def theta0_out_of_bounds(theta0: np.ndarray, pars: list[ParDef]) -> bool:
    for j, p in enumerate(pars):
        if not (p.low <= float(theta0[j]) <= p.high):
            return True
    return False

def make_mixed_design(n_runs: int, pars: list[ParDef], theta0: np.ndarray, seed: int) -> np.ndarray:
    npars = len(pars)
    n_remaining = n_runs - 1
    n_local = int(round(FRAC_LOCAL * n_remaining))
    n_global = n_remaining - n_local

    U = lhs_unit(n_global, npars, seed=seed)
    Xg = scale_samples(U, pars)
    Xl = local_cloud(theta0, pars, n_local, seed=seed + 1001)

    X = np.vstack([theta0[None, :], Xg, Xl])

    for j, p in enumerate(pars):
        X[:, j] = np.clip(X[:, j], p.low, p.high)

    return X

def _read_sim_sgd(run_ws: Path, sgd_ids: np.ndarray) -> np.ndarray:
    p = run_ws / "sim_sgd.dat"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (SGD enabled but runner did not write it).")
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["obs_id", "sim"])
    m = dict(zip(df["obs_id"].astype(str), df["sim"].astype(float)))
    y = np.array([m.get(oid, np.nan) for oid in sgd_ids], dtype=float)
    if np.any(np.isnan(y)):
        missing = sgd_ids[np.isnan(y)][:10]
        raise RuntimeError(f"Missing simulated SGD for obs ids (first 10): {missing}")
    return y

# -----------------------------
# RUN MANAGEMENT
# -----------------------------
def copy_base_to_run(run_ws: Path) -> None:
    if run_ws.exists():
        shutil.rmtree(run_ws)
    shutil.copytree(BASE_WS, run_ws)


def write_params_csv(run_ws: Path, pars: list[ParDef], xrow: np.ndarray) -> None:
    df = pd.DataFrame({"parnme": [p.name for p in pars], "parval1": xrow.astype(float)})
    df.to_csv(run_ws / "params.csv", index=False)


def run_forward(run_ws: Path) -> None:
    script_path = run_ws / RUNNER_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(f"Runner script not found in run folder: {script_path}")

    p = subprocess.run([PYTHON_EXE, RUNNER_SCRIPT], cwd=str(run_ws), capture_output=True, text=True)
    if p.returncode != 0:
        print("STDOUT:\n", p.stdout)
        print("STDERR:\n", p.stderr)
        raise RuntimeError(f"Forward run failed in {run_ws.name}")


def read_sim_heads(run_ws: Path) -> np.ndarray:
    f = run_ws / "sim_heads.dat"
    if not f.exists():
        raise FileNotFoundError(f"Missing sim_heads.dat in {run_ws}")

    df = pd.read_csv(f, sep=r"\s+", header=None, names=["obs_id", "head_sim"])
    y = df["head_sim"].to_numpy(dtype=float)

    if y.size != N_OBS:
        raise ValueError(f"Expected {N_OBS} heads but got {y.size} in {run_ws}")
    return y

# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    
    if theta0_out_of_bounds(THETA0, PARS):
        raise ValueError("THETA0 is outside the declared bounds.")

    exp_name = f"exp_mixed_{N_RUNS}_seed{SEED}"
    OUT_WS = OUT_ROOT / exp_name
    OUT_WS.mkdir(parents=True, exist_ok=True)

    manifest = {
        "BASE_WS": str(BASE_WS),
        "OUT_WS": str(OUT_WS),
        "N_RUNS": N_RUNS,
        "SEED": SEED,
        "RUNNER_SCRIPT": RUNNER_SCRIPT,
        "N_OBS": N_OBS,
        "INCLUDE_SGD_IN_DATASET": INCLUDE_SGD_IN_DATASET,
        "PARS": [asdict(p) for p in PARS],
        "THETA0": THETA0.tolist(),
        "FRAC_LOCAL": FRAC_LOCAL,
        "SIGMA_LOG10": SIGMA_LOG10,
        "SIGMA_LIN_FRAC": SIGMA_LIN_FRAC,
    }
    (OUT_WS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sgd_ids = _maybe_load_sgd_ids(BASE_WS) if INCLUDE_SGD_IN_DATASET else None

    X = make_mixed_design(N_RUNS, PARS, THETA0, seed=SEED)

    n_sgd = 0 if sgd_ids is None else int(sgd_ids.size)
    n_out = N_OBS + n_sgd

    Y = np.full((N_RUNS, n_out), np.nan, dtype=float)
    status = np.zeros(N_RUNS, dtype=int)

    for i in range(N_RUNS):
        run_ws = OUT_WS / f"run_{i:05d}"
        try:
            copy_base_to_run(run_ws)
            write_params_csv(run_ws, PARS, X[i, :])
            run_forward(run_ws)

            y_heads = read_sim_heads(run_ws)

            if sgd_ids is not None:
                y_sgd = _read_sim_sgd(run_ws, sgd_ids)
                y_all = np.concatenate([y_heads, y_sgd])
            else:
                y_all = y_heads

            Y[i, :] = y_all
            status[i] = 1

        except Exception as e:
            status[i] = 0
            print(f"[FAIL] {run_ws.name}: {e}")

        if (i + 1) % 25 == 0:
            ok = int(status[: i + 1].sum())
            print(f"Completed {i+1}/{N_RUNS} (ok={ok}, fail={(i+1-ok)})")

    np.savez(
        OUT_WS / "ml_dataset.npz",
        X=X,
        Y=Y,
        par_names=np.array([p.name for p in PARS], dtype=object),
        status=status,
    )

    pd.DataFrame(X, columns=[p.name for p in PARS]).to_csv(OUT_WS / "X_params.csv", index=False)
    pd.DataFrame({"status": status}).to_csv(OUT_WS / "run_status.csv", index=False)

    print("Saved:", OUT_WS / "ml_dataset.npz")


if __name__ == "__main__":
    main()