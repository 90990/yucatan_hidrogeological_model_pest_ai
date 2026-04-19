"""
    SCRIPT THAT CALIBRATE THE PARAMETERS USING THE EMULATOR OF MF6 (AI CALIBRATION): Minimizes misfit between emulator-predicted heads and the observed heads.
    This script produces:
        - theta_emulator_star.csv
        - predicted heads on the emulator and residual arrays
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.optimize import differential_evolution

# Path to the EXPERIMENT WORKSPACE
EXP_WS = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\AI\exp_mixed_200_seed200")
# EXP_WS = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\AI\exp_mixed_200_seed200")

EMULATOR_FILE = EXP_WS / "forward_emulator.joblib"
MANIFEST_FILE = EXP_WS / "manifest.json"

BASE_WS = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\quadtreeGrid\gridgen_disu")
OBS_HEAD_FILE = BASE_WS / "obs_heads.csv"
OBS_VALUE_COL_CANDIDATES = ["head_obs"]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log10_bounds(lo: float, hi: float) -> tuple[float, float]:
    return (math.log10(lo), math.log10(hi))


def build_bounds_from_manifest(manifest: dict):
    pars = manifest["PARS"]
    par_names = [str(p["name"]) for p in pars]
    sampling = [str(p.get("sampling", "linear")).lower() for p in pars]

    bounds_z = []
    for p in pars:
        lo = float(p["low"])
        hi = float(p["high"])
        if str(p.get("sampling", "linear")).lower() == "log":
            bounds_z.append(log10_bounds(lo, hi))
        else:
            bounds_z.append((lo, hi))

    return par_names, sampling, bounds_z


def z_to_theta(z: np.ndarray, sampling: list[str]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    theta = np.zeros(len(z), dtype=float)
    for j, smp in enumerate(sampling):
        theta[j] = 10.0 ** z[j] if smp == "log" else z[j]
    return theta


def load_emulator(path: Path):
    return load(path)


def predict_heads(pack, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(1, -1)

    if isinstance(pack, dict) and "model" in pack:
        model = pack["model"]
        xsc = pack.get("x_scaler", None)
        ysc = pack.get("y_scaler", None)

        X = xsc.transform(theta) if xsc is not None else theta
        Yp = model.predict(X)
        if ysc is not None:
            Yp = ysc.inverse_transform(Yp)
        return np.asarray(Yp).ravel()

    if hasattr(pack, "predict"):
        return np.asarray(pack.predict(theta)).ravel()

    raise TypeError("Unsupported emulator format.")


def load_observed_heads(obs_file: Path, obs_names):
    df = pd.read_csv(obs_file)
    if "obs_id" not in df.columns:
        raise ValueError("obs_heads.csv must contain 'obs_id' column.")

    df["obs_id"] = df["obs_id"].astype(str)

    value_col = None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in OBS_VALUE_COL_CANDIDATES:
        if cand.lower() in lower_map:
            value_col = lower_map[cand.lower()]
            break
    if value_col is None:
        raise ValueError("Could not find observed head column in obs_heads.csv.")

    if "weight" not in df.columns:
        raise ValueError("obs_heads.csv has no 'weight' column.")

    tab = df.set_index("obs_id")[[value_col, "weight"]].astype(float)

    obs_names = [str(x) for x in obs_names]
    missing = [n for n in obs_names if n not in tab.index]
    if missing:
        raise ValueError(f"Missing obs ids in obs_heads.csv (example: {missing[:5]})")

    y_obs = tab.loc[obs_names, value_col].to_numpy(dtype=float)
    w = tab.loc[obs_names, "weight"].to_numpy(dtype=float)
    return y_obs, w


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def wrmse_meters(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)

    res = b - a
    num = np.sum((w * res) ** 2)
    den = np.sum(w ** 2) if np.sum(w ** 2) > 0 else len(w)
    return float(np.sqrt(num / den))


def main():
    manifest = load_manifest(MANIFEST_FILE)
    manifest_par_names, sampling, bounds_z = build_bounds_from_manifest(manifest)

    pack = load_emulator(EMULATOR_FILE)

    if isinstance(pack, dict) and "par_names" in pack:
        emu_par_names = [str(x) for x in pack["par_names"]]
        if emu_par_names != manifest_par_names:
            raise ValueError(
                f"Mismatch between manifest PAR_NAMES and emulator PAR_NAMES.\n"
                f"manifest={manifest_par_names}\n"
                f"emulator={emu_par_names}"
            )
        par_names = emu_par_names
    else:
        par_names = manifest_par_names

    if isinstance(pack, dict) and "obs_names" in pack:
        obs_names = [str(x) for x in pack["obs_names"]]
    else:
        obs_names = pd.read_csv(OBS_HEAD_FILE)["obs_id"].astype(str).to_numpy().tolist()

    y_obs, w = load_observed_heads(OBS_HEAD_FILE, obs_names)

    USE_PRIOR = False
    LAMBDA_PRIOR = 0.0

    def prior_penalty(theta: np.ndarray) -> float:
        pen = 0.0
        for val, smp in zip(theta, sampling):
            if smp == "log":
                pen += (np.log10(val)) ** 2
            else:
                pen += (val - 1.0) ** 2
        return float(pen)

    def objective(z):
        theta = z_to_theta(z, sampling)
        y_pred = predict_heads(pack, theta)
        res = y_pred - y_obs
        phi_data = float(np.sum((w * res) ** 2))
        if not USE_PRIOR:
            return phi_data
        return phi_data + LAMBDA_PRIOR * prior_penalty(theta)

    print("Starting emulator-based calibration (differential evolution)...")
    result = differential_evolution(
        objective,
        bounds=bounds_z,
        maxiter=400,
        popsize=25,
        tol=1e-7,
        polish=True,
        seed=7,
        updating="deferred",
        workers=1,
    )

    theta_star = z_to_theta(result.x, sampling)
    y_star = predict_heads(pack, theta_star)

    phi_star = float(np.sum((w * (y_star - y_obs)) ** 2))
    rmse_star = rmse(y_obs, y_star)
    wrmse_star = wrmse_meters(y_obs, y_star, w)

    print("\n=== Emulator-calibrated parameters (theta*) ===")
    for n, v in zip(par_names, theta_star):
        print(f"{n:15s} = {v:.10g}")

    print("\n=== Emulator fit to observations (on emulator) ===")
    print(f"RMSE    = {rmse_star:.6f} m")
    print(f"WRMSE_m = {wrmse_star:.6f} m")
    print(f"PHI     = {phi_star:.6f}")

    out_csv = EXP_WS / "theta_emulator_star.csv"
    out_pred = EXP_WS / "heads_emulator_star.csv"

    pd.DataFrame({"parnme": par_names, "parval1": theta_star}).to_csv(out_csv, index=False)
    pd.DataFrame(
        {
            "obs_id": obs_names,
            "head_obs": y_obs,
            "head_pred_emulator": y_star,
            "residual": y_star - y_obs,
        }
    ).to_csv(out_pred, index=False)

    print(f"\nSaved calibrated parameters: {out_csv}")
    print(f"Saved emulator predictions:  {out_pred}")


if __name__ == "__main__":
    main()