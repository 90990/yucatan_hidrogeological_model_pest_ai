"""
Step C — Verify emulator-calibrated parameters with a true MODFLOW 6 run.

- Copies BASE_WS -> VERIFY_WS
- Writes params.csv from a theta CSV (parnme, parval1)
- Runs forward runner script
- Reads sim_heads.dat and compares to obs_heads.csv (head_obs + optional weight)
- Prints RMSE / WRMSE and saves residuals


"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


OUT_WS = Path(r"C:\\Users\\sebas\\Documents\\AYUDANTE SNI\\TESIS\\AI\\exp_mixed_1000_seed10")
BASE_WS = Path(r"C:\\Users\\sebas\\Documents\\AYUDANTE SNI\\TESIS\\yucatan_modelD\\quadtreeGrid\\gridgen_disu")
# ---------------------------------

VERIFY_WS = OUT_WS / "verify_mf6_theta_star"

RUNNER_SCRIPT_CANDIDATES = [
    "forward_run_ai_mult.py",
    "forward_run_ai_abs.py",
]

OBS_HEADS_FILE = BASE_WS / "obs_heads.csv"
OBS_VALUE_COL_CANDIDATES = ["head_obs"]
OBS_WEIGHT_COL_CANDIDATES = ["weight", "weights", "w"]  # allow flexibility


def find_runner_script(ws: Path, preferred: str | None = None) -> str:
    if preferred is not None:
        if (ws / preferred).exists():
            return preferred
        raise FileNotFoundError(f"Runner '{preferred}' not found in {ws}")

    for name in RUNNER_SCRIPT_CANDIDATES:
        if (ws / name).exists():
            return name

    raise FileNotFoundError(f"No runner script found in {ws}. Tried: {RUNNER_SCRIPT_CANDIDATES}")


def run_python(script_name: str, ws: Path) -> None:
    p = subprocess.run(["python", script_name], cwd=str(ws))
    if p.returncode != 0:
        raise RuntimeError("Forward run failed during verification.")


def read_sim_heads_dat(ws: Path) -> pd.DataFrame:
    f = ws / "sim_heads.dat"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    df = pd.read_csv(f, sep=r"\s+", header=None, names=["obs_id", "head_sim"])
    df["obs_id"] = df["obs_id"].astype(str)
    df["head_sim"] = df["head_sim"].astype(float)
    return df


def _pick_col_case_insensitive(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_observed(obs_file: Path, obs_ids: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    df = pd.read_csv(obs_file)
    if "obs_id" not in df.columns:
        raise ValueError(f"{obs_file} must contain an 'obs_id' column.")
    df["obs_id"] = df["obs_id"].astype(str)

    vcol = _pick_col_case_insensitive(df, OBS_VALUE_COL_CANDIDATES)
    if vcol is None:
        raise ValueError(f"{obs_file} missing observed head column. Tried: {OBS_VALUE_COL_CANDIDATES}")

    wcol = _pick_col_case_insensitive(df, OBS_WEIGHT_COL_CANDIDATES)

    df = df.set_index("obs_id")

    missing = [oid for oid in obs_ids if oid not in df.index]
    if missing:
        raise ValueError(f"obs_heads.csv missing {len(missing)} obs ids (example: {missing[:5]})")

    y_obs = df.loc[obs_ids, vcol].astype(float).to_numpy()
    w = None
    if wcol is not None:
        w = df.loc[obs_ids, wcol].astype(float).to_numpy()

    return y_obs, w


def rmse(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_sim - y_obs) ** 2)))


def phi_and_wrmse(y_obs: np.ndarray, y_sim: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    res = y_sim - y_obs
    wres = w * res
    phi = float(np.sum(wres ** 2))
    wrmse = float(np.sqrt(phi / len(res)))
    return phi, wrmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="theta_emulator_star.csv", help="Theta CSV in OUT_WS (parnme, parval1)")
    ap.add_argument("--runner", default=None, help="Runner script name (optional)")
    args = ap.parse_args()

    theta_arg = Path(args.theta)
    theta_csv = theta_arg if theta_arg.is_absolute() else (Path.cwd() / theta_arg)
    if not theta_csv.exists():
        raise FileNotFoundError(f"Missing {theta_csv}")

    # Building clean verify workspace
    if VERIFY_WS.exists():
        shutil.rmtree(VERIFY_WS)
    shutil.copytree(BASE_WS, VERIFY_WS)

    # Write params.csv for runner
    df_theta = pd.read_csv(theta_csv)
    if not {"parnme", "parval1"}.issubset(df_theta.columns):
        raise ValueError("Theta CSV must contain columns: parnme, parval1")
    df_theta.to_csv(VERIFY_WS / "params.csv", index=False)

    runner = find_runner_script(VERIFY_WS, preferred=args.runner)
    print(f"Running MF6 verification using: {runner}")
    run_python(runner, VERIFY_WS)

    simdf = read_sim_heads_dat(VERIFY_WS)
    obs_ids = simdf["obs_id"].tolist()

    y_sim_all = simdf["head_sim"].to_numpy(dtype=float)
    y_obs_all, w_all = load_observed(OBS_HEADS_FILE, obs_ids)

    valid = np.isfinite(y_sim_all) & (y_sim_all > -1e20) & np.isfinite(y_obs_all)
    n_bad = int((~valid).sum())
    if n_bad > 0:
        print(f"Warning: dropping {n_bad}/{len(valid)} invalid simulated heads from metrics.")

    y_sim = y_sim_all[valid]
    y_obs = y_obs_all[valid]
    w = w_all[valid] if w_all is not None else None

    print("\n=== MF6 verification (true forward run) ===")
    print(f"N used = {len(y_obs)} / {len(obs_ids)}")
    print(f"RMSE   = {rmse(y_obs, y_sim):.6f} m")

    if w is not None:
        phi, wr = phi_and_wrmse(y_obs, y_sim, w)
        print(f"PHI    = {phi:.6f}")
        print(f"WRMSE (PHI/N) = {wr:.6f}")
    else:
        print("WRMSE  = (no weight column found in obs_heads.csv)")

    wrmse_m = float(np.sqrt(np.sum((w*(y_sim-y_obs))**2) / np.sum(w**2)))
    print(f"WRMSE_m = {wrmse_m:.6f} m")

    out = pd.DataFrame(
        {
            "obs_id": np.array(obs_ids, dtype=object),
            "head_obs": y_obs_all,
            "head_sim": y_sim_all,
            "residual": (y_sim_all - y_obs_all),
        }
    )
    out.to_csv(OUT_WS / "verify_mf6_residuals.csv", index=False)
    print(f"Saved residual table: {OUT_WS / 'verify_mf6_residuals.csv'}")


if __name__ == "__main__":
    main()
