import os 
import sys
import math
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pyemu
except ImportError as e:
    raise SystemExit("pyemu package not found. Please install pyemu to run this script.") from e

try:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
except ImportError as e:
    raise SystemExit("SALib package not found. Please install SALib to run this script.") from e



def run_forward(root: Path, quiet: bool = True) -> None:
    cmd = [sys.executable, "forward_run.py"]
    if quiet:
        subprocess.run(cmd, cwd=str(root), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, cwd=str(root), check=True)


def rmse_head_from_pst(obs_csv: Path, sim_dat: Path) -> float:
    obs = pd.read_csv(obs_csv)
    #expected columns : obs_id, head_obs, weight 
    if "obs_id" not in obs.columns or "head_obs" not in obs.columns:
        raise ValueError(f"Observation CSV file {obs_csv} is missing required columns.")
    
    sim = pd.read_csv(sim_dat, sep=r"\s+", header=None, names=["obs_id", "head_sim"])
    obs["obs_id"] = obs["obs_id"].astype(str) 
    sim["obs_id"] = sim["obs_id"].astype(str)

    df = obs.merge(sim, on="obs_id", how="left")
    if df["head_sim"].isna().any():
        missing = df.loc[df["head_sim"].isna(), "obs_id"].head(10).tolist()
        raise RuntimeError(f"MIssing sim heads for obs_ids, e.g.: {missing}")

    # Robnust filter for MF6 dry/invalid nonsense values
    hs = df["head_sim"].astype(float).values
    ho = df["head_obs"].astype(float).values
    good = np.isfinite(hs) & np.isfinite(ho) & (np.abs(hs) < 1e20)

    if good.sum() < max(5, int(0.5 * len(df))):
        raise RuntimeError(f"Too many invalid heads: valid={good.sum()} / {len(df)}")
    
    err = hs[good] - ho[good]
    return float(np.sqrt(np.mean(err**2)))

def rmse_sgd_from_pst(pst_path: Path, sim_sgd_dat: Path) -> float:
    pst = pyemu.Pst(str(pst_path))
    od = pst.observation_data.copy()
    od.index = od.index.astype(str)

    # Only SGD group
    sgd_obs = od.loc[od.obgnme.astype(str).str.lower() == "sgd", ["obsval"]].copy()
    if sgd_obs.empty:
        # No SGD constraints in this case; return NaN (or 0.0)
        return float("nan")
    
    sim = pd.read_csv(sim_sgd_dat, sep=r"\s+", header=None, names=["obsnme", "sim"])
    sim["obsnme"] = sim["obsnme"].astype(str).str.strip()
    sgd_obs = sgd_obs.reset_index().rename(columns={"index": "obsnme"})
    sgd_obs["obsnme"] = sgd_obs["obsnme"].astype(str).str.strip()

    m = sgd_obs.merge(sim, on='obsnme', how='inner')
    if m.empty:
        raise ValueError("No matching SGD observations names between model.pst and sim_sgd.dat")
    err = m["sim"].astype(float) - m["obsval"].astype(float)
    return float(np.sqrt(np.mean(err**2)))


def write_params_csv(params_csv: Path, names: list[str], values: np.ndarray) -> None:
    # forward_run.py accepts either headered or headerless;
    # template expects headerless: parnme, value
    with open(params_csv, "w", encoding="utf-8") as f:
        for n,v in zip(names, values):
            f.write(f"{n},{v:.16g}\n")

    

# def main(N: int = 12, num_levels: int = 4, grid_jump: int = 2, seed: int = 123, quiet: bool = True):
def main(N: int = 12, num_levels: int = 4, seed: int = 123, quiet: bool = True):
    root = Path(".").resolve()
    pst_path = root / "model.pst"
    if not pst_path.exists():
        raise FileNotFoundError(f"Pest control file not found at {pst_path}")

    pst = pyemu.Pst(str(pst_path))
    par_names = list(pst.adj_par_names)

    has_sgd = (pst.observation_data.obgnme.astype(str).str.lower() == "sgd").any()


    if len(par_names) == 0:
        raise RuntimeError("No adjustable parameters found in the PEST control file (model.pst). !!")
    
    # using bounds from pst; build_pest_clean.py sets partrans='log' for all pars
    pdict = pst.parameter_data.loc[par_names, ["parlbnd", "parubnd", "partrans"]].copy()
    lb = pdict["parlbnd"].astype(float).values
    ub = pdict["parubnd"].astype(float).values

    if np.any(lb <= 0.0) or np.any(ub <= 0.0):
        raise ValueError("All parameter bounds must be >0 for log-space sampling.")
    
    #sampling uniformly in log10-space (consistent with parftrans=log philosophy)"
    lb_log = np.log10(lb)
    ub_log = np.log10(ub)

    problem = {
        "num_vars":len(par_names),
        "names": par_names,
        "bounds": list(map(list, zip(lb_log, ub_log))) # SALib wants list of [low,high]
    }

    # Morris sample in log10 space:
    Xlog = morris_sample.sample(
        problem, 
        N=N,
        num_levels=num_levels,
        #grid_jump=grid_jump,
        optimal_trajectories=None,
        local_optimization=True,
        seed=seed
    )

    # Transforming to linear multipliers for the model:
    X = 10.0 ** Xlog

    params_csv = root / "params.csv"
    obs_csv = root / "obs_heads.csv"
    sim_dat = root / "sim_heads.dat"
    sim_sgd = root / "sim_sgd.dat"

    if not obs_csv.exists():
        raise FileNotFoundError(f"Observation CSV file not found at {obs_csv}")

    # Backup existing params.csv (calibrated) so i can restore afterward
    backup = None
    if params_csv.exists():
        backup = root / "params.csv__bak__"
        shutil.copy2(params_csv, backup)
    
    Y_head = np.zeros(X.shape[0], dtype=float)
    Y_sgd = np.zeros(X.shape[0], dtype=float)

    failures = 0

    try:
        for i in range(X.shape[0]):
            write_params_csv(params_csv, par_names, X[i,:])
            try:
                run_forward(root, quiet=quiet)
                y_head = rmse_head_from_pst(obs_csv, sim_dat)

                if has_sgd:
                    y_sgd = rmse_sgd_from_pst(pst_path, sim_sgd)
                else:
                    y_sgd = np.nan # wont be used

            except Exception as e:
                # Penalize failed runs with a large RMSE so analysis can proceed
                failures += 1
                y_head = 1e6
                y_sgd = 1e6 if has_sgd else np.nan
                if not quiet:
                    print(f"[run {i}] FAILED: {e}")

            Y_head[i] = y_head
            Y_sgd[i] = y_sgd

    
        # Analyze in the same space Morris head was sampled (log-space)
        Si_head = morris_analyze.analyze(
            problem,
            Xlog,
            Y_head,
            conf_level=0.95,
            print_to_console=False,
            num_levels = num_levels,
            #grid_jump = grid_jump,
            num_resamples = 100,
            seed=seed
        )

        df_head = pd.DataFrame({
            "parnme": par_names,
            "mu_star": Si_head["mu_star"],
            "mu":Si_head["mu"],
            "sigma":Si_head["sigma"],
            "mu_star_conf":Si_head["mu_star_conf"],
        }).sort_values("mu_star", ascending=False)

        df_head.to_csv("gsa_morris_rmse_head.csv", index=False)
        pd.DataFrame(X, columns=par_names).to_csv("gsa_morris_head_samples_linear.csv", index=False)
        pd.DataFrame({"rmse_head":Y_head}).to_csv("gsa_morris_Y_head.csv", index=False)

        print("wrote: gsa_morris_rmse_head.csv")
        print("Top parameters by mu_star (global influence on RMSE):")
        print(df_head.head(10).to_string(index=False))
        if failures:
            print(f"WARNING: {failures} model failures were penalized (RMSE=1e6). Consider tightening bounds if many failures.")

        # Analyze Morris for SGD (drop NaNs in case has no SGD)
        if has_sgd:
            Si_sgd = morris_analyze.analyze(
                problem,
                Xlog,
                Y_sgd,
                conf_level=0.95,
                print_to_console=False,
                num_levels = num_levels,
                num_resamples = 100,
                seed=seed
            )

            df_sgd = pd.DataFrame({
                "parnme": par_names,
                "mu_star": Si_sgd["mu_star"],
                "mu":Si_sgd["mu"],
                "sigma":Si_sgd["sigma"],
                "mu_star_conf":Si_sgd["mu_star_conf"],
            }).sort_values("mu_star", ascending=False)

            df_sgd.to_csv("gsa_morris_rmse_sgd.csv", index=False)
            pd.DataFrame(X, columns=par_names).to_csv("gsa_morris_sgd_samples_linear.csv", index=False)
            pd.DataFrame({"rmse_sgd":Y_sgd}).to_csv("gsa_morris_Y_sgd.csv", index=False)

            print("wrote: gsa_morris_rmse_sgd.csv")
            print("Top parameters by mu_star (global influence on RMSE):")
            print(df_sgd.head(10).to_string(index=False))
            if failures:
                print(f"WARNING: {failures} model failures were penalized (RMSE=1e6). Consider tightening bounds if many failures.")

        else:
            print("No SGD group in 'model.pst' file, skipping SGD Morris !!")

    finally:
        print("Wrote: gsa_morris_rmse_head.csv and gsa_morris_rmse_sgd.csv. | *_* | ")
        # Restoring original params.csv file
        if backup is not None and backup.exists():
            shutil.move(str(backup), str(params_csv))
    

if __name__ == "__main__":
    # can tune these default values
    main(N=16, num_levels=8, seed=123, quiet=True)
