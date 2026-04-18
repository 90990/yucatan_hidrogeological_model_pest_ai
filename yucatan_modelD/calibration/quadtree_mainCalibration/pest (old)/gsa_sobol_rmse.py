import sys 
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pyemu
except ImportError as e:
    raise SystemExit("pyemu is fucking required!") from e

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
except ImportError as e:
    raise SystemExit("SALib is fucking required as well !!") from e


def run_forward(root: Path, i: int) -> None:
    p = subprocess.run(
        [sys.executable, "forward_run.py"],
        cwd=str(root),
        capture_output=True,
        text=True
    )
    if p.returncode != 0:
        logdir = root / "_sobol_fail_logs"
        logdir.mkdir(exist_ok=True)
        (logdir / f"fail_{i:04d}.txt").write_text(
            "STDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr,
            encoding="utf-8"
        )
        raise RuntimeError(f"forward_run.py failed for sample {i} (see _sobol_fail_logs/fail_{i:04d}.txt)")
    
def write_params_csv(params_csv: Path, names: list[str], values: np.ndarray) -> None:
    #headerless: parnme, value 
    with open(params_csv, "w", encoding="utf-8") as f:
        for n,v in zip(names, values):
            f.write(f"{n},{v:.16g}\n")

def rmse_head(obs_csv: Path, sim_dat: Path) -> float:
    obs = pd.read_csv(obs_csv)
    if "obs_id" not in obs.columns or "head_obs" not in obs.columns:
        raise ValueError(f"{obs_csv} must contain columns: obs_id, head_obs")

    sim = pd.read_csv(sim_dat, sep=r"\s+", header=None, names=["obs_id", "head_sim"])
    obs["obs_id"] = obs["obs_id"].astype(str)
    sim["obs_id"] = sim["obs_id"].astype(str)

    df = obs.merge(sim, on="obs_id", how='left')
    if df["head_sim"].isna().any():
        missing = df.loc[df["head_sim"].isna(), "obs_id"].head(10).tolist()
        raise RuntimeError(f"Missing sim heads for obs_id, e.g.: {missing}")

    hs = df["head_sim"].astype(float).values
    ho = df["head_obs"].astype(float).values
    good = np.isfinite(hs) & np.isfinite(ho) & (np.abs(hs) < 1e20)

    if good.sum() < max(5, int(0.5 * len(df))):
        raise RuntimeError(f"Too many invalid heads: valid={good.sum()} / {len(df)}")

    err = hs[good] - ho[good]
    return float(np.sqrt(np.mean(err ** 2)))

def rmse_sgd_from_pst(pst_path, Path, sim_sgd_dat: Path) -> float:
    pst = pyemu.Pst(str(pst_path))
    od = pst.observation_data.copy()
    od.index = od.index.astype(str)

    #SGD observations are those with obgnme == "sgd"
    sgd_obs = od.loc[od.obgnme.astype(str).str.lower() == "sgd", ["obsval"]].copy()
    if sgd_obs.empty:
        return float("nan")

    sim = pd.read_csv(sim_sgd_dat, sep=r"\s+", header=None, names=["obsnme", "sim"])
    sim["obsnme"] = sim["obsnme"].astype(str).str.strip()

    m = sgd_obs.merge(sim, on="obsnme", how="inner")
    if m.empty:
        raise ValueError("No matching SGD observations names between 'model.pst' and 'sim_sgd.dat'.")
    
    err = m["sim"].astype(float) - m["obsval"].astype(float)
    return float(np.sqrt(np.mean(err ** 2)))

 
def get_top_union_from_morris(k_head: int=5, k_sgd: int=5) -> list[str]:
    # Building the TOP_union automatically from the Morris outputs
    top = set()

    head_csv = Path("gsa_morris_rmse_head.csv")
    if head_csv.exists():
        df = pd.read_csv(head_csv)
        if "parnme" in df.columns:
            top |= set(df["parnme"].astype(str).head(k_sgd).tolist())

    return sorted(top)


def main(
    top_union: list[str] | None = None,
    N: int = 128,
    seed: int = 123,
    quiet: bool = True,
    calc_second_order: bool = False
    ) -> None:

    root = Path(".").resolve()
    pst_path = root / "model.pst"
    if not pst_path.exists():
        raise FileNotFoundError(f"Missing {pst_path}")

    pst = pyemu.Pst(str(pst_path))
    adj = set(pst.adj_par_names)

    if top_union is None or len(top_union) == 0:
        top_union = get_top_union_from_morris(k_head=5, k_sgd=5)

    names = [p for p in top_union if p in adj]
    if len(names) == 0:
        raise RuntimeError("TOP_union is empty or none of those parameters are adjustable in model.pst")

    # bounds from pst
    pdat = pst.parameter_data.loc[names, ["parlbnd", "parubnd"]].astype(float)
    lb = pdat["parlbnd"].values
    ub = pdat["parubnd"].values
    if np.any(lb <= 0.0) or np.any(ub <= 0.0):
        raise ValueError("All bounds must be >0 to sample in log-space.")

    lb_log = np.log10(lb)
    ub_log = np.log10(ub)

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": list(map(list, zip(lb_log, ub_log)))  # correct key: "bounds"
    }

    np.random.seed(seed)

    # Saltelli sample in log space
    Xlog = sobol_sample.sample(problem, N, calc_second_order=calc_second_order, seed=seed)
    X = 10.0 ** Xlog # back to linear multipliers
    X = np.clip(X, lb, ub) #safety clamp

    params_csv = root / "params.csv"
    obs_csv = root / "obs_heads.csv"
    sim_heads = root / "sim_heads.dat"
    sim_sgd = root / "sim_sgd.dat"

    # backup calibrated params.csv so we restore after runs
    backup = None
    if params_csv.exists():
        backup = root / "params.csv.__bak__"
        shutil.copy2(params_csv, backup)

    Y_head = np.zeros(X.shape[0], dtype=float)
    Y_sgd = np.zeros(X.shape[0], dtype=float)
    failures = 0

    try:
        for i in range(X.shape[0]):
            print(f"Iteration: {i} of {X.shape[0]}")
            write_params_csv(params_csv, names, X[i, :])
            try:
                run_forward(root, i)
                Y_head[i] = rmse_head(obs_csv, sim_heads)

                # If SGD exists and pst has sgd group, compute; else NaN
                if sim_sgd.exists():
                    Y_sgd[i] = rmse_sgd_from_pst(pst_path, sim_sgd)
                else:
                    Y_sgd[i] = float("nan")

            except Exception as e:
                failures += 1
                Y_head[i] = 1e6
                Y_sgd[i] = 1e6
                if not quiet:
                    print(f"[sample {i}] FAILED: {e}")

        # Sobol indices
        Si_head = sobol.analyze(problem, Y_head, calc_second_order=calc_second_order, print_to_console=False)

        out_head = pd.DataFrame({
            "parnme": names,
            "S1": Si_head["S1"],
            "S1_conf": Si_head["S1_conf"],
            "ST": Si_head["ST"],
            "ST_conf": Si_head["ST_conf"],
        }).sort_values("ST", ascending=False)

        out_head.to_csv("gsa_sobol_head.csv", index=False)

        # SGD Sobol (only if valid values exist)
        if np.isfinite(Y_sgd).any():
            Si_sgd = sobol.analyze(problem, Y_sgd, calc_second_order=calc_second_order, print_to_console=False)
            out_sgd = pd.DataFrame({
                "parnme": names,
                "S1": Si_sgd["S1"],
                "S1_conf": Si_sgd["S1_conf"],
                "ST": Si_sgd["ST"],
                "ST_conf": Si_sgd["ST_conf"],
            }).sort_values("ST", ascending=False)
            out_sgd.to_csv("gsa_sobol_sgd.csv", index=False)
        else:
            out_sgd = None

        pd.DataFrame(X, columns=names).to_csv("gsa_sobol_samples_linear.csv", index=False)
        pd.DataFrame({"rmse_head": Y_head, "rmse_sgd": Y_sgd}).to_csv("gsa_sobol_Y.csv", index=False)

        print("Wrote: gsa_sobol_head.csv, gsa_sobol_sgd.csv (if applicable), gsa_sobol_samples_linear.csv, gsa_sobol_Y.csv")
        print("\nSobol (Heads RMSE) sorted by ST:")
        print(out_head.to_string(index=False))

        if out_sgd is not None:
            print("\nSobol (SGD RMSE) sorted by ST:")
            print(out_sgd.to_string(index=False))

        if failures:
            print(f"\nWARNING: {failures} model failures were penalized (RMSE=1e6). If this is large, tighten bounds or add failure diagnostics.")

    finally:
        if backup is not None and backup.exists():
            shutil.move(str(backup), str(params_csv))


if __name__ == "__main__":

    TOP_UNION = []  # e.g. ["mk_rest_l2","mr_rest","mr_upland","mfgw","mk_rest_l1"]

    main(top_union=TOP_UNION, N=128, seed=123, quiet=True, calc_second_order=False)

    