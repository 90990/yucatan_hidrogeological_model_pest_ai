"""
Forward run script for PEST/PEST++ (runs MANY times).

This script must stay light:
  - No absolute paths
  - No heavy GIS
  - Only: read params -> apply multipliers -> run MF6 -> write sim_heads.dat
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import flopy
from flopy.utils import HeadFile

#### ---- OLD 'read_params' FUNCTION 
# def read_params(params_csv: Path) -> dict[str, float]:
#     df = pd.read_csv(params_csv)
#     if not {"parnme", "parval1"}.issubset(df.columns):
#         raise ValueError(f"{params_csv} must have columns parnme, parval1")
#     return {str(r.parnme): float(r.parval1) for r in df.itertuples(index=False)}

def read_params(params_csv):
    # trying normal headered CSV first
    try:
        df = pd.read_csv(params_csv, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(params_csv, header=None)
    
    cols = [c.lower() if isinstance(c, str) else c for c in df.columns]

    if "parnme" in cols and "parval1" in cols:
        # hadered case 
        df.columns = cols
        df = df[["parnme", "parval1"]]
    else:
        #haderless case: assume two columns (par name, value)
        df = pd.read_csv(params_csv, header=None, sep=None, engine='python')
        if df.shape[1] < 2:
            raise ValueError(f"{params_csv} must have at least 2 columns (parnme,parval1)")
        df = df.iloc[:, :2]
        df.columns = ["parnme", "parval1"]

    # drop a possible header row accidentally read as data
    df["parnme"] = df["parnme"].astype(str).str.strip().str.strip('"').str.strip("'")
    df = df[df["parnme"].str.lower() != "parnme"].copy()

    df["parval1"] = pd.to_numeric(df["parval1"], errors="raise")
    return dict(zip(df["parnme"], df["parval1"]))


def find_mf6_exe(root: Path) -> str:
    for cand in ["mf6.exe", "mf6"]:
        if (root / cand).exists():
            return str(root / cand)
    return "mf6"


def run_mf6(root: Path) -> None:
    exe = find_mf6_exe(root)
    p = subprocess.run([exe], cwd=str(root), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("MF6 failed")


def load_sim(root: Path):
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(root), verbosity_level=0)
    mnames = list(sim.model_names)
    if not mnames:
        raise RuntimeError("No models found in simulation")
    gwf = sim.get_model(mnames[0])
    return sim, gwf



def apply_parameters(gwf, pars: dict[str, float], root: Path) -> None:

    # parameters loading
    mK_upland_l1 = float(pars.get("mk_upland_l1", 1.0))
    mK_upland_l2 = float(pars.get("mk_upland_l2", 1.0))
    mK_ring_l1 = float(pars.get("mk_ring_l1", 1.0))
    mK_ring_l2 = float(pars.get("mk_ring_l2", 1.0))
    mK_rest_l1 = float(pars.get("mk_rest_l1", 1.0))
    mK_rest_l2 = float(pars.get("mk_rest_l2", 1.0))

    mR_upland = float(pars.get("mr_upland", 1.0))
    mR_rest = float(pars.get("mr_rest", 1.0))
    mFGW = float(pars.get("mfgw", 1.0))

    K_base = np.load(root / "K_base.npy")
    zone_ring = np.load(root / "zone_ring.npy").astype(bool)
    zone_upland = np.load(root / "zone_upland.npy").astype(bool)
    zone_rest = np.load(root / "zone_rest.npy").astype(bool)
    
    K_base = np.asarray(K_base).ravel()

    n1 = int(gwf.modelgrid.ncpl[0])
    ntot = K_base.size
    sl1 = slice(0, n1)
    sl2 = slice(n1, ntot)

    u1, u2 = zone_upland[sl1], zone_upland[sl2]
    r1, r2 = zone_ring[sl1],   zone_ring[sl2]
    z1, z2 = zone_rest[sl1],   zone_rest[sl2]

    K_new = K_base.copy()

    K_new[np.where(u1)[0]] *= mK_upland_l1
    K_new[n1 + np.where(u2)[0]] *= mK_upland_l2

    K_new[np.where(r1)[0]] *= mK_ring_l1
    K_new[n1 + np.where(r2)[0]] *= mK_ring_l2

    K_new[np.where(z1)[0]] *= mK_rest_l1
    K_new[n1 + np.where(z2)[0]] *= mK_rest_l2

    k_upland_l1 = (1e-3)*(mK_upland_l1)
    k_upland_l2 = (1e-2)*(mK_upland_l2)
    k_ring_l1 = (9e-2)*(mK_ring_l1)
    k_ring_l2 = (9e-1)*(mK_ring_l2)
    k_rest_l1 = (9e-3)*(mK_rest_l1)
    k_rest_l2 = (9e-2)*(mK_rest_l2)

    r_upland = (mR_upland) 
    r_rest = (mR_rest)

    mfgw = (mFGW)

    print(f"K upland l1: {k_upland_l1}")
    print(f"K upland l2: {k_upland_l2}")
    print(f"K ring l1: {k_ring_l1}")
    print(f"K ring l2: {k_ring_l2}")
    print(f"K rest l1: {k_rest_l1}")
    print(f"K rest l2: {k_rest_l2}")
    print(f"R upland: {r_upland}")
    print(f"R rest: {r_rest}")
    print(f"mfgw: {mfgw}")

    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found")
    npf.k.set_data(K_new)

    rch = gwf.get_package("rch")
    spd_path = root / "rch_spd_base.npy"

    if rch is not None and spd_path.exists():
        spd = np.load(spd_path, allow_pickle=True)
        spd_new = spd.copy()

        # find the recharge field name safely
        names = spd_new.dtype.names
        rfield = "recharge" if "recharge" in names else names[-1]
        # recharge values (should be ncpl(/layer1) lenght in my setup)
        rch_arr = spd_new[rfield].astype(float) 
        print("ncpl layer1:", gwf.modelgrid.ncpl[0])
        print("zone_upland size:", zone_upland.size)
        print("rch_arr size:", rch_arr.size)


        #building a top layer upland-mask with the same legnth as rch_arr
        #zone upland is full-model length ---> slice to layer 1
        upland_top = zone_upland[:n1]

        # safety check: ensure the mask matches recharge array length
        if rch_arr.size != upland_top.size:
            raise ValueError(
                f"Recharge array length ({rch_arr.size}) != upland_top mask length ({upland_top.size}). "
                "This means your rch_spd_base.npy is not top-layer-only or ncpl indexing differs."
            )   

        # apply zoned multipliers
        rch_arr[upland_top] *= mR_upland
        rch_arr[~upland_top] *= mR_rest

        # assign back and set stress period 0
        spd_new[rfield] = rch_arr
        rch.stress_period_data.set_data({0: spd_new})

    evt = gwf.get_package("evt")
    evt_spd_path = root / "evt_spd_base.npy"

    if evt is not None and evt_spd_path.exists():
        spd_evt = np.load(evt_spd_path, allow_pickle=True)

    # Must be structured with named fields
    if spd_evt.dtype.names is None:
        raise ValueError(
            "evt_spd_base.npy has no named fields (dtype.names is None). "
            "Recreate it as a structured array with fields: node, surf, rate, exdp."
        )

    spd_evt_new = spd_evt.copy()

    # rate field (in our recommended file it is exactly 'rate')
    if "rate" not in spd_evt_new.dtype.names:
        raise ValueError(f"evt_spd_base.npy missing 'rate' field. Fields: {spd_evt_new.dtype.names}")

    spd_evt_new["rate"] = spd_evt_new["rate"].astype(float) * mFGW
    spd_evt_new["rate"] = np.maximum(spd_evt_new["rate"], 0.0)

    evt.stress_period_data.set_data({0: spd_evt_new})

    return k_upland_l1, k_upland_l2, k_ring_l1, k_ring_l2, k_rest_l1, k_rest_l2, r_upland, r_rest, mfgw


def find_headfile(root: Path) -> Path:
    cands = sorted(root.glob("*.hds"))
    if not cands:
        raise FileNotFoundError("No *.hds file found after MF6 run")
    return cands[0]

def extract_sim_heads(headfile_path: Path, obs_cellids_path: Path) -> pd.DataFrame:
    obs = pd.read_csv(obs_cellids_path)
    obs["obs_id"] = obs["obs_id"].astype(str)

    hf = HeadFile(str(headfile_path))
    #totim = hf.get_times()[-1]
    #h = hf.get_data(totim=totim)
    h = hf.get_data()[0][0]

    cols = set(obs.columns.str.lower())
    if {"cellid"}.issubset(cols):
        #cellid = obs["cellid"].astype(int).to_numpy() - 1
        cellid = obs["cellid"].astype(int).to_numpy() 
        # if h.ndim != 2:
        #     raise ValueError(f"Expected heads (nlay, ncpl). Got {h.shape}")
        head_sim = h[cellid]
    elif {"k", "i", "j"}.issubset(cols):
        k = obs["k"].astype(int).to_numpy() - 1
        i = obs["i"].astype(int).to_numpy() - 1
        j = obs["j"].astype(int).to_numpy() - 1
        if h.ndim != 3:
            raise ValueError(f"Expected heads (nlay, nrow, ncol). Got {h.shape}")
        head_sim = h[k, i, j]
    else:
        raise ValueError("obs_cellids.csv must contain (layer,cellid) or (k,i,j)")

    return pd.DataFrame({"obs_id": obs["obs_id"].to_numpy(), "head_sim": head_sim})


def write_sim_heads_dat(root: Path, simdf: pd.DataFrame) -> None:
    obs_ids = pd.read_csv(root / "obs_heads.csv")["obs_id"].astype(str).tolist()

    sim = simdf.copy()
    sim["obs_id"] = sim["obs_id"].astype(str)
    sim = sim.set_index("obs_id").reindex(obs_ids)

    if sim["head_sim"].isna().any():
        missing = sim.index[sim["head_sim"].isna()].tolist()[:10]
        raise RuntimeError(f"Missing simulated heads for obs_ids, e.g.: {missing}")

    sim.reset_index()[["obs_id", "head_sim"]].to_csv(
        root / "sim_heads.dat",
        sep=" ",
        header=False,
        index=False,
        float_format="%.10f"
    )


def main():
    root = Path(__file__).resolve().parent
    pars = read_params(root / "params.csv")
    sim, gwf = load_sim(root)

    k_upland_l1,k_upland_l2,k_ring_l1,k_ring_l2,k_rest_l1,k_rest_l2,r_upland,r_rest,mfgw = apply_parameters(gwf, pars, root)
    sim.write_simulation()
    run_mf6(root)

    headfile = find_headfile(root)
    simdf = extract_sim_heads(headfile, root / "obs_cellids.csv")
    write_sim_heads_dat(root, simdf)

    params_path = root / "absolute_params.csv"
    if not params_path.exists():    
        pd.DataFrame([
            ["k_upland_l1", k_upland_l1],
            ["k_upland_l2", k_upland_l2],
            ["k_ring_l1",   k_ring_l1],
            ["k_ring_l2",   k_ring_l2],
            ["k_rest_l1",   k_rest_l1],
            ["k_rest_l2",   k_rest_l2],
            ["r_upland",    r_upland],
            ["r_rest",      r_rest],
            ["mfgw", mfgw]
        ]).to_csv(params_path, index=False, header=False)


if __name__ == "__main__":
    main()
