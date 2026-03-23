"""
    Forward run script for AI dataset generation (ABSOLUTE parameter values).
    This script provides workflow structure:
        - Reads 'params.csv' file
        - Apply
        - Run
        - Extract 135 observed heads
         - Write 'sim_heads.dat'
    Applies absolute values instead of multipliers.....
"""

from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import flopy
from flopy.utils import HeadFile
from flopy.discretization import StructuredGrid, UnstructuredGrid

"""
IO helpers
"""
def read_params(params_csv: Path) -> dict[str, float]:
    df = pd.read_csv(params_csv, sep=None, engine='python')
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if not {"parnme", "parval1"}.issubset(df.columns):
        raise ValueError(f"{params.csv} must have 2 columns: parnme, parval1")

    df["parnme"] = df["parnme"].astype(str).str.strip()
    df["parval1"] = pd.to_numeric(df["parval1"], errors='raise')
    return dict(zip(df["parnme"], df["parval1"]))


def find_mf6_exe(root: Path) -> str:
    for cand in ["mf6.exe", "mf6"]:
        if (root / cand).exists():
            return str(root / cand)
    return "mf6"

def _build_zone_masks_disu_disv(gwf, ring_cellids_0based: np.ndarray, upland_cellids_0based: np.ndarray):
    mg = gwf.modelgrid
    if isinstance(mg, StructuredGrid):
        raise ValueError(
            "This script currently expects DISU/DISV (UnstructuredGrid) cell2d indices for masks. "
        )

    ncpl = mg.nnodes
    zone_ring = np.zeros(ncpl, dtype=bool)
    zone_upland = np.zeros(ncpl, dtype=bool)

    ring_cellids_0based = np.asarray(ring_cellids_0based, dtype=int)
    upland_cellids_0based = np.asarray(upland_cellids_0based, dtype=int)

    if ring_cellids_0based.size:
        zone_ring[ring_cellids_0based] = True
    if upland_cellids_0based.size:
        zone_upland[upland_cellids_0based] = True

    zone_rest = ~(zone_ring | zone_upland)
    return zone_ring, zone_upland, zone_rest

def export_base_arrays_and_masks(gwf, template_dir: Path, zone_ring: np.ndarray, zone_upland: np.ndarray, zone_rest: np.ndarray):
    np.save(template_dir / "zone_ring.npy", zone_ring.astype(bool))
    np.save(template_dir / "zone_upland.npy", zone_upland.astype(bool))
    np.save(template_dir / "zone_rest.npy", zone_rest.astype(bool))

    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found (needed for save the K base values).")
    np.save(template_dir / "K_base.npy", npf.k.array)

    rch = gwf.get_package("rch")
    if rch is not None:
        spd0 = rch.stress_period_data.get_data(0)
        np.save(template_dir / "rch_spd_base.npy", spd0)

    evt = gwf.get_package("evt")
    if evt is not None:
        spd0_evt = evt.stress_period_data.get_data(0)
    if spd0_evt is None:
        raise RuntimeError("EVT exists but stress_period_data for kper=0 is empty.")
    np.save(template_dir / "evt_spd_base.npy", spd0_evt)



def run_mf6(root: Path) -> None:
    exe = find_mf6_exe(root)
    p = subprocess.run([exe], cwd=str(root), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError("FUCKING MF6 FAILED !!!!!")

def load_sim(root:Path):
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(root), verbosity_level=0)
    mnames = list(sim.model_names)
    if not mnames:
        raise RuntimeError("No models were found in this simulation.")
    gwf = sim.get_model(mnames[0])
    return sim, gwf


"""
ABSOLUTE PARAMETER APPLICATION (DISU)
"""
def apply_parameters_abs(gwf, pars:dict[str, float], root: Path) -> None:

    mK_upland_l1 = float(pars.get("mk_upland_l1", 1.0))
    mK_upland_l2 = float(pars.get("mk_upland_l2", 1.0))
    mK_ring_l1   = float(pars.get("mk_ring_l1",   1.0))
    mK_ring_l2   = float(pars.get("mk_ring_l2",   1.0))
    mK_rest_l1   = float(pars.get("mk_rest_l1",   1.0))
    mK_rest_l2   = float(pars.get("mk_rest_l2",   1.0))

    mR_upland = float(pars.get("mr_upland", 1.0))
    mR_rest   = float(pars.get("mr_rest",   1.0))
    mFGW      = float(pars.get("mfgw",  1.0))

    # Zone masks from cellid lists
    ring_cellids = np.load(root / "ring_cellids.npy")
    upland_cellids = np.load(root / "upland_cellids.npy")
    zone_ring, zone_upland, zone_rest = _build_zone_masks_disu_disv(gwf, ring_cellids, upland_cellids)
    export_base_arrays_and_masks(gwf, root, zone_ring, zone_upland, zone_rest)

    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found in loaded GWF model.")

    K_base = (npf.k.array).ravel()

    zone_ring   = np.asarray(np.load(root / "zone_ring.npy")).astype(bool).ravel()
    zone_upland = np.asarray(np.load(root / "zone_upland.npy")).astype(bool).ravel()
    zone_rest   = np.asarray(np.load(root / "zone_rest.npy")).astype(bool).ravel()

    if not (K_base.size == zone_ring.size == zone_upland.size == zone_rest.size):
        raise ValueError(
            f"Size mismatch: K_base({K_base.size}), ring({zone_ring.size}), upland({zone_upland.size}), rest({zone_rest.size}). "
            "These arrays must all be full-model length (nodes)."
        )

    n1 = int(gwf.modelgrid.ncpl[0])
    ntot = K_base.size
    if n1 <= 0 or n1 >= ntot:
        raise ValueError(f"Bad ncpl[0]={n1} vs total nodes={ntot}.")

    sl1 = slice(0, n1)
    sl2 = slice(n1, ntot)

    u1, u2 = zone_upland[sl1], zone_upland[sl2]
    r1, r2 = zone_ring[sl1],   zone_ring[sl2]
    z1, z2 = zone_rest[sl1],   zone_rest[sl2]

    ## Build absolute K array
    K_new = K_base.copy()

    K_new[np.flatnonzero(u1)] *= mK_upland_l1
    K_new[n1 + np.flatnonzero(u2)] *= mK_upland_l2

    K_new[np.flatnonzero(r1)] *= mK_ring_l1
    K_new[n1 + np.flatnonzero(r2)] *= mK_ring_l2

    K_new[np.flatnonzero(z1)] *= mK_rest_l1
    K_new[n1 + np.flatnonzero(z2)] *= mK_rest_l2

    npf.k.set_data(K_new)

    rch = gwf.get_package("rch")
    spd_path = root / "rch_spd_base.npy"
    if rch is not None and spd_path.exists():
        spd = np.load(spd_path, allow_pickle=True)
        spd_new = spd.copy()

        names = spd_new.dtype.names
        if names is None:
            raise ValueError("rch_spd_base.npy must be a structured array (dtype.names is None).")
        rfield = "recharge" if "recharge" in names else names[-1]

        rch_arr = spd_new[rfield].astype(float)
        upland_top = zone_upland[:n1]  # layer-1 mask

        if rch_arr.size != upland_top.size:
            raise ValueError(
                f"Recharge array length ({rch_arr.size}) != layer-1 node count ({upland_top.size}). "
                "Confirm that rch_spd_base.npy is top-layer only."
            )

        rch_arr[upland_top] *= mR_upland
        rch_arr[~upland_top] *= mR_rest

        spd_new[rfield] = rch_arr
        rch.stress_period_data.set_data({0: spd_new})

    # --- EVT (scale rate by mFGW) ---
    evt = gwf.get_package("evt")
    evt_spd_path = root / "evt_spd_base.npy"
    if evt is not None and evt_spd_path.exists():
        spd_evt = np.load(evt_spd_path, allow_pickle=True)
        if spd_evt.dtype.names is None:
            raise ValueError(
                "evt_spd_base.npy has no named fields (dtype.names is None). "
                "Recreate it as a structured array with fields like: node, surf, rate, exdp."
            )
        spd_evt_new = spd_evt.copy()

        # rate field: use 'rate' if present, else last numeric field
        if "rate" in spd_evt_new.dtype.names:
            rate_field = "rate"
        else:
            # try a reasonable fallback
            rate_field = spd_evt_new.dtype.names[-2] if len(spd_evt_new.dtype.names) >= 2 else spd_evt_new.dtype.names[-1]

        spd_evt_new[rate_field] = spd_evt_new[rate_field].astype(float) * mFGW
        spd_evt_new[rate_field] = np.maximum(spd_evt_new[rate_field], 0.0)

        evt.stress_period_data.set_data({0: spd_evt_new})


def find_headfile(root: Path) -> Path:
    cands = sorted(root.glob("*.hds"))
    if not cands:
        raise FileNotFoundError("No *.hds file found after MF6 run")
    return cands[0]

def extract_sim_heads(headfile_path: Path, obs_cellids_path: Path) -> pd.DataFrame:
    """
    Robust extraction for DISU-style observations by cellid.
    Keeps the existing concept: obs_cellids.csv has 'obs_id' + 'cellid' (0-based)
    """
    obs = pd.read_csv(obs_cellids_path)
    obs["obs_id"] = obs["obs_id"].astype(str)

    hf = HeadFile(str(headfile_path))
    totim = hf.get_times()[-1]
    h = hf.get_data(totim=totim)

    # normalizing to 1D vector for indexing by "cellid"
    if isinstance(h, np.ndarray):
        if h.ndim == 1:
            hvec = h
        elif h.ndim == 2:
            #typical: (nlay, nnode_layer). for heads obs, assuming layer 1
            hvec = h[0, :]
        elif h.ndim == 3:
            hvec = h[-1, 0, :]
        else:
            raise ValueError(f"Unexpected head array shape: {h.shape}")
    else:
        raise ValueError("HeadFile.get_data() did not return a numpy array.")
    
    cols = set(obs.columns.str.lower())
    if "cellid" not in cols:
        raise ValueError("obs_cellids.csv must contain a 'cellid' column for DISU indexing.")
    
    cellid = obs["cellid"].astype(int).to_numpy()
    head_sim = hvec[cellid]

    return pd.DataFrame({"obs_id": obs["obs_id"].to_numpy(), "head_sim": head_sim})


def write_sim_heads_dat(root: Path, simdf: pd.DataFrame) -> None:
    """
    Exactly like the PEST++ forward runner: enforce the ordering in obs_heads.csv,
    then write sim_heads.dat (2 columns).
    """

    obs_id = pd.read_csv(root / "obs_heads.csv")["obs_id"].astype(str).tolist()

    sim = simdf.copy()
    sim["obs_id"] = sim["obs_id"].astype(str)
    sim = sim.set_index("obs_id").reindex(obs_id)

    if sim["head_sim"].isna().any():
        missing = sim.index[sim["head_sim"].isna()].tolist()[:10]
        raise RuntimeError(f"Missing simulated heads for obs_ids, e.g. {missing}")
    
    sim.reset_index()[["obs_id", "head_sim"]].to_csv(
        root / "sim_heads.dat",
        sep=" ",
        header=False,
        index=False,
        float_format = "%.10f"
    )


########## MAIN FUNCTION
def main():
    root = Path(__file__).resolve().parent
    pars = read_params(root / "params.csv")
    sim, gwf = load_sim(root)

    apply_parameters_abs(gwf, pars, root)
    sim.write_simulation()
    run_mf6(root)

    headfile = find_headfile(root)
    simdf = extract_sim_heads(headfile, root / "obs_cellids.csv")
    write_sim_heads_dat(root, simdf)


if __name__ == "__main__":
    main()


