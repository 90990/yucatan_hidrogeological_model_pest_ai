"""
Phase A: build the portable PEST template folder.

What this script DOES (run outside PEST):
  1) Copies the existing MF6 model folder (my calibrated "base" run) into ./template
  2) Builds/exports:
        - zone masks (zone_ring.npy, zone_upland.npy, zone_rest.npy)
        - base arrays (K_base.npy, R_base.npy if present)
        - obs_cellids.csv (maps each observation to a cell index)
        - sim_heads.ins (static instruction file for PEST to read sim_heads.dat)

Inputs:
  - base_model_dir: folder that already contains the runnable MF6 simulation (mfsim.nam, etc.)
  - obs_heads.csv: columns include: obs_id, x, y, head_obs (optional: weight)
  - ring_cellids.npy and upland_cellids.npy: 0-based cell2d indices for the grid (DISU/DISV)

Run:
  python prep_template_clean.py --base_model_dir "C:/path/to/your/mf6_run" --obs_heads_csv "C:/path/to/obs_heads.csv" 
        --ring_cellids_npy "C:/path/to/ring_cellids.npy" --upland_cellids_npy "C:/path/to/upland_cellids.npy"
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import flopy
from flopy.discretization import StructuredGrid


def _ensure_clean_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _load_first_gwf(sim: flopy.mf6.MFSimulation):
    mnames = list(sim.model_names)
    if not mnames:
        raise RuntimeError("No models found in the simulation.")
    return sim.get_model(mnames[0])


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




def build_obs_cellids_disu_disv(gwf, obs_heads_csv: Path, out_csv: Path, layer_default: int = 1):
    obs = pd.read_csv(obs_heads_csv)
    req = {"obs_id", "x", "y"}
    if not req.issubset(set(obs.columns)):
        raise ValueError(f"{obs_heads_csv} must contain columns {req}. Found: {list(obs.columns)}")

    mg = gwf.modelgrid

    cellids = []
    for _, r in obs.iterrows():
        x = float(r["x"]); y = float(r["y"])
        icell2d = mg.intersect(x, y)
        if icell2d is None or int(icell2d) < 0:
            raise RuntimeError(f"Point not in grid: obs_id={r['obs_id']} x={x} y={y}")
        cellids.append(int(icell2d) + 1)  # 1-based for the CSV file

    out = pd.DataFrame({
        "obs_id": obs["obs_id"].astype(str).to_list(),
        "layer": [int(layer_default)] * len(obs),
        "cellid": cellids
    })
    out.to_csv(out_csv, index=False)


def write_sim_heads_ins(obs_heads_csv: Path, ins_path: Path):
    obs = pd.read_csv(obs_heads_csv)
    obs_ids = obs["obs_id"].astype(str).tolist()

    with open(ins_path, "w", newline="\n") as f:
        f.write("pif ~\n")
        for oid in obs_ids:
            f.write(f"l1 w !{oid}!\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--obs_heads_csv", required=True)
    ap.add_argument("--ring_cellids_npy", required=True)
    ap.add_argument("--upland_cellids_npy", required=True)
    ap.add_argument("--template_dir", default="template_1")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    obs_heads_csv = Path(args.obs_heads_csv).expanduser().resolve()
    template_dir = (root / args.template_dir).resolve()

    _ensure_clean_dir(template_dir)

    # Copy runnable MF6 folder into template_dir
    shutil.copytree(base_model_dir, template_dir, dirs_exist_ok=True)

    # Load simulation from template_dir
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(template_dir), verbosity_level=0)
    gwf = _load_first_gwf(sim)

    # Zone masks from cellid lists
    ring_cellids = np.load(Path(args.ring_cellids_npy))
    upland_cellids = np.load(Path(args.upland_cellids_npy))
    zone_ring, zone_upland, zone_rest = _build_zone_masks_disu_disv(gwf, ring_cellids, upland_cellids)

    export_base_arrays_and_masks(gwf, template_dir, zone_ring, zone_upland, zone_rest)

    # Build obs_cellids.csv and copy obs_heads.csv
    build_obs_cellids_disu_disv(gwf, obs_heads_csv, template_dir / "obs_cellids.csv")
    shutil.copy2(obs_heads_csv, template_dir / "obs_heads.csv")

    # Static instruction file
    write_sim_heads_ins(template_dir / "obs_heads.csv", template_dir / "sim_heads_1.ins")

    ## SEED: sim_budget.dat + sim_budget.ins (template placeholders)
    budget_dat = template_dir / "sim_budget_1.dat"
    budget_ins = template_dir / "sim_budget_1.ins"

    if not budget_dat.exists():
        # 3 lines, one value per line, fixed orders:
        # 1) RCH_IN, 2) EVT_OUT, 3) CHD_OUT
        budget_dat.write_text("0.0\n0.0\n0.0\n")

    if not budget_ins.exists():
        with open(budget_ins, "w") as f:
            f.write("pif ~\n")
            f.write("l1 !rch_in!\n")
            f.write("l1 !evt_out!\n")
            f.write("l1 !chd_out!\n")

    # Seed params.csv if missing

    params_path = template_dir / "params_1.csv"
    if not params_path.exists():    
        pd.DataFrame([
            ["mk_upland_l1", 1.0],
            ["mk_upland_l2", 1.0],
            ["mk_ring_l1",   1.0],
            ["mk_ring_l2",   1.0],
            ["mk_rest_l1",   1.0],
            ["mk_rest_l2",   1.0],
            ["mr_upland",    1.0],
            ["mr_rest",      1.0],
            ["mFGW_upland", 1.0],
            ["mFGW_rest", 1.0],
        ]).to_csv(params_path, index=False, header=False)
        
    print("Template built at:", template_dir)


if __name__ == "__main__":
    main()
