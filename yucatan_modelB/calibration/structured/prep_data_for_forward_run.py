"""
    What this script actually DOES:
        1 - Copy the existing MF6 model folder for any case or grid
        2 - Build/exports:
            - Discretization zone masks
            - base arrays
            - obs_cellids.csv

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
from flopy.discretization import StructuredGrid, UnstructuredGrid

def _ensure_clean_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def _load_first_gwf(sim: flopy.mf6.MFSimulation):
    mnames = list(sim.model_names)
    if not mnames:
        raise RuntimeError("No models found in the simulation")
    return sim.get_model(mnames[0])

def _build_zone_masks_disu_disv(gwf, ring_cellids_0based: np.ndarray, upland_cellids_0based: np.ndarray):
    mg = gwf.modelgrid

    nrow = mg.nrow
    ncol = mg.ncol

    zone_ring = np.zeros((nrow,ncol), dtype=bool) 
    zone_upland = np.zeros((nrow,ncol), dtype=bool) 

    ring_cellids_0based = np.asarray(ring_cellids_0based, dtype=int)
    upland_cellids_0based = np.asarray(upland_cellids_0based, dtype=int)

    if ring_cellids_0based.size:
        zone_ring[ring_cellids_0based] = True
    if upland_cellids_0based.size:
        zone_upland[upland_cellids_0based] = True

    return zone_ring, zone_upland

def export_base_arrays_and_masks(gwf, template_dir: Path, zone_ring: np.ndarray, zone_upland: np.ndarray):
    np.save(template_dir / "zone_ring.npy", zone_ring.astype(bool))
    np.save(template_dir / "zone_upland.npy", zone_upland.astype(bool))

    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found !")
    np.save(template_dir / "K_base.npy", npf.k.array)

    rch = gwf.get_package("rch")
    if rch is not None:
        spd0 = rch.stress_period_data.get_data(0)
        np.save(template_dir / "rch_spd_base.npy", spd0)

    evt = gwf.get_package("evt")
    if evt is not None:
        spd0_evt = evt.stress_period_data.get_data(0)
        np.save(template_dir / "evt_spd_base.npy", spd0_evt)

def build_obs_cellids_disu_disv(gwf, obs_heads_csv: Path, out_csv: Path, layer_default: int=1):
    obs = pd.read_csv(obs_heads_csv)
    req = {"obs_id", "x", "y"}
    if not req.issubset(set(obs.columns)):
        raise ValueError(f"{obs_heads_csv} must contain columns {req}. Found: {list(obs.columns)}")
    
    mg = gwf.modelgrid
    xoff = 142384.3855
    yoff = 2161902.4815
    crs = 32616
    angrot = 0
    mg.set_coord_info(xoff=xoff, yoff=yoff, angrot=angrot, crs=crs)

    rows = []
    cols = []

    for _,r in obs.iterrows():
        x = float(r["x"]); y = float(r["y"])
        ij = [gwf.modelgrid.intersect(x,y)]

        if ij is None:
            raise RuntimeError(f"Point not in grid: x={x} y={y}")

        rows.append(ij[0][0])
        cols.append(ij[0][1])
        

    out = pd.DataFrame({
        "obs_id": obs["obs_id"].astype(str).to_list(),
        "layer": [int(layer_default)] * len(obs),
        "rows": rows,
        "cols": cols
    })
    out.to_csv(out_csv, index=False)

def write_sim_heads_ins(obs_heads_csv:Path, ins_path:Path):
    obs = pd.read_csv(obs_heads_csv)
    obs_ids = obs["obs_id"].astype(str).tolist()

    with open(ins_path, "w", newline='\n') as f:
        f.write("pif ~\n")
        for oid in obs_ids:
            f.write(f"l1 w !{oid}!\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--obs_heads_csv", required=True)
    ap.add_argument("--ring_cellids_npy", required=True)
    ap.add_argument("--upland_cellids_npy", required=True)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    obs_heads_csv = Path(args.obs_heads_csv).expanduser().resolve()
    data_dir = (root / args.data_dir).resolve()

    _ensure_clean_dir(data_dir)

    shutil.copytree(base_model_dir, data_dir, dirs_exist_ok=True)
    
    shutil.copy2(root / "forward_run.py", data_dir / "forward_run.py")
    shutil.copy2(root / "compute_metrics.py", data_dir / "compute_metrics.py")
    shutil.copy2(root / "absolute_params1_caseB.csv", data_dir / "absolute_params1_caseB.csv")

    # loading simulation
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(data_dir), verbosity_level=0)
    gwf = _load_first_gwf(sim)
    
    # zone masks from cellids list
    ring_cellids = np.load(Path(args.ring_cellids_npy))
    upland_cellids = np.load(Path(args.upland_cellids_npy))
    zone_ring, zone_upland = _build_zone_masks_disu_disv(gwf, ring_cellids, upland_cellids)

    export_base_arrays_and_masks(gwf, data_dir, zone_ring, zone_upland)

    # build obs_cellids.csv and copy obs_heads.csv
    build_obs_cellids_disu_disv(gwf, obs_heads_csv, data_dir / "obs_cellids.csv")
    shutil.copy2(obs_heads_csv, data_dir / "obs_heads.csv")

    # static instruction file
    write_sim_heads_ins(data_dir / "obs_heads.csv", data_dir / "sim_heads.ins")

if __name__ == "__main__":
    main()


