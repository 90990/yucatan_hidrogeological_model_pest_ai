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
import matplotlib.pyplot as plt
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

    # ABSOLUTE parameters loading
    K_upland_l1 = float(pars.get("k_upland_l1", 1.0))
    K_upland_l2 = float(pars.get("k_upland_l2", 1.0))
    K_ring_l1 = float(pars.get("k_ring_l1", 1.0))
    K_ring_l2 = float(pars.get("k_ring_l2", 1.0))
    K_rest_l1 = float(pars.get("k_rest_l1", 1.0))
    K_rest_l2 = float(pars.get("k_rest_l2", 1.0))

    mR_upland = float(pars.get("r_upland", 1.0))
    mR_rest = float(pars.get("r_rest", 1.0))
    mFGW_upland = float(pars.get("fgw_upland", 1.0))
    mFGW_rest = float(pars.get("fgw_rest", 1.0))

    #ncpl = gwf.modelgrid.ncpl
    # --- load base arrays / masks ---
    K_base = np.asarray(np.load(root / "K_base.npy")).ravel()

    zone_ring   = np.asarray(np.load(root / "zone_ring.npy")).astype(bool)
    zone_upland = np.asarray(np.load(root / "zone_upland.npy")).astype(bool)
    zone_rest   = np.asarray(np.load(root / "zone_rest.npy")).astype(bool)
    
    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found")

    karr = npf.k.array
    if karr.ndim != 3:
        raise ValueError(f"Expected NPF.k.array to have 2 dimensions (nlay, nrow, ncol). Got {karr.shape}")

    nlay,nrow,ncol = karr.shape
    if nlay < 2:
        raise ValueError(f"Expected at least 2 layers. Got {nlay}")
    if zone_upland.size != (gwf.modelgrid.ncpl):
        raise ValueError(f"zone_upland size {zone_upland.size} != ncpl {ncpl}")
    
    K2D = np.empty((nlay, nrow, ncol), dtype=float)

    zone_upland = (zone_upland) & (gwf.modelgrid.idomain[0])
    zone_ring = (zone_ring) & (gwf.modelgrid.idomain[0])
    
    # K layer 1
    K2D[0][:][:] = K_rest_l1
    K2D[0][zone_upland] = K_upland_l1
    K2D[0][zone_ring] = K_ring_l1

    K2D[1][:][:] = K_rest_l2
    K2D[1][zone_upland] = K_upland_l2
    K2D[1][zone_ring] = K_ring_l2

    npf.k.set_data(K2D)

    rch = gwf.get_package("rch")
    spd_path = root / "rch_spd_base.npy"

    if rch is not None and spd_path.exists():
        spd = np.load(spd_path, allow_pickle=True)
        spd_new = spd.copy()
        names = spd_new.dtype.names
        rfield = "recharge" if "recharge" in names else names[-1]
        rch_arr = spd_new[rfield].astype(float) 

        # if rch_arr.size != gwf.modelgrid.ncpl:
        #     raise ValueError(f"RCH array size {rch_arr.size} != ncpl {gwf.modelgrid.ncpl}")

        zone_upland = (zone_upland) & (gwf.modelgrid.idomain[0])
        rch_arr[zone_upland] *= mR_upland
        rch_arr[~zone_upland] *= mR_rest

        spd_new[rfield] = rch_arr
        rch.stress_period_data.set_data({0: spd_new})

      # --- EVT (scale rate by zoned multipliers) ---
    evt = gwf.get_package("evt")
    evt_spd_path = root / "evt_spd_base.npy"
    if evt is not None and evt_spd_path.exists():
        spd_evt = np.load(evt_spd_path, allow_pickle=True)
        if spd_evt.dtype.names is None:
            raise ValueError("evt_spd_base.npy must be a structured array with named fields.")

    spd_evt_new = spd_evt.copy()

    # 1) localizar campo de nodos (según cómo lo guardaste)
    node_field = None
    for cand in ("node", "nodenumber", "cellid", "icell", "id"):
        if cand in spd_evt_new.dtype.names:
            node_field = cand
            break
    if node_field is None:
        raise ValueError(f"Cannot find node field in evt_spd_base.npy. Fields: {spd_evt_new.dtype.names}")

    evt_nodes = spd_evt_new[node_field]

    # Si viene como tuplas/objetos (a veces cellid se guarda raro), lo convertimos a int
    if evt_nodes.dtype == object:
        tmp = []
        for v in evt_nodes:
            # si v es (k, node) o (node,) toma el último
            if isinstance(v, (tuple, list, np.ndarray)):
                tmp.append(int(v[-1]))
            else:
                tmp.append(int(v))
        evt_nodes = np.array(tmp, dtype=int)
    else:
        evt_nodes = evt_nodes.astype(int)

    # 2) auto-detección 1-based vs 0-based (muy común en archivos “externos”)
    #    Si tus nodos van 1..N, pásalos a 0..N-1
    if evt_nodes.min() >= 1 and evt_nodes.max() <= zone_upland.size and (evt_nodes == 0).sum() == 0:
        # ojo: esto asume que realmente están 1-based
        # si ya están 0-based, normalmente aparece algún 0
        evt_nodes = evt_nodes - 1

    # 3) validar rango
    if evt_nodes.min() < 0 or evt_nodes.max() >= zone_upland.size:
        raise ValueError(
            f"EVT node ids out of range. min={evt_nodes.min()}, max={evt_nodes.max()}, zone_upland.size={zone_upland.size}"
        )

    # 4) máscara EVT-length: para cada celda EVT, ¿está en upland?
    evt_upland = zone_upland[evt_nodes]     # tamaño = len(spd_evt_new) (=18987)

    # 5) campo de tasa
    if "rate" in spd_evt_new.dtype.names:
        rate_field = "rate"
    else:
        raise ValueError(f"evt_spd_base.npy missing 'rate' field. Fields: {spd_evt_new.dtype.names}")

    rate = spd_evt_new[rate_field].astype(float)

    # 6) aplicar multiplicadores por zona
    rate[evt_upland] *= mFGW_upland
    rate[~evt_upland] *= mFGW_rest
    rate = np.maximum(rate, 0.0)

    # 7) guardar de vuelta
    spd_evt_new[rate_field] = rate
    evt.stress_period_data.set_data({0: spd_evt_new})


def find_headfile(root: Path) -> Path:
    cands = sorted(root.glob("*.hds"))
    if not cands:
        raise FileNotFoundError("No *.hds file found after MF6 run")
    return cands[0]

def extract_sim_heads(headfile_path: Path, obs_cellids_path: Path, gwf) -> pd.DataFrame:
    obs = pd.read_csv(obs_cellids_path)
    obs["obs_id"] = obs["obs_id"].astype(str)

    hf = HeadFile(str(headfile_path))
    totim = hf.get_times()[-1]
    h = hf.get_data(totim=totim)
    #h = hf.get_data()[0][0]

    if h.ndim != 3:
        raise ValueError(f"Expected heads 2D (nlay,nrow,ncol) for DIS, Got {h.shape}")

    #layer = obs["layer"].astype(int).to_numpy - 1 # Layer column is 1-based in file
    rows = obs["rows"].astype(int).to_numpy()  # Must be 0-based
    cols = obs["cols"].astype(int).to_numpy()  # Must be 0-based

    head_sim = h[0, rows, cols]

    return pd.DataFrame({"obs_id":obs["obs_id"].astype(str),"rows": obs["rows"].to_numpy(), "cols":obs["cols"].to_numpy(), "head_sim": head_sim})


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

def plot_results(gwf, headFile_path: Path, title, layer, out_png, cbar_label):

    gwf.modelgrid.set_coord_info(xoff=142384.3855, yoff=2161902.4815, angrot=0, crs=32616)

    hf = HeadFile(str(headFile_path))
    totim = hf.get_times()[-1]
    h = hf.get_data(totim=totim)

    fig,ax = plt.subplots()
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
    hd = pmv.plot_array(h[layer], ax=ax, alpha=1.0, cmap='viridis')
    contour = pmv.contour_array(h[layer], ax=ax, levels=5, colors='black')
    #pmv.plot_grid(lw=0.05, alpha=0.15)
    plt.colorbar(hd, label=cbar_label)
    plt.clabel(contour, fmt="%1.0f")
    ax.set_title(title)
    ax.set_xlabel("Este [m]")
    ax.set_ylabel("Norte [m]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    pars = read_params(root / "absolute_params1_caseD.csv")
    sim, gwf = load_sim(root)

    apply_parameters(gwf, pars, root)
    sim.write_simulation()
    run_mf6(root)

    headfile = find_headfile(root)
    simdf = extract_sim_heads(headfile, root / "obs_cellids.csv", gwf)
    write_sim_heads_dat(root, simdf)
    for layer in (0,1):
        plot_results(gwf, headfile, f"Caso D - Calibrado\nMalla estructurada - Capa {int(layer)+1}", layer, root / f"estruc_casoD_calibrado_layer{int(layer)+1}", "Cargas hidráulicas [m.s.n.m.]")


if __name__ == "__main__":
    main()
