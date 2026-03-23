
from __future__ import annotations
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import flopy
from flopy.utils import CellBudgetFile
from flopy.utils import HeadFile
import matplotlib.pyplot as plt


def read_params(params_csv: Path) -> dict[str, float]:
    """
    Read parameters from params.csv.
    Accepts both:
      - headered CSV: parnme,parval1
      - headerless CSV: <parnme>,<parval1>
    """
    df = pd.read_csv(params_csv)

    if {"parnme", "parval1"}.issubset(df.columns):
        pass
    else:
        # Headerless fallback: force column names
        df = pd.read_csv(params_csv, header=None, names=["parnme", "parval1"])
    return dict(zip(df["parnme"].astype(str), df["parval1"].astype(float)))


def _find_budget_file(root: Path) -> Path:
    for cand in [
        root / "gwf.cbc",
        root / "gwf.bud",
        root / "gwf_1.cbc",
        root / "gwf_1.bud",
    ]:
        if cand.exists():
            return cand
    # any *.cbc or *.bud
    for ext in ("*.cbc", "*.bud"):
        hits = list(root.glob(ext))
        if hits:
            return hits[0]
    raise FileNotFoundError("No cell-by-cell budget file (*.cbc or *.bud) found in model run folder.")


def apply_parameters(gwf, pars: dict[str, float], root: Path) -> None:
    mK_ring_l1   = float(pars.get("k_ring_l1",   1.0))
    mK_ring_l2   = float(pars.get("k_ring_l2",   1.0))
    mK_rest_l1   = float(pars.get("k_rest_l1",   1.0))
    mK_rest_l2   = float(pars.get("k_rest_l2",   1.0))

    mR_upland = float(pars.get("r_upland",1.0))
    mR_rest   = float(pars.get("r_rest",1.0))
    mFGW_upland = float(pars.get("fgw_upland",1.0))
    mFGW_rest = float(pars.get("fgw_rest",1.0))

    # --- load base arrays / masks ---
    rch = gwf.get_package("rch")
    spd_path = root / "rch_spd_base.npy"
    K_base = np.load(root / "K_base.npy")
    K_base = np.asarray(K_base).ravel()

    zone_ring   = np.asarray(np.load(root / "zone_ring.npy")).astype(bool).ravel()
    zone_upland = np.asarray(np.load(root / "zone_upland.npy")).astype(bool).ravel()
    zone_rest   = np.asarray(np.load(root / "zone_rest.npy")).astype(bool).ravel()

    if not (K_base.size == zone_ring.size == zone_upland.size == zone_rest.size):
        raise ValueError(
            f"Size mismatch: K_base({K_base.size}), ring({zone_ring.size}), upland({zone_upland.size}), rest({zone_rest.size}). "
            "These arrays must all be full-model length (nodes)."
        )

    n1 = int(gwf.modelgrid.ncpl[0])  # layer 1 node count (DISU/DISV convention)
    ntot = K_base.size
    if n1 <= 0 or n1 >= ntot:
        raise ValueError(f"Bad ncpl[0]={n1} vs total nodes={ntot}.")

    sl1 = slice(0, n1)
    sl2 = slice(n1, ntot)

    u1, u2 = zone_upland[sl1], zone_upland[sl2]
    r1, r2 = zone_ring[sl1],   zone_ring[sl2]
    z1, z2 = zone_rest[sl1],   zone_rest[sl2]

    K_new = K_base.copy()

    # Important: use indices so we modify the original array (boolean slicing on a slice returns a copy)

    K_new[np.flatnonzero(r1)] = mK_ring_l1
    K_new[n1 + np.flatnonzero(r2)] = mK_ring_l2

    K_new[np.flatnonzero(z1)] = mK_rest_l1
    K_new[n1 + np.flatnonzero(z2)] = mK_rest_l2

    npf = gwf.get_package("npf")
    if npf is None:
        raise RuntimeError("NPF package not found in loaded GWF model.")
    npf.k.set_data(K_new)

    # --- Recharge (top layer only in your setup) ---
 
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
        root / "sim_heads_1.dat",
        sep=" ",
        header=False,
        index=False,
        float_format="%.10f"
    )

def write_sim_sgd_segments_dat(
    root: Path,
    sgd_obs_csv: str = "sgd_obs.csv",
    node_to_seg_csv: str = "chd_node_to_seg.csv",
    ) -> None:
    """
    Compute segment-averaged SGD from CHD cell-by-cell flows and write sim_sgd.dat:
        sgd_### <m/d>
    Units:
      - CHD flow in budget is [L^3/T] (m3/s)
      - Convert to flux [L/T] by dividing by (segment_length_m * width_m)
      - Convert m/s -> m/d by *86400
    """
    sgd_path = root / sgd_obs_csv
    map_path = root / node_to_seg_csv
    if not sgd_path.exists() or not map_path.exists():
        # Optional output
        return RuntimeError(f"SGD observation or mapping file not found: {sgd_path}, {map_path}")

    obs = pd.read_csv(sgd_path)
    seg_col = "seg_id" if "seg_id" in obs.columns else obs.columns[0]
    len_col = "length_m" if "length_m" in obs.columns else None
    width_col = "width_m" if "width_m" in obs.columns else None

    mapping = pd.read_csv(map_path)
    if "node" not in mapping.columns or ("seg_id" not in mapping.columns and seg_col not in mapping.columns):
        raise ValueError("chd_node_to_seg.csv must have columns: node, seg_id")

    map_seg_col = "seg_id" if "seg_id" in mapping.columns else seg_col

    # Determine which segments we can compute (must appear in mapping)
    map_seg_ids = set(int(x) for x in mapping[map_seg_col].unique())
    seg_ids = [int(x) for x in obs[seg_col].tolist()]
    seg_ids_use = [sid for sid in seg_ids if sid in map_seg_ids]
    if len(seg_ids_use) == 0:
        raise RuntimeError("No seg_id in sgd_obs.csv matched seg_id in chd_node_to_seg.csv.")

    # node numbering: MF6 budget 'node' is 1-based. Shift mapping if needed.
    node_map = mapping[["node", map_seg_col]].copy()
    if node_map["node"].min() == 0:
        node_map["node"] = node_map["node"] + 1
    node_map["node"] = node_map["node"].astype(int)
    node_map[map_seg_col] = node_map[map_seg_col].astype(int)

    # build dict seg_id -> nodes
    seg_to_nodes = {}
    for sid, grp in node_map.groupby(map_seg_col):
        seg_to_nodes[int(sid)] = grp["node"].to_numpy(dtype=int)

    # read CHD flows from budget file
    bud_file = _find_budget_file(root)
    cbc = flopy.utils.CellBudgetFile(bud_file, precision="double")

    # record name that contains CHD; prefer exact match if possible
    recnames = [rn.decode().strip() if isinstance(rn, (bytes, bytearray)) else str(rn).strip() for rn in cbc.get_unique_record_names()]
    chd_rec = None
    for rn in recnames:
        if rn.upper() == "CHD":
            chd_rec = rn
            break
    if chd_rec is None:
        for rn in recnames:
            if "CHD" in rn.upper():
                chd_rec = rn
                break
    if chd_rec is None:
        raise RuntimeError(f"Could not find a CHD record in budget file. Records: {recnames}")

    times = cbc.get_times()
    totim = times[-1]
    rec = cbc.get_data(text=chd_rec, totim=totim)

    # rec is typically a list with one array of dtype (node, q)
    if isinstance(rec, list):
        rec = rec[0]
    node = rec["node"].astype(int)
    q = rec["q"].astype(float)

    # out of aquifer is negative q; take magnitude of discharge only
    discharge = np.where(q < 0.0, -q, 0.0)  # m3/s

    # map node->discharge for fast lookup
    # many nodes may have 0, but dict is fine at this size
    node_to_q = dict(zip(node.tolist(), discharge.tolist()))

    out = root / "sim_sgd.dat"
    with open(out, "w", encoding="utf-8") as f:
        norm_path = root / "sgd_norm_factor.txt"
        norm_factor = 1.0
        if norm_path.exists():
            norm_factor = float(norm_path.read_text().strip())
            if not (norm_factor > 0.0):
                raise ValueError(f"Bad norm_factor in {norm_path}: {norm_factor}")

        for sid in seg_ids_use:
            nodes = seg_to_nodes.get(int(sid), np.array([], dtype=int))
            qsum = float(sum(node_to_q.get(int(n), 0.0) for n in nodes))  # m3/s

            # segment length (m)
            if len_col is not None:
                seg_len = float(obs.loc[obs[seg_col].astype(int) == sid, len_col].iloc[0])
            else:
                seg_len = 1.0

            # representative cross-shore width (m): default 500 m if not supplied
            if width_col is not None:
                width_m = float(obs.loc[obs[seg_col].astype(int) == sid, width_col].iloc[0])
            else:
                width_m = 500.0

            area = max(seg_len * width_m, 1.0)  # m2
            flux_mps = qsum / area              # m/s
            flux_mpd = flux_mps * 86400.0       # m/d
            #f.write(f"sgd_{sid:03d} {flux_mpd:.6f}\n")
            f.write(f"sgd_{sid:03d} {flux_mpd / norm_factor:.6f}\n")


def find_mf6_exe(root: Path) -> str:
    for cand in ["mf6.exe", "mf6"]:
        if (root / cand).exists():
            return str(root / cand)
    return "mf6"

def find_headfile(root: Path) -> Path:
    cands = sorted(root.glob("*.hds"))
    if not cands:
        raise FileNotFoundError("No *.hds file found after MF6 run")
    return cands[0]

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


def plot_results(gwf, headFile_path: Path, title, layer, out_png, cbar_label):

    hf = HeadFile(str(headFile_path))
    totim = hf.get_times()[-1]
    h = hf.get_data(totim=totim)

    fig,ax = plt.subplots()
    pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
    hd = pmv.plot_array(h[0][0], ax=ax, alpha=1.0, cmap='viridis')
    contour = pmv.contour_array(h[0][0], ax=ax, levels=5, colors='black')
    #pmv.plot_grid(lw=0.05, alpha=0.15)
    plt.colorbar(hd, label=cbar_label)
    plt.clabel(contour, fmt="%1.0f")
    ax.set_title(title)
    ax.set_xlabel("Este [m]")
    ax.set_ylabel("Norte [m]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    plt.close(fig)

def main() -> None:
    root = Path(__file__).resolve().parent
    pars = read_params(root / "absolute_params1_caseB.csv")
    sim, gwf = load_sim(root)

    sim_ws = root  # / "model"
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), verbosity_level=0)
    gwf = sim.get_model()

    apply_parameters(gwf, pars, root)
    sim.write_simulation()
    run_mf6(root)
    #write_sim_budget_dat(root)

    # Write PEST outputs
    headfile = find_headfile(root)
    simdf = extract_sim_heads(headfile, root / "obs_cellids.csv")
    write_sim_heads_dat(root,simdf)
    write_sim_sgd_segments_dat(root)  # optional; does nothing if sgd files are missing
    for layer in (0,1):
        plot_results(gwf, headfile, f"Caso B - Calibrado\nMalla quadtree - Capa {int(layer)+1}", layer, root / f"quad_casoB_calibrado_layer{int(layer)+1}", "Cargas hidráulicas [m.s.n.m.]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)