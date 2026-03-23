\
# build_pest_clean.py
# Build a PEST++-GLM setup from an existing "template/" directory.
#
# This version additionally supports SGD/CHD segment observations:
#   - expects sgd_obs.csv (seg_id, obsval, weight, length_m[, width_m])
#   - expects chd_node_to_seg.csv (node, seg_id)
#   - forward_run.py will write sim_sgd.dat; sim_sgd.ins is generated here.

from __future__ import annotations

import os
import shutil
from pathlib import Path
import pandas as pd
import pyemu


def _col_pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of columns {candidates} found. Available: {list(df.columns)}")


def _ensure_params_files(pest_dir: Path, par_names: list[str]) -> None:
    """Ensure params.csv and params.csv.tpl exist and are headered (parnme, parval1)."""
    params_csv = pest_dir / "params_1.csv"
    params_tpl = pest_dir / "params_1.csv.tpl"

    if not params_csv.exists():
        pd.DataFrame({"parnme": par_names, "parval1": [1.0] * len(par_names)}).to_csv(params_csv, index=False)

    if not params_tpl.exists():
        # Template that preserves the header row
        with open(params_tpl, "w", encoding="utf-8") as f:
            f.write("ptf ~\n")
            f.write("parnme,parval1\n")
            for p in par_names:
                # second column is replaced by PEST
                f.write(f"{p},~{p}~\n")


def _build_sim_sgd_ins_and_seed(
    pest_dir: Path,
    sgd_obs_path: Path,
    chd_node_to_seg_path: Path | None,
    ) -> list[int]:
    """
    Create sim_sgd.ins and seed sim_sgd.dat.
    Returns seg_ids used (ints). If mapping is provided, only seg_ids present in mapping are used.
    """
    sgd = pd.read_csv(sgd_obs_path)
    seg_col = _col_pick(sgd, ["seg_id", "segid", "segment", "segment_id"])
    seg_ids = [int(x) for x in sgd[seg_col].tolist()]

    if chd_node_to_seg_path is not None and chd_node_to_seg_path.exists():
        m = pd.read_csv(chd_node_to_seg_path)
        m_seg_col = _col_pick(m, ["seg_id", "segid", "segment", "segment_id"])
        map_seg_ids = set(int(x) for x in m[m_seg_col].unique())
        seg_ids_use = [sid for sid in seg_ids if sid in map_seg_ids]
    else:
        seg_ids_use = seg_ids

    if len(seg_ids_use) == 0:
        raise RuntimeError(
            "No seg_id in sgd_obs.csv matched any seg_id in chd_node_to_seg.csv. "
            "Fix your seg_id numbering or mapping."
        )

    # Instruction file reads second token in each line: 'sgd_### <value>'
    ins_path = pest_dir / "sim_sgd.ins"
    with open(ins_path, "w", encoding="utf-8") as f:
        f.write("pif ~\n")
        for sid in seg_ids_use:
            f.write(f"l1 w !sgd_{sid:03d}!\n")

    # Seed output file
    dat_path = pest_dir / "sim_sgd.dat"
    if not dat_path.exists():
        with open(dat_path, "w", encoding="utf-8") as f:
            for sid in seg_ids_use:
                f.write(f"sgd_{sid:03d} 0.0\n")

    return seg_ids_use


def main() -> None:
    root = Path(__file__).resolve().parent
    template_dir = root / "template_1"
    pest_dir = root / "pest_1"

    if not template_dir.exists():
        raise FileNotFoundError(f"Template folder not found: {template_dir}")

    if pest_dir.exists():
        shutil.rmtree(pest_dir)
    shutil.copytree(template_dir, pest_dir)

    # ensuring the forward run script is in pest folder
    shutil.copy2(root / "forward_run.py", pest_dir / "forward_run.py")
    for fn in ["sgd_obs.csv", "chd_node_to_seg.csv"]:
        src = root / fn
    if src.exists():
        shutil.copy2(src, pest_dir / fn)

    # Copy auxiliary CSVs (if present at project root) into pest dir
    for fn in ["sgd_obs.csv", "chd_node_to_seg.csv", "obs_heads.csv"]:
        src = root / fn
        if src.exists():
            shutil.copy2(src, pest_dir / fn)

    # --- parameter names (must match forward_run.py read_params / apply_parameters) ---
    par_names = [
        "mk_ring_l1", "mk_ring_l2",
        "mk_rest_l1", "mk_rest_l2",
        "mr_upland", "mr_rest",
        "mFGW_upland", "mFGW_rest",
    ]
    _ensure_params_files(pest_dir, par_names)

    # --- SGD instruction + seed, if sgd_obs exists ---
    sgd_obs_path = pest_dir / "sgd_obs.csv"
    chd_node_to_seg_path = pest_dir / "chd_node_to_seg.csv"
    seg_ids_use: list[int] = []
    if sgd_obs_path.exists():
        seg_ids_use = _build_sim_sgd_ins_and_seed(pest_dir, sgd_obs_path, chd_node_to_seg_path)

    # --- Ensure sim_heads.dat exists (seed) ---
    heads_dat = pest_dir / "sim_heads_1.dat"
    if not heads_dat.exists():
        # seed with obs ids from obs_heads.csv if available
        obs_path = pest_dir / "obs_heads.csv"
        if not obs_path.exists():
            raise FileNotFoundError(
                f"{heads_dat} missing and {obs_path} not found to seed it. "
                "Run prep_template_clean.py first (or copy obs_heads.csv into the project root)."
            )
        obs = pd.read_csv(obs_path)
        oid_col = _col_pick(obs, ["obs_id", "obsnme", "oid"])
        with open(heads_dat, "w", encoding="utf-8") as f:
            for oid in obs[oid_col].tolist():
                f.write(f"{str(oid)} 0.0\n")

    # --- Build PST (IMPORTANT: do this INSIDE pest_dir so the control file has NO absolute paths) ---
    tpl_files = ["params_1.csv.tpl"]
    in_files = ["params_1.csv"]
    ins_files = ["sim_heads_1.ins"]
    out_files = ["sim_heads_1.dat"]
    if (pest_dir / "sim_sgd.ins").exists() and (pest_dir / "sim_sgd.dat").exists():
        ins_files.append("sim_sgd.ins")
        out_files.append("sim_sgd.dat")

    cwd = os.getcwd()
    os.chdir(pest_dir)
    try:
        pst = pyemu.Pst.from_io_files(
            tpl_files=tpl_files, #params.csv.tpl
            in_files=in_files, # params.csv
            ins_files=ins_files, # sim_heads.ins
            out_files=out_files, # sim_heads.dat
            pst_filename="model.pst", 
        )
    finally:
        os.chdir(cwd)

    # --- assign observed heads + weights ---
    obs_path = pest_dir / "obs_heads.csv"
    if not obs_path.exists():
        raise FileNotFoundError(f"obs_heads.csv not found in {pest_dir}")

    obs = pd.read_csv(obs_path)
    oid_col = _col_pick(obs, ["obs_id", "obsnme", "oid"])
    val_col = _col_pick(obs, ["head_obs", "obsval", "head", "h_obs", "value"])
    w_col = "weight" if "weight" in obs.columns else None

    pst_obs_map = {o.lower(): o for o in pst.observation_data.index}
    n_set, n_miss = 0, 0
    for _, r in obs.iterrows():
        key = str(r[oid_col]).strip().lower()
        if key in pst_obs_map:
            true = pst_obs_map[key]
            pst.observation_data.loc[true, "obsval"] = float(r[val_col])
            pst.observation_data.loc[true, "weight"] = float(r[w_col]) if w_col else 1.0
            pst.observation_data.loc[true, "obgnme"] = "head"
            n_set += 1
        else:
            n_miss += 1
    print(f"Head obs matched: {n_set} / {len(obs)} (missing in pst: {n_miss})")
    if n_set < len(obs):
        raise RuntimeError("Not all head obs_ids were found in pst.observation_data.index. Fix sim_heads.ins naming.")

    # --- assign SGD obs + weights (optional) ---
    if sgd_obs_path.exists() and len(seg_ids_use) > 0:
        sgd = pd.read_csv(sgd_obs_path)
        seg_col = _col_pick(sgd, ["seg_id", "segid", "segment", "segment_id"])
        obsval_col = _col_pick(sgd, ["obsval", "sgd_md", "value"])
        weight_col = "weight" if "weight" in sgd.columns else None

        sgd_use = sgd.loc[sgd[seg_col].astype(int).isin(seg_ids_use)].copy()

        if sgd_use.empty:
            raise RuntimeError("No SGD segments matched seg_ids_use; cannot normalize.")

        norm_factor = float(sgd_use[obsval_col].astype(float).mean())
        if not (norm_factor > 0.0):
            raise RuntimeError(f"Bad SGD norm_factor={norm_factor}. Check obs values/units.")

        (pest_dir / "sgd_norm_factor.txt").write_text(f"{norm_factor:.10e}\n", encoding="utf-8")

        for sid in seg_ids_use:
            obsnme = f"sgd_{sid:03d}"
            if obsnme not in pst.observation_data.index:
                raise RuntimeError(f"SGD obsnme {obsnme} not found in pst. Check sim_sgd.ins.")
            row = sgd.loc[sgd[seg_col].astype(int) == sid]
            if row.empty:
                continue
            #pst.observation_data.loc[obsnme, "obsval"] = float(row.iloc[0][obsval_col])
            obsval = float(row.iloc[0][obsval_col])
            pst.observation_data.loc[obsnme, "obsval"] = obsval / norm_factor
            
            pst.observation_data.loc[obsnme, "weight"] = float(row.iloc[0][weight_col]) if weight_col else 1.0
            pst.observation_data.loc[obsnme, "obgnme"] = "sgd"

    # --- parameter transforms and bounds (multipliers) ---
    bounds = {
        "mk_ring_l1":   (1/9,  2.2222222222),         # [1e-2,2e-1] / 9e-2
        "mk_rest_l1":   (1/9,  2.2222222222),         # [1e-3,2e-2] / 9e-3

        "mk_ring_l2":   (1/9,  1.1111111111),         # [1e-1,1.0]  / 9e-1
        "mk_rest_l2":   (1/9,  2.2222222222),         # [1e-2,2e-1] / 9e-2

        "mr_upland": (0.05/0.25, 0.45/0.25),  # (0.20, 1.80)
        "mr_rest":   (0.05/0.25, 0.45/0.25),  # (0.20, 1.80)
        
        "mFGW_upland": (0.05/0.30 , 0.50/0.30),
        "mFGW_rest": (0.05/0.30 , 0.50/0.30),
    }

    bounds = {k.lower(): v for k,v in bounds.items()}

    # setting parameters bounds + transforms:
    for p in pst.parameter_data.index:
        key = p.lower()
        if key not in bounds:
            raise KeyError(f"Missing bounds for parameter '{key}' (add it to bounds dict)")
        lb, ub = bounds[key]
        pst.parameter_data.loc[p, "partrans"] = "log"
        pst.parameter_data.loc[p, "parlbnd"] = lb
        pst.parameter_data.loc[p, "parubnd"] = ub

    # Control settings (edit as desired)
    pst.control_data.noptmax = 10

    pst.write(pest_dir / "model.pst")
    print("Built pest setup at:", pest_dir)


if __name__ == "__main__":
    main()
