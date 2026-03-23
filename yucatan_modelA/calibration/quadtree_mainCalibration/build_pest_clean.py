
##### --------------------------------------------- NEW 'build_pest_clean.py' SCRIPT FILE
from pathlib import Path
import shutil
import pandas as pd
import pyemu


def main():
    root = Path(__file__).resolve().parent
    template_dir = root / "template"
    pest_dir = root / "pest"

    if pest_dir.exists():
        shutil.rmtree(pest_dir)
    shutil.copytree(template_dir, pest_dir)

    # ensuring the forward run script is in pest folder
    shutil.copy2(root / "forward_run.py", pest_dir / "forward_run.py")

    # ensuring sim_heads.ins exists in pest folder
    if not (pest_dir / "sim_heads.dat").exists():
        obs = pd.read_csv(pest_dir / "obs_heads.csv")
        with open(pest_dir / "sim_heads.dat", "w", encoding='utf-8') as f:
            for oid in obs["obs_id"].astype(str):
                f.write(f"{oid} 0.0\n")
    
    # ensuring the params.csv.tpl file exists. If not, build from params.csv
    # assuming params.csv is two columns: parname, value (no header) like the Pstfrom log indicates
    tpl = pest_dir / "params.csv.tpl"
    if not tpl.exists():
        dfp = pd.read_csv(pest_dir / "params.csv", header=None)
        parnames = dfp.iloc[:,0].astype(str).tolist()
        with open(tpl, "w", encoding="utf-8") as f:
            f.write("ptf ~\n")
            for p in parnames:
                f.write(f"{p},~{p}~\n")
    
    pst = pyemu.Pst.from_io_files(
        tpl_files = [str(pest_dir / "params.csv.tpl")],
        in_files = [str(pest_dir / "params.csv")],
        ins_files = [str(pest_dir / "sim_heads.ins")],
        out_files = [str(pest_dir / "sim_heads.dat")],
        pst_filename = str(pest_dir / "model.pst")
    )

    # Set the commands
    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 10

    # Load TRUE observed values + weights
    # obs = pd.read_csv(pest_dir / "obs_heads.csv")
    # obs["obs_id"] = obs["obs_id"].astype(str)

    # for _, r in obs.iterrows():
    #     oid = r["obs_id"]
    #     if oid in pst.observation_data.index:
    #         pst.observation_data.loc[oid, "obsval"] = float(r["head_obs"])
    #         pst.observation_data.loc[oid, "weight"] = float(r.get("weight", 1.0))

    obs = pd.read_csv(pest_dir / "obs_heads.csv")

    # map pst obs names by lowercase
    pst_obs_map = {str(n).strip().lower(): n for n in pst.observation_data.index}

    n_set = 0
    n_miss = 0

    for _, r in obs.iterrows():
        key = str(r["obs_id"]).strip().lower()
        if key in pst_obs_map:
            true_name = pst_obs_map[key]
            pst.observation_data.loc[true_name, "obsval"] = float(r["head_obs"])
            pst.observation_data.loc[true_name, "weight"] = float(r.get("weight", 1.0))
            pst.observation_data.loc[true_name, "obgnme"] = "head"
            n_set += 1
        else:
            n_miss += 1

    print(f"Obs matched: {n_set} / {len(obs)}   missing in pst: {n_miss}")

    # HARD STOP if you didn't match everything
    if n_set < len(obs):
        raise RuntimeError("Not all obs_ids were found in pst.observation_data.index. Fix naming before running PEST++.")

    bounds = {

    "mk_rest_l1":   (1/9,  2.2222222222), #[1e-3,2e-2] / 9e-3
    "mk_rest_l2":   (1/9,  2.2222222222), #[1e-2, 2e-1] / 9e-2

    "mr_rest":   (0.05/0.25, 0.4/0.2), 
    "mr_upland": (0.05/0.25, 0.4/0.2),  

    "mfgw": (0.05/0.30 , 0.50/0.30),
    }

    bounds = {k.lower(): v for k,v in bounds.items()}

    # setting parameters bounds + transforms:
    for p in pst.parameter_data.index:
        key = p.lower()
        if key not in bounds:
            raise KeyError(f"Missing bounds for parameter '{p}' (add it to bounds dict)")
        lb, ub = bounds[key]
        pst.parameter_data.loc[p, "partrans"] = "log"
        pst.parameter_data.loc[p, "parlbnd"] = lb
        pst.parameter_data.loc[p, "parubnd"] = ub
    
    # pst.pestpp_options["phiredstp"] = 0.001
    # pst.pestpp_options["nphistp"] = 5

    pst.write(pest_dir / "model.pst")
    print("Built pest setup at:", pest_dir)


if __name__ == "__main__":
    main()
