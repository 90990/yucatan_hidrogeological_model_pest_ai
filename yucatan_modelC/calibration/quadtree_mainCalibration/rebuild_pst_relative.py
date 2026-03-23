# rebuild_pst_relative.py
from pathlib import Path
import os
import pandas as pd
import pyemu

def main():
    root = Path(__file__).resolve().parent
    pest_dir = root / "pest"

    if not pest_dir.exists():
        raise FileNotFoundError("pest folder not found. Run your build step first.")

    # IMPORTANT: build from inside the pest folder so paths are relative
    os.chdir(pest_dir)

    pst = pyemu.Pst.from_io_files(
        tpl_files=["params.csv.tpl"],
        in_files=["params.csv"],
        ins_files=["sim_heads.ins"],
        out_files=["sim_heads.dat"],
        pst_filename="model.pst",
    )

    pst.model_command = ["python forward_run.py"]
    pst.control_data.noptmax = 10

    # inject observed targets
    obs = pd.read_csv("obs_heads.csv")
    obs["obs_id"] = obs["obs_id"].astype(str)

    for _, r in obs.iterrows():
        oid = r["obs_id"]
        if oid in pst.observation_data.index:
            pst.observation_data.loc[oid, "obsval"] = float(r["head_obs"])
            pst.observation_data.loc[oid, "weight"] = float(r.get("weight", 1.0))
            pst.observation_data.loc[oid, "obgnme"] = "head"

    # parameter bounds/transforms
    par = pst.parameter_data
    par.loc[:, "partrans"] = "log"
    par.loc[:, "parchglim"] = "factor"
    par.loc[:, "parlbnd"] = 0.2
    par.loc[:, "parubnd"] = 5.0
    if "mR" in par.index:
        par.loc["mR", "parlbnd"] = 0.1
        par.loc["mR", "parubnd"] = 10.0

    # finite-diff step size (helps jacobian)
    pg = pst.parameter_groups
    pg.loc[:, "derinc"] = 0.1

    pst.write("model.pst")
    print("Rewrote pest/model.pst with relative file paths.")

if __name__ == "__main__":
    main()
