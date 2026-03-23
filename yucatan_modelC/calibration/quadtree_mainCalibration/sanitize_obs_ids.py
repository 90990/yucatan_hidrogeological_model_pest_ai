import pandas as pd
import re

def make_safe_ids(series):
    safe = []
    for s in series.astype(str):
        s = s.strip().strip('"').strip("'") # remove literal quote chars
        s = re.sub(r"[^A-Za-z0-9_]+", "_", s)  #replacing bad charts with _
        if not re.match(r"^[A-Za-z]", s):
            s = "w_" + s
        safe.append(s[:20])
    return safe


obs = pd.read_csv("obs_heads.csv")

#keeping original well id for the thesis reporting
obs["obs_id_raw"] = obs["obs_id"].astype(str)

# making safe obs ids
obs["obs_id"] = make_safe_ids(obs["obs_id_raw"])

# writting mapping (important for the report)
obs[["obs_id", "obs_id_raw"]].to_csv("obs_id_map.csv", index=False)

# overwritting obs_heads.csv with safe obs_id
obs.drop(columns=["obs_id_raw"]).to_csv("obs_heads.csv", index=False)

print("Wrote obs_id_map.csv and updated obs_heads.csv with PEST-safe obs_id.")
