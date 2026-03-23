from pathlib import Path
import numpy as np
import pandas as pd

SIM_WS = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\calibration\quadtree_mainCalibration\final_calibration")   # must contain sim_heads.dat
OBS_CSV = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\calibration\quadtree_mainCalibration\final_calibration\obs_heads.csv")

OBS_COL = "head_obs"   # edit if needed

sim = pd.read_csv(SIM_WS / "sim_heads.dat", sep=r"\s+", header=None, names=["obs_id","head_sim"])
obs = pd.read_csv(OBS_CSV)

sim["obs_id"] = sim["obs_id"].astype(str)
obs["obs_id"] = obs["obs_id"].astype(str)

df = sim.merge(obs[["obs_id", OBS_COL]], on="obs_id", how="left")
if df[OBS_COL].isna().any():
    raise ValueError("Some obs_id not found in obs file.")

y_sim = df["head_sim"].to_numpy(float)
y_obs = df[OBS_COL].to_numpy(float)

bad = (~np.isfinite(y_sim)) | (np.abs(y_sim) > 1e20)
good = ~bad

rmse = float(np.sqrt(np.mean((y_sim[good] - y_obs[good])**2)))
print("Valid obs used:", int(good.sum()), "/", len(good))
print("RMSE:", rmse)
print("Invalid obs ids:", df.loc[bad, "obs_id"].tolist())
