import pandas as pd
import pyemu

pst = pyemu.Pst("model.pst")
od = pst.observation_data.copy()

print("nobs in pst:", pst.nobs)
print("npar adj:", pst.npar_adj)
print("weight stats:", od.weight.describe())
print("weights == 0:", int((od.weight == 0).sum()))
print("weights NaN:", int(od.weight.isna().sum()))
print("unique obgnme (first 10):", od.obgnme.unique()[:10])

# read sim output produced by forward run
sim = pd.read_csv("sim_heads.dat", sep=r"\s+", header=None, names=["obs_id", "head_sim"])
sim["obs_id"] = sim["obs_id"].astype(str).str.strip().str.strip('"').str.strip("'")
sim = sim.set_index("obs_id")

# read intended observed targets
obs = pd.read_csv("obs_heads.csv")
obs["obs_id"] = obs["obs_id"].astype(str).str.strip().str.strip('"').str.strip("'")
obs = obs.set_index("obs_id")

# compare names
pst_ids = set(od.index.astype(str))
sim_ids = set(sim.index.astype(str))
obs_ids = set(obs.index.astype(str))

print("\n--- ID matching ---")
print("obs_heads.csv not in pst:", len(obs_ids - pst_ids))
print("pst not in obs_heads.csv:", len(pst_ids - obs_ids))
print("sim_heads.dat not in pst:", len(sim_ids - pst_ids))
print("pst not in sim_heads.dat:", len(pst_ids - sim_ids))

# compare values where keys overlap
common = list(pst_ids & sim_ids & obs_ids)
common = common[: min(200, len(common))]

tmp = pd.DataFrame(index=common)
tmp["pst_obsval"] = od.loc[common, "obsval"].astype(float).values
tmp["pst_weight"] = od.loc[common, "weight"].astype(float).values
tmp["sim_head"] = sim.loc[common, "head_sim"].astype(float).values
tmp["obs_head"] = obs.loc[common, "head_obs"].astype(float).values

tmp["absdiff_pst_vs_sim"] = (tmp["pst_obsval"] - tmp["sim_head"]).abs()
tmp["absdiff_pst_vs_obs"] = (tmp["pst_obsval"] - tmp["obs_head"]).abs()
tmp["absdiff_sim_vs_obs"] = (tmp["sim_head"] - tmp["obs_head"]).abs()

print("\n--- Value checks (first 10) ---")
print(tmp[["pst_obsval","sim_head","obs_head","pst_weight",
           "absdiff_pst_vs_sim","absdiff_pst_vs_obs","absdiff_sim_vs_obs"]].head(10))

print("\nMax |pst_obsval - sim|:", tmp["absdiff_pst_vs_sim"].max())
print("Max |pst_obsval - obs|:", tmp["absdiff_pst_vs_obs"].max())
print("Max |sim - obs|:", tmp["absdiff_sim_vs_obs"].max())
