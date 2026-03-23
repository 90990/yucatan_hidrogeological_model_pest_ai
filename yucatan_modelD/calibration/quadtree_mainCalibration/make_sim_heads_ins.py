import pandas as pd

obs = pd.read_csv("obs_heads.csv")
with open("sim_heads.ins", "w", encoding="utf-8") as f:
    f.write("pif ~\n")
    for oid in obs["obs_id"].astype(str):
        f.write(f"l1 w !{oid}!\n")

print("Wrote sim_heads.ins")