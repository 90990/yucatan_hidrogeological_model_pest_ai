import numpy as np
import pandas as pd

try:
    import pyemu
except ImportError as e:
    raise SystemExit("pyemu is required bro!") from e

def find_jacobian():
    for fn in ["model.jcb", "model.jco", "model.jac", "model.jcb.bin"]:
        try:
            open(fn, "rb").close()
            return fn
        except Exception:
            pass
    # fallback : first *.jcb or *.jco in folder
    import glob
    cands = glob.glob("*.jcb") + glob.glob("*.jco")
    if not cands:
        raise FileNotFoundError("no jacobian file found :c")
    return cands[0]

pst = pyemu.Pst("model.pst")

jfn = find_jacobian()
jco = pyemu.Jco.from_binary(jfn)

# use only non-zero weight obs and adjustable parameters
obs_names = pst.nnz_obs_names
par_names = pst.adj_par_names

j = jco.get(obs_names, par_names).to_dataframe()

w = pst.observation_data.loc[obs_names, "weight"].astype(float).values
w = w.reshape(-1,1)

# Weighted Jacobian: rows scales by observation weights
jw = j.values * w

# ---- Composite sensitivity metrics ----
# Unscaled composite sensitivity (units: obs units / parameter units)
css = np.sqrt((jw**2).sum(axis=0))

# A dimensionless-ish scaling: multiply by |parval| (useful for multipliers)
parvals = pst.parameter_data.loc[par_names, "parval1"].astype(float).values
css_scaled = css * np.abs(parvals)

out = pd.DataFrame({
    "parnme":par_names,
    "parval1":parvals,
    "css":css,
    "css_scaled":css_scaled
}).sort_values("css_scaled", ascending=False)

out.to_csv("sensitivity_css.csv", index=False)

# Correlation (approx) from Finisher information
# F = J^T W^2 J
F = jw.T @ jw
cov = np.linalg.inv(F) # Pseudo-inverse for stability
d = np.sqrt(np.clip(np.diag(cov), 1e-30, np.inf))
corr = cov / np.outer(d,d)

corr_df = pd.DataFrame(corr, index=par_names, columns=par_names)
corr_df.to_csv("parameter_correlation.csv")

# Observation leverage:
# Hat matrix H = J (J^T J)^-1 J^T (on weighted system)
H = jw @ cov @ jw.T
lev = np.clip(np.diag(H), 0.0, 1.0)
lev_df = pd.DataFrame({"obsnme":obs_names, "leverage":lev})
lev_df.to_csv("observation_leverage.csv", index=False)

print("Wrote: sensitivity_css.csv, parameter_correlation.csv, observation_leverage.csv")
print("Top parameters by css_scaled:")
print(out.head(10).to_string(index=False))

