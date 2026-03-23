# css_by_group.py
import os
import numpy as np
import pandas as pd

def load_jco():
    # PEST++ suele dejar model.jcb (binario). A veces hay .jco.
    if os.path.exists("model.jcb"):
        import pyemu
        return pyemu.Jco.from_binary("model.jcb")
    elif os.path.exists("model.jco"):
        import pyemu
        return pyemu.Jco.from_ascii("model.jco")
    else:
        raise FileNotFoundError("No encuentro model.jcb ni model.jco en este folder.")

def css_from_jco(jco_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    # jco_df: (nobs x npar) dataframe
    # weights: length nobs
    w = weights.values.astype(float)
    JW = jco_df.values * w[:, None]
    css = np.sqrt(np.sum(JW * JW, axis=0))
    return pd.Series(css, index=jco_df.columns)

def main():
    import pyemu

    pst = pyemu.Pst("model.pst")
    jco = load_jco()

    # Observaciones no-cero peso
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.weight.astype(float) != 0.0].copy()

    # Parámetros ajustables
    par = pst.parameter_data.copy()
    adj = pst.adj_par_names
    par = par.loc[adj].copy()

    # Jacobiano como DataFrame y recortado
    jdf = jco.to_dataframe()
    jdf = jdf.loc[obs.index, adj]

    # CSS total
    css_total = css_from_jco(jdf, obs.weight.astype(float))

    # CSS por grupos (si no existe, se queda en 0)
    out = pd.DataFrame({"parnme": adj})
    out["css_total"] = css_total.reindex(adj).values

    for gname in ["head", "sgd"]:
        og = obs.loc[obs.obgnme == gname]
        if og.shape[0] == 0:
            out[f"css_{gname}"] = 0.0
            continue
        jg = jdf.loc[og.index, :]
        out[f"css_{gname}"] = css_from_jco(jg, og.weight.astype(float)).reindex(adj).values

    # (Opcional) CSS escalado por parval1 (útil si partrans=log)
    parval = par.parval1.astype(float).reindex(adj).values
    out["css_total_scaled"] = out["css_total"].values * parval
    out["css_head_scaled"]  = out["css_head"].values * parval
    out["css_sgd_scaled"]   = out["css_sgd"].values * parval

    # Rankings
    out["rank_total"] = out["css_total"].rank(ascending=False, method="min").astype(int)
    out["rank_head"]  = out["css_head"].rank(ascending=False, method="min").astype(int)
    out["rank_sgd"]   = out["css_sgd"].rank(ascending=False, method="min").astype(int)

    out = out.sort_values("css_total", ascending=False)

    out.to_csv("css_by_group.csv", index=False)
    print("Wrote: css_by_group.csv")
    print("\nTop 10 por CSS_total:")
    print(out[["parnme","css_total","css_head","css_sgd","rank_total","rank_head","rank_sgd"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
