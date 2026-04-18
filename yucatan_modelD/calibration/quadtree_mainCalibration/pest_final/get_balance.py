import re
from pathlib import Path

PCT_PATS = [
    r"PERCENT\s+DISCREPANCY\s*=?\s*([-+0-9\.Ee]+)",
    r"PERCENT\s+DIFFERENCE\s*=?\s*([-+0-9\.Ee]+)",
    r"PERCENT\s+ERROR\s*=?\s*([-+0-9\.Ee]+)",
]

TOTIN_PATS  = [r"TOTAL\s+IN\s*=?\s*([-+0-9\.Ee]+)"]
TOTOUT_PATS = [r"TOTAL\s+OUT\s*=?\s*([-+0-9\.Ee]+)"]

def _find_all_floats(txt, pats):
    vals = []
    for pat in pats:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            try:
                vals.append(float(m.group(1)))
            except:
                pass
    return vals

def get_balance_pct_from_lst(lst_path: Path):
    txt = lst_path.read_text(errors="ignore")

    # 1) intenta extraer directamente el porcentaje si existe
    pct_vals = _find_all_floats(txt, PCT_PATS)
    if pct_vals:
        # criterio conservador: máximo absoluto
        return max(pct_vals, key=lambda v: abs(v))

    # 2) si no hay porcentaje, calcula con TOTAL IN/OUT
    tot_in  = _find_all_floats(txt, TOTIN_PATS)
    tot_out = _find_all_floats(txt, TOTOUT_PATS)

    if not tot_in or not tot_out:
        raise RuntimeError(
            f"No encontré ni PERCENT (discrepancy/difference/error) ni TOTAL IN/OUT en {lst_path.name}. "
            "Este archivo probablemente NO es el listing del GWF."
        )

    # Empareja por orden de aparición (timestep/stress period)
    n = min(len(tot_in), len(tot_out))
    pct_calc = []
    for i in range(n):
        I = tot_in[i]
        O = tot_out[i]
        denom = (abs(I) + abs(O)) / 2.0
        if denom <= 0.0:
            continue
        pct = 100.0 * (I - O) / denom   # percent difference clásico
        pct_calc.append(pct)

    if not pct_calc:
        raise RuntimeError(f"No pude calcular porcentaje desde TOTAL IN/OUT en {lst_path.name}.")

    return max(pct_calc, key=lambda v: abs(v))

def find_gwf_listing(run_dir: Path):
    cands = []
    for p in run_dir.glob("*.lst"):
        txt = p.read_text(errors="ignore")
        if re.search(r"VOLUME\s+BUDGET|BUDGET\s+FOR\s+ENTIRE\s+MODEL|TOTAL\s+IN", txt, flags=re.IGNORECASE):
            cands.append(p)
    return cands

if __name__ == "__main__":
    run_dir = Path(".")
    cands = find_gwf_listing(run_dir)
    if not cands:
        raise SystemExit("No encontré ningún .lst con VOLUME BUDGET/TOTAL IN. Estás en la carpeta correcta?")
    print("Candidatos:")
    for c in cands:
        print(" -", c.name)

    # toma el primero (o cambia aquí al que tú quieras)
    lst = cands[0]
    pct = get_balance_pct_from_lst(lst)
    print(f"\nArchivo usado: {lst.name}")
    print(f"Error balance (%): {pct:.6g}")
