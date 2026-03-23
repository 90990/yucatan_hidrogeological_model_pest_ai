import re
from pathlib import Path

def get_balance_pct_from_lst(lst_path: Path):
    txt = lst_path.read_text(errors="ignore")
    # captura varios posibles nombres
    pat = re.compile(r"(PERCENT\s+(?:DISCREPANCY|DIFFERENCE|ERROR).{0,20}?)([-+0-9\.Ee]+)")
    vals = []
    for m in pat.finditer(txt):
        try:
            vals.append(float(m.group(2)))
        except:
            pass
    if not vals:
        raise RuntimeError(f"No encontré PERCENT DISCREPANCY/DIFFERENCE/ERROR en {lst_path}")
    # devuelve el máximo absoluto (criterio conservador)
    return max(vals, key=lambda v: abs(v))


lst = Path(r"C:\\Users\\sebas\\Documents\\AYUDANTE SNI\\TESIS\\yucatan_modelA\\quadtreeGrid\\gridgen_disu\\mfsim.lst")  # o gwf_*.lst, o el nombre que tengas
print("Balance error (%):", get_balance_pct_from_lst(lst))
