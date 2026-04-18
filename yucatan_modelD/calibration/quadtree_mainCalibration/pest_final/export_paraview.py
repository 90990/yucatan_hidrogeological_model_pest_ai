from pathlib import Path
import numpy as np
import flopy
from flopy.export.vtk import Vtk

# ==========
# AJUSTA AQUÍ
# ==========
sim_ws = Path(r"C:\Users\sebas\Documents\AYUDANTE SNI\TESIS\yucatan_modelD\quadtreeGrid\gridgen_disu")  # carpeta con mfsim.nam
outdir = sim_ws / "vtk_export"
outdir.mkdir(exist_ok=True)

# 1) Cargar simulación MF6
sim = flopy.mf6.MFSimulation.load(sim_name="mfsim.nam", sim_ws=str(sim_ws), verbosity_level=0)

# 2) Tomar el primer modelo GWF (o especifica por nombre si tienes más de uno)
gwf_names = sim.model_names
if len(gwf_names) == 0:
    raise RuntimeError("No se encontraron modelos dentro de la simulación.")
gwf = sim.get_model(gwf_names[0])

# 3) Leer heads (último registro)
hobj = gwf.output.head()  # HeadFile-like
kstpkper_list = hobj.get_kstpkper()
kstpkper_last = kstpkper_list[-1]  # último
head = hobj.get_data(kstpkper=kstpkper_last)

# 4) Exportar a VTU
vtk = Vtk(model=gwf, xml=True, binary=True)

# a) heads como arreglo principal
vtk.add_array(head, name="head")

# b) (opcional) idomain para filtrar celdas inactivas en ParaView
#    si no existe, no pasa nada; exportas todo.
try:
    idomain = gwf.disu.idomain.array
    vtk.add_array(idomain, name="idomain")
except Exception:
    pass

# c) (opcional) capa/lay para colorear por layer en ParaView
try:
    nlay = gwf.modelgrid.nlay
    ncpl = gwf.modelgrid.ncpl
    lay_id = np.vstack([np.full(ncpl, k+1) for k in range(nlay)])  # 1..nlay
    vtk.add_array(lay_id, name="layer")
except Exception:
    pass

# 5) Escribir archivo
vtu_path = outdir / "caseD_quadtree_heads_LAST.vtu"
vtk.write(str(vtu_path))

print(f"Listo. Archivo VTU: {vtu_path}")
print(f"Se exportó kstpkper={kstpkper_last} (último registro).")
