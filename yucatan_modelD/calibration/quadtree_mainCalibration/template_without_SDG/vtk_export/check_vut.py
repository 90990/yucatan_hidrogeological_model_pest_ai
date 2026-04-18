import numpy as np
import meshio

path = "caseD_quadtree_heads_LAST.vtu"
m = meshio.read(path)

print(m)
print("points:", m.points.shape)

# chequeo NaN/Inf en coordenadas
print("coords finite:", np.isfinite(m.points).all())

# chequeo NaN/Inf en data
for k,v in (m.point_data or {}).items():
    arr = np.asarray(v)
    print("point_data", k, arr.shape, "finite:", np.isfinite(arr).all())

for k,v in (m.cell_data or {}).items():
    # cell_data puede ser lista por tipo de celda
    try:
        arr = np.asarray(v)
        print("cell_data", k, arr.shape, "finite:", np.isfinite(arr).all())
    except:
        print("cell_data", k, "non-array")
