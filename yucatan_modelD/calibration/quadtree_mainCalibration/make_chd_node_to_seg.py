from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import flopy


def _extract_chd_nodes(gwf, kper: int = 0) -> np.ndarray:
    chd = gwf.get_package("chd")
    if chd is None:
        raise RuntimeError("CHD package not found in this model.")

    spd = chd.stress_period_data.get_data(kper)
    if spd is None or len(spd) == 0:
        raise RuntimeError(f"CHD stress_period_data is empty for kper={kper}.")

    if spd.dtype.names and "cellid" in spd.dtype.names:
        cellids = spd["cellid"]
    else:
        # fallback: assume first column is cellid
        cellids = spd[:, 0]

    nodes = []
    for cid in cellids:
        # DISU usually: cid is int (node number)
        if isinstance(cid, (int, np.integer)):
            nodes.append(int(cid))
        else:
            # sometimes cellid might be tuple/list (k,i,j). This is a fallback.
            # For DISU you should not hit this.
            try:
                nodes.append(int(cid[0]))
            except Exception as e:
                raise RuntimeError(f"Unsupported cellid format: {cid} ({type(cid)})") from e

    return np.array(nodes, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_ws", required=True,
                    help="Path to model workspace to load (use your template folder).")
    ap.add_argument("--gpkg", required=True,
                    help="Path to GeoPackage containing segments (or survey points) with seg_id.")
    ap.add_argument("--layer", default=None,
                    help="Layer name inside gpkg (optional). If omitted, first layer is used.")
    ap.add_argument("--seg_id_field", default="seg_id",
                    help="Field name holding stable IDs (default: seg_id).")
    ap.add_argument("--kper", type=int, default=0,
                    help="Stress period to read CHD from (default: 0).")
    ap.add_argument("--max_dist_m", type=float, default=None,
                    help="Optional max distance (m). Rows farther than this are dropped.")
    ap.add_argument("--out_csv", default="chd_node_to_seg.csv",
                    help="Output CSV name (default: chd_node_to_seg.csv).")

    args = ap.parse_args()

    sim_ws = Path(args.sim_ws).resolve()
    gpkg = Path(args.gpkg).resolve()
    out_csv = Path(args.out_csv).resolve()

    if not sim_ws.exists():
        raise FileNotFoundError(sim_ws)
    if not gpkg.exists():
        raise FileNotFoundError(gpkg)

    # Load MF6
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), verbosity_level=0)
    gwf = sim.get_model()

    mg = gwf.modelgrid
    if not hasattr(mg, "xcellcenters") or not hasattr(mg, "ycellcenters"):
        raise RuntimeError("Modelgrid does not expose xcellcenters/ycellcenters as 1D arrays (expected for DISU).")

    # CHD nodes
    nodes = _extract_chd_nodes(gwf, kper=args.kper)

    # Build CHD node point layer
    xs = np.asarray(mg.xcellcenters).ravel()
    ys = np.asarray(mg.ycellcenters).ravel()

    if nodes.max() >= xs.size:
        raise RuntimeError(
            f"Max CHD node id {nodes.max()} >= xcellcenters size {xs.size}. "
            "This indicates a mismatch between CHD node ids and modelgrid indexing."
        )

    pts = gpd.GeoDataFrame(
        {"node": nodes},
        geometry=[Point(xs[n], ys[n]) for n in nodes],
        crs=None
    )

    # Load segments/points with seg_id
    if args.layer:
        segs = gpd.read_file(gpkg, layer=args.layer)
    else:
        segs = gpd.read_file(gpkg)

    if args.seg_id_field not in segs.columns:
        raise RuntimeError(
            f"'{args.seg_id_field}' not found in gpkg columns. Found: {list(segs.columns)}"
        )

    segs = segs[[args.seg_id_field, "geometry"]].copy()
    segs = segs.rename(columns={args.seg_id_field: "seg_id"})

    # Force CRS consistency (assumes model coords match gpkg coords)
    # If segs has a CRS, assign it to pts.
    if segs.crs is not None:
        pts = pts.set_crs(segs.crs, allow_override=True)

    # Nearest join
    joined = gpd.sjoin_nearest(
        pts, segs,
        how="left",
        distance_col="dist_m"
    )

    out = joined[["node", "seg_id", "dist_m"]].copy()

    # optional distance filter
    if args.max_dist_m is not None:
        out = out[out["dist_m"] <= float(args.max_dist_m)].copy()

    out = out.sort_values("node")
    out[["node", "seg_id"]].to_csv(out_csv, index=False)

    # Quick sanity summary
    print(f"Wrote: {out_csv}")
    print("Rows:", len(out))
    print("Unique seg_id:", out["seg_id"].nunique(dropna=True))
    print("Distance (m) min/median/max:",
          float(out["dist_m"].min()),
          float(out["dist_m"].median()),
          float(out["dist_m"].max()))
    print("\nCounts per seg_id (top 10):")
    print(out["seg_id"].value_counts().head(10))


if __name__ == "__main__":
    main()
