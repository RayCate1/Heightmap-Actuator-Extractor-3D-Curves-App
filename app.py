import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
import plotly.graph_objects as go
from io import BytesIO
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves ahhh")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in mm)", type=["stl", "obj"])

# ── 3) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    width_val      = st.number_input("Bounds Width (ft)", value=6.0)
    height_val     = st.number_input("Bounds Height (ft)", value=4.0)
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1)
    nz            = st.slider("Z-Resolution (# slices)", 10, 10_000, 200)
    # NEW: checkbox to shift zero
    shift_zero = st.checkbox(
        "Re-zero at mid-height (shift all heights down by half the bounding-box Y)", 
        value=False
    )
    zero_disp = st.checkbox(
        "Relative Movment (Actuators will start at zero, and be given heights relative to their start position)",
        value=False
    )
# ── 4) LAUNCH PROCESS ────────────────────────────────────────
if st.button("Process"):
    if not uploaded:
        st.error("Please upload a model file.")
        st.stop()

    # 4.1 Convert bounds (ft→mm)
    bounds_width_mm  = width_val  * 304.8 
    bounds_height_mm = height_val * 304.8
    
    # 4.2 Load mesh
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()

    # 4.3 X positions in mm
    if num_actuators > 1:
        xs_mm = np.linspace(0, bounds_width_mm, num_actuators)
    else:
        xs_mm = np.array([0.0])
    # 4.3b Convert actuator X positions to inches
    xs_in = xs_mm / 25.4

    # 4.4 Map into mesh coords
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_mm / bounds_width_mm) * (xmax - xmin)

    # ── **4.5 Z slices** ⬅️ moved **here**, before ray‐cast
    zs = np.linspace(zmin, zmax, nz)

    # 4.6 Nudge inwards
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps

    # 4.7 Ray-cast heights (mm)
    H_mm = np.full((len(xs_mesh), nz), np.nan)
    for i, x0 in enumerate(xs_mesh):
        origins = np.column_stack([
            np.full(nz, x0),
            np.full(nz, ymax + (ymax - ymin) * 0.1),
            zs
        ])
        dirs = np.tile([0.0, -1.0, 0.0], (nz, 1))
        locs, idxs, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idxs):
            H_mm[i, idxs] = locs[:, 1]

    # Apply the vertical shift if requested
    if shift_zero:
        half_y_span = (ymax - ymin) / 2.0
        H_mm = H_mm - half_y_span
        
    # ── 4.8b Smooth/spline-interpolate any remaining NaNs ─────────
    for i in range(len(xs_mesh)):
        row   = H_mm[i, :]           # the mm heights for actuator i
        idx   = np.arange(nz)        # sample indices
        valid = ~np.isnan(row)       # mask of good points
    
        # Only fit if we have at least 4 non-NaNs (for a cubic spline)
        if valid.sum() >= 4:
            # fit a cubic spline through the known points
            spline = UnivariateSpline(idx[valid], row[valid], k=3, s=0)
            # evaluate it at every slice index (fills NaNs smoothly)
            H_mm[i, :] = spline(idx)
        else:
            # fallback: linear/constant fill if too few points
            H_mm[i, :] = pd.Series(row).interpolate(
                method='linear', limit_direction='both'
            ).values
        
    # 4.9 Convert outputs to Imperial
    H_in = H_mm / 25.4
    xs_in = xs_mm / 25.4

    # ── 4.11 Heights table (inches) ─────────────────────────
    rows = []
    for i, xi in enumerate(xs_in, start=1):
        row = {"Actuator": i, "X (in)": float(round(xi, 3))}
        for j in range(nz):
            v = H_in[i-1, j]
            row[f"Z[{j}]"] = None if np.isnan(v) else float(round(v, 3))
        rows.append(row)
    df = pd.DataFrame(rows)
    with st.expander("Parent Height Data (inches)", expanded=False):
        st.subheader("Parent Height Data (inches)")
        st.dataframe(df, use_container_width=True)
