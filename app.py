import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves (in inches)")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in mm)", type=["stl", "obj"])

# ── 2) MACHINE PARAMETERS ─────────────────────────────────
st.markdown("### Machine Parameters (defaults shown)")

col1, col2, col3 = st.columns(3)
with col1:
    pull_speed      = st.number_input("Pull Speed", value=15.0, format="%.2f", help="Pull speed")
    pull_speed_unit = st.selectbox("Pull Speed Unit", ["cm/min", "mm/min", "in/min"], index=0)
    dye_temp        = st.number_input("Dye Temperature", value=150.0)
    dye_temp_unit   = st.selectbox("Dye Temp Unit", ["°C", "°F"], index=0)
with col2:
    wet_temp        = st.number_input("Wet Fiber Temp", value=15.0)
    wet_temp_unit   = st.selectbox("Wet Fiber Temp Unit", ["°C", "°F"], index=0)
    after_temp      = st.number_input("After-Dye Temp", value=170.0)
    after_temp_unit = st.selectbox("After-Dye Temp Unit", ["°C", "°F"], index=0)
    resin_ratio     = st.text_input("Resin:Fiber Ratio", value="1:1")
with col3:
    comp_force      = st.number_input("Compressive Force", value=15.0)
    comp_force_unit = st.selectbox("Compressive Force Unit", ["psi", "Pa"], index=0)
    comp_thickness  = st.number_input("Composite Thickness", value=1.0)
    comp_thickness_u= st.selectbox("Composite Thick Unit", ["in", "mm"], index=0)
    dye_thickness   = st.number_input("Dye Thickness", value=0.0)
    dye_thickness_u = st.selectbox("Dye Thick Unit", ["mm", "in"], index=0)

# ── 3) MACHINE BOUNDS & ACTUATORS ──────────────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    width_val   = st.number_input("Bounds Width", value=6.0)
    width_unit  = st.selectbox("Width Unit", ["ft", "mm"], index=0)
    height_val  = st.number_input("Bounds Height", value=4.0)
    height_unit = st.selectbox("Height Unit", ["ft", "mm"], index=0)
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1)
    nz            = st.slider("Z-Resolution (# samples)", 10, 10000, 200)

# ── 4) LAUNCH PROCESS ───────────────────────────────────────
if st.button("Process"):
    if not uploaded:
        st.error("Please upload a model file.")
        st.stop()

    # 4.1 Convert bounds to mm
    def to_mm(v, u):
        return v * 304.8 if u == "ft" else v

    bounds_width_mm  = to_mm(width_val, width_unit)
    bounds_height_mm = to_mm(height_val, height_unit)

    # 4.2 Load mesh
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()

    # 4.3 Prepare actuator X positions in mesh coords
    if num_actuators > 1:
        xs_mm = np.linspace(0, bounds_width_mm, num_actuators)
    else:
        xs_mm = np.array([0.0])

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_mm / bounds_width_mm) * (xmax - xmin)

    # 4.4 Z samples in mesh coords
    zs = np.linspace(zmin, zmax, nz)

    # 4.5 Nudge edges slightly inward
    if num_actuators > 1:
        mesh_span    = xmax - xmin
        mesh_spacing = mesh_span / (num_actuators - 1)
        eps          = mesh_spacing * 0.01
        xs_mesh[0]   = xmin + eps
        xs_mesh[-1]  = xmax - eps

    # 4.6 Ray‐cast for heights (in mm)
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

    # 4.7 Fallback for missing rays
    for i in range(len(xs_mesh)):
        if np.isnan(H_mm[i]).all():
            if i > 0 and not np.isnan(H_mm[i-1]).all():
                H_mm[i] = H_mm[i-1]
            elif i < len(xs_mesh)-1 and not np.isnan(H_mm[i+1]).all():
                H_mm[i] = H_mm[i+1]

    # 4.8 Convert heights to inches
    H_in = H_mm / 25.4

    # 4.9 Build params dict
    params = {
        "model_file": uploaded.name,
        "pull_speed":      {"value": pull_speed,      "unit": pull_speed_unit},
        "dye_temperature": {"value": dye_temp,         "unit": dye_temp_unit},
        "wet_fiber_temp":  {"value": wet_temp,         "unit": wet_temp_unit},
        "after_dye_temp":  {"value": after_temp,       "unit": after_temp_unit},
        "resin_to_fiber_ratio": resin_ratio,
        "compressive_force":    {"value": comp_force,   "unit": comp_force_unit},
        "composite_thickness":  {"value": comp_thickness,"unit": comp_thickness_u},
        "dye_thickness":        {"value": dye_thickness, "unit": dye_thickness_u},
        "bounds_width":         {"value": width_val,     "unit": width_unit},
        "bounds_height":        {"value": height_val,    "unit": height_unit},
        "number_of_actuators":  num_actuators,
        "z_resolution":         nz
    }

    # 4.10 Show & download params.json
    st.subheader("Machine Params JSON file")
    st.download_button(
        "Download params.json",
        data=json.dumps(params, indent=2),
        file_name="params.json",
        mime="application/json"
    )

    # 4.11 Build & show heights table (in inches)
    st.subheader("Height Data (in inches)")
    rows = []
    for i in range(len(xs_mesh)):
        row = {"Actuator": i+1}
        for j in range(nz):
            val = None if np.isnan(H_in[i, j]) else float(f"{H_in[i, j]:.4f}")
            row[f"Sample[{j}]"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # ── 4.12 BUILD & PLOT OFFSET CURVES ───────────────────────────
    # compute composite thickness in inches
    thickness_in = comp_thickness if comp_thickness_u == "in" else comp_thickness / 25.4

    st.subheader("Actuator Curves ± Offset (in inches)")
    fig = go.Figure()
    z_coords = np.arange(nz)

    for i in range(len(xs_mesh)):
        y_orig = H_in[i, :]
        x_orig = np.full(nz, i+1)  # actuator index on X
        z_orig = z_coords

        # local deltas
        dy = np.diff(y_orig)
        dz = np.diff(z_orig)                # ones
        lengths    = np.hypot(dy, dz)
        cos_angles = np.where(lengths > 1e-6, dy / lengths, 1.0)
        disp       = thickness_in / cos_angles

        uy = dy / lengths
        uz = dz / lengths

        # pad once to length nz
        uy_full   = np.concatenate(([uy[0]], uy))
        uz_full   = np.concatenate(([uz[0]], uz))
        disp_full = np.concatenate(([disp[0]], disp))

        # offset curves
        y_top = y_orig + uy_full * disp_full
        z_top = z_orig + uz_full * disp_full
        y_bot = y_orig - uy_full * disp_full
        z_bot = z_orig - uz_full * disp_full

        fig.add_trace(go.Scatter3d(
            x=x_orig, y=y_top, z=z_top,
            mode='lines', name=f"Act {i+1} Top"
        ))
        fig.add_trace(go.Scatter3d(
            x=x_orig, y=y_bot, z=z_bot,
            mode='lines', name=f"Act {i+1} Bottom"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Actuator Index",
            yaxis_title="Height (inches)",
            zaxis_title="Sample Index"
        ),
        height=600, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
