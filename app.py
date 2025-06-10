import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planer geometry (OBJ/STL in mm)", type=["stl","obj"])

# ── 2) MACHINE PARAMETERS ─────────────────────────────────
st.markdown("### Machine Parameters (defaults shown)")

col1, col2, col3 = st.columns(3)
with col1:
    pull_speed      = st.number_input("Pull Speed", value=15.0, help="Pull speed", format="%.2f", key="pull_speed")
    pull_speed_unit = st.selectbox("Pull Speed Unit", ["cm/min","mm/min","in/min"], index=0, key="pull_unit")
    dye_temp        = st.number_input("Dye Temperature", value=150.0, key="dye_temp")
    dye_temp_unit   = st.selectbox("Dye Temp Unit", ["°C","°F"], index=0, key="dye_unit")
with col2:
    wet_temp        = st.number_input("Wet Fiber Temp", value=15.0, key="wet_temp")
    wet_temp_unit   = st.selectbox("Wet Fiber Temp Unit", ["°C","°F"], index=0, key="wet_unit")
    after_temp      = st.number_input("After-Dye Temp", value=170.0, key="after_temp")
    after_temp_unit = st.selectbox("After-Dye Temp Unit", ["°C","°F"], index=0, key="after_unit")
    resin_ratio     = st.text_input("Resin:Fiber Ratio", value="1:1", key="resin_ratio")
with col3:
    comp_force      = st.number_input("Compressive Force", value=15.0, key="comp_force")
    comp_force_unit = st.selectbox("Compressive Force Unit", ["psi","Pa"], index=0, key="comp_unit")
    comp_thickness  = st.number_input("Composite Thickness", value=1.0, key="comp_thick")
    comp_thickness_u= st.selectbox("Composite Thick Unit", ["in","mm"], index=0, key="comp_thick_unit")
    dye_thickness   = st.number_input("Dye Thickness", value=0.0, key="dye_thick")
    dye_thickness_u = st.selectbox("Dye Thick Unit", ["mm","in"], index=0, key="dye_thick_unit")

# ── 3) MACHINE BOUNDS & ACTUATORS ──────────────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    width_val   = st.number_input("Bounds Width", value=6.0, key="bounds_width")
    width_unit  = st.selectbox("Width Unit", ["ft","mm"], index=0, key="width_unit")
    height_val  = st.number_input("Bounds Height", value=4.0, key="bounds_height")
    height_unit = st.selectbox("Height Unit", ["ft","mm"], index=0, key="height_unit")
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1, key="n_act")
    nz            = st.slider("Z-Resolution (# slices)", 10, 10000, 200, key="z_res")
    steps_per_mm  = st.number_input("Steps per mm", value=1.0, key="spm")

# ── 4) LAUNCH PROCESS ───────────────────────────────────────
if st.button("Process", key="process_btn"):
    if not uploaded:
        st.error("Please upload a model file."); st.stop()

    # 4.1 Convert bounds to mm
    def to_mm(v,u):
        return v*304.8 if u=="ft" else v
    bounds_width_mm  = to_mm(width_val, width_unit)
    bounds_height_mm = to_mm(height_val, height_unit)

    # 4.2 Load mesh
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    if mesh.is_empty:
        st.error("Mesh is empty."); st.stop()

    # 4.3 Prepare actuator X positions in user units [0…width]
    if num_actuators > 1:
        xs_mm = np.linspace(0, bounds_width_mm, num_actuators)
    else:
        xs_mm = np.array([0.0])

    # 4.4 Map mm→mesh coords
    (xmin,ymin,zmin),(xmax,ymax,zmax) = mesh.bounds
    xs_mesh = xmin + (xs_mm/bounds_width_mm)*(xmax-xmin)

    # 4.5 Z slices in mesh coords
    zs = np.linspace(zmin, zmax, nz)

 # 4.5 Nudge the edges slightly inward so rays hit
    if num_actuators > 1:
        # compute a small epsilon (1% of actuator‐spacing in mesh coordinates)
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
            H_mm[i, idxs] = locs[:,1]

    # 4.7 Fallback: if an entire actuator row is still empty, copy the nearest neighbor
    for i in range(len(xs_mesh)):
        if np.isnan(H_mm[i]).all():
            if i > 0 and not np.isnan(H_mm[i-1]).all():
                H_mm[i] = H_mm[i-1]
            elif i < len(xs_mesh)-1 and not np.isnan(H_mm[i+1]).all():
                H_mm[i] = H_mm[i+1]

    # 4.8 Convert to steps
    H_steps = np.round(H_mm * steps_per_mm).astype(int)

    # 4.9 Ray-cast for heights
    H_mm = np.full((len(xs_mesh),nz), np.nan)
    for i,x0 in enumerate(xs_mesh):
        origins   = np.column_stack([np.full(nz,x0),
                                     np.full(nz,ymax+(ymax-ymin)*0.1),
                                     zs])
        dirs      = np.tile([0,-1,0], (nz,1))
        locs, idxs,_ = mesh.ray.intersects_location(origins,dirs,multiple_hits=False)
        if len(idxs):
            H_mm[i,idxs] = locs[:,1]

    # 4.10 Convert to steps
    H_steps = np.round(H_mm*steps_per_mm).astype(int)

    # 4.11 Build params dict
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
        "z_resolution":         nz,
        "steps_per_mm":         steps_per_mm
    }

    # 4.12 Show & download params.json
    st.subheader("Machine Params")
    st.json(params)
    st.download_button(
        "Download params.json",
        data=json.dumps(params, indent=2),
        file_name="params.json",
        mime="application/json",
        key="download_params"
    )

    # 4.13 Build & show heights table
    st.subheader("Height Data (in steps)")
    rows=[]
    for i,xm in enumerate(xs_mm, start=1):
        row={"Actuator":i,"X (mm)":float(xm)}
        for j in range(nz):
            val = None if np.isnan(H_steps[i-1,j]) else int(H_steps[i-1,j])
            row[f"Z[{j}]"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True)

    # 4.14 3D Curves: X(mm), Y=steps, Z=slice index
    st.subheader("Actuator Curves in 3D")
    fig=go.Figure()
    z_steps=np.arange(nz)
    for i,xm in enumerate(xs_mm, start=1):
        fig.add_trace(go.Scatter3d(
            x=np.full(nz,xm),
            y=H_steps[i-1,:],
            z=z_steps,
            mode='lines',
            name=f"Act {i}"
        ))
    fig.update_layout(
        scene=dict(xaxis_title="X (mm)",
                   yaxis_title="Y (steps)",
                   zaxis_title="Z (slice idx)"),
        height=600,margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig,use_container_width=True)
