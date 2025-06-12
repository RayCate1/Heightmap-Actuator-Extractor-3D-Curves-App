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
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in mm)", type=["stl","obj"])

# ── 3) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    width_val  = st.number_input("Bounds Width (ft)", value=6.0)
    height_val = st.number_input("Bounds Height (ft)", value=4.0)
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1)
    nz            = st.slider("Z-Resolution (# slices)", 10, 10_000, 200)
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)

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

    # 4.4 Map into mesh coords
    (xmin,ymin,zmin),(xmax,ymax,zmax) = mesh.bounds
    xs_mesh = xmin + (xs_mm/bounds_width_mm)*(xmax-xmin)

    # 4.5 Z slices
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
            np.full(nz, ymax + (ymax-ymin)*0.1),
            zs
        ])
        dirs = np.tile([0.0, -1.0, 0.0], (nz,1))
        locs, idxs, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idxs):
            H_mm[i, idxs] = locs[:,1]

    # 4.8 Fill any fully-empty actuator rows
    for i in range(len(xs_mesh)):
        if np.isnan(H_mm[i]).all():
            if i>0 and not np.isnan(H_mm[i-1]).all():
                H_mm[i] = H_mm[i-1]
            elif i<len(xs_mesh)-1 and not np.isnan(H_mm[i+1]).all():
                H_mm[i] = H_mm[i+1]

    # 4.9 Convert outputs to Imperial
    H_in = H_mm / 25.4
    xs_in = xs_mm / 25.4

    # 4.11 Heights table (inches)
    st.subheader("Height Data (inches)")
    rows = []
    for i, xi in enumerate(xs_in, start=1):
        row = {"Actuator": i, "X (in)": float(round(xi,3))}
        for j in range(nz):
            v = H_in[i-1,j]
            row[f"Z[{j}]"] = None if np.isnan(v) else float(round(v,3))
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # 4.12 3D Plot: X=actuator #, Y=sample #, Z=height (in)
    st.subheader("Actuator Curves in 3D")
    fig = go.Figure()
    samp = np.arange(nz)
    for i in range(len(xs_in)):
        fig.add_trace(go.Scatter3d(
            x=np.full(nz, i+1),     # actuator number 1…N
            y=samp,                  # sample (slice) index
            z=H_in[i, :],            # height in inches
            mode='lines',
            name=f"Act {i+1}"
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Actuator #",
            xaxis=dict(autorange="reversed"),
            yaxis_title="Sample #",
            zaxis_title="Height (in)"
        ),
        height=600, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 2) MACHINE PARAMETERS (Imperial) ────────────────────────
st.markdown("### Machine Parameters (Imperial Defaults)")
col1, col2, col3 = st.columns(3)
with col1:
    after_temp     = st.number_input("After-Dye Temperature (°F)", value=338.0)
    dye_temp       = st.number_input("Dye Temperature (°F)", value=302.0)
    wet_temp       = st.number_input("Wet Fiber Temperature (°F)", value=59.0)
with col2:
    pull_speed     = st.number_input("Pull Speed (in/min)", value=15.0, format="%.2f")
    resin_ratio    = st.text_input("Resin:Fiber Ratio", value="1:1")
    comp_force     = st.number_input("Compressive Force (psi)", value=15.0)
with col3:
    dye_thickness  = st.number_input("Dye Thickness (in)", value=0.0)

        # Params JSON (all Imperial)
params = {
        "model_file":            uploaded.name,
        "pull_speed_in_per_min": pull_speed,
        "dye_temperature_F":     dye_temp,
        "wet_fiber_temp_F":      wet_temp,
        "after_dye_temp_F":      after_temp,
        "resin_to_fiber_ratio":  resin_ratio,
        "compressive_force_psi": comp_force,
        "composite_thickness_in": comp_thickness,
        "dye_thickness_in":      dye_thickness,
        "bounds_width_ft":       width_val,
        "bounds_height_ft":      height_val,
        "number_of_actuators":   num_actuators,
        "z_resolution":          nz
}
st.subheader("Machine Params JSON")
st.download_button(
        "Download params.json",
        data=json.dumps(params, indent=2),
        file_name="params.json",
        mime="application/json"
)

# ── compute height‐derivative (vy) and slice‐derivative (vz) ──
vy     = np.gradient(H_in, axis=1)       # shape (num_actuators, nz)
slices = np.arange(nz)
vz     = np.gradient(slices)             # shape (nz,) — all ones

# vx is zeros (not used)
vx = np.zeros_like(vy)

# ── compute actuator displacement with clamp ─────────────────
thickness_in = comp_thickness             # in inches
eps          = 1e-3                       # slope threshold

abs_vy = np.abs(vy)
# raw geometric ratio = sqrt(vy^2 + vz^2) / |vy|
ratio  = np.sqrt(vy**2 + vz[None, :]**2) / abs_vy
ratio  = np.where(abs_vy < eps, 1.0, ratio)
actuator_displacement = thickness_in * ratio
# shape (num_actuators, nz)

# ── compute angle θ between (0,1,0) and (0,vy,vz) ─────────────
# cosθ = vy / sqrt(vy^2 + vz^2)
mag = np.sqrt(vy**2 + vz[None, :]**2)
# avoid division by zero
mag_safe = np.where(mag == 0, 1e-6, mag)
cos_theta = vy / mag_safe
# clamp into [-1,1]
cos_theta = np.clip(cos_theta, -1.0, 1.0)
theta = np.arccos(cos_theta)  # in radians, shape (num_actuators, nz)

# ── build flat table of velocity vectors, θ & displacement ────
vec_rows = []
for i in range(len(xs_in)):        # each actuator
    for j in range(nz):            # each slice
        vy_ij    = vy[i, j]
        vz_j     = vz[j]
        theta_ij = theta[i, j]
        disp_ij  = actuator_displacement[i, j]
        vec_rows.append({
            "Actuator": i+1,
            "Slice":    j,
            "vx":       0.0,
            "vy":       float(round(vy_ij,   4)),
            "vz":       float(round(vz_j,    4)),
            "θ (rad)":   float(round(theta_ij,4)),
            "disp (in)": float(round(disp_ij, 4)),
            "vector":   f"{{0, {vy_ij:.4f}, {vz_j:.4f}}}"
        })

vec_df = pd.DataFrame(vec_rows)
st.subheader("Velocity, Angle & Displacement")
st.dataframe(vec_df, use_container_width=True)

# ── 4.12 3D Plot: curves + velocity vectors ─────────────────
st.subheader("Actuator Curves with Velocity Vectors")

fig = go.Figure()

# 1) draw each actuator’s curve
for i in range(len(xs_in)):
    fig.add_trace(go.Scatter3d(
        x=np.full(nz, i+1),
        y=np.arange(nz),
        z=H_in[i, :],
        mode='lines',
        name=f"Act {i+1}"
    ))

# 2) prepare the vector field (u,v,w) at each point
A = len(xs_in)
# grid of positions
Xg = np.repeat(np.arange(1, A+1)[:, None], nz, axis=1)
Yg = np.repeat(np.arange(nz)[None, :], A, axis=0)
Zg = H_in

# vector components in plot axes:
#   u = 0  (no x‐movement)
#   v = 1  (1 slice per sample along y‐axis)
#   w = vy (dH/dslice along z‐axis)
Ug = np.zeros_like(Zg)
Vg = np.ones_like(Zg)
Wg = vy  # vy from your np.gradient(H_in)

fig.add_trace(go.Cone(
    x=Xg.flatten(),
    y=Yg.flatten(),
    z=Zg.flatten(),
    u=Ug.flatten(),
    v=Vg.flatten(),
    w=Wg.flatten(),
    anchor="tail",
    sizemode="absolute",
    sizeref=2,        # tweak this to scale arrow length
    showscale=False
))

fig.update_layout(
    scene=dict(
        xaxis_title="Actuator #",
        yaxis_title="Sample #",
        zaxis_title="Height (in)"
    ),
    height=700,
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)
