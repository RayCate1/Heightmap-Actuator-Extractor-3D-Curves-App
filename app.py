import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
import plotly.graph_objects as go
from io import BytesIO
from scipy.interpolate import UnivariateSpline

st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves m ")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in mm)", type=["stl","obj"])

# ── 3) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    width_val  = st.number_input("Bounds Width (ft)", value=6.0)
    height_val = st.number_input("Bounds Height (ft)", value=4.0)
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1)
    nz            = st.slider("Z-Resolution (# slices)", 10, 10_000, 200)

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
st.subheader("Parent Height Data (inches)")
rows = []
for i, xi in enumerate(xs_in, start=1):
    row = {"Actuator": i, "X (in)": float(round(xi, 3))}
    for j in range(nz):
        v = H_in[i-1, j]
        row[f"Z[{j}]"] = None if np.isnan(v) else float(round(v, 3))
    rows.append(row)
df = pd.DataFrame(rows)

with st.expander("Parent Height Data (inches)", expanded=False):
    st.dataframe(df, use_container_width=True)
# ── 4.11 fit a spline through each actuator’s height curve and get dy/ds ────────────────
s = np.arange(nz)                                 # parameter (slice index)
vy = np.zeros_like(H_in)                          # will store dy/ds

for i in range(len(xs_in)):
    H_i = H_in[i, :]                              # heights for actuator i
    # cubic spline, exact fit (no smoothing)
    spline = UnivariateSpline(s, H_i, k=3, s=0)
    # analytic first derivative at each s
    vy[i, :] = spline.derivative(n=1)(s)

# ── 4.12 build normals & normal‐based displacement ──────────
thickness_in = comp_thickness                      # inches

# arc‐length factor ‖r′(s)‖ = sqrt((dy/ds)^2 + 1)
v_norm    = np.sqrt(vy**2 + 1.0)                   # shape (A, nz)

# unit normal N = (0, 1, -dy/ds) / ‖r′(s)‖
nx        = np.zeros_like(vy)
ny        = 1.0    / v_norm
nz_norm   = -vy    / v_norm

# displacement along the normal = thickness * ‖r′(s)‖
disp_normal = thickness_in * v_norm                # shape (A, nz)

# angle between N and vertical = arccos(ny)
theta_n     = np.arccos(np.clip(ny, -1.0, 1.0))     # radians

# ── 4.13 build flat table of normals + θ + disp ────────────
st.subheader("Normal Vectors & Normal-Based Displacement")
vec_rows = []
A = len(xs_in)
for i in range(A):
    for j in range(nz):
        vec_rows.append({
            "Actuator":  i+1,
            "Slice":     j,
            "nx":        float(round(nx[i,j],     4)),
            "ny":        float(round(ny[i,j],     4)),
            "nz":        float(round(nz_norm[i,j],4)),
            "θₙ (rad)":  float(round(theta_n[i,j],4)),
            "disp (in)": float(round(disp_normal[i,j],4)),
            "normal":    f"{{{nx[i,j]:.4f}, {ny[i,j]:.4f}, {nz_norm[i,j]:.4f}}}"
        })

vec_df = pd.DataFrame(vec_rows)
with st.expander("Normal Vectors & Normal-Based Displacement", expanded=False):
    st.dataframe(vec_df, use_container_width=True)
# # ── 4.14 3D Plot: curves + normal vectors ────────────────────
# st.subheader("Actuator Curves with Surface Normals")

# fig = go.Figure()

# # draw each actuator’s curve
# for i in range(A):
#     fig.add_trace(go.Scatter3d(
#         x=np.full(nz, i+1),
#         y=np.arange(nz),
#         z=H_in[i, :],
#         mode='lines',
#         name=f"Act {i+1}"
#     ))

# # set up grid & normal components
# Xg = np.repeat(np.arange(1, A+1)[:,None], nz, axis=1)
# Yg = np.repeat(np.arange(nz)[None,:],        A, axis=0)
# Zg = H_in

# Ug = nx
# Vg = ny
# Wg = nz_norm

# fig.add_trace(go.Cone(
#     x=Xg.flatten(),
#     y=Yg.flatten(),
#     z=Zg.flatten(),
#     u=Ug.flatten(),
#     v=Vg.flatten(),
#     w=Wg.flatten(),
#     anchor="tail",
#     sizemode="absolute",
#     sizeref=10,      # adjust to scale your normals
#     showscale=False
# ))

# fig.update_layout(
#     scene=dict(
#         xaxis_title="Actuator #",
#         yaxis_title="Sample #",
#         zaxis_title="Height (in)"
#     ),
#     height=700,
#     margin=dict(l=20, r=20, t=40, b=20),
# )
# st.plotly_chart(fig, use_container_width=True)
# ── 4.14 Build & show “top”/“bottom” displaced height table ──

st.subheader("Displaced Height Data (inches) — Top & Bottom Curves")

# half-displacement
disp_half = disp_normal / 2.0   # shape (A, nz)

disp_rows = []
A = len(xs_in)
for i in range(A):
    actuator_id = i+1
    # start two rows: one for the top-displaced curve, one for bottom
    row_top  = {"Actuator": actuator_id, "Type": "top"}
    row_bot  = {"Actuator": actuator_id, "Type": "bottom"}
    for j in range(nz):
        h   = H_in[i, j]
        dh  = disp_half[i, j]
        row_top[f"Z[{j}]"] = float(round(h + dh, 3))
        row_bot[f"Z[{j}]"] = float(round(h - dh, 3))
    disp_rows.extend([row_top, row_bot])

disp_df = pd.DataFrame(disp_rows)
with st.expander("Displaced Height Data (inches) — Top & Bottom Curves", expanded=False):
    st.dataframe(disp_df, use_container_width=True)
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

# ── 4.15 Plot Displaced Curves in 3D ─────────────────────────
st.subheader("Displaced Actuator Curves in 3D")
fig = go.Figure()
samp = np.arange(nz)

for i in range(len(xs_in)):
    top_z = H_in[i, :] + disp_normal[i, :] / 2
    bot_z = H_in[i, :] - disp_normal[i, :] / 2

    fig.add_trace(go.Scatter3d(
        x=np.full(nz, i+1),
        y=samp,
        z=top_z,
        mode='lines',
        name=f"Act {i+1} Top"
    ))
    fig.add_trace(go.Scatter3d(
        x=np.full(nz, i+1),
        y=samp,
        z=bot_z,
        mode='lines',
        name=f"Act {i+1} Bottom"
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="Actuator #",
        xaxis=dict(autorange="reversed"),
        yaxis_title="Sample #",
        zaxis_title="Displaced Height (in)"
    ),
    height=600,
    margin=dict(l=20, r=20, t=40, b=20)
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
