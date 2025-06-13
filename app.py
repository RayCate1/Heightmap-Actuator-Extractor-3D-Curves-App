import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
import plotly.graph_objects as go
from io import BytesIO
from scipy.interpolate import UnivariateSpline

st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves ahh")

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

    # 4.8 Fill any fully-empty actuator rows
    for i in range(len(xs_mesh)):
        if np.isnan(H_mm[i]).all():
            if i > 0 and not np.isnan(H_mm[i-1]).all():
                H_mm[i] = H_mm[i-1]
            elif i < len(xs_mesh)-1 and not np.isnan(H_mm[i+1]).all():
                H_mm[i] = H_mm[i+1]

    # NEW: apply the vertical shift if requested
    if shift_zero:
        half_y_span = (ymax - ymin) / 2.0
        H_mm = H_mm - half_y_span

    # 4.9 Convert outputs to Imperial
    H_in = H_mm / 25.4
    xs_in = xs_mm / 25.4

    # 4.8b Fill any remaining NaNs by linear interpolation along the Z-axis
    for i in range(len(xs_mesh)):
        row = H_mm[i, :]                     # this is in mm
        idx = np.arange(nz)
        mask = np.isnan(row)
        if mask.all():
            continue                         # already handled by full-row fallback
        # interpolate only where row isn’t NaN
        valid = ~mask
        row[mask] = np.interp(idx[mask], idx[valid], row[valid])
        H_mm[i, :] = row

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

    # ── 4.12 Fit a spline through each actuator’s height curve and get dy/ds ──
    s  = np.arange(nz)
    vy = np.zeros_like(H_in)
    for i in range(len(xs_in)):
        H_i      = H_in[i, :]
        spline   = UnivariateSpline(s, H_i, k=3, s=0)
        vy[i, :] = spline.derivative(n=1)(s)

    # ── 4.13 Build normals & normal‐based displacement ─────────────────────
    thickness_in = comp_thickness
    v_norm       = np.sqrt(1 + vy**2)
    nx           = np.zeros_like(vy)
    ny           = -vy     / v_norm
    nz_norm      = 1.0     / v_norm
    disp_normal  = thickness_in * v_norm
    theta_n      = np.arccos(np.clip(nz_norm, -1.0, 1.0))

    vec_rows = []
    A = len(xs_in)
    for i in range(A):
        for j in range(nz):
            vec_rows.append({
                "Actuator":  i+1,
                "Slice":     j,
                "nx":        float(round(nx[i, j],     4)),
                "ny":        float(round(ny[i, j],     4)),
                "nz":        float(round(nz_norm[i, j],4)),
                "θₙ (rad)":  float(round(theta_n[i, j],4)),
                "disp (in)": float(round(disp_normal[i, j],4)),
                "normal":    f"{{{nx[i, j]:.4f}, {ny[i, j]:.4f}, {nz_norm[i, j]:.4f}}}"
            })
    vec_df = pd.DataFrame(vec_rows)
    with st.expander("Normal Vectors & Normal-Based Displacement", expanded=False):
        st.subheader("Normal Vectors & Normal-Based Displacement")
        st.dataframe(vec_df, use_container_width=True)

    # ── 4.14 Build & show “top”/“bottom” height table ─────────────────────────
    disp_half = disp_normal / 2.0
    
    # 1) form the displaced curves (top & bottom)
    top_curve = H_in + disp_half     # shape (A, nz)
    bot_curve = H_in - disp_half

    # 2) optionally zero‐first *only* the displaced data
    if zero_disp:
        top_curve = top_curve - top_curve[:, 0][:, None]
        bot_curve = bot_curve - bot_curve[:, 0][:, None]
    disp_rows = []
    for i in range(A):
        for kind, curve in (("top", top_curve), ("bottom", bot_curve)):
            row = {"Actuator": i+1, "Type": kind}
            for j in range(nz):
                row[f"Z[{j}]"] = float(round(curve[i, j], 3))
            disp_rows.append(row)
    disp_df = pd.DataFrame(disp_rows)
    with st.expander("Displaced Height Data (inches) — Top & Bottom Curves", expanded=False):
        st.dataframe(disp_df, use_container_width=True)

    # ── 4.15 Plot Displaced Curves in 3D ─────────────────────────
    st.subheader("Displaced Curves in 3D")
    fig = go.Figure()
    samp = np.arange(nz)
    for i in range(len(xs_in)):
        top_z = H_in[i, :] + disp_normal[i, :] / 2
        bot_z = H_in[i, :] - disp_normal[i, :] / 2
        fig.add_trace(go.Scatter3d(
            x=np.full(nz, i + 1),
            y=samp,
            z=top_z,
            mode='lines',
            name=f"Act {i + 1} Top"
        ))
        fig.add_trace(go.Scatter3d(
            x=np.full(nz, i + 1),
            y=samp,
            z=bot_z,
            mode='lines',
            name=f"Act {i + 1} Bottom"
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
    # # ── Visualize Curves + Normals ───────────────────────────────
    # st.subheader("3D Curves with Surface Normals")
    
    # fig = go.Figure()
    # samp = np.arange(nz)
    # A    = len(xs_in)
    
    # # 1) Plot each actuator’s curve
    # for i in range(A):
    #     fig.add_trace(go.Scatter3d(
    #         x=np.full(nz, i+1),
    #         y=samp,
    #         z=H_in[i, :],
    #         mode='lines',
    #         line=dict(width=4),
    #         name=f"Actuator {i+1}"
    #     ))
    
    # # 2) Prepare flattened grids for normals
    # Xg = np.repeat(np.arange(1, A+1)[:, None], nz, axis=1).ravel()
    # Yg = np.repeat(samp[None, :],        A,        axis=0).ravel()
    # Zg = H_in.ravel()
    
    # Ug = nx.ravel()       # x-component (should be zero)
    # Vg = ny.ravel()       # y-component
    # Wg = nz_norm.ravel()  # z-component
    
    # # 3) Overlay normals as cones
    # fig.add_trace(go.Cone(
    #     x=Xg, y=Yg, z=Zg,
    #     u=Ug, v=Vg, w=Wg,
    #     anchor="tail",
    #     sizemode="absolute",
    #     sizeref=0.5,      # ← adjust to scale arrow lengths
    #     showscale=False
    # ))
    
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="Actuator #", autorange="reversed"),
    #         yaxis=dict(title="Slice #"),
    #         zaxis=dict(title="Height (in)")
    #     ),
    #     height=700,
    #     margin=dict(l=20, r=20, t=30, b=20)
    # )
    
    # st.plotly_chart(fig, use_container_width=True)

    # ── 2) MACHINE PARAMETERS (Imperial) ────────────────────────
    # st.markdown("### Machine Parameters (Imperial Defaults)")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     after_temp     = st.number_input("After-Dye Temperature (°F)", value=338.0)
    #     dye_temp       = st.number_input("Dye Temperature (°F)", value=302.0)
    #     wet_temp       = st.number_input("Wet Fiber Temperature (°F)", value=59.0)
    # with col2:
    #     pull_speed     = st.number_input("Pull Speed (in/min)", value=15.0, format="%.2f")
    #     resin_ratio    = st.text_input("Resin:Fiber Ratio", value="1:1")
    #     comp_force     = st.number_input("Compressive Force (psi)", value=15.0)
    # with col3:
    #     dye_thickness  = st.number_input("Dye Thickness (in)", value=0.0)
    #
    # params = {
    #     "model_file":            uploaded.name,
    #     "pull_speed_in_per_min": pull_speed,
    #     "dye_temperature_F":     dye_temp,
    #     "wet_fiber_temp_F":      wet_temp,
    #     "after_dye_temp_F":      after_temp,
    #     "resin_to_fiber_ratio":  resin_ratio,
    #     "compressive_force_psi": comp_force,
    #     "composite_thickness_in": comp_thickness,
    #     "dye_thickness_in":      dye_thickness,
    #     "bounds_width_ft":       width_val,
    #     "bounds_height_ft":      height_val,
    #     "number_of_actuators":   num_actuators,
    #     "z_resolution":          nz
    # }
    # st.subheader("Machine Params JSON")
    # st.download_button(
    #     "Download params.json",
    #     data=json.dumps(params, indent=2),
    #     file_name="params.json",
    #     mime="application/json"
    # )
