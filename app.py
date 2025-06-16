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
st.title("Heightmap Actuator Extractor & 3D Curves ahh")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in inches)", type=["stl", "obj"])

# ── 2) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    #Prompt user input
    width_val      = st.number_input("Bounds Width (in)", value=62)
    height_val     = st.number_input("Bounds Height (in)", value=14)
    num_actuators  = st.number_input("Number of Actuator Pairs", min_value=1, value=7, step=1)
    nz             = st.number_input("Z-Resolution (# slices)", value=1000)
with b2:
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)
    wheel_radius   = st.number_input("Wheel Radius (in)", value=0.625)
    heat_k         = st.number_input("Heating Element Thickness (in)", value=0.019685)
    # Checkbox to shift zero
    shift_zero = st.checkbox(
        "Re-zero at mid-height (shift all heights down by half the bounding-box Y)", 
        value=False
    )
    # Checkbox for relative movment
    zero_disp = st.checkbox(
        "Relative Movment (Actuators will start at zero, and be given heights relative to their start position)",
        value=False
    )
# ── 3) LAUNCH PROCESS ────────────────────────────────────────
if st.button("Process"):
    # If no mesh -> Error message
    if not uploaded:
        st.error("Please upload a model file.")
        st.stop()
        
    # 2) Bounds & actuator X positions (inches)
    #    width_val, height_val are now inches
    bounds_width_in  = width_val
    bounds_height_in = height_val
    
    # Load mesh (inches assumed)
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    
    # Error if mesh empty
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()
        
    # 4) Actuator X positions in mesh‐space
    if num_actuators > 1:
        xs_in = np.linspace(0, bounds_width_in, num_actuators)
    else:
        xs_in = np.array([0.0])
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_in / bounds_width_in) * (xmax - xmin)
    
    # 5) Z‐slice positions (inches)
    zs = np.linspace(zmin, zmax, nz)
    
    # 6) Nudge at the edges
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps
        
    # 7) Ray‐cast heights directly in inches
    H_in = np.full((len(xs_mesh), nz), np.nan)
    for i, x0 in enumerate(xs_mesh):
        origins = np.column_stack([
            np.full(nz, x0),
            np.full(nz, ymax + (ymax - ymin) * 0.1),
            zs
        ])
        dirs = np.tile([0.0, -1.0, 0.0], (nz,1))
        locs, idxs, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idxs):
            H_in[i, idxs] = locs[:,1]

    # 8) Optional mid‐height re‐zero
    if shift_zero:
        H_in -= (ymax - ymin) / 2.0
        
    # 9) Smooth/spline‐interpolate any remaining NaNs
    for i in range(len(xs_mesh)):
        row   = H_in[i, :]
        idx   = np.arange(nz)
        valid = ~np.isnan(row)
        if valid.sum() >= 4:
            spline    = UnivariateSpline(idx[valid], row[valid], k=3, s=0)
            H_in[i,:] = spline(idx)
        else:
            H_in[i,:] = pd.Series(row).interpolate(method='linear', limit_direction='both').values
            
    A = len(xs_in)   # number of actuators
    
    # The equation relating theta θ (angle between x axis and curve), the distance between axles k, of the frp and the 
    # displacment d (disance the vertical actuators need to add onto the original cuve to compansate for bending), is 
    # d=k/Cos(θ). From there, you simply add plus or minus 1/2 d to the parent curves. 
    
    # 1) Compute physical slice spacing
    #   - dz_in: horizontal distance per slice, in inches (in)
    dz_in = (zmax - zmin) / (nz - 1)               # inches per slice
    
    # 2) Fit a cubic spline per actuator to obtain smooth derivative dH/dz
    #    - s_phys: physical coordinate along horizontal (Z) axis in inches
    #    - H_in[i, :] holds heights in inches
    s_phys = np.arange(nz) * dz_in      # inches along Z-axis
    vy = np.zeros_like(H_in)            # slope array (dimensionless: in/in)
    for i in range(A):
        # Fit exact cubic spline through (s_phys, H_in[i,:])
        spline = UnivariateSpline(s_phys, H_in[i, :], k=3, s=0)
        # Derivative dy/dz_phys at each slice (unitless)
        vy[i, :] = spline.derivative(n=1)(s_phys)
    
    # 3) Compute tangent angle relative to horizontal axis (in degrees)
    #    - tangent vector in (horizontal, vertical) plane = (Δs, ΔH) = (1, m)
    #    - arctan2(vertical_component, horizontal_component) returns radians; convert to degrees
    angle_vs_horizontal = np.degrees(np.arctan2(vy, 1.0))  # degrees
    
    # 4) Determine effective span k (inches) between actuators/components
    k = (heat_k * 2) + (wheel_radius * 2) + comp_thickness
    
    # 5) Compute displacement: full d = k / cos(θ), then use half-displacement
    #    NumPy’s cos() expects radians, so convert degrees back to radians
    d_full = k / np.cos(np.radians(angle_vs_horizontal))  # total displacement d = k/Cos(θ)
    disp = d_full / 2.0                                     # half-displacement to apply (inches)
    
    
    # 6) Build new top/bottom curves using pointwise half-displacement
    #    New curves: H_top = H_in + disp, H_bot = H_in - disp
    if zero_disp:
        # zero-relative: subtract each actuator's starting height afterward
        top_curve = H_in + disp
        bot_curve = H_in - disp
        top_curve -= top_curve[:, 0][:, None]
        bot_curve -= bot_curve[:, 0][:, None]
    else:
        top_curve = H_in + disp
        bot_curve = H_in - disp

    






    
    #DISPLAY STUFF
    #Parent data
    rows = []
    for i, xi in enumerate(xs_in, start=1):
        row = {"Actuator": i, "X (in)": float(xi)}
        for j in range(nz):
            v = H_in[i-1, j]
            row[f"Z[{j}]"] = None if np.isnan(v) else float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    with st.expander("Parent Height Data (inches)", expanded=False):
        st.dataframe(df, use_container_width=True)
    #Basically a debug table
    df_rows = []
    for i in range(A):
        for j in range(nz):
            df_rows.append({
                "Actuator":            i+1,
                "Slice":               j,
                "slope (in/in)":       float(vy[i, j]),
                "angle vs horiz (°)":   float(angle_vs_horizontal[i, j]),
                "disp_half (in)":      float(disp[i, j])
            })
    disp_half_df = pd.DataFrame(df_rows)
    with st.expander("Half-Displacement per Point", expanded=False):
        st.dataframe(disp_half_df, use_container_width=True)
    # Display new curves table
    df_rows = []
    for i in range(A):
        for kind, curve in (("top", top_curve), ("bottom", bot_curve)):
            row = {"Actuator": i+1, "Type": kind}
            for j in range(nz):
                row[f"Z[{j}]"] = float(curve[i, j])
            df_rows.append(row)
    new_curves_df = pd.DataFrame(df_rows)
    with st.expander("Displaced Curves Table", expanded=False):
        st.dataframe(new_curves_df, use_container_width=True)
    #3D viewers 
    # Parent Curves 3D Visualizer 
    with st.expander("Parent Actuator Curves in 3D", expanded=False):
        fig = go.Figure()
        A    = len(xs_in)
        samp = np.arange(nz)
    
        for i in range(A):
            fig.add_trace(go.Scatter3d(
                x=np.full(nz, i+1),      # actuator number
                y=samp,                   # slice index
                z=H_in[i, :],             # height in inches
                mode='lines',
                line=dict(width=4),
                name=f"Act {i+1}"
            ))
    
        fig.update_layout(
            scene=dict(
                xaxis_title="Actuator #",
                xaxis=dict(autorange="reversed"),
                yaxis_title="Slice #",
                zaxis_title="Height (in)"
            ),
            height=700,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # 3D Viewer: plot top & bottom displaced curves
    fig3d = go.Figure()
    samp = np.arange(nz)
    for i in range(A):
        fig3d.add_trace(go.Scatter3d(
            x=np.full(nz, i+1), y=samp, z=top_curve[i, :],
            mode='lines', name=f"Act {i+1} Top"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=np.full(nz, i+1), y=samp, z=bot_curve[i, :],
            mode='lines', name=f"Act {i+1} Bottom"
        ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Actuator #",
            xaxis=dict(autorange="reversed"),
            yaxis_title="Slice #",
            zaxis_title="Height (in)"
        ),
        height=600, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig3d, use_container_width=True)
