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
st.title("Heightmap Actuator Extractor & 3D Curves ah")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in mm)", type=["stl", "obj"])

# ── 2) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    #Prompt user input
    width_val      = st.number_input("Bounds Width (ft)", value=6.0)
    height_val     = st.number_input("Bounds Height (ft)", value=4.0)
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)
with b2:
    num_actuators = st.number_input("Number of Actuators", min_value=1, value=10, step=1)
    nz            = st.slider("Z-Resolution (# slices)", 10, 10_000, 200)
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
        
    # Convert bounds (ft→mm)
    bounds_width_mm  = width_val  * 304.8 
    bounds_height_mm = height_val * 304.8
    
    # Load mesh (mm assumed)
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    
    # Error if mesh empty
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()
        
    # X positions in mm
    if num_actuators > 1:
        xs_mm = np.linspace(0, bounds_width_mm, num_actuators)
    else:
        xs_mm = np.array([0.0])
        
    # Convert actuator X positions to inches
    xs_in = xs_mm / 25.4
    
    # Map into mesh coords
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_mm / bounds_width_mm) * (xmax - xmin)
    
    # Z slices
    zs = np.linspace(zmin, zmax, nz)
    
    # Nudge inwards for rays to hit edges properly
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps
        
    # Ray-cast heights (mm)
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
        
    # Smooth/spline-interpolate any remaining NaNs 
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
        
    # Convert outputs to Imperial
    H_in = H_mm / 25.4
    xs_in = xs_mm / 25.4
    A = len(xs_in)   # number of actuators

    # Heights table (inches) 
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

    # Parent Curves 3D Visualizer 
    st.subheader("Parent Actuator Curves in 3D")

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
    
    # The equation relating theta θ (angle between x axis and curve), the specified thickness k, of the frp and the 
    # displacment d (disance the vertical actuators need to add onto the original cuve to compansate for bending), is 
    # d=(k-kCos(θ))/(2Cos(θ)). To make things simpler on the code end, we will write the angle in terms of the slope 
    # of the tangent line m or dy/dz. d=(k-kCos(arctan(m)))/(2Cos(arctan(m))). Using trig identities this simplifies to
    # d=(1/2)k(Sqrt(1+m^2)-1). A beutiful formula that takes the desired thickness and spits out displacment compensation
    # for every point! From there, you simply add plus or minus 1/2 thickness+d to the parent curves uwu. 

    # 1) Compute physical slice spacing
    #   - dz_mm: horizontal distance between consecutive Z-slices, in millimeters (mm)
    #   - ds_in: horizontal distance per slice, in inches (in)
    dz_mm = (zmax - zmin) / (nz - 1)   # mm per slice
    ds_in = dz_mm / 25.4               # inches per slice
    
    # 2) Fit a cubic spline per actuator to obtain smooth derivative dH/ds
    #    - s_phys: physical coordinate along horizontal (Z) axis in inches
    #    - H_in[i, :] holds heights in inches
    s_phys = np.arange(nz) * ds_in      # inches along Z-axis
    vy = np.zeros_like(H_in)            # slope array (dimensionless: in/in)
    for i in range(A):
        # Fit exact cubic spline through (s_phys, H_in[i,:])
        spline = UnivariateSpline(s_phys, H_in[i, :], k=3, s=0)
        # Derivative dy/ds_phys at each slice (unitless)
        vy[i, :] = spline.derivative(n=1)(s_phys)
    
    # 3) Compute tangent angle relative to horizontal axis (in degrees)
    #    - tangent vector in (horizontal, vertical) plane = (Δs, ΔH) = (1, m)
    #    - arctan2(vertical_component, horizontal_component) returns radians; convert to degrees
    grid = vy  # using vy for clarity
    angle_vs_horizontal = np.degrees(np.arctan2(vy, 1.0))  # degrees
    
    # 4) Compute displacement using angle-based formula: d = (k/cos(θ) - k)/2
    disp = (comp_thickness / np.cos(np.radians(angle_vs_horizontal)) - comp_thickness) / 2.0  # inches
    
    # 5) Build and display table: Actuator, Slice, slope, angle vs horizontal, and displacement
    df_rows = []
    for i in range(A):
        for j in range(nz):
            df_rows.append({
                "Actuator":            i+1,
                "Slice":               j,
                "slope (in/in)":       float(round(vy[i, j],      4)),
                "angle vs horiz (°)":   float(round(angle_vs_horizontal[i, j], 2)),
                "disp (in)":           float(round(disp[i, j],     4))
            })
    angle_disp_df = pd.DataFrame(df_rows)
    
    # Display in Streamlit under expander
    st.subheader("Tangent Angle vs Horizontal & Displacement")
    st.dataframe(angle_disp_df, use_container_width=True)



