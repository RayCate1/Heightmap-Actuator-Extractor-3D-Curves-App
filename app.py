# PPPPPP  RRRRRR  OOOOOO  JJJJJJ  EEEEEE  CCCCCC  TTTTTT
# P    P  R    R  O    O      JJ  E       C         TT    
# PPPPPP  RRRRRR  O    O      JJ  EEE     C         TT    
# P       R   R   O    O  J   JJ  E       C         TT    
# P       R    R  OOOOOO   JJJJ   EEEEEE  CCCCCC    TT    

# FFFFFF  RRRRRR  EEEEEE  DDDDD 
# F       R    R  E       D    D
# FFFF    RRRRRR  EEE     D    D
# F       R   R   E       D    D
# F       R    R  EEEEEE  DDDDD 

import streamlit as st
import numpy as np
import trimesh
import pandas as pd
import json
import plotly.graph_objects as go
from io import BytesIO
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import open3d as o3d
import pydeck as pdk
from pydeck.types import CoordinateSystem
import tempfile

st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in inches)", type=["stl", "obj"])
scan_file = st.file_uploader("Upload Scan Point Cloud", type=["ply","pcd","xyz"])
mesh_file = st.file_uploader("Upload CAD Mesh (STL/OBJ/PLY)", type=["stl","obj","ply"])
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
    # Checkbox for millimeter units (not currently implemented)
    #     mm_units = st.checkbox(
    #     "millimeter units (not operational)",
    #     value=False
    # )
    # Checkbox for intermediate smoothing (trade acuracy to the file to smooth bumps out) (not currently implemented)
    #     mm_units = st.checkbox(
    #     "intermediate smoothing (not operational)",
    #     value=False
    # )
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
        
    #Construct bounding box
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_in / bounds_width_in) * (xmax - xmin)
    
    # 5) Z‐slice positions (inches)
    zs = np.linspace(zmin, zmax, nz)
    
    # 6) Nudge at the edges. Before, the rays where not hitting the edges of the mesh so we are scooting the edge slices in here.
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps
        
    # 7) Ray‐cast heights directly in inches.For each actuator position, you “fire” a line of sight straight down from just above the mesh through each Z-slice, detect where it first strikes the surface, and record that Y (height) in your 2D array. Wherever no hit occurs, the entry stays NaN, so you can fill or interpolate later.This gives you a full height map in inches, organized as H_in[actuator_index, slice_index]
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

    #Right now we have data that is completely filled in but is still a bit choppy because the mesh used is not a B spline. we will add an opion at the top foe extra smoothing. The mesh a set of triangles doing its best to represent a B spline. What we need to do to fix this is spline‐interpolate all of the data and then sample these splines at the sample rate.

    
    # The equation relating theta θ (angle between x axis and curve), the distance between axles k, of the frp and the displacment d    (disance the vertical actuators need to add onto the original cuve to compansate for bending), is d=k/Cos(θ). From there, you simply add plus or minus 1/2 d to the parent curves. 
    
    # 1) Compute physical actuator spacing along X-axis (inches)
    # xs_in: actuator X positions in inches (already defined)
    A = len(xs_in)   # number of actuators
    s_act = xs_in    # shape (A,)
    
    # 2) Fit a cubic spline per Z-slice to obtain smooth derivative dH/dx
    #    - For each slice j, H_in[:, j] gives heights across actuators at that Z.
    #    - slopes_x[i,j] = ∂H/∂x at actuator i for slice j.
    slopes_x = np.zeros_like(H_in)  # shape (A, nz)
    for j in range(nz):
        H_slice   = H_in[:, j]      # heights at slice j across A actuators
        # If enough points, fit a cubic spline; else use finite differences
        if A >= 4:
            spline      = UnivariateSpline(s_act, H_slice, k=3, s=0)
            slopes_x[:, j] = spline.derivative(n=1)(s_act)
        else:
            slopes_x[:, j] = np.gradient(H_slice, s_act)
    
    # 3) Compute tangent angle relative to X-axis (in degrees)
    #    - Tangent vector in (horizontal, vertical) plane = (Δx, ΔH) = (1, m)
    #    - arctan2(vertical_component, horizontal_component) returns radians; convert to degrees
    angle_vs_x = np.degrees(np.arctan2(slopes_x, 1.0))  # shape (A, nz)
    
    # 4) Determine effective span k (inches) between components
    eff_span = (heat_k * 2) + (wheel_radius * 2) + comp_thickness
    
    # 5) Compute displacement: full d = k / cos(θ), then half-displacement
    d_full    = eff_span / np.cos(np.radians(angle_vs_x))  # total displacement (inches)
    disp_half = d_full / 2.0                               # half-displacement to apply (inches)
    
    # 6) Build new top/bottom curves using pointwise half-displacement
    #    New curves: H_top = H_in + disp_half, H_bot = H_in - disp_half
    if zero_disp:
        top_curve = H_in + disp_half
        bot_curve = H_in - disp_half
        # zero-relative: subtract each actuator's starting height afterward
        top_curve -= top_curve[:, 0][:, None]
        bot_curve -= bot_curve[:, 0][:, None]
    else:
        top_curve = H_in + disp_half
        bot_curve = H_in - disp_half
















    st.title("📐 Scan vs. CAD Alignment Viewer (Trimesh ICP)")
    
    if scan_file and mesh_file:
        # Save temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=scan_file.name) as tmp:
            tmp.write(scan_file.getbuffer())
            scan_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=mesh_file.name) as tmp:
            tmp.write(mesh_file.getbuffer())
            mesh_path = tmp.name
    
        # Load scan and mesh
        scan_data = trimesh.load(scan_path)
        pts_scan = np.asarray(scan_data.vertices) if hasattr(scan_data, 'vertices') else np.loadtxt(scan_path)
        mesh = trimesh.load(mesh_path)
        pts_mesh, _ = trimesh.sample.sample_surface(mesh, 500000)
    
        # ICP registration
        matrix, _ = trimesh.registration.icp(
            pts_scan,
            pts_mesh,
            max_iterations=50,
            tolerance=1e-5
        )
        # Apply transform
        ones = np.ones((pts_scan.shape[0],1))
        pts_hom = np.hstack([pts_scan, ones])
        pts_aligned = (matrix @ pts_hom.T).T[:,:3]
    
        # Prepare data for PyDeck
        df_scan = pd.DataFrame(pts_aligned, columns=["x","y","z"])
        vertices = np.asarray(mesh.vertices)
        faces    = np.asarray(mesh.faces)
    
        mesh_layer = pdk.Layer(
            "MeshLayer",
            data=[{"positions": vertices.tolist(), "indices": faces.tolist()}],
            get_color=[180, 100, 200],
            opacity=0.4,
            wireframe=True,
            coordinate_system=CoordinateSystem.CARTESIAN
        )
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_scan,
            get_position=["x","y","z"],
            get_color=[255,0,0],
            get_radius=0.002,
            coordinate_system=CoordinateSystem.CARTESIAN
        )
    
        center = vertices.mean(axis=0)
        view_state = pdk.ViewState(
            latitude=center[1],
            longitude=center[0],
            zoom=0,
            pitch=45,
            bearing=0
        )
    
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[mesh_layer, scatter_layer]
        ))

























    
    #DISPLAY STUFF for streamlit app
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
                "Actuator":        i+1,
                "Slice":           j,
                "slope_x (in/in)": float(round(slopes_x[i, j], 4)),
                "angle vs X (°)":   float(round(angle_vs_x[i, j], 2)),
                "disp_half (in)":   float(round(disp_half[i, j], 4))
            })
    tangent_df = pd.DataFrame(df_rows)
    with st.expander("Tangent Angle vs X & Half-Displacement", expanded=False):
        st.dataframe(tangent_df, use_container_width=True)
        
    # Display new curves table
    # 6) Build and display table: Actuator, Slice, slope_x, angle_vs_x, half-displacement
    df_rows = []
    for i in range(A):
        for kind, curve in (("top", top_curve), ("bottom", bot_curve)):
            row = {"Actuator": i+1, "Type": kind}
            for j in range(nz):
                row[f"Z[{j}]"] = float(round(curve[i, j], 4))
            df_rows.append(row)
    new_curves_df = pd.DataFrame(df_rows)
    with st.expander("Displaced Curves Table (Top & Bottom)", expanded=False):
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

st.subheader("Rcate3@vols.utk.edu")
