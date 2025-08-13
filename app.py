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

"""
Heightmap Actuator Extractor & 3D Curves (Modified)
--------------------------------------------------

This Streamlit application extracts a height map from a planar geometry mesh and
computes actuator displacements for a bending machine.  The original version
produced absolute world‐space positions for the top and bottom actuators,
with an option to convert the curves to relative movement by subtracting the
first slice.  However, the machine expects actuator movements to be expressed
relative to each actuator's starting position (P₀) and along its own travel
direction.  In particular, the bottom actuators move along the positive
vertical axis, while the top actuators move along the negative vertical axis.

This modified implementation introduces a consistent parameterisation for the
actuators using **start positions** and **direction vectors**.  For each
actuator pair, the baseline (P₀) positions are derived from the measured
height map `H_in` and a fixed span `eff_span` representing the vertical
separation between the paired actuators.  The resulting displacement arrays
`top_curve` and `bot_curve` are then expressed either in world space (when
``zero_disp`` is unchecked) or as offsets relative to these P₀ positions and
along the actuator's travel direction (when ``zero_disp`` is checked).

The core logic is contained within the ``if st.button("Process")`` block and
can be reused or adapted for different mesh inputs or machine configurations.
"""

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
st.title("Heightmap Actuator Extractor & 3D Curves (Meters)")

# ── 1) MODEL INPUT ─────────────────────────────────────────
uploaded = st.file_uploader("Upload planar geometry (OBJ/STL in inches)", type=["stl", "obj"])

# ── 2) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")
b1, b2 = st.columns(2)
with b1:
    # Prompt user input
    width_val      = st.number_input("Bounds Width (in)", value=62)
    height_val     = st.number_input("Bounds Height (in)", value=14)
    num_actuators  = st.number_input("Number of Actuator Pairs", min_value=1, value=7, step=1)
    nz             = st.number_input("Z-Resolution (# slices)", value=1000)
with b2:
    comp_thickness = st.number_input("Composite Thickness (in)", value=1.0)
    wheel_diam     = st.number_input("Wheel Diameter (in)", value=1.5)
    heat_k         = st.number_input("Heating Element Thickness (in)", value=0.019685)
    # Checkbox to shift zero
    shift_zero = st.checkbox(
        "Re-zero at mid-height (shift all heights down by half the bounding-box Y)", 
        value=False
    )
    # Checkbox for relative movement
    zero_disp = st.checkbox(
        "Relative Movement (actuators start at zero; values are offsets from P₀)",
        value=False
    )

# ── 3) LAUNCH PROCESS ────────────────────────────────────────
if st.button("Process"):
    # If no mesh -> Error message
    if not uploaded:
        st.error("Please upload a model file.")
        st.stop()
        
    # 2) Bounds & actuator X positions
    #    Convert all user‐supplied lengths from inches to meters so that the
    #    entire pipeline runs in Blender's default units (meters).  Many
    #    geometries are authored in inches, so if the uploaded mesh is in
    #    inches, this conversion normalises it to metres.  If the mesh is
    #    already in metres, the conversion still preserves ratios because
    #    widths and heights are scaled consistently.
    inch_to_m = 0.0254
    bounds_width_m  = width_val * inch_to_m
    bounds_height_m = height_val * inch_to_m
    
    # Load mesh (inches assumed)
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    
    # Error if mesh empty
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()
        
    # 4) Actuator X positions in mesh-space (metres)
    if num_actuators > 1:
        xs_in = np.linspace(0, bounds_width_m, num_actuators)
    else:
        xs_in = np.array([0.0])
        
    # Construct bounding box
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_in / bounds_width_m) * (xmax - xmin)
    
    # 5) Z-slice positions (mesh units).  These are unaffected by unit
    # conversions because they derive from the mesh directly.  We assume
    # ``zmin`` and ``zmax`` are already in metres if the mesh is authored
    # in Blender's default units.
    zs = np.linspace(zmin, zmax, nz)
    
    # 6) Nudge at the edges.  Before, the rays were not hitting the edges of the
    # mesh so we are scooting the edge slices in here.
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps
        
    # 7) Ray-cast heights directly in inches.  For each actuator position,
    # you “fire” a line of sight straight down from just above the mesh through
    # each Z-slice, detect where it first strikes the surface, and record that Y
    # (height) in your 2D array.  Wherever no hit occurs, the entry stays NaN,
    # so you can fill or interpolate later.  This gives you a full height map in
    # inches, organised as H_in[actuator_index, slice_index]
    H_in = np.full((len(xs_mesh), nz), np.nan)
    for i, x0 in enumerate(xs_mesh):
        origins = np.column_stack([
            np.full(nz, x0),
            np.full(nz, ymax + (ymax - ymin) * 0.1),
            zs
        ])
        dirs = np.tile([0.0, -1.0, 0.0], (nz, 1))
        locs, idxs, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idxs):
            H_in[i, idxs] = locs[:, 1]
            
    # 8) Optional mid-height re-zero
    if shift_zero:
        H_in -= (ymax - ymin) / 2.0
    
    # 9) Smooth/spline-interpolate any remaining NaNs
    for i in range(len(xs_mesh)):
        row   = H_in[i, :]
        idx   = np.arange(nz)
        valid = ~np.isnan(row)
        if valid.sum() >= 4:
            spline      = UnivariateSpline(idx[valid], row[valid], k=3, s=0)
            H_in[i, :] = spline(idx)
        else:
            # Fall back to linear interpolation if too few valid points
            H_in[i, :] = pd.Series(row).interpolate(method='linear', limit_direction='both').values

    # ---------------------------------------------------------------------------
    # Convert heights from mesh units (presumed inches) to metres.  If the mesh
    # is authored in metres, multiplying by ``inch_to_m`` will scale values
    # proportionally; however, because ``xs_in`` and ``width_val`` have already
    # been scaled to metres, the ratio between geometry and actuator positions
    # remains correct.
    H_in = H_in * inch_to_m
    
    # ---------------------------------------------------------------------------
    # Geometry processing for actuator displacements
    #
    # The equation relating the tangent angle θ (angle between the X axis and
    # surface slope), the original actuator spacing k (eff_span), and the
    # displacement d is d = k / cos(θ).  We apply ±½ d to the surface
    # heights to obtain the top and bottom curves.  See the docstring for
    # further detail.
    
    # 1) Compute physical actuator spacing along X-axis (metres)
    A     = len(xs_in)   # number of actuators
    s_act = xs_in        # actuator positions along the width (metres)
    
    # 2) Fit a cubic spline per Z-slice to obtain smooth derivative dH/dx
    # slopes_x[i, j] = ∂H/∂x at actuator i for slice j
    slopes_x = np.zeros_like(H_in)  # shape (A, nz)
    for j in range(nz):
        H_slice = H_in[:, j]      # heights at slice j across A actuators
        # If enough points, fit a cubic spline; else use finite differences
        if A >= 4:
            spline = UnivariateSpline(s_act, H_slice, k=3, s=0)
            slopes_x[:, j] = spline.derivative(n=1)(s_act)
        else:
            slopes_x[:, j] = np.gradient(H_slice, s_act)
    
    # 3) Compute tangent angle relative to X-axis (in degrees)
    #    - Tangent vector in (horizontal, vertical) plane = (Δx, ΔH) = (1, m)
    #    - arctan2(vertical_component, horizontal_component) returns radians; convert to degrees
    angle_vs_x = np.degrees(np.arctan2(slopes_x, 1.0))  # shape (A, nz)
    
    # 4) Determine effective span k (metres) between components
    # Convert component dimensions from inches to metres before summing
    comp_thickness_m = comp_thickness * inch_to_m
    wheel_diam_m     = wheel_diam * inch_to_m
    heat_k_m         = heat_k * inch_to_m
    eff_span = (heat_k_m * 2) + wheel_diam_m + comp_thickness_m
    
    # 5) Compute displacement: full d = k / cos(θ), then half-displacement (metres)
    d_full    = eff_span / np.cos(np.radians(angle_vs_x))  # total displacement (metres)
    disp_half = d_full / 2.0                               # half-displacement to apply (metres)
    
    # 6) Build new top/bottom curves using pointwise half-displacement
    #    New curves: H_top = H_in + disp_half, H_bot = H_in - disp_half (metres)
    #
    # Optionally re-express these curves as offsets relative to the actuators' starting
    # positions (P₀) and along their travel directions.  When ``zero_disp`` is True,
    # the outputs represent how far each actuator should move from its baseline rather
    # than absolute world coordinates.
    
    # Compute absolute positions first
    top_curve = H_in + disp_half
    bot_curve = H_in - disp_half
    
    # If relative movement requested, convert to offsets
    if zero_disp:
        # Baseline (P₀) positions for bottom actuators (metres)
        baseline_bottom = H_in
        
        # Baseline (P₀) positions for top actuators
        baseline_top = H_in + eff_span
        
        # Offsets relative to P₀ (metres)
        offset_bottom = bot_curve - baseline_bottom  # negative means moved down; positive means up
        offset_top_raw = top_curve - baseline_top    # positive means moved up in +Z if ``baseline_top`` is along Z
        offset_top_dir = -offset_top_raw             # convert to movement along −Z (downwards)
        
        # Replace the curves with these offsets so downstream logic uses relative movement
        bot_curve = offset_bottom
        top_curve = offset_top_dir
    
    # At this point, ``bot_curve`` and ``top_curve`` contain either absolute
    # positions (if zero_disp=False) or offsets relative to each actuator's
    # baseline along the actuator's travel direction (if zero_disp=True).  Further
    # processing—such as exporting these arrays, plotting, or generating G-code—
    # should operate on ``bot_curve`` and ``top_curve`` accordingly.
    
    # ---------------------------------------------------------------------------
    # Assemble output in the format specified by the user.  Each actuator
    # generates two rows: one for the top actuator (is_top=1) and one for
    # the bottom actuator (is_top=0).  The columns include parameter
    # metadata, start positions, direction vectors, and the computed
    # height values for each slice.  Units are metres throughout.
    
    # Determine absolute positions (metres) for top and bottom actuators
    if zero_disp:
        # Convert offsets back to absolute positions for reporting.  See
        # comments above for sign conventions: ``top_curve`` and
        # ``bot_curve`` currently hold offsets.
        baseline_bottom = H_in
        baseline_top = H_in + eff_span
        bottom_positions_m = baseline_bottom + bot_curve
        top_positions_m = baseline_top - top_curve  # invert sign because top_curve holds downward offsets
    else:
        bottom_positions_m = bot_curve
        top_positions_m = top_curve

    # X positions for actuators (metres)
    xs_in_m = xs_in

    # Define parameter names for the first four actuator pairs; beyond that, use 'N/A'
    param_mapping = [
        ('Emax', 'Emin'),
        ('r', 'R'),
        ('L', 'Vmax'),
        ('W', 'D'),
    ]

    rows = []
    # Build slice column names as strings ('0', '1', '2', ..., str(nz-1))
    slice_cols = [str(j) for j in range(nz)]
    for i in range(A):
        # Determine parameter names
        if i < len(param_mapping):
            param_top, param_bottom = param_mapping[i]
        else:
            param_top = param_bottom = 'N/A'

        # Placeholder parameter values; users can customise these as needed
        param_value_top = 0.0
        param_value_bottom = 0.0

        # Top actuator row
        row_top = {
            'parameter_name': param_top,
            'parameters': param_value_top,
            'actuator': i + 1,
            'is_top': 1,
            'x_0': xs_in_m[i],
            'y_0': 0.0,
            'z_0': 0.0,
            'd_x': 0.0,
            'd_y': 0.0,
            'd_z': -1.0,
        }
        # Add slice values for top actuator
        for j, col in enumerate(slice_cols):
            row_top[col] = float(top_positions_m[i, j])
        rows.append(row_top)

        # Bottom actuator row
        row_bottom = {
            'parameter_name': param_bottom,
            'parameters': param_value_bottom,
            'actuator': i + 1,
            'is_top': 0,
            'x_0': xs_in_m[i],
            'y_0': 0.0,
            'z_0': 0.0,
            'd_x': 0.0,
            'd_y': 0.0,
            'd_z': 1.0,
        }
        # Add slice values for bottom actuator
        for j, col in enumerate(slice_cols):
            row_bottom[col] = float(bottom_positions_m[i, j])
        rows.append(row_bottom)

    df_output = pd.DataFrame(rows)
    
    # Show a preview of the DataFrame in the Streamlit app
    st.markdown("#### Preview of Actuator Control Table (metres)")
    # To avoid overwhelming the display, show only the first few slices
    preview_cols = ['parameter_name', 'parameters', 'actuator', 'is_top', 'x_0', 'y_0', 'z_0', 'd_x', 'd_y', 'd_z'] + slice_cols[:min(3, nz)]
    st.dataframe(df_output)
