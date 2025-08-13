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
height map and a fixed span ``eff_span`` representing the vertical
separation between the paired actuators.  The resulting displacement arrays
``top_curve`` and ``bot_curve`` are then expressed either in world space
(when ``zero_disp`` is unchecked) or as offsets relative to these P₀
positions and along the actuator's travel direction (when ``zero_disp`` is
checked).

In addition to this geometric transformation, the app provides full unit
flexibility: every numeric input is accompanied by a unit selector (m, mm,
cm, in, ft for lengths; corresponding per‑second units for velocities).  All
values are converted internally to metres (Blender's default unit) so the
output table is always in metres regardless of the input units.  This
conveniently allows you to work with whatever units your mesh and machine
dimensions are expressed in.

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

# -----------------------------------------------------------------------------
# Unit conversion factors for length units to metres.  These are used to
# convert all user inputs into Blender's default units (metres).  The keys
# correspond to the options presented to the user in the unit dropdowns.  For
# velocity units (e.g., 'cm/s'), we derive the factor by stripping the '/s' and
# using the corresponding length factor.
unit_factors = {
    'm': 1.0,
    'mm': 1e-3,
    'cm': 1e-2,
    'in': 0.0254,
    'ft': 0.3048,
}

# Length and velocity unit options presented to the user.  All numeric inputs
# accept a unit dropdown; the selected unit is converted to metres using
# ``unit_factors`` above.  Velocity units are expressed as length per second.
length_unit_options = ['m', 'mm', 'cm', 'in', 'ft']
velocity_unit_options = ['m/s', 'mm/s', 'cm/s', 'in/s', 'ft/s']


st.set_page_config(layout="wide")
st.title("Heightmap Actuator Extractor & 3D Curves (Metres)")

# -----------------------------------------------------------------------------
# 1) MODEL INPUT
#
# Upload a planar surface mesh (OBJ or STL).  We assume the mesh is authored in
# the same units as your machine dimensions (for example, inches if you select
# inches for the width/height inputs).  All internal computations convert
# geometry and user‑supplied values to metres, Blender's default unit system,
# before assembling the actuator control table.  If your mesh is not in the
# same unit as your input values, the spacing between actuators may be
# incorrect.
uploaded = st.file_uploader(
    "Upload planar geometry (OBJ/STL)",
    type=["stl", "obj"],
    help=(
        "Upload the height‑map surface as an OBJ or STL file.  The mesh should be"
        " expressed in the same unit system as your numeric inputs (e.g. inches,"
        " millimetres, metres).  All results will be output in metres."
    ),
)

# ── 2) MACHINE BOUNDS & ACTUATORS (Imperial) ─────────────────
st.markdown("### Machine Bounds & Actuators")

# Create two primary columns for layout.  All numeric inputs are paired with a
# corresponding unit selectbox.  Help strings clarify the meaning of each
# parameter and emphasise that units are user‑selectable.
b1, b2 = st.columns(2)
with b1:
    # Bounds Width (Emax) and its unit
    width_cols = st.columns([3, 1])
    width_val = width_cols[0].number_input(
        "Bounds Width",
        value=62.0,
        min_value=0.0,
        help=(
            "Total width of the machine's working area (Emax) measured along"
            " the X‑axis.  This determines the physical spacing between actuators."
        ),
        key="width_val",
    )
    width_unit = width_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=3,  # default to inches for legacy defaults
        key="width_unit",
        help="Select the unit for the width value.  All results will be converted to metres.",
    )

    # Bounds Height (also Emax) and its unit
    height_cols = st.columns([3, 1])
    height_val = height_cols[0].number_input(
        "Bounds Height",
        value=14.0,
        min_value=0.0,
        help=(
            "Vertical height of the machine's working area (Emax) measured along"
            " the Y‑axis.  This corresponds to the intersection space between"
            " top and bottom actuators."
        ),
        key="height_val",
    )
    height_unit = height_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=3,
        key="height_unit",
        help="Select the unit for the height value.  All results will be converted to metres.",
    )

    # Number of actuator pairs (dimensionless)
    num_actuators = st.number_input(
        "Number of Actuator Pairs",
        min_value=1,
        value=7,
        step=1,
        help=(
            "Number of paired actuators arranged along the X‑axis.  Each pair"
            " consists of a top actuator (directed downwards) and a bottom"
            " actuator (directed upwards)."
        ),
    )

    # Z‑Resolution (number of slices along Z)
    nz = st.number_input(
        "Z‑Resolution (# slices)",
        value=1000,
        min_value=1,
        step=1,
        help=(
            "Number of sampling slices along the Z‑axis (depth).  Higher values"
            " provide a smoother height map but increase computation time."
        ),
    )

with b2:
    # Composite thickness and unit (part of eff_span)
    comp_cols = st.columns([3, 1])
    comp_thickness = comp_cols[0].number_input(
        "Composite Thickness",
        value=1.0,
        min_value=0.0,
        help=(
            "Thickness of the composite material situated between the wheel"
            " and the heating element.  Contributes to the effective span"
            " between top and bottom actuators."
        ),
        key="comp_thickness",
    )
    comp_unit = comp_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=3,
        key="comp_unit",
        help="Select the unit for the composite thickness value.",
    )

    # Wheel diameter and unit (part of eff_span)
    wheel_cols = st.columns([3, 1])
    wheel_diam = wheel_cols[0].number_input(
        "Wheel Diameter",
        value=1.5,
        min_value=0.0,
        help=(
            "Diameter of the wheel used in the bending machine.  Included in"
            " the effective span calculation."
        ),
        key="wheel_diam",
    )
    wheel_unit = wheel_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=3,
        key="wheel_unit",
        help="Select the unit for the wheel diameter value.",
    )

    # Heating element thickness and unit (part of eff_span)
    heat_cols = st.columns([3, 1])
    heat_k = heat_cols[0].number_input(
        "Heating Element Thickness",
        value=0.019685,
        min_value=0.0,
        help=(
            "Thickness of the heating element sandwiched in the composite. "
            "Used twice in the effective span calculation (top and bottom)."
        ),
        key="heat_k",
    )
    heat_unit = heat_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=3,
        key="heat_unit",
        help="Select the unit for the heating element thickness value.",
    )

    # Checkbox to shift zero (no unit)
    shift_zero = st.checkbox(
        "Re‑zero at mid‑height",
        value=False,
        help=(
            "If checked, subtract half the mesh's Y bounding dimension from all"
            " height values.  This can help centre the height map around zero."
        ),
    )

    # Checkbox for relative movement (express offsets rather than absolute)
    zero_disp = st.checkbox(
        "Relative Movement",
        value=False,
        help=(
            "If checked, actuator positions are expressed as offsets from their"
            " starting positions (P₀) along their travel directions.  The"
            " resulting values correspond to actuator displacement rather"
            " than world coordinates."
        ),
    )

    # Additional actuator parameters (all convertible to metres or metres per second)
    # Minimum actuator extension (Emin) and unit
    min_cols = st.columns([3, 1])
    min_extension = min_cols[0].number_input(
        "Minimum Extension Emin",
        value=0.0,
        min_value=0.0,
        help=(
            "Minimum extension of the actuator (Emin).  Purely descriptive:"
            " does not affect the computed height map but appears in the"
            " output table for reference."
        ),
        key="min_extension",
    )
    min_unit = min_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=0,
        key="min_unit",
        help="Select the unit for the minimum extension value.",
    )

    # Shaft radius r and unit
    shaft_cols = st.columns([3, 1])
    shaft_radius = shaft_cols[0].number_input(
        "Shaft Radius r",
        value=0.0,
        min_value=0.0,
        help=(
            "Radius of the actuator shaft (r).  Appears in the output table and"
            " does not influence the height computation."
        ),
        key="shaft_radius",
    )
    shaft_unit = shaft_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=0,
        key="shaft_unit",
        help="Select the unit for the shaft radius value.",
    )

    # Body radius R and unit
    body_cols = st.columns([3, 1])
    body_radius = body_cols[0].number_input(
        "Body Radius R",
        value=0.0,
        min_value=0.0,
        help=(
            "Radius of the actuator body (R).  Appears in the output table."
        ),
        key="body_radius",
    )
    body_unit = body_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=0,
        key="body_unit",
        help="Select the unit for the body radius value.",
    )

    # Body length L and unit
    length_cols = st.columns([3, 1])
    body_length = length_cols[0].number_input(
        "Body Length L",
        value=0.0,
        min_value=0.0,
        help=(
            "Length of the actuator body (L).  Appears in the output table."
        ),
        key="body_length",
    )
    length_unit_select = length_cols[1].selectbox(
        "Unit",
        length_unit_options,
        index=0,
        key="length_unit",
        help="Select the unit for the body length value.",
    )

    # Maximum actuator velocity Vmax and unit
    vel_cols = st.columns([3, 1])
    max_velocity = vel_cols[0].number_input(
        "Maximum Actuator Velocity Vmax",
        value=0.0,
        min_value=0.0,
        help=(
            "Maximum actuator velocity (Vmax).  Appears in the output table"
            " and does not affect the height map."
        ),
        key="max_velocity",
    )
    velocity_unit = vel_cols[1].selectbox(
        "Unit",
        velocity_unit_options,
        index=0,
        key="velocity_unit",
        help=(
            "Select the unit for the maximum actuator velocity value (e.g."
            " metres per second).  Only the length component of the unit"
            " (m, mm, cm, in, ft) is used for conversion to metres."
        ),
    )

# ── 3) LAUNCH PROCESS ────────────────────────────────────────
if st.button("Process"):
    # If no mesh -> Error message
    if not uploaded:
        st.error("Please upload a model file.")
        st.stop()
        
    # -----------------------------------------------------------------------
    # Convert all user inputs into metres.  Each numeric field has an
    # associated unit selection; we use ``unit_factors`` to scale the values
    # accordingly.  For the maximum velocity, only the length component of
    # ``velocity_unit`` is used (e.g. ``mm/s`` → ``mm`` → 0.001 m).
    bounds_width_m  = width_val * unit_factors[width_unit]
    bounds_height_m = height_val * unit_factors[height_unit]
    comp_thickness_m = comp_thickness * unit_factors[comp_unit]
    wheel_diam_m     = wheel_diam * unit_factors[wheel_unit]
    heat_k_m         = heat_k * unit_factors[heat_unit]
    min_extension_m  = min_extension * unit_factors[min_unit]
    shaft_radius_m   = shaft_radius * unit_factors[shaft_unit]
    body_radius_m    = body_radius * unit_factors[body_unit]
    body_length_m    = body_length * unit_factors[length_unit_select]
    # Extract base length unit for velocity (strip '/s')
    vel_length_unit = velocity_unit.split("/")[0]
    max_velocity_m   = max_velocity * unit_factors[vel_length_unit]
    
    # Load mesh (inches assumed)
    mesh = trimesh.load(BytesIO(uploaded.read()),
                        file_type=uploaded.name.split('.')[-1])
    
    # Error if mesh empty
    if mesh.is_empty:
        st.error("Mesh is empty.")
        st.stop()
        
    # 4) Actuator X positions in mesh‑space.  ``xs_in`` gives the physical
    # positions of the actuators along the width, measured in metres.  These
    # positions are then mapped into the mesh's X coordinate range via
    # ``xs_mesh`` below.
    if num_actuators > 1:
        xs_in = np.linspace(0.0, bounds_width_m, num_actuators)
    else:
        xs_in = np.array([0.0])
        
    # Construct bounding box
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    xs_mesh = xmin + (xs_in / bounds_width_m) * (xmax - xmin)
    
    # 5) Z‑slice positions (mesh units).  These derive directly from the
    # mesh's Z bounds and are assumed to be in the same unit system as the
    # mesh.  The resulting heights are converted to metres later based on
    # ``width_unit``.
    zs = np.linspace(zmin, zmax, int(nz))
    
    # 6) Nudge at the edges.  Before, the rays were not hitting the edges of the
    # mesh so we are scooting the edge slices in here.
    if num_actuators > 1:
        span    = xmax - xmin
        spacing = span / (num_actuators - 1)
        eps     = spacing * 0.01
        xs_mesh[0]  = xmin + eps
        xs_mesh[-1] = xmax - eps
        
    # 7) Ray‑cast heights.  For each actuator position, you fire a line of
    # sight straight down from just above the mesh through each Z slice,
    # detect where it first strikes the surface, and record that Y (height) in
    # your 2D array.  Wherever no hit occurs, the entry stays NaN, so you
    # can fill or interpolate later.  This yields a full height map organised
    # as H_in[actuator_index, slice_index].  The values are initially in the
    # mesh's units and converted to metres after the interpolation step.
    H_in = np.full((len(xs_mesh), int(nz)), np.nan)
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
            
    # 8) Optional mid‑height re‑zero
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
    # Convert height values from the mesh's unit system to metres.  We assume
    # the mesh uses the same unit as the width input; thus we apply the
    # conversion factor associated with ``width_unit``.  This ensures the
    # height map and actuator spacings are in the same physical units.
    H_in = H_in * unit_factors[width_unit]
    
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
    eff_span = (heat_k_m * 2.0) + wheel_diam_m + comp_thickness_m
    
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

        # Determine parameter values based on actuator index.  The mapping
        # corresponds to user-defined mechanical properties:
        #   - i=0 (first actuator pair): top parameter Emax is the bounds height;
        #     bottom parameter Emin is the minimum extension value.
        #   - i=1 (second actuator pair): top parameter r is the shaft radius;
        #     bottom parameter R is the body radius.
        #   - i=2 (third actuator pair): top parameter L is the body length;
        #     bottom parameter Vmax is the maximum actuator velocity.
        #   - i=3 (fourth actuator pair): top parameter W is the wheel diameter;
        #     bottom parameter D is the element thickness (heater element thickness).
        #   - i>=4: unused pairs with no specific parameters.
        if i == 0:
            param_value_top = bounds_height_m
            param_value_bottom = min_extension_m
        elif i == 1:
            param_value_top = shaft_radius_m
            param_value_bottom = body_radius_m
        elif i == 2:
            param_value_top = body_length_m
            param_value_bottom = max_velocity_m
        elif i == 3:
            param_value_top = wheel_diam_m
            param_value_bottom = heat_k_m
        else:
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
    # To avoid overwhelming the display, show only the first few slice
    st.dataframe(df_output)
