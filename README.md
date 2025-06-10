# Heightmap-Actuator-Extractor-3D-Curves-App

A Streamlit-powered GUI tool to  
1. Load a 3D mesh (STL/OBJ)  
2. Sample its surface along evenly-spaced “actuator” X-positions  
3. Cast vertical rays to build a height-map (in machine steps)  
4. Export per-actuator height data and machine parameters  
5. Visualize each actuator’s profile as a 3D curve  

---

## 🔍 Features

- **Mesh input**: supports STL or OBJ files in your CAD units (mm).  
- **Machine parameters**: default values (pull speed, temperatures, forces, thicknesses, bounds, etc.) with flexible unit selectors.  
- **Actuator sampling**: choose number of actuators, bounds width & height, Z-resolution, and steps-per-mm conversion.  
- **Edge-handling**: “nudges” first/last actuators inward to ensure boundary rays hit the mesh.  
- **Exportable JSON**: download `params.json` with all machine settings.  
- **Height data table**: step counts for each actuator at each Z-slice.  
- **Interactive 3D plot**: each actuator’s height profile rendered as a 3D curve.

---

## 🚀 Quick Start

### 1. Clone & enter project

```bash
git clone https://github.com/RayCate1/Heightmap-Actuator-Extractor-3D-Curves-App
cd heightmap-actuator-extractor
