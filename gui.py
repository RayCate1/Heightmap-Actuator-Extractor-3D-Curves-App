#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import trimesh

class SimpleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Heightmap Actuator Extractor")
        self.geometry("500x300")

        # Bounds Width
        tk.Label(self, text="Bounds Width:").place(x=20, y=20)
        self.width_var = tk.StringVar()
        tk.Entry(self, textvariable=self.width_var).place(x=150, y=20, width=100)

        # Number of Actuators
        tk.Label(self, text="Number of Actuators:").place(x=20, y=60)
        self.num_var = tk.StringVar()
        tk.Entry(self, textvariable=self.num_var).place(x=150, y=60, width=100)

        # Load Mesh button
        tk.Button(self, text="Load STL/OBJ", command=self.load_mesh)\
          .place(x=20, y=100, width=120)

        # Process button
        tk.Button(self, text="Process", command=self.process)\
          .place(x=150, y=100, width=120)

        # Text output
        self.out = tk.Text(self, wrap="none")
        self.out.place(x=20, y=140, width=460, height=140)

        self.xs = None
        self.H  = None

    def load_mesh(self):
        path = filedialog.askopenfilename(
            filetypes=[("Mesh","*.stl *.obj"),("All","*.*")]
        )
        if not path:
            return
        try:
            mesh = trimesh.load(path, force="mesh")
            (min_x, min_y, min_z), (max_x, max_y, max_z) = mesh.bounds
            # quick heightmap
            nx, nz = 300, 300
            xs = np.linspace(min_x, max_x, nx)
            zs = np.linspace(min_z, max_z, nz)
            origins = np.column_stack([
                np.repeat(xs, nz),
                np.full(nx*nz, max_y + (max_y-min_y)*0.1),
                np.tile(zs, nx)
            ])
            dirs = np.tile([0,-1,0], (nx*nz,1))
            locs, idxs, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
            H = np.full((nx, nz), np.nan)
            for rid, pt in zip(idxs, locs):
                i, j = divmod(rid, nz)
                H[i,j] = pt[1]
            self.xs, self.H = xs, H
            messagebox.showinfo("Mesh Loaded", f"Heightmap ready ({nx}×{nz})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process(self):
        if self.xs is None:
            messagebox.showwarning("No mesh", "Load an STL/OBJ first.")
            return
        try:
            width = float(self.width_var.get())
            nact  = int(self.num_var.get())
            if nact <= 0: raise ValueError
        except:
            messagebox.showerror("Input error", "Enter numeric bounds & positive actuators.")
            return

        spacing = width / nact
        positions = [i*spacing for i in range(nact)]

        self.out.delete("1.0", "end")
        self.out.insert("end", f"Width={width}, Actuators={nact}, Spacing={spacing}\n\n")
        for i, x in enumerate(positions, 1):
            ix = int(np.abs(self.xs - x).argmin())
            heights = self.H[ix, :].tolist()
            self.out.insert("end", f"Act {i}: x={self.xs[ix]:.3f} → heights: {heights}\n")

if __name__ == "__main__":
    SimpleGUI().mainloop()
