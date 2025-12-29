import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from spectral import open_image
from tkinter import Tk, filedialog
from matplotlib.patches import Patch

# ---- Configuration ----
id_to_name = {
    1:"cardboard_none", 2:"cardboard_PS", 3:"cardboard_silicon", 4:"cardboard_sugar",
    5:"wood_none", 6:"wood_PS", 7:"wood_silicon", 8:"wood_sugar",
    9:"ceramics_none", 10:"ceramics_PS", 11:"ceramics_silicon", 12:"ceramics_sugar"
}
name_to_id = {v: k for k, v in id_to_name.items()}

substrates = ["cardboard", "wood", "ceramics"]
materials = ["PS", "silicon", "sugar"]  # contaminants
fill_colors = {
    "surface": (0.7, 0.7, 0.7, 0.35),
    "PS": (1.0, 0.0, 0.0, 0.45),
    "silicon": (0.0, 1.0, 0.0, 0.45),
    "sugar": (0.0, 0.5, 1.0, 0.45)
}
edge_colors = {
    "surface": "gray",
    "PS": "red",
    "silicon": "green",
    "sugar": "blue"
}

# ---- I/O helpers ----
def choose_hdr():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select ENVI .hdr", filetypes=[("ENVI header", "*.hdr")])

def load_cube(hdr_path):
    img = open_image(hdr_path)
    cube = img.load()
    wl = np.array(img.bands.centers)
    meta = img.metadata
    return cube, wl, meta

def build_false_color(cube, meta):
    bands = [120, 42, 0]
    if 'default bands' in meta:
        try:
            b = [int(x) - 1 for x in meta['default bands'].strip("{}").split(",")]
            if all(0 <= bi < cube.shape[2] for bi in b):
                bands = b
        except Exception:
            pass
    rgb = cube[:, :, bands].astype(float)
    rgb[rgb < 0] = 0
    for i in range(3):
        m = rgb[:, :, i].max()
        if m > 0:
            rgb[:, :, i] /= m
    return rgb

# ---- Geometry helpers ----
def polygons_to_mask(polygons, shape):
    if not polygons:
        return np.zeros(shape, dtype=bool)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    mask = np.zeros((h, w), dtype=bool)
    for verts in polygons:
        path = Path(verts)
        mask |= path.contains_points(pts).reshape(h, w)
    return mask

# ---- Interactive polygon collector ----
class PolygonCollector:
    """
    Collect multiple polygons.
    Controls:
      - Draw polygon vertices by clicking, double-click to close.
      - z: undo last polygon
      - q: finish session
    Constraints:
      - Optional constraint_mask: polygons must be fully inside (else rejected).
      - Optional exclusion_mask: overlapping parts are clipped (inform user).
    """
    def __init__(self, base_image, existing_surface_polys, existing_material_polys_dict,
                 stage_label, fill_color, edge_color,
                 constraint_mask=None, exclusion_mask=None):
        self.img = base_image
        self.H, self.W = self.img.shape[:2]
        self.surface_polys = existing_surface_polys
        self.material_polys_dict = existing_material_polys_dict  # {material: [polys]}
        self.stage_label = stage_label  # "surface" or material name
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.constraint_mask = constraint_mask
        self.exclusion_mask = exclusion_mask
        self.new_polys = []

        self.fig = None
        self.ax = None
        self.selector = None

    def start(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self._init_selector()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()
        plt.show()
        return self.new_polys  # return newly drawn polygons

    def _init_selector(self):
        try:
            self.selector = PolygonSelector(
                self.ax,
                self.onselect,
                useblit=True,
                props=dict(color='yellow', linewidth=2, alpha=0.9)
            )
        except TypeError:
            self.selector = PolygonSelector(
                self.ax,
                self.onselect,
                useblit=True,
                lineprops=dict(color='yellow', linewidth=2, alpha=0.9),
                markerprops=dict(marker='o', markersize=3, mec='yellow', mfc='yellow', alpha=0.9)
            )

    def onselect(self, verts):
        verts = np.array(verts)
        # Build mask for new polygon
        mask_new = polygons_to_mask([verts], (self.H, self.W))

        # Constraint: must lie fully inside constraint_mask
        if self.constraint_mask is not None:
            outside = mask_new & (~self.constraint_mask)
            if np.any(outside):
                print("Polygon rejected: extends outside allowed surface.")
                return

        # Clip against exclusion (other materials)
        if self.exclusion_mask is not None and self.stage_label != "surface":
            overlap = mask_new & self.exclusion_mask
            if np.any(overlap):
                mask_new &= ~self.exclusion_mask
                if not np.any(mask_new):
                    print("Polygon entirely overlapped by existing materials; discarded.")
                    return
                print("Overlapping area with existing materials clipped.")

            # Replace verts by an approximate polygon from remaining mask (optional):
            # For simplicity keep original polygon; mask clipping only affects final labeling.

        self.new_polys.append(verts)
        print(f"Added polygon ({len(self.new_polys)} new this session).")
        self.redraw()

    def on_key(self, event):
        if event.key == 'z':
            if self.new_polys:
                self.new_polys.pop()
                print("Undid last polygon.")
                self.redraw()
            else:
                print("No new polygons to undo.")
        elif event.key == 'q':
            print("Finishing this selection session.")
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)

        # Overlay existing surfaces
        surf_mask = polygons_to_mask(self.surface_polys, (self.H, self.W))
        if np.any(surf_mask):
            overlay = np.zeros((self.H, self.W, 4))
            overlay[surf_mask] = fill_colors["surface"]
            self.ax.imshow(overlay)

        # Overlay existing materials
        for mat_name, plist in self.material_polys_dict.items():
            if not plist:
                continue
            m_mask = polygons_to_mask(plist, (self.H, self.W))
            if np.any(m_mask):
                ov = np.zeros((self.H, self.W, 4))
                ov[m_mask] = fill_colors[mat_name]
                self.ax.imshow(ov)

        # Draw existing polygon outlines
        for poly in self.surface_polys:
            xs, ys = poly[:, 0], poly[:, 1]
            self.ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], '-', color=edge_colors["surface"], linewidth=1.2)
        for mat_name, plist in self.material_polys_dict.items():
            for poly in plist:
                xs, ys = poly[:, 0], poly[:, 1]
                self.ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], '-', color=edge_colors[mat_name], linewidth=1.2)

        # Draw new polygons in current session
        for poly in self.new_polys:
            xs, ys = poly[:, 0], poly[:, 1]
            self.ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], '--', color='yellow', linewidth=2)

        self.ax.set_axis_off()
        self.ax.set_title(
            f"Stage: {self.stage_label} | Draw polygon (double-click). z=undo, q=finish"
        )
        self.fig.canvas.draw_idle()

# ---- Label map construction ----
def build_label_map(H, W, surface_polys, contam_polys):
    label_map = np.zeros((H, W), dtype=np.uint8)
    for s in substrates:
        s_polys = surface_polys[s]
        if not s_polys:
            continue
        surf_mask = polygons_to_mask(s_polys, (H, W))
        if not np.any(surf_mask):
            continue

        # Build contaminant masks
        contam_union = np.zeros_like(surf_mask)
        for m in materials:
            polys = contam_polys[s][m]
            if not polys:
                continue
            cmask = polygons_to_mask(polys, (H, W))
            # Ensure inside surface only
            cmask &= surf_mask
            # Remove overlaps with previously assigned contaminant pixels
            cmask &= ~contam_union
            if np.any(cmask):
                lname = f"{s}_{m}"
                lid = name_to_id[lname]
                label_map[cmask] = lid
                contam_union |= cmask

        clean_mask = surf_mask & (~contam_union)
        if np.any(clean_mask):
            lname = f"{s}_none"
            lid = name_to_id[lname]
            label_map[clean_mask] = lid
    return label_map

def save_gt(label_map, wl, out_base):
    np.savez(out_base + "_GT.npz", label_map=label_map, wavelengths=wl, mapping=id_to_name)
    csv_path = out_base + "_GT.csv"
    h, w = label_map.shape
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("pixel_index,row,col,label_id,label_name\n")
        idx = 0
        for r in range(h):
            for c in range(w):
                lid = int(label_map[r, c])
                lname = id_to_name.get(lid, "unlabeled")
                f.write(f"{idx},{r},{c},{lid},{lname}\n")
                idx += 1
    print(f"Saved: {out_base}_GT.npz and {csv_path}")

def save_gt_preview(rgb, label_map, out_base):
    """
    Save a debug preview PNG showing the RGB image with GT map overlay and legend.
    """
    # Define per-class colors (RGBA)
    label_colors = {
        "cardboard_none":  (0.80, 0.80, 0.80, 0.55),
        "wood_none":       (0.60, 0.60, 0.60, 0.55),
        "ceramics_none":   (0.90, 0.90, 0.90, 0.55),
        "cardboard_PS":    (1.00, 0.20, 0.20, 0.55),
        "wood_PS":         (0.85, 0.00, 0.00, 0.55),
        "ceramics_PS":     (0.70, 0.00, 0.00, 0.55),
        "cardboard_silicon": (0.20, 1.00, 0.20, 0.55),
        "wood_silicon":      (0.00, 0.85, 0.00, 0.55),
        "ceramics_silicon":  (0.00, 0.70, 0.00, 0.55),
        "cardboard_sugar": (0.20, 0.55, 1.00, 0.55),
        "wood_sugar":      (0.10, 0.45, 0.90, 0.55),
        "ceramics_sugar":  (0.00, 0.35, 0.80, 0.55),
    }
    H, W = label_map.shape
    overlay = np.zeros((H, W, 4), dtype=float)

    present = []
    for lid, lname in id_to_name.items():
        mask = (label_map == lid)
        if np.any(mask):
            rgba = label_colors.get(lname, (1, 1, 1, 0.5))
            overlay[mask] = rgba
            present.append(lname)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    if overlay[..., 3].max() > 0:
        ax.imshow(overlay)
    ax.set_axis_off()

    # Legend only for present classes
    patches = [Patch(facecolor=label_colors.get(name, (1, 1, 1, 0.5)), edgecolor='k', label=name) for name in present]
    if patches:
        ax.legend(handles=patches, loc='upper right', fontsize=8, frameon=True)

    out_png = out_base + "_GT_preview.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved GT preview image: {out_png}")

# ---- Main terminal-driven workflow ----
def main():
    hdr = choose_hdr()
    if not hdr:
        print("No HDR chosen.")
        return
    cube, wl, meta = load_cube(hdr)
    rgb = build_false_color(cube, meta)
    H, W = rgb.shape[:2]

    # Data structures
    surface_polys = {s: [] for s in substrates}
    contam_polys = {s: {m: [] for m in materials} for s in substrates}

    print("Terminal-driven labeling:")
    print("You can revisit a surface anytime to add more polygons.")
    print("Polygons are collected by drawing and double-clicking to finish, q to close that window.")

    while True:
        # Surface selection menu
        print("\nSelect surface to edit:")
        for i, s in enumerate(substrates, 1):
            print(f"[{i}] {s} (polygons: {len(surface_polys[s])})")
        print("[0] DONE (finish all surfaces)")
        sel = input("Choice: ").strip()
        if not sel.isdigit():
            print("Invalid input.")
            continue
        sel = int(sel)
        if sel == 0:
            break
        if not (1 <= sel <= len(substrates)):
            print("Out of range.")
            continue
        sname = substrates[sel - 1]

        # Collect surface polygons
        print(f"\nEditing surface '{sname}'. Open window...")
        collector = PolygonCollector(
            base_image=rgb,
            existing_surface_polys=surface_polys[sname],
            existing_material_polys_dict=contam_polys[sname],
            stage_label="surface",
            fill_color=fill_colors["surface"],
            edge_color=edge_colors["surface"],
            constraint_mask=None,
            exclusion_mask=None
        )
        new = collector.start()
        surface_polys[sname].extend(new)
        print(f"Surface '{sname}' total polygons: {len(surface_polys[sname])}")

        # Ask for contaminants
        yn = input("Add contaminant polygons for this surface? (y/n): ").strip().lower()
        if yn != 'y':
            continue

        # Build current surface mask for constraints
        surface_mask = polygons_to_mask(surface_polys[sname], (H, W))
        if not np.any(surface_mask):
            print("Surface mask empty; cannot add contaminants.")
            continue

        while True:
            print(f"\nSurface '{sname}' contaminant menu:")
            for i, m in enumerate(materials, 1):
                count_m = len(contam_polys[sname][m])
                print(f"[{i}] {m} (polygons: {count_m})")
            print("[0] Done contaminants for this surface")
            csel = input("Choice: ").strip()
            if not csel.isdigit():
                print("Invalid input.")
                continue
            csel = int(csel)
            if csel == 0:
                break
            if not (1 <= csel <= len(materials)):
                print("Out of range.")
                continue
            mname = materials[csel - 1]

            # Exclusion mask = union of other contaminants (to clip overlaps)
            exclusion = np.zeros_like(surface_mask)
            for other_m in materials:
                if other_m == mname:
                    continue
                if contam_polys[sname][other_m]:
                    exclusion |= polygons_to_mask(contam_polys[sname][other_m], (H, W))

            print(f"Editing contaminant '{mname}' on '{sname}'. Open window...")
            collector_c = PolygonCollector(
                base_image=rgb,
                existing_surface_polys=surface_polys[sname],
                existing_material_polys_dict=contam_polys[sname],
                stage_label=mname,
                fill_color=fill_colors[mname],
                edge_color=edge_colors[mname],
                constraint_mask=surface_mask,
                exclusion_mask=exclusion
            )
            new_c = collector_c.start()
            contam_polys[sname][mname].extend(new_c)
            print(f"Contaminant '{mname}' polygons now: {len(contam_polys[sname][mname])}")

    # Build label map
    label_map = build_label_map(H, W, surface_polys, contam_polys)

    # Save
    base_dir = os.path.dirname(hdr)
    base_name = os.path.splitext(os.path.basename(hdr))[0]
    out_base = os.path.join(base_dir, base_name)
    save_gt(label_map, wl, out_base)
    # NEW: save debug preview image
    save_gt_preview(rgb, label_map, out_base)

    # Summary
    print("\nSummary counts:")
    for s in substrates:
        print(f"{s}: surface_polys={len(surface_polys[s])}")
        for m in materials:
            print(f"  {m}: {len(contam_polys[s][m])}")

if __name__ == "__main__":
    main()