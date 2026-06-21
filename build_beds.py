"""
Split two Arabic-text STLs into their separable parts, lay each one flat
(smallest bounding-box dimension becomes vertical), shelf-pack them onto
220x220 mm Ender 3 V3 SE beds, and emit one PrusaSlicer-compatible
3MF project per bed.

No external 3D libs required beyond numpy + numpy-stl. The 3MF writer
produces the minimal Core 3D Manufacturing Format payload that
PrusaSlicer happily ingests.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from io import StringIO

import numpy as np
from stl import mesh

# ---- Constants -------------------------------------------------------------

BED_X = 220.0
BED_Y = 220.0
MARGIN = 5.0           # keep parts away from absolute bed edge
GAP = 4.0              # gap between parts on the bed
SOURCE_FILES = [
    "LWR_RA_NEW NAME&DESIGN.stl",
    "UPR_RA_NEW NAME_DESIGN.stl",
]
OUT_DIR = "output"
PARTS_DIR = "parts"


# ---- Step 1: load + split into connected components ------------------------

def load_components(path: str, tol: float = 0.01):
    """Return a list of (name, vectors[N,3,3]) for each connected piece."""
    m = mesh.Mesh.from_file(path)
    pts = m.vectors.reshape(-1, 3)
    q = np.round(pts / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    tri_v = inv.reshape(-1, 3)
    n = inv.max() + 1

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for tri in tri_v:
        union(int(tri[0]), int(tri[1]))
        union(int(tri[1]), int(tri[2]))

    roots = np.array([find(i) for i in range(n)])
    tri_root = roots[tri_v[:, 0]]

    base = os.path.splitext(os.path.basename(path))[0]
    comps = []
    for idx, root in enumerate(np.unique(tri_root)):
        sub_vecs = m.vectors[tri_root == root].copy()
        comps.append((f"{base}__c{idx:02d}", sub_vecs))
    return comps


# ---- Step 2: lie flat (smallest bounding dim becomes Z) --------------------

def axis_aligned_rotate(vecs: np.ndarray, smallest_axis: int) -> np.ndarray:
    """Rotate so axis `smallest_axis` (0=x,1=y,2=z) maps to world Z.

    Preserves a right-handed coordinate frame.
    """
    if smallest_axis == 2:
        return vecs  # already flat
    if smallest_axis == 0:
        # swap X and Z
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    else:  # smallest_axis == 1
        # swap Y and Z
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    return vecs @ R.T


def lay_flat(vecs: np.ndarray) -> np.ndarray:
    pts = vecs.reshape(-1, 3)
    size = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmin(size))
    out = axis_aligned_rotate(vecs, axis)
    # drop to z=0, shift to non-negative XY
    pts = out.reshape(-1, 3)
    mn = pts.min(axis=0)
    out = out - mn  # translate so min corner is at origin
    return out


# ---- Step 3: shelf bin-pack parts onto 220x220 beds ------------------------

@dataclass
class Placed:
    name: str
    vecs: np.ndarray          # already laid flat, origin at (0,0,0)
    x: float                  # placement on bed
    y: float
    w: float
    h: float


def pack(parts):
    """parts: list[(name, vecs)] (already lying flat).
    Returns list[bed]; each bed = list[Placed]."""
    avail_x = BED_X - 2 * MARGIN
    avail_y = BED_Y - 2 * MARGIN

    items = []
    for name, vecs in parts:
        pts = vecs.reshape(-1, 3)
        sz = pts.max(axis=0) - pts.min(axis=0)
        w, h = float(sz[0]), float(sz[1])
        if w > avail_x or h > avail_y:
            # try rotating 90° in plane
            if h <= avail_x and w <= avail_y:
                R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
                vecs = vecs @ R.T
                # shift back to origin
                vecs = vecs - vecs.reshape(-1, 3).min(axis=0)
                w, h = h, w
            else:
                print(f"  !! {name} ({w:.1f}x{h:.1f}) exceeds bed; placing anyway")
        items.append((name, vecs, w, h))

    # sort by height desc for shelf packing
    items.sort(key=lambda t: -t[3])

    beds = []
    current = []
    shelves = []  # list of (shelf_y, shelf_h, cursor_x)
    next_y = MARGIN

    def new_bed():
        nonlocal current, shelves, next_y
        if current:
            beds.append(current)
        current = []
        shelves = []
        next_y = MARGIN

    for name, vecs, w, h in items:
        placed = False
        # try existing shelves
        for i, (sy, sh, cx) in enumerate(shelves):
            if h <= sh and cx + w <= MARGIN + avail_x:
                shelves[i] = (sy, sh, cx + w + GAP)
                current.append(Placed(name, vecs, cx, sy, w, h))
                placed = True
                break
        if placed:
            continue
        # new shelf on this bed?
        if next_y + h <= MARGIN + avail_y and w <= avail_x:
            sy = next_y
            shelves.append((sy, h, MARGIN + w + GAP))
            current.append(Placed(name, vecs, MARGIN, sy, w, h))
            next_y = sy + h + GAP
            placed = True
            continue
        # new bed
        new_bed()
        if h > avail_y or w > avail_x:
            # part bigger than bed — place at origin anyway
            current.append(Placed(name, vecs, MARGIN, MARGIN, w, h))
            next_y = MARGIN + h + GAP
            shelves = [(MARGIN, h, MARGIN + w + GAP)]
        else:
            sy = MARGIN
            shelves = [(sy, h, MARGIN + w + GAP)]
            current.append(Placed(name, vecs, MARGIN, sy, w, h))
            next_y = sy + h + GAP

    if current:
        beds.append(current)
    return beds


# ---- Step 4: minimal 3MF writer (one bed = one file) ----------------------

_CT = b"""<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

_RELS = b"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rel0" Target="/3D/3dmodel.model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _write_model_xml(placed_parts, out_stream):
    """Write 3MF Core model XML containing each placed part as an <object>
    plus a <build> section with translations to the bed positions."""
    w = out_stream
    w.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    w.write(
        '<model unit="millimeter" xml:lang="en-US"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
    )
    w.write(' <resources>\n')
    for oid, p in enumerate(placed_parts, start=1):
        verts = p.vecs.reshape(-1, 3)
        # dedupe verts
        q = np.round(verts * 1000).astype(np.int64)
        uniq, inv = np.unique(q, axis=0, return_inverse=True)
        uniq_f = uniq.astype(np.float64) / 1000.0
        tris = inv.reshape(-1, 3)
        w.write(f'  <object id="{oid}" type="model" name="{p.name}">\n')
        w.write('   <mesh>\n    <vertices>\n')
        for v in uniq_f:
            w.write(f'     <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>\n')
        w.write('    </vertices>\n    <triangles>\n')
        for t in tris:
            w.write(f'     <triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}"/>\n')
        w.write('    </triangles>\n   </mesh>\n  </object>\n')
    w.write(' </resources>\n <build>\n')
    for oid, p in enumerate(placed_parts, start=1):
        # 3MF transform: a row-major 4x3 like "m00 m10 m20  m01 m11 m21  m02 m12 m22  tx ty tz"
        # Spec text: 11 12 13 21 22 23 31 32 33 t1 t2 t3
        # For identity rotation + translation (tx,ty,0):
        tx, ty, tz = p.x, p.y, 0.0
        tr = f"1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}"
        w.write(f'  <item objectid="{oid}" transform="{tr}"/>\n')
    w.write(' </build>\n</model>\n')


def write_3mf(path, placed_parts):
    buf = StringIO()
    _write_model_xml(placed_parts, buf)
    model_xml = buf.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("3D/3dmodel.model", model_xml)


# ---- Step 5: orchestrate ---------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_parts = []
    print("Splitting STLs into components:")
    for f in SOURCE_FILES:
        comps = load_components(f)
        print(f"  {f}: {len(comps)} pieces")
        for name, vecs in comps:
            flat = lay_flat(vecs)
            pts = flat.reshape(-1, 3)
            sz = pts.max(axis=0) - pts.min(axis=0)
            print(f"    {name}: footprint {sz[0]:7.1f} x {sz[1]:7.1f} x {sz[2]:6.1f}")
            all_parts.append((name, flat))

    print(f"\nPacking {len(all_parts)} parts onto {BED_X:.0f}x{BED_Y:.0f} beds…")
    beds = pack(all_parts)
    print(f"  -> {len(beds)} bed(s)")

    for i, bed in enumerate(beds, start=1):
        # Re-center the packed parts on the bed (works whether the slicer
        # treats (0,0) as front-left or as bed center).
        xs = [p.x for p in bed]; ys = [p.y for p in bed]
        xe = [p.x + p.w for p in bed]; ye = [p.y + p.h for p in bed]
        bbox_cx = (min(xs) + max(xe)) / 2.0
        bbox_cy = (min(ys) + max(ye)) / 2.0
        dx = BED_X / 2.0 - bbox_cx
        dy = BED_Y / 2.0 - bbox_cy
        for p in bed:
            p.x += dx
            p.y += dy
        out = os.path.join(OUT_DIR, f"bed_{i}.3mf")
        write_3mf(out, bed)
        names = ", ".join(p.name.split("__")[-1] for p in bed)
        print(f"  bed_{i}.3mf  ({len(bed)} parts: {names})")


if __name__ == "__main__":
    main()
