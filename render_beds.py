"""
Render each bed_*.3mf as top-down + side-view PNGs (PIL, no matplotlib)
and emit a self-contained viewer.html presenting the print plan.
"""

from __future__ import annotations

import json
import os
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
BED_X = 220.0
BED_Y = 220.0
BED_Z = 250.0
OUT_DIR = "output"

PX_PER_MM = 4         # 880x880 px image for 220mm bed
PAD_MM = 12           # padding around bed
PART_ALPHA = 165
EDGE_ALPHA = 90
LABEL_BG_ALPHA = 215

PALETTE = [
    (230, 25, 75),  (60, 180, 75),   (255, 225, 25), (67, 99, 216),  (245, 130, 49),
    (145, 30, 180), (66, 212, 244),  (240, 50, 230), (191, 239, 69), (250, 190, 212),
    (70, 153, 144), (220, 190, 255), (154, 99, 36),  (255, 250, 200),(128, 0, 0),
    (170, 255, 195),(128, 128, 0),   (255, 216, 177),(0, 0, 117),    (169, 169, 169),
]

BG = (17, 20, 27)         # page bg
BED_BG = (27, 31, 42)     # bed surface
BED_BORDER = (90, 120, 181)
GRID = (44, 49, 64)
TEXT = (216, 221, 232)

try:
    FONT = ImageFont.truetype("/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf", 13)
    FONT_SMALL = ImageFont.truetype("/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf", 10)
except OSError:
    FONT = ImageFont.load_default()
    FONT_SMALL = FONT


def read_3mf(path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    objs = {}
    for o in root.iter(NS + "object"):
        verts, tris = [], []
        for v in o.iter(NS + "vertex"):
            verts.append([float(v.attrib["x"]), float(v.attrib["y"]), float(v.attrib["z"])])
        for t in o.iter(NS + "triangle"):
            tris.append([int(t.attrib["v1"]), int(t.attrib["v2"]), int(t.attrib["v3"])])
        objs[o.attrib["id"]] = {
            "name": o.attrib.get("name", "?"),
            "verts": np.array(verts),
            "tris": np.array(tris),
        }
    placed = []
    for item in root.iter(NS + "item"):
        oid = item.attrib["objectid"]
        t = list(map(float, item.attrib["transform"].split()))
        M = np.array([[t[0], t[3], t[6]],
                      [t[1], t[4], t[7]],
                      [t[2], t[5], t[8]]])
        tr = np.array(t[9:12])
        o = objs[oid]
        placed.append({"name": o["name"],
                       "verts": o["verts"] @ M.T + tr,
                       "tris": o["tris"]})
    return placed


def short(name):
    tag = "L" if name.startswith("LWR") else "U"
    return f"{tag}·{name.split('__')[-1]}"


def _make_canvas(world_w, world_h):
    """RGB canvas with the bed drawn and grid, plus mm→px mapper.
    world_w, world_h in mm.
    """
    img_w = int(world_w * PX_PER_MM) + 2 * int(PAD_MM * PX_PER_MM)
    img_h = int(world_h * PX_PER_MM) + 2 * int(PAD_MM * PX_PER_MM)
    img = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)
    pad_px = int(PAD_MM * PX_PER_MM)

    def to_px(x_mm, y_mm):
        # y inverted (image y grows downward; world y grows upward)
        return (pad_px + x_mm * PX_PER_MM,
                img_h - pad_px - y_mm * PX_PER_MM)

    # bed
    x0, y0 = to_px(0, 0)            # bottom-left
    x1, y1 = to_px(world_w, world_h)  # top-right
    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                   fill=BED_BG, outline=BED_BORDER, width=3)
    # grid every 20mm
    for k in range(0, int(world_w) + 1, 20):
        a = to_px(k, 0); b = to_px(k, world_h)
        draw.line([a, b], fill=GRID, width=1)
    for k in range(0, int(world_h) + 1, 20):
        a = to_px(0, k); b = to_px(world_w, k)
        draw.line([a, b], fill=GRID, width=1)

    return img, draw, to_px


def render_top(parts, out_path):
    img, draw, to_px = _make_canvas(BED_X, BED_Y)
    info = []
    for i, p in enumerate(parts):
        color = PALETTE[i % len(PALETTE)]
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(layer)
        # draw all triangles (XY projection)
        verts = p["verts"]
        for tri in p["tris"]:
            a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
            pts = [to_px(a[0], a[1]), to_px(b[0], b[1]), to_px(c[0], c[1])]
            ldraw.polygon(pts, fill=color + (PART_ALPHA,))
        img.paste(layer, (0, 0), layer)

        mn = verts.min(0); mx = verts.max(0)
        cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        lx, ly = to_px(cx, cy)
        text = short(p["name"])
        bbox = draw.textbbox((lx, ly), text, font=FONT, anchor="mm")
        pad = 3
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                       fill=color + (LABEL_BG_ALPHA,), outline=(0, 0, 0))
        draw.text((lx, ly), text, font=FONT, fill=(255, 255, 255), anchor="mm")
        info.append(f"{text:10s}  {mx[0]-mn[0]:6.1f} × {mx[1]-mn[1]:6.1f} × {mx[2]-mn[2]:5.1f} mm")

    # axis tick labels
    for k in range(0, int(BED_X) + 1, 40):
        x, y = to_px(k, 0)
        draw.text((x, y + 4), f"{k}", font=FONT_SMALL, fill=TEXT, anchor="mt")
    for k in range(0, int(BED_Y) + 1, 40):
        x, y = to_px(0, k)
        draw.text((x - 6, y), f"{k}", font=FONT_SMALL, fill=TEXT, anchor="rm")

    img.save(out_path, optimize=True)
    return info


def render_side(parts, out_path):
    img, draw, to_px = _make_canvas(BED_X, BED_Z)
    for i, p in enumerate(parts):
        color = PALETTE[i % len(PALETTE)]
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(layer)
        verts = p["verts"]
        for tri in p["tris"]:
            a = verts[tri[0]]; b = verts[tri[1]]; c = verts[tri[2]]
            pts = [to_px(a[0], a[2]), to_px(b[0], b[2]), to_px(c[0], c[2])]
            ldraw.polygon(pts, fill=color + (PART_ALPHA,))
        img.paste(layer, (0, 0), layer)

    for k in range(0, int(BED_X) + 1, 40):
        x, y = to_px(k, 0); draw.text((x, y + 4), f"{k}", font=FONT_SMALL, fill=TEXT, anchor="mt")
    for k in range(0, int(BED_Z) + 1, 50):
        x, y = to_px(0, k); draw.text((x - 6, y), f"{k}", font=FONT_SMALL, fill=TEXT, anchor="rm")
    img.save(out_path, optimize=True)


def main():
    beds = sorted(p for p in os.listdir(OUT_DIR)
                  if p.startswith("bed_") and p.endswith(".3mf"))
    summary = []
    for path in beds:
        idx = int(path.split("_")[1].split(".")[0])
        print(f"Rendering bed_{idx}…", flush=True)
        parts = read_3mf(os.path.join(OUT_DIR, path))
        top = os.path.join(OUT_DIR, f"bed_{idx}_top.png")
        side = os.path.join(OUT_DIR, f"bed_{idx}_side.png")
        info = render_top(parts, top)
        render_side(parts, side)

        allv = np.concatenate([p["verts"] for p in parts], axis=0)
        mn = allv.min(0); mx = allv.max(0); sz = mx - mn
        margin_x = min(mn[0], BED_X - mx[0])
        margin_y = min(mn[1], BED_Y - mx[1])
        warn = []
        if margin_x < 1: warn.append(f"only {margin_x:.1f} mm X clearance — bed edge risk")
        if margin_y < 1: warn.append(f"only {margin_y:.1f} mm Y clearance — bed edge risk")
        if sz[2] > BED_Z: warn.append(f"Z {sz[2]:.1f} mm exceeds 250 mm Ender height")

        summary.append({
            "idx": idx, "file": path, "parts": len(parts),
            "footprint": [round(sz[0], 1), round(sz[1], 1), round(sz[2], 1)],
            "margin_x": round(margin_x, 1), "margin_y": round(margin_y, 1),
            "info": info, "warn": warn,
            "top": f"bed_{idx}_top.png", "side": f"bed_{idx}_side.png",
        })

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_viewer(summary)
    print(f"Done. {len(summary)} beds → output/viewer.html")


def write_viewer(summary):
    total_parts = sum(b["parts"] for b in summary)
    max_z = max(b["footprint"][2] for b in summary)
    rows = []
    for b in summary:
        warn = "".join(f'<div class="warn">⚠ {w}</div>' for w in b["warn"])
        info = "<br>".join(b["info"])
        rows.append(f"""
        <section class="bed">
          <h2>Bed {b['idx']} <span class="meta">— {b['parts']} part{'s' if b['parts']!=1 else ''} · footprint {b['footprint'][0]} × {b['footprint'][1]} × {b['footprint'][2]} mm · margins {b['margin_x']} / {b['margin_y']} mm</span></h2>
          {warn}
          <div class="views">
            <figure><img src="{b['top']}" alt="bed {b['idx']} top"><figcaption>Top view</figcaption></figure>
            <figure><img src="{b['side']}" alt="bed {b['idx']} side"><figcaption>Side view (X–Z)</figcaption></figure>
          </div>
          <pre class="parts">{info}</pre>
          <p class="dl"><a href="{b['file']}" download>⬇ {b['file']}</a></p>
        </section>""")

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tazeen — print plan ({len(summary)} beds, Ender 3 V3 SE)</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px;
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #11141b; color: #d8dde8; max-width: 1100px; margin-inline: auto; }}
  header {{ border-bottom: 1px solid #2c3140; padding-bottom: 14px; margin-bottom: 22px; }}
  h1 {{ margin: 0 0 4px 0; font-size: 22px; color: #e8ecf5; }}
  header p {{ margin: 4px 0; color: #8c96ac; }}
  .stats {{ display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; }}
  .stat {{ background: #1b1f2a; padding: 10px 14px; border-radius: 8px; border: 1px solid #2c3140; min-width: 100px; }}
  .stat b {{ display: block; font-size: 18px; color: #e8ecf5; }}
  .stat span {{ color: #8c96ac; font-size: 12px; }}
  section.bed {{ background: #161a23; border: 1px solid #2c3140; border-radius: 10px;
                  padding: 18px; margin-bottom: 22px; }}
  section.bed h2 {{ margin: 0 0 10px 0; font-size: 16px; color: #e8ecf5; }}
  .meta {{ color: #8c96ac; font-weight: normal; font-size: 13px; }}
  .views {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  figure {{ margin: 0; flex: 1 1 380px; min-width: 0; }}
  figure img {{ width: 100%; border-radius: 6px; border: 1px solid #2c3140; display: block; }}
  figcaption {{ font-size: 12px; color: #8c96ac; padding: 4px 2px 0; }}
  pre.parts {{ background: #0d1017; border: 1px solid #2c3140; border-radius: 6px;
               padding: 10px 14px; font-size: 12px; overflow-x: auto;
               color: #c5cde0; white-space: pre; }}
  .warn {{ background: #3a2814; border: 1px solid #b07b3a; color: #ffd9a8;
            padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 13px; }}
  .dl a {{ color: #6fa8ff; text-decoration: none; font-size: 13px; }}
  .dl a:hover {{ text-decoration: underline; }}
  footer {{ color: #6a7383; font-size: 12px; text-align: center; margin-top: 24px; }}
</style>
</head><body>
<header>
  <h1>Tazeen — Arabic sign print plan</h1>
  <p>Two STLs split into <b>{total_parts} connected parts</b>, laid flat &amp; packed onto <b>{len(summary)} Ender 3 V3 SE beds</b> (220 × 220 × 250 mm).</p>
  <div class="stats">
    <div class="stat"><b>{len(summary)}</b><span>beds total</span></div>
    <div class="stat"><b>{total_parts}</b><span>parts</span></div>
    <div class="stat"><b>{max_z:.0f} mm</b><span>tallest part Z</span></div>
    <div class="stat"><b>220 × 220</b><span>bed (mm)</span></div>
  </div>
</header>
{''.join(rows)}
<footer>Generated by render_beds.py · open any bed_<i>N</i>.3mf in PrusaSlicer to slice.</footer>
</body></html>
"""
    with open(os.path.join(OUT_DIR, "viewer.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
