#!/usr/bin/env python3
"""Assemble the Fig. 1 draft (183 mm wide) from the panel elements: task_intro.svg/.pdf/.png.

Row 1  a  data: task, recordings, timeline       elements/panel_d.svg   (make_panel_d.py)
       b  grand average per trial type (wide)    elements/panel_b.svg   (make_panels_bc.py)
Row 2  c  grand-average feature vector           elements/panel_c.svg   (make_panels_bc.py)
       d  how homogeneous is response diversity  elements/panel_a.svg   (make_panel_a.py)
       e  latency tiling, stimulus window        elements/panel_latency.svg (make_panel_latency.py)
The mixed-selectivity schematic (make_panel_mixing.py, elements/panel_mixing.svg) was
removed 2026-10-12 with the mixed-selectivity analysis; see fig_response_correlations/si/mixed_selectivity.
Every element is nested as vector SVG; element ids are prefixed per file so that
clip paths, gradients and symbols of different files cannot collide.
"""

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EL = HERE / "elements"
FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
PT = 0.3528
W = 183.0


def nested(path, x, y, w=None, h=None, view=None, prefix="p"):
    """Embed an SVG file at (x, y) mm with width w (height from the aspect) or height h."""
    src = path.read_text()
    src = re.sub(r"<\?xml[^>]*\?>", "", src)
    src = re.sub(r"<!DOCTYPE[^>]*>", "", src)
    vb = view or [float(v) for v in re.search(r'viewBox="([^"]+)"', src).group(1).replace(",", " ").split()]
    if w is None:
        w = h * vb[2] / vb[3]
    if h is None:
        h = w * vb[3] / vb[2]
    inner = src[src.index(">", src.index("<svg")) + 1: src.rindex("</svg>")]
    ids = set(re.findall(r'\bid="([^"]+)"', inner))
    inner = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}_{m.group(1)}"', inner)
    inner = re.sub(r'url\(#([^)]+)\)', lambda m: f"url(#{prefix}_{m.group(1)})" if m.group(1) in ids else m.group(0), inner)
    inner = re.sub(r'((?:xlink:)?href)="#([^"]+)"',
                   lambda m: f'{m.group(1)}="#{prefix}_{m.group(2)}"' if m.group(2) in ids else m.group(0), inner)
    vbs = " ".join(f"{v:.4f}" for v in vb)
    # explicit clip to the element's box (Inkscape does not clip nested svg viewports)
    clip = (f'<clipPath id="{prefix}_box"><rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" '
            f'height="{h:.3f}"/></clipPath>')
    return (f'<defs>{clip}</defs><g clip-path="url(#{prefix}_box)">'
            f'<svg x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" viewBox="{vbs}" '
            f'preserveAspectRatio="xMinYMin meet" overflow="hidden">{inner}</svg></g>'), w, h


def letter(x, y, s):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{8 * PT:.3f}" '
            f'font-weight="bold">{s}</text>')


def main():
    el = []
    # row 1
    y1 = 3.0
    s, wa, ha = nested(EL / "panel_d.svg", 2.5, y1, w=62.0, prefix="a"); el += [s, letter(0, y1 + 1.2, "a")]
    s, wb, hb = nested(EL / "panel_b.svg", W - 115.0, y1, h=ha, prefix="b"); el += [s, letter(W - 115.0 - 1.0, y1 + 1.2, "b")]
    # row 2
    y2 = y1 + ha + 3.0
    s, wc, hc = nested(EL / "panel_c.svg", 2.5, y2, w=58.0, prefix="c"); el += [s, letter(0, y2 + 1.2, "c")]
    hd = hc
    s, wd, _ = nested(EL / "panel_a.svg", 62.0, y2, h=hd, prefix="d"); el += [s, letter(61.0, y2 + 1.2, "d")]
    s, we, he = nested(EL / "panel_latency.svg", W - 62.0 * hd / 36.0, y2, h=hd, prefix="e")
    el += [s, letter(W - we - 0.5, y2 + 1.2, "e")]
    H = y2 + hd + 1.0

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}mm" height="{H:.2f}mm" viewBox="0 0 {W} {H:.3f}">\n'
           f'<rect width="{W}" height="{H:.3f}" fill="#fff"/>\n' + "\n".join(el) + "\n</svg>\n")
    out = HERE / "task_intro.svg"
    out.write_text(svg)
    for ext, extra in (("pdf", []), ("png", ["--export-dpi=400", "--export-background=white"])):
        subprocess.run(["inkscape", str(out), f"--export-type={ext}",
                        f"--export-filename={out.with_suffix('.' + ext)}", *extra],
                       check=True, capture_output=True)
    print(f"Saved {out} (+ .pdf, .png), {W:.0f} x {H:.1f} mm; "
          f"widths a {wa:.0f}, b {wb:.0f}, c {wc:.0f}, d {wd:.0f}, e {we:.0f} mm")


if __name__ == "__main__":
    main()
