#!/usr/bin/env python3
"""Assemble Fig. 1d (task, recordings, timeline) at final size: elements/panel_d.svg/.pdf/.png.

Same content and arrangement as the earlier panel d. Element positions are taken
from the manuscript PDF (figure points, 976.45 pt = 183 mm) and converted to mm:
task cartoon (elements/d_task_cartoon.png), 3D rendering of the recorded
insertions (elements/d_insertions_brain.png, upright, panel letter and "15 mice"
removed), Neuropixels tip inset (elements/d_probe_tip.png, cropped to the view
shown before), "~50K cells", and the vector timeline (elements/d_timeline.svg,
make_timeline.py) below them. PNGs are embedded, so the SVG is self-contained.
The panel letter is left to the figure assembly.
"""

import base64
import io
import re
import subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
EL = HERE / "elements"
F = 183.0 / 976.45          # mm per figure point
X0, Y0 = 660.0, -7.3        # figure-point origin of the panel
PT = 0.3528
FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
W = 62.0
Y_TIMELINE = 17.4           # mm, top of the timeline


def mm(xpt, ypt):
    return (xpt - X0) * F, (ypt - Y0) * F


def png_uri(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image(im, x, y, w, h):
    return (f'<image x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" '
            f'preserveAspectRatio="none" href="{png_uri(im)}"/>')


def main():
    el = []
    # task cartoon: PDF box (673.1, 3.9)-(809.3, 70.2) pt; image keeps its aspect
    task = Image.open(EL / "d_task_cartoon.png").convert("RGB")
    x, y = mm(673.1, 3.9)
    w = (809.3 - 673.1) * F
    el.append(image(task, x, y, w, w * task.height / task.width))

    # brain: PDF box (857.6, -7.3)-(947.3, 82.7) pt; height kept, width from aspect
    brain = Image.open(EL / "d_insertions_brain.png").convert("RGB")
    h = (82.7 + 7.3) * F
    wb = h * brain.width / brain.height
    xc = mm((857.6 + 947.3) / 2, 0)[0]
    el.append(image(brain, xc - wb / 2, 0.0, wb, h))

    # probe tip inset: PDF box (847.6, -3.0)-(873.6, 27.3) pt shows the bottom of the image
    tip = Image.open(EL / "d_probe_tip.png").convert("RGB")
    bw, bh = 873.6 - 847.6, 27.3 + 3.0
    keep = round(tip.width * bh / bw)
    tip = tip.crop((0, tip.height - keep, tip.width, tip.height))
    x, y = mm(847.6, -3.0)
    el.append(image(tip, x, y, bw * F, bh * F))

    # "~50K cells": PDF text box (812.0, 35.6)-(862.2, 47.6) pt
    x, y = mm(812.0, 45.6)
    el.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{6.0 * PT:.3f}" '
              f'fill="#000">~50K cells</text>')

    # timeline (vector), inserted as a nested svg
    tl = (EL / "d_timeline.svg").read_text()
    tw, th = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', tl).groups())
    inner = tl[tl.index(">", tl.index("<svg")) + 1: tl.rindex("</svg>")]
    el.append(f'<svg x="{(W - tw) / 2:.3f}" y="{Y_TIMELINE:.3f}" width="{tw}" height="{th}" '
              f'viewBox="0 0 {tw} {th}">{inner}</svg>')
    H = Y_TIMELINE + th

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H:.2f}mm" '
           f'viewBox="0 0 {W} {H:.3f}">\n' + "\n".join(el) + "\n</svg>\n")
    out = EL / "panel_d.svg"
    out.write_text(svg)
    for ext, extra in (("pdf", []), ("png", ["--export-dpi=600", "--export-background=white"])):
        subprocess.run(["inkscape", str(out), f"--export-type={ext}",
                        f"--export-filename={out.with_suffix('.' + ext)}", *extra],
                       check=True, capture_output=True)
    print(f"Saved {out} (+ .pdf, .png), {W} x {H:.1f} mm")


if __name__ == "__main__":
    main()
