#!/usr/bin/env python3
"""Fig. 1d timeline as a vector SVG at final size (elements/d_timeline.svg).

The PETH windows are drawn to scale, as defined in dmn_bwm.concat_PETHs:
rest [-1.15, -1] s and quiescence [-0.4, -0.1] s before stimulus onset; stimulus
window [0, 0.15] s; choice window [-0.15, 0] s and movement window [0, 0.15] s
around movement onset; feedback window [0, 0.3] s. One time scale (MM_PER_S)
for all windows; the event-aligned segments are separated by gaps (the axis
break marks the gap between rest and quiescence).

Layout (top to bottom): event names over the onset bars with the symbols of the
IBL task schematic (stimulus: Gabor patch and go-cue tone, blue; feedback: water
drop, blue, and noise burst, red), names of the windows without an event
(rest, quiescence, choice), the windows, the axis with
times (0 under each event), "time (s)". Units are mm at the figure's final size
(panel d is ~62 mm wide); Arial, 5.5 pt labels, 5 pt ticks.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "elements"
PT = 0.3528  # mm per point
FONT = "Arial, 'Liberation Sans', Helvetica, sans-serif"
INK = "#000000"
WIN = (125, 107, 104)  # window colour (taupe, as in the original panel)
REWARD, ERROR = "#2b8cd6", "#d62728"  # go cue tone and water: blue; noise burst: red
LABEL, TICK = 5.5, 5.0  # font sizes (pt)
MM_PER_S = 30.0
W, H = 61.5, 15.0

REST = dict(t0=-1.15, x0=2.6)                                   # stimulus-aligned
STIM = dict(t0=-0.4, x0=11.5)                                   # stimulus-aligned
MOVE = dict(t0=-0.15, x0=STIM["x0"] + 0.55 * MM_PER_S + 5.5)    # movement-aligned
FDBK = dict(t0=0.0, x0=MOVE["x0"] + 0.30 * MM_PER_S + 6.5)      # feedback-aligned

Y_EVENT = 2.3              # baseline of the event names
Y_WINLAB = 5.6             # baseline of the window names
Y_WIN0, Y_WIN1 = 6.4, 8.8  # window boxes
Y_BAR0 = 3.2               # onset bars from here down to the axis
Y_AXIS = 9.6
Y_TICK = 11.9
Y_XLABEL = 14.4


def x(seg, t):
    return seg["x0"] + (t - seg["t0"]) * MM_PER_S


def text(xx, yy, s, size=LABEL, anchor="middle"):
    return (f'<text x="{xx:.2f}" y="{yy:.2f}" font-family="{FONT}" font-size="{size * PT:.3f}" '
            f'text-anchor="{anchor}" fill="{INK}">{s}</text>')


def gradient(gid, dark_at_right):
    r, g, b = WIN
    a0, a1 = (0.0, 1.0) if dark_at_right else (1.0, 0.0)
    return (f'<linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">'
            f'<stop offset="0" stop-color="rgb({r},{g},{b})" stop-opacity="{a0}"/>'
            f'<stop offset="1" stop-color="rgb({r},{g},{b})" stop-opacity="{a1}"/></linearGradient>')


def box(x0, x1, fill):
    return (f'<rect x="{x0:.2f}" y="{Y_WIN0:.2f}" width="{x1 - x0:.2f}" '
            f'height="{Y_WIN1 - Y_WIN0:.2f}" fill="{fill}"/>')


def speaker(x0, cy, s=1.9, color=INK):
    """Small loudspeaker icon (body, cone, two sound arcs) starting at x0, stroke only."""
    h = s / 2
    cx = x0 + h
    sw = 0.2
    body = (f"M{cx - h:.2f},{cy - h * 0.35:.2f} h{h * 0.45:.2f} l{h * 0.55:.2f},{-h * 0.45:.2f} "
            f"v{h * 1.6:.2f} l{-h * 0.55:.2f},{-h * 0.45:.2f} h{-h * 0.45:.2f} z")
    arcs = "".join(
        f'<path d="M{cx + h * k:.2f},{cy - h * 0.25 * (1 + k):.2f} q{h * 0.35:.2f},{h * 0.25 * (1 + k):.2f} '
        f'0,{h * 0.5 * (1 + k):.2f}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        for k in (0.35, 0.75))
    return f'<path d="{body}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>' + arcs


def gabor(x0, cy, d=1.9, n=3):
    """Gabor patch: a disc of n dark vertical bars on white, thin outline (as in the IBL task schematic)."""
    r = d / 2
    cx = x0 + r
    cid = f"gabor{int(x0 * 100)}"
    w = d / (2 * n)
    bars = "".join(f'<rect x="{cx - r + (2 * k + 0.5) * w:.3f}" y="{cy - r:.3f}" width="{w:.3f}" '
                   f'height="{d:.3f}" fill="{INK}"/>' for k in range(n))
    return (f'<clipPath id="{cid}"><circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}"/></clipPath>'
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" fill="#fff"/>'
            f'<g clip-path="url(#{cid})">{bars}</g>'
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" fill="none" stroke="{INK}" stroke-width="0.15"/>')


def drop(x0, cy, h=1.9, color=REWARD):
    """Water drop (reward)."""
    w = h * 0.62
    cx = x0 + w / 2
    top, bot = cy - h / 2, cy + h / 2
    r = w / 2
    path = (f"M{cx:.3f},{top:.3f} C{cx + r * 0.55:.3f},{top + h * 0.35:.3f} {cx + r:.3f},{bot - r * 1.3:.3f} "
            f"{cx + r:.3f},{bot - r:.3f} A{r:.3f},{r:.3f} 0 0 1 {cx - r:.3f},{bot - r:.3f} "
            f"C{cx - r:.3f},{bot - r * 1.3:.3f} {cx - r * 0.55:.3f},{top + h * 0.35:.3f} {cx:.3f},{top:.3f} z")
    return f'<path d="{path}" fill="{color}"/>', w


def text_width(s, size):
    """Approximate width (mm) of Arial text: 0.52 em per character."""
    return len(s) * 0.52 * size * PT


def event(xx, name, icons=()):
    """Onset bar with the event name centred over it and symbols to the right of the name."""
    out = [f'<rect x="{xx - 0.35:.2f}" y="{Y_BAR0:.2f}" width="0.7" height="{Y_AXIS - Y_BAR0:.2f}" fill="{INK}"/>',
           text(xx, Y_EVENT, name)]
    xi = xx + text_width(name, LABEL) / 2 + 0.5
    cy = Y_EVENT - 0.65
    for icon in icons:
        if icon == "gabor":
            out.append(gabor(xi, cy)); xi += 2.6
        elif icon == "tone":
            out.append(speaker(xi, cy, color=REWARD)); xi += 2.3
        elif icon == "water":
            svg, w = drop(xi, cy); out.append(svg); xi += w + 0.5
        elif icon == "noise":
            out.append(speaker(xi, cy, color=ERROR)); xi += 2.3
    return out


def tick(xx, s):
    return [f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{Y_AXIS:.2f}" y2="{Y_AXIS + 0.6:.2f}" '
            f'stroke="{INK}" stroke-width="0.2"/>', text(xx, Y_TICK, s, TICK)]


def main():
    OUT.mkdir(exist_ok=True)
    r, g, b = WIN
    el = ['<defs>', gradient("up", True), gradient("down", False), '</defs>']

    # windows, to scale
    el.append(box(x(REST, -1.15), x(REST, -1.0), f"rgb({r},{g},{b})"))
    el.append(box(x(STIM, -0.4), x(STIM, -0.1), "url(#up)"))      # quiescence
    el.append(box(x(STIM, 0.0), x(STIM, 0.15), "url(#down)"))     # stimulus
    el.append(box(x(MOVE, -0.15), x(MOVE, 0.0), "url(#up)"))      # choice
    el.append(box(x(MOVE, 0.0), x(MOVE, 0.15), "url(#down)"))     # movement
    el.append(box(x(FDBK, 0.0), x(FDBK, 0.3), "url(#down)"))      # feedback

    # window names for the windows without an event of their own; the windows after
    # stimulus, movement and feedback onset are named by the event above them
    gap = 0.6
    el.append(text((x(REST, -1.15) + x(REST, -1.0)) / 2, Y_WINLAB, "rest"))
    el.append(text((x(STIM, -0.4) + x(STIM, -0.1)) / 2, Y_WINLAB, "quiescence"))
    el.append(text(x(MOVE, 0) - gap, Y_WINLAB, "choice", anchor="end"))

    # axis with a break between rest and quiescence, arrow at the end
    xb = (x(REST, -1.0) + x(STIM, -0.4)) / 2
    sw = 0.25
    el.append(f'<line x1="0.6" x2="{xb - 0.6:.2f}" y1="{Y_AXIS}" y2="{Y_AXIS}" stroke="{INK}" stroke-width="{sw}"/>')
    el.append(f'<line x1="{xb + 0.6:.2f}" x2="{W - 1.6:.2f}" y1="{Y_AXIS}" y2="{Y_AXIS}" stroke="{INK}" stroke-width="{sw}"/>')
    for dx in (-0.6, 0.6):
        el.append(f'<line x1="{xb + dx - 0.4:.2f}" x2="{xb + dx + 0.4:.2f}" y1="{Y_AXIS + 0.7}" '
                  f'y2="{Y_AXIS - 0.7}" stroke="{INK}" stroke-width="{sw}"/>')
    el.append(f'<path d="M{W - 0.3:.2f},{Y_AXIS} l-1.6,-0.65 v1.3 z" fill="{INK}"/>')

    # times
    for seg, t, s in ((REST, -1.15, "−1.15"), (REST, -1.0, "−1"), (STIM, -0.4, "−0.4"),
                      (STIM, -0.1, "−0.1"), (STIM, 0, "0"), (STIM, 0.15, "0.15"),
                      (MOVE, -0.15, "−0.15"), (MOVE, 0, "0"), (MOVE, 0.15, "0.15"),
                      (FDBK, 0, "0"), (FDBK, 0.3, "0.3")):
        el += tick(x(seg, t), s)

    # events
    el += event(x(STIM, 0), "stimulus", icons=("gabor", "tone"))
    el += event(x(MOVE, 0), "movement")
    el += event(x(FDBK, 0), "feedback", icons=("water", "noise"))
    el.append(text((x(STIM, 0) + x(FDBK, 0)) / 2, Y_XLABEL, "time (s)"))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" '
           f'viewBox="0 0 {W} {H}">\n' + "\n".join(el) + "\n</svg>\n")
    path = OUT / "d_timeline.svg"
    path.write_text(svg)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
