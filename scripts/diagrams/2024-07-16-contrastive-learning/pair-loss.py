"""Pair loss around an anchor: shading is the loss, arrows the gradient-descent step on the other point."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as MARGIN, CATEGORICAL, INK, rect, save, svg_open, text

ANCHOR, POS, NEG = CATEGORICAL
W, H = 840, 380
RF, RM = 120, 82            # radius of the plotted plane, margin radius (sqrt(eps))
OP = 0.45                   # shading opacity at the maximum loss


def gradient(id, c, f):
    """Radial gradient whose opacity follows f(r), r in [0, 1] of the plotted radius."""
    stops = "".join(f'<stop offset="{k / 20:.2f}" stop-color="{c}" stop-opacity="{OP * f(k / 20):.3f}"/>'
                    for k in range(21))
    return f'<radialGradient id="{id}" cx="0.5" cy="0.5" r="0.5">{stops}</radialGradient>'


m = RM / RF
o = svg_open(W, H, [gradient("gp", POS, lambda r: r * r),
                    gradient("gn", NEG, lambda r: max(0.0, 1 - (r / m) ** 2))])
BOLD = 'font-weight="600"'


def arrow(x, y, dx, dy, c):
    """Arrow centred on (x, y) with a head that shrinks for short arrows."""
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    x0, y0, x1, y1 = x - dx / 2, y - dy / 2, x + dx / 2, y + dy / 2
    hl = min(6.5, 0.55 * n)
    hw = hl * 0.6
    bx, by = x1 - ux * hl, y1 - uy * hl
    return [f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{c}" stroke-width="1.6"/>',
            f'<path d="M{x1:.1f},{y1:.1f} L{bx - uy * hw:.1f},{by + ux * hw:.1f} L{bx + uy * hw:.1f},{by - ux * hw:.1f} z" fill="{c}"/>']


panels = [
    (6, POS, "gp", "Positive pair", "ℒ = D(z, z⁺)²", "positives are pulled towards the anchor", "z⁺"),
    (424, NEG, "gn", "Negative pair", "ℒ = max(0, ε − D(z, z⁻)²)", "negatives are pushed out past the margin", "z⁻")]
for x0, c, gid, title, formula, foot, sym in panels:
    pw = 410
    cx, cy = x0 + pw / 2, 192
    o.append(rect(x0, 10, pw, H - 20, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(cx, 36, title, 16, extra=BOLD))
    o.append(text(cx, 56, formula, 12.5))
    o.append(text(cx, H - 24, foot, 13, extra=f'fill="{c}" {BOLD}'))

    o.append(f'<circle cx="{cx}" cy="{cy}" r="{RF}" fill="url(#{gid})" stroke="{INK}" stroke-opacity="0.3" stroke-width="1"/>')
    for ring in range(2, 7):
        r = ring * RF / 6.6
        for k in range(12):
            a = math.radians(30 * k + 15 * (ring % 2))
            px, py = cx + r * math.cos(a), cy + r * math.sin(a)
            # -grad of the loss w.r.t. the other point has norm 2D, inwards for positives and
            # outwards for negatives inside the margin
            if c == POS:
                L = 0.3 * r
                o += arrow(px, py, -L * math.cos(a), -L * math.sin(a), c)
            elif r < RM:
                L = 0.3 * r
                o += arrow(px, py, L * math.cos(a), L * math.sin(a), c)
            else:
                o.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.4" fill="{INK}" fill-opacity="0.6"/>')
    if c == NEG:
        o.append(f'<circle cx="{cx}" cy="{cy}" r="{RM}" fill="none" stroke="{MARGIN}" stroke-width="1.6" stroke-dasharray="4 3"/>')
        a = math.radians(-35)
        o.append(text(cx + (RM + 8) * math.cos(a), cy + (RM + 8) * math.sin(a), "margin √ε", 11.5, "start",
                      f'fill="{MARGIN}" {BOLD}'))
    o.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{ANCHOR}"/>')
    o.append(text(cx, cy - 10, "z", 13, extra=f'fill="{ANCHOR}" {BOLD}'))
    o.append(text(cx, cy + RF + 22, f"arrows: gradient step on {sym}", 11.5))
save(__file__, o)
