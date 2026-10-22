"""Data / pipeline / tensor parallelism: what each scheme splits and how often GPUs communicate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT, CATEGORICAL as GPU, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

W, H = 840, 330
o = svg_open(W, H, [arrow_marker("a", ACCENT, "auto-start-reverse"), arrow_marker("i", INK), hatch_pattern()])
# legend
lx = 250
for g in range(3):
    o.append(rect(lx, 12, 14, 14, GPU[g]))
    o.append(text(lx + 20, 24, f"GPU {g+1}", 13, "start"))
    lx += 82
o.append(line(lx, 19, lx + 30, 19, ACCENT, 'stroke-dasharray="4 3" marker-end="url(#a)"'))
o.append(text(lx + 38, 24, "communication", 13, "start"))

PY, PH = 40, 280
for p, (title, sub, foot) in enumerate([
        ("Data parallelism", "full copy per GPU, batch is split", "sync gradients once per step"),
        ("Pipeline parallelism", "model split by depth (layers)", "send activations between stages"),
        ("Tensor parallelism", "every layer split by width", "sync inside every layer")]):
    x0 = p * 280 + 6
    o.append(rect(x0, PY, 268, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    cx = x0 + 134
    o.append(text(cx, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(cx, PY + 46, sub, 12.5))
    o.append(text(cx, PY + PH - 16, foot, 13, extra=f'fill="{ACCENT}" font-weight="600"'))


def grid(x, y, b, gx, gy, colors_fn, cols=6, rows=3):
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(rect(x + c * (b + gx), y + r * (b + gy), b, b, colors_fn(r, c), 2))
    return out


# Data panel
b, g = 14, 3
gw, gh = 6 * b + 5 * g, 3 * b + 2 * g
gx0 = 6 + 134 - gw / 2 + 4
ys = [100, 154, 208]
for k, y in enumerate(ys):
    o += grid(gx0, y, b, g, g, lambda r, c, k=k: GPU[k])
    my = y + gh / 2
    o.append(line(gx0 - 34, my, gx0 - 6, my, INK, 'marker-end="url(#i)"'))
    o.append(text(gx0 - 44, my + 5, f"x<tspan baseline-shift=\"sub\" font-size=\"10\">{k+1}</tspan>", 14))
    o.append(line(gx0 + gw + 4, my, gx0 + gw + 22, my, ACCENT, 'stroke-dasharray="4 3"'))
bx = gx0 + gw + 22
o.append(text(gx0 + gw / 2, 274, "layers →", 11.5))
o.append(line(bx, ys[0] + gh / 2, bx, ys[2] + gh / 2, ACCENT, 'stroke-dasharray="4 3"'))
ry = (ys[0] + ys[2] + gh) / 2
o.append(text(bx + 10, ry, "all-reduce", 11.5, "middle", f'fill="{ACCENT}" transform="rotate(90 {bx + 10} {ry})"'))

# Pipeline panel
b, g, bg = 24, 4, 26
band = 2 * b + g
tw = 3 * band + 2 * bg
th = 3 * b + 2 * g
px0 = 286 + 134 - tw / 2 + 14
py0 = 138
for s in range(3):
    bx0 = px0 + s * (band + bg)
    o += grid(bx0, py0, b, g, g, lambda r, c, s=s: GPU[s], cols=2)
    o.append(text(bx0 + band / 2, py0 + th + 18, f"layers {2*s+1}–{2*s+2}", 11.5))
    if s < 2:
        o.append(line(bx0 + band + 3, py0 + th / 2, bx0 + band + bg - 3, py0 + th / 2, ACCENT, 'stroke-dasharray="4 3" marker-end="url(#a)"'))
o.append(line(px0 - 24, py0 + th / 2, px0 - 5, py0 + th / 2, INK, 'marker-end="url(#i)"'))
o.append(text(px0 - 31, py0 + th / 2 + 5, "x", 14))
o.append(text(px0 + tw / 2, 274, "layers →", 11.5))

# Tensor panel
b, cg, rg = 24, 10, 4
tw = 6 * b + 5 * cg
th = 3 * b + 2 * rg
tx0 = 620
ty0 = 138
o += grid(tx0, ty0, b, cg, rg, lambda r, c: GPU[r])
for c in range(6):
    sx = tx0 + c * (b + cg) + b + (cg / 2 if c < 5 else 6)
    o.append(line(sx, ty0 - 6, sx, ty0 + th + 6, ACCENT, 'stroke-dasharray="3 3"'))
# broadcast input
ix = tx0 - 16
o.append(line(ix - 14, ty0 + th / 2, ix, ty0 + th / 2, INK))
o.append(line(ix, ty0 + b / 2, ix, ty0 + th - b / 2, INK))
for r in range(3):
    yy = ty0 + r * (b + rg) + b / 2
    o.append(line(ix, yy, tx0 - 4, yy, INK, 'marker-end="url(#i)"'))
o.append(text(ix - 22, ty0 + th / 2 + 5, "x", 14))
o.append(text(tx0 + tw / 2, 274, "layers →", 11.5))

save(__file__, o)
