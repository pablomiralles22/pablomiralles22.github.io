"""One batch of 8 vs 4 accumulated micro-batches of 2: same update, less activation memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as GRAD, CATEGORICAL, INK, arrow_marker, line, rect, save, svg_open, text

SAMPLE = CATEGORICAL[0]
W, H = 840, 330
o = svg_open(W, H, [arrow_marker("mi", INK), arrow_marker("ma", GRAD)])

PY, PH = 10, 310
BLOCK = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
GBUF = f'fill-opacity="0.15" stroke="{GRAD}" stroke-width="1.5"'
S, SG = 12, 4                 # sample square size, gap
CY = 122                      # centre of the samples / F+B row
MY = 264                      # activation memory bar
U = 18                        # memory bar width per sample

panels = [(6, 330, "One batch of 8", "one forward/backward pass over all samples",
           "activations of all 8 samples at once", INK),
          (346, 488, "4 micro-batches of 2", "add up their gradients, then take one step",
           "same update with ¼ of the activations, but slower", GRAD)]
for x0, w, title, subtitle, foot, foot_fill in panels:
    o.append(rect(x0, PY, w, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + w / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + w / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + w / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))


def samples(x, rows, cols):
    """Grid of sample squares centred on CY; returns its right edge."""
    top = CY - (rows * (S + SG) - SG) / 2
    for r in range(rows):
        for c in range(cols):
            o.append(rect(x + c * (S + SG), top + r * (S + SG), S, S, SAMPLE, 2))
    return x + cols * (S + SG) - SG


def fb(x, h, w=48):
    o.append(rect(x, CY - h / 2, w, h, INK, 4, BLOCK))
    o.append(text(x + w / 2, CY + 4.5, "F + B", 12.5, extra='font-weight="600"'))
    return x + w


def step(x, y, w=50, h=30):
    o.append(rect(x, y - h / 2, w, h, GRAD, 4))
    o.append(text(x + w / 2, y + 4.5, "step", 13, extra='fill="#fff" font-weight="600"'))


def memory(x, n):
    o.append(text(x, MY - 8, "activation memory", 11.5, "start"))
    o.append(rect(x, MY, n * U, 14, INK, 3, BLOCK))


# ---- One batch ----
x0 = panels[0][0] + 44
right = samples(x0 + 24, 4, 2)
o.append(line(right + 4, CY, x0 + 80, CY, INK, 'marker-end="url(#mi)"'))
right = fb(x0 + 82, 4 * (S + SG) - SG + 8)
o.append(line(right + 4, CY, x0 + 156, CY, INK, 'marker-end="url(#mi)"'))
o.append(rect(x0 + 158, CY - 15, 30, 30, GRAD, 4, GBUF))
o.append(text(x0 + 173, CY + 5, "g", 14, extra=f'fill="{GRAD}" font-weight="600"'))
o.append(line(x0 + 192, CY, x0 + 212, CY, GRAD, 'marker-end="url(#ma)"'))
step(x0 + 214, CY)
memory(x0 + 24, 8)

# ---- Accumulated micro-batches ----
x0 = panels[1][0]
BY, BH = 182, 26              # gradient buffer
bx0, pitch = x0 + 24, 92
for k in range(4):
    x = bx0 + k * pitch
    right = samples(x, 2, 1)
    o.append(line(right + 4, CY, x + 30, CY, INK, 'marker-end="url(#mi)"'))
    fx = x + 32
    fb(fx, 2 * (S + SG) - SG + 8)
    o.append(line(fx + 24, CY + 18, fx + 24, BY - 4, GRAD, 'marker-end="url(#ma)"'))
    o.append(text(fx + 31, (CY + 18 + BY) / 2 + 4, "+=", 12.5, "start", f'fill="{GRAD}" font-weight="600"'))
bx1 = bx0 + 3 * pitch + 80
o.append(rect(bx0, BY, bx1 - bx0, BH, GRAD, 4, GBUF))
o.append(text((bx0 + bx1) / 2, BY + BH / 2 + 5, "gradient buffer g", 13, extra=f'fill="{GRAD}" font-weight="600"'))
o.append(line(bx1 + 4, BY + BH / 2, bx1 + 26, BY + BH / 2, GRAD, 'marker-end="url(#ma)"'))
step(bx1 + 28, BY + BH / 2)
# time axis under the micro-batches
ty = BY + BH + 20
o.append(line(bx0, ty, bx1, ty, INK, 'marker-end="url(#mi)"'))
o.append(text(bx1 + 6, ty + 4, "time", 11.5, "start"))
memory(bx0, 2)

save(__file__, o)
