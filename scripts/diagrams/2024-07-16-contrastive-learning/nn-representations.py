"""A small fully connected network whose hidden layer outputs are the representations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as REPR, CATEGORICAL, INK, rect, save, svg_open, text

IN, OUT = CATEGORICAL[0], CATEGORICAL[1]
W, H = 840, 345
o = svg_open(W, H)

R, STEP, CY = 19, 52, 140      # node radius, vertical spacing, vertical centre of the layers
layers = [  # x, colour, label, values
    (110, IN, "input x", [0.60, 0.54, 0.58, 0.78]),
    (320, REPR, "representation 1", [0.91, 0.39, 0.76, 0.75, 0.19]),
    (530, REPR, "representation 2", [0.25, 0.46, 0.84, 0.13, 0.51]),
    (740, OUT, "output ŷ", [0.43, 0.55, 0.37])]
ys = [[CY + (k - (len(v) - 1) / 2) * STEP for k in range(len(v))] for _, _, _, v in layers]

# hidden layers sit on a soft band so they read as one vector each
for (x, c, _, v), y in zip(layers, ys):
    if c == REPR:
        o.append(rect(x - R - 9, y[0] - R - 9, 2 * R + 18, y[-1] - y[0] + 2 * R + 18, c, 10,
                      f'fill-opacity="0.1" stroke="{c}" stroke-opacity="0.6" stroke-width="1.2"'))

for (x0, *_), (x1, *_), y0s, y1s in zip(layers, layers[1:], ys, ys[1:]):
    for y0 in y0s:
        for y1 in y1s:
            o.append(f'<line x1="{x0 + R}" y1="{y0}" x2="{x1 - R}" y2="{y1}" stroke="{INK}" stroke-opacity="0.35" stroke-width="1.1"/>')

for (x, c, label, vals), y in zip(layers, ys):
    for yy, val in zip(y, vals):
        o.append(f'<circle cx="{x}" cy="{yy}" r="{R}" fill="{c}"/>')
        o.append(text(x, yy + 4, f"{val:.2f}", 11.5, extra='fill="#fff" font-weight="600"'))
    o.append(text(x, CY + 2 * STEP + R + 34, label, 13, extra=f'fill="{c}" font-weight="600"'))

o.append(text(W / 2, H - 12, "each hidden layer outputs a new vector representation of the input", 13,
              extra='font-weight="600"'))
save(__file__, o)
