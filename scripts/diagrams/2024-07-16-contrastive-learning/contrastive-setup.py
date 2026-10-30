"""Two inputs go through encoder and projection head; the loss compares their embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as LOSS, CATEGORICAL, INK, arrow_marker, curve, line, rect, save, svg_open, text

W, H = 840, 284
o = svg_open(W, H, [arrow_marker("mi", INK), arrow_marker("m0", CATEGORICAL[0]),
                    arrow_marker("m1", CATEGORICAL[1]), arrow_marker("ma", LOSS)])

BOLD = 'font-weight="600"'
WHITE = 'fill="#fff" font-weight="600"'
NEUTRAL = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
CELL = 11
ROWS = [95, 225]
MID = sum(ROWS) / 2

ix, iw = 24, 44        # input
ex, bw, bh = 110, 90, 50  # encoder, block width/height
vx = 255               # representation (d cells)
hx = 345               # projection head
zx = 487               # embedding (m cells)
sx, sw = 575, 120      # similarity / distance
lx, lw = 755, 60       # loss
VW = 16

for x, s in [(ix + iw / 2, "input"), (ex + bw / 2, "encoder"), (vx + VW / 2, "representation"),
             (hx + bw / 2, "projection head"), (zx + VW / 2, "embedding")]:
    o.append(text(x, 28, s, 12.5))


def vector(x, cy, n, c):
    for k in range(n):
        o.append(rect(x + 0.75, cy + (k - n / 2) * CELL + 0.75, VW - 1.5, CELL - 1.5, c, 2,
                      f'fill-opacity="0.15" stroke="{c}" stroke-width="1.5"'))


for i, (cy, c) in enumerate(zip(ROWS, CATEGORICAL)):
    n = "₁₂"[i]
    m = f'marker-end="url(#m{i})"'
    o.append(rect(ix, cy - 14, iw, 28, c, 4))
    o.append(text(ix + iw / 2, cy + 5, f"x{n}", 14, extra=WHITE))
    o.append(line(ix + iw, cy, ex - 2, cy, c, m))
    o.append(rect(ex, cy - bh / 2, bw, bh, INK, 5, NEUTRAL))
    o.append(text(ex + bw / 2, cy + 6, f"e{n}", 16, extra=BOLD))
    o.append(line(ex + bw, cy, vx - 4, cy, c, m))
    vector(vx, cy, 5, c)
    o.append(text(vx + VW / 2, cy + 5 * CELL / 2 + 17, f"v{n}", 13, extra=f'fill="{c}" {BOLD}'))
    o.append(line(vx + VW + 2, cy, hx - 2, cy, c, m))
    o.append(rect(hx, cy - bh / 2, bw, bh, INK, 5, NEUTRAL))
    o.append(text(hx + bw / 2, cy + 6, f"h{n}", 16, extra=BOLD))
    o.append(line(hx + bw, cy, zx - 4, cy, c, m))
    vector(zx, cy, 3, c)
    o.append(text(zx + VW / 2, cy + 3 * CELL / 2 + 17, f"z{n}", 13, extra=f'fill="{c}" {BOLD}'))
    o.append(curve(zx + VW + 2, cy, sx - 2, MID + (i - 0.5) * 16, c, m))

# note: the two encoders (and the two heads) are often the same network
nx, ny = vx + VW / 2, MID + 12
for x in (ex + bw / 2, hx + bw / 2):
    o.append(line(x, ROWS[0] + bh / 2 + 4, x, ROWS[1] - bh / 2 - 4, INK, 'stroke-dasharray="4 3"'))
o.append(text(nx, ny + 4, "often shared", 11.5))
o.append(line(nx - 42, ny, ex + bw / 2 + 4, ny, INK, 'stroke-dasharray="4 3" marker-end="url(#mi)"'))
o.append(line(nx + 42, ny, hx + bw / 2 - 4, ny, INK, 'stroke-dasharray="4 3" marker-end="url(#mi)"'))

o.append(rect(sx, MID - bh / 2, sw, bh, INK, 5, NEUTRAL))
o.append(text(sx + sw / 2, MID + 6, "S or D", 16, extra=BOLD))
o.append(text(sx + sw / 2, MID + bh / 2 + 17, "similarity or distance", 11.5))
o.append(line(sx + sw, MID, lx - 2, MID, LOSS, 'marker-end="url(#ma)"'))
o.append(rect(lx, MID - 20, lw, 40, LOSS, 5))
o.append(text(lx + lw / 2, MID + 6, "ℒ", 17, extra=WHITE))
o.append(text(lx + lw / 2, MID + 37, "loss", 11.5, extra=f'fill="{LOSS}"'))

# note on what the loss does, pointing down at it
for k, s in enumerate(["pulls similar pairs together,", "pushes dissimilar ones apart"]):
    o.append(text(lx + lw, 86 + 15 * k, s, 11.5, "end", f'fill="{LOSS}"'))
o.append(line(lx + lw / 2, 109, lx + lw / 2, MID - 20 - 3, LOSS, 'stroke-dasharray="4 3" marker-end="url(#ma)"'))
save(__file__, o)
