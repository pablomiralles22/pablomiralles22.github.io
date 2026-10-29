"""LoRA (frozen W plus trainable low-rank AB) vs soft prompting (trainable embeddings prepended)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as TRAIN, CATEGORICAL, INK, arrow_marker, curve, line, rect, save, svg_open, text

TOK = CATEGORICAL[0]
W, H = 840, 370
o = svg_open(W, H, [arrow_marker("mi", INK), arrow_marker("mt", TOK), arrow_marker("ma", TRAIN)])

PY, PH = 10, 350
FROZEN = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
BOLD = 'font-weight="600"'
WHITE = 'fill="#fff" font-weight="600"'


def label(x, y, parts, size, extra=""):
    """Text roughly centred at x made of (text, subscript) parts. It is left-anchored with
    dy tspans, because cairosvg misplaces tspans in middle-anchored text."""
    k = 0.58 if "font-weight" in extra else 0.56
    width = sum(k * size * len(s) + 0.5 * size * len(i) for s, i in parts)
    body, low = "", False
    for s, i in parts:
        body += f'<tspan dy="-3">{s}</tspan>' if low else s
        low = bool(i)
        if i:
            body += f'<tspan dy="3" font-size="{size * 0.72:.1f}">{i}</tspan>'
    return [text(x - width / 2, y, body, size, "start", extra)]


for x0, w, title, subtitle, foot in [
        (6, 470, "LoRA", "add a trainable low-rank update AB to a frozen W",
         [("trains r(n", "in"), (" + n", "out"), (") instead of n", "in"), ("·n", "out"), (" weights", "")]),
        (486, 348, "Soft prompting", "prepend trainable embeddings to every input",
         [("trains only a few embeddings", "")])]:
    o.append(rect(x0, PY, w, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + w / 2, PY + 26, title, 16, extra=BOLD))
    o.append(text(x0 + w / 2, PY + 46, subtitle, 12.5))
    o += label(x0 + w / 2, PY + PH - 16, foot, 13, f'fill="{TRAIN}" {BOLD}')

# ---- LoRA ----
dx = 30
S, R = 100, 16                 # W is S x S (n_in x n_out), the rank r is R
wy, ay = 80, 210               # tops of W and A
wc, ac = wy + S / 2, ay + S / 2
xc = (wc + ac) / 2
CW, CH = 44, 24
xx = 28 + dx
ax_, wx = 140 + dx, 180 + dx   # A and W / B left edges
sx, hx = 340 + dx, 380 + dx    # sum and output

o.append(rect(xx, xc - CH / 2, CW, CH, TOK, 4))
o.append(text(xx + CW / 2, xc + 4.5, "x", 13, extra=WHITE))
o.append(curve(xx + CW, xc, wx - 2, wc, INK, 'marker-end="url(#mi)"'))
o.append(curve(xx + CW, xc, ax_ - 2, ac, INK, 'marker-end="url(#mi)"'))

o.append(rect(wx, wy, S, S, INK, 5, FROZEN))
o.append(text(wx + S / 2, wc + 2, "W", 16, extra=BOLD))
o.append(text(wx + S / 2, wc + 20, "frozen", 11.5))

o.append(rect(ax_, ay, R, S, TRAIN, 3))
o.append(text(ax_ + R / 2, ac + 4.5, "A", 13, extra=WHITE))
o.append(rect(wx, ac - R / 2, S, R, TRAIN, 3))
o.append(text(wx + S / 2, ac + 4.5, "B", 13, extra=WHITE))
o.append(line(ax_ + R, ac, wx - 2, ac, TRAIN, 'marker-end="url(#ma)"'))
o.append(text(wx + S / 2, ac + R / 2 + 22, "trainable, rank r", 11.5, extra=f'fill="{TRAIN}"'))

o.append(curve(wx + S, wc, sx - 10, xc, INK))
o.append(curve(wx + S, ac, sx - 10, xc, TRAIN))
o.append(f'<circle cx="{sx}" cy="{xc}" r="10" fill="none" stroke="{INK}" stroke-width="1.8"/>')
o.append(f'<path d="M{sx - 5},{xc} h10 M{sx},{xc - 5} v10" stroke="{INK}" stroke-width="1.8"/>')
o.append(line(sx + 10, xc, hx - 2, xc, INK, 'marker-end="url(#mi)"'))
o.append(rect(hx + 0.75, xc - CH / 2 + 0.75, CW - 1.5, CH - 1.5, TOK, 4, f'fill-opacity="0.15" stroke="{TOK}" stroke-width="1.5"'))
o.append(text(hx + CW / 2, xc + 4.5, "h", 13, extra=f'fill="{TOK}" {BOLD}'))

# ---- Soft prompting ----
tx, mx, mw = 530, 650, 140
ty0, step = 96, 30
seq = [("p", "1"), ("p", "2"), ("p", "3"), ("the", ""), ("cat", ""), ("sat", ""), ("on", "")]
for k, (s, i) in enumerate(seq):
    y = ty0 + k * step
    c = TRAIN if i else TOK
    o.append(rect(tx, y, CW, CH, c, 4))
    o += label(tx + CW / 2, y + CH / 2 + 4.5, [(s, i)], 13, WHITE)
    o.append(line(tx + CW, y + CH / 2, mx - 2, y + CH / 2, c, f'marker-end="url(#{"ma" if i else "mt"})"'))
my0, my1 = ty0 - 6, ty0 + (len(seq) - 1) * step + CH + 6
o.append(text(tx + CW / 2, ty0 - 12, "soft prompt", 11.5, extra=f'fill="{TRAIN}"'))
o.append(text(tx + CW / 2, my1 + 12, "input tokens", 11.5, extra=f'fill="{TOK}"'))
o.append(rect(mx, my0, mw, my1 - my0, INK, 5, FROZEN))
o.append(text(mx + mw / 2, (my0 + my1) / 2, "LLM", 16, extra=BOLD))
o.append(text(mx + mw / 2, (my0 + my1) / 2 + 18, "frozen", 11.5))

save(__file__, o)
