"""Dense feed-forward vs mixture-of-experts layer with top-2 routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as ROUTER, CATEGORICAL as TOK, INK, arrow_marker, curve, rect, save, svg_open, text

W, H = 840, 320
o = svg_open(W, H, [arrow_marker(f"m{k}", c) for k, c in enumerate(TOK)])

PY, PH = 10, 300
for (x0, w, title, sub, foot) in [
        (6, 250, "Dense feed-forward", "every token goes through one block", "every token uses all weights"),
        (266, 568, "Mixture-of-experts feed-forward", "the router sends each token to its top-2 experts", "2× the weights, same compute per token")]:
    o.append(rect(x0, PY, w, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    cx = x0 + w / 2
    o.append(text(cx, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(cx, PY + 46, sub, 12.5))
    o.append(text(cx, PY + PH - 16, foot, 13, extra='font-weight="600"'))

WORDS = ["the", "cat", "sat"]
TY = [118, 178, 238]          # token centres
CW, CH = 44, 24


def tok_in(x, k):
    return [rect(x, TY[k] - CH / 2, CW, CH, TOK[k], 4),
            text(x + CW / 2, TY[k] + 4.5, WORDS[k], 13, extra='fill="#fff" font-weight="600"')]


def tok_out(x, k):
    return [rect(x + 0.75, TY[k] - CH / 2 + 0.75, CW - 1.5, CH - 1.5, TOK[k], 4, f'fill-opacity="0.15" stroke="{TOK[k]}" stroke-width="1.5"'),
            text(x + CW / 2, TY[k] + 4.5, WORDS[k], 13, extra=f'fill="{TOK[k]}" font-weight="600"')]


EH, EG = 36, 12
ET = 88
EC = [ET + i * (EH + EG) + EH / 2 for i in range(4)]
BLOCK = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'

# ---- Dense panel ----
fh = 2 * EH + EG                      # same size as two experts
fx, fw, fy = 98, 52, 178 - fh / 2
o.append(rect(fx, fy, fw, fh, INK, 5, BLOCK))
o.append(text(fx + fw / 2, 178 + 5, "FFN", 14, extra='font-weight="600"'))
for k in range(3):
    o += tok_in(24, k)
    o.append(curve(24 + CW, TY[k], fx, 178 + (k - 1) * 14, TOK[k]))
    o.append(curve(fx + fw, 178 + (k - 1) * 14, 200, TY[k], TOK[k], f'marker-end="url(#m{k})"'))
    o += tok_out(202, k)

# ---- MoE panel ----
route = {0: [0, 2], 1: [1, 2], 2: [1, 3]}       # token -> experts
users = {e: [k for k in range(3) if e in route[k]] for e in range(4)}


def port(e, k):
    u = users[e]
    return EC[e] + (u.index(k) - (len(u) - 1) / 2) * 10


tx, rx, rw = 290, 378, 14
ex, ew = 486, 100
cx_ = 680
ox = 740
o.append(rect(rx, TY[0] - 22, rw, TY[2] - TY[0] + 44, ROUTER, 4))
o.append(text(rx + rw / 2, TY[0] - 30, "router", 12.5, extra=f'fill="{ROUTER}" font-weight="600"'))
for e in range(4):
    o.append(rect(ex, EC[e] - EH / 2, ew, EH, INK, 5, BLOCK))
    o.append(text(ex + ew / 2, EC[e] + 5, f"Expert {e+1}", 13, extra='font-weight="600"'))
for k in range(3):
    o += tok_in(tx, k)
    o.append(f'<line x1="{tx + CW}" y1="{TY[k]}" x2="{rx}" y2="{TY[k]}" stroke="{TOK[k]}" stroke-width="1.8"/>')
    for e in route[k]:
        o.append(curve(rx + rw, TY[k], ex, port(e, k), TOK[k]))
        o.append(curve(ex + ew, port(e, k), cx_ - 10, TY[k], TOK[k]))
    o.append(f'<circle cx="{cx_}" cy="{TY[k]}" r="10" fill="none" stroke="{TOK[k]}" stroke-width="1.8"/>')
    o.append(f'<path d="M{cx_ - 5},{TY[k]} h10 M{cx_},{TY[k] - 5} v10" stroke="{TOK[k]}" stroke-width="1.8"/>')
    o.append(f'<line x1="{cx_ + 10}" y1="{TY[k]}" x2="{ox - 2}" y2="{TY[k]}" stroke="{TOK[k]}" stroke-width="1.8" marker-end="url(#m{k})"/>')
    o += tok_out(ox, k)
o.append(text(cx_, TY[2] + 36, "weighted sum", 11.5))

save(__file__, o)
