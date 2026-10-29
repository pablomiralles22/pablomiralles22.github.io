"""Serving 3 LoRA adapters: one merged model each vs one shared base model plus small adapters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT, CATEGORICAL as ADP, INK, arrow_marker, curve, rect, save, svg_open, text

W, H = 840, 330
o = svg_open(W, H, [arrow_marker(f"m{k}", c) for k, c in enumerate(ADP)])

PY, PH, PW = 10, 310, 409
NEUTRAL = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
RW, RH = 34, 20                  # request chip
BW, BH = 130, 54                 # base model block
SW, SH = 14, 14                  # adapter strip
ROWS = [112, 180, 248]           # row centres in the left panel


def model(x, yc, adapters, sub):
    """Base model block with small adapter strips attached to its right edge."""
    out = [rect(x, yc - BH / 2, BW, BH, INK, 5, NEUTRAL),
           text(x + BW / 2, yc + 1, "base model", 13, extra='font-weight="600"'),
           text(x + BW / 2, yc + 17, sub, 11.5)]
    y0 = yc - (len(adapters) * (SH + 3) - 3) / 2
    for j, k in enumerate(adapters):
        out.append(rect(x + BW + 3, y0 + j * (SH + 3), SW, SH, ADP[k], 2))
    return out


for x0, title, subtitle, foot, foot_fill in [
        (6, "One model per adapter", "each adapter is merged into its own copy", "3 copies of the base model in memory", INK),
        (425, "Shared base model", "adapters are applied per request at runtime", "1 base model + 3 small adapters, one batch", ACCENT)]:
    shared = x0 > 6
    o.append(rect(x0, PY, PW, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + PW / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + PW / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + PW / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))

    mx = x0 + 186
    if not shared:
        o.append(text(x0 + 48 + RW + 3, ROWS[0] - RH / 2 - 10, "requests", 11.5))
        for k, yc in enumerate(ROWS):
            for j in range(2):
                cx = x0 + 48 + j * (RW + 6)
                o.append(rect(cx, yc - RH / 2, RW, RH, ADP[k], 4))
            o.append(curve(x0 + 48 + 2 * RW + 6 + 2, yc, mx - 2, yc, ADP[k], f'marker-end="url(#m{k})"'))
            o += model(mx, yc, [k], "copy " + str(k + 1))
            o.append(text(mx + BW + SW + 10, yc + 4, f"LoRA {k + 1}", 11.5, "start", f'fill="{ADP[k]}"'))
    else:
        yc = ROWS[1]
        mix = [0, 1, 0, 2, 1, 2]
        cx = x0 + 68
        for j, k in enumerate(mix):
            y = yc + (j - 2.5) * (RH + 8)
            o.append(rect(cx, y - RH / 2, RW, RH, ADP[k], 4))
            o.append(curve(cx + RW + 2, y, mx - 2, yc + (j - 2.5) * 7, ADP[k], f'marker-end="url(#m{k})"'))
        o += model(mx, yc, [0, 1, 2], "shared")
        o.append(text(cx + RW / 2, yc - 3 * (RH + 8) - 4, "mixed batch", 11.5))
        o.append(text(mx + BW + SW + 10, yc + 4, "LoRA 1–3", 11.5, "start"))

save(__file__, o)
