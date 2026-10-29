"""Storing every activation vs gradient checkpointing (keep a few, recompute the rest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as RECOMP, CATEGORICAL, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

ACT = CATEGORICAL[0]
W, H = 840, 278
o = svg_open(W, H, [arrow_marker("mi", INK), arrow_marker("ma", RECOMP), hatch_pattern()])

PY, PH, PW = 10, 258, 409
L, BW, GAP = 8, 30, 14           # layers, block width, gap between blocks
LY, LH = 92, 40                  # layer row
AY, AH = 158, 24                 # activation row
CKPT = {0, 4}                    # layers whose input is kept as a checkpoint
SEG = (4, L - 1)                 # segment recomputed in the figure

for x0, title, subtitle, foot, foot_fill in [
        (6, "Store all activations", "every layer keeps its input for the backward pass", "memory for all 8 activations", INK),
        (425, "Gradient checkpointing", "keep only every 4th input, recompute the rest", "2 of 8 activations, plus an extra forward pass", RECOMP)]:
    ckpt = x0 > 6
    o.append(rect(x0, PY, PW, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + PW / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + PW / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + PW / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))

    lx = x0 + (PW - (L * (BW + GAP) - GAP)) / 2 + 10
    xs = [lx + k * (BW + GAP) for k in range(L)]
    o.append(line(lx - 26, LY + LH / 2, lx - 3, LY + LH / 2, INK, 'marker-end="url(#mi)"'))
    for k, x in enumerate(xs):
        o.append(rect(x, LY, BW, LH, INK, 4, f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'))
        o.append(text(x + BW / 2, LY + LH / 2 + 4.5, f"L{k + 1}", 12.5, extra='font-weight="600"'))
        if k < L - 1:
            o.append(line(x + BW + 1, LY + LH / 2, x + BW + GAP - 2, LY + LH / 2, INK, 'marker-end="url(#mi)"'))
        # activation (layer input) kept for backward, or dropped
        o.append(f'<line x1="{x + BW / 2}" y1="{LY + LH + 2}" x2="{x + BW / 2}" y2="{AY - 2}" stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"/>')
        if not ckpt or k in CKPT:
            o.append(rect(x + 2, AY, BW - 4, AH, ACT, 3))
        else:
            o.append(rect(x + 2, AY, BW - 4, AH, "url(#h)", 3))

    if ckpt:
        # recompute the dropped activations of one segment from its checkpoint
        a, b = SEG
        ry = AY + AH + 18
        x1, x2 = xs[a] + BW / 2, xs[b] + BW / 2
        o.append(f'<path d="M{x1},{AY + AH + 2} V{ry} H{x2 - 2}" fill="none" stroke="{RECOMP}" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#ma)"/>')
        o.append(text((x1 + x2) / 2, ry + 18, "recomputed during backward", 11.5, extra=f'fill="{RECOMP}"'))
        o.append(text((xs[0] + xs[3] + BW) / 2, ry + 18, "checkpoints kept", 11.5, extra=f'fill="{ACT}"'))
    else:
        o.append(text((xs[0] + xs[-1] + BW) / 2, AY + AH + 36, "all activations kept until backward", 11.5, extra=f'fill="{ACT}"'))

save(__file__, o)
