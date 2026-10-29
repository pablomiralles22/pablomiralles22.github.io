"""KV cache with contiguous per-sequence reservations vs paged attention (fixed-size blocks on demand)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT, CATEGORICAL as SEQ, INK, arrow_marker, curve, hatch_pattern, rect, save, svg_open, text

W, H = 840, 350
o = svg_open(W, H, [arrow_marker("mi", INK), hatch_pattern()])

PY, PH, PW = 10, 330, 409
BW, BH, GX, GY = 34, 24, 6, 8    # physical block size and gaps
LW, LG = 30, 5                   # logical block width and gap
COLS = 4
GTOP = 100                       # top of the memory grid
NAMES = ["A", "B", "C"]
LEN = [3, 2, 3]                  # blocks used per sequence
LY = [GTOP + BH + GY / 2, GTOP + 3 * (BH + GY) + BH / 2 - (BH + GY) / 2, GTOP + 4 * (BH + GY) + 26]
PAGED = {0: [1, 6, 12], 1: [3, 9], 2: [4, 10, 15]}   # physical slots of each sequence's blocks


def block(x, y, w, s, j, outline=False):
    """Block j of sequence s: filled once allocated, outlined while it waits."""
    c, lab = SEQ[s], f"{NAMES[s]}{j + 1}"
    if outline:
        return [rect(x + 0.75, y + 0.75, w - 1.5, BH - 1.5, c, 3, f'fill-opacity="0.15" stroke="{c}" stroke-width="1.5"'),
                text(x + w / 2, y + BH / 2 + 4, lab, 11.5, extra=f'fill="{c}" font-weight="600"')]
    return [rect(x, y, w, BH, c, 3), text(x + w / 2, y + BH / 2 + 4, lab, 11.5, extra='fill="#fff" font-weight="600"')]


def slot(k):
    return GX0 + (k % COLS) * (BW + GX), GTOP + (k // COLS) * (BH + GY)


for x0, title, subtitle, foot, foot_fill in [
        (6, "Contiguous KV cache", "each sequence reserves room for its maximum length", "11 of 16 blocks reserved but empty, C has to wait", INK),
        (425, "Paged attention", "fixed-size blocks are allocated as the sequence grows", "blocks allocated on demand, so C fits too", ACCENT)]:
    paged = x0 > 6
    o.append(rect(x0, PY, PW, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + PW / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + PW / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + PW / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))

    LX0 = x0 + 50
    GX0 = x0 + 216
    o.append(text(LX0 + 40, GTOP - 16, "sequences", 12.5, extra='font-weight="600"'))
    o.append(text(GX0 + (COLS * (BW + GX) - GX) / 2, GTOP - 16, "GPU memory", 12.5, extra='font-weight="600"'))

    # the memory grid: what each slot holds
    owner = {}
    if paged:
        for s, ks in PAGED.items():
            for j, k in enumerate(ks):
                owner[k] = (s, j)
    else:
        for s in range(2):
            for j in range(8):
                owner[8 * s + j] = (s, j)
    for k in range(COLS * 4):
        x, y = slot(k)
        if k not in owner:
            o.append(rect(x + 0.75, y + 0.75, BW - 1.5, BH - 1.5, "none", 3, f'stroke="{INK}" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="3 3"'))
            continue
        s, j = owner[k]
        if j < LEN[s]:
            o += block(x, y, BW, s, j)
        else:
            o.append(rect(x + 0.75, y + 0.75, BW - 1.5, BH - 1.5, "url(#h)", 3, f'stroke="{SEQ[s]}" stroke-opacity="0.6" stroke-width="1.2"'))

    # the sequences, as lists of logical blocks
    for s in range(3):
        y = LY[s] - BH / 2
        o.append(text(LX0 - 12, LY[s] + 4.5, NAMES[s], 13, "end", f'fill="{SEQ[s]}" font-weight="600"'))
        for j in range(LEN[s]):
            lx = LX0 + j * (LW + LG)
            o += block(lx, y, LW, s, j, outline=not paged and s == 2)
        if not paged and s < 2:
            ex = LX0 + LEN[s] * (LW + LG) - LG
            o.append(curve(ex + 2, LY[s], GX0 - 4, LY[s], INK, 'marker-end="url(#mi)"'))
        if not paged and s == 2:
            o.append(text(LX0 + 140, LY[s] + 4.5, "no room left", 11.5, "start", f'fill="{SEQ[2]}"'))

    # legend
    ly = GTOP + 4 * (BH + GY) + 52
    if paged:
        lx = x0 + PW / 2 - 24
        o.append(rect(lx + 0.75, ly + 0.75, 14.5, 14.5, "none", 3, f'stroke="{INK}" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="3 3"'))
        o.append(text(lx + 22, ly + 12.5, "free", 12.5, "start"))
    else:
        lx = x0 + PW / 2 - 64
        o.append(rect(lx, ly, 16, 16, "url(#h)", 3, f'stroke="{INK}" stroke-opacity="0.6" stroke-width="1.2"'))
        o.append(text(lx + 22, ly + 12.5, "reserved, empty", 12.5, "start"))

save(__file__, o)
