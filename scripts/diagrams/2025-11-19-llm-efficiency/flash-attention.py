"""Standard attention vs FlashAttention: full N x N score matrix vs one tile at a time."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as SRAM, CATEGORICAL, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

Q, K, O = CATEGORICAL
W, H = 840, 396
o = svg_open(W, H, [arrow_marker("mi", INK), hatch_pattern()])

PY, PH, PW = 10, 376, 409
N, C = 4, 44                 # blocks per side, cell size
BW = 28                      # width of the Q / K / O strips
OUTLINE = 'fill-opacity="0.15" stroke-width="1.5"'


def sub(x, y, s, i, size, extra=""):
    """Label s_i centred at x; the subscript is its own <text> so every renderer places it alike."""
    sw = (0.8 if "font-weight" in extra else 0.62) * size * len(s)
    iw = 0.45 * size * len(i)
    left = x - (sw + iw) / 2
    return [text(left, y, s, size, "start", extra),
            text(left + sw, y + 3, i, size * 0.72, "start", extra)]


for x0, title, subtitle, foot, foot_fill in [
        (6, "Standard attention", "compute all scores, softmax them, then multiply by V", "N×N scores written to and read from HBM", INK),
        (425, "FlashAttention", "per Q block, stream K/V blocks through on-chip SRAM", "one tile at a time: memory linear in N", SRAM)]:
    flash = x0 > 6
    o.append(rect(x0, PY, PW, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + PW / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + PW / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + PW / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))

    qx = x0 + 48                      # Q strip
    gx, gy = qx + BW + 8, 128         # score grid
    ky = gy - 8 - BW                  # K strip
    ox = gx + N * C + 52              # O strip
    ai, aj = 1, 2                     # active tile in the FlashAttention panel

    o.append(text(qx - 10, gy + N * C / 2 + 5, "Q", 13, "end", 'font-weight="600"'))
    o.append(text(gx + N * C / 2, ky - 10, "Kᵀ", 13, extra='font-weight="600"'))
    o.append(text(ox + BW / 2, gy - 10, "O", 13, extra='font-weight="600"'))

    if not flash:
        # everything materialized at once: one block per matrix
        o.append(rect(qx + 1, gy + 2, BW - 2, N * C - 4, Q, 3))
        o.append(rect(gx + 2, ky + 1, N * C - 4, BW - 2, K, 3))
        o.append(rect(ox + 1, gy + 2, BW - 2, N * C - 4, O, 3))
        o.append(rect(gx + 2, gy + 2, N * C - 4, N * C - 4, INK, 5, f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'))
        o.append(text(gx + N * C / 2, gy + N * C / 2 + 6, "N×N", 18, extra='font-weight="600"'))
    else:
        for k in range(N):
            cy, cx = gy + k * C, gx + k * C
            o.append(rect(qx + 1, cy + 2, BW - 2, C - 4, Q, 3, "" if k == ai else f'{OUTLINE} stroke="{Q}"'))
            o.append(rect(cx + 2, ky + 1, C - 4, BW - 2, K, 3, "" if k == aj else f'{OUTLINE} stroke="{K}"'))
            o.append(rect(ox + 1, cy + 2, BW - 2, C - 4, O, 3, "" if k == ai else f'{OUTLINE} stroke="{O}"'))
        white = 'fill="#fff" font-weight="600"'
        o += sub(qx + BW / 2, gy + ai * C + C / 2 + 4, "Q", "i", 13, white)
        o += sub(gx + aj * C + C / 2, ky + BW / 2 + 4, "K", "j", 13, white)
        o += sub(ox + BW / 2, gy + ai * C + C / 2 + 4, "O", "i", 13, white)

        # score grid
        for r in range(N):
            for c in range(N):
                x, y = gx + c * C + 2, gy + r * C + 2
                if (r, c) == (ai, aj):
                    o.append(rect(x, y, C - 4, C - 4, SRAM, 3))
                    o += sub(x + C / 2 - 2, y + C / 2 + 3, "S", "ij", 14, 'fill="#fff" font-weight="600"')
                elif r == ai and c < aj:
                    o.append(rect(x, y, C - 4, C - 4, "url(#h)", 3))
                else:
                    o.append(rect(x, y, C - 4, C - 4, "none", 3, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2" stroke-dasharray="4 3"'))

    # multiply by V into O
    if flash:
        y = gy + ai * C + C / 2
        o.append(line(gx + N * C + 4, y, ox - 4, y, INK, 'marker-end="url(#mi)"'))
        o += sub((gx + N * C + ox) / 2, y - 8, "× V", "j", 11.5)
        o.append(text(ox + BW + 8, y + 4, "updated", 11.5, "start"))
        o.append(text(ox + BW + 8, y + 18, "per tile", 11.5, "start"))
        # sweep over K/V blocks
        by = gy + N * C + 22
        o.append(line(gx + 4, by, gx + N * C - 4, by, INK, 'marker-end="url(#mi)"'))
        o.append(text(gx + N * C / 2, by + 18, "loop over K/V blocks (Q blocks run in parallel)", 11.5))
    else:
        y = gy + N * C / 2
        o.append(line(gx + N * C + 4, y, ox - 4, y, INK, 'marker-end="url(#mi)"'))
        o.append(text((gx + N * C + ox) / 2, y - 8, "× V", 11.5))
        o.append(text(gx + N * C / 2, gy + N * C + 26, "S = QKᵀ and P = softmax(S), both N×N", 11.5))

save(__file__, o)
