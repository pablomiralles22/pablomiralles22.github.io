"""Pipeline parallelism schedule on 3 GPUs, with and without micro-batching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT, CATEGORICAL as GPU, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

U, RH, LW = 26, 26, 64          # time unit width, row height, label column width
W = LW + 24 * U + 20
o = svg_open(W, 330, [arrow_marker("a", ACCENT, "auto-start-reverse"), arrow_marker("i", INK), hatch_pattern()])


def chart(y0, title, blocks, total):
    out = [text(LW, y0, title, 14, "start", 'font-weight="600"')]
    for r in range(3):
        y = y0 + 12 + r * (RH + 6)
        out.append(text(LW - 10, y + RH / 2 + 5, f"GPU {r+1}", 13, "end"))
        out.append(rect(LW, y, total * U, RH, "none", 3, f'stroke="{INK}" stroke-opacity="0.35" stroke-dasharray="3 3"'))
        # hatch the idle gaps
        t = 0
        for (t0, dur, _, _) in sorted(blocks[r]) + [(total, 0, "", 0)]:
            if t0 > t:
                out.append(rect(LW + t * U + 1.5, y + 1.5, (t0 - t) * U - 3, RH - 3, "url(#h)", 3))
            t = t0 + dur
        for (t0, dur, lab, bwd) in blocks[r]:
            if bwd:
                out.append(rect(LW + t0 * U + 1.5, y + 1.5, dur * U - 3, RH - 3, GPU[r], 3, f'fill-opacity="0.15" stroke="{GPU[r]}" stroke-width="1.5"'))
            else:
                out.append(rect(LW + t0 * U + 1, y + 1, dur * U - 2, RH - 2, GPU[r], 3))
            out.append(text(LW + (t0 + dur / 2) * U, y + RH / 2 + 5, lab, 12, extra=f'fill="{GPU[r] if bwd else "#fff"}" font-weight="600"'))
    return out


# blocks per GPU: (start, duration, label, is_backward)
naive = [[(0, 4, "F", 0), (20, 4, "B", 1)],
         [(4, 4, "F", 0), (16, 4, "B", 1)],
         [(8, 4, "F", 0), (12, 4, "B", 1)]]
# backward: GPU3 starts at 6, GPU2 at 7, GPU1 at 8
gp = [[(r + i, 1, f"F{i+1}", 0) for i in range(4)] + [(6 + (2 - r) + i, 1, f"B{4-i}", 1) for i in range(4)] for r in range(3)]
o += chart(54, "Without micro-batching (whole batch at once)", naive, 24)
o += chart(196, "With 4 micro-batches", gp, 12)
# time axis
ay = 310
o.append(line(LW, ay, LW + 24 * U, ay, INK, 'marker-end="url(#i)"'))
o.append(text(LW + 24 * U, ay + 16, "time", 12, "end"))
# idle annotation
o.append(text(LW + 12 * U + 10, 196 + 12 + RH + 6 + RH / 2 + 5, "← finished in half the time", 12.5, "start"))
# key
kx = LW + 24 * U - 250
o.append(rect(kx, 10, 16, 16, INK, 3))
o.append(text(kx + 22, 23, "forward", 12.5, "start"))
o.append(rect(kx + 85.75, 10.75, 14.5, 14.5, INK, 3, f'fill-opacity="0.15" stroke="{INK}" stroke-width="1.5"'))
o.append(text(kx + 107, 23, "backward", 12.5, "start"))
o.append(rect(kx + 180, 10, 16, 16, "url(#h)", 3))
o.append(text(kx + 202, 23, "idle", 12.5, "start"))

save(__file__, o)
