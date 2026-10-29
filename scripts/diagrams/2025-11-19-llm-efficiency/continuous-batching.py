"""Static batching (wait for the longest request) vs continuous batching (refill free slots right away)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as JOIN, CATEGORICAL, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

U, RH, LW = 52, 24, 70          # time unit width, row height, label column width
STEPS = 14
W = 840
o = svg_open(W, 416, [arrow_marker("i", INK), hatch_pattern()])

LENS = [2, 8, 4, 5, 5, 3, 6, 2]  # decode steps of requests R1..R8
SLOTS = 4
REQ = CATEGORICAL[0]


def static():
    """Two fixed batches of 4; the second starts when the longest of the first ends."""
    rows = [[] for _ in range(SLOTS)]
    t = 0
    for b in range(2):
        batch = range(4 * b, 4 * b + 4)
        for s, r in enumerate(batch):
            rows[s].append((t, r))
        t += max(LENS[r] for r in batch)
    return rows, t


def continuous():
    """Each new request takes the first slot that frees up."""
    rows = [[(0, r)] for r in range(SLOTS)]
    free = [LENS[r] for r in range(SLOTS)]
    for r in range(SLOTS, len(LENS)):
        s = min(range(SLOTS), key=lambda k: (free[k], k))
        rows[s].append((free[s], r))
        free[s] += LENS[r]
    return rows, max(free)


def chart(y0, title, rows, total, joins):
    out = [text(LW, y0, title, 14, "start", 'font-weight="600"')]
    for s, row in enumerate(rows):
        y = y0 + 12 + s * (RH + 6)
        out.append(text(LW - 10, y + RH / 2 + 5, f"slot {s + 1}", 13, "end"))
        out.append(rect(LW, y, STEPS * U, RH, "none", 3, f'stroke="{INK}" stroke-opacity="0.35" stroke-dasharray="3 3"'))
        t = 0
        for t0, r in row + [(total, None)]:
            if t0 > t:
                out.append(rect(LW + t * U + 1.5, y + 1.5, (t0 - t) * U - 3, RH - 3, "url(#h)", 3))
            if r is None:
                break
            out.append(rect(LW + t0 * U + 1, y + 1, LENS[r] * U - 2, RH - 2, REQ, 3))
            out.append(text(LW + (t0 + LENS[r] / 2) * U, y + RH / 2 + 5, f"R{r + 1}", 12, extra='fill="#fff" font-weight="600"'))
            if joins and t0 > 0:
                x = LW + t0 * U
                out.append(f'<path d="M{x - 5},{y - 4} L{x + 5},{y - 4} L{x},{y + 3} z" fill="{JOIN}"/>')
            t = t0 + LENS[r]
    end = y0 + 12 + SLOTS * (RH + 6) - 6
    x = LW + total * U
    out.append(f'<line x1="{x}" y1="{y0 + 6}" x2="{x}" y2="{end + 6}" stroke="{JOIN if joins else INK}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    out.append(text(x, end + 24, f"all done after {total} steps", 12.5, "end", f'fill="{JOIN}" font-weight="600"' if joins else 'font-weight="600"'))
    return out


rows, total = static()
o += chart(54, "Static batching: the next batch waits for the longest request", rows, total, False)
rows, total = continuous()
o += chart(228, "Continuous batching: a new request joins as soon as a slot frees up", rows, total, True)

# time axis
ay = 394
o.append(line(LW, ay, LW + STEPS * U, ay, INK, 'marker-end="url(#i)"'))
o.append(text(LW + STEPS * U, ay + 16, "time (decoding steps)", 12, "end"))
# key
kx = LW + STEPS * U - 250
o.append(f'<path d="M{kx + 3},{14} L{kx + 13},{14} L{kx + 8},{21} z" fill="{JOIN}"/>')
o.append(text(kx + 22, 23, "new request joins", 12.5, "start"))
o.append(rect(kx + 160, 10, 16, 16, "url(#h)", 3))
o.append(text(kx + 182, 23, "idle", 12.5, "start"))

save(__file__, o)
