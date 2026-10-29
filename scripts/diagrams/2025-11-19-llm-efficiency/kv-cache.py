"""Generation without a KV cache (recompute keys/values of every token at every step) vs with one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as CACHE, CATEGORICAL, INK, arrow_marker, line, rect, save, svg_open, text

TOK = CATEGORICAL[0]
W, H = 840, 330
o = svg_open(W, H, [arrow_marker("mi", INK)])

PY, PH, PW = 10, 310, 409
WORDS = ["The", "cat", "sat", "on", "the", "mat"]
N = 5                            # generation steps
CW, CH, GX, GY = 40, 24, 6, 8    # cell size and gaps
RY = 96                          # top of the first row
OUTLINE = f'fill-opacity="0.15" stroke="{CACHE}" stroke-width="1.5"'

for x0, title, subtitle, foot, foot_fill in [
        (6, "Without a KV cache", "every step recomputes K and V of all tokens", "K and V computed 15 times", INK),
        (425, "With a KV cache", "every step computes K and V of the new token only", "K and V computed 5 times, the rest is reused", CACHE)]:
    cached = x0 > 6
    o.append(rect(x0, PY, PW, PH, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(x0 + PW / 2, PY + 26, title, 16, extra='font-weight="600"'))
    o.append(text(x0 + PW / 2, PY + 46, subtitle, 12.5))
    o.append(text(x0 + PW / 2, PY + PH - 16, foot, 13, extra=f'fill="{foot_fill}" font-weight="600"'))

    gx = x0 + 74                 # left edge of the grid
    for k in range(N):
        o.append(text(gx + k * (CW + GX) + CW / 2, RY - 10, WORDS[k], 12.5, extra='font-weight="600"'))
    for s in range(N):
        y = RY + s * (CH + GY)
        o.append(text(gx - 12, y + CH / 2 + 4.5, f"step {s + 1}", 12.5, "end"))
        for k in range(s + 1):
            x = gx + k * (CW + GX)
            if cached and k < s:
                o.append(rect(x + 0.75, y + 0.75, CW - 1.5, CH - 1.5, CACHE, 4, OUTLINE))
                o.append(text(x + CW / 2, y + CH / 2 + 4, "KV", 11.5, extra=f'fill="{CACHE}" font-weight="600"'))
            else:
                o.append(rect(x, y, CW, CH, TOK, 4))
                o.append(text(x + CW / 2, y + CH / 2 + 4, "KV", 11.5, extra='fill="#fff" font-weight="600"'))
        # the step predicts the next token
        ex = gx + (s + 1) * (CW + GX) - GX
        o.append(line(ex + 6, y + CH / 2, ex + 26, y + CH / 2, INK, 'marker-end="url(#mi)"'))
        o.append(text(ex + 32, y + CH / 2 + 4.5, WORDS[s + 1], 13, "start", f'fill="{TOK}" font-weight="600"'))

    # legend
    ly = RY + N * (CH + GY) + 8
    items = [(TOK, "", "computed")] + ([(CACHE, OUTLINE, "read from the cache")] if cached else [])
    lx = x0 + PW / 2 - (80 if not cached else 130)
    for c, extra, lab in items:
        if extra:
            o.append(rect(lx + 0.75, ly + 0.75, 14.5, 14.5, c, 3, extra))
        else:
            o.append(rect(lx, ly, 16, 16, c, 3))
        o.append(text(lx + 22, ly + 12.5, lab, 12.5, "start"))
        lx += 110

save(__file__, o)
