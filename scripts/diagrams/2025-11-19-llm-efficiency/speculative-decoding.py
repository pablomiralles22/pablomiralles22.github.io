"""Speculative decoding: a draft model guesses 4 tokens, the large model checks them in one pass."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as LARGE, CATEGORICAL, INK, arrow_marker, hatch_pattern, line, rect, save, svg_open, text

TOK, DRAFT = CATEGORICAL[0], CATEGORICAL[1]
W, H = 840, 380
o = svg_open(W, H, [arrow_marker("mi", INK), hatch_pattern()])

CW, CH = 44, 24
DW, DH = 52, 28
BOLD = 'font-weight="600"'
NEUTRAL = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
PROMPT, GUESS = ["The", "cat"], ["sat", "on", "a", "mat"]


def chip(x, yc, s, c, filled=True):
    if filled:
        return [rect(x, yc - CH / 2, CW, CH, c, 4), text(x + CW / 2, yc + 4.5, s, 13, extra=f'fill="#fff" {BOLD}')]
    return [rect(x + 0.75, yc - CH / 2 + 0.75, CW - 1.5, CH - 1.5, c, 4, f'fill-opacity="0.15" stroke="{c}" stroke-width="1.5"'),
            text(x + CW / 2, yc + 4.5, s, 13, extra=f'fill="{c}" {BOLD}')]


# ---- 1. draft: one small forward pass per token ----
A = 20                                           # arrow length
chain = 2 * CW + 8 + 4 * (2 * A + 8 + DW + CW)
x = (W - chain) / 2
y1 = 72
o.append(text(W / 2, 26, "1. The draft model guesses 4 tokens, one at a time", 14, extra=BOLD))
for k, s in enumerate(PROMPT):
    o += chip(x + k * (CW + 8), y1, s, TOK)
x += 2 * CW + 8
for s in GUESS:
    o.append(line(x + 3, y1, x + A, y1, INK, 'marker-end="url(#mi)"'))
    x += A + 4
    o.append(rect(x, y1 - DH / 2, DW, DH, INK, 4, NEUTRAL))
    o.append(text(x + DW / 2, y1 + 4.5, "draft", 12.5, extra=BOLD))
    x += DW
    o.append(line(x + 3, y1, x + A, y1, INK, 'marker-end="url(#mi)"'))
    x += A + 4
    o += chip(x, y1, s, DRAFT)
    x += CW
o.append(text(W / 2, y1 + 34, "4 cheap forward passes of the small model", 11.5))

# ---- 2. verify: one forward pass of the large model over all guesses ----
o.append(text(W / 2, 156, "2. The large model checks all 4 guesses in one forward pass", 14, extra=BOLD))
G = 16
xs = [(W - (6 * CW + 5 * G)) / 2 + k * (CW + G) for k in range(6)]
y2, my, mh, y3 = 188, 214, 40, 282
for k, s in enumerate(PROMPT + GUESS):
    o += chip(xs[k], y2, s, TOK if k < 2 else DRAFT)
    o.append(line(xs[k] + CW / 2, y2 + CH / 2 + 2, xs[k] + CW / 2, my - 3, INK, 'marker-end="url(#mi)"'))
mx0, mx1 = xs[0] - 10, xs[-1] + CW + 10
o.append(rect(mx0, my, mx1 - mx0, mh, INK, 5, NEUTRAL))
o.append(text(W / 2, my + mh / 2 + 5, "large model", 13, extra=BOLD))

# verdict under each guess
for k, (s, verdict) in enumerate(zip(GUESS, ["ok", "ok", "fix", "drop"]), start=2):
    x = xs[k]
    o.append(line(x + CW / 2, my + mh + 2, x + CW / 2, y3 - CH / 2 - 3, INK, 'marker-end="url(#mi)"'))
    if verdict == "ok":
        o += chip(x, y3, s, DRAFT, filled=False)
        o.append(text(x + CW / 2, y3 + 32, "✓", 15, extra=f'fill="{DRAFT}" {BOLD}'))
    elif verdict == "fix":
        o += chip(x, y3, "the", LARGE)
    else:
        o.append(rect(x + 0.75, y3 - CH / 2 + 0.75, CW - 1.5, CH - 1.5, "url(#h)", 4, f'stroke="{INK}" stroke-opacity="0.6" stroke-width="1.2"'))
o.append(text((xs[2] + xs[3] + CW) / 2, y3 + 50, "guesses kept", 11.5, extra=f'fill="{DRAFT}"'))
o.append(text(xs[4] + CW / 2, y3 + 32, "replaced", 11.5, extra=f'fill="{LARGE}"'))
o.append(text(xs[5] + CW / 2, y3 + 32, "discarded", 11.5))

o.append(text(W / 2, H - 12, "3 new tokens from a single forward pass of the large model", 13, extra=f'fill="{LARGE}" {BOLD}'))

save(__file__, o)
