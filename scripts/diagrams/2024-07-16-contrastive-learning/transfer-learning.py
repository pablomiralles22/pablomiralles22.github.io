"""Transfer learning: pretrain encoder + pretext head, then reuse the encoder with a new head."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import ACCENT as TRAIN, CATEGORICAL, INK, arrow_marker, line, rect, save, svg_open, text

FROZEN, V, PRED = CATEGORICAL
FIRE, SNOW = "🔥", "❄️"

BOLD = 'font-weight="600"'
WHITE = 'fill="#fff" font-weight="600"'
NEUTRAL = f'fill-opacity="0.12" stroke="{INK}" stroke-width="1.5"'
PX, PW = 6, 828
CELL, NCELL = 11, 5

ix, iw = 40, 44                  # input
ex, ew, bh = 160, 120, 56        # encoder, block height
vx = 352                         # representation vector
hx, hw = 425, 160                # head
px, pw = 650, 150                # prediction
TILT = 7                         # horizontal offset of the encoder split at top and bottom

P1, H1 = 10, 156                 # pretraining panel top and height
GAP = 34
P2, H2 = P1 + H1 + GAP, 160      # fine-tuning panel
W, H = 840, P2 + H2 + 34

o = svg_open(W, H, [arrow_marker("mi", INK),
                    f'<clipPath id="enc"><rect x="{ex}" y="{P2 + 96 - bh / 2}" width="{ew}" height="{bh}" rx="5"/></clipPath>'])


def badge(x, y, emoji):
    """Emoji sitting on a block corner."""
    o.append(text(x, y + 5, emoji, 14))


panels = [
    (P1, H1, "Pretraining", "train on a pretext task with plenty of data, e.g. predicting masked words",
     "Pretext head", "e.g. masked word"),
    (P2, H2, "Fine-tuning", "train on the downstream task we care about",
     "Downstream head", "e.g. class label")]

for py, ph, title, subtitle, head, pred in panels:
    cy = py + 96
    top = cy - bh / 2
    o.append(rect(PX, py, PW, ph, "none", 10, f'stroke="{INK}" stroke-opacity="0.45" stroke-width="1.2"'))
    o.append(text(PX + PW / 2, py + 26, title, 16, extra=BOLD))
    o.append(text(PX + PW / 2, py + 46, subtitle, 12.5))

    o.append(rect(ix, cy - 14, iw, 28, INK, 4, NEUTRAL))
    o.append(text(ix + iw / 2, cy + 5, "x", 14, extra=BOLD))
    o.append(line(ix + iw, cy, ex - 2, cy, INK, 'marker-end="url(#mi)"'))

    if py == P1:
        o.append(rect(ex, top, ew, bh, TRAIN, 5))
        badge(ex + ew, top, FIRE)
    else:
        # frozen, trainable or anything in between: left part frozen, right part trained
        m, bot = ex + ew / 2, top + bh
        o.append(f'<g clip-path="url(#enc)">'
                 f'<path d="M{ex},{top} H{m + TILT} L{m - TILT},{bot} H{ex} z" fill="{FROZEN}"/>'
                 f'<path d="M{m + TILT},{top} H{ex + ew} V{bot} H{m - TILT} z" fill="{TRAIN}"/></g>')
        badge(ex, top, SNOW)
        badge(ex + ew, top, FIRE)
        o.append(text(ex + ew / 2, bot + 16, "frozen, trainable or partially frozen", 11.5))
    o.append(text(ex + ew / 2, cy + 5, "Encoder", 14, extra=WHITE))
    o.append(line(ex + ew, cy, vx - 4, cy, INK, 'marker-end="url(#mi)"'))

    for k in range(NCELL):
        o.append(rect(vx + 0.75, cy + (k - NCELL / 2) * CELL + 0.75, 16 - 1.5, CELL - 1.5, V, 2,
                      f'fill-opacity="0.15" stroke="{V}" stroke-width="1.5"'))
    o.append(text(vx + 8, cy + NCELL * CELL / 2 + 16, "v", 13, extra=f'fill="{V}" {BOLD}'))
    o.append(line(vx + 18, cy, hx - 2, cy, INK, 'marker-end="url(#mi)"'))

    o.append(rect(hx, top, hw, bh, TRAIN, 5))
    o.append(text(hx + hw / 2, cy + 5, head, 14, extra=WHITE))
    badge(hx + hw, top, FIRE)
    o.append(line(hx + hw, cy, px - 2, cy, INK, 'marker-end="url(#mi)"'))

    o.append(rect(px + 0.75, cy - 14 + 0.75, pw - 1.5, 28 - 1.5, PRED, 4,
                  f'fill-opacity="0.15" stroke="{PRED}" stroke-width="1.5"'))
    o.append(text(px + pw / 2, cy + 4.5, pred, 13, extra=f'fill="{PRED}" {BOLD}'))

# the pretrained encoder weights carry over to the second stage
ax = ex + ew / 2
o.append(line(ax, P1 + 96 + bh / 2, ax, P2 + 96 - bh / 2 - 2, INK, 'stroke-dasharray="4 3" marker-end="url(#mi)"'))
o.append(text(ax + 10, P1 + H1 + GAP / 2 + 4, "keep the weights", 11.5, "start", BOLD))

# legend
o.append(text(W / 2 - 16, H - 10, f"{FIRE} trainable", 12.5, "end", f'fill="{TRAIN}" {BOLD}'))
o.append(text(W / 2 + 16, H - 10, f"{SNOW} frozen", 12.5, "start", f'fill="{FROZEN}" {BOLD}'))
save(__file__, o)
