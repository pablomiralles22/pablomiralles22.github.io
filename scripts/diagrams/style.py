"""Shared palette and SVG helpers for blog diagrams (see "Diagram style" in CLAUDE.md).

Each figure script lives in scripts/diagrams/<post-slug>/<figure-name>.py and writes
assets/img/posts/<post-slug>/<figure-name>.svg via save().
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CATEGORICAL = ["#4c8bd6", "#2fa58f", "#9b6dd1"]  # blue, teal, violet
ACCENT = "#e8833a"
INK = "#858585"
FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def svg_open(w, h, defs=()):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}" fill="{INK}">',
            '<defs>', *defs, '</defs>']


def arrow_marker(id, color, orient="auto"):
    return f'<marker id="{id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="{orient}"><path d="M0,0 L8,4 L0,8 z" fill="{color}"/></marker>'


def hatch_pattern(id="h"):
    return f'<pattern id="{id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="{INK}" stroke-opacity="0.4" stroke-width="1.5"/></pattern>'


def rect(x, y, w, h, fill, rx=3, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" {extra}/>'


def text(x, y, s, size=13, anchor="middle", extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" {extra}>{s}</text>'


def line(x1, y1, x2, y2, stroke, extra=""):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="1.6" {extra}/>'


def curve(x1, y1, x2, y2, stroke, extra=""):
    """Cubic Bezier with horizontal tangents at both ends."""
    mx = (x1 + x2) / 2
    return f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" stroke="{stroke}" stroke-width="1.8" {extra}/>'


def save(script_file, elements):
    """Write the SVG next to the post's other assets, named after the calling script."""
    script = Path(script_file).resolve()
    out = REPO / "assets/img/posts" / script.parent.name / f"{script.stem}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(elements + ["</svg>"]))
    print(f"wrote {out.relative_to(REPO)}")
