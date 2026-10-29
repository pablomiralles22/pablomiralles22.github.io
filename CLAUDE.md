# Coding rules

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

# Project conventions

## Figures in posts
- Theme is `jekyll-bear-theme` (not al-folio). Dark mode follows `prefers-color-scheme`; backgrounds are `#F5F7F3` (light) and `#0D1B12` (dark). Site styles live in `assets/css/style.css`.
- Insert figures with `{% include figure.liquid path="assets/img/posts/<post-slug>/<file>" alt="..." caption="..." %}`. Add a `caption` (styled via `figcaption` in `style.css`) and an `alt` to every new figure; avoid double quotes inside both. The `alt` is one plain sentence saying what the image shows (for search engines and screen readers); the caption carries the explanation.
- The `img-fluid rounded z-depth-1` classes are al-folio leftovers with no effect here; don't add them to new figures.
- Diagrams are hand-written SVGs with a transparent background and mid-tone colours that read on both the light and dark backgrounds, so no theme switching is needed.

## Links and SEO
- `<head>` metadata (title, description, canonical, Open Graph/Twitter cards, JSON-LD) comes from `jekyll-seo-tag` via `{% seo %}` in `_includes/head.html`. Don't hand-write those tags.
- Every post needs a `description` in its front matter (one sentence, ideally under ~160 characters); it becomes the search snippet and link-preview text.
- Keep `title` under ~60 characters so search results don't truncate it. If the post deserves a longer on-page heading, put it in `heading` (rendered as the `<h1>` by `_layouts/post.html`).
- Link previews default to the profile photo (`defaults` in `_config.yml`). For a post with a good raster figure, set `image: /assets/img/posts/<post-slug>/<file>.png` (leading slash, PNG/JPG, not SVG, which most preview cards can't show) plus `twitter: {card: summary_large_image}`.
- Internal links in templates use `relative_url` (e.g. `{{ "/blog" | relative_url }}`), never `site.url`, so they work when serving locally. Use `absolute_url` only where a full URL is required (e.g. feeds).
- Editable diagram sources (`*.drawio`) stay next to their exports but are excluded from the build in `_config.yml`.

## Diagram style
Keep new diagrams consistent with `assets/img/posts/2025-11-19-llm-efficiency/*.svg` (the reference examples).
- **Canvas:** `viewBox` up to 840 wide (fits the 720px content column), height as needed. Transparent background, no `<style>` blocks and no `prefers-color-scheme` logic.
- **Font:** `font-family="-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"` on the root `<svg>`. Sizes: panel title 16 bold, subtitle 12.5, labels 13, small annotations 11.5.
- **Colours** (never pure black/white text, except white labels on filled blocks):
  - Ink (all text, outlines, neutral arrows): `#858585`.
  - Categorical, in this order: `#4c8bd6` (blue), `#2fa58f` (teal), `#9b6dd1` (violet). Use them for the main entity of the figure (GPUs, tokens, ...) and label them in place or with a legend.
  - Accent: `#e8833a` (orange), reserved for the one thing the figure highlights (communication, router, ...). Footer takeaways use it when they describe that thing.
- **Panels:** side-by-side comparisons go in rounded panels (`rx=10`, ink stroke 1.2 at opacity 0.45, no fill), each with a bold title, a one-line subtitle, and a bold one-line takeaway at the bottom.
- **Blocks:** rounded rects (`rx` 2–5). Filled = active/primary (white bold label inside); outlined = secondary (colour stroke 1.5, `fill-opacity=0.15`, coloured label); neutral components = ink fill at `fill-opacity=0.12` with ink stroke; idle/empty = 45° hatch pattern in ink at opacity 0.4.
- **Lines:** stroke width 1.6–1.8. Small triangular arrowhead markers in the line's colour. Communication/accent links are dashed (`4 3`). Use cubic Béziers with horizontal tangents for routing lines.
- **Layout:** data flows left → right. Keep labels off panel borders and leave even spacing; inputs enter from the left with a short arrow.
- **Workflow:** each figure has a generator at `scripts/diagrams/<post-slug>/<figure-name>.py` that imports the palette and helpers from `scripts/diagrams/style.py` and writes `assets/img/posts/<post-slug>/<figure-name>.svg` (run it with `python3 scripts/diagrams/<post-slug>/<figure-name>.py`). Edit figures through their script, never the SVG by hand. Render previews with cairosvg on both `#F5F7F3` and `#0D1B12` (outside the repo) and check them visually. `scripts/` is in `exclude` in `_config.yml`, so it is never deployed.
