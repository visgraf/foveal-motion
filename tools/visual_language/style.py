"""Visual Language 1: the single source of truth for semantic roles.

Contract: docs/methodology/visual-language-1-contract.md.  Every semantic role has a color
and a non-color cue (shape, outline, hatch, dash or glyph); the legend, the frames and the
checker all read this module.  The palette is Okabe-Ito based (validated with the dataviz
palette validator; see docs/methodology/visual-language-1.md), on a light surface.

Drawing primitives here define how each cue looks; tools/visual_language/vl1_draw.py composes
them into panels.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- colors (sRGB)

INK, INK2, MUTED, FAINT = (20, 20, 20), (70, 70, 70), (130, 130, 130), (200, 199, 195)
SURFACE, PANEL, WHITE, GRID = (250, 250, 248), (255, 255, 255), (255, 255, 255), (226, 225, 221)
OI_BLUE, OI_SKY, OI_GREEN = (0, 114, 178), (86, 180, 233), (0, 158, 115)
OI_ORANGE, OI_VERM, OI_PURPLE, OI_YELLOW = (230, 159, 0), (213, 94, 0), (204, 121, 167), (240, 228, 66)
SLATE = (157, 180, 207)          # QUIET: a deliberately calm neutral (reads gray), always with a check glyph
GEOM = (160, 168, 181)           # persistent geometry, depth-shaded between GEOM_NEAR and GEOM_FAR
GEOM_NEAR, GEOM_FAR = (196, 203, 214), (104, 114, 132)
OTHER_SURF, UNKNOWN_BG, STIPPLE = (205, 201, 193), (247, 246, 242), (178, 175, 166)
CORE_GREEN = (0, 120, 88)
WINE, REF_BROWN = (136, 34, 85), (133, 94, 20)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR, FONT_BOLD = FONT_DIR / "DejaVuSans.ttf", FONT_DIR / "DejaVuSans-Bold.ttf"
_FONTS: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (bold, int(size))
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), int(size))
    return _FONTS[key]


def font_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (FONT_REGULAR, FONT_BOLD)}


# Typography at 1440p (the 1080p derivative scales by 0.75; body >= 22 px keeps >= 16.5 px).
T_TITLE, T_HEAD, T_BODY, T_SMALL, T_BIG, T_HUGE = 30, 26, 22, 19, 40, 64

# ---------------------------------------------------------------- primitives

def dashed_line(d: ImageDraw.ImageDraw, pts, fill, width=3, dash=10, gap=7):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            d.line([x0 + ux * s, y0 + uy * s, x0 + ux * e, y0 + uy * e], fill=fill, width=width)
            s = e + gap


def dashed_rect(d, box, fill, width=3, dash=10, gap=7):
    x0, y0, x1, y1 = box
    dashed_line(d, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], fill, width, dash, gap)


def dashed_ellipse(d, box, fill, width=3, n=16):
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 0.55) / n
        d.arc([cx - rx, cy - ry, cx + rx, cy + ry], math.degrees(a0), math.degrees(a1), fill=fill, width=width)


def hatch(img: Image.Image, box, color, spacing=9, width=2, cross=False, mask: Image.Image | None = None):
    """Diagonal hatch (``cross`` adds the other diagonal) inside ``box`` (optionally masked)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for k in range(-h, w + h, spacing):
        ld.line([k, 0, k + h, h], fill=color + (255,), width=width)
        if cross:
            ld.line([k + h, 0, k, h], fill=color + (255,), width=width)
    if mask is not None:  # a mask in the image's coordinates, cropped to the box
        m = np.asarray(mask.crop((x0, y0, x1, y1)), np.uint8)
        layer.putalpha(Image.fromarray(np.minimum(np.asarray(layer.getchannel("A")), m)))
    img.paste(layer, (x0, y0), layer)


def stipple(img: Image.Image, box, color=STIPPLE, spacing=8, r=1):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    for yy in range(y0 + spacing // 2, y1, spacing):
        off = (spacing // 2) if ((yy - y0) // spacing) % 2 else 0
        for xx in range(x0 + off + spacing // 2, x1, spacing):
            d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color)


def crosshair(d, x, y, r=16, solid=True, color=INK, halo=WHITE, width=3):
    """Current fixation: solid crosshair + ring.  Proposed fixation: dashed crosshair + dashed ring."""
    for col, w in ((halo, width + 4), (color, width)):
        if solid:
            d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=w)
            for a, b in (((x - 2 * r, y), (x - r * 0.45, y)), ((x + r * 0.45, y), (x + 2 * r, y)),
                         ((x, y - 2 * r), (x, y - r * 0.45)), ((x, y + r * 0.45), (x, y + 2 * r))):
                d.line([a, b], fill=col, width=w)
        else:
            dashed_ellipse(d, [x - r, y - r, x + r, y + r], col, width=w, n=10)
            for a, b in (((x - 2 * r, y), (x - r * 0.45, y)), ((x + r * 0.45, y), (x + 2 * r, y)),
                         ((x, y - 2 * r), (x, y - r * 0.45)), ((x, y + r * 0.45), (x, y + 2 * r))):
                dashed_line(d, [a, b], col, width=w, dash=6, gap=5)


def ring(d, x, y, r=6, color=OI_VERM, width=2, dashed=False):
    if dashed:
        dashed_ellipse(d, [x - r, y - r, x + r, y + r], color, width=width, n=8)
    else:
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)


def xmark(d, x, y, r=5, color=OI_SKY, width=2):
    d.line([x - r, y - r, x + r, y + r], fill=color, width=width)
    d.line([x - r, y + r, x + r, y - r], fill=color, width=width)


def square(d, x, y, r=4, color=OI_GREEN):
    d.rectangle([x - r, y - r, x + r, y + r], fill=color)


def arrow(d, x0, y0, x1, y1, color=INK, width=4, head=18, double=False, halo=WHITE):
    """Attention move: single arrow (retain).  Attention switch: double-line arrow."""
    ang = math.atan2(y1 - y0, x1 - x0)
    nx, ny = -math.sin(ang), math.cos(ang)
    for col, extra in ((halo, 4), (color, 0)):
        if double:
            for s in (-4, 4):
                d.line([x0 + nx * s, y0 + ny * s, x1 - head * 0.6 * math.cos(ang) + nx * s,
                        y1 - head * 0.6 * math.sin(ang) + ny * s], fill=col, width=max(2, width - 1) + extra)
        else:
            d.line([x0, y0, x1 - head * 0.5 * math.cos(ang), y1 - head * 0.5 * math.sin(ang)], fill=col,
                   width=width + extra)
        p = [(x1, y1), (x1 - head * math.cos(ang - 0.42), y1 - head * math.sin(ang - 0.42)),
             (x1 - head * math.cos(ang + 0.42), y1 - head * math.sin(ang + 0.42))]
        d.polygon(p, fill=col)


def text(d, xy, s, size=T_BODY, bold=False, fill=INK, outline=WHITE, plate=None, anchor="la", stroke=3):
    """Text with an outline (busy backgrounds) or on a plate; returns the text bbox."""
    f = font(size, bold)
    if plate is not None:
        bb = d.textbbox(xy, s, font=f, anchor=anchor)
        d.rectangle([bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4], fill=plate)
        d.text(xy, s, font=f, fill=fill, anchor=anchor)
        return bb
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke if outline else 0,
           stroke_fill=outline if outline else None)
    return d.textbbox(xy, s, font=f, anchor=anchor)


# ---------------------------------------------------------------- truth badges

BADGES = {
    "CONTROLLER-TIME": {"color": OI_BLUE, "cue": "solid filled badge", "label": "CONTROLLER-TIME"},
    "DERIVED": {"color": INK2, "cue": "dashed-outline badge, no fill", "label": "DERIVED"},
    "REFERENCE / EVALUATION": {"color": REF_BROWN, "cue": "diagonally hatched badge", "label": "REFERENCE / EVALUATION"},
    "ORACLE INPUT": {"color": WINE, "cue": "double-outline badge", "label": "ORACLE INPUT"},
}


def badge(img: Image.Image, x_right: float, y: float, name: str, size=T_SMALL) -> float:
    """Draw a truth badge whose right edge is at ``x_right``; return its left edge."""
    d = ImageDraw.Draw(img)
    b = BADGES[name]
    f = font(size, True)
    w = d.textlength(b["label"], font=f) + 24
    h = size + 14
    box = [x_right - w, y, x_right, y + h]
    col = b["color"]
    if name == "CONTROLLER-TIME":
        d.rounded_rectangle(box, radius=6, fill=col)
        d.text((box[0] + 12, y + 6), b["label"], font=f, fill=WHITE)
    elif name == "DERIVED":
        d.rounded_rectangle(box, radius=6, fill=WHITE)
        dashed_rect(d, box, col, width=2, dash=7, gap=5)
        d.text((box[0] + 12, y + 6), b["label"], font=f, fill=col)
    elif name == "REFERENCE / EVALUATION":
        d.rounded_rectangle(box, radius=6, fill=(246, 238, 222))
        hatch(img, box, (224, 205, 170), spacing=8, width=2)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(box, radius=6, outline=col, width=2)
        tb = d.textbbox((box[0] + 12, y + 6), b["label"], font=f)
        d.rectangle([tb[0] - 3, tb[1] - 1, tb[2] + 3, tb[3] + 1], fill=(246, 238, 222))
        d.text((box[0] + 12, y + 6), b["label"], font=f, fill=col)
    else:  # ORACLE INPUT
        d.rounded_rectangle(box, radius=6, fill=WHITE, outline=col, width=2)
        d.rounded_rectangle([box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4], radius=4, outline=col, width=2)
        d.text((box[0] + 12, y + 6), b["label"], font=f, fill=col)
    return box[0]


# ---------------------------------------------------------------- service-state tiles

def state_tile(img: Image.Image, box, state: str, label: str = "", size=T_BODY):
    """A roster tile: fill / outline / hatch and a glyph per service state."""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    if state == "QUIET":
        d.rectangle(box, fill=SLATE, outline=INK2, width=1)
        glyph, gcol = "✓", INK
    elif state == "ACTIONABLE":
        d.rectangle(box, fill=WHITE)
        d.rectangle(box, outline=OI_BLUE, width=5)
        glyph, gcol = "▶", OI_BLUE
    elif state == "SEEDABLE":
        d.rectangle(box, fill=WHITE)
        dashed_rect(d, box, MUTED, width=2, dash=6, gap=5)
        glyph, gcol = "○", MUTED
    elif state == "DEFERRED":
        d.rectangle(box, fill=(253, 243, 222))
        hatch(img, box, OI_ORANGE, spacing=10, width=3)
        d = ImageDraw.Draw(img)
        d.rectangle(box, outline=OI_ORANGE, width=4)
        glyph, gcol = "II", INK
    elif state == "RESIDUAL":
        d.rectangle(box, fill=(250, 232, 220))
        hatch(img, box, OI_VERM, spacing=11, width=2, cross=True)
        d = ImageDraw.Draw(img)
        d.rectangle(box, outline=OI_VERM, width=3)
        d.rectangle([x0 + 6, y0 + 6, x1 - 6, y1 - 6], outline=OI_VERM, width=3)
        glyph, gcol = "R", INK
    elif state == "UNLOCATED":
        d.rectangle(box, fill=(236, 235, 231))
        for xx in range(int(x0), int(x1), 6):
            d.point([(xx, y0), (xx, y1)], fill=INK2)
        for yy in range(int(y0), int(y1), 6):
            d.point([(x0, yy), (x1, yy)], fill=INK2)
        glyph, gcol = "?", INK2
    else:
        raise ValueError(state)
    if label:
        f = font(size, True)
        tw = d.textlength(label, font=f)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        tb = d.textbbox((cx - tw / 2, cy - size / 2 - 2), label, font=f)
        d.rectangle([tb[0] - 3, tb[1] - 1, tb[2] + 3, tb[3] + 2], fill=WHITE)
        d.text((cx - tw / 2, cy - size / 2 - 2), label, font=f, fill=INK)
    gf = font(max(14, int(size * 0.8)), True)
    gw = d.textlength(glyph, font=gf)
    gb = [x1 - gw - 8, y0 + 3, x1 - 3, y0 + 5 + gf.size]
    d.rectangle(gb, fill=WHITE)
    d.text((gb[0] + 2, y0 + 3), glyph, font=gf, fill=gcol)


# ---------------------------------------------------------------- the roles

def _sw_badge(name):
    def f(img, box):
        x0, y0, x1, y1 = box
        badge(img, x1 - 2, y0 + (y1 - y0 - T_SMALL - 14) / 2, name, size=T_SMALL - 3)
    return f


def _sw_outline(img, box):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(232, 240, 250))
    d.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=WHITE, width=8)
    d.rectangle([x0 + 5, y0 + 5, x1 - 5, y1 - 5], outline=INK, width=4)


def _sw_cross(solid):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        crosshair(d, (x0 + x1) / 2, (y0 + y1) / 2, r=13, solid=solid, width=3)
    return f


def _sw_arrow(double):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        arrow(d, x0 + 6, (y0 + y1) / 2, x1 - 6, (y0 + y1) / 2, width=3, head=14, double=double)
    return f


def _sw_fill(color, outline=None):
    def f(img, box):
        d = ImageDraw.Draw(img)
        d.rectangle(box, fill=color)
        if outline:
            d.rectangle(box, outline=outline, width=3)
    return f


def _sw_unknown(img, box):
    d = ImageDraw.Draw(img)
    d.rectangle(box, fill=UNKNOWN_BG, outline=FAINT, width=1)
    stipple(img, box, spacing=7)


def _sw_lines(color):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=OTHER_SURF)
        d.line([x0, y1 - 4, (x0 + x1) / 2, y0 + 6, x1, y0 + 12], fill=color, width=4)
    return f


def _sw_rings(color, dashed=False):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=WHITE, outline=FAINT, width=1)
        for k in range(3):
            ring(d, x0 + 12 + k * ((x1 - x0 - 24) / 2), (y0 + y1) / 2, r=7, color=color, width=3, dashed=dashed)
    return f


def _sw_points(color, kind):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=WHITE, outline=FAINT, width=1)
        for k in range(3):
            cx = x0 + 12 + k * ((x1 - x0 - 24) / 2)
            cy = (y0 + y1) / 2 + (k - 1) * 4
            if kind == "square":
                square(d, cx, cy, r=6, color=color)
            elif kind == "x":
                xmark(d, cx, cy, r=6, color=color, width=3)
            elif kind == "dots":
                for dx in (-6, 0, 6):
                    d.ellipse([cx + dx - 2, cy - 2, cx + dx + 2, cy + 2], fill=color)
    return f


def _sw_footprint(img, box):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=WHITE, outline=FAINT, width=1)
    dashed_rect(d, [x0 + 6, y0 + 6, x1 - 6, y1 - 6], INK, width=3, dash=5, gap=4)


def _sw_state(state):
    def f(img, box):
        state_tile(img, box, state)
    return f


def _sw_core(dashed):
    def f(img, box):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = box
        d.rectangle(box, fill=WHITE, outline=FAINT, width=1)
        b = [x0 + 8, y0 + 5, x1 - 8, y1 - 5]
        (dashed_rect(d, b, CORE_GREEN, width=4, dash=7, gap=5) if dashed else d.rectangle(b, outline=CORE_GREEN, width=4))
    return f


def _sw_nodepth(img, box):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(214, 206, 190), outline=FAINT, width=1)
    hatch(img, box, INK, spacing=8, width=2)


def _sw_binocular(img, box):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=WHITE, outline=FAINT, width=1)
    d.rectangle([x0 + 18, y0 + 6, x1 - 8, y1 - 6], fill=(214, 236, 226))
    dashed_rect(d, [x0 + 6, y0 + 4, x1 - 20, y1 - 8], CORE_GREEN, width=2, dash=5, gap=4)
    dashed_rect(d, [x0 + 18, y0 + 8, x1 - 6, y1 - 4], CORE_GREEN, width=2, dash=5, gap=4)


# group, key, label, color, non-color cue, swatch renderer
ROLES = [
    ("TRUTH / PROVENANCE", "controller_time", "CONTROLLER-TIME: logged or replayed controller state", OI_BLUE,
     "solid filled badge", _sw_badge("CONTROLLER-TIME")),
    ("TRUTH / PROVENANCE", "derived", "DERIVED: recomputed by accepted functions, checked against records", INK2,
     "dashed-outline badge", _sw_badge("DERIVED")),
    ("TRUTH / PROVENANCE", "oracle_input", "ORACLE INPUT: Blender catalog / seeds; current-pair Position + Object Index", WINE,
     "double-outline badge", _sw_badge("ORACLE INPUT")),
    ("TRUTH / PROVENANCE", "reference", "REFERENCE / EVALUATION: never used by the controller", REF_BROWN,
     "hatched badge", _sw_badge("REFERENCE / EVALUATION")),
    ("ATTENTION", "current_object", "current object (target)", INK, "thick dark outline with white halo", _sw_outline),
    ("ATTENTION", "current_fixation", "current fixation", INK, "solid crosshair + solid ring", _sw_cross(True)),
    ("ATTENTION", "proposed_fixation", "proposed fixation", INK, "dashed crosshair + dashed ring", _sw_cross(False)),
    ("ATTENTION", "attention_move", "attention move (same object)", INK, "single-line arrow", _sw_arrow(False)),
    ("ATTENTION", "attention_switch", "attention switch (new object)", INK, "double-line arrow + SWITCH tag", _sw_arrow(True)),
    ("EPISTEMIC", "target_support", "target support", OI_SKY, "solid fill inside a dark target outline",
     _sw_fill(OI_SKY, (10, 60, 110))),
    ("EPISTEMIC", "other_surface", "other surface (other objects)", OTHER_SURF, "flat fill, no outline", _sw_fill(OTHER_SURF)),
    ("EPISTEMIC", "unknown", "unknown (never measured)", UNKNOWN_BG, "dot stipple", _sw_unknown),
    ("EPISTEMIC", "ambiguous_boundary", "ambiguous boundary", OI_ORANGE, "thin line strokes", _sw_lines(OI_ORANGE)),
    ("EPISTEMIC", "unresolved_support", "unresolved support (FSG6f OPEN / Cyclopean eligible)", OI_VERM, "hollow rings",
     _sw_rings(OI_VERM)),
    ("MEASUREMENT / GEOMETRY", "persistent_geometry", "persistent geometry (active fused maps)", GEOM, "fine dot texture",
     _sw_points(GEOM, "dots")),
    ("MEASUREMENT / GEOMETRY", "new_measurement", "this look's target measurements (footprint)", INK, "dashed footprint outline",
     _sw_footprint),
    ("MEASUREMENT / GEOMETRY", "new_surfel", "new spatial support: new surfel", OI_GREEN, "filled squares",
     _sw_points(OI_GREEN, "square")),
    ("MEASUREMENT / GEOMETRY", "repeated_measurement", "repeated measurement (no new surfel)", OI_PURPLE, "hollow rings",
     _sw_rings(OI_PURPLE)),
    ("MEASUREMENT / GEOMETRY", "cross_target", "cross-target measurement (another object)", OI_SKY, "x marks",
     _sw_points(OI_SKY, "x")),
    ("SERVICE STATE", "quiet", "QUIET: local policy has nothing to do", SLATE, "filled tile + check glyph", _sw_state("QUIET")),
    ("SERVICE STATE", "actionable", "ACTIONABLE: local policy proposes a look", OI_BLUE, "heavy outline + play glyph",
     _sw_state("ACTIONABLE")),
    ("SERVICE STATE", "seedable", "SEEDABLE: localized, not yet looked at", MUTED, "dashed outline + circle glyph",
     _sw_state("SEEDABLE")),
    ("SERVICE STATE", "deferred", "DEFERRED: ordinary budget spent, still ACTIONABLE", OI_ORANGE,
     "single diagonal hatch + pause glyph", _sw_state("DEFERRED")),
    ("SERVICE STATE", "finalized_residual", "FINALIZED RESIDUAL: closed with explicit residue", OI_VERM,
     "cross-hatch + double outline + R glyph", _sw_state("RESIDUAL")),
    ("SERVICE STATE", "unlocated", "UNLOCATED: in the catalog, never localized", MUTED, "dotted outline + ? glyph",
     _sw_state("UNLOCATED")),
    ("SENSOR", "actual_core", "actual depth-measuring core", CORE_GREEN, "solid rectangle outline", _sw_core(False)),
    ("SENSOR", "predicted_core", "predicted depth-measuring core (no render)", CORE_GREEN, "dashed rectangle outline",
     _sw_core(True)),
    ("SENSOR", "binocular_support", "binocular support (inside both eyes' cores)", CORE_GREEN, "pale wash inside two dashed outlines",
     _sw_binocular),
    ("SENSOR", "visible_no_depth", "visible target pixel without valid stereo depth", INK, "black diagonal hatch",
     _sw_nodepth),
    ("SENSOR", "previously_interrogated", "previously interrogated binocularly", OI_PURPLE, "dashed ring", _sw_rings(OI_PURPLE, True)),
]
ROLE = {r[1]: {"group": r[0], "key": r[1], "label": r[2], "color": r[3], "cue": r[4], "swatch": r[5]} for r in ROLES}
GROUPS = list(dict.fromkeys(r[0] for r in ROLES))
REQUIRED_ROLES = {
    "TRUTH / PROVENANCE": {"controller_time", "derived", "reference", "oracle_input"},
    "ATTENTION": {"current_object", "current_fixation", "proposed_fixation", "attention_switch"},
    "EPISTEMIC": {"target_support", "other_surface", "unknown", "ambiguous_boundary", "unresolved_support"},
    "MEASUREMENT / GEOMETRY": {"persistent_geometry", "new_measurement", "new_surfel", "repeated_measurement", "cross_target"},
    "SERVICE STATE": {"quiet", "actionable", "deferred", "finalized_residual", "unlocated"},
    "SENSOR": {"actual_core", "predicted_core", "binocular_support"},
}
# Action sources on the timeline: color + shape.
SOURCES = {"oracle_seed": ("seed look", INK2, "square"), "fsg6f": ("FSG6f look", OI_BLUE, "circle"),
           "cyclopean_epistemic": ("Cyclopean look", OI_GREEN, "diamond")}
DISTINCTIONS = ["measurement != new geometry", "visible != depth measured", "actionable != worth a final observation",
                "deferred != quiet", "residual != failure", "SCENE_CLOSED != global quiescence"]


def swatch(key: str, w=64, h=34) -> Image.Image:
    img = Image.new("RGB", (w, h), PANEL)
    ROLE[key]["swatch"](img, (1, 1, w - 2, h - 2))
    return img


def declaration() -> list[dict]:
    """The machine-readable language (the legend and the checker compare against it)."""
    return [{"group": r[0], "key": r[1], "label": r[2], "color": list(r[3]), "cue": r[4]} for r in ROLES]
