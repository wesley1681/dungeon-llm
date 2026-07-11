"""Web combat view helpers — pure, front-end-agnostic.

The narrative game's combat is fully mechanical (engine/combat.py); the human's
turn is solicited via a `CombatPrompt` and answered with a COMMAND STRING that
`combat_policy._parse_command` parses. This module turns the combat state into
(a) a structured payload the web UI can render, (b) a battlefield image, and
(c) synthesises the command string from a UI selection (skill + target/cell) so
the whole thing rides the existing text-command input path (no new game seam).

Everything here is a pure function of (world_state, actor) — testable headless.
"""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont

from .engine.skill import available_skills, TargetType


def _load_font(size: int = 12):
    """A CJK-capable TrueType font (character names are Chinese) with a graceful
    fallback to PIL's bitmap default (renders CJK as tofu, but never crashes)."""
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msjh.ttc",
                 "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_FONT = _load_font(12)
_SMALL_FONT = _load_font(11)


# ── (a) structured payload ─────────────────────────────────────────────────────
def combat_view_state(actor, ctx, world_state) -> dict:
    """Structured snapshot for the web combat view, from the acting human's
    perspective. `ctx` is the engine CombatContext (has enemies/allies/resources)."""
    ws = world_state
    def _row(cid, team):
        c = ws.characters[cid]
        return {"id": cid, "name": c.name, "team": team,
                "x": round(c.position.x, 2), "y": round(c.position.y, 2),
                "hp": max(0, c.hp), "max_hp": getattr(c, "max_hp", c.hp),
                "ac": getattr(c, "ac", None), "alive": c.is_alive(),
                "dying": c.is_dying()}
    combatants = [_row(ctx.actor_id, "self")]
    combatants += [_row(cid, "ally") for cid in ctx.allies if cid in ws.characters]
    combatants += [_row(cid, "enemy") for cid in ctx.enemies if cid in ws.characters]

    skills = []
    for s in available_skills(actor, ws):
        tt = int(s.features.target_type)
        skills.append({"skill_id": s.skill_id, "display_name": s.display_name,
                       "target_type": tt, "target_kind": _target_kind(s.skill_id, tt)})
    return {
        "actor_id": ctx.actor_id, "actor_name": actor.name,
        "round": ctx.round_num, "resources": dict(ctx.resources),
        "combatants": combatants,
        "enemies": dict(ctx.enemies), "allies": dict(ctx.allies),
        "skills": skills,
    }


def _target_kind(skill_id: str, target_type: int) -> str:
    """How the UI must gather a target for this skill: 'none' / 'enemy' / 'ally'
    / 'point' (grid cell) / 'multi_enemy' / 'multi_ally'."""
    if skill_id in ("dodge", "hide", "disengage") or skill_id.startswith("end"):
        return "none"
    if skill_id == "move":
        return "point"          # click a destination cell (also accepts a target id)
    if target_type == TargetType.SELF:
        return "none"
    if target_type == TargetType.SINGLE_ENEMY:
        return "enemy"
    if target_type == TargetType.SINGLE_ALLY:
        return "ally"
    if target_type in (TargetType.POINT, TargetType.LINE, TargetType.CONE):
        return "point"
    if target_type == TargetType.MULTI_ENEMY:
        return "multi_enemy"
    if target_type == TargetType.MULTI_ALLY:
        return "multi_ally"
    return "none"


# ── (b) battlefield image (PIL — gr.Image accepts it directly) ─────────────────
_TERRAIN_FILL = {1: (120, 100, 60), 2: (140, 60, 60), 3: (40, 40, 48)}  # DIFFICULT/DANGEROUS/BLOCKED
_TEAM_FILL = {"self": (70, 130, 235), "ally": (70, 170, 120), "enemy": (210, 70, 70)}


def render_battlefield(world_state, actor_id: str, px: int = 600):
    """Draw the battlefield (terrain + combatants + HP bars) to a PIL image,
    AUTO-FIT to the combatants so a fight clustered in one corner of the 30×30m
    field fills the view instead of being a tiny blob. Returns
    (image, (origin_x_m, origin_y_m, scale)) — the view transform, so a click
    pixel maps back to metres via  m = pixel / scale + origin."""
    bf = world_state.combat.battlefield
    order = world_state.combat.initiative_order

    # view window (metres) = bounding box of live combatants + padding, min 14 m,
    # square, clamped inside the field.
    live = [world_state.characters[c] for c in order
            if world_state.characters.get(c) and world_state.characters[c].is_alive()]
    xs = [c.position.x for c in live] or [bf.width / 2]
    ys = [c.position.y for c in live] or [bf.height / 2]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    win = min(max(14.0, span + 6.0), max(bf.width, bf.height))
    ox = max(0.0, min(cx - win / 2, bf.width - win))
    oy = max(0.0, min(cy - win / 2, bf.height - win))
    scale = px / win

    def _px(mx, my):
        return (mx - ox) * scale, (my - oy) * scale

    img = Image.new("RGB", (px, px), (24, 26, 32))
    d = ImageDraw.Draw(img)

    # terrain (only non-NORMAL cells inside the window)
    res = bf.grid_resolution
    ix0, iy0 = int(ox / res), int(oy / res)
    ix1, iy1 = int((ox + win) / res) + 1, int((oy + win) / res) + 1
    for iy in range(max(0, iy0), min(len(bf.cells), iy1)):
        for ix in range(max(0, ix0), min(len(bf.cells[0]), ix1)):
            fill = _TERRAIN_FILL.get(int(bf.cells[iy][ix]))
            if fill:
                x0, y0 = _px(ix * res, iy * res)
                d.rectangle([x0, y0, x0 + res * scale, y0 + res * scale], fill=fill)
    # 1 m grid lines (zoomed in, so a manageable count) to help cell-clicking
    m0 = int(ox)
    for m in range(m0, int(ox + win) + 1):
        x, _ = _px(m, 0)
        d.line([(x, 0), (x, px)], fill=(40, 44, 52))
    for m in range(int(oy), int(oy + win) + 1):
        _, y = _px(0, m)
        d.line([(0, y), (px, y)], fill=(40, 44, 52))

    # combatants — only this fight's participants (not the whole world)
    r = max(7, min(13, int(0.45 * scale)))
    for c in live:
        is_self = (c is world_state.characters.get(actor_id))
        team = "self" if is_self else (
            "enemy" if (c.is_npc and getattr(c, "attitude", 1) == 0) else "ally")
        cx_p, cy_p = _px(c.position.x, c.position.y)
        d.ellipse([cx_p - r, cy_p - r, cx_p + r, cy_p + r], fill=_TEAM_FILL[team],
                  outline=(255, 215, 60) if is_self else (16, 16, 20),
                  width=3 if is_self else 1)
        mhp = getattr(c, "max_hp", c.hp) or 1
        frac = max(0.0, min(1.0, c.hp / mhp))
        d.rectangle([cx_p - r, cy_p - r - 6, cx_p + r, cy_p - r - 2], fill=(50, 50, 55))
        d.rectangle([cx_p - r, cy_p - r - 6, cx_p - r + 2 * r * frac, cy_p - r - 2],
                    fill=(90, 200, 90) if frac > 0.3 else (210, 120, 60))
        d.text((cx_p - r, cy_p + r + 1), f"{c.name} {c.hp}", fill=(225, 225, 230),
               font=_SMALL_FONT)
    return img, (ox, oy, scale)


# ── (c) UI selection → command string (rides _parse_command) ───────────────────
def command_for(skill: dict, target_id: str | None = None,
                cell: tuple[float, float] | None = None) -> str:
    """Synthesise the command string `_parse_command` expects from a UI choice.
    `skill` is one entry of combat_view_state()['skills']."""
    sid = skill["skill_id"]
    kind = skill["target_kind"]
    if sid.startswith("end"):
        return "end"
    if sid in ("dodge", "hide", "disengage"):
        return sid
    if sid == "move":
        if cell is not None:
            return f"move {cell[0]:.1f} {cell[1]:.1f}"
        return f"move {target_id}"
    if sid.startswith("weapon:"):
        return f"攻擊 {target_id} {sid[len('weapon:'):]}"
    if sid.startswith("spell:"):
        name = sid[len('spell:'):]
        if kind == "point" and cell is not None:
            return f"spell {name} {cell[0]:.1f} {cell[1]:.1f}"
        return f"spell {name} {target_id}"
    # generic ability → 招式 <skill_id> [args by target_type]
    if kind == "none":
        return f"招式 {sid}"
    if kind == "point" and cell is not None:
        return f"招式 {sid} {cell[0]:.1f} {cell[1]:.1f}"
    if kind in ("multi_enemy", "multi_ally"):
        return f"招式 {sid} {target_id}"   # target_id already comma-joined by caller
    return f"招式 {sid} {target_id}"
