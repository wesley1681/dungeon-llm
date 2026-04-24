import re
from ..engine import combat
from ..engine.world_state import WorldState

_TAG = re.compile(r"\[([A-Z]+):\s*([^\]]+)\]")


def parse_and_resolve(text: str, world_state: WorldState,
                      skip_tags: set[str] | None = None) -> tuple[str, list[str]]:
    results: list[str] = []

    def _replace(match: re.Match) -> str:
        tag = match.group(1)
        if skip_tags and tag in skip_tags:
            return match.group(0)   # leave tag text as-is, don't execute
        try:
            result = _dispatch(tag, match.group(2).strip(), world_state)
        except Exception as e:
            result = f"標籤解析錯誤：{match.group(0)}（{e}）"
        results.append(result)
        return f"（{result}）"

    cleaned = _TAG.sub(_replace, text)
    return cleaned, results


def parse_travel_only(text: str, world_state: WorldState) -> list[str]:
    """Parse only TRAVEL tags from text (safe to call on the pre-narrative section).
    Returns list of result strings."""
    results = []
    for match in re.finditer(r"\[TRAVEL:\s*([^\]]+)\]", text):
        try:
            result = _dispatch("TRAVEL", match.group(1).strip(), world_state)
        except Exception as e:
            result = f"TRAVEL 錯誤：{e}"
        results.append(result)
    return results


def _dispatch(tag: str, args: str, ws: WorldState) -> str:

    if tag == "INITIATIVE":
        # No-op if TRAVEL already started combat in this same turn
        if ws.combat and ws.combat.active:
            return f"先攻順序：{'、'.join(ws.characters[n].name for n in ws.combat.initiative_order if n in ws.characters)}"
        # Scope to current room characters if dungeon_map is active
        char_ids = None
        if ws.dungeon_map:
            room = ws.dungeon_map.current_room
            pc_ids = [cid for cid, c in ws.characters.items() if not c.is_npc]
            char_ids = list(room.enemy_ids) + pc_ids
        state = combat.roll_initiative(ws, char_ids)
        ws.combat = state
        order = "、".join(
            ws.characters[n].name for n in state.initiative_order if n in ws.characters
        )
        return f"先攻順序：{order}"

    if tag == "TRAVEL":
        direction = args.lower().strip()
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.move(direction)
        if not room:
            current = ws.dungeon_map.current_room
            valid = "、".join(current.exits.keys()) or "無"
            return f"無法往{direction}走（有效出口：{valid}）"
        # Auto-trigger combat if room has living enemies
        enemies = room.alive_enemies(ws.characters)
        if enemies and not room.cleared:
            pc_ids = [cid for cid, c in ws.characters.items() if not c.is_npc]
            char_ids = list(enemies.keys()) + pc_ids
            ws.combat = combat.roll_initiative(ws, char_ids)
            enemy_names = "、".join(c.name for c in enemies.values())
            return f"移動至：{room.name}。發現敵人：{enemy_names}！先攻順序：{'、'.join(ws.characters[n].name for n in ws.combat.initiative_order if n in ws.characters)}"
        return f"移動至：{room.name}"

    if tag == "PICKUP":
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            return f"無效 PICKUP 格式：{args}"
        char_id, item_name = parts
        char = ws.characters.get(char_id)
        if not char:
            return f"找不到角色：{char_id}"
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room
        for i, item in enumerate(room.loot):
            name = item.name if hasattr(item, "name") else str(item)
            if name == item_name:
                room.loot.pop(i)
                from ..engine.items import Weapon, Consumable
                if isinstance(item, Weapon):
                    char.weapons.append(item)
                    return f"{char.name} 拾取了 {item.name}"
                elif isinstance(item, Consumable):
                    existing = char.get_consumable(item.name)
                    if existing:
                        existing.quantity += item.quantity
                    else:
                        char.consumables.append(item)
                    return f"{char.name} 拾取了 {item.name}×{item.quantity}"
                else:
                    char.gear.append(str(item))
                    return f"{char.name} 拾取了 {item}"
        return f"房間內沒有「{item_name}」"

    if tag == "ATTACK":
        parts = [p.strip() for p in args.split("->")]
        if len(parts) != 2:
            return f"無效 ATTACK 格式：{args}"
        attacker = ws.characters.get(parts[0])
        target   = ws.characters.get(parts[1])
        if not attacker:
            return f"找不到攻擊者：{parts[0]}"
        if not target:
            return f"找不到目標：{parts[1]}"
        hit, total = combat.resolve_attack(attacker, target)
        result = "命中" if hit else "未命中"
        return f"{attacker.name} 攻擊 {target.name}，擲骰 {total} vs AC {target.ac}：{result}"

    if tag == "DAMAGE":
        m = re.match(r"(.+)\s+to\s+(.+)", args)
        if not m:
            return f"無效 DAMAGE 格式：{args}"
        dice_str, target_id = m.group(1).strip(), m.group(2).strip()
        target = ws.characters.get(target_id)
        if not target:
            return f"找不到角色：{target_id}"
        damage = combat.apply_damage(target, dice_str)
        return f"{target.name} 受到 {damage} 點傷害，剩餘 HP：{target.hp}/{target.max_hp}"

    if tag == "HEAL":
        parts = args.split()
        if len(parts) != 2:
            return f"無效 HEAL 格式：{args}"
        char_id, dice_str = parts
        char = ws.characters.get(char_id)
        if not char:
            return f"找不到角色：{char_id}"
        healed = combat.apply_heal(char, dice_str)
        return f"{char.name} 恢復 {healed} HP，現在 HP：{char.hp}/{char.max_hp}"

    if tag == "ROLL":
        parts = args.split()
        if len(parts) != 3:
            return f"無效 ROLL 格式：{args}"
        char_id, stat, dc_str = parts
        char = ws.characters.get(char_id)
        if not char:
            return f"找不到角色：{char_id}"
        try:
            dc = int(dc_str.upper().replace("DC", ""))
        except ValueError:
            return f"無效 ROLL 格式（DC值應為數字，收到：{dc_str}）：{args}"
        success, total = combat.make_saving_throw(char, stat.upper(), dc)
        result = "成功" if success else "失敗"
        return f"{char.name} {stat.upper()} 檢定 {total} vs DC {dc}：{result}"

    if tag == "STATUS":
        parts = args.split()
        if len(parts) != 2:
            return f"無效 STATUS 格式：{args}"
        char_id, effect = parts
        char = ws.characters.get(char_id)
        if not char:
            return f"找不到角色：{char_id}"
        if effect.startswith("+"):
            char.status_effects.append(effect[1:])
            return f"{char.name} 獲得狀態：{effect[1:]}"
        if effect.startswith("-"):
            eff = effect[1:]
            if eff in char.status_effects:
                char.status_effects.remove(eff)
            return f"{char.name} 移除狀態：{eff}"
        return f"無效 STATUS 效果：{effect}"

    return f"未知標籤：{tag}"
