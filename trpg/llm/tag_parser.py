import re
from ..engine import combat
from ..engine.world_state import WorldState

_TAG      = re.compile(r"\[([A-Z]+):\s*([^\]]+)\]")
_BARE_TAG = re.compile(r"\[([A-Z]+)\]")   # tags without arguments e.g. [TRAVEL]


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

    # Catch bare tags like [TRAVEL] that are missing arguments
    def _bare_replace(m: re.Match) -> str:
        tag = m.group(1)
        msg = f"標籤格式錯誤：[{tag}] 缺少參數（例如應寫 [{tag}: direction]）"
        results.append(msg)
        return f"（{msg}）"

    cleaned = _BARE_TAG.sub(_bare_replace, cleaned)
    return cleaned, results


_COMBAT_TAGS = {"INITIATIVE", "ATTACK", "DAMAGE"}  # never execute these from pre-narrative


def parse_pre_narrative(text: str, world_state: WorldState) -> tuple[list[str], set[str]]:
    """Process non-combat tags from the 機制 section in order.
    Returns (result_strings, set_of_tag_types_that_fired)."""
    results: list[str] = []
    fired:   set[str]  = set()
    for match in re.finditer(r"\[([A-Z]+):\s*([^\]]+)\]", text):
        tag = match.group(1)
        if tag in _COMBAT_TAGS:
            continue
        try:
            result = _dispatch(tag, match.group(2).strip(), world_state)
        except Exception as e:
            result = f"標籤解析錯誤：[{tag}]（{e}）"
        results.append(result)
        fired.add(tag)
    return results, fired


def parse_travel_only(text: str, world_state: WorldState) -> list[str]:
    """Execute only TRAVEL tags found in text. Returns result strings."""
    results = []
    for m in re.finditer(r"\[TRAVEL:\s*([^\]]+)\]", text):
        try:
            results.append(_dispatch("TRAVEL", m.group(1).strip(), world_state))
        except Exception as e:
            results.append(f"TRAVEL 錯誤：{e}")
    return results


def _find_char(key: str, ws: WorldState):
    """Look up a character by ID first, then by name as fallback."""
    char = ws.characters.get(key)
    if char is None:
        char = next((c for c in ws.characters.values() if c.name == key), None)
    return char


def _dispatch(tag: str, args: str, ws: WorldState) -> str:

    if tag == "INITIATIVE":
        # No-op if combat already active (e.g. started by TRAVEL this same turn)
        if ws.combat and ws.combat.active:
            return f"先攻順序：{'、'.join(ws.characters[n].name for n in ws.combat.initiative_order if n in ws.characters)}"
        # Scope to current room when dungeon_map is active
        char_ids = None
        if ws.dungeon_map:
            room = ws.dungeon_map.current_room
            pc_ids = [cid for cid, c in ws.characters.items() if not c.is_npc]
            alive_enemy_ids = [
                eid for eid in room.enemy_ids
                if eid in ws.characters and ws.characters[eid].is_alive()
            ]
            # No enemies in room → don't start combat
            if not alive_enemy_ids:
                return "（當前房間無敵人，跳過先攻）"
            char_ids = alive_enemy_ids + pc_ids
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
        char = _find_char(char_id, ws)
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
        attacker = _find_char(parts[0], ws)
        target   = _find_char(parts[1], ws)
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
        target = _find_char(target_id, ws)
        if not target:
            return f"找不到角色：{target_id}"
        damage = combat.apply_damage(target, dice_str)
        return f"{target.name} 受到 {damage} 點傷害，剩餘 HP：{target.hp}/{target.max_hp}"

    if tag == "HEAL":
        parts = args.split()
        if len(parts) != 2:
            return f"無效 HEAL 格式：{args}"
        char_id, dice_str = parts
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        healed = combat.apply_heal(char, dice_str)
        return f"{char.name} 恢復 {healed} HP，現在 HP：{char.hp}/{char.max_hp}"

    if tag == "ROLL":
        parts = args.split()
        if len(parts) != 3:
            return f"無效 ROLL 格式：{args}"
        char_id, stat, dc_str = parts
        char = _find_char(char_id, ws)
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
        char = _find_char(char_id, ws)
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
