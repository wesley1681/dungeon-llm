import re
from ..engine import combat
from ..engine.world_state import WorldState

_TAG = re.compile(r"\[([A-Z]+):\s*([^\]]+)\]")


def parse_and_resolve(text: str, world_state: WorldState) -> tuple[str, list[str]]:
    """
    Find all tags in text, resolve via game engine, return (cleaned_text, result_strings).
    Tags are replaced inline with their resolved result in parentheses.
    """
    results: list[str] = []

    def _replace(match: re.Match) -> str:
        result = _dispatch(match.group(1), match.group(2).strip(), world_state)
        results.append(result)
        return f"（{result}）"

    cleaned = _TAG.sub(_replace, text)
    return cleaned, results


def _dispatch(tag: str, args: str, ws: WorldState) -> str:
    if tag == "INITIATIVE":
        state = combat.roll_initiative(ws)
        ws.combat = state
        order = "、".join(
            ws.characters[n].name for n in state.initiative_order if n in ws.characters
        )
        return f"先攻順序：{order}"

    if tag == "ATTACK":
        parts = [p.strip() for p in args.split("->")]
        if len(parts) != 2:
            return f"無效 ATTACK 格式：{args}"
        attacker = ws.characters.get(parts[0])
        target = ws.characters.get(parts[1])
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
        dc = int(dc_str.upper().replace("DC", ""))
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
