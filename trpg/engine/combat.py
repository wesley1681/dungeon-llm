from .dice import roll
from .character import Character, CombatState
from .world_state import WorldState


def roll_initiative(world_state: WorldState,
                    character_ids: list[str] | None = None) -> CombatState:
    """Roll initiative. If character_ids given, only include those characters."""
    chars = world_state.characters
    if character_ids is not None:
        chars = {k: v for k, v in chars.items() if k in character_ids}
    results = {
        cid: roll("1d20") + char.stats.modifier("DEX")
        for cid, char in chars.items()
        if char.is_alive()
    }
    order = sorted(results, key=lambda n: results[n], reverse=True)
    return CombatState(initiative_order=order)


def resolve_attack(attacker: Character, target: Character,
                   weapon=None) -> tuple[bool, int]:
    """Returns (hit, total_roll). Uses STR, or DEX if weapon has 精巧 and DEX is higher."""
    if weapon is None:
        weapon = attacker.get_weapon()
    if "精巧" in (weapon.properties if weapon else []):
        stat_mod = max(attacker.stats.modifier("STR"), attacker.stats.modifier("DEX"))
    else:
        stat_mod = attacker.stats.modifier("STR")
    attack_roll = roll("1d20")
    total = attack_roll + stat_mod + attacker.proficiency_bonus
    return total >= target.ac, total


def apply_damage(target: Character, damage_notation: str) -> int:
    damage = roll(damage_notation)
    target.hp = max(0, target.hp - damage)
    return damage


def apply_heal(target: Character, dice_notation: str) -> int:
    amount = roll(dice_notation)
    target.hp = min(target.max_hp, target.hp + amount)
    return amount


def make_saving_throw(character: Character, stat: str, dc: int) -> tuple[bool, int]:
    roll_result = roll("1d20")
    modifier = character.stats.modifier(stat)
    if stat in character.proficiencies:
        modifier += character.proficiency_bonus
    total = roll_result + modifier
    return total >= dc, total


def execute_action(action: dict, world_state: WorldState) -> dict:
    """Execute a parsed action JSON from the arbiter. Returns a result summary dict."""
    t = action.get("type")

    # ── ATTACK ────────────────────────────────────────────────────────────────
    if t == "ATTACK":
        attacker = world_state.characters.get(action.get("attacker", ""))
        target   = world_state.characters.get(action.get("target",   ""))
        if not attacker or not target:
            return {"type": "ERROR", "message": "找不到攻擊者或目標"}
        if not target.is_alive():
            return {"type": "ERROR", "message": f"{target.name} 已倒下，無法攻擊"}

        weapon = attacker.get_weapon(action.get("weapon", ""))

        # Consume ammo for ranged weapons
        if weapon.ammo:
            if not attacker.has_ammo(weapon.ammo):
                return {"type": "ERROR", "message": f"{attacker.name} 沒有 {weapon.ammo} 了"}
            attacker.consume(weapon.ammo)

        # Ability modifier used for both attack and damage (finesse takes higher of STR/DEX)
        if "精巧" in weapon.properties:
            dmg_mod = max(attacker.stats.modifier("STR"), attacker.stats.modifier("DEX"))
        elif weapon.range_type == "遠程":
            dmg_mod = attacker.stats.modifier("DEX")
        else:
            dmg_mod = attacker.stats.modifier("STR")

        hit, roll_total = resolve_attack(attacker, target, weapon)
        result = {
            "type":          "ATTACK",
            "attacker_name": attacker.name,
            "target_name":   target.name,
            "weapon_name":   weapon.name,
            "roll":          roll_total,
            "target_ac":     target.ac,
            "hit":           hit,
        }
        if hit:
            base_dmg = roll(weapon.damage_dice)
            damage   = max(1, base_dmg + dmg_mod)
            target.hp = max(0, target.hp - damage)
            result.update({
                "damage":        damage,
                "damage_dice":   weapon.damage_dice,
                "damage_mod":    dmg_mod,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "target_alive":  target.is_alive(),
            })
        return result

    # ── AOE ───────────────────────────────────────────────────────────────────
    if t == "AOE":
        attacker   = world_state.characters.get(action.get("attacker", ""))
        target_ids = action.get("targets", [])
        item_name  = action.get("item", "")
        damage_dice = action.get("damage_dice", "1d6")
        save_stat   = action.get("save_stat", "DEX").upper()
        save_dc     = int(action.get("save_dc", 13))
        half_on_save = action.get("half_on_save", True)

        if item_name and attacker:
            if not attacker.consume(item_name):
                return {"type": "ERROR", "message": f"{attacker.name} 沒有「{item_name}」了"}

        target_results = []
        for tid in target_ids:
            target = world_state.characters.get(tid)
            if not target or not target.is_alive():
                continue
            success, save_roll = make_saving_throw(target, save_stat, save_dc)
            full_dmg   = roll(damage_dice)
            actual_dmg = full_dmg // 2 if (success and half_on_save) else full_dmg
            target.hp  = max(0, target.hp - actual_dmg)
            target_results.append({
                "target_name":  target.name,
                "save_roll":    save_roll,
                "save_success": success,
                "damage":       actual_dmg,
                "target_hp":    target.hp,
                "target_max_hp": target.max_hp,
                "target_alive": target.is_alive(),
            })

        remaining = None
        if item_name and attacker:
            c = attacker.get_consumable(item_name)
            remaining = c.quantity if c else 0

        return {
            "type":           "AOE",
            "attacker_name":  attacker.name if attacker else "未知",
            "item":           item_name,
            "damage_dice":    damage_dice,
            "save_stat":      save_stat,
            "save_dc":        save_dc,
            "target_results": target_results,
            "remaining":      remaining,
        }

    # ── USE_ITEM ──────────────────────────────────────────────────────────────
    if t == "USE_ITEM":
        char = world_state.characters.get(action.get("character", ""))
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        item_name = action.get("item", "")
        c = char.get_consumable(item_name)
        if not c:
            return {"type": "ERROR", "message": f"找不到道具：{item_name}"}
        if c.quantity <= 0:
            return {"type": "ERROR", "message": f"{item_name} 已用完"}

        if c.effect_type == "heal":
            target_id = action.get("target", action.get("character", ""))
            target = world_state.characters.get(target_id, char)
            healed = apply_heal(target, c.effect_value)
            c.quantity -= 1
            return {
                "type":          "USE_ITEM",
                "item":          item_name,
                "character":     char.name,
                "target":        target.name,
                "healed":        healed,
                "target_hp":     target.hp,
                "target_max_hp": target.max_hp,
                "remaining":     c.quantity,
            }
        # Non-heal consumables (light, utility)
        c.quantity -= 1
        return {
            "type":      "USE_ITEM",
            "item":      item_name,
            "character": char.name,
            "remaining": c.quantity,
        }

    # ── ROLL ──────────────────────────────────────────────────────────────────
    if t == "ROLL":
        char = world_state.characters.get(action.get("character", ""))
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        stat = action.get("stat", "DEX").upper()
        dc   = int(action.get("dc", 12))
        success, total = make_saving_throw(char, stat, dc)
        return {
            "type":           "ROLL",
            "character_name": char.name,
            "stat":           stat,
            "total":          total,
            "dc":             dc,
            "success":        success,
            "skill":          action.get("skill_description", ""),
        }

    # ── MOVE ──────────────────────────────────────────────────────────────────
    if t == "MOVE":
        return {
            "type":        "MOVE",
            "character":   action.get("character", ""),
            "description": action.get("description", "移動"),
        }

    # ── HIDE ──────────────────────────────────────────────────────────────────
    if t == "HIDE":
        char = world_state.characters.get(action.get("character", ""))
        if not char:
            return {"type": "ERROR", "message": "找不到角色"}
        success, total = make_saving_throw(char, "DEX", 12)
        if success:
            char.status_effects.append("hidden")
        return {
            "type":    "HIDE",
            "success": success,
            "total":   total,
        }

    return {"type": "ERROR", "message": f"未知行動類型：{t}"}


def format_result(player_description: str, result: dict, actor_name: str = "") -> str:
    """Convert execute_action result into a text summary for the GM."""
    label = f"【{actor_name}的行動】" if actor_name else "【玩家行動】"
    lines = [f"{label}{player_description}"]
    t = result.get("type")

    if t == "ATTACK":
        hit_str = "命中" if result["hit"] else "未命中"
        lines.append(
            f"使用 {result['weapon_name']}，攻擊骰 {result['roll']} vs AC {result['target_ac']}：{hit_str}"
        )
        if result["hit"]:
            alive    = "存活" if result.get("target_alive") else "倒下"
            dmg_mod  = result.get("damage_mod", 0)
            mod_str  = f"+{dmg_mod}" if dmg_mod > 0 else (str(dmg_mod) if dmg_mod < 0 else "")
            dice_str = f"{result['damage_dice']}{mod_str}"
            lines.append(
                f"造成 {result['damage']} 點傷害（{dice_str}），"
                f"{result['target_name']} HP {result['target_hp']}/{result['target_max_hp']}（{alive}）"
            )

    elif t == "AOE":
        lines.append(
            f"投擲 {result['item']}（{result['damage_dice']}，"
            f"{result['save_stat']} DC{result['save_dc']} 豁免半傷）"
        )
        for tr in result.get("target_results", []):
            save_str  = "豁免成功（半傷）" if tr["save_success"] else "豁免失敗"
            alive_str = "存活" if tr["target_alive"] else "倒下"
            lines.append(
                f"  {tr['target_name']}：{save_str}，受 {tr['damage']} 傷害，"
                f"HP {tr['target_hp']}/{tr['target_max_hp']}（{alive_str}）"
            )
        if result.get("remaining") is not None:
            lines.append(f"剩餘 {result['item']}：{result['remaining']} 個")

    elif t == "USE_ITEM":
        if result.get("healed") is not None:
            lines.append(
                f"使用 {result['item']}：{result['target']} 恢復 {result['healed']} HP"
                f"（{result['target_hp']}/{result['target_max_hp']}），剩餘 {result['remaining']} 個"
            )
        else:
            lines.append(f"使用 {result['item']}，剩餘 {result['remaining']} 個")

    elif t == "ROLL":
        outcome = "成功" if result["success"] else "失敗"
        skill = f"（{result['skill']}）" if result.get("skill") else ""
        lines.append(
            f"{result['character_name']} {result['stat']}{skill} 檢定 "
            f"{result['total']} vs DC {result['dc']}：{outcome}"
        )

    elif t == "MOVE":
        lines.append(f"移動：{result['description']}")

    elif t == "HIDE":
        outcome = "成功隱身" if result["success"] else "躲藏失敗"
        lines.append(f"躲藏 DEX 檢定 {result['total']}：{outcome}")

    elif t == "ERROR":
        lines.append(f"錯誤：{result['message']}")

    return "\n".join(lines)
