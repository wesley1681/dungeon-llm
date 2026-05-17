import re
from ..engine import combat
from ..engine.world_state import WorldState

_TAG      = re.compile(r"\[([A-Z_]+):\s*([^\]]+)\]")
_BARE_TAG = re.compile(r"\[([A-Z_]+)\]")   # tags without arguments e.g. [TRAVEL]


def parse_and_resolve(text: str, world_state: WorldState,
                      skip_tags: set[str] | None = None) -> tuple[str, list[str]]:
    results: list[str] = []
    # TRAVEL and INITIATIVE should only fire once per narrative — ignore duplicates
    _once_tags_fired: set[str] = set()

    def _replace(match: re.Match) -> str:
        tag = match.group(1)
        if skip_tags and tag in skip_tags:
            return match.group(0)   # leave tag text as-is, don't execute
        if tag in {"TRAVEL", "INITIATIVE"} and tag in _once_tags_fired:
            return match.group(0)   # skip duplicate TRAVEL/INITIATIVE in same narrative
        try:
            result = _dispatch(tag, match.group(2).strip(), world_state)
        except Exception as e:
            result = f"標籤解析錯誤：{match.group(0)}（{e}）"
        if tag in {"TRAVEL", "INITIATIVE"}:
            _once_tags_fired.add(tag)
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


# Tags that must only fire in the narrative section (after ---), never from 機制 planning section.
# Only ROLL (and STATUS) are executed in the plan phase so the GM knows the result before writing.
_NARRATIVE_ONLY_TAGS = {"INITIATIVE", "ATTACK", "DAMAGE", "TRAVEL", "PICKUP", "HEAL", "CONSUME"}


_ERROR_PREFIXES = ("無效", "找不到", "標籤解析錯誤", "未知標籤")


def execute_all_tags(text: str, world_state: WorldState,
                     log_to_narrative: bool = False) -> tuple[list[str], list[str]]:
    """Execute every tag found in text (used for TagAgent output).

    Unlike parse_pre_narrative, does not skip TRAVEL/ATTACK/etc.
    Unlike parse_and_resolve, does not modify the source text.
    Returns (ok_results, error_results).

    log_to_narrative: if True, each ok result is pushed to narrative_log
        immediately after the tag executes. This preserves the per-tag
        snapshot of current_room/present — critical for tags like TRAVEL
        that mutate the visible scene mid-batch.
    """
    ok:     list[str] = []
    errors: list[str] = []
    seen_once: set[str] = set()   # TRAVEL / INITIATIVE / TALK / ATTACK_NPC / FLEE fire once per call
    for m in re.finditer(r"\[([A-Z_]+):\s*([^\]]+)\]", text):
        tag = m.group(1)
        if tag in {"TRAVEL", "INITIATIVE", "TALK", "ATTACK_NPC", "FLEE", "QUEST_ACCEPT", "QUEST_TURNIN", "RECRUIT"} and tag in seen_once:
            continue
        try:
            result = _dispatch(tag, m.group(2).strip(), world_state)
        except Exception as e:
            result = f"標籤解析錯誤：[{tag}]（{e}）"
        if any(result.startswith(p) for p in _ERROR_PREFIXES):
            errors.append(result)
        else:
            ok.append(result)
            if log_to_narrative:
                world_state.log_event("system", result)
            if tag in {"TRAVEL", "INITIATIVE", "TALK", "ATTACK_NPC", "FLEE", "QUEST_ACCEPT", "QUEST_TURNIN", "RECRUIT"}:
                seen_once.add(tag)
    return ok, errors


def parse_pre_narrative(text: str, world_state: WorldState) -> tuple[list[str], list[str], set[str]]:
    """Execute plan-phase tags (ROLL, STATUS) from the 機制 section.

    Only tags NOT in _NARRATIVE_ONLY_TAGS are executed here so the GM can receive
    actual dice results before writing the narrative (Option C two-pass generation).
    Returns (ok_results, error_results, set_of_tag_types_that_fired).
    ok_results: successful executions — injected into GM as pre_roll_results.
    error_results: format/parse failures — logged only, NOT sent to GM.
    """
    ok:    list[str] = []
    errors: list[str] = []
    fired: set[str]  = set()
    for match in re.finditer(r"\[([A-Z_]+):\s*([^\]]+)\]", text):
        tag = match.group(1)
        if tag in _NARRATIVE_ONLY_TAGS:
            continue
        try:
            result = _dispatch(tag, match.group(2).strip(), world_state)
        except Exception as e:
            result = f"標籤解析錯誤：[{tag}]（{e}）"
        if any(result.startswith(p) for p in _ERROR_PREFIXES):
            errors.append(result)
            # Don't add to fired: tag didn't actually execute
        else:
            ok.append(result)
            fired.add(tag)
    return ok, errors, fired


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


def _find_npc(key: str, ws: WorldState) -> tuple[str | None, object]:
    """Resolve (npc_id, char) from a key that may be ID or display name.
    Returns (None, None) when nothing matches. Used by TALK/ATTACK_NPC/FLEE
    which need the canonical npc_id for room lookups even if the LLM passed a name."""
    char = ws.characters.get(key)
    if char:
        return key, char
    for cid, c in ws.characters.items():
        if c.name == key:
            return cid, c
    return None, None


def _resolve_room_entry(room, ws: WorldState) -> str:
    """Dispatch hostile NPCs on room entry by their hostile_reaction.

    - attitude==0 + reaction=="flee" → execute FLEE (remove from room)
    - attitude==0 + reaction=="attack" → add to combat
    Returns a narrative summary string (may be empty).
    """
    attackers: list[str] = []
    flee_names: list[str] = []

    for nid in list(room.npc_ids):
        char = ws.characters.get(nid)
        if not char or not char.is_alive() or char.attitude != 0:
            continue
        if char.hostile_reaction == "flee":
            room.npc_ids.remove(nid)
            flee_names.append(char.name)
        else:   # "attack" or anything else defaults to attack
            attackers.append(nid)

    parts: list[str] = []
    if flee_names:
        parts.append(f"{'、'.join(flee_names)} 見到你們便慌忙逃走")

    if attackers:
        pc_ids = [cid for cid, c in ws.characters.items() if not c.is_npc]
        party_npc_ids = [nid for nid in room.npc_ids
                         if nid in ws.party_ids and ws.characters[nid].is_alive()]
        ws.combat = combat.roll_initiative(ws, attackers + pc_ids + party_npc_ids)
        attacker_names = "、".join(ws.characters[n].name for n in attackers)
        order = "、".join(
            ws.characters[n].name for n in ws.combat.initiative_order if n in ws.characters
        )
        parts.append(f"發現敵人：{attacker_names}！先攻順序：{order}")

    return "。".join(parts)


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
                nid for nid in room.npc_ids
                if nid in ws.characters
                   and ws.characters[nid].is_alive()
                   and ws.characters[nid].attitude == 0
            ]
            # No hostile NPCs in room → don't start combat
            if not alive_enemy_ids:
                return "（當前房間無敵人，跳過先攻）"
            party_npc_ids = [nid for nid in room.npc_ids
                             if nid in ws.party_ids and ws.characters[nid].is_alive()]
            char_ids = alive_enemy_ids + pc_ids + party_npc_ids
        state = combat.roll_initiative(ws, char_ids)
        ws.combat = state
        order = "、".join(
            ws.characters[n].name for n in state.initiative_order if n in ws.characters
        )
        return f"先攻順序：{order}"

    if tag == "UNLOCK":
        # Format: [UNLOCK: <char_id> <target_name> <stat>]
        # stat: DEX = lockpick (original DC), STR = brute force (DC+3)
        parts = args.split(maxsplit=2)
        if len(parts) < 2:
            return f"無效 UNLOCK 格式（需要：角色ID 目標名稱 [屬性]）：{args}"
        char_id    = parts[0]
        target_name = parts[1]
        stat       = parts[2].upper() if len(parts) == 3 else "DEX"
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        if not ws.dungeon_map:
            return "沒有地圖"
        from ..engine.items import Chest
        room = ws.dungeon_map.current_room
        # Search room loot for any Chest matching the target name
        for i, item in enumerate(room.loot):
            if not isinstance(item, Chest):
                continue
            if target_name not in item.name and item.name not in target_name:
                continue
            if item.lock_dc == 0:
                # Already unlocked — just reveal contents (chest stays in room)
                item.opened = True
                names = "、".join(c.name if hasattr(c, "name") else str(c) for c in item.loot) or "空的"
                return f"{item.name} 已解鎖，{char.name} 打開後發現：{names}"
            # DC depends on approach
            _DC_PENALTY = {"STR": 3}
            dc = item.lock_dc + _DC_PENALTY.get(stat, 0)
            success, total = combat.make_saving_throw(char, stat, dc)
            if not success:
                return f"{char.name} 解鎖失敗（{stat} {total} vs DC{dc}）——{item.name} 依然緊閉"
            # Success: unlock — chest stays in room.loot, contents remain inside.
            # PICKUP will reach inside any opened chest.
            item.lock_dc = 0
            item.opened  = True
            names = "、".join(c.name if hasattr(c, "name") else str(c) for c in item.loot) or "空的"
            return f"{char.name} 成功解鎖 {item.name}（{stat} {total} vs DC{dc}），發現：{names}"
        return f"找不到可解鎖的物件：{target_name}"

    if tag == "TRAVEL":
        if ws.combat and ws.combat.active:
            return "戰鬥中無法移動"
        direction = args.lower().strip()
        # Tolerate "[TRAVEL: aria east]" — strip leading character ID if present
        _valid_dirs = {"north", "south", "east", "west"}
        parts = direction.split()
        if len(parts) == 2 and parts[0] not in _valid_dirs and parts[1] in _valid_dirs:
            direction = parts[1]
        if not ws.dungeon_map:
            return "沒有地圖"
        old_room = ws.dungeon_map.current_room
        room = ws.dungeon_map.move(direction)
        if not room:
            valid = "、".join(old_room.exits.keys()) or "無"
            return f"無法往{direction}走（有效出口：{valid}）"
        # Follower NPCs in party tag along — move them to the new room's npc_ids
        for npc_id in list(ws.party_ids):
            c = ws.characters.get(npc_id)
            if not c or not c.is_npc or not c.is_alive():
                continue
            if npc_id in old_room.npc_ids:
                old_room.npc_ids.remove(npc_id)
            if npc_id not in room.npc_ids:
                room.npc_ids.append(npc_id)
        # A pending TALK from earlier in the same tag batch is stale once we've
        # left the room — the target NPC isn't here anymore.
        if ws.pending_conversation and ws.pending_conversation not in room.npc_ids:
            ws.pending_conversation = ""
        # Dispatch hostile NPCs by their reaction (attack vs flee)
        summary = _resolve_room_entry(room, ws) if not room.cleared else ""
        if summary:
            return f"移動至：{room.name}。{summary}"
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
        from ..engine.items import Weapon, Consumable, Chest

        def _give(item) -> str:
            if isinstance(item, Weapon):
                char.weapons.append(item)
                return f"{char.name} 拾取了 {item.name}"
            if isinstance(item, Consumable):
                existing = char.get_consumable(item.name)
                if existing:
                    existing.quantity += item.quantity
                else:
                    char.consumables.append(item)
                return f"{char.name} 拾取了 {item.name}×{item.quantity}"
            char.gear.append(str(item))
            return f"{char.name} 拾取了 {item}"

        # 1. Try top-level room loot
        for i, item in enumerate(room.loot):
            if isinstance(item, Chest):
                continue   # chests aren't directly pickupable
            name = item.name if hasattr(item, "name") else str(item)
            if name == item_name:
                room.loot.pop(i)
                return _give(item)

        # 2. Try inside any opened chest
        for i, item in enumerate(room.loot):
            if not isinstance(item, Chest) or not item.opened:
                continue
            for j, sub in enumerate(item.loot):
                sub_name = sub.name if hasattr(sub, "name") else str(sub)
                if sub_name == item_name:
                    item.loot.pop(j)
                    msg = _give(sub)
                    if not item.loot:
                        room.loot.pop(i)
                        msg += f"（{item.name} 已被清空）"
                    return msg

        # 3. Try inside any discovered hideout
        for i, h in enumerate(room.hideouts):
            if not h.discovered:
                continue
            for j, sub in enumerate(h.contents):
                sub_name = sub.name if hasattr(sub, "name") else str(sub)
                if sub_name == item_name:
                    h.contents.pop(j)
                    msg = _give(sub)
                    if not h.contents:
                        room.hideouts.pop(i)
                        msg += f"（{h.description} 已搜刮殆盡）"
                    return msg

        # Helpful hint when player tries to grab something only undiscovered hideouts might hold
        if any(not h.discovered for h in room.hideouts):
            return f"房間內沒有明顯可見的「{item_name}」（也許還有沒搜出來的角落）"
        return f"房間內沒有「{item_name}」"

    if tag == "GIVE":
        # Format: [GIVE: <from_id> <to_id> <item_name>]
        parts = args.split(maxsplit=2)
        if len(parts) != 3:
            return f"無效 GIVE 格式（需要：給的人ID 收的人ID 物品名）：{args}"
        from_id, to_id, item_name = parts
        giver = _find_char(from_id, ws)
        if not giver:
            return f"找不到角色：{from_id}"
        recipient = _find_char(to_id, ws)
        if not recipient:
            return f"找不到角色：{to_id}"
        if not giver.is_alive():
            return f"{giver.name} 已經倒下，無法交付"
        if not recipient.is_alive():
            return f"{recipient.name} 已經倒下，無法接收"
        # Same-room gate (when dungeon_map exists)
        if ws.dungeon_map:
            room = ws.dungeon_map.current_room
            def _in_room(char_id, char):
                if not char.is_npc:
                    return True   # PCs implicitly in current room
                return char_id in room.npc_ids
            # Find char_id for giver/recipient (may be name fallback)
            def _resolve_id(ref, char):
                if ref in ws.characters:
                    return ref
                for cid, c in ws.characters.items():
                    if c is char:
                        return cid
                return ref
            gid = _resolve_id(from_id, giver)
            rid = _resolve_id(to_id, recipient)
            if not _in_room(gid, giver) or not _in_room(rid, recipient):
                return f"{giver.name} 和 {recipient.name} 不在同一個房間，無法交付"

        # 1. Try weapons (whole weapon transfer)
        for i, w in enumerate(giver.weapons):
            if w.name == item_name:
                giver.weapons.pop(i)
                recipient.weapons.append(w)
                return f"{giver.name} 把 {w.name} 交給 {recipient.name}"

        # 2. Try consumables (transfer 1 unit; merge into recipient's existing stack)
        for c in giver.consumables:
            if c.name == item_name and c.quantity > 0:
                c.quantity -= 1
                existing = recipient.get_consumable(item_name)
                if existing:
                    existing.quantity += 1
                else:
                    from ..engine.items import Consumable
                    new_c = Consumable(c.name, 1, c.effect_type, c.effect_value)
                    recipient.consumables.append(new_c)
                # Clean up empty stack on giver
                if c.quantity <= 0:
                    giver.consumables.remove(c)
                return f"{giver.name} 把 {item_name} 交給 {recipient.name}"

        return f"{giver.name} 身上沒有「{item_name}」"

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

    if tag == "CONSUME":
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            return f"無效 CONSUME 格式：{args}"
        char_id, item_name = parts
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        if not char.consume(item_name):
            return f"{char.name} 沒有 {item_name} 可使用"
        return f"{char.name} 消耗了 {item_name} 1 個"

    if tag == "ROLL":
        _DEFAULT_DC = 12
        parts = args.split()
        # Accept 4-token form "[ROLL: aria DEX Investigation DC14]" — drop skill description
        if len(parts) == 4:
            parts = [parts[0], parts[1], parts[3]]
        # Tolerate missing DC: "[ROLL: aria WIS]" → use default DC
        if len(parts) == 2:
            parts = [parts[0], parts[1], f"DC{_DEFAULT_DC}"]
        if len(parts) != 3:
            return f"無效 ROLL 格式（無法解析：{args}）"
        char_id, stat, dc_str = parts
        # Tolerate "INT/WIS" style — take the first stat only
        stat = stat.split("/")[0]
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        dc_clean = dc_str.upper().replace("DC", "")
        # Tolerate placeholder like "DCXX" — fall back to default DC
        if not dc_clean.lstrip("-").isdigit():
            dc = _DEFAULT_DC
            dc_note = f"（DC佔位符，使用預設DC{_DEFAULT_DC}）"
        else:
            dc = int(dc_clean)
            dc_note = ""
        success, total = combat.make_saving_throw(char, stat.upper(), dc)
        result = "成功" if success else "失敗"
        return f"{char.name} {stat.upper()} 檢定 {total} vs DC {dc}{dc_note}：{result}"

    if tag == "STATUS":
        parts = args.split()
        if len(parts) != 2:
            return f"無效 STATUS 格式：{args}"
        char_id, effect = parts
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        if effect.startswith("+"):
            from ..engine.status import StatusEffect
            name = effect[1:]
            char.add_status(StatusEffect(name=name))
            return f"{char.name} 獲得狀態：{name}"
        if effect.startswith("-"):
            name = effect[1:]
            char.remove_status(name)
            return f"{char.name} 移除狀態：{name}"
        return f"無效 STATUS 效果：{effect}"

    if tag == "TALK":
        key = args.strip().split()[0]
        npc_id, char = _find_npc(key, ws)
        if not char:
            return f"找不到角色：{key}"
        if not char.is_alive():
            return f"無法與 {char.name} 交談——他已經倒下"
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room
        if npc_id not in room.npc_ids:
            return f"找不到 NPC：{char.name}（當前房間無此 NPC）"
        ws.pending_conversation = npc_id
        return f"開始與 {char.name} 對話"

    if tag == "ATTACK_NPC":
        key = args.strip().split()[0]
        npc_id, char = _find_npc(key, ws)
        if not char:
            return f"找不到角色：{key}"
        if not char.is_alive():
            return f"無法攻擊 {char.name}——他已經倒下"
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room
        if npc_id not in room.npc_ids:
            return f"找不到 NPC：{char.name}（當前房間無此 NPC）"
        char.attitude = 0          # 立刻變敵意 → 進入戰鬥
        room.cleared = False
        ws.pending_conversation = ""   # cancel any pending TALK; combat takes precedence
        # Betrayal: attacking a follower instantly removes them from the party
        if npc_id in ws.party_ids:
            ws.party_ids.remove(npc_id)
            agent = _npc_agent_registry.get(npc_id) if _npc_agent_registry else None
            if agent is not None:
                agent.in_party = False
        pc_ids = [cid for cid, c in ws.characters.items() if not c.is_npc]
        party_npc_ids = [nid for nid in room.npc_ids
                         if nid in ws.party_ids and ws.characters[nid].is_alive()]
        ws.combat = combat.roll_initiative(ws, [npc_id] + pc_ids + party_npc_ids)
        order = "、".join(ws.characters[n].name for n in ws.combat.initiative_order if n in ws.characters)
        return f"{char.name} 變為敵人！先攻順序：{order}"

    if tag == "FLEE":
        key = args.strip().split()[0]
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room
        npc_id, char = _find_npc(key, ws)
        if not char:
            return f"找不到角色：{key}"
        if npc_id in room.npc_ids:
            room.npc_ids.remove(npc_id)
        if ws.combat and ws.combat.active and npc_id in ws.combat.initiative_order:
            ws.combat.initiative_order.remove(npc_id)
        return f"{char.name} 逃跑離開了"

    if tag == "SEARCH":
        # Format: [SEARCH: <char_id> [stat]]    stat defaults to INT
        parts = args.split()
        if not parts:
            return f"無效 SEARCH 格式：{args}"
        char_id = parts[0]
        stat = (parts[1].upper() if len(parts) > 1 else "INT")
        char = _find_char(char_id, ws)
        if not char:
            return f"找不到角色：{char_id}"
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room

        undiscovered = [h for h in room.hideouts if not h.discovered]
        if not undiscovered:
            return f"{char.name} 仔細搜索，沒有發現任何特別之物"

        # Roll once. Reveal every hideout whose find_dc ≤ total.
        min_dc = min(h.find_dc for h in undiscovered)
        _, total = combat.make_saving_throw(char, stat, min_dc)
        revealed = []
        for h in undiscovered:
            if total >= h.find_dc:
                h.discovered = True
                names = "、".join(
                    c.name if hasattr(c, "name") else str(c) for c in h.contents
                ) or "（空）"
                revealed.append(f"{h.description}（{names}）")
        if revealed:
            return f"{char.name} 搜索（{stat} {total}）→ 發現：{'；'.join(revealed)}"
        return f"{char.name} 搜索（{stat} {total}）→ 沒看出名堂，可能還漏看了什麼"

    if tag == "RECRUIT":
        key = args.strip().split()[0]
        npc_id, char = _find_npc(key, ws)
        if not char:
            return f"找不到角色：{key}"
        if not char.is_alive():
            return f"無法招募 {char.name}——他已經倒下"
        if npc_id in ws.party_ids:
            return f"{char.name} 已經跟著你們了"
        if char.attitude < 2:
            return f"{char.name} 不夠信任你，不會跟你走"
        agent = _npc_agent_registry.get(npc_id) if _npc_agent_registry else None
        if agent is None:
            return f"找不到 {char.name} 的 NpcAgent"
        agent.pending_join_decision = True
        return f"{char.name} 看著你，似乎在考慮…"

    if tag == "QUEST_ACCEPT":
        qid = args.strip().split()[0]
        quest = ws.quests.get(qid)
        if not quest:
            return f"找不到任務：{qid}"
        if quest.status == "active":
            return f"任務「{quest.title}」已在進行中"
        if quest.status in ("completed", "turned_in"):
            return f"任務「{quest.title}」已完成"
        quest.status = "active"
        return f"接受任務：{quest.title}"

    if tag == "QUEST_TURNIN":
        qid = args.strip().split()[0]
        quest = ws.quests.get(qid)
        if not quest:
            return f"找不到任務：{qid}"
        if quest.status != "completed":
            return f"任務「{quest.title}」尚未達成完成條件（目前 {quest.status}）"
        from ..engine.quests import apply_reward
        lines = apply_reward(quest, ws)
        # Pre-scripted reward delivery: NPC MUST tell the player exactly these info
        # lines this turn (via pending_reveal). Also extend permanent secrets so later
        # turns can reference them via the normal reveal flow.
        info = (quest.reward or {}).get("info") or []
        agent = _npc_agent_registry.get(quest.giver_id) if _npc_agent_registry else None
        if info and agent is not None:
            agent._secrets.extend(info)
            agent.pending_reveal = list(info)
        quest.status = "turned_in"
        suffix = "；" + "，".join(lines) if lines else ""
        return f"完成任務：{quest.title}{suffix}"

    return f"未知標籤：{tag}"


# Optional registry for NpcAgent so QUEST_TURNIN can push info into NPC secrets.
# Set externally (e.g. game.py) after building NpcAgents.
_npc_agent_registry: dict | None = None


def set_npc_agent_registry(registry: dict) -> None:
    global _npc_agent_registry
    _npc_agent_registry = registry
