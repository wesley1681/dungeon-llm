import json
import pathlib
import re
from .rulebook import COMBAT_RULEBOOK
from .backend import complete_chat

OLLAMA_URL = "http://localhost:11434"
_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEBUG_DIR.mkdir(exist_ok=True)


class ArbiterAgent:
    def __init__(self, model: str, base_url: str = OLLAMA_URL, backend: str = "ollama"):
        self.model = model
        self.base_url = base_url
        self.backend = backend
        self.options = {"temperature": 0.1, "num_predict": 512}

    def parse(
        self,
        player_action: str,
        actor_id: str,
        actor_name: str,
        available_targets: dict[str, str],  # {id: name}，只含存活目標
        actor_char=None,                    # Character 物件，提供武器/消耗品列表
    ) -> dict:
        # Arbiter only translates intent → JSON. Range / distance / resources /
        # damage are engine concerns and not shown — keeping the prompt narrow
        # makes parsing more reliable. Engine errors trigger a retry upstream.
        if actor_char:
            weapons_str = "、".join(w.name for w in actor_char.weapons) or "無武器"
            heals = [
                f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
                for c in actor_char.consumables
                if c.effect_type == "heal" and c.quantity > 0
            ]
            throws = [
                f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
                for c in actor_char.consumables
                if c.effect_type == "throw" and c.quantity > 0
            ]
            items_str  = "、".join(heals)  or "無"
            throws_str = "、".join(throws) or "無"
        else:
            weapons_str = "未知"
            items_str   = "未知"
            throws_str  = "未知"

        target_lines = [f"{name}（{cid}）" for cid, name in available_targets.items()]
        targets_str = "\n  ".join(target_lines) if target_lines else "無"

        situation = (
            f"## 當前情況\n"
            f"行動者：{actor_name}（{actor_id}）\n"
            f"可用武器：{weapons_str}\n"
            f"可用治療道具：{items_str}\n"
            f"可用投擲物：{throws_str}\n"
            f"可攻擊目標：\n  {targets_str}\n"
        )

        messages = [{
            "role": "user",
            "content": (
                f"{COMBAT_RULEBOOK}\n\n{situation}\n"
                f"## 玩家行動\n{player_action}"
            ),
        }]

        (_DEBUG_DIR / f"arbiter_{actor_id}_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )

        content = complete_chat(
            self.base_url, self.model, messages, self.options,
            backend=self.backend, timeout=60,
        )
        return self._extract_json(content, player_action)

    _CONSUMES_DEFAULT = {
        "ATTACK": ["action"],
        "AOE":    ["action"],
        "USE_ITEM": ["action"],
        "ROLL":   ["action"],
        "DODGE":  ["action"],
        "HIDE":   ["action"],
        "MOVE":   ["movement"],
    }

    def _apply_defaults(self, action: dict) -> dict:
        """Fill in `consumes` if missing for known valid action types.
        Invalid actions are left untouched."""
        if not action.get("valid"):
            return action
        t = action.get("type", "").upper()
        if "consumes" not in action and t in self._CONSUMES_DEFAULT:
            action["consumes"] = list(self._CONSUMES_DEFAULT[t])
        return action

    def _extract_json(self, text: str, original_action: str) -> dict:
        # Strip markdown code fences (```json ... ```)
        cleaned = re.sub(r"```\w*", "", text).strip()

        start = cleaned.find("{")
        if start == -1:
            return self._fail(text)

        depth = end = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end:
            try:
                return self._apply_defaults(json.loads(cleaned[start:end]))
            except json.JSONDecodeError:
                pass

        fragment = re.sub(r"[,\s\)\]]*$", "", cleaned[start:])
        open_count = fragment.count("{") - fragment.count("}")
        if open_count > 0:
            try:
                return self._apply_defaults(json.loads(fragment + "}" * open_count))
            except json.JSONDecodeError:
                pass

        return self._fail(text)

    def _fail(self, text: str) -> dict:
        return {
            "valid": False,
            "reason": f"判定器無法解析（原始輸出：{text[:80]}）",
            "suggestion": "請重新描述你的行動",
        }
