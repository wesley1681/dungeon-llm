import json
import pathlib
from .backend import stream_chat

_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"

_SYSTEM = """你是 D&D 5e 社交難度估算員。根據玩家的社交嘗試內容與對話歷史，輸出適當的 DC 值。

## NPC 資訊
{personality}
當前態度：{attitude_label}（{attitude}/4）

## DC 估算規則

### 欺騙（deceive）
DC 取決於謊言可信度、脈絡貼合度、NPC 疑心程度：
- DC 8：完全符合常識，毫不可疑（「我是附近村莊的人」）
- DC10：普通小謊言，無明顯漏洞（「我是過路的旅人」）
- DC12：需要一定信任的謊言（「我奉命來調查這裡」）
- DC14：有漏洞但仍可能的謊言（「我是皇家密探，沒有證件是機密任務」）
- DC16：明顯可疑（「我認識你們頭目，他命令你服從我」）
- DC18：非常荒謬，正常人不會相信（「我是龍的使者，龍命令你服從我」）
- DC22：超出理性範圍（「我是神明的化身」「我是上帝」「我有永生的力量」）
- DC28：完全不可能信（「整個世界是我創造的」「你其實是我的夢境」）

### 說服（persuade）
- DC 8：符合對方利益，有明確好處（「幫我帶路，我給你一筆錢」）
- DC10：合理的請求（「告訴我這裡的狀況，我們可以幫你脫困」）
- DC12：需要信任的請求（「把你知道的秘密告訴我們」）
- DC14：對方有顧慮或損失的請求（「跟我們一起冒險，很危險」）
- DC16：明顯違反對方意願（「把你所有錢都給我們」）
- DC20：幾乎不可能同意的請求

### 恐嚇（intimidate）
- DC 8：對方已極度恐懼，隨便說說就嚇到
- DC10：輕微威脅姿態，拔武器示威
- DC12：直接言語威脅，語氣強硬
- DC14：高壓逼迫，逼到牆角
- DC16：涉及肢體傷害或傷害第三方的威脅
- DC20：極端恐嚇，對方可能選擇魚死網破

## 重複嘗試懲罰
若對話記錄中玩家已嘗試相同或高度相似的手段：
- 第 2 次同樣方式：DC +3
- 第 3 次：DC +6
- 第 4 次以上：DC +10（對方已完全免疫這套說詞）

## 輸出格式
只輸出一個整數，不加任何說明或文字。
"""

_TYPE_LABELS = {"intimidate": "恐嚇", "persuade": "說服", "deceive": "欺騙"}


class SocialDcAgent:
    def __init__(self, model: str, base_url: str, backend: str, options: dict = None):
        self.model    = model
        self.base_url = base_url
        self.backend  = backend
        self.options  = options or {"temperature": 0.1, "num_predict": 10}

    def estimate(self, social_type: str, attempt_text: str,
                 personality: str, attitude: int, attitude_label: str,
                 conv_log_text: str) -> int:
        """Return estimated DC for a social skill attempt.

        conv_log_text: pre-rendered multi-line string of prior dialogue
        (one `speaker：text` per line). Pass "" for first contact.
        """
        log_text = conv_log_text if conv_log_text.strip() else "（尚無對話記錄）"

        type_label = _TYPE_LABELS.get(social_type, social_type)
        user_content = (
            f"[{type_label}嘗試]：{attempt_text}\n\n"
            f"【完整對話記錄】\n{log_text}"
        )

        system = _SYSTEM.format(
            personality=personality,
            attitude=attitude,
            attitude_label=attitude_label,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ]
        (_DEBUG_DIR / "social_dc_context.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace"
        )

        result = stream_chat(
            self.base_url, self.model, messages, self.options,
            think=False, on_chunk=None, backend=self.backend, timeout=30,
        )

        for word in result.strip().split():
            try:
                return int(word)
            except ValueError:
                continue
        return 12  # fallback
