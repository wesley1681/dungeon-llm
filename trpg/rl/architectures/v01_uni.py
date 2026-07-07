"""v01 · 代號 uni · 生產統一線(uni_v1…uni_v10)的架構。

12 個 per-archetype 頭 + FiLM 職業條件化 + 手刻免疫/抗性 join、無技能 codeword。
最早、也是目前交付模型(models/unified/uni_v10.pt)所用的架構。後續實驗都從這裡分支。
前身:無(基準)。
"""
from ..model import CombatPolicyNet

CODENAME = "uni"
PARENT = None
DATE = "≤2026-06"


class Net(CombatPolicyNet):
    """uni:CombatPolicyNet 全預設(12頭 / FiLM / 免疫 join / 無 codeword)。"""
    def __init__(self, hidden: int = 128):
        super().__init__(hidden=hidden)
