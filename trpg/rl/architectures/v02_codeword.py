"""v02 · 代號 codeword · 單一共享頭 + 技能組合 codeword。

前身 uni。改動(相對 uni):
  - 12 個 per-arch 頭 → **1 個共享頭**(n_head_groups=1)
  - **加技能 codeword**(skill_combo_dim=8):把技能池壓成一個向量餵頭,逼從技能池推身分
  - 拿掉 skill/entity 頭的死權重常數 h(drop_noop_h)
  仍保留 FiLM 職業條件化與手刻免疫 join。
「接下來實驗預設」(架構指令);exp_scratch fighter/mixed/battlemaster、general5 baseline 用它。
"""
from ..model import CombatPolicyNet

CODENAME = "codeword"
PARENT = "uni"
DATE = "2026-07-05"


class Net(CombatPolicyNet):
    """codeword:單頭 + skill_combo_dim=8 + drop_noop_h(FiLM/免疫 join 仍在)。"""
    def __init__(self, hidden: int = 128):
        super().__init__(hidden=hidden, n_head_groups=1, skill_combo_dim=8,
                         drop_noop_h=True)
