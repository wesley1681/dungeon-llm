"""v03 · 代號 codeword-noimmune · codeword 拿掉兩條手刻免疫 join。

前身 codeword。改動(相對 codeword):
  - **結構性移除** typed-resist join + condition-immunity join(ablate_immunity_joins),
    逼模型只靠 sk_ent_ctx(技能×實體注意力)自己讀免疫,別無手刻捷徑。
免疫實驗(exp_scratch_immune)用它,證表徵層讀得出免疫(desc-ON 對、desc-OFF 塌回亂猜)。
"""
from ..model import CombatPolicyNet

CODENAME = "codeword-noimmune"
PARENT = "codeword"
DATE = "2026-07-06"


class Net(CombatPolicyNet):
    """codeword-noimmune:codeword + ablate_immunity_joins(免疫只能靠 sk_ent_ctx)。"""
    def __init__(self, hidden: int = 128):
        super().__init__(hidden=hidden, n_head_groups=1, skill_combo_dim=8,
                         drop_noop_h=True, ablate_immunity_joins=True)
