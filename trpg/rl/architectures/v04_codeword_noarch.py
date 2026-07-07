"""v04 · 代號 codeword-noarch · codeword **結構性移除職業**(非遮罩,是不存在)。

前身 codeword。改動(相對 codeword)——4 處消費職業的地方全部拔掉(ablate_archetype):
  - FiLM(film_gamma/beta,職業條件化 h=γh+β)整條**不建、不套用**
  - entity_mlp 入寬 127→115:進第一層前**切掉**每列的職業 one-hot 欄(權重不存在)
  - end_head 切掉 end_features 的職業尾
  - per-arch 路由固定 head0,不讀職業欄
逼模型只靠技能 codeword 推自己身分。實驗(general5 no_arch)結論:codeword 足以承載身分、
職業標籤非必要(總分 59%≈baseline 60%);但收斂慢2×且失穩 → FiLM 真作用=梯度隔離讓 5 職
平行穩定收斂。驗收:tests/test_ablate_archetype.py(擾動任何職業欄→輸出變化 exactly 0)。
"""
from ..model import CombatPolicyNet

CODENAME = "codeword-noarch"
PARENT = "codeword"
DATE = "2026-07-06"


class Net(CombatPolicyNet):
    """codeword-noarch:codeword + ablate_archetype(職業結構性不存在於架構)。"""
    def __init__(self, hidden: int = 128):
        super().__init__(hidden=hidden, n_head_groups=1, skill_combo_dim=8,
                         drop_noop_h=True, ablate_archetype=True)
