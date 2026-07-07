"""v05 · 代號 codeword-noarch-enemyskill · 讓模型讀「敵人技能組」。

前身 codeword-noarch。改動(相對 v04)——加 obs v8 + `encode_entity_skills`:
  - obs 新增 `entity_skills`[10,20,66]＋mask:每個實體的**靜態 kit**(kit_features)機制向量。
  - 模型用**共享的** skill encoder(self 那把)把每個實體的技能總結成向量,經 **zero-init**
    投影加進該實體的 ent_emb。zero-init ⇒ 從 v04 warm-start bit-exact,直到投影訓練。
動機(數據,scripts 探針):capability_descriptor 這份「威脅輪廓」把技能組不同、數值接近的
敵人糊成一團——raging berserker vs champion(L2 1.29,暴怒抗性隱形)、兩個 L6 法師
evocation vs divination(L2 **0.00** 完全相同)。entity_skills 下同樣兩對 L2=37.8 / 19.4
＝分得開。為「扮演標準職業＋各種怪物」的更難泛化任務做準備(全職/怪物 roster 下大量 kit
會撞同一份聚合)。驗收:tests/test_obs_v8_entity_skills.py(zero-init 惰性、訓練後真讀、
全空實體不 NaN);obs 改動對 v04/uni_v10 bit-exact(eval_gate 38/0)。
"""
from ..model import CombatPolicyNet

CODENAME = "codeword-noarch-enemyskill"
PARENT = "codeword-noarch"
DATE = "2026-07-06"


class Net(CombatPolicyNet):
    """codeword-noarch + encode_entity_skills(讀每個實體的靜態 kit)。"""
    def __init__(self, hidden: int = 128):
        super().__init__(hidden=hidden, n_head_groups=1, skill_combo_dim=8,
                         drop_noop_h=True, ablate_archetype=True,
                         encode_entity_skills=True)
