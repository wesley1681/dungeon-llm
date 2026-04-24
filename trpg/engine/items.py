from dataclasses import dataclass, field


@dataclass
class Weapon:
    name: str
    damage_dice: str
    damage_type: str        # 斬擊 / 穿刺 / 鈍擊
    range_type: str         # 近戰 / 遠程
    range_normal: int = 1   # 公尺
    range_long: int = 0     # 遠程武器長射程
    properties: list = field(default_factory=list)  # 輕巧 / 精巧 / 投擲
    ammo: str = ""          # 需要消耗的彈藥消耗品名稱 (e.g. "箭")


@dataclass
class Consumable:
    name: str
    quantity: int
    effect_type: str   # heal / light / ammo / utility
    effect_value: str  # heal 用骰子式 e.g."1d4+2"；其餘為 ""


# ── 標準武器庫 ─────────────────────────────────────────────────────────────────
WEAPON_DEFS: dict[str, Weapon] = {
    "短劍":   Weapon("短劍",   "1d6", "穿刺", "近戰", properties=["輕巧", "精巧"]),
    "長劍":   Weapon("長劍",   "1d8", "斬擊", "近戰"),
    "手斧":   Weapon("手斧",   "1d6", "斬擊", "近戰", properties=["投擲"]),
    "匕首":   Weapon("匕首",   "1d4", "穿刺", "近戰", properties=["輕巧", "精巧", "投擲"]),
    "彎刀":   Weapon("彎刀",   "1d6", "斬擊", "近戰", properties=["輕巧"]),
    "短弓":   Weapon("短弓",   "1d6", "穿刺", "遠程",
                    range_normal=24, range_long=96, ammo="箭"),
    "無武器": Weapon("無武器", "1d4", "鈍擊", "近戰"),
}
