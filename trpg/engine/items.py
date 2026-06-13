from dataclasses import dataclass, field


@dataclass
class Weapon:
    name: str
    damage_dice: str
    damage_type: str        # 斬擊 / 穿刺 / 鈍擊
    range_type: str         # 近戰 / 遠程
    range_normal: float = 1.5  # 公尺：近戰=伸手距離 / 遠程=正常射程
    range_long: float = 0.0    # 遠程武器最大射程（normal 內無 disadvantage、normal~long 有 disadvantage、long 外不可達）
    properties: list = field(default_factory=list)  # 輕巧 / 精巧 / 投擲
    ammo: str = ""          # 需要消耗的彈藥消耗品名稱 (e.g. "箭")
    # 命中附帶效果（怪物天然武器用，資料驅動）。支援欄位見
    # combat._apply_weapon_hit_rider 與 _resolve_single_attack 的 rider 區塊：
    #   status/save_stat/save_dc/save_each/rounds —— 命中附帶狀態（豁免否則）
    #   damage_dice/damage_type/save_stat/save_dc/save_half —— 附帶傷害
    #   drain_stat=("STR","1d4") / drain_max_hp=True —— 屬性/血上限吸取
    on_hit: dict | None = None


@dataclass
class Consumable:
    name: str
    quantity: int
    effect_type: str   # heal / light / ammo / utility / quest
    effect_value: str  # heal 用骰子式 e.g."1d4+2"；其餘為 ""


@dataclass
class Chest:
    name: str
    loot: list         # list[Weapon | Consumable | str]
    lock_dc: int = 0   # 0 = 未上鎖；正整數 = 需要 DEX 撬鎖
    opened: bool = False

    def visible_name(self) -> str:
        if self.opened:
            return f"{self.name}（已開啟）"
        if self.lock_dc > 0:
            return f"{self.name}（上鎖 DC{self.lock_dc}）"
        return self.name


@dataclass
class Hideout:
    """A concealed spot in a room. Items placed inside are not visible until SEARCH
    succeeds (roll >= find_dc). Once discovered, the spot is revealed permanently
    and its contents become pickupable, kept under the hideout for narrative context."""
    description: str           # 「牆角堆疊的木箱深處」「王座後的陰影」
    contents: list             # list[Weapon | Consumable | str]
    find_dc: int = 13
    discovered: bool = False


# ── 標準武器庫 ─────────────────────────────────────────────────────────────────
WEAPON_DEFS: dict[str, Weapon] = {
    "短劍":   Weapon("短劍",   "1d6", "穿刺", "近戰", range_normal=1.5, properties=["輕巧", "精巧"]),
    "長劍":   Weapon("長劍",   "1d8", "斬擊", "近戰", range_normal=1.5),
    "手斧":   Weapon("手斧",   "1d6", "斬擊", "近戰", range_normal=1.5, properties=["投擲"]),
    "匕首":   Weapon("匕首",   "1d4", "穿刺", "近戰", range_normal=1.5, properties=["輕巧", "精巧", "投擲"]),
    "彎刀":   Weapon("彎刀",   "1d6", "斬擊", "近戰", range_normal=1.5, properties=["輕巧", "精巧"]),
    "短弓":   Weapon("短弓",   "1d6", "穿刺", "遠程",
                    range_normal=24, range_long=96, ammo="箭"),
    "無武器": Weapon("無武器", "1d4", "鈍擊", "近戰", range_normal=1.5),
}
