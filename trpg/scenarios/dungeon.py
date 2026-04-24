from ..engine.character import Character, Stats
from ..engine.items import WEAPON_DEFS, Weapon, Consumable
from ..engine.world_state import WorldState
from ..engine.dungeon_map import DungeonMap, Room

OPENING_SCENE = (
    "你們是一對受僱的冒險者，任務是從地精手中奪回小鎮鎮長的護身符。"
    "線索指向鎮外廢棄的石造地下城。\n\n"
    "火把的光芒在潮濕的石壁上搖曳，眼前是一條向下延伸的台階走廊，"
    "空氣中瀰漫著霉味和隱約的腐臭氣息。某處傳來地精粗嘎的笑聲。"
)

THOR_PERSONALITY = (
    "索爾是一位直率、勇敢的北方戰士。他崇尚榮耀戰鬥，堅決保護隊友。"
    "說話簡短有力，行動優先於思考，面對危險時第一個衝上去。"
)


def build_world_state() -> WorldState:
    # ── 玩家角色 ────────────────────────────────────────────────────────────────
    aria = Character(
        name="艾里亞", race="人類", class_="盜賊", level=3,
        stats=Stats(STR=10, DEX=16, CON=12, INT=13, WIS=11, CHA=14),
        hp=22, max_hp=22, ac=14,
        weapons=[
            WEAPON_DEFS["短劍"],
            WEAPON_DEFS["匕首"],
            Weapon("神話長劍", "4d6+10", "斬擊", "近戰"),
        ],
        consumables=[
            Consumable("火把", 3, "light",  ""),
            Consumable("火瓶", 2, "throw",  "2d6"),   # AOE 投擲，DEX DC13 豁免半傷
            Consumable("毒煙彈", 1, "throw", "1d6"),  # AOE 投擲，CON DC12 豁免半傷
        ],
        gear=["盜賊工具", "繩索 15 尺"],
        proficiencies=["DEX", "INT", "潛行", "開鎖", "察覺", "欺騙"],
        is_npc=False,
    )

    thor = Character(
        name="索爾", race="人類", class_="戰士", level=3,
        stats=Stats(STR=17, DEX=12, CON=15, INT=9, WIS=10, CHA=11),
        hp=31, max_hp=31, ac=16,
        weapons=[WEAPON_DEFS["長劍"], WEAPON_DEFS["手斧"]],
        consumables=[
            Consumable("急救包", 2, "heal", "1d4+2"),
            Consumable("火把",   2, "light", ""),
        ],
        gear=["鏈甲", "盾牌"],
        proficiencies=["STR", "CON", "運動", "恐嚇"],
        is_npc=False,
    )

    # ── NPC（各自屬於不同房間）──────────────────────────────────────────────────
    goblin_1 = Character(
        name="地精甲", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        weapons=[WEAPON_DEFS["彎刀"]],
        is_npc=True,
    )

    goblin_2 = Character(
        name="地精乙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        weapons=[WEAPON_DEFS["彎刀"]],
        is_npc=True,
    )

    goblin_3 = Character(
        name="地精丙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=5, max_hp=5, ac=12,
        weapons=[WEAPON_DEFS["短弓"]],
        consumables=[Consumable("箭", 20, "ammo", "")],
        is_npc=True,
    )

    goblin_boss = Character(
        name="地精頭目葛茲", race="地精", class_="—", level=3,
        stats=Stats(STR=13, DEX=14, CON=12, INT=10, WIS=9, CHA=12),
        hp=21, max_hp=21, ac=15,
        weapons=[WEAPON_DEFS["彎刀"], WEAPON_DEFS["短弓"]],
        consumables=[Consumable("箭", 10, "ammo", "")],
        gear=["破舊皮甲", "護符（贓物）"],
        proficiencies=["STR", "DEX"],
        is_npc=True,
    )

    # ── 地圖 ────────────────────────────────────────────────────────────────────
    dungeon_map = DungeonMap(
        rooms={
            "entrance": Room(
                id="entrance",
                name="入口走廊",
                description=(
                    "潮濕的石造走廊，火把在生鏽的壁架上燃燒，地面覆滿苔蘚。"
                    "走廊向北延伸，隱約傳來嘈雜聲。"
                ),
                exits={"north": "guard_room"},
                cleared=True,
                visited=True,
            ),
            "guard_room": Room(
                id="guard_room",
                name="守衛室",
                description=(
                    "寬敞的石室，地上散落著啃過的骨頭和破爛的毯子。"
                    "兩隻地精正懶洋洋地靠在牆邊，手邊放著生鏽的彎刀。"
                    "東邊有扇半開的木門，南邊是來時的走廊。"
                ),
                exits={"south": "entrance", "east": "storage_room"},
                enemy_ids=["goblin_1", "goblin_2"],
                loot=[Consumable("治療藥水", 1, "heal", "2d4+2")],
            ),
            "storage_room": Room(
                id="storage_room",
                name="儲藏室",
                description=(
                    "堆滿雜物的小房間，木箱和麻袋靠牆堆疊，角落有一個上鎖的鐵箱。"
                    "北邊有一扇厚重的石門，西邊通往守衛室。"
                ),
                exits={"west": "guard_room", "north": "boss_chamber"},
                loot=[
                    WEAPON_DEFS["手斧"],
                    Consumable("急救包", 1, "heal", "1d4+2"),
                ],
                cleared=True,
            ),
            "boss_chamber": Room(
                id="boss_chamber",
                name="頭目室",
                description=(
                    "較大的石室，中央擺著一把用骨頭裝飾的破舊王座。"
                    "地精頭目葛茲正坐在上面，脖子上掛著一枚閃亮的護符，"
                    "旁邊站著一隻持短弓的地精衛兵。"
                    "這裡只有南邊一條退路。"
                ),
                exits={"south": "storage_room"},
                enemy_ids=["goblin_boss", "goblin_3"],
                loot=["護符"],
            ),
        },
        current_room_id="entrance",
    )

    return WorldState(
        characters={
            "aria":        aria,
            "thor":        thor,
            "goblin_1":    goblin_1,
            "goblin_2":    goblin_2,
            "goblin_3":    goblin_3,
            "goblin_boss": goblin_boss,
        },
        scene=OPENING_SCENE,
        dungeon_map=dungeon_map,
        scenario_name="地下城探索：失竊的護符",
    )
