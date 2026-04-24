from ..engine.character import Character, Stats
from ..engine.world_state import WorldState

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
    aria = Character(
        name="艾里亞", race="人類", class_="盜賊", level=3,
        stats=Stats(STR=10, DEX=16, CON=12, INT=13, WIS=11, CHA=14),
        hp=22, max_hp=22, ac=14,
        inventory=["短劍", "匕首×2", "盜賊工具", "火把×3", "繩索 15 尺"],
        proficiencies=["DEX", "INT", "潛行", "開鎖", "察覺", "欺騙"],
        is_npc=False,
    )

    thor = Character(
        name="索爾", race="人類", class_="戰士", level=3,
        stats=Stats(STR=17, DEX=12, CON=15, INT=9, WIS=10, CHA=11),
        hp=31, max_hp=31, ac=16,
        inventory=["長劍", "手斧", "鏈甲", "盾牌", "火把×2", "急救包"],
        proficiencies=["STR", "CON", "運動", "恐嚇"],
        is_npc=False,
    )

    goblin_1 = Character(
        name="地精甲", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        inventory=["彎刀"],
        is_npc=True,
    )

    goblin_2 = Character(
        name="地精乙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        inventory=["彎刀", "小盾"],
        is_npc=True,
    )

    goblin_3 = Character(
        name="地精丙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=5, max_hp=5, ac=12,
        inventory=["短弓", "箭×20"],
        is_npc=True,
    )

    return WorldState(
        characters={
            "aria": aria,
            "thor": thor,
            "goblin_1": goblin_1,
            "goblin_2": goblin_2,
            "goblin_3": goblin_3,
        },
        scene=OPENING_SCENE,
        scenario_name="地下城探索：失竊的護符",
    )
