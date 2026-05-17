from ..engine.character import Character, Stats
from ..engine.items import WEAPON_DEFS, Weapon, Consumable, Chest, Hideout
from ..engine.world_state import WorldState
from ..engine.dungeon_map import DungeonMap, Room
from ..engine.quests import Quest
from ..llm.npc_agent import NpcAgent

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

THOR_TACTICS = """## 行為原則（依你的個性決定何時做什麼）
- 戰鬥前環視場景：若房裡有比你現用武器更合適的兵器（對遠程敵人時撿弓、敵人多時撿火瓶），戰鬥結束後撿起、適時換裝
- 戰鬥中分階段：
  - HP > 50%：衝鋒主攻，必要時用「我衝向 X，再用 [武器] 攻擊 X」一次移動加攻擊
  - HP 30~50%：考慮閃避（「我專注防禦」→ 對方擲劣勢）或拋擲手斧拉節奏
  - HP < 30%：使用急救包恢復；或閃避保命
- 弓手敵人退到後排時：選擇「衝上去」靠近，否則手斧投擲（短弓範圍不會碰到）
- 對 NPC 不老實或藏匿情報：你體格高大、聲音渾厚，**主動嘗試威嚇**（「不說我就讓你說」）——不要只當沉默後援讓凱恩單打獨鬥
- 探索中若有明確下一步，主動建議：「往北」「先回去找老柯」「檢查那個寶箱」
- 隊友（凱恩）已經在發起社交時，你選擇 [SILENT]，不搶話；但隊友被攻擊或威脅時，第一個衝出去護衛
"""

CIVILIAN_ID = "civilian"
CIVILIAN_PERSONALITY = (
    "你叫老柯，是個採藥人，中年男子，善良但膽小，說話時帶著顫抖。"
    "你今天進入這座廢棄地下城採集罕見藥草，結果迷路走不出去了。"
    "你已經在入口走廊縮了好幾個小時，北邊傳來地精的嘈雜聲讓你嚇得動彈不得。"
    "你看到兩個武裝的陌生人，不確定他們是好人還是壞人，謹慎地觀察著。"
    "如果對方表現友善，你會慢慢放鬆，並分享你聽到的情報；"
    "如果對方粗魯或恐嚇你，你會更加恐懼，閉口不言。"
)

CIVILIAN_TACTICS = """## 行為原則（依你的個性決定何時做什麼）
- 你不會打架、沒武器：戰鬥中**主動退到後排**（「我躲到索爾後面」），讓位置變遠離敵人
- 被近戰敵人追上時：用閃避動作（「我抱頭蹲下」「我躲開」）撐到隊友幫忙
- HP < 50% 直接 <FLEE>
- 採藥人本能：房間裡若提到藥草、植物、菌類，主動補一句「這個我認得 / 這個有毒」
- 對話中對方友善時：可以主動補充你聽到的線索（顫抖地、不確定地），不必硬等對方問
- 對方粗魯時：縮起來、不回應、結巴；不要編造資訊討好他們
- 若加入隊伍跟著走，每進一個新房間可以表達緊張（「這裡好暗⋯⋯」），但不要每次都講
"""


def build_world_state() -> WorldState:
    # ── 玩家角色 ────────────────────────────────────────────────────────────────
    aria = Character(
        name="凱恩", race="人類", class_="盜賊", level=3,
        stats=Stats(STR=100, DEX=16, CON=12, INT=13, WIS=11, CHA=50),
        hp=22, max_hp=22, ac=14,
        weapons=[
            WEAPON_DEFS["短劍"],
            WEAPON_DEFS["匕首"],
            Weapon("神話長劍", "4d6+10", "斬擊", "近戰"),
        ],
        consumables=[
            Consumable("火把", 3, "light",  ""),
            Consumable("火瓶", 2, "throw",  "2d100"),   # AOE 投擲，DEX DC13 豁免半傷
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

    # ── NPC（個性、初始態度由 NpcAgent 控制；attitude=0 即敵意）─────────────────
    civilian = Character(
        name="老柯", race="人類", class_="平民", level=1,
        stats=Stats(STR=9, DEX=10, CON=10, INT=11, WIS=12, CHA=11),
        hp=6, max_hp=6, ac=10,
        is_npc=True,
        attitude=1,                   # 戒備
        hostile_reaction="flee",      # 膽小，敵意時會逃
    )

    goblin_1 = Character(
        name="地精甲", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        weapons=[WEAPON_DEFS["彎刀"]],
        is_npc=True,
        attitude=0,   # 敵意
    )

    goblin_2 = Character(
        name="地精乙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13,
        weapons=[WEAPON_DEFS["彎刀"]],
        is_npc=True,
        attitude=0,
    )

    goblin_3 = Character(
        name="地精丙", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=5, max_hp=5, ac=12,
        weapons=[WEAPON_DEFS["短弓"]],
        consumables=[Consumable("箭", 20, "ammo", "")],
        is_npc=True,
        attitude=0,
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
        attitude=0,
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
                    "角落裡縮著一個衣衫凌亂的中年男子，看起來嚇得不輕。"
                ),
                exits={"north": "guard_room"},
                npc_ids=["civilian"],
                cleared=True,
                visited=True,
            ),
            "guard_room": Room(
                id="guard_room",
                name="守衛室",
                description=(
                    "寬敞的石室，地上散落著啃過的骨頭和破爛的毯子，"
                    "角落堆著生鏽的武器架。"
                    "東邊有扇半開的木門，南邊是來時的走廊。"
                ),
                exits={"south": "entrance", "east": "storage_room"},
                npc_ids=["goblin_1", "goblin_2"],
                loot=[
                    Consumable("治療藥水", 1, "heal", "2d4+2"),
                    # 守衛室的月光草明顯可見（散落在骨堆旁）
                    Consumable("月光草", 1, "quest", ""),
                ],
            ),
            "storage_room": Room(
                id="storage_room",
                name="儲藏室",
                description=(
                    "堆滿雜物的小房間，木箱和麻袋靠牆堆疊。"
                    "北邊有一扇厚重的石門，西邊通往守衛室。"
                ),
                exits={"west": "guard_room", "north": "boss_chamber"},
                hideouts=[
                    Hideout(
                        description="牆角堆疊的木箱深處",
                        contents=[Consumable("月光草", 1, "quest", "")],
                        find_dc=13,
                    ),
                ],
                loot=[
                    WEAPON_DEFS["手斧"],
                    Chest(
                        name="鐵箱",
                        lock_dc=13,
                        loot=[
                            Consumable("治療藥水", 2, "heal", "2d4+2"),
                            Consumable("急救包",   1, "heal", "1d4+2"),
                            "金幣 30 枚",
                        ],
                    ),
                ],
                cleared=True,
            ),
            "boss_chamber": Room(
                id="boss_chamber",
                name="頭目室",
                description=(
                    "較大的石室，中央擺著一把用骨頭裝飾的破舊王座，"
                    "牆上掛著粗糙的戰旗，地上零散著被啃過的骨頭與酒甕。"
                    "這裡只有南邊一條退路。"
                ),
                exits={"south": "storage_room"},
                npc_ids=["goblin_boss", "goblin_3"],
                loot=[
                    Consumable("護符", 1, "quest", ""),
                ],
                hideouts=[
                    Hideout(
                        description="骨頭王座背後的陰影",
                        contents=[Consumable("月光草", 1, "quest", "")],
                        find_dc=12,
                    ),
                ],
            ),
        },
        current_room_id="entrance",
    )

    # ── 任務 ────────────────────────────────────────────────────────────────────
    moonlight_quest = Quest(
        id="moonlight_grass",
        title="採集月光草",
        description="你想拜託對方替你採集三朵月光草——你聽說地下城深處的潮濕陰暗角落長著這種藥草，但你自己不敢進去。作為回報你願意把自己藏的應急藥草分給對方。",
        giver_id=CIVILIAN_ID,
        objective={"type": "collect", "item": "月光草", "count": 3},
        reward={
            "items": [
                Consumable("治療藥水", 2, "heal", "2d4+2"),
                Consumable("急救包",   1, "heal", "1d4+2"),
            ],
            "attitude_delta": 2,        # 戒備(1) → 友好(3)，直接越過 reveal_threshold
            "recipient_id":   "aria",
        },
    )

    return WorldState(
        characters={
            "aria":        aria,
            "thor":        thor,
            "civilian":    civilian,
            "goblin_1":    goblin_1,
            "goblin_2":    goblin_2,
            "goblin_3":    goblin_3,
            "goblin_boss": goblin_boss,
        },
        scene=OPENING_SCENE,
        dungeon_map=dungeon_map,
        scenario_name="地下城探索：失竊的護符",
        quests={moonlight_quest.id: moonlight_quest},
    )


CIVILIAN_SECRETS = [
    "守衛室裡有兩隻地精，一隻拿彎刀，一隻拿弓，躲在骨頭堆後面。",
    "我聽到更北邊有一個比其他地精更低沉的吼聲在指揮，應該是個頭目。",
    "我聽他大聲罵其他地精『別靠近火把』，那聲音特別兇。",
]

GOBLIN_GRUNT_PERSONALITY = (
    "你是一隻地精小兵，矮小、骯髒、貪婪又膽小，講話沙啞且夾雜咒罵。"
    "你怕死但更怕被頭目處罰，所以面對入侵者通常會擺出兇狠姿態。"
    "如果對方明顯比你強大很多，你會猶豫、甚至想逃，但會虛張聲勢掩飾。"
    "你只認得簡單的通用語，常用「人類」「侏儒」等詞罵人。"
)

GOBLIN_ARCHER_PERSONALITY = (
    "你是一隻拿弓的地精，比同伴稍微機警一點，會躲在掩護後射箭。"
    "你怕近身戰鬥，若敵人逼近你會慌張地後退。"
    "說話尖銳急促，常用威脅但缺乏底氣。"
)

GOBLIN_BOSS_PERSONALITY = (
    "你是地精頭目葛茲，比一般地精更強壯、更狡猾，懂得策略與恐嚇。"
    "你貪婪、自大，認為自己是這座地下城的主人。"
    "你怕火（火把、火瓶會讓你慌張），但會用憤怒掩飾恐懼。"
    "對手下兇狠，對敵人傲慢，被激怒會做出魯莽決定。"
)


def build_npc_agents(world_state: WorldState, model: str,
                     base_url: str, backend: str) -> dict:
    """Build all NPC agents for this scenario. Add new NPCs here only."""
    def _agent(cid, personality, tactics="", secrets=None, reveal=3, quests=None):
        return NpcAgent(
            model=model,
            char_id=cid,
            character=world_state.characters[cid],
            personality=personality,
            tactics=tactics,
            secrets=secrets,
            reveal_threshold=reveal,
            quests=quests,
            world_state=world_state,
            base_url=base_url,
            backend=backend,
        )
    civilian_quests = [q for q in world_state.quests.values() if q.giver_id == CIVILIAN_ID]
    return {
        CIVILIAN_ID:   _agent(CIVILIAN_ID, CIVILIAN_PERSONALITY,
                              tactics=CIVILIAN_TACTICS,
                              secrets=CIVILIAN_SECRETS, reveal=3,
                              quests=civilian_quests),
        "goblin_1":    _agent("goblin_1",    GOBLIN_GRUNT_PERSONALITY),
        "goblin_2":    _agent("goblin_2",    GOBLIN_GRUNT_PERSONALITY),
        "goblin_3":    _agent("goblin_3",    GOBLIN_ARCHER_PERSONALITY),
        "goblin_boss": _agent("goblin_boss", GOBLIN_BOSS_PERSONALITY),
    }
