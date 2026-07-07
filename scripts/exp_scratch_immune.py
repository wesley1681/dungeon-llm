"""實驗:拔掉 typed-join + cimmun-join,模型只能靠 sk_ent_ctx 學會「避開免疫傷害型」。

問題:v10 的免疫能力靠一條**手刻** join(skill 傷害型 ⋅ 敵人 typed-resist 原始欄位)
直接算出來餵給 skill head。使用者質疑:sk_ent_ctx 既然是 sk_emb×entity 的注意力交互,
按理該能自己讀出免疫。本實驗把兩條 join 全部歸零(CombatPolicyNetNoArch(ablate_
immunity_joins=True)),讓 sk_ent_ctx 成為免疫資訊的**唯一**通路,從零 PPO 直接驗。

設定(全部沿用 train_switch_lab 的機制,見 scripts/train_switch_lab.py):
  - agent = 一個 kit 同時持兩把**同傷害、不同型**的合成武器:火焰之刃(1d8火)/
    冰霜之刃(1d8冰)。兩者都是 at-will,available_skills 同時列出 → 唯一差別=傷害型。
  - 敵人站樁(EndTurnPolicy),開局就擺在近戰觸及內(1.0m<1.5m reach)→ 位移不是變數,
    唯一要決策的就是「用哪一型打」。
  - 免疫 = damage_multipliers[type]=0.0 → 引擎 int(dmg*0)=0(真免疫);obs 端該型
    I_DESC_RESIST 欄 = mult-1 = -1。

  --mode fixed  = 實驗1:免疫恆為火。注意這只是**能力下限**——固定免疫可用「冰霜之刃
                  是死招」記住通關,不需要讀敵人。descriptor-off 仍 ~0% 即記憶而非讀取。
  --mode random = 實驗2:每局在 火/冰 隨機挑一型免疫。哪把是對的**逐局翻轉**,模型
                  **必須**從敵人 obs 讀免疫才能選對 → 這才是 sk_ent_ctx 的真測。

判讀:wrong%(造成傷害的攻擊裡,砸在免疫型上的比例;→0=每次選對)+ WR。
  因果檢:eval 時把敵人 I_DESC_RESIST 欄歸零(zero_enemy_resist)——若 random 模式
  wrong% 從 ~0% 跳回 ~50%,證明模型確實靠 sk_ent_ctx 讀免疫(join 已拔,別無他路)。

用法:
  python scripts/exp_scratch_immune.py --smoke                      # 先驗機制
  python scripts/exp_scratch_immune.py --mode fixed  --updates 40 --out_dir models/exp_immune_fixed
  python scripts/exp_scratch_immune.py --mode random --updates 40 --out_dir models/exp_immune_random
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TRPG_WASTED_MOVE_COST", "0")   # fresh PPO:別讓 wmc 擋探索

import argparse, random
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry use
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import (build_obs, I_DESC_RESIST, N_DAMAGE_TYPES, I_ENT_ENEMY,
                         ENEMY_SLOT_START)
from trpg.engine.skill import available_skills
from trpg.engine.items import WEAPON_DEFS, Weapon
from trpg.engine.combat_policy import EndTurnPolicy, CombatDecision
from trpg.engine.vec2 import Vec2
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, ClassDef, _factory)
from probe_fire_switch import action_damage_type

FIRE, COLD = "火", "冰"
W_FIRE, W_COLD = "火焰之刃", "冰霜之刃"
KIT = "imm2"          # agent:同持兩把同傷害不同型的武器,只有這兩個傷害來源
DUMMY = "champion"    # 站樁敵人,HP/AC 覆蓋
MELEE_REACH_M = 1.5


class StationaryAttacker:
    """站樁但會還手:在武器 reach 內就打最近敵人,**永不移動**(honor 站樁=不走位)。
    引入「還手」才讓選錯型(免疫→0 傷)的浪費回合有真代價(己方挨打掉血),
    否則對不還手假人選錯零代價、模型學不到辨別(見 exp_immune_calib.py 診斷)。"""
    def _nearest_enemy(self, actor_id, ws):
        actor = ws.characters[actor_id]; is_party = ws.is_party_ally(actor_id)
        best, bd = None, 1e9
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive():
                continue
            if ws.is_party_ally(cid) == is_party:
                continue
            d = actor.position.distance_to(c.position)
            if d < bd:
                bd, best = d, cid
        return best, bd

    def decide(self, actor_id, actor, ws, resources, round_number):
        tid, d = self._nearest_enemy(actor_id, ws)
        if tid is None:
            return CombatDecision(ended=True)
        wpn = actor.get_weapon() if actor.weapons else None
        reach = (wpn.range_normal or 1.5) if wpn else 1.5
        if d <= reach + 1e-6 and resources.get("action", 0) > 0 and wpn is not None:
            skills = available_skills(actor, ws)
            sk = next((s for s in skills if s.skill_id == f"weapon:{wpn.name}"), None)
            if sk is not None:
                return CombatDecision(action=sk.build_action(
                    actor_id, tid, ws.characters[tid].position))
        return CombatDecision(ended=True)          # 搆不到就原地結束,絕不移動


def register_kit():
    # 兩把合成武器:同 1d8,不同傷害型 → 唯一差別=型
    WEAPON_DEFS.setdefault(W_FIRE, Weapon(W_FIRE, "1d8", FIRE, "近戰", range_normal=1.5))
    WEAPON_DEFS.setdefault(W_COLD, Weapon(W_COLD, "1d8", COLD, "近戰", range_normal=1.5))
    cd = ClassDef(
        archetype_id=KIT, default_name=KIT, class_display="免疫實驗",
        role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=10, CHA=10),
        hp_base=10, hp_per_level=6, ac=15,
        weapons=(W_FIRE, W_COLD), proficiencies=("STR", "CON"),
        skills=(), traits=())
    CLASS_DEFS[KIT] = cd
    ARCHETYPE_FACTORIES[KIT] = _factory(KIT)
    ARCHETYPE_ROLES[KIT] = cd.role


def make_env(seed, immune_type, hp=24, ac=13, level=3, opp_level=3,
             enemy_fights=False, agent_hp=0, enemy_regen=0):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[KIT], opp_archs=[DUMMY],
              level=level, opp_level=opp_level, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    o = env.ws.characters[oid]
    # 站樁=不移動;enemy_fights 決定會不會還手
    env._opp_policies[oid] = StationaryAttacker() if enemy_fights else EndTurnPolicy()
    o.damage_multipliers[immune_type] = 0.0           # 注入免疫
    o.max_hp = hp; o.hp = hp; o.ac = ac
    if enemy_regen > 0:
        # 每回合回血 enemy_regen(blocked_by=[] → 恆回),與傷害型無關(不洩漏免疫):
        # 選錯型=0 傷,被回血抵銷 → 唯有「正確型輸出 > 回血」才有淨進度 = 辨別壓力
        o.regeneration = {"amount": int(enemy_regen), "blocked_by": []}
    if agent_hp > 0:
        a = env.ws.characters[aid]; a.max_hp = agent_hp; a.hp = agent_hp
    # 把 agent 擺進近戰觸及內 → 位移非變數,只剩「選哪一型」
    op = o.position
    env.ws.characters[aid].position = Vec2(min(29.5, op.x + 1.0), op.y)
    return env


def zero_enemy_resist(ob):
    """因果控制:把敵人列的 typed-resist 欄歸零(移除免疫訊號,型全變中性)。
    join 已拔,若模型仍選對就只能是靠 sk_ent_ctx 讀這段 → 歸零後該塌回亂猜。"""
    ob = dict(ob)
    ent = ob["entities"].copy()                        # [E, D]
    enemy = ent[:, I_ENT_ENEMY] > 0.5
    ent[enemy, I_DESC_RESIST:I_DESC_RESIST + N_DAMAGE_TYPES] = 0.0
    ob["entities"] = ent
    return ob


# ── rollout ────────────────────────────────────────────────────────────────
_BUF_KEYS = ("obs", "act", "lp", "rew", "val", "done", "skm", "enm", "grm")


def _immune_of(mode, rng):
    if mode == "fixed":
        return FIRE
    return FIRE if rng.random() < 0.5 else COLD


def collect(net, n_steps, seed, rng, mode, cfg, gamma=0.99, lam=0.95):
    net.eval()
    buf = {k: [] for k in _BUF_KEYS}
    ep = 0; wrong = ndmg = 0
    while len(buf["rew"]) < n_steps:
        imm = _immune_of(mode, rng)
        env = make_env(seed * 1_000_003 + ep, imm, cfg.hp, cfg.ac,
                       cfg.level, cfg.opp_level, enemy_fights=bool(cfg.enemy_fights),
                       enemy_regen=cfg.enemy_regen, agent_hp=cfg.agent_hp)
        aid0 = env.agent_ids[0]
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid,
                    mask_immune_null=(not bool(cfg.ablate)))
            a = action.numpy().tolist()
            if aid == aid0 and a[0] > 0:               # 追蹤選錯型
                sks = available_skills(env.ws.characters[aid], env.ws)
                if a[0] < len(sks) and sks[a[0]].features.expected_damage > 0:
                    o = env.ws.characters[env.opp_ids[0]]
                    dt = action_damage_type(
                        sks[a[0]].build_action(aid, env.opp_ids[0],
                                               (o.position.x, o.position.y)),
                        env.ws.characters[aid])
                    if dt:
                        ndmg += 1; wrong += (dt == imm)
            obs2, r, term, trunc, _ = env.step(a)
            buf["obs"].append(obs); buf["act"].append(a); buf["lp"].append(lp.numpy())
            buf["rew"].append(float(r)); buf["val"].append(val)
            buf["skm"].append(skm.numpy()); buf["enm"].append(enm.numpy())
            buf["grm"].append(grm.numpy())
            done = term or trunc; buf["done"].append(bool(done))
            obs = obs2
        ep += 1
    rewards = np.array(buf["rew"], np.float32); values = np.array(buf["val"], np.float32)
    dones = np.array(buf["done"], np.float32)
    adv, ret = _compute_gae(rewards, values, dones, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in buf["obs"]]) for k in buf["obs"][0]},
        "actions": np.array(buf["act"], np.int64),
        "log_probs": np.array(buf["lp"], np.float32),
        "skill_masks": np.stack(buf["skm"]).astype(np.bool_),
        "entity_masks": np.stack(buf["enm"]).astype(np.bool_),
        "grid_masks": np.stack(buf["grm"]).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, ep, wrong / max(1, ndmg)


# ── 探針(greedy):descriptor-on 與 descriptor-off(因果)各一 ────────────────
def probe(net, n, mode, cfg, kill_resist, max_steps=200, seed0=40_000):
    net.eval(); wrong = ndmg = wins = 0; ahp, whp = [], []
    for gi in range(n):
        imm = FIRE if mode == "fixed" else (FIRE if gi % 2 == 0 else COLD)
        env = make_env(seed0 + gi, imm, cfg.hp, cfg.ac, cfg.level, cfg.opp_level,
                       enemy_fights=bool(cfg.enemy_fights),
                       enemy_regen=cfg.enemy_regen, agent_hp=cfg.agent_hp)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done = False; steps = 0
        while not done and steps < max_steps:
            steps += 1; actor = env.current_agent_id
            ob = build_obs(env.ws, actor, env.resources)
            if kill_resist:
                ob = zero_enemy_resist(ob)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            _mn = not bool(cfg.ablate)
            s = apply_resource_mask(s, env.resources, env.ws, actor, mask_immune_null=_mn)
            e = apply_entity_mask(e, ot, env.ws, actor, mask_immune_null=_mn)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            if actor == aid and act[0] > 0:
                sks = available_skills(env.ws.characters[actor], env.ws)
                if act[0] < len(sks) and sks[act[0]].features.expected_damage > 0:
                    o = env.ws.characters[oid]
                    dt = action_damage_type(
                        sks[act[0]].build_action(actor, oid,
                                                 (o.position.x, o.position.y)),
                        env.ws.characters[actor])
                    if dt:
                        ndmg += 1; wrong += (dt == imm)
            _, _, term, trunc, _ = env.step(act); done = term or trunc
        A = env.ws.characters[aid]; W = env.ws.characters[oid]
        wins += (not W.is_alive() and A.is_alive())
        ahp.append(max(0., A.hp) / max(1, A.max_hp))
        whp.append(max(0., W.hp) / max(1, W.max_hp))
    net.train()
    return dict(wrong=wrong / max(1, ndmg), wr=wins / n,
                ahp=float(np.mean(ahp)), whp=float(np.mean(whp)))


def _report(tag, mode, net, cfg):
    on = probe(net, cfg.eval_games, mode, cfg, kill_resist=False)
    off = probe(net, cfg.eval_games, mode, cfg, kill_resist=True)
    print(f"[{tag:<5}] desc-ON : wrong={on['wrong']:4.0%} WR={on['wr']:4.0%} "
          f"己血={on['ahp']:3.0%} 敵血={on['whp']:3.0%}  ‖  "
          f"desc-OFF(因果): wrong={off['wrong']:4.0%} WR={off['wr']:4.0%} "
          f"敵血={off['whp']:3.0%}", flush=True)


# ── 機制自檢:免疫真的把傷害歸零嗎?obs 真的是 -1 嗎? ──────────────────────
def smoke():
    register_kit()
    print("── smoke:驗免疫機制 + obs 編碼 + 雙武器可選 ──")
    env = make_env(123, FIRE, hp=24, ac=13)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    o = env.ws.characters[oid]
    print(f"敵人 HP={o.hp}/{o.max_hp} AC={o.ac} 免疫={FIRE} "
          f"(mult={o.damage_multipliers})  距離={env.ws.characters[aid].position.distance_to(o.position):.2f}m")

    sks = available_skills(env.ws.characters[aid], env.ws)
    wnames = [s.skill_id for s in sks]
    print(f"available_skills = {wnames}")
    fire_sk = next((s for s in sks if s.skill_id == f"weapon:{W_FIRE}"), None)
    cold_sk = next((s for s in sks if s.skill_id == f"weapon:{W_COLD}"), None)
    assert fire_sk is not None and cold_sk is not None, "兩把武器都該在 available_skills"

    # obs:敵人列 I_DESC_RESIST 火欄該=-1,冰欄該=0
    ob = build_obs(env.ws, aid, env.resources)
    ent = ob["entities"]
    er = ent[ent[:, I_ENT_ENEMY] > 0.5][0]
    from trpg.engine.damage import DAMAGE_TYPES
    fi = DAMAGE_TYPES.index(FIRE); ci = DAMAGE_TYPES.index(COLD)
    print(f"敵 obs resist[火]={er[I_DESC_RESIST + fi]:+.1f}  resist[冰]={er[I_DESC_RESIST + ci]:+.1f}  "
          f"(火該=-1 免疫, 冰該=0 中性)")
    assert abs(er[I_DESC_RESIST + fi] + 1.0) < 1e-5, "火欄該是 -1"
    assert abs(er[I_DESC_RESIST + ci]) < 1e-5, "冰欄該是 0"

    # RL 動作三元組 [skill_idx, entity_idx, grid];敵人在 slot ENEMY_SLOT_START
    fire_idx = wnames.index(f"weapon:{W_FIRE}")
    cold_idx = wnames.index(f"weapon:{W_COLD}")

    # 火(免疫)打過去 → HP 不變;冰打過去 → HP 掉
    hp0 = o.hp
    env.step([fire_idx, ENEMY_SLOT_START, 0])
    hp_after_fire = env.ws.characters[oid].hp
    print(f"火焰之刃(免疫)攻擊後 敵 HP: {hp0} → {hp_after_fire}  (該不變)")

    env2 = make_env(123, FIRE, hp=24, ac=13)
    aid2 = env2.agent_ids[0]; oid2 = env2.opp_ids[0]; o2 = env2.ws.characters[oid2]
    hp0b = o2.hp
    # 多打幾下(可能 miss),量是否曾造成傷害
    dropped = False
    for _ in range(8):
        if not env2.ws.characters[oid2].is_alive():
            break
        if env2.current_agent_id != aid2:
            env2.step([0, 0, 0]); continue
        hb = env2.ws.characters[oid2].hp
        env2.step([cold_idx, ENEMY_SLOT_START, 0])
        if env2.ws.characters[oid2].hp < hb:
            dropped = True
    print(f"冰霜之刃(非免疫)攻擊數次後 敵 HP: {hp0b} → {env2.ws.characters[oid2].hp}  "
          f"(該掉血, dropped={dropped})")

    assert abs(hp_after_fire - hp0) < 1e-9, "免疫型該 0 傷害!"
    assert dropped, "非免疫型該能造成傷害!"

    # 免手寫免疫的實驗網:codeword 版全網(非 NoArch),join 欄結構性移除,
    # 且 skill/entity 頭拿掉常數 h(對選招/選目標無效的死權重)
    net = CombatPolicyNet(hidden=128, n_head_groups=1, skill_combo_dim=8,
                          ablate_immunity_joins=True, drop_noop_h=True)
    assert net.ablate_immunity_joins is True and net.drop_noop_h is True
    assert net.skill_heads[0].weight.shape[1] == 160, "skill 頭該=160(拔 join+h)"
    assert net.entity_heads[0].weight.shape[1] == 96, "entity 頭該=96(拔 join+h)"
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, sl, ee, gg = net(ot)               # forward 跑得動
    print(f"CombatPolicyNet(codeword,ablate joins,drop-h) forward OK, "
          f"skill_head.in={net.skill_heads[0].weight.shape[1]} "
          f"entity_head.in={net.entity_heads[0].weight.shape[1]} "
          f"skill_logits shape={tuple(sl.shape)}")
    print("── smoke 全過:機制為真,可以開訓 ──")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--mode", choices=["fixed", "random"], default="random")
    p.add_argument("--out_dir", default="models/exp_immune")
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--opp_level", type=int, default=3)
    p.add_argument("--hp", type=int, default=24,
                   help="站樁還手敵人 HP(校準 24HP/己48/AC13/還手:全對→WR100%、"
                        "各半→WR50%、全錯→必死,correct%→WR 平滑單調)")
    p.add_argument("--ac", type=int, default=13)
    p.add_argument("--agent_hp", type=int, default=48,
                   help="agent HP(校準值:撐得過全對速殺、撐不過各半的雙倍挨打)")
    p.add_argument("--enemy_fights", type=int, default=1,
                   help="1=站樁但每回合還手(選錯型挨打=平滑代價);0=不還手")
    p.add_argument("--enemy_regen", type=int, default=0,
                   help="敵每回合回血(0=關;regen 會造成閾值式無梯度,已棄用)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablate", type=int, default=1,
                   help="1=拔掉 typed/cimmun join+免疫遮罩門(只能靠 sk_ent_ctx);"
                        "0=保留(對照組)")
    p.add_argument("--skill_combo_dim", type=int, default=8,
                   help="技能組合 codeword 維度(單一 kit 下恆定=不影響免疫訊號,"
                        "開著只為與 codeword 架構一致)")
    p.add_argument("--drop_noop_h", type=int, default=1,
                   help="1=skill/entity 頭拿掉常數 h(對選招/選目標為 no-op 的死權重;"
                        "grid 頭仍保留 h);0=保留")
    args = p.parse_args()

    if args.smoke:
        smoke(); return

    register_kit()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    # codeword 版全網(非 NoArch);ablate=結構性移除兩條手寫免疫 join;
    # drop_noop_h=skill/entity 頭拿掉常數 h(死權重)
    net = CombatPolicyNet(hidden=128, n_head_groups=1,
                          skill_combo_dim=args.skill_combo_dim,
                          ablate_immunity_joins=bool(args.ablate),
                          drop_noop_h=bool(args.drop_noop_h))
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    _en = f"還手{'+regen'+str(args.enemy_regen) if args.enemy_regen else ''}" \
        if args.enemy_fights else "不還手"
    print(f"fresh CombatPolicyNet(codeword,無手寫免疫) params={n_par} "
          f"combo_dim={args.skill_combo_dim} skill_head.in={net.skill_heads[0].weight.shape[1]} "
          f"ablate(joins+gates)={bool(args.ablate)} mode={args.mode} | "
          f"{KIT}(火/冰同傷害,己血{args.agent_hp}) vs 站樁{_en}{DUMMY} 敵HP{args.hp}/AC{args.ac} "
          f"免疫={'恆火' if args.mode=='fixed' else '每局隨機火/冰'} | open adjacent", flush=True)
    _report("u0", args.mode, net, args)

    for update in range(1, args.updates + 1):
        batch, neps, wrate = collect(net, args.steps, args.seed * 7919 + update,
                                     rng, args.mode, args)
        net.train()
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu", value_only=(update <= args.value_warmup))
        mean_r = float(batch["rewards"].sum() / max(1, neps))
        print(f"U{update:3d}/{args.updates}{' [wu]' if update <= args.value_warmup else ''} "
              f"eps={neps} wrong%={wrate:.2f} R/ep={mean_r:+.2f} "
              f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.2f} "
              f"ent={info['entropy']:.3f}", flush=True)
        if update % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"immune_u{update:04d}.pt")
            _report(f"u{update}", args.mode, net, args)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
