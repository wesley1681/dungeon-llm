"""v10 抗性全網格 lab：訓練分布設計成「讓捷徑輸」，唯一全格拿滿獎勵的策略
＝真 per-type EV 語義。純 PPO、無 oracle、凍結手術。

背景（probe_resist_curve + 權重考古，見 project_resist_shortcut）：
  v7 rsw 的課程只有「單根 ×0 釘在自己型上、永遠有替代」→ pooled 欄學到
  「有條目就消極」捷徑；且 ncol-1 漂移 bug 讓真 typed join 全程凍結。
本 lab 的網格維度（每一維殺一條已證捷徑）：
  幅度 0/0.25/0.5/2      → 0.5×照打才贏（殺「不讀值」）
  符號 含 2.0 易傷        → 易傷更該打（殺「不讀符號」）
  外型欄注入             → 亮著照打才贏（殺「外型→消極」）
  多欄疊加＋天然怪檔案    → 疊加校準（shadow/gargoyle/skeleton/zombie 進敵池）
  單型 kit×0.5×弱敵      → 打穿抗性是唯一贏法（殺「有亮就躲」）
  雙型 kit×一免一抗      → 挑 0.5 那把（值比較，不只切離免疫）
  1v2 一注入一乾淨       → 目標選擇（重校 entity join，v9 該欄 −0.413 反號）
  真職業（berserker/champion/devotion）→ join 量級對真招 logit 校準
公平性：單型 kit 永不注 ×0；多欄注入後若無有意義 EV（best_after <
max(2, 0.2×before)）逐步把自身型抬回 0.5 —— 不可贏局不進課程（1vN
game-impossible 教訓同族）。
凍結（resist-isolated，per net.tjoin_col 具名索引，不再 ncol-1）：
  entity_mlp[0] 的 13 個 typed-resist 輸入欄 + skill/entity head 的 typed
  join 欄 + critic；其餘位元級凍結 → 無抗性局為恆等式不變。
Hold-out（永不進訓練）：wight、ghoul（零樣本守門用）。

用法:
  python scripts/train_resist_grid.py --warm models/unified/_v10_reset.pt \
      --out_dir models/resist_grid --updates 80
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry mutation
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs, I_DESC_RESIST, N_DAMAGE_TYPES
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.damage import DAMAGE_TYPES
from trpg.engine.items import WEAPON_DEFS, Weapon
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, ClassDef, TraitGrant)
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from eval_routed import stable_seed
from seed_switch_bc import damaging_options
from train_switch_lab import (register_train_kits, KITS, SPELL_KITS,
                              _TRAIN_WEAPONS)

MONO_KITS = [
    ("ckit_mono_sw", ("長劍",),    ("斬擊",)),
    ("ckit_mono_cl", ("棍棒",),    ("鈍擊",)),
    ("ckit_mono_fr", ("火焰之刃",), ("火",)),
]
# run3：+施法者（spell→spell 切換＝miner immune_hit 主家族：mage_npc 火球 vs
# 火免疫不改丟 magic_missile）+聖武士系（sacred_weapon lift）；等級格含 20
#（dlvl:L20 lift ×9＝join 量級對 L20 高 logit 招外推不足）
REAL_IDENTS = ["berserker", "champion", "devotion", "vengeance", "life",
               "mage_npc", "evocation"]
# 困難格聚焦池（run4 u25 拆解：合成/life/mage_npc 已收斂、聖騎士家族 85%+ 未動）
HARD_FOCUS = ["devotion", "vengeance", "mage_npc"]
NAT_ENEMIES = ["shadow", "gargoyle", "skeleton", "zombie"]
HELDOUT = ["wight", "ghoul"]          # 永不進訓練（守門腳本用）
CLEAN_ENEMY = "orc"
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


def register_mono_kits():
    for aid, weapons, _t in MONO_KITS:
        cd = ClassDef(
            archetype_id=aid, default_name=aid, class_display="訓練合成",
            role="front",
            stat_block=dict(STR=16, DEX=14, CON=14, INT=10, WIS=10, CHA=10),
            hp_base=10, hp_per_level=6, ac=16,
            weapons=weapons, proficiencies=("STR", "CON"), skills=(),
            traits=(TraitGrant("extra_attack", min_level=5),))
        CLASS_DEFS[aid] = cd
        ARCHETYPE_FACTORIES[aid] = ARCHETYPE_FACTORIES.get(aid) or __import__(
            "trpg.scenarios.archetypes", fromlist=["_factory"])._factory(aid)
        ARCHETYPE_ROLES[aid] = cd.role


def _my_types(ws, aid, oids):
    ts = set()
    for o in oids:
        try:
            for t in damaging_options(ws, aid, o):
                ts.add(t[3])
        except Exception:
            pass
    return sorted(ts)


def _best_ev(ws, aid, oids):
    best = 0.0
    for o in oids:
        try:
            best = max(best, max((t[4] for t in damaging_options(ws, aid, o)),
                                 default=0.0))
        except Exception:
            pass
    return best


def apply_grid_mutation(env, rng, targets=None):
    """對 targets（預設全對手）注入網格突變；回傳 kind 字串。含公平性守門。"""
    ws = env.ws
    aid = env.agent_ids[0]
    oids = targets if targets is not None else list(env.opp_ids)
    my = _my_types(ws, aid, env.opp_ids)
    if not my:
        return "none"
    before = _best_ev(ws, aid, env.opp_ids)
    r = rng.random()
    prof = {}
    if r < 0.10:
        kind = "none"
    elif r < 0.45:
        kind = "my"
        dt = rng.choice(my)
        # ×0 加權（run1 診斷：切招格太稀→join 梯度事件不足、wrong0 卡 63%）
        m = rng.choice([0.0, 0.0, 0.25, 0.5, 2.0])
        if len(my) == 1 and m == 0.0:
            m = rng.choice([0.25, 0.5])       # 單型永不 ×0（不可贏局不進課程）
        prof[dt] = m
    elif r < 0.65:
        kind = "foreign"
        pool = [t for t in DAMAGE_TYPES if t not in my]
        # run3 診斷：單獨外型正欄（光耀=2.0）→消極 捷徑在訓練中期長出來，
        # 該構型出現率 ~0.2% 壓不住 → 2.0 加權 ×2＋提高單欄採樣比重
        for dt in rng.sample(pool, min(len(pool),
                                       rng.choice([1, 1, 2, 3]))):
            prof[dt] = rng.choice([0.0, 0.5, 2.0, 2.0])
    else:
        kind = "multi"
        for dt in rng.sample(list(DAMAGE_TYPES),
                             rng.randint(2, min(6, len(DAMAGE_TYPES)))):
            prof[dt] = rng.choice([0.0, 0.25, 0.5, 2.0])
    for o in oids:
        ws.characters[o].damage_multipliers.update(prof)
    # 公平性：自身型被壓到不可贏 → 逐步抬回 0.5
    for dt in my:
        if _best_ev(ws, aid, env.opp_ids) >= max(2.0, 0.2 * before):
            break
        for o in oids:
            dmm = ws.characters[o].damage_multipliers
            if dmm.get(dt, 1.0) < 0.5:
                dmm[dt] = 0.5
    if _best_ev(ws, aid, env.opp_ids) < max(2.0, 0.2 * before):
        for o in oids:                        # 仍不可贏（不該發生）→ 全撤
            for dt in prof:
                ws.characters[o].damage_multipliers.pop(dt, None)
        return "revert"
    return kind


ALL_IDENTS = ([k[0] for k in KITS] + [k[0] for k in SPELL_KITS]
              + [k[0] for k in MONO_KITS] + REAL_IDENTS)


def build_episode(rng, seed):
    """抽一個網格格子 → (env, cell_tag)。"""
    r = rng.random()
    if r < 0.30:                                   # 天然抗性怪（檔案原樣）
        ident = rng.choice(ALL_IDENTS)
        enemy = rng.choice(NAT_ENEMIES)
        # +L1/L3（run6 後 0087 診斷：L1 施法者對帶毒:0 條目的 ghoul 風箏不開火、
        # ablate typed_resist 翻轉＝載體因果、低等級不在格＝外推洞）
        lvl = rng.choice([1, 3, 6, 6, 10, 10, 20])
        env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
        env.reset(agent_archs=[ident], opp_archs=[enemy], level=lvl,
                  opp_level=MONSTER_DEFS[enemy].natural_level)
        return env, f"nat:{enemy}"
    if r < 0.42:                                   # 1v2 目標選擇（entity join）
        ident = rng.choice(ALL_IDENTS)
        env = CombatEnvV2(seed=seed, n_agents=1, n_opps=2)
        env.reset(agent_archs=[ident], opp_archs=[CLEAN_ENEMY, CLEAN_ENEMY],
                  level=rng.choice([6, 10, 20]), opp_level=4)
        ws = env.ws; aid = env.agent_ids[0]
        my = _my_types(ws, aid, env.opp_ids)
        if my:                                     # 只注入第一隻：另一隻=正解目標
            dt = rng.choice(my)
            ws.characters[env.opp_ids[0]].damage_multipliers[dt] = \
                rng.choice([0.0, 0.25, 0.5])
        return env, "sel:1v2"
    if r < 0.54:                                   # 困難格聚焦：主型×0 收斂最慢的
        ident = rng.choice(HARD_FOCUS)             # 身分（run4 u25 拆解：聖騎士
        lvl = rng.choice([6, 10, 20])              # wrong0 85%+，格子太稀=每
        env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)   # update 僅 ~1 次）
        # run5 u35 教訓：對等級敵浪費揮擊「還是會贏」＝獎勵差太小、PPO 平原
        # （vengeance 卡 43%、join 再長只蝕 through）→ 敵人加強 +0~2 級，
        # 讓浪費揮擊真的輸掉比賽＝捷徑在格子裡輸錢，而非只是次優
        env.reset(agent_archs=[ident], opp_archs=[CLEAN_ENEMY],
                  level=lvl, opp_level=min(lvl + rng.choice([0, 1, 2]), 12))
        ws = env.ws; aid = env.agent_ids[0]
        try:
            opts = damaging_options(ws, aid, env.opp_ids[0])
            if opts:
                dt = max(opts, key=lambda t: t[4])[3]
                for o in env.opp_ids:
                    ws.characters[o].damage_multipliers[dt] = \
                        rng.choice([0.0, 0.0, 0.5])
        except Exception:
            pass
        return env, "inj:hard"
    ident = rng.choice(ALL_IDENTS)                 # orc + 網格注入
    lvl = rng.choice([1, 3, 6, 6, 10, 10, 20])
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[ident], opp_archs=[CLEAN_ENEMY],
              level=lvl, opp_level=min(lvl, 8))
    kind = apply_grid_mutation(env, rng)
    return env, f"inj:{kind}"


def collect(net, n_steps, seed, rng, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    cells = {}
    while len(rew_l) < n_steps:
        env, tag = build_episode(rng, seed * 1_000_003 + ep)
        cells[tag] = cells.get(tag, 0) + 1
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            obs2, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); lp_l.append(lp.numpy())
            rew_l.append(float(r)); val_l.append(val)
            skm_l.append(skm.numpy()); enm_l.append(enm.numpy())
            grm_l.append(grm.numpy())
            done = term or trunc
            done_l.append(bool(done))
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
    rewards = np.array(rew_l, np.float32); values = np.array(val_l, np.float32)
    dones = np.array(done_l, np.float32)
    adv, ret = _compute_gae(rewards, values, dones, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]},
        "actions": np.array(act_l, np.int64),
        "log_probs": np.array(lp_l, np.float32),
        "skill_masks": np.stack(skm_l).astype(np.bool_),
        "entity_masks": np.stack(enm_l).astype(np.bool_),
        "grid_masks": np.stack(grm_l).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, ep, cells


# ── 訓練中 battery（greedy、固定種子）──────────────────────────────────────────

def _greedy_game(net, ident, enemy, lvl, olvl, profile, key, n_opps=1,
                 max_steps=400):
    """回 (win, atk_n, wrong0_n, seq)。wrong0=砸進 ×0 型且當回合有非 0 替代。"""
    random.seed(stable_seed(key))
    env = CombatEnvV2(seed=stable_seed(key) ^ 0x5A5A5A, n_agents=1,
                      n_opps=n_opps)
    env.reset(agent_archs=[ident], opp_archs=[enemy] * n_opps, level=lvl,
              opp_level=olvl)
    aid = env.agent_ids[0]
    if profile:
        for o in env.opp_ids:
            env.ws.characters[o].damage_multipliers.update(profile)
    done, steps, atk, wrong0, seq = False, 0, 0, 0, []
    while not done and steps < max_steps:
        steps += 1
        actor = env.current_agent_id
        if actor != aid:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        obs = build_obs(env.ws, actor, env.resources)
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                               agent_id=actor))
        sks = available_skills(env.ws.characters[actor], env.ws)
        sk = sks[act[0]] if 0 < act[0] < len(sks) else None
        seq.append(sk.skill_id if sk is not None else "END")
        if (sk is not None and getattr(sk.features, "expected_damage", 0) > 0
                and sk.features.target_type in _OFFENSE_TT):
            atk += 1
            try:
                opts = []
                for o in env.opp_ids:
                    if env.ws.characters[o].is_alive():
                        opts += damaging_options(env.ws, actor, o)
                mine = [t for t in opts if t[0] == act[0]]
                if (mine and max((t[4] for t in mine), default=0.0) <= 0.05
                        and max((t[4] for t in opts), default=0.0) >= 2.0):
                    wrong0 += 1
            except Exception:
                pass
        _, _, term, trunc, _ = env.step(act)
        done = term or trunc
    win = all(not env.ws.characters[o].is_alive() for o in env.opp_ids) \
        and env.ws.characters[aid].is_alive()
    return win, atk, wrong0, seq


def battery(net, games, tag):
    rows, score = [], 0.0
    pooled = float(net.entity_mlp[0].weight[
        :, I_DESC_RESIST:I_DESC_RESIST + N_DAMAGE_TYPES].norm())
    tj = net.tjoin_col
    sj = float(np.mean([h.weight[0, tj].item() for h in net.skill_heads]))
    ejw = float(np.mean([h.weight[0, tj].item() for h in net.entity_heads]))
    rows.append(f"  [{tag}] pooled={pooled:.3f} skill-tjoin={sj:+.3f} "
                f"entity-tjoin={ejw:+.3f}")
    # 1) 切招保持：雙型 kit、自身主型 ×0 → 不砸 0、贏。
    #    run3 加真職業案例：mage_npc 火免疫（spell→spell）＋devotion L20 斬免疫
    #   （L20 高 logit 外推）＝miner immune_hit 兩個子家族的直接量尺
    sw_w = sw_g = sw_wrong = sw_atk = 0
    for kit, dt, lv, ol in (("ckit_sw_fr", "斬擊", 6, 6),
                            ("ckit_zap_cl", "閃電", 6, 6),
                            ("mage_npc", "火", 6, 6),
                            ("devotion", "斬擊", 20, 8),
                            ("vengeance", "斬擊", 6, 6),
                            ("life", "光耀", 10, 6)):
        for gi in range(games):
            w, a, w0, _ = _greedy_game(net, kit, "orc", lv, ol, {dt: 0.0},
                                       f"bat_sw|{kit}|{gi}")
            sw_w += w; sw_g += 1; sw_wrong += w0; sw_atk += a
    rows.append(f"    switch : WR={sw_w/sw_g:3.0%} wrong0={sw_wrong}/"
                f"{max(1,sw_atk)}")
    score += (sw_w / sw_g) - 2.0 * (sw_wrong / max(1, sw_atk))
    # 2) 打穿抗性：單型×0.5×弱敵 + berserker vs shadow 天然
    th_w = th_g = th_atk = 0
    for ident, enemy, lvl, olvl, prof in (
            ("ckit_mono_sw", "orc", 6, 4, {"斬擊": 0.5}),
            ("berserker", "shadow", 10, 1, None),
            ("champion", "gargoyle", 8, 2, None),
            # 單獨外型易傷欄（run3 中期塌縮格）：物理 kit 面對光耀+2 必須照打
            ("ckit_mono_sw", "orc", 6, 6, {"光耀": 2.0}),
            ("berserker", "orc", 6, 6, {"光耀": 2.0, "冰": 2.0}),
            # 0087 家族：L1 施法者 vs 帶毒:0 條目的 ghoul——量接戰（有攻擊）
            ("evocation", "ghoul", 1, 1, None)):
        for gi in range(games):
            w, a, _, _ = _greedy_game(net, ident, enemy, lvl, olvl, prof,
                                      f"bat_th|{ident}|{gi}")
            th_w += w; th_g += 1; th_atk += (a > 0)
    rows.append(f"    through: WR={th_w/th_g:3.0%} 有攻擊={th_atk}/{th_g}")
    score += 1.5 * (th_w / th_g) + 0.5 * (th_atk / th_g)
    # 3) 外型對照：外型條目在場，行為仍須「攻擊＋贏」（序列一致只當資訊印出——
    #    位元級一致非獎勵相關，會自然漂移；守門標準是不消極）
    same = tot = fo_w = fo_atk = 0
    for gi in range(games):
        w1, a1, _, s1 = _greedy_game(net, "ckit_mono_sw", "orc", 6, 6,
                                     {"火": 0.5, "冰": 0.5, "毒": 0.0},
                                     f"bat_fo|{gi}")
        _, _, _, s0 = _greedy_game(net, "ckit_mono_sw", "orc", 6, 6, None,
                                   f"bat_fo|{gi}")
        same += (s1 == s0); tot += 1
        fo_w += w1; fo_atk += (a1 > 0)
    rows.append(f"    foreign: WR={fo_w/tot:3.0%} 有攻擊={fo_atk}/{tot} "
                f"(序列一致={same}/{tot} 僅資訊)")
    score += fo_w / tot + 0.5 * (fo_atk / tot)
    # 4) 值/符號語義由 probe_resist_curve 守門（A/B 掃描+--decomp），battery 不重複量
    print("\n".join(rows), flush=True)
    return score


def freeze_except_resist(net):
    rs, re_ = I_DESC_RESIST, I_DESC_RESIST + N_DAMAGE_TYPES
    for p_ in net.parameters():
        p_.requires_grad_(False)
    for p_ in net.critic.parameters():
        p_.requires_grad_(True)

    def _cols_only(lo, hi):
        def hook(g):
            gg = torch.zeros_like(g); gg[:, lo:hi] = g[:, lo:hi]; return gg
        return hook

    ew = net.entity_mlp[0].weight
    ew.requires_grad_(True)
    ew.register_hook(_cols_only(rs, re_))
    tj = net.tjoin_col
    for h in net.skill_heads:
        h.weight.requires_grad_(True)
        h.weight.register_hook(_cols_only(tj, tj + 1))
    for h in net.entity_heads:
        h.weight.requires_grad_(True)
        h.weight.register_hook(_cols_only(tj, tj + 1))
    n_tr = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
    print(f"[freeze] trainable={n_tr}（pooled 欄 {rs}:{re_} + skill/entity "
          f"tjoin@{tj} + critic；具名索引、不再 ncol-1）", flush=True)


def join_feature_sanity(net):
    """typed join 特徵必須非零流動（v7 的 cimmun 欄零特徵教訓）。"""
    env = CombatEnvV2(seed=7, n_agents=1, n_opps=1)
    env.reset(agent_archs=["berserker"], opp_archs=["shadow"], level=10,
              opp_level=1)
    ob = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                   env.resources))
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    tj = net.tjoin_col
    with torch.no_grad():
        _, s0, _, _ = net(ot)
        old = [h.weight[:, tj].clone() for h in net.skill_heads]
        for h in net.skill_heads:
            h.weight[:, tj] += 5.0
        _, s1, _, _ = net(ot)
        for h, o in zip(net.skill_heads, old):
            h.weight[:, tj] = o
    moved = float((s1 - s0).abs().max())
    print(f"[sanity] typed-join 特徵流動 |Δlogit|max={moved:.4f} "
          f"{'OK' if moved > 1e-6 else '** 特徵死線，中止 **'}", flush=True)
    if moved <= 1e-6:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/_v10_reset.pt")
    p.add_argument("--out_dir", default="models/resist_grid")
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--join_lr", type=float, default=5e-3,
                   help="skill/entity head join 欄的獨立學習率（純量載體）")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.02)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("v10 resist-grid pure-PPO (blind)\n")
    register_monsters()
    register_train_kits()
    register_mono_kits()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = load_student(args.warm)
    join_feature_sanity(net)
    freeze_except_resist(net)
    print(f"warm={args.warm} idents={len(ALL_IDENTS)} nat={NAT_ENEMIES} "
          f"heldout={HELDOUT}", flush=True)

    # 參數組學習率（run1 診斷：uniform 1e-4 下 join 純量 20 updates 只長 +0.07、
    # wrong0 不動——純量載體要能長到翻高 logit 的量級，需要獨立高 lr）
    head_ws = ([h.weight for h in net.skill_heads]
               + [h.weight for h in net.entity_heads])
    head_ids = {id(w) for w in head_ws}
    optim = torch.optim.Adam([
        {"params": head_ws, "lr": args.join_lr},
        {"params": [p_ for p_ in net.parameters()
                    if p_.requires_grad and id(p_) not in head_ids],
         "lr": args.lr},
    ])
    print("baseline:", flush=True)
    best = battery(net, args.eval_games, "u0")
    for update in range(1, args.updates + 1):
        batch, neps, cells = collect(net, args.steps,
                                     args.seed * 7919 + update, rng)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        print(f"U{update:3d}/{args.updates}"
              f"{' [wu]' if update <= args.value_warmup else ''} eps={neps} "
              f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.2f} "
              f"ent={info['entropy']:.3f} cells={dict(sorted(cells.items()))}",
              flush=True)
        if update % args.eval_every == 0:
            net.eval()
            torch.save(net.state_dict(), out_dir / f"rg_u{update:04d}.pt")
            battery(net, args.eval_games, f"u{update}")
            net.train()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
