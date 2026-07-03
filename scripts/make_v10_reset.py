"""v10 Step 0：把 uni_v9 的「pooled 抗性欄」歸零（entity_mlp[0] 的 13 個
typed-resist 輸入欄），存 _v10_reset.pt。

依據（probe_resist_curve + 版本鏈指紋 + 權重考古）：
  - v7 rsw 手術的毒**只在** pooled 欄（norm 2.1→7.5；v7=v9 同值）；
  - typed join（skill/entity head 第 net.tjoin_col 欄）v7 全程凍結在 v3 的
    +0.314（rsw 的 ncol-1 漂移 bug 訓到 cimmun 欄）——符號正確、量級小，
    是 DAgger 底座學來的健全語義 → **保留**，v10 網格 lab 把它練大；
  - cimmun 欄本來就是 0.000，不動。

數學保證：這些權重的輸入在對手無抗性條目時恆 0（obs 編碼=倍率−1、空表=全 0）
→ 無抗性局行為位元級不變。本腳本仍實測驗證，不只靠推導：
  V1) 職業對戰（champion vs battle_master, 5 場同種子）v9/reset 逐決策 logits
      位元級相同；
  V2) 對天然抗性怪（shadow）reset 行為 = v9 的 empty-profile 行為（抗性盲基線）。

用法: python scripts/make_v10_reset.py [--src models/unified/uni_v9.pt]
      [--out models/unified/_v10_reset.pt]
"""
from __future__ import annotations
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import I_DESC_RESIST, N_DAMAGE_TYPES
from distill_routed import load_student
from eval_routed import stable_seed
from train_population import blind_np_single


def zero_resist_carrier(net):
    """只清 pooled 毒欄；typed join（net.tjoin_col）與 cimmun 欄(-1)不動。"""
    rs, re_ = I_DESC_RESIST, I_DESC_RESIST + N_DAMAGE_TYPES
    with torch.no_grad():
        net.entity_mlp[0].weight[:, rs:re_].zero_()
    return rs, re_


def _greedy_episode(net, agent, opp, key, n_dec=40, record_logits=False):
    """同種子跑一場，回 (動作序列, logits 串接)。"""
    random.seed(stable_seed(key))
    env = CombatEnvV2(seed=stable_seed(key) ^ 0x5A5A5A, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent], opp_archs=[opp], level=6, opp_level=6)
    aid = env.agent_ids[0]
    obs, acts, logs = env._last_obs if hasattr(env, "_last_obs") else None, [], []
    obs = None
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[opp],
                       level=6, opp_level=6)
    done, steps = False, 0
    while not done and len(acts) < n_dec and steps < 400:
        steps += 1
        actor = env.current_agent_id
        if actor != aid:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        if record_logits:
            logs.append(torch.cat([s.flatten(), e.flatten(), g.flatten()]))
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        acts.append(tuple(act))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return acts, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/unified/uni_v9.pt")
    ap.add_argument("--out", default="models/unified/_v10_reset.pt")
    args = ap.parse_args()

    net9 = load_student(args.src); net9.eval()
    netR = load_student(args.src); netR.eval()
    rs, re_ = zero_resist_carrier(netR)
    tj = netR.tjoin_col
    sj = float(np.mean([h.weight[0, tj].item() for h in netR.skill_heads]))
    ej = float(np.mean([h.weight[0, tj].item() for h in netR.entity_heads]))
    pn = float(net9.entity_mlp[0].weight[:, rs:re_].norm())
    print(f"歸零: entity_mlp[0][:, {rs}:{re_}]（v9 pooled norm 原為 {pn:.3f}）")
    print(f"保留: typed join @col{tj}  skill={sj:+.3f} entity={ej:+.3f}"
          f"（v3 健全語義底、v10 lab 續訓）")

    # V1: 無抗性局位元級等價
    print("\nV1 無抗性局（champion vs battle_master ×5、同種子逐決策 logits）:")
    ok = True
    for gi in range(5):
        a9, l9 = _greedy_episode(net9, "champion", "battle_master",
                                 f"v10rst|{gi}", record_logits=True)
        aR, lR = _greedy_episode(netR, "champion", "battle_master",
                                 f"v10rst|{gi}", record_logits=True)
        same_act = a9 == aR
        same_log = (len(l9) == len(lR)
                    and all(torch.equal(x, y) for x, y in zip(l9, lR)))
        ok &= same_act and same_log
        print(f"  game{gi}: 決策數={len(a9)} 動作相同={same_act} "
              f"logits位元級相同={same_log}")
    print(f"  => {'PASS 位元級等價' if ok else '** FAIL —— 不可繼續 **'}")
    if not ok:
        sys.exit(1)

    # V2: 對天然抗性怪 = 抗性盲基線（v9 empty-profile 行為）
    print("\nV2 對 shadow（天然多欄抗性、L10 vs L1）:")
    for ident in ("totem_bear", "berserker"):
        try:
            from chimera_defs import register_chimeras
            register_chimeras()
        except Exception:
            pass
        random.seed(stable_seed(f"v10rst2|{ident}"))
        env = CombatEnvV2(seed=stable_seed(f"v10rst2|{ident}") ^ 0x5A5A5A,
                          n_agents=1, n_opps=1)
        obs, _ = env.reset(agent_archs=[ident], opp_archs=["shadow"],
                           level=10, opp_level=1)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done, atk, steps = False, 0, 0
        from trpg.engine.skill import available_skills, TargetType
        while not done and steps < 400:
            steps += 1
            actor = env.current_agent_id
            if actor != aid:
                obs, _, term, trunc, _ = env.step([0, 0, 0])
                done = term or trunc
                continue
            ob = blind_np_single(obs)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = netR(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))
            sks = available_skills(env.ws.characters[actor], env.ws)
            sk = sks[act[0]] if 0 < act[0] < len(sks) else None
            if (sk is not None and getattr(sk.features, "expected_damage", 0) > 0
                    and sk.features.target_type in (
                        TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                        TargetType.POINT, TargetType.LINE, TargetType.CONE)):
                atk += 1
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        win = not env.ws.characters[oid].is_alive()
        print(f"  {ident:12s} 攻擊數={atk} win={int(win)} "
              f"{'PASS(攻擊獲勝=抗性盲基線)' if atk > 0 and win else '** 檢查 **'}")

    torch.save(netR.state_dict(), args.out)
    print(f"\n已存 {args.out}")


if __name__ == "__main__":
    main()
