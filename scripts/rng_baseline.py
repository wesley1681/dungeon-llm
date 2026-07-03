"""隨機對照：回答「模型是不是演算法套神經網路殼？亂數選招是不是也打得不錯？」

三臂，全部走【完全相同】的守門管線（apply_resource_mask/apply_entity_mask/
pick_action），唯一差別在餵給守門的 logits 來源：

  A model   ： net 吐 logits（真模型）
  B rng     ： 純隨機噪音 logits（= 守門演算法 + 亂數生成器；el 強制不主動結束＝
               給隨機最好發揮空間，只在資源耗盡被迫結束回合）
  C rng_end ： 同 B 但 end 也隨機（更「純」的亂選，會自己浪費回合）

若 B ≈ A → 守門在扛、網路是裝飾（用戶假設成立）。
若 A ≫ B → 守門只是地板，勝負由網路學到的 logits 決定。

對手席永遠是腳本專家 OPP_PANEL（4 個），成對種子（同 key→同引擎骰）。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_gate as EG  # noqa: E402
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action  # noqa: E402
from train_population import blind_np_single  # noqa: E402


def _rng_driver(net, random_end: bool):
    """走 _model_driver 完全相同的管線，但把 logits 換成 randn。

    仍呼叫 net 只為了取得【張量形狀】(架構固定)，其輸出值被 randn_like 徹底覆蓋＝
    對決策零影響。end：random_end=False → 強制 -50(≈永不主動結束，給隨機最好發揮)。"""
    def drv(env, obs, actor):
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        el = torch.randn_like(el) if random_end else torch.full_like(el, -50.0)
        s = torch.randn_like(s)
        e = torch.randn_like(e)
        g = torch.randn_like(g)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    return drv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--games", type=int, default=4, help="每職每對手局數")
    ap.add_argument("--arms", default="model,rng,rng_end")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(0)
    net = EG.load_student(args.model)

    ARMS = {
        "model":   EG._model_driver(net),
        "rng":     _rng_driver(net, random_end=False),
        "rng_end": _rng_driver(net, random_end=True),
    }
    arms = [a for a in args.arms.split(",") if a in ARMS]
    G = args.games
    n_cell = G * len(EG.OPP_PANEL)
    sig = EG._sigma_frac(n_cell)

    print(f"model={args.model}  對手={EG.OPP_PANEL}  G={G} "
          f"(n/職={n_cell}, σ≈{sig:.0%})  L5v5")
    print(f"守門管線三臂皆相同，只換 logits 來源。臂: {arms}\n")

    hdr = f"{'職業':16s}"
    for a in arms:
        hdr += f" {a+'WR':>9s} {a+'DPR':>8s}"
    print(hdr)

    tot = {a: {"wr": 0.0, "dpr": 0.0} for a in arms}
    for arch in EG.CLASSES:
        line = f"{arch:16s}"
        row = {}
        for a in arms:
            r = EG._agg_wr(ARMS[a], arch, EG.OPP_PANEL, G)
            row[a] = r
            tot[a]["wr"] += r["wr"]; tot[a]["dpr"] += r["dpr"]
            line += f" {r['wr']:9.0%} {r['dpr']:8.2f}"
        print(line)
    n = len(EG.CLASSES)
    avg = f"{'—— 平均 ——':16s}"
    for a in arms:
        avg += f" {tot[a]['wr']/n:9.0%} {tot[a]['dpr']/n:8.2f}"
    print("\n" + avg)

    if "model" in arms and "rng" in arms:
        dwr = (tot["model"]["wr"] - tot["rng"]["wr"]) / n
        ddpr = (tot["model"]["dpr"] - tot["rng"]["dpr"]) / n
        print(f"\nΔ(model−rng): WR {dwr:+.0%}  DPR {ddpr:+.2f}   "
              f"(2σ≈{2*sig:.0%})")
        if dwr > 2 * sig:
            print("→ 網路 logits 顯著優於「守門＋亂數」：網路在做事，守門只是地板。")
        else:
            print("→ 兩臂無顯著差：守門可能在扛，需檢視。")


if __name__ == "__main__":
    main()
