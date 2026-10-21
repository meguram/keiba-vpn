"""tp-63（アンサンブルの幅出し・メンバー別/条件別のチューニング）の結果を docs の 1 章（ensemble_diversity.html）にする。

python -m src.scripts.data.tp63_report --dir <結果ディレクトリ（ev/tp63_results.json, ev/members/*.json, tune/sweep_*.json, summary.html, conclusion.html があるところ）>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
TEXT = Path(__file__).resolve().parent / "tp63_text"      # 手書きの要点・結論（結果の数値は章に直接書く）
CHAPTER = ROOT / "docs/html/modeling/chapters/ensemble_diversity.html"
MEMBERS = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
KIND_JP = {
    "lgb_extras_ped": "LightGBM 二値（血統つき）", "xgb_extras_ped": "XGBoost 二値（血統つき）", "lgb_extras": "LightGBM 二値", "lgb_softmax": "LightGBM レース内 softmax（1 着のみ）",
    "linear": "線形（隠れ層なしの NN）", "mtl_nn_extras_ped": "補助損失つき NN（血統つき）", "mtl_nn_s": "補助損失つき NN（小）",
    "tree5": "単一の決定木（深さ 5）", "bagtree4": "浅い決定木の袋詰め（深さ 4、150 本）", "rf_shallow_cal": "ランダムフォレスト（浅い: 深さ 8、200 本、較正つき）",
    "rf_deep_cal": "ランダムフォレスト（深い: 深さ 14、120 本、較正つき）", "et_deep": "ExtraTrees（深さ 16、150 本、較正つき）", "rf_full": "ランダムフォレスト（深さ無制限）",
    "lgb_xendcg": "LightGBM ランキング（rank_xendcg）", "lgb_lambdarank": "LightGBM ランキング（lambdarank）", "lgb_surface": "LightGBM 芝・ダート別の専門モデル",
    "lgb_recent3": "LightGBM 直近 3 年だけで学習", "lgb_half2": "LightGBM 半減期 2 年の重み", "lgb_dart": "LightGBM dart", "logreg_l2": "ロジスティック回帰（L2）",
    "lgb_split_dist": "LightGBM 距離帯別の専門モデル", "lgb_split_going": "LightGBM 馬場別の専門モデル", "lgb_split_class": "LightGBM クラス帯別の専門モデル",
    "lgb_q_pace": "LightGBM レース質（ペース段階）別の専門モデル（事前の確率で混合）", "lgb_q_pace_oracle": "同（実測の段階で選ぶ。オラクル＝上限、実運用では使えない）",
}


DONE_TABLE = """<h3>これまでに試していた範囲（再実行はしない。引用）</h3>
<table><tr><th>種類</th><th>どこで</th><th>結果（勝率 LogLoss の差。負 = 改善）</th></tr>
<tr><td>CatBoost・LightGBM extra_trees・深い XGBoost・TabM 風 NN・ノイズ付き NN・DART</td><td>tp-05n（§19）</td><td>単体は v2 を超えない。DART は +0.0096 と悪化。基準との logit 相関 0.95〜0.999</td></tr>
<tr><td>クラス・コース・競馬場・競馬場×路面ごとの専用モデル</td><td>tp-05j・05k（§19）</td><td>専用だけは全て悪化（+0.0026〜+0.0049）。全体との重み付き混合も、全体の seed 違い平均（対照）と同じで不採用</td></tr>
<tr><td>lambdarank・rank_xendcg の確率化</td><td>tp-05l（§19）</td><td>単体は二値に及ばない。v2 と logit 合成で勝率 −0.00018（効果 &lt; 0.0002）</td></tr>
<tr><td>多層スタッキング・L3（現行 + クロス LR 等）</td><td>tp-05m（§19）、tp-51（§33）</td><td>L3 だけが基準の境界（V5: LOYO 勝率 −0.00024、前進 −0.00017）</td></tr>
<tr><td>CatBoost GPU・extra_trees・全列 XGBoost・seed 3→6・特徴量グルーピング型・メタ GBM/NN・較正</td><td>tp-51（§33）</td><td>V4 の 7 メンバーに足して ±0.00003（seed 6 本は −0.00016 で効果量未満）</td></tr>
<tr><td>DCN-v2・FM・DeepFM・素の MLP・残差ブースティング</td><td>tp-58（§44）</td><td>L3 の上で −0.00014 で基準未満</td></tr>
<tr><td>ハイパーパラメータ調整</td><td>tp-05d（§8）、NN の再調整 tp-05i</td><td>Optuna（LightGBM 40 試行・XGBoost 30 試行・NN 20 試行）。<b>220 列</b>の特徴量で、valid 2022・2023 で調整、2024 は確認。NN は 30 試行で再調整して −0.00002（不採用）。LightGBM・XGBoost は <b>1,142 列では再調整していない</b></td></tr>
<tr><td><b>未実施</b>: sklearn のランダムフォレスト・浅い決定木の袋詰め・単一の決定木、レース質の特徴を持つ学習器、レース質別の専門モデル、V5 の 1,142 列での感度と条件別の最適設定</td><td>本章（tp-63）</td><td>以下</td></tr></table>"""

HISTORY = """<p>V5 の LightGBM 系メンバーの設定は tp-05d の Optuna（TPE、LightGBM 40 試行。学習率 0.033・葉 17・min_data 69・feature_fraction 0.83・bagging 0.78・lambda_l2 99.5・789 ラウンド）。
評価は valid 2022・2023 の勝率・連対・複勝の合算（2024 は確認専用）。<b>この調整は 220 列の特徴量のとき</b>のもので、V5 の 1,142 列（5 倍強）では、列が増えたぶん最適な feature_fraction・正則化・木数が動いているかもしれない。
下の感度は、その点を V5 の入力で確かめる（学習率だけ 3 倍にした簡易設定。木数は学習中に 400 本までの途中経過で評価）。</p>"""


def _f(x, nd=5, sign=True):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _cell(c: dict | None) -> str:
    if not c:
        return "-"
    ok = "<b>採用</b>" if c.get("採用") else "不採用"
    return f"{_f(c['Δwin'])}（上限 {_f(c['win_hi'])}） / {_f(c['Δtop2'])} / {_f(c['Δtop3'])} / {ok}"


def inventory_tables(res: dict) -> str:
    inv, inv2 = res.get("inventory"), res.get("inventory2")
    if not inv:
        return "<p>（未実施）</p>"
    rows = []
    w = inv["weights_all_years"]
    for i, m in enumerate(MEMBERS):
        d = inv["drop_one"][m]
        rows.append([f"<code>{m}</code>", html.escape(KIND_JP[m]), _f(inv["members"][m]["win"], 5, False), _f(inv["members"][m]["top2"], 5, False), _f(inv["members"][m]["top3"], 5, False),
                     _f(w["win"][i], 3, False), f"{_f(d['Δwin'])}（{_f(d['win_hi'])}） / {_f(d['Δtop2'])} / {_f(d['Δtop3'])}"])
    t1 = r29._tbl(["メンバー", "種類", "勝率 LL", "連対 LL", "複勝 LL", "勝率の重み", "このメンバーを外したときの劣化（Δ勝率（上限） / Δ連対 / Δ複勝、LOYO）"], rows, ["s", "s", "n", "n", "n", "n", "n"])
    lc = inv["logit_corr_win"]
    rows = [[f"<code>{MEMBERS[i]}</code>"] + [f"{lc[i][j]:.3f}" for j in range(len(MEMBERS))] for i in range(len(MEMBERS))]
    t2 = r29._tbl(["勝率の logit 相関"] + [m.replace("_extras", "").replace("mtl_", "") for m in MEMBERS], rows, ["s"] + ["n"] * len(MEMBERS))
    t3 = ""
    if inv2:
        rows = [[t, f"{inv2[t]['dev_corr_mean']:+.3f}", " / ".join(f"{x * 100:.0f}%" for x in inv2[t]["pc_var_share"][:4]), f"{inv2[t]['participation_ratio']:.2f}"] for t in ("win", "top2", "top3")]
        t3 = r29._tbl(["対象", "『平均 logit からのずれ』の相関の平均", "ずれの主成分の分散比（第 1〜4）", "実効的な次元数"], rows, ["s", "n", "n", "n"])
    return t1 + "<h3>メンバー間の相関</h3>" + t2 + ("<h3>平均からのずれで見た多様性</h3>" + t3 if t3 else "")


def members_table(res: dict, minfo: dict) -> str:
    rows = []
    for k, v in res.items():
        parts = k.split("|")
        if len(parts) != 3 or parts[1] != "P" or parts[2] != "loyo":
            continue
        n = parts[0]
        if n.startswith("combo_"):
            continue
        sa = res.get("standalone", {}).get(n, {})
        l3 = res.get(f"{n}|L3win2|loyo", {})
        fw = res.get(f"{n}|P|forward", {})
        fl = res.get(f"{n}|L3win2|forward", {})
        info = minfo.get(n, {})
        size = info.get("model_mb_per_fold")
        rows.append([f"<code>{n}</code>", html.escape(KIND_JP.get(n, n)), _f(sa.get("ll_win"), 5, False), _f(sa.get("ll_top3"), 5, False), f"{sa.get('resid_corr_composite_win', float('nan')):.3f}",
                     _cell(v.get("vs_S0")), _cell(fw.get("vs_S0")), _cell(l3.get("vs_S0")), _cell(l3.get("vs_L3")), _cell(fl.get("vs_L3")),
                     (f"{max(size):.1f} MB/fold" if size else "-")])
    return r29._tbl(["追加メンバー", "種類", "単独 勝率 LL", "単独 複勝 LL", "現行合成との残差相関(勝率)", "①現行合成に足す LOYO", "①前進", "②L3 型に足す（V5 の現行合成比）LOYO", "②L3 型比 LOYO", "②L3 型比 前進", "モデルの大きさ"],
                    rows, ["s", "s", "n", "n", "n", "n", "n", "n", "n", "n", "n"]) if rows else "<p>（未実施）</p>"


def combos_table(res: dict) -> str:
    rows = []
    for k, v in res.items():
        parts = k.split("|")
        if len(parts) == 3 and parts[0].startswith("combo_") and parts[2] == "loyo":
            fw = res.get(f"{parts[0]}|{parts[1]}|forward", {})
            rows.append([f"<code>{parts[0]}</code>", "、".join(v.get("members", [])), parts[1], _cell(v.get("vs_S0")), _cell(v.get("vs_L3")), _cell(fw.get("vs_L3"))])
    return r29._tbl(["組", "メンバー", "合成", "現行合成比 LOYO", "L3 型比 LOYO", "L3 型比 前進"], rows, ["s", "s", "s", "n", "n", "n"]) if rows else "<p>（未実施）</p>"


def sweep_tables(sw: dict) -> str:
    if not sw:
        return "<p>（未実施）</p>"
    cfgs = []
    for k in sw:
        c = k.split("|")[0]
        if c not in cfgs:
            cfgs.append(c)
    years = sorted({k.split("|")[1] for k in sw})
    tgt = "win"
    rows = []
    for c in cfgs:
        cells = []
        tot = []
        for y in years:
            r = sw.get(f"{c}|{y}|{tgt}")
            if not r:
                cells.append("-")
                continue
            ks = [x for x in r if x != "seconds"]
            kb = min(ks, key=lambda x: r[x]["all"])
            cells.append(f"{r[kb]['all']:.5f} @{kb}")
            tot.append(r[kb]["all"])
        rows.append([f"<code>{c}</code>"] + cells + [f"{sum(tot) / len(tot):.5f}" if tot else "-"])
    return r29._tbl(["設定（学習率 3 倍の簡易設定、他は V5 の LightGBM と同じ）"] + [f"{y} の最良（LL @木数）" for y in years] + ["平均"], rows, ["s"] + ["n"] * (len(years) + 1))


def cond_sweep_table(sw: dict) -> str:
    """条件ごとに、最良の設定と、全体で最良の設定との差（条件別に最適化したときの見込みの利得）。"""
    if not sw:
        return ""
    cfgs, keys = [], set()
    for k, v in sw.items():
        c, y, t = k.split("|")
        if t != "win":
            continue
        if c not in cfgs:
            cfgs.append(c)
        for kk in v:
            if kk != "seconds":
                keys.update(v[kk].keys())
    conds = sorted(x for x in keys if "=" in x)
    years = sorted({k.split("|")[1] for k in sw})

    def ll(c, cond):
        vals = []
        for y in years:
            r = sw.get(f"{c}|{y}|win")
            if not r:
                return None
            ks = [x for x in r if x != "seconds"]
            vals.append(min(r[x].get(cond, float("inf")) for x in ks))        # 条件ごとに最良の木数を選ぶ（早期停止を条件別にした場合）
        return sum(vals) / len(vals)

    def ll_fixed(c, cond):
        vals = []
        for y in years:
            r = sw.get(f"{c}|{y}|win")
            if not r:
                return None
            ks = [x for x in r if x != "seconds"]
            kb = min(ks, key=lambda x: r[x]["all"])                             # 全体で最良の木数に固定
            vals.append(r[kb].get(cond, float("inf")))
        return sum(vals) / len(vals)

    rows = []
    for cond in conds:
        base = ll_fixed("base", cond)
        scored = [(ll_fixed(c, cond), c) for c in cfgs if ll_fixed(c, cond) is not None]
        if not scored or base is None:
            continue
        best, bc = min(scored)
        rows.append([cond, f"{base:.5f}", f"<code>{bc}</code>", f"{best - base:+.5f}"])
    return r29._tbl(["条件", "基準設定の LL（全体最良の木数）", "その条件で最良の設定", "基準比（負ほど良い）"], rows, ["s", "n", "s", "n"]) if rows else ""


def build(d: Path) -> None:
    res = json.loads((d / "ev/tp63_results.json").read_text(encoding="utf-8")) if (d / "ev/tp63_results.json").exists() else {}
    minfo = {}
    for f in sorted((d / "ev/members").glob("*.json")):
        minfo[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    sw = {}
    for f in sorted((d / "tune").glob("sweep_*.json")):
        sw.update(json.loads(f.read_text(encoding="utf-8")))
    opt = (TEXT / "tune_optuna.html").read_text(encoding="utf-8") if (TEXT / "tune_optuna.html").exists() else ""
    summary = (TEXT / "summary.html").read_text(encoding="utf-8") if (TEXT / "summary.html").exists() else ""
    hist = (d / "history.html").read_text(encoding="utf-8") if (d / "history.html").exists() else HISTORY
    done = (d / "done_table.html").read_text(encoding="utf-8") if (d / "done_table.html").exists() else DONE_TABLE
    concl = (TEXT / "conclusion.html").read_text(encoding="utf-8") if (TEXT / "conclusion.html").exists() else "<p>（結論は結果が出そろってから記載）</p>"
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>アンサンブルの幅出しとメンバー別チューニング — tp-63</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>アンサンブルの幅出し（ランダムフォレスト・決定木・別の目的関数）とメンバー別・条件別のチューニング — tp-63</h1>
<p class="muted">ユーザー要望: 今使っているモデル（LightGBM・NN）で幅を広げられないか。シンプルなランダムフォレスト（決定木）をアンサンブルに使えないか。さらに、レース質や全体特徴量を持つ学習器ごと、レース条件（コース・馬場など）ごとにハイパーパラメータを調整する余地。</p>
{summary}
<h2>1. 評価の枠と、これまでに済んでいた範囲</h2>
<ul>
 <li><b>土台</b>: V5 の 7 メンバーの OOF（valid 2022・2023・2024 の 140,865 行）。現行の合成（勝率 weighted_logit、連対・複勝 greedy）と、L3 型（勝率・連対を『現行 + クロス対象 LR』の logit 平均、複勝は現行。tp-51 の S11e）が基準。追加メンバーは V5 と同じ 1,142 列、年の前進（検証年より前の全年）で学習した OOF。</li>
 <li>採否は既定の基準（95% paired-bootstrap・レース単位の上限 &lt; 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。LOYO と前進の両方を出す。test 2025・2026 年は使っていない。市場オッズ・人気は使っていない。</li>
</ul>
{done}
<h2>2. V5 の 7 メンバーの棚卸し</h2>
{inventory_tables(res)}
<h2>3. 追加メンバーを足したときの増分</h2>
<p class="muted">①は『V5 の 7 + そのメンバー』を現行の合成にしたとき（現行合成に対する増分）。②は同じ 8 メンバーで L3 型にしたとき（現行合成比と、V5 の L3 型比）。</p>
{members_table(res, minfo)}
<h3>複数を同時に足す</h3>
{combos_table(res)}
<h2>4. ハイパーパラメータの調整履歴と感度</h2>
{hist}
<h3>主要パラメータを 1 つずつ動かした感度（勝率、valid 2022・2023）</h3>
{sweep_tables(sw)}
<h3>条件別の最適設定（同じ予測を条件で切って集計）</h3>
{cond_sweep_table(sw)}
{opt}
<h2>5. 結論</h2>
{concl}
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    build(Path(ap.parse_args().dir))


if __name__ == "__main__":
    main()
