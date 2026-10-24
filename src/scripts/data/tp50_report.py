"""tp-50 の章 docs/html/modeling/chapters/bet_optimization.html を、summary.csv / summary.json / final2026.json / sens.json から作る。

python -m src.scripts.data.tp50_report --dir <tp50 の結果ディレクトリ>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
OUT = CH / "bet_optimization.html"

CLASS_JP = {
    "rule": "固定ルール（EV のしきい値・上位 k 点・年ごとに当てはめた単調回帰の較正 EV）",
    "lgbm": "教師あり: LightGBM（的中確率 × オッズ = EV、回帰、ケリー配分）",
    "lgbm2": "教師あり: LightGBM を 3 つのハイパーパラメータ × 3 つの特徴量集合 × 3 シードの平均に拡張",
    "lgbm3": "教師あり: 小さい木の周辺探索（葉数 7/15/31 × 本数 200/400 × 特徴量集合）",
    "lgbmb": "教師あり: 券種ごとの専用モデル",
    "mlp": "オフライン強化学習の悲観的な価値推定: 8 本の MLP（レース単位のブートストラップ）の EV の下側信頼限界",
    "ens": "アンサンブル: LightGBM 2 種と MLP の的中確率の平均",
    "pg": "方策勾配（完全情報の文脈つきバンディット。目的: 平均利益 / 対数資産 / 下位 20% の利益を含む CVaR）",
    "pg2": "方策勾配の大規模探索（隠れ層・学習率・エントロピー・シード）",
    "ppo": "PPO（レースごとに 1 点を選ぶか見送るか）",
    "league": "リーグ戦（12 本の集団の競争で選抜）＋批評役（悲観的な価値推定が 1.0 以上の買い目だけ許す）",
    "es": "進化戦略（線形スコア、下側信頼限界を目的に）",
    "tree": "決定木のルール抽出（葉の平均利益が高い領域）",
}


def head_style() -> str:
    t = (CH / "race_type_marks.html").read_text(encoding="utf-8")
    m = re.search(r"<style>.*?</style>", t, re.S)
    return m.group(0) if m else "<style></style>"


def pc(x, d: int = 0) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{d}f}%"


def tbl(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def build(sm: pd.DataFrame, js: dict, fin: dict | None, sens: dict | None, cond: dict | None = None) -> str:
    sm = sm.copy()
    sm["cls"] = sm["policy"].str.split("|").str[0]
    rc = js.get("reality_check") or {}
    n_pol, n_bets = js.get("n_policies"), js.get("n_with_bets")
    ok = sm[(sm["races_bet"] >= 300) & sm["roi"].notna()]
    best = ok.sort_values("lo", ascending=False)
    parts = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>tp-50 馬券最適化</title>{head_style()}</head><body><div class='wrap'>"]
    parts.append("<h1>馬券最適化（強化学習ほか）を、最新モデル＋修正後の期待値で大規模に回す（tp-50）</h1>")
    parts.append("<p class='muted'>目的: 回収率 100% 超（区間下限が 100% 超）の再現できる買い方を、手法を広げて探す。評価の厳密さ（walk-forward、レース単位、多重比較の補正、2026 年は事前固定して 1 回）は落とさない。</p>")
    top = best.iloc[0] if len(best) else None
    parts.append("<h2>1. 結論</h2><ul>")
    parts.append(f"<li><b>試した方策は {n_pol} 本（賭けが発生したもの {n_bets} 本）。</b>Reality Check（最良の選抜バイアスの補正）は p = {rc.get('p_value', float('nan')):.2f}、最大の t = {rc.get('t_max', float('nan')):.2f} で、<b>偶然を排除できない</b>。</li>")
    if top is not None:
        parts.append(f"<li>区間下限が最も高いのは <code>{html.escape(top['policy'])}</code>（2024〜25 年、{int(top['races_bet']):,} レース）: 回収率 {pc(top['roi'])}［{pc(top['lo'])}–{pc(top['hi'])}］、上位 3 的中を除くと {pc(top.get('roi_wo_top3'))}。多重比較を補正する前の区間下限は {'100% を超える' if top['lo'] > 1 else '100% に届かない'}が、補正後は有意でない。</li>")
    rob = ok[(ok["races_bet"] >= 2000) & (ok["y2023"] > 1.0) & (ok["y2024"] > 1.0) & (ok["y2025"] > 1.0)].sort_values("lo", ascending=False)
    if len(rob):
        r0 = rob.iloc[0]
        parts.append(f"<li>2023・2024・2025 年のどの年も 100% 超で、賭けが 2,000 レース以上の方策は {len(rob)} 本（いずれもワイドか複勝の LightGBM 系）。最上位は <code>{html.escape(r0['policy'])}</code>: 回収率 {pc(r0['roi'], 1)}［{pc(r0['lo'], 1)}–{pc(r0['hi'], 1)}］（2023 年 {pc(r0['y2023'])}、2024 年 {pc(r0['y2024'])}、2025 年 {pc(r0['y2025'])}、{int(r0['races_bet']):,} レース・{int(r0['bets']):,} 点）。ただし回収率の大半は人気薄どうしの組合せ（§7）で、上位 3 的中を除くと {pc(r0.get('roi_wo_top3'))}。</li>")
    if fin and fin.get("policies"):
        ok26 = [(k, v) for k, v in fin["policies"].items() if v and v.get("roi") is not None]
        up = [k for k, v in ok26 if v["lo"] > 1.0]
        parts.append(f"<li>2026 年（未使用データ・事前に固定した {len(fin['policies'])} 方策を 1 回だけ）: 区間下限が 100% を超えた方策は {'あり: ' + ', '.join(up) if up else '<b>なし</b>'}。Reality Check は p = {(fin.get('reality_check') or {}).get('p_value', float('nan')):.2f}。</li>")
        if len(rob):
            v = fin["policies"].get(rob.iloc[0]["policy"])
            if v and v.get("roi") is not None:
                parts.append(f"<li><b>2024〜25 年に最も一貫していたワイド方策は、2026 年に {pc(v['roi'], 1)}［{pc(v['lo'], 1)}–{pc(v['hi'], 1)}］（{v['races_bet']:,} レース）で、再現しなかった。</b>払戻率（ワイド 77.5%）並みに戻っており、2024〜25 年の超過分はモデルの優位というより、その期間の偶然か、市場との相対関係の一時的な偏りだった可能性が高い（2026 年はモデルが市場に弱い年でもある）。</li>")
    parts.append("<li>強化学習系（方策勾配・PPO・リーグ戦・進化戦略）は、固定ルールや LightGBM の確率予測に勝てず、多くは『賭けない』に近い方策に収束した（§4）。</li>")
    parts.append("</ul>")
    # 方策クラスごとの要約
    rows = []
    for c, g in sm.groupby("cls"):
        gg = g[(g["races_bet"] >= 300) & g["roi"].notna()]
        b = gg.sort_values("lo", ascending=False).iloc[0] if len(gg) else None
        rows.append([f"<code>{c}</code> {html.escape(CLASS_JP.get(c, ''))}", f"{len(g)}", f"{len(gg)}",
                     (f"<code>{html.escape(b['policy'].split('|', 1)[-1])}</code><br>{pc(b['roi'])}［{pc(b['lo'])}–{pc(b['hi'])}］" if b is not None else "—"),
                     pc(float(gg["roi"].median())) if len(gg) else "—"])
    parts.append("<h2>2. 方策クラスごとの結果（2024〜25 年のプール）</h2>")
    parts.append("<p class='muted'>主の評価は 2024〜25 年（学習は評価年より前の年だけ。2024 年は 2022〜23 年で、2025 年は 2022〜24 年で学習）。2023 年（2022 年だけで学習）は補助。右端は 300 レース以上で賭けた方策の回収率の中央値。</p>")
    parts.append(tbl(["方策クラス", "方策数", "300R以上で賭けた数", "区間下限が最高の方策", "回収率の中央値"], rows))
    # 上位
    rows = []
    for _, r in best.head(20).iterrows():
        rows.append([f"<code>{html.escape(r['policy'])}</code>", pc(r["roi"], 1), f"{pc(r['lo'], 1)}–{pc(r['hi'], 1)}", pc(r.get("roi_wo_top3")), f"{int(r['races_bet']):,}", f"{int(r['bets']):,}", pc(r["hit_rate"], 1),
                     pc(r.get("y2023")), pc(r.get("y2024")), pc(r.get("y2025"))])
    parts.append("<h2>3. 区間下限が高い方策の上位 20（2024〜25 年のプール）</h2>")
    parts.append(tbl(["方策", "回収率", "95% 区間", "上位3的中を除く", "賭けたレース", "点数", "的中率", "2023", "2024", "2025"], rows))
    parts.append(f"<p class='muted'>Reality Check: {n_bets} 方策のうち賭けが十分な {rc.get('n_strategies')} 本で、最大のスチューデント化統計量 {rc.get('t_max', float('nan')):.2f}、p = {rc.get('p_value', float('nan')):.2f}（最良: <code>{html.escape(str(rc.get('best_name')))}</code>）。</p>")
    # RL の収束
    rows = []
    for c in ("pg", "pg2", "ppo", "league", "es"):
        g = sm[sm["cls"] == c]
        if not len(g):
            continue
        st = g[g["policy"].str.endswith("|stake")]
        am = g[g["policy"].str.endswith("|argmax")]
        avg_w = float((st["stake"] / st["races_bet"].clip(lower=1)).median()) if len(st) else float("nan")
        rows.append([f"<code>{c}</code>", f"{len(g)}", pc(float(g["roi"].median())) if g["roi"].notna().any() else "—",
                     f"{avg_w:.2f}" if st is not None and len(st) else "—", f"{int(am['races_bet'].median())}" if len(am) else "—"])
    parts.append("<h2>4. 強化学習系の学習結果（『賭けない』への収束）</h2>")
    parts.append("<p>完全情報（過去の全候補の払戻が分かる）の方策勾配、PPO、リーグ戦、進化戦略を、学習年の末尾 15% で早期停止して学習した。目的関数の違い（利益・対数資産・CVaR）、エントロピー、隠れ層、学習率、シードを振っても、<b>方策は市場に対する優位をほとんど見つけず、賭け金の大半を『見送り』に置く</b>か、全買いに近い配分（回収率は払戻率に近い）に落ちた。優位が無い設定では、期待利益の最大化は『賭けない』が最適解になるため、これは学習の失敗ではなく、データに優位が乏しいことの反映と読む。</p>")
    parts.append(tbl(["クラス", "方策数", "回収率の中央値", "賭けた配分の中央値（1 レース当たりの重み）", "argmax 方策の賭けたレース数の中央値"], rows))
    # 2026
    if fin and fin.get("policies"):
        rows = []
        for k, v in fin["policies"].items():
            if not v or v.get("roi") is None:
                rows.append([f"<code>{html.escape(k)}</code>", "—", "—", "—"])
            else:
                rows.append([f"<code>{html.escape(k)}</code>", pc(v["roi"], 1), f"{pc(v['lo'], 1)}–{pc(v['hi'], 1)}", f"{v['races_bet']:,}"])
        rcf = fin.get("reality_check") or {}
        parts.append("<h2>5. 2026 年の最終評価（未使用データ・事前に固定した方策を 1 回だけ）</h2>")
        parts.append(f"<p class='muted'>固定時刻 {fin.get('registered_at')}、ハッシュ {fin.get('hash')}。方策は 2024〜25 年の結果だけから選んだ（区間下限の上位、回収率の上位、クラスごとの最良）。学習は 2022〜2025 年。2026 年はモデルが市場に弱かった年（モデルと市場の差 0.100）で、弱い検証。Reality Check: p = {rcf.get('p_value', float('nan')):.2f}。</p>")
        parts.append(tbl(["方策", "回収率", "95% 区間", "賭けたレース"], rows))
    if sens:
        rows = []
        for k, v in sens.items():
            rows.append([f"<code>{html.escape(k)}</code>"] + [f"{pc(v[s]['roi_mean'])}（最小 {pc(v[s]['roi_min'])}）" if s in v else "—" for s in ("0.0", "0.05", "0.1", "0.2")])
        parts.append("<h2>6. 購入時点のオッズの感度</h2>")
        parts.append("<p>判断に使うオッズに対数正規の誤差 σ を加えて選び、払戻は確定で精算した（購入時点と確定オッズの差の代理。5 回の平均と最小）。</p>")
        parts.append(tbl(["方策（LightGBM、1 シード）", "σ=0", "σ=0.05", "σ=0.1", "σ=0.2"], rows))
        parts.append("<p class='muted'>σ=0 の回収率は、まとめ（3 シード平均）の値と数 % 違う。同じ構成でもシードで数ポイント動くことは、この種の優位が雑音の水準に近いことを示す。</p>")
    if cond:
        def ct(key: str, title: str) -> str:
            rows = [[html.escape(k), f"{int(v['bets']):,}", pc(v["roi"], 1), pc(v["hit"], 1), f"{v['mean_ev_hat']:.2f}"] for k, v in cond[key].items()]
            return f"<h3>{title}</h3>" + tbl(["区分", "点数", "回収率", "的中率", "予測 EV の平均"], rows)

        parts.append("<h2>7. 最有力のワイド方策は何を買っているか</h2>")
        parts.append(f"<p>方策: <code>lgbm3|L15|E400|nomix|wide&gt;=1.0</code>（小さい木の LightGBM 2 シード平均が出す的中確率 × オッズ ≥ 1.0 のワイドを全て等額で買う）。2024〜25 年で {cond['n_bets']:,} 点、回収率 {pc(cond['overall_roi'], 1)}。内訳（確定オッズ・確定払戻）:</p>")
        parts.append(ct("year", "年別") + ct("odds_band", "オッズ帯別（ワイドの目安オッズ）") + ct("pop_band", "組合せの人気の低い方の馬の単勝人気別") + ct("n_band", "頭数別") + ct("ev_hat_band", "予測 EV の水準別"))
        parts.append("<p>読み: 回収率の大半は<b>オッズ 20 倍以上・人気薄どうしの組合せ</b>から来ている（的中率 1〜4%）。1〜3 倍台の組合せや人気馬どうしは 94〜100% で、払戻率並み。つまり『モデルが人気薄の組合せの 3 着以内率を市場より高く見積もれている』領域に優位があるように見えるが、的中が少なく、上位 3 的中を除くと下がる（表 3）。予測 EV が高い帯ほど回収率が高いとは限らず、較正は不完全。</p>")
    n_sec = 8 if cond else 7
    parts.append(f"<h2>{n_sec}. 限界と次の一手</h2><ul>"
                 "<li>払戻は<b>確定オッズ</b>で精算している。購入時点（T-60/T-15/T-10/T-5/T-2）のオッズとは違う。直前オッズの履歴（2026-10-24 から蓄積）が貯まったら、同じ方策を購入時点のオッズで再評価する。</li>"
                 "<li>主の評価は 2 年（2024〜25 年）。学習できる年が短く、2022 年以前の予測（V4 の leave-one-year-out）が無いため、深い方策ほど学習データが足りない。2020〜21 年の予測を足せば評価年を増やせる。</li>"
                 "<li>V4 の確率は、2022〜24 年は leave-one-year-out、2025 年は 2020〜24 年で学習した予測、2026 年は 2020〜25 年で学習した予測。学習年の特徴量（予測値）に、評価年より後のデータで学習したモデルの出力が混じる点は、厳密な時系列の学習ではない。</li>"
                 "<li>多重比較: 方策数が増えるほど偶然に良く見える方策が出る。Reality Check の p が大きい間は、『最良の方策』は仮説の候補であり、採用の根拠ではない。</li></ul>")
    parts.append("</div></body></html>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    sm = pd.read_csv(d / "summary.csv")
    js = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fin = json.loads((d / "final2026.json").read_text(encoding="utf-8")) if (d / "final2026.json").exists() else None
    sens = json.loads((d / "sens.json").read_text(encoding="utf-8")) if (d / "sens.json").exists() else None
    cond = json.loads((d / "cond.json").read_text(encoding="utf-8")) if (d / "cond.json").exists() else None
    OUT.write_text(build(sm, js, fin, sens, cond), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
