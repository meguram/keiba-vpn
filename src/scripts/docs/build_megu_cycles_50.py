"""めぐサイクル（50 サイクル）のまとめ（docs/html/operations/final/megu_cycles_50.html）を、記録の JSON から作る（数字を手で写さない）。

入力（src/research/megu_cycles/ の道具が書く）:
  docs/operations/megu_cycles/confirm_log.json（2022〜24 の確かめ）・final_once.json（2025 年の 1 回の確かめ）・drift_check.json（再現の楽観）
  docs/operations/megu_cycles/cycle_NN.json / cycle_NN.md（各サイクル）・trials.json（試した数）・cycle_50.json（割引き）
使い方: python -m src.scripts.docs.build_megu_cycles_50
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.docs.build_latest_strategy_tables import CSS
from src.scripts.docs.build_megu_why import esc, num, pc, table

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/megu_cycles"
DST = ROOT / "docs/html/operations/final/megu_cycles_50.html"
TITLE = "めぐサイクル 50 回のまとめ"
JST = timezone(timedelta(hours=9))
LABEL = {"c13_flat": "c13 平場・T-60", "c18_flat_t05": "c18 平場・T-5", "c21_maiden_t60": "c21 新馬未勝利・T-60",
         "c23_flat_t15": "c23 平場・T-15", "c24_flat_v7": "c24 平場・T-60・s_* なし"}


def ci(r: dict | None, key: str = "roi_ci") -> str:
    if not r or not r.get(key):
        return "－"
    a, b = r[key]
    return f"{a * 100:.0f}〜{b * 100:.0f}%" if key == "roi_ci" else f"{a:.2f}〜{b:.2f}"


def cycle_rows() -> list[list]:
    T = json.loads((SRC / "trials.json").read_text(encoding="utf-8"))["cycles"]
    rows = []
    for n in range(1, 51):
        md = SRC / f"cycle_{n:02d}.md"
        title = re.sub(r"^# サイクル \d+: ", "", md.read_text(encoding="utf-8").splitlines()[0]) if md.exists() else "－"
        t = T.get(str(n), {})
        res = "目標を満たした" if t.get("goal_met") else ("関門を通過（当てない）" if t.get("gate_passed") else ("検査（探さない）" if t.get("trials") == 0 else "不合格"))
        rows.append([str(n), esc(title), str(t.get("trials", "－")), res])
    return rows


def build() -> str:
    CL = {r["spec"]["name"]: r for r in json.loads((SRC / "confirm_log.json").read_text(encoding="utf-8"))}
    FO = json.loads((SRC / "final_once.json").read_text(encoding="utf-8"))["results"]
    DR = json.loads((SRC / "drift_check.json").read_text(encoding="utf-8"))
    C50 = json.loads((SRC / "cycle_50.json").read_text(encoding="utf-8"))
    TR = json.loads((SRC / "trials.json").read_text(encoding="utf-8"))
    made = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    names = list(CL)
    passed25 = [n for n in names if FO[n]["pass"]]
    C43 = json.loads((SRC / "cycle_43.json").read_text(encoding="utf-8"))["sensitivity"]
    corr = [C50["corr"][a][b] for i, a in enumerate(names) for b in names[i + 1:]]
    surv = [n for n in names if C50["discount"][n]["lb_simultaneous_95"] > 1]
    import numpy as np

    def adj(n):
        """差し引いた見積もり = 凍結の 2017〜21 × 曲線(s*) ÷ 曲線(1)（サイクル 43 の曲線の比。絶対値の引き直しの揺れを除く）。"""
        cur = C43[n]
        xs = np.array([float(k) for k in cur])
        ys = np.array([v["roi"] for v in cur.values()])
        s_ = DR["adjusted_estimate"][n]["s_star"]
        if s_ is None:
            return None, None
        r = float(np.interp(s_, xs, ys) / np.interp(1.0, xs, ys))
        return r, CL[n]["select"]["roi"] * r

    o = [f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{TITLE}</h1>
<p class="sub">作成 {made}（JST）。数字は記録の JSON から生成（<code>src/research/megu_cycles/</code>・<code>src/scripts/docs/build_megu_cycles_50.py</code>）。
記録は <code>docs/operations/megu_cycles/</code>（cycle_01〜50.md・lessons.md）。担当 megu-cycles。費用 0 円（手元の表だけ。GCS を読まない）。</p>
"""]

    o.append(f"""<div class="box">
<b>結論</b>
<ul>
<li>目標（選ぶ期間 2017〜21 と確かめ 2022〜24 の両方で 100% 以上・差が小さい・年 50 件以上）を満たしたのは、<b>平場の単勝の 5 候補</b>。<b>重賞は 0</b>。</li>
<li>5 候補は 1 つの考えの変種。市場の列を使わない自前の確率（約 1,000 列の XGBoost 3 本の平均）と、発走前のオッズ（過去の年は、確定オッズに 2026 年の発走前と確定のずれを当てた再現）の確率を条件付き logit で組み、期待値 1.3 以上・50 倍以下の単勝を買う。上積みは新馬・未勝利と人気薄に集まる。</li>
<li>2025 年の 1 回だけの確かめで事前の線（2025 ≥ 100% かつ ≥ 2022〜24 × 0.9）を満たしたのは {len(passed25)} 候補（{'・'.join(LABEL[n] for n in passed25)}）。ただし 2025 年の 95% 区間はどの候補も 100% を含み、5 候補の同時の下限は {min(FO[n]['lb_simultaneous_95'] for n in names) * 100:.0f}〜{max(FO[n]['lb_simultaneous_95'] for n in names) * 100:.0f}%。<b>「100% を超える」と言い切れる候補は無い</b>。</li>
<li>相関を入れた割引き（2022〜24）の後も同時の下限が 100% を超えたのは {'・'.join(LABEL[n] for n in surv) or 'なし'}。本物の発走前オッズでの粗い測りでは、T-5 の再現が一番楽観寄り（s* {num(DR['t05']['s_star'])}。区間の無い目安）。</li>
<li>次の本当の確かめは、前向きの記録（2026-11-21〜、本物の発走前オッズと推論の経路の特徴量）。2025 年の線を満たした {'・'.join(LABEL[n] for n in passed25)} を前向きに数えることを勧める。</li>
</ul>
</div>
""")

    o.append("<h2>目標の判定（サイクル 1 の前に固定。config/megu_cycles/goal.json）</h2>")
    o.append("""<ul>
<li>重賞・平場のそれぞれで: 選ぶ期間（2017〜21）と確かめ（2022〜24）の両方で回収率 100% 以上、確かめ ≥ 選ぶ期間 × 0.9、確かめで年 50 件以上。</li>
<li>判定は発走前に分かる値だけ（T-60 などの再現オッズ 5 通りの平均）。精算は確定の払戻。</li>
<li>2022〜24 に当てる前の関門（2017〜19 で選び、2020〜21 で一次の確かめ・帰無・市場比の下限 &gt; 1）。2022〜24 に当てたのは通算 5 候補（上限 10）。</li>
<li>2025 年は 50 サイクルの後に 1 回だけ（config/megu_cycles/final_prereg.json を先に commit）。2026 年は特徴量の表に行が無く数えていない。</li>
</ul>""")

    o.append("<h2>目標を満たした候補（すべて平場の単勝）</h2>")
    rows = []
    for n in names:
        r, f = CL[n], FO[n]
        rows.append([LABEL[n], pc(r["select"]["roi"]), f"{r['select']['bets_per_year']:.0f}",
                     f"{pc(r['confirm']['roi'])}（{ci(r['confirm'])}）", f"{r['confirm']['bets_per_year']:.0f}", f"{num(r['confirm']['market_ratio'])}（{ci(r['confirm'], 'mr_ci')}）",
                     pc(r["confirm"]["base_same_pop"]), pc(r.get("secondary_smartrc_roi")),
                     f"{pc(f['roi'])}（{ci(f)}）", f"{f['bets_per_year']:.0f}", f"{num(f['market_ratio'])}（{ci(f, 'mr_ci')}）", pc(f["base_same_pop"]),
                     pc((f.get("smartrc_2025") or {}).get("roi")), "合格" if f["pass"] else "不合格"])
    o.append(table(["候補", "2017〜21", "件/年", "2022〜24（95% 区間）", "件/年", "市場比（区間）", "同じ人気の基準線", "副 SmartRC", "2025 年（95% 区間）", "件/年", "市場比（区間）", "基準線",
                    "副 SmartRC 2025", "2025 の判定"], rows, num_cols=frozenset(range(1, 13))))
    o.append("<p>市場比 = 勝ち数 ÷ 確定の市場の勝率の和。同じ人気の基準線 = 同じ期間の確定の人気ごとの全馬の回収率を、買った馬の人気で重みづけした値。副 SmartRC は判定のオッズを SmartRC の想定人気（0.8/E_q）に替えた値で、取得時刻の記録が無く確定に近い情報を含むおそれがあるため、独立の確かめとは読まない。</p>")

    o.append("<h2>割引き（相関を入れた同時の下限）</h2>")
    rows = []
    for n in names:
        x = C50["discount"][n]
        rows.append([LABEL[n], pc(x["roi_2022_24"]), pc(x["lb_single_95"]), pc(x["lb_simultaneous_95"]), num(x["p_holm"], 3),
                     pc(FO[n]["roi"]), pc(FO[n]["lb_simultaneous_95"])])
    o.append(table(["候補", "2022〜24", "単独の下限（片側 95%）", "同時の下限（2022〜24）", "Holm（近似）", "2025 年", "同時の下限（2025）"], rows, num_cols=frozenset(range(1, 7))))
    o.append(f"<p>5 候補の開催日ごとの回収率の相関は {min(corr):.2f}〜{max(corr):.2f}。開催日を同時にブートストラップし、標準化した偏差の最大の 95% 分位（2022〜24: {num(C50['c_max'])}）で割り引いた。"
             f"試した設定は通算 {TR['total_trials']:,}、2020〜21 を見た回数は {TR['select_b_looks']['count']}、2022〜24 に当てたのは {TR['total_confirm']} 候補。"
             "割引きは確かめに当てた候補の数（5）で行った。ただし c18〜c24 は c13 の 2022〜24 の結果を見た後に作った変種で、2022〜24 は後の候補にとって完全な外ではない（この割引きは楽観寄り）。</p>")

    o.append("<h2>判定のオッズの再現の楽観（本物の発走前オッズで測った）</h2>")
    rows = [[ph, str(DR[ph]["races"]), num(DR[ph]["ll_real"], 4), num(DR[ph]["ll_final"], 4), num(DR[ph]["ll_sim"]["1.0"], 4), num(DR[ph]["s_star"])] for ph in ("t60", "t15", "t05")]
    o.append(table(["時点", "レース", "勝ち馬の対数損失: 本物", "確定", "再現（倍率 1）", "合う倍率 s*"], rows, num_cols=frozenset(range(1, 6))))
    rows = []
    for n, v in DR["adjusted_estimate"].items():
        r, a = adj(n)
        rows.append([LABEL[n], v["phase"], num(v["s_star"]), pc(CL[n]["select"]["roi"]), num(r, 3), pc(a), pc(CL[n]["confirm"]["roi"]), pc(CL[n]["confirm"]["roi"] * r if r else None)])
    o.append(table(["候補", "時点", "s*（目安）", "凍結の 2017〜21", "曲線(s*) ÷ 曲線(1)", "見積もり 2017〜21", "2022〜24", "見積もり 2022〜24"], rows, num_cols=frozenset(range(2, 8))))
    o.append(f"<p>本物の発走前オッズ（2026 年の秋。全頭がそろったレース: t60 {DR['t60']['races']}・t15 {DR['t15']['races']}・t05 {DR['t05']['races']} レース。市場の値だけ）。"
             "s* &lt; 1 は再現が本物より厳しめ、&gt; 1 は楽観。見積もりは、凍結の値にサイクル 43 の曲線の比を掛けたもの。</p>"
             "<p><b>この測りの限界（批評の指摘）</b>: s* に区間が無い（倍率ごとに 5 回の繰り返し・レースのブートストラップなし。t05 は曲線が滑らかでない）。"
             "drift の再現は全頭がそろったレースから、凍結の再現は 5 頭以上が対応したレースから借りるので、s* はサイクル 43 の倍率と単位が厳密には合わない。"
             "勝ち馬の対数損失を市場全体で合わせただけで、買う馬（人気薄・自前の確率の高い馬）の条件付きの持ち込みは測れていない。"
             "曲線の外（倍率 2 より上）は端の値になる。目安として読む。</p>")

    o.append("<h2>壁（目標を満たさなかったところ）</h2>")
    o.append("""<ul>
<li><b>重賞</b>（10 サイクル）: 2017〜19 が 88〜96% で止まる。市場比は 1.1 前後で、平場（1.15〜1.4）より小さく、控除の壁（同じ人気の基準線 約 78% から 100% に約 1.28 倍が要る）に届かない。件数も年 130 レース程度で区間が広い。</li>
<li><b>条件の切り口だけ</b>（めぐ分析の水準・人気の区画）: 選び過ぎで、帰無と区別できない（megu-why と同じ）。</li>
<li><b>券種</b>: 複勝・ワイド・単勝＋複勝は控除の壁で上積みが残らない。</li>
<li><b>見送り</b>: レースの値の 1 水準・1 レース 1 点・順位の制限は、上積みの源（人気薄・新馬未勝利）を捨てるか、選び過ぎになる。</li>
<li><b>データのリーク</b>: 過去の年の調教の記録の偏り（tl_*・wc_*）、全期間の集計（rb_*）、上流の補助モデル（mtl_*）が、2017〜19 の成績を水増ししていた。結果を使わない決まり（記録の率・最頻値の段）で外した。</li>
</ul>""")

    o.append("<h2>汚れと限界</h2>")
    o.append("""<ul>
<li>確率の特徴量の表（model-cycles の X_xgb）には、上流で未来の年を含む学習（s_* の元の EASE2・LAP）と、2021 年末までの定数の当てはめ（sr・f3）が残る。s_* を外した c24 は 2025 年で市場比 1.00 に落ちた。</li>
<li>判定のオッズは、確定オッズに 2026 年のずれを当てた再現。本物の発走前オッズで測ると T-60 は厳しめ・T-5 は楽観（上の表）。</li>
<li>2020〜21 は関門のたびに見た（70 回）。2022〜24 は 5 候補で 5 回見た。</li>
<li>2026 年の候補の成績は数えていない（特徴量の表に行が無い）。運用には推論の経路で同じ特徴量を作る必要があり、学習と推論のずれの問題が残る。</li>
<li>判定のオッズの再現のずれの母集団（data/local/research/persona1/pre_odds.parquet）には、前向きの期間（2026-11-21〜22）の発走前オッズ（約 48 レース。オッズの値だけで結果ではない）が入っていた。表を作った時点（2026-11-23）のファイルで、凍結の候補（2017〜24）と 2025 年の確かめは同じ母集団を使ったので互いには整合するが、事前登録の「11-21 以降を使わない」には反する（drift の測りの側は 11-20 までで切った）。</li>
<li>台帳: drift の測りで 2026 年の結果の表を年の初めから読み込んだ（range_outside の警告。計算に使ったのは発走前オッズのある 10/18〜11/15 のレースだけ）。build_phase は 2022〜24 の確定オッズ（結果ではない）を申告なしに読んだ。</li>
<li>候補 c24 の凍結ファイルの columns_rule は v5 と書かれているが、確率のファイルは v7（s_* なし）。数えた値は v7 で正しい（記録の誤記）。</li>
</ul>""")

    o.append("<h2>50 サイクルの一覧</h2>")
    o.append(table(["回", "方向", "試した設定", "結果"], cycle_rows(), num_cols=frozenset({0, 2})))
    o.append("""<h2>道具と記録</h2>
<ul>
<li>道具: <code>src/research/megu_cycles/</code>（mc_data・mc_eval・cycles・mc_final）。台帳の系統 megu_cycles（src/ops/data_ledger.py の台帳）。</li>
<li>記録: <code>docs/operations/megu_cycles/</code>（README・lessons・cycle_01〜50・confirm_log・final_once・drift_check・trials）。候補の凍結は config/megu_cycles/candidates/。</li>
</ul>
</main>
</body>
</html>
""")
    return "\n".join(o)


def main() -> None:
    DST.write_text(build(), encoding="utf-8")
    print("書いた", DST)


if __name__ == "__main__":
    main()
