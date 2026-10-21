"""tp-70 の章 docs/html/modeling/chapters/prob_calibration_gap.html を、tp70_*.json から生成する。

python -m src.scripts.data.tp70_report --in DIR [--out docs/html/modeling/chapters/prob_calibration_gap.html]
DIR には tp70_diag.json / tp70_compare.json / tp70_ev.json / tp70_exotic.json / tp70_marks.json / tp70_cond.json がある想定。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
code { background: #1b2a4d; padding: 1px 5px; border-radius: 4px; font-size: 13px; }"""

TNAME = {"win": "勝率", "top2": "連対率", "top3": "複勝率"}
MNAME = {"C0": "生（モデルそのまま）", "C1": "市場のみ（単勝オッズ→Harville）", "C2": "Platt（モデルのみ）", "C3": "ブレンド（モデル+市場、3 係数）", "C4": "ブレンド+食い違い項",
         "C5": "単調制約 GBM", "C6": "人気区分別ブレンド", "C3r": "ブレンド（直近 52 週で毎月再当てはめ）"}
CLS = ["1人気", "2人気", "3人気", "4-5人気", "6-9人気", "10人気以下"]


def tbl(rows: list[list], head: list[str]) -> str:
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td class=\"num\">{c}</td>" if isinstance(c, (int, float)) else f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def f4(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.4f}"


def f5(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.5f}"


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", default="docs/html/modeling/chapters/prob_calibration_gap.html")
    a = ap.parse_args()
    D = Path(a.indir)
    diag = json.load(open(D / "tp70_diag.json"))
    comp = pd.DataFrame(json.load(open(D / "tp70_compare.json")))
    ev = json.load(open(D / "tp70_ev.json"))
    exo = json.load(open(D / "tp70_exotic.json"))
    marks = json.load(open(D / "tp70_marks.json"))
    cond = pd.DataFrame(json.load(open(D / "tp70_cond.json")))
    summ = pd.DataFrame(diag["summary"])

    s = []
    s.append("<h2>1. 診断: 平均では合っているが、人気区分ごとに系統的にずれる</h2>")
    s.append("<p>全体の平均（勝率・連対率・複勝率）は実績と 0.0001〜0.0002 以内で合っており、ECE も 0.002〜0.007 と小さい。ずれは<b>人気区分ごと</b>に出る。複勝率（V4 の OOF、2022〜2025 年。市場は単勝オッズから Harville で作った複勝率）:</p>")
    rows = []
    for c in CLS:
        r = summ[(summ["period"] == "2022-25") & (summ["target"] == "top3") & (summ["pc"] == c)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([c, int(r["n"]), pct(r["pred"]), pct(r["mkt"]), pct(r["act"]), f"{(r['act'] - r['pred']) * 100:+.1f}pt", f4(r["ll_model"]), f4(r["ll_mkt"])])
    s.append(tbl(rows, ["人気", "頭数", "モデル予測", "市場(単勝→Harville)", "実績", "実績−予測", "LL モデル", "LL 市場"]))
    s.append("<p>人気馬（1〜3 人気）はモデルが<b>低く</b>見積もり（1 人気は実績がモデルより 5.9pt 上）、10 人気以下は<b>高く</b>見積もる（実績がモデルより 1.8pt 下）。市場の複勝率は Harville が勝率の高い馬に寄るため 1 人気を 8.9pt 過大に見積もる（市場単独も較正されていない）。</p>")
    s.append("<h3>1-1. 低確率域の人気馬（ユーザーの指摘）</h3>")
    rows = []
    for r in diag["by_popularity"]["top3"]:
        if r["pc"] in ("3人気", "4-5人気") and r["pred"] < 0.16 and r["n"] >= 20:
            rows.append([r["pc"], r["b"], int(r["n"]), pct(r["pred"]), pct(r["act"]), f"{r['gap'] * 100:+.1f}pt"])
    s.append(tbl(rows, ["人気", "予測ビン", "頭数", "予測", "実績", "実績−予測"]))
    s.append("<p>3〜5 人気で複勝確率が 3〜10% と予測された馬は、実績が 13〜29%（予測の 2〜6 倍）。逆に 10 人気以下では予測 4.4% に対し実績 3.3%、予測 7.8% に対し 5.7%（過大）。</p>")
    s.append("<h3>1-2. 食い違いが大きいほど実績は市場寄り（モデルの自信過剰）</h3>")
    rows = []
    for r in diag["disagreement"]["top3"]:
        rows.append([int(r["dq"]) + 1, f"{r['gap']:+.2f}", pct(r["pred"]), pct(r["mkt"]), pct(r["act"]), "—" if r["w_model"] is None else f"{r['w_model']:.2f}"])
    s.append(tbl(rows, ["食い違い十分位（モデル logit − 市場 logit、1=モデルが市場より低い）", "平均差", "モデル", "市場", "実績", "実績がモデル寄りの度合い（0=市場、1=モデル）"]))
    s.append("<p>実績は両端の十分位で市場とモデルの<b>間</b>（モデル寄りの度合い 0.34〜0.48。中央の十分位は差が小さく比が不安定）。モデルが市場から離れるほど（両端）、その差の約半分以上は実現しない。これが『人気なのに低評価』の馬で複勝確率が過小になり、『穴なのに高評価』で過大になる直接の原因で、<b>モデルの確率を市場側へ縮める</b>（Benter 型の対数線形ブレンド）と、人気区分別のずれも一度に直る。</p>")
    s.append("<h3>1-3. モデルは市場に LogLoss で負けている（2026 年に拡大）</h3>")
    rows = []
    for per in ("2022-25", "2026"):
        for t in ("win", "top2", "top3"):
            r = summ[(summ["period"] == per) & (summ["target"] == t) & (summ["pc"] == "全体")].iloc[0]
            rows.append([per, TNAME[t], f4(r["ll_model"]), f4(r["ll_mkt"]), f5(r["ll_model"] - r["ll_mkt"]), f4(r["ece_model"]), f4(r["ece_mkt"])])
    s.append(tbl(rows, ["期間", "目的", "LL モデル", "LL 市場", "モデル−市場", "ECE モデル", "ECE 市場"]))
    s.append("<p class=\"muted\">1 頭 1 行の二値 LogLoss。モデル−市場が正＝市場のほうが良い。月別でも 2026 年 1〜10 月のすべての月で正（勝率 +0.004〜+0.013）。診断（drift_diagnosis）の『モデル−市場差の拡大』と同じ現象。</p>")

    s.append("<h2>2. 較正器の比較（ウォークフォワード）</h2>")
    s.append("<p>各評価年は<b>前年までの OOF だけ</b>で較正器を当てはめる（2023←2022、2024←2022〜23、2025←2022〜24、2026←2022〜25）。目的（勝率・連対率・複勝率）ごとに別の較正器を作り、レース内で合計を 1・2・3 に合わせる。市場の確率は単勝オッズだけから作る（複勝オッズは 2026 年に無く、あっても改善しない: 2.3 節）。</p>")
    s.append("<ul>" + "".join(f"<li><code>{k}</code> {v}</li>" for k, v in MNAME.items()) + "</ul>")
    for per in ("2023-25", "2026"):
        rows = []
        for m in MNAME:
            r = [MNAME[m]]
            for t in ("win", "top2", "top3"):
                x = comp[(comp["period"] == per) & (comp["target"] == t) & (comp["method"] == m)].iloc[0]
                r += [f4(x["ll"]), f4(x["ece"])]
            rows.append(r)
        s.append(f"<h3>2-1. {per}</h3>")
        s.append(tbl(rows, ["較正器", "勝率 LL", "勝率 ECE", "連対 LL", "連対 ECE", "複勝 LL", "複勝 ECE"]))
    s.append("<h3>2-2. 人気区分別の平均のずれ（複勝、実績−予測）</h3>")
    rows = []
    for per in ("2023-25", "2026"):
        for m in ("C0", "C1", "C3", "C6"):
            x = comp[(comp["period"] == per) & (comp["target"] == "top3") & (comp["method"] == m)].iloc[0]
            rows.append([per, MNAME[m]] + [f"{x['gap_' + c] * 100:+.1f}pt" for c in CLS])
    s.append(tbl(rows, ["期間", "較正器"] + CLS))
    s.append("<h3>2-3. LogLoss の差（レース単位ブートストラップ 95%）</h3>")
    ll = pd.DataFrame(ev["ll_diff"])
    rows = []
    for _, r in ll[ll["method"].isin(["C3", "C6"])].iterrows():
        rows.append([r["period"], TNAME[r["target"]], r["method"], "生" if r["vs"] == "C0" else "市場", f5(r["diff"]), f"[{r['lo']:+.5f}, {r['hi']:+.5f}]"])
    s.append(tbl(rows, ["期間", "目的", "較正器", "比較先", "差（負＝較正のほうが良い）", "95% 区間"]))
    s.append("<p><b>読み方:</b> 市場と組み合わせた較正（C3）は、生のモデルに対して全目的・全期間で有意に改善（勝率 −0.0049、連対 −0.0073、複勝 −0.0076。2026 年は −0.0077／−0.0114／−0.0124）し、市場単独にも 2023〜25 年は全目的で勝つ。<b>2026 年は勝率・連対では市場と区別がつかない</b>（勝率 +0.0003 [−0.0002, +0.0009]、連対 −0.0005 [−0.0012, +0.0005]）＝この期間のモデルは市場に対して勝率・連対の追加情報をほとんど持たない（drift の主因は未確定）。複勝は 2026 年も市場に勝つ（−0.0040）。人気区分別ブレンド（C6）は連対・複勝でだけ C3 より僅かに良い（複勝 −0.0005、連対 −0.0002）。単調制約 GBM（C5）・食い違い項（C4）・ローリング再当てはめ（C3r）は C3 に対して実質差なし（C3r は 2026 年の勝率で −0.00018、連対 −0.0002）。</p>")
    s.append("<h3>2-4. 識別力（AUC）も上がる</h3>")
    rows = []
    for per in ("2023-25", "2026"):
        for t in ("win", "top2", "top3"):
            a_ = ev["auc_all"][per][t]
            rows.append([per, TNAME[t], f"{a_['C0']:.4f}", f"{a_['C1']:.4f}", f"{a_['C3']:.4f}", f"{a_['C6']:.4f}"])
    s.append(tbl(rows, ["期間", "目的", "AUC 生", "AUC 市場", "AUC C3", "AUC C6"]))
    s.append("<p>較正は『確率の水準を直す』だけでなく、市場情報とモデル情報の合成なので順位づけの精度も上がる（全馬の複勝 AUC 0.813→0.822、2026 年 0.806→0.821）。</p>")
    s.append("<h3>2-5. 複勝オッズを市場情報に足すか</h3>")
    s.append("<p>複勝オッズ（2022〜2025 年）から作った市場の複勝率は、単勝由来（Harville）の市場と同じ精度（LogLoss 0.4065 対 0.4066）で、ブレンドに足しても 0.3995→0.3994 と変わらない。<b>単勝オッズだけで足りる</b>ので、複勝オッズの無い期間（2026 年のバックフィル）や、複勝オッズが未発表の時点でも同じ較正器が使える。</p>")

    s.append("<h2>3. 人気馬の低確率域は縮むか</h2>")
    rows = []
    for r in ev["popular_low"]:
        if isinstance(r.get("raw_bin"), str):
            rows.append([r["period"], r["raw_bin"], int(r["n"]), pct(r["raw"]), pct(r["C3"]), pct(r["C6"]), pct(r["mkt"]), pct(r["act"])])
    s.append(tbl(rows, ["期間", "生の複勝確率(区間)", "頭数(1〜5人気)", "生", "較正 C3", "較正 C6", "市場", "実績"]))
    s.append("<p>1〜5 人気で生の複勝確率が 10% 以下の馬（559 頭）は、生の予測 7.5% に対し実績 14.9%。較正後は 16.5%（C3）・17.9%（C6）で、ずれは 7.4pt（過小）→ 1.6〜3.0pt（やや過大）に縮んだ（市場は 24.4% で大きく過大）。0.10〜0.15 の区間も生 12.9%→実績 22.7% が、較正 20.9〜22.2% に合う。2026 年も同じ（生 7.6%→較正 15.7〜17.6%、実績 17.1%）。</p>")
    s.append("<div class=\"note\"><b>縮まらない部分:</b> 較正で直るのは『確率の水準』で、『その馬が来る・来ないの見分け』ではない。1〜5 人気で生の複勝確率が最低の層は、較正後も約 16〜18% 来る（消印の検証で、1〜5 人気で 10% 以下と言える集合が無かったのと同じ）。人気馬に対し『来ない』と自信を持てる情報は、モデルにも市場にも無い。確率が<b>正しくなる</b>ので EV の計算は良くなるが、消印の対象は増えない。</div>")

    s.append("<h2>4. 期待値（EV）の信頼性</h2>")
    s.append("<p>EV = 確率 × 確定オッズ（単勝）／× 複勝の実効オッズ（下限+0.3×(上限−下限)）。EV の帯別に『平均の予測 EV』と『実現回収率』を比べ、件数加重の |予測−実現| と傾き（45 度線なら 1）で要約した（複勝は払戻のある 2023〜2025 年のみ）。</p>")
    mae = pd.DataFrame([r for r in ev["selection"] if r["metric"] == "ev_gap_mae"])
    rows = []
    for _, r in mae.iterrows():
        rows.append([r["kind"], r["period"], r["ev_type"], f4(r["value"]), f"{r['slope']:.2f}"])
    s.append(tbl(rows, ["券種", "期間", "EV の作り方", "|予測EV−実現| 件数加重", "実現÷予測の傾き"]))
    s.append("<p>生の確率による EV は、高い帯ほど実現と離れる（単勝 EV 2.0 超の帯は平均 2.85 に対し実現 1.10）。較正後の確率による EV は平均の差が 0.31→0.05〜0.08（単勝）、0.19→0.03〜0.06（複勝）に縮む。tp-49 の EV 単調回帰（生の EV を直接補正）は 2023〜25 年は 0.016/0.013 と最も小さいが、これは当てはめと同じ期間での評価（甘い）で、<b>2026 年は 0.077</b> と較正後の確率（C3 0.045）より悪い。</p>")
    s.append("<h3>4-1. 同じ購入数で EV 上位を買った実現回収率（2023〜25 年プール）</h3>")
    sel = pd.DataFrame([r for r in ev["selection"] if str(r["metric"]).startswith("roi_top")])
    rows = []
    for kind in ("win", "place"):
        for q in ("roi_top1pct", "roi_top2pct", "roi_top5pct"):
            for et in ("raw", "C3", "C6", "mkt"):
                x = sel[(sel["kind"] == kind) & (sel["period"] == "2023-25") & (sel["metric"] == q) & (sel["ev_type"] == et)]
                if x.empty:
                    continue
                x = x.iloc[0]
                rows.append([kind, q.replace("roi_top", "上位 ").replace("pct", "%"), et, int(x["n"]), pct(x["hit"]), f"{x['ev_mean']:.2f}", f"{x['value'] * 100:.0f}% [{x['lo'] * 100:.0f}, {x['hi'] * 100:.0f}]"])
    s.append(tbl(rows, ["券種", "購入", "EV の作り方", "購入数", "的中率", "平均 EV", "実現回収率 [95%]"]))
    s.append("<p>生の EV の上位は平均 EV 3〜5 倍の『穴の外れ値』で的中率 1〜2%、回収率は 87%〜149% と区間が広くばらつく。較正後は平均 EV が 1.05〜1.4 に収まり的中率が 4.6〜8.6%（生の 1.5〜2.2% の約 3〜5 倍）になる。単勝の較正後は区間の下限が 82〜94% で 100% を超えない。生の EV の上位 2% だけ 149% [103, 206] と下限が 100% を超えるが、隣の上位 1%（87%）・5%（123% [99, 152]）と整合せず、的中率 2% の超大穴 1〜2 本に依る外れ値で、見た組み合わせも多い（較正器 4 × 購入量 3 × 券種 2）ため頑健とは言えない。複勝 C6 は 2023〜25 年プールで上位 1% 123% [101, 151]・2% 115% [100, 131] と下限が 100% 近辺まで来るが、見た組み合わせの数（較正器 4 × 購入量 3 × 券種 2）を考えると有意とは言えない。2026 年は複勝の払戻が無く未確認。</p>")
    s.append("<h3>4-2. EV 閾値で買う（複勝、2023〜25 年）</h3>")
    th = pd.DataFrame([r for r in ev["selection"] if str(r["metric"]).startswith("roi_ev") and r["kind"] == "place"])
    rows = []
    for et in ("raw", "C3", "C6", "mkt"):
        for m in ("roi_ev1.0", "roi_ev1.1", "roi_ev1.2"):
            x = th[(th["ev_type"] == et) & (th["metric"] == m)]
            if x.empty or pd.isna(x.iloc[0]["value"]):
                continue
            x = x.iloc[0]
            rows.append([et, m.replace("roi_ev", "EV≥"), int(x["n"]), pct(x["hit"]), f"{x['value'] * 100:.1f}% [{x['lo'] * 100:.1f}, {x['hi'] * 100:.1f}]"])
    s.append(tbl(rows, ["EV の作り方", "閾値", "購入数", "的中率", "実現回収率 [95%]"]))
    s.append("<p>生の EV で買うと 89〜93%、較正（C3）で 93%、人気区分別（C6）で 101〜108%（区間の下限 96〜98%）。件数が多い（4 千〜1.2 万）ぶん区間は狭いが、下限は 100% を超えない。<b>前向き追跡の候補</b>として設定（C6、複勝 EV≥1.1、10 点以内など）を凍結する価値はあるが、現時点では採用の根拠にならない。</p>")

    s.append("<h2>5. 組合せ券（馬連・ワイド・3連複）の確率</h2>")
    s.append("<p>割引 Harville（exotic_probabilities）の入力を、生／市場のみ／較正後の勝率・連対率・複勝率に替え、評価期間ごとに無作為抽出したレース（2023〜25 年 3,000R、2026 年 1,500R）で、全ペア・全 3 頭組の『予測確率 vs 的中』の LogLoss を比べた。</p>")
    rows = []
    for per in exo:
        for k, nm in (("um", "馬連"), ("wd", "ワイド"), ("tr", "3連複")):
            r = exo[per][k]
            rows.append([per, nm, int(r["n_items"]), f"{r['raw']['ll']:.5f}", f"{r['mkt']['ll']:.5f}", f"{r['C3']['ll']:.5f}", f"{r['C6']['ll']:.5f}", f"{(r['C3']['ll'] / r['raw']['ll'] - 1) * 100:+.1f}%"])
    s.append(tbl(rows, ["期間", "券種", "組数", "LL 生", "LL 市場", "LL C3", "LL C6", "C3/生−1"]))
    s.append("<p>三確率を較正すると、組合せ券の LogLoss も 2〜3% 改善する（2026 年はいずれも 市場単独とほぼ同じか僅かに良い）。確率が高いほうの 1%（買う候補になる組）の予測と実績は、生で 16.1%／16.6%（馬連）と合っているが、市場単独は予測 20.8%／実績 18.6% で過大、較正後は 18.6%／18.9% と合う。</p>")

    s.append("<h2>6. 印への影響</h2>")
    for per, v in marks.items():
        s.append(f"<h3>6-{1 if per == '2023-25' else 2}. {per}（{v['horses']:,} 頭）</h3>")
        rows = []
        for sym in ("◎", "○", "▲", "★", "△", "消", "-"):
            r = [sym]
            for k in ("raw", "C3"):
                x = v["marks"][k].get(sym)
                r += [("—" if not x else f"{x['n']:,}"), ("—" if not x else pct(x["win"])), ("—" if not x else pct(x["top3"]))]
            rows.append(r)
        s.append(tbl(rows, ["印", "生 頭数", "生 勝率", "生 3着内", "較正 頭数", "較正 勝率", "較正 3着内"]))
        s.append(f"<p class=\"muted\">印が変わる馬は {v['changed_share']['C3'] * 100:.1f}%（C3）。</p>")
    s.append("<p>◎の勝率が 32.2%→33.8%（3 着内 63.3%→65.2%）、2026 年は 29.8%→33.0%（61.8%→65.5%）と上がる。★は期待値の水準が下がるぶん 7,039→2,051 頭に減る（生の EV が膨らんでいた分）。消印は固定しきい値（10 人気以下の複勝確率 8%）のまま適用すると 33,766→39,836 頭に増え、3 着内率は 2.5%→3.0%（目標 5% 以下は満たす）。<b>しきい値は較正後の確率で設計し直す</b>（kesi_mark の手順）。</p>")

    s.append("<h2>7. 条件別・月別</h2>")
    c1 = cond[cond["cond"] == "field"]
    rows = []
    for _, r in c1.iterrows():
        rows.append([r["value"], r["period"], int(r["n"]), f5(r["llgap_raw"]), f5(r["llgap_C3"]), f5(r["llgap_C6"]), f5(r["llgap_C3r"])])
    s.append("<h3>7-1. 頭数帯（複勝 LL −市場、負＝市場より良い）</h3>" + tbl(rows, ["頭数", "期間", "頭数(行)", "生", "C3", "C6", "C3r"]))
    c2 = cond[(cond["cond"] == "month") & (cond["period"] == "2026")]
    rows = [[r["value"], int(r["n"]), f5(r["llgap_raw"]), f5(r["llgap_C3"]), f5(r["llgap_C3r"])] for _, r in c2.iterrows()]
    s.append("<h3>7-2. 2026 年の月別（複勝 LL −市場）</h3>" + tbl(rows, ["月", "行数", "生", "C3", "C3r"]))
    s.append("<p>生のモデルは 2026 年のすべての月・頭数帯で市場より悪い（+0.002〜+0.019）が、較正後は全月・全頭数帯で市場を上回る（−0.0009〜−0.0081）。ローリング再当てはめ（C3r）と固定（C3）の差は小さい（月次の係数の変動は小さく、ドリフトの主因は係数の古さではない）。</p>")

    s.append("<h2>8. 導入案（本番反映はしていない）</h2>")
    s.append("<ul><li><b>係数</b>: <code>config/prob_calibration.json</code>（2022〜2025 年の V4 OOF で当てはめた、目的ごとの 3 係数と人気区分別 6×3 係数。再作成は <code>python -m src.scripts.data.tp70_fit export --base base.parquet --out config/prob_calibration.json</code>）。既定は勝率=ブレンド(C3)、連対・複勝=人気区分別(C6)。</li>"
             "<li><b>適用</b>: <code>src/utils/prob_calibration.py</code> の <code>calibrate_entries(entries, odds_by_horse, market_is_real=True)</code>。<b>フラグ <code>KEIBA_PROB_CALIBRATION=1</code>（既定オフ）</b>のときだけ有効。8 頭未満・単勝オッズが全頭揃わない・オッズが予測オッズ（モデル由来で独立な市場ではない）のときは何もしない（従来の確率）。</li>"
             "<li><b>差し込み位置</b>: <code>src/api/local_predictions.py</code> のエントリ生成ループの直前（<code>pw, p2, p3 = …</code> の行）と、<code>src/api/app.py</code> の <code>prob_list</code> の直後（ライブ採点）。較正後の確率を <code>win_prob/top2_prob/top3_prob</code> に入れ、元の確率は <code>win_prob_model/top2_prob_model/top3_prob_model</code> に退避して API で両方返す。以降の EV・印・買い判断・推奨・割引 Harville は、そのまま較正後を使う。UI は『AI 予測の確率』（モデル）と『市場補正後の確率』（EV・印・推奨に使う）を並べて出す。</li>"
             "<li><b>二重較正の整理</b>: tp-49 の EV 単調回帰（<code>config/ev_calibration.json</code>）は生の EV 用。フラグ有効時は、較正後の確率から作った EV で<code>calibrate_ev.py</code> を当てはめ直す（またはフラグ有効時は tp-49 を恒等に）。そのままだと二重に補正する。</li>"
             "<li><b>消印</b>: しきい値 8% は生の V4 確率で設計した。フラグを有効にするなら、較正後の確率で tp67 の手順（Clopper-Pearson 上側限界）をやり直す。</li>"
             "<li><b>V5</b>: V4 の OOF で当てはめた係数を V5 の 2025 年予測（final_eval_v5）に適用しても V4 と同じ改善（複勝 LL 0.4039→0.3971、V4 は 0.4047→0.3974、人気区分別のずれも同じ形）。V5 は V4 と同じ方向のずれなので係数は共用できる。V5 の LOYO OOF が揃ったら当てはめ直す。<span class=\"muted\">（この確認の係数は 2025 年を含めた当てはめで、評価は甘い。差は V4/V5 間の比較として読む。）</span></li>"
             "<li><b>オッズの時点</b>: 係数は確定オッズで当てはめた。T-60/T-15 のオッズで使うと、市場の情報量が少ないぶん市場側の係数が大きすぎる。直前オッズ（T-10/T-5/T-2）が貯まったら、時点別に当てはめを比べる。</li>"
             "<li><b>ドリフト監視</b>: 週次の drift_monitor に『較正後の ECE（人気区分別）とモデル−市場差』を足す（<code>tp70_fit compare</code> の指標）。係数の再当てはめは、警告（連続 2 週）が出たときに実行。</li>"
             "<li><b>ロールバック</b>: <code>KEIBA_PROB_CALIBRATION</code> を外して API を再起動（コードは変更しない設計）。</li></ul>")

    s.append("<h2>9. 結論（正直な整理）</h2>")
    s.append("<ul><li><b>縮まる部分</b>: 人気区分別の平均のずれ（1 人気 +5.6pt → −2.3〜−0.3pt、低確率の人気馬 7.4pt → 1.6〜3.0pt）、LogLoss（生に対し勝率 −0.0049／連対 −0.0073／複勝 −0.0076）、EV の信頼性（|予測−実現| 0.31→0.05〜0.08）、組合せ券の確率（−1〜3%）、◎の勝率（+1.6pt）。2026 年も同方向。</li>"
             "<li><b>縮まらない部分</b>: (1) 個々の馬が来るか来ないか（人気馬で確率が低い馬が約 16% 来る）。(2) 2026 年の勝率・連対ではモデルが市場に追加情報を持たない（C3 でも市場と同等）。モデル側の情報が弱っている（調教データの欠損など、drift 診断の主因）ことは較正では直らない。(3) 100% 超と頑健に言える馬券は見つからない（単勝の較正後は区間下限 82〜94%、複勝 C6 の閾値買いは下限 96〜98%）。</li>"
             "<li><b>推奨</b>: フラグ付きで導入して、画面は『AI 予測の確率』と『市場補正後の確率』を併記（ユーザー判断）。複勝 EV（C6）の閾値買いは前向き追跡へ。</li></ul>")
    s.append("<h2>10. 再現</h2>")
    s.append("<p><code>python -m src.scripts.data.tp70_diag prep --frame frame_all.parquet --rank-dir DIR --out DIR</code> → <code>tp70_diag run</code> → <code>tp70_fit compare</code> → <code>tp70_ev run --preds tp70_cal_preds.parquet</code> → <code>tp70_exotic run</code> → <code>tp70_marks run</code> → <code>tp70_cond run</code> → <code>tp70_report --in DIR</code>。入力は tp-36 の frame_all（V4 の OOF、確定の単勝・複勝オッズと払戻）と rank_tbl。2026 年は最終確認のみで、較正器の選択・係数には使っていない。</p>")

    head = ("<!doctype html>\n<html lang=\"ja\"><head><meta charset=\"utf-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
            "<title>確率の較正（予測と実績の差）— 市場と組み合わせて人気馬の「来ない」の過小評価を縮め、期待値の信頼性を上げる（tp-70）— keiba-vpn</title>"
            f"<style>{CSS}</style></head>\n<body><a href=\"../../index.html\" class=\"muted\">← ドキュメント一覧</a>\n"
            "<h1>確率の較正（予測と実績の差）— 市場と組み合わせて人気馬の「来ない」の過小評価を縮め、期待値の信頼性を上げる（tp-70）</h1>\n"
            "<p class=\"muted\">依頼（2026-10-21）: 『現行の複勝確率は、人気馬の来ないを過小に見積もっている（予測 4〜9% の馬が実績 15〜17%）。この差分は期待値計算の大きな障害になる。実測値と予測値との差分が大きすぎる。改善したい』。"
            "市場のオッズ・人気は<b>モデルの特徴量にはしない</b>。較正と馬券の判断にだけ使う（Benter 型ブレンドと同じ位置づけ）。2026 年はテストセット②で、較正器の選択・係数には使わず最終確認だけ。</p>\n"
            "<div class=\"note\"><b>要点.</b> ずれの原因は『モデルが市場から離れるほど自信過剰』で、実績はモデルと市場の間（食い違いの両端でモデル寄り 0.34〜0.48）にある。"
            "logit(p_cal) = a + w1×logit(p_model) + w2×logit(p_market) をレース内で合計 1/2/3 に合わせる較正（勝率の係数: モデル 0.31・市場 0.76）で、"
            "生に対し LogLoss は勝率 −0.0049・連対 −0.0073・複勝 −0.0076（2023〜25 年、2026 年は −0.0077/−0.0114/−0.0124）、人気区分別のずれ・低確率の人気馬のずれ（7.4pt→1.6〜3.0pt）・EV の信頼性（0.31→0.05〜0.08）・組合せ券の確率が改善する。"
            "<b>ただし</b> 個々の馬の見分け（人気馬の低確率層は較正後も約 16% 来る）は直らず、2026 年は勝率・連対でモデルが市場に追加情報を持たない。頑健に 100% 超と言える馬券は見つからない。"
            "導入はフラグ（KEIBA_PROB_CALIBRATION=1、既定オフ）つきの提案まで。</div>\n")
    html = head + "\n".join(s) + "\n</body></html>\n"
    Path(a.out).write_text(html, encoding="utf-8")
    print("wrote", a.out, len(html))


if __name__ == "__main__":
    main()
