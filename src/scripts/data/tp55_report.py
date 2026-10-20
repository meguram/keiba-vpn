"""tp-55 の結果を docs の 1 章（lap_forecast.html）にする。

python -m src.scripts.data.tp55_report --dir <結果ディレクトリ（main_*.json / ablate_2024.json / final2026_*.json / win_eval.json）> [--dir-v1 <特徴量 v1 の結果>]
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/lap_forecast.html"


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _main(d: Path, scen: str, tag: str, year: int):
    return _j(d / f"main_{scen}_{tag}_{year}.json")


def cls_tbl(d: Path) -> str:
    rows = []
    for scen, tag, name in (("A", "full", "A（スロー/ミドル/ハイ）全特徴量"), ("A", "ten", "A テン1F なし"), ("B5", "full", "B（形状 5 クラスタ）")):
        for y in (2022, 2023, 2024, 2025):
            r = _main(d, scen, tag, y)
            if not r:
                continue
            c = r["classify"]
            bp = r.get("baseline_pressure") or {}
            rows.append([name, y, f"{r['n_ok']:,}", _f(c["logloss"]), _f(c["logloss_prior"]), f"{(c['logloss'] / c['logloss_prior'] - 1) * 100:+.1f}%",
                         _f(c["brier"]), _f(c["brier_prior"]), _f(c["acc"], 2), _f(sum(c["ece"]) / len(c["ece"])),
                         _f(bp.get("logloss")) if bp else "-"])
    return r29._tbl(["シナリオ定義", "評価年", "レース数", "log loss", "事前分布", "改善率", "Brier", "事前分布", "正解率", "ECE", "先行頭数だけのロジット(log loss)"], rows,
                    ["s", "n", "n", "n", "n", "n", "n", "n", "n", "n", "n"])


def agg_tbl(d: Path) -> str:
    rows = []
    for scen, tag, name in (("A", "full", "A 全特徴量"), ("B5", "full", "B（5 クラスタ）")):
        for y in (2022, 2023, 2024, 2025):
            r = _main(d, scen, tag, y)
            if not r:
                continue
            for k, jp in (("first3f", "前半3F"), ("last3f", "後半3F")):
                a = r["agg"][k]
                rows.append([name, y, jp, _f(a["par_mae"]), _f(a["direct_mae"]), _f(a["mixture_mae"]), _f(a["oracle_mae"]), _f(a["direct_r2"], 2), _f(a["mixture_r2"], 2)])
    return r29._tbl(["シナリオ定義", "評価年", "対象", "par の誤差(秒)", "直接回帰", "混合の期待値", "正解シナリオ既知(上限)", "R² 直接", "R² 混合"], rows,
                    ["s", "n", "s", "n", "n", "n", "n", "n", "n"])


def lap_tbl(d: Path, year: int = 2024) -> str:
    r = _main(d, "A", "full", year)
    if not r:
        return ""
    rows = []
    for j, v in sorted(r["mae_by_lap"].items(), key=lambda kv: int(kv[0])):
        rows.append([f"{int(j) + 1}F 目" + ("（テン）" if j == "0" else ""), f"{v['n']:,}", _f(v["par"]), _f(v["direct"]), _f(v["mixture"]), _f(v["oracle"])])
    return r29._tbl(["ラップ（スタートから）", "レース数", "par の誤差(秒)", "直接回帰", "混合の期待値", "正解シナリオ既知"], rows, ["s", "n", "n", "n", "n", "n"])


def dens_tbl(d: Path) -> str:
    rows = []
    for y in (2022, 2023, 2024, 2025):
        r = _main(d, "A", "full", y)
        if not r:
            continue
        de, cv = r["density"], r["coverage"]
        rows.append([y, _f(de["loglik_per_lap_par"]), _f(de["loglik_per_lap_direct"]), _f(de["loglik_per_lap_mixture"]),
                     _f(cv["lap0"]["coverage80"], 2), _f(cv["lap1"]["coverage80"], 2), _f(cv["first3f"]["coverage80"], 2), _f(cv["last3f"]["coverage80"], 2)])
    return r29._tbl(["評価年", "par(単一ガウス)", "直接回帰(単一ガウス)", "混合(提案)", "80%帯のカバー率 テン1F", "2F 目", "前半3F", "後半3F"], rows,
                    ["n", "n", "n", "n", "n", "n", "n", "n"])


def rel_tbl(d: Path, year: int = 2024) -> str:
    r = _main(d, "A", "full", year)
    if not r:
        return ""
    names = ["スロー", "ミドル", "ハイ"]
    rows = []
    for k, bins in enumerate(r["classify"]["reliability"]):
        for b in bins:
            rows.append([names[k], _f(b["pred"], 2), _f(b["obs"], 2), f"{b['n']:,}"])
    return r29._tbl(["傾向", "予測確率(帯の平均)", "実際の発生率", "レース数"], rows, ["s", "n", "n", "n"])


def ablate_tbl(d: Path) -> str:
    a = _j(d / "ablate_2024.json")
    if not a:
        return ""
    res = a["results"]
    full = res["full"]
    jp = {"full": "全特徴量", "style": "脚質の分布を除く", "gate": "枠並びを除く", "ten": "テン1F を除く", "jockey": "騎手の積極性を除く", "sameday": "同日の先のレースのラップを除く",
          "field": "メンバー構成を除く", "style+gate+ten+jockey": "脚質・枠並び・テン1F・騎手をすべて除く", "course": "コース・距離などを除く", "cond": "馬場・天候・クラスを除く",
          "geo": "コース形状を除く", "recent": "コース別の直近のラップ傾向を除く"}
    rows = []
    for k, v in res.items():
        rows.append([jp.get(k, k), _f(v["logloss"]), _f(v["logloss"] - full["logloss"], 4, True), _f(v["lap0_mae"], 4), _f(v["lap0_mae"] - full["lap0_mae"], 4, True),
                     _f(v["first3f_mae"], 4), _f(v["first3f_mae"] - full["first3f_mae"], 4, True), _f(v["last3f_mae"], 4), _f(v["last3f_mae"] - full["last3f_mae"], 4, True)])
    return r29._tbl(["抜いた特徴量", "シナリオ log loss", "差", "テン1F 誤差", "差", "前半3F 誤差", "差", "後半3F 誤差", "差"], rows, ["s", "n", "n", "n", "n", "n", "n", "n", "n"])


def seg_tbl(d: Path, year: int = 2024) -> str:
    r = _main(d, "A", "full", year)
    if not r:
        return ""
    rows = [[k, f"{v['n']:,}", _f(v["f3_par_mae"]), _f(v["f3_mae"]), _f(v["l3_par_mae"]), _f(v["l3_mae"])] for k, v in r["segments"].items()]
    return r29._tbl(["区分", "レース数", "前半3F par", "前半3F 混合", "後半3F par", "後半3F 混合"], rows, ["s", "n", "n", "n", "n", "n"])


def final_tbl(d: Path) -> str:
    rows = []
    for lab, name in (("final2026_A_noten", "A テン1F なし（2026 年用）"), ("final2026_A_full", "A テン1F あり（2026 年は表が無く欠損）"), ("final2026_B_noten", "B（5 クラスタ）テン1F なし")):
        r = _j(d / f"{lab}.json")
        if not r:
            continue
        c, a, de, cv = r["classify"], r["agg"], r["density"], r["coverage"]
        rows.append([name, f"{r['n_ok']:,}", _f(c["logloss"]), _f(c["logloss_prior"]), _f(c["acc"], 2), _f(a["first3f"]["par_mae"]), _f(a["first3f"]["mixture_mae"]),
                     _f(a["last3f"]["par_mae"]), _f(a["last3f"]["mixture_mae"]), _f(de["loglik_per_lap_mixture"]), _f(cv["first3f"]["coverage80"], 2),
                     _f(cv["last3f"]["coverage80"], 2)])
    return r29._tbl(["モデル", "レース数", "log loss", "事前分布", "正解率", "前半3F par", "混合", "後半3F par", "混合", "対数尤度/ラップ", "80%帯 前半3F", "後半3F"], rows,
                    ["s", "n", "n", "n", "n", "n", "n", "n", "n", "n", "n", "n"])


def version_tbl(dirs: dict) -> str:
    rows = []
    for y in (2022, 2023, 2024, 2025):
        cells = [y]
        for name, dd in dirs.items():
            r = _main(dd, "A", "full", y) if dd else None
            if r:
                cells += [_f(r["classify"]["logloss"], 4), _f(r["agg"]["first3f"]["mixture_mae"], 4), _f(r["agg"]["last3f"]["mixture_mae"], 4)]
            else:
                cells += ["-", "-", "-"]
        rows.append(cells)
    head = ["評価年"]
    for name in dirs:
        head += [f"{name} log loss", "前半3F 誤差", "後半3F 誤差"]
    return r29._tbl(head, rows, ["n"] * len(head))


def cluster_tbl(d: Path) -> str:
    r = None
    for lab in ("final2026_B_noten",):
        r = _j(d / f"{lab}.json")
    if not r or "cluster_centers" not in r:
        return ""
    cc = r["cluster_centers"]
    nm = ["形状1（最もスロー寄り）", "形状2", "形状3", "形状4", "形状5（最もハイ寄り）"]
    rows = [[nm[i], _f(c[0], 2, True), _f(c[1], 2, True), _f(c[2], 2, True), f"{100 * cc['share_train'][i]:.1f}%"] for i, c in enumerate(cc["centers_f3_mid_l3_dev"])]
    return r29._tbl(["クラスタ", "前半3F の par との差(秒)", "中盤の平均(秒)", "後半3F の par との差(秒)", "学習データでの割合"], rows, ["s", "n", "n", "n", "n"])


def example_block(d: Path) -> str:
    ex = _j(d / "examples_2026.json")
    if not ex:
        return ""
    out = []
    for e in ex:
        s = e["summary"]
        m = s["n_laps"]
        head = ["ラップ"] + [f"{i + 1}F" for i in range(m)] + ["前半3F", "後半3F"]
        rows = []
        for sc in s["scenarios"]:
            rows.append([f"{sc['name']}（{100 * sc['prob']:.0f}%）"] + [_f(x, 1) for x in sc["laps"]] + [_f(sc["first3f"], 1), _f(sc["last3f"], 1)])
        rows.append(["期待ラップ"] + [_f(x, 1) for x in s["expected_laps"]] + [_f(s["expected_first3f"], 1), _f(s["expected_last3f"], 1)])
        rows.append(["80%帯"] + [(f"{_f(a, 1)}〜{_f(b, 1)}" if a is not None else "-") for a, b in zip(s["band80_lo"], s["band80_hi"])] + ["", ""])
        al = e["actual_laps"]
        rows.append(["実際のラップ"] + [_f(x, 1) for x in al] + [_f(sum(al[:3]), 1), _f(sum(al[-3:]), 1)])
        out.append(f"<h3>{html.escape(e['venue'])} {html.escape(e['surface'])}{e['distance']}m（{e['date']}、{e['race_id']}、{e['field']} 頭）</h3>"
                   + r29._tbl(head, rows, ["s"] + ["n"] * (m + 2)))
    return "".join(out)


def win_block(d: Path) -> str:
    w = _j(d / "win_eval.json")
    if not w:
        return "<p class='muted'>（勝率予測への増分は評価の完了待ち）</p>"
    rows = []
    for lab, v in w.items():
        c = v["overall"]
        by = v["by_year"]
        rows.append([html.escape(lab), _f(c["Δwin"], 5, True), _f(c["win_hi"], 5, True), _f(c["Δtop2"], 5, True), _f(c["Δtop3"], 5, True), "採用" if c.get("採用") else "不採用",
                     " / ".join(f"{y}: {_f(b['Δwin'], 5, True)}" for y, b in by.items())])
    return r29._tbl(["比較", "Δ勝率", "95%上限", "Δ連対", "Δ複勝", "採否", "年別 Δ勝率"], rows, ["s", "n", "n", "n", "n", "s", "s"])


def build(d: Path, d1: Path | None, d2: Path | None = None) -> None:
    a24 = _main(d, "A", "full", 2024)
    b24 = _main(d, "B5", "full", 2024)
    t24 = _main(d, "A", "ten", 2024)
    if not a24:
        raise SystemExit("main_A_full_2024.json が無い")
    ca, ab = a24["classify"], a24["agg"]
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    f26 = _j(d / "final2026_A_noten.json")
    f26_v2 = _j(d2 / "final2026_A_noten.json") if d2 else None
    f26txt = ""
    if f26:
        f26txt = (f"<b>2026 年</b>（未使用。学習は 2020〜2025 年。2026 年は SmartRC の表が無いのでテン1F なしのモデル）: シナリオの log loss {_f(f26['classify']['logloss'])}"
                  f"（事前分布 {_f(f26['classify']['logloss_prior'])}）、前半3F の誤差 {_f(f26['agg']['first3f']['mixture_mae'])} 秒（par {_f(f26['agg']['first3f']['par_mae'])}）、"
                  f"後半3F の誤差 {_f(f26['agg']['last3f']['mixture_mae'])} 秒（par {_f(f26['agg']['last3f']['par_mae'])}）。"
                  + (f"なお 2026 年の 1 回目の評価は特徴量 v2（コース別の直近の傾向を足す前）で行い（log loss {_f(f26_v2['classify']['logloss'])}、前半3F {_f(f26_v2['agg']['first3f']['mixture_mae'])} 秒）、"
                     f"v3 に決めたあとの評価はこの 2 回目。" if f26_v2 else ""))
    v1txt = "特徴量の版の比較は第 4 節。"
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>ハロンごとのラップ予想（ラップ傾向別のタイムと発生確率）（tp-55）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>ハロンごとのラップ予想（ラップ傾向別のタイムと発生確率）（tp-55）</h1>
<p class="muted">ユーザー要望: メンバー構成、枠並び、脚質の分布、逃げ先行馬の直近テン1F、会場の特徴などをふんだんに使って、全レースで 1F ごとのラップ予想を出したい。ミドル・スロー・ハイでラップの傾向が変わるので、傾向ごとにハロンごとのタイムを予測し、それぞれの傾向がどのくらいの確率で起きそうかという粒度のモデルにしたい。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>『傾向（シナリオ）の発生確率』＋『傾向ごとのハロン別タイム』の混合として、全レースの予想を出せるようにした.</b>シナリオはコース別の基準（par）に対する『前半3F と後半3F の差』の分位でスロー/ミドル/ハイ（25%・50%・25%）と定義。評価年 2022〜2025 年（学習は評価年より前）でシナリオの log loss は事前分布の {_f(ca['logloss_prior'])} から {_f(ca['logloss'])}（{(ca['logloss'] / ca['logloss_prior'] - 1) * 100:+.1f}%）、確率の較正は良好（ECE {_f(sum(ca['ece']) / len(ca['ece']))}）。</li>
 <li><b>確率の分布の当たりが、単一の予測より大きく良くなる.</b>ハロンあたりの対数尤度は、par を中心とした単一のガウス {_f(a24['density']['loglik_per_lap_par'], 2)}、直接回帰の単一ガウス {_f(a24['density']['loglik_per_lap_direct'], 2)} に対し、混合は {_f(a24['density']['loglik_per_lap_mixture'], 2)}（2024 年）。80% 帯のカバー率は前半3F {_f(a24['coverage']['first3f']['coverage80'], 2)}、後半3F {_f(a24['coverage']['last3f']['coverage80'], 2)}（目標 0.80）。</li>
 <li><b>期待ラップの点の精度は、直接回帰と同じ水準.</b>前半3F の誤差は par の {_f(ab['first3f']['par_mae'])} 秒から {_f(ab['first3f']['mixture_mae'])} 秒、後半3F は {_f(ab['last3f']['par_mae'])} 秒から {_f(ab['last3f']['mixture_mae'])} 秒（R² 約 0.3）。正解のシナリオが分かれば {_f(ab['first3f']['oracle_mae'])}／{_f(ab['last3f']['oracle_mae'])} 秒まで縮むので、残りの誤差の大半は『どの傾向になるか』が事前には決まらないことによる。</li>
 <li><b>効く特徴量はコース・脚質の分布・同日の先のレースのラップ・馬場など。枠並びとテン1F は単独ではほとんど効かない.</b>（第 4 節）{v1txt}</li>
 <li>{f26txt}</li>
 </ul></div>
<h2>1. 出力の形</h2>
<p>各レースについて、(a) ラップ傾向（シナリオ）の発生確率 P(k)、(b) 各シナリオ k のときのハロン（1F）ごとの予想タイム（スタートから。最初の区間が『テン1F』）、(c) 確率で重み付けした期待ラップと 80% 帯。1700m のように 200m で割り切れない距離は、最初の区間が短い（100m など）。API の <code>lap_forecast</code> に載る形は第 8 節。</p>
<h2>2. データと先読みの安全性</h2>
<ul>
 <li><b>実測のラップ</b>: race_result_flat のハロンごとのラップ（2020〜2026 年、芝・ダート。障害は除く）。ラップを読めて区間数が距離と合うレースだけ（全体の約 97%）。</li>
 <li><b>脚質・位置取り</b>: 過去走の最初のコーナーの通過順位を（順位−1）/（頭数−1）にし、直近 5 走を 0.35・0.25・0.20・0.12・0.08 で重み付け（position_forecast と同じ定義）。<b>その走より前の走だけ</b>を使う（日付が厳密に前）。</li>
 <li><b>テン1F</b>: SmartRC の馬ごとの『前走のスタート1F』（h1_ten1f）と『最近の最速』（ten1f_best）。smartrc_race_flat（2020〜2025 年）から。2026 年の表は無い（欠損）。<b>先読みの検査</b>: h1_ten1f は本走自身のレース最初のラップと一致する割合が 3.0%（偶然水準）で、本走の値ではなく前走までの値。前走で逃げた馬では前走の最初のラップと中央値の差 0.0 秒で一致する。データの出どころの経緯（DEC-024 でスタートデータを一度除外）を踏まえ、ここでは『前走までの値を日付で切る』使い方に限り、2026 年のようにテン1F が無いレースには、テン1F を使わないモデルで予測する。</li>
 <li><b>同日の傾向</b>: 同じ日・同じ会場でラウンドが前のレース（同じ面）のラップの par との差。ラウンド順で前のレースだけ。</li>
 <li><b>完全一致の検査</b>: 2024-07-01 以降の結果をすべて消して特徴量を作り直しても、それ以前の 15,053 レースの特徴量は全列で完全に一致した（先読みなし）。</li>
 <li><b>使わないもの</b>: 市場オッズ・人気は使わない。</li>
</ul>
<h2>3. ラップ傾向（シナリオ）の定義と予測しやすさ</h2>
<ul>
 <li><b>A（既定）</b>: コース別の par（過去の同コースの平均ラップ。件数が少ないコースは距離×面の平均へ縮約）との差で、『後半3F の差 − 前半3F の差』が学習データの下位 25% ならスロー、上位 25% ならハイ、それ以外はミドル。直感的で、予測も当たりやすい。</li>
 <li><b>B</b>: 前半3F・中盤・後半3F の par との差を k-means（5 クラスタ）でまとめ、スロー寄りからハイ寄りに並べる。形の違いは細かいが、傾向の当てやすさは下がる（正解率 0.47〜0.49）。</li>
</ul>
{cluster_tbl(d)}
{cls_tbl(d)}
<p class="muted">A の事前分布は 25%・50%・25% なので、『常にミドル』の正解率は約 0.50。正解率の改善は小さいが、log loss と Brier は 9〜10% 良くなり、確率の信頼度は目標どおり（下表）。『先行頭数だけのロジット』は、逃げ・先行の頭数や圧力だけで傾向を決める簡易版（position_forecast の発想）。</p>
<h3>確率の較正（A、2024 年）</h3>
{rel_tbl(d)}
<h2>4. 特徴量の効き（2024 年、抜き取り）</h2>
<p class="muted">学習は 2020〜2023 年。シナリオ A の分類器と、テン1F・前半3F・後半3F の直接回帰だけの軽い版で、特徴量のグループを抜いたときの差（正の差=悪化）。</p>
{ablate_tbl(d)}
<h3>特徴量の版の比較（シナリオ A、全特徴量）</h3>
<p class="muted">v1 = 脚質・枠並び・テン1F・同日・コース条件。v2 = v1 にコース形状（初角までの距離・直線・起伏など 7 列）とテン1F の面・距離別の標準化を足したもの。v3（既定）= v2 に、同じコース（直近 30 レース）・同じ会場×面（直近 60 レース）のラップ傾向を足したもの。</p>
{version_tbl({'v1': d1, 'v2': d2, 'v3': d})}
<ul>
 <li>v1→v2 はほぼ同じ（コース形状はコースの種類で既に表せている）。v3 は log loss が 2023〜2025 年で 0.002 前後良く、後半3F の誤差が 2024・2025 年で 0.005〜0.006 秒縮んだ（2022 年は学習が 2 年と少なく、log loss はわずかに悪化）。差は小さいが、後半3F に効く傾向が一貫するので v3 を既定にした。</li>
 <li>最も効くのはコース・距離などの条件、次いで脚質の分布（逃げ・先行の頭数、最前の馬の位置取り）。同日の先のレースのラップは後半3F の誤差に大きく効く（馬場や当日の傾向の代理）。</li>
 <li>枠並びとテン1F は、単独ではほとんど効かない。枠並びの情報は、脚質の分布と中身が重なる。テン1F（SmartRC の前走までのスタート1F）は、前走の距離やコースが今回と違うため、脚質の分布が決まれば追加の情報が小さい。</li>
</ul>
<h2>5. 精度（期待ラップと分布）</h2>
{agg_tbl(d)}
<h3>ラップ位置別の誤差（A、2024 年、秒）</h3>
{lap_tbl(d)}
<h3>距離帯・面別の 3F の誤差（A、2024 年、秒）</h3>
{seg_tbl(d)}
<h3>分布の当たり（ラップあたりの対数尤度。大きいほど良い）と 80% 帯のカバー率</h3>
{dens_tbl(d)}
<h2>6. 2026 年（未使用、1 回のみ）</h2>
{final_tbl(d)}
<h2>7. 全レースで出せるか・応用</h2>
<ul>
 <li><b>被覆</b>: ラップが分かる全レースで予想を出せる（評価の被覆率 100%）。出馬表の段階で、過去走が無い馬が多いレースは、脚質の特徴が『不明』の割合（unknown_ratio）として扱われる。テン1F が無いレース（2026 年・取得前）は、テン1F を使わないモデルで出す（評価では差はほぼ無い）。</li>
 <li><b>追走難度（tp-54）への土台</b>: 想定ラップ（期待ラップ・シナリオ別ラップ）を、『想定位置取りを取るのに必要な通過ラップ』の要求の土台として使える。形は第 8 節の <code>lap_forecast</code>。</li>
 <li><b>整合層（pace_coherence）との関係</b>: 整合層のペースは先行勢の頭数から推定する簡易なもの。ここの確率つきシナリオは、先行頭数だけのロジットより log loss が良く（第 3 節の最右列）、置き換え候補。ただし画面での置き換えは UI 担当の判断。</li>
 <li><b>勝率予測への増分</b>（V5 の列セット d_CY_BEST に、ラップ予想の特徴量 12 列を足す。valid 2022〜2024 年、既定の採否基準）:</li>
</ul>
{win_block(d)}
<h2>8. API・フラグ・運用</h2>
<ul>
 <li>予測 API（<code>/api/race/{{id}}/predictions</code>）に、環境変数 <code>KEIBA_LAP_FORECAST=1</code> のときだけ <code>lap_forecast</code> を付ける（既定は無効）。保存済みの予想を読み出すだけ（<code>src/pipeline/inference/lap_forecast_service.py</code>）。</li>
 <li>保存: <code>python -m src.scripts.data.tp55_backfill --dates YYYY-MM-DD ...</code>（出馬表が書き出された日の全レース。過去分のバックフィルも同じ）。保存先 <code>data/local/inference/lap_forecast/&lt;日付&gt;/&lt;race_id&gt;.json</code>。モデルは final2026 の A（テン1F あり／なし）。</li>
 <li><code>lap_forecast</code>: <code>version, race_id, date, distance, n_laps, lap_meters, model(A_full|A_noten), pace_label, pace_prob, scenarios[{{name, prob, laps, sd, first3f, last3f}}], expected_laps, band80_lo, band80_hi, expected_first3f, expected_last3f, par</code>。laps はスタートから 1 区間ずつの秒。</li>
 <li>本番へは未反映。T-60 の確定出走表（枠順確定後）で動かす。出馬表の書き出し（export_tables の差分）が前提で、書き出し前は暫定にならない（保存なし＝付かない）。</li>
</ul>
<h2>9. 限界と今後</h2>
<ul>
 <li>ラップ傾向の予測は、正解のシナリオが分かれば誤差が約 3 割縮むが、事前にはあいまい（正解率は『常にミドル』並み）。確率で持つ意味はここにある。</li>
 <li>特徴量は、馬の過去走の脚質とコースが中心。騎手の意図や当日の馬場の直前変化は入らない。2026 年のテン1F は取得できていない（SmartRC の 2026 年の表が無い）。</li>
 <li>勝率予測への応用は、第 7 節の増分の結果に従う。ペース予測は tp-25b でも不採用だった。</li>
</ul>
<p class="muted">コード: <code>src/pipeline/features/tp55_lapfeat.py</code>、<code>src/scripts/data/tp55_build.py</code>（表の作成）、<code>src/pipeline/models/lap_forecast.py</code>（モデル）、<code>tp55_eval.py</code>（評価）、<code>tp55_win_eval.py</code>（勝率への増分）、<code>tp55_backfill.py</code>、<code>tp55_report.py</code>。結果の生データ: ホスト <code>/home/jovyan/work/tp55_out3/</code>。</p>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="特徴量 v3（既定）の結果")
    ap.add_argument("--dir-v1", default=None)
    ap.add_argument("--dir-v2", default=None)
    a = ap.parse_args()
    build(Path(a.dir), Path(a.dir_v1) if a.dir_v1 else None, Path(a.dir_v2) if a.dir_v2 else None)


if __name__ == "__main__":
    main()
