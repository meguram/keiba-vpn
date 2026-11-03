"""めぐコラム『距離で騎手の差は変わるか』を実データから集計し、frontend/content/columns/distance-jockey.json を書く。

問い: 距離が長いほど騎手の影響が大きい（騎手で差がつく）か。

物差し: 距離帯ごとの「騎手間の本当のばらつき」（標準偏差）。騎手ごとの率のばらつきから、騎乗数が有限なことによる偶然のばらつき
  （率の分散 / 騎乗数 の平均）を差し引いたもの（ランダム効果のモーメント推定。騎乗数で重み付け）。その距離帯で 50 騎乗以上の騎手。
  - 勝率・複勝率そのもの: 騎手の腕と、乗る馬の強さ（上位騎手ほど強い馬に乗る）の両方を含む。
  - 実際 − 人気から見込まれる率（jockey_common の res_win・res_top3）: 市場が織り込んだ以上の騎手の差。
  95% 区間は騎手を単位にしたブートストラップ（B 回）。距離帯の差（例: 長距離 − 短距離）も同じブートストラップで区間を出す。
以前の版は、生の標準偏差（偶然のばらつきを含む）と、同じデータから選んだ上位 10 名の勝率で比べていた。

比較グラフ「距離帯別 騎手勝率の格差（上位10名 vs それ以外の騎手）」は compare_series が書くので、既存の JSON から引き継ぐ。

使い方:
    python -m src.research.columns.distance_jockey [--boot 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.jockey_common import frame
from src.research.columns.wind_venue import pct

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/distance-jockey.json"
KEEP_CHART = "距離帯別 騎手勝率の格差（上位10名 vs それ以外の騎手）"
BANDS = ["短距離", "マイル", "中距離", "長距離"]
XS = ["短距離\n〜1400m", "マイル\n1401〜1800m", "中距離\n1801〜2200m", "長距離\n2201m〜"]
METRICS = {"win": "勝率", "top3": "複勝率", "res_win": "勝率（人気の見込みとの差）", "res_top3": "複勝率（人気の見込みとの差）"}
MIN_N = 50


def true_sd(g: pd.DataFrame) -> float:
    """g: 騎手ごとの mean・var・size。偶然のばらつきを差し引いた騎手間の標準偏差。"""
    w = g["size"].to_numpy(float)
    m = np.average(g["mean"], weights=w)
    obs = np.average((g["mean"] - m) ** 2, weights=w)
    noise = np.average(g["var"] / g["size"], weights=w)
    return float(np.sqrt(max(obs - noise, 0.0)))


def analyze(d: pd.DataFrame, boot: int, seed: int = 0) -> dict[str, Any]:
    d = d.copy()
    d["db"] = d["dist_band"].astype(str).str.split("\n").str[0]
    rng = np.random.default_rng(seed)
    a: dict[str, Any] = {"est": {}, "ci": {}, "diff": {}, "n_j": {}, "n_runs": {}, "base": {}}
    stats = {}
    for b in BANDS:
        x = d[d["db"] == b]
        g = x.groupby("jockey_id")[list(METRICS)].agg(["mean", "var", "size"])
        keep = g[("win", "size")] >= MIN_N
        stats[b] = {m: g.loc[keep, m] for m in METRICS}
        a["n_j"][b], a["n_runs"][b], a["base"][b] = int(keep.sum()), int(len(x)), {m: float(x[m].mean()) for m in ("win", "top3")}
        a["est"][b] = {m: true_sd(stats[b][m]) for m in METRICS}
    draws = {b: {m: [] for m in METRICS} for b in BANDS}
    for _ in range(boot):
        for b in BANDS:
            k = len(stats[b]["win"])
            idx = rng.integers(0, k, k)
            for m in METRICS:
                draws[b][m].append(true_sd(stats[b][m].iloc[idx]))
    for b in BANDS:
        a["ci"][b] = {m: [float(np.percentile(draws[b][m], 2.5)), float(np.percentile(draws[b][m], 97.5))] for m in METRICS}
    for b in BANDS[1:]:
        for m in METRICS:
            dd = np.array(draws[b][m]) - np.array(draws["短距離"][m])
            a["diff"][(b, m)] = (a["est"][b][m] - a["est"]["短距離"][m], float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
    dd = np.array(draws["長距離"]["win"]) - np.array(draws["マイル"]["win"])
    a["long_vs_mile"] = (a["est"]["長距離"]["win"] - a["est"]["マイル"]["win"], float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
    a["boot"] = boot

    def chart(ms: list[str], title: str, caption: str) -> dict[str, Any]:
        return {"type": "chart", "title": title, "xLabel": "距離帯", "yLabel": "騎手間の標準偏差", "xs": XS,
                "series": [{"name": METRICS[m], "values": [round(a["est"][b][m], 4) for b in BANDS], "n": [a["n_j"][b] for b in BANDS],
                            "ci": [[round(a["ci"][b][m][0], 4), round(a["ci"][b][m][1], 4)] for b in BANDS]} for m in ms],
                "caption": caption}
    a["c_raw"] = chart(["win", "top3"], "距離帯別 騎手間の勝率・複勝率のばらつき（偶然の分を除く）",
                       f"その距離帯で {MIN_N} 騎乗以上の騎手（n は騎手の人数）。乗る馬の強さの差も含む。縦線は騎手単位のブートストラップ（{boot} 回）による 95% 区間。")
    a["c_res"] = chart(["res_win", "res_top3"], "距離帯別 人気の見込みを超える騎手の差（偶然の分を除く）",
                       "実際の率 − 人気（単勝オッズ）と頭数から見込まれる率 の、騎手間の標準偏差。市場が織り込んだ以上の騎手の差。縦線は 95% 区間。")
    return a


def article(a: dict[str, Any], keep: dict[str, Any] | None) -> dict[str, Any]:
    E, C, D = a["est"], a["ci"], a["diff"]

    def pt(v: float, k: int = 1) -> str:
        return f"{v * 100:.{k}f}pt"

    def rng_(m: str, b: str) -> str:
        return f"{pt(C[b][m][0])}〜{pt(C[b][m][1])}"

    rows = [[b, f"{a['n_j'][b]}", f"{a['n_runs'][b]:,}", pct(a["base"][b]["win"]), f"{pt(E[b]['win'])}（{rng_('win', b)}）",
             f"{E[b]['win'] / a['base'][b]['win']:.2f}", f"{pt(E[b]['res_win'], 2)}（{rng_('res_win', b)}）"] for b in BANDS]
    lm = a["long_vs_mile"]
    sig_short = [b for b in BANDS[1:] if D[(b, "win")][1] > 0]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "騎手間の勝率の差（短距離）", "value": pt(E["短距離"]["win"]), "sub": f"標準偏差。95%区間 {rng_('win', '短距離')}"},
            {"label": "騎手間の勝率の差（長距離）", "value": pt(E["長距離"]["win"]), "sub": f"マイル {pt(E['マイル']['win'])}・中距離 {pt(E['中距離']['win'])}"},
            {"label": "人気の見込みを超える騎手の差", "value": f"{pt(min(E[b]['res_win'] for b in BANDS), 2)}〜{pt(max(E[b]['res_win'] for b in BANDS), 2)}",
             "sub": "勝率。どの距離でもごく小さい"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            "「距離が長いほど騎手の影響が大きい」は、はっきりとは言えません。\n\n"
            f"騎乗数が限られることによる偶然のばらつきを除いた、騎手間の勝率の差（標準偏差）は、短距離 {pt(E['短距離']['win'])}・マイル {pt(E['マイル']['win'])}・"
            f"中距離 {pt(E['中距離']['win'])}・長距離 {pt(E['長距離']['win'])} です。"
            + (f"短距離より大きいと言える（95% 区間が 0 をまたがない）のは {'・'.join(sig_short)} で、" if sig_short else "短距離との差も、95% 区間がどれも 0 をまたぎ、")
            + f"長距離とマイルの差は {pt(lm[0])}（{pt(lm[1])}〜{pt(lm[2])}）と区別できません。"
            "以前の版の「マイルが最も差がつく」も、偶然の範囲です。\n\n"
            f"この差には、騎手の腕と、上位騎手ほど強い馬に乗ることの両方が含まれ、このデータでは分けられません。人気（オッズ）から見込まれる勝率との差で見ると、騎手間の差はどの距離でも "
            f"{pt(min(E[b]['res_win'] for b in BANDS), 2)}〜{pt(max(E[b]['res_win'] for b in BANDS), 2)} とごく小さく、95% 区間の下限は 0 です（差が無いことと区別できない）。"
            "騎手の差は距離を問わず、ほぼオッズに織り込まれています。")},
        a["c_raw"],
        {"type": "table", "columns": ["距離帯", "騎手数", "騎乗数", "全体の勝率", "騎手間の勝率の差（95%区間）", "差 ÷ 全体の勝率", "人気の見込みを超える差（95%区間）"],
         "rows": rows, "caption": "距離が長いほど頭数が少なく全体の勝率が高いので、差を全体の勝率で割った値（相対的なばらつき）も示す。"},
        a["c_res"],
    ]
    if keep:
        body += [{"type": "heading", "text": "上位 10 名とそれ以外（以前の版のグラフ）"}, keep,
                 {"type": "text", "text": "このグラフは、距離帯ごとに勝率の上位 10 名を同じデータから選んでいるので、偶然よく勝った騎手も上位に入り、差が大きめに出ます。"
                                          "また上位騎手は強い馬に乗るので、騎手の腕だけの差ではありません。距離での違いは、上の偶然の分を除いた比較で見てください。"}]
    body += [
        {"type": "heading", "text": "馬券への活用"},
        {"type": "text", "text": (
            "距離によって騎手の重みを変える根拠は、データからは見つかりません。騎手の差は距離を問わずオッズにほぼ織り込まれているので、"
            "「長距離だから上位騎手を重視」「短距離だから乗り替わりを気にしない」といった調整は不要です。")},
        {"type": "accordion", "title": "データと方法・以前の版からの変更", "blocks": [
            {"type": "text", "text": (
                "・対象: 2020〜2025 年の JRA 平地（障害・取消・除外を除く）。距離帯は 〜1400m・1401〜1800m・1801〜2200m・2201m〜。\n\n"
                "・騎手間の差は、騎手ごとの率の（騎乗数で重み付けした）ばらつきから、騎乗数が有限なことによる偶然のばらつきを差し引いて求めました。"
                "騎乗数が少ない長距離ほど、生のばらつきは偶然で大きく見えます。\n\n"
                "・以前の版は生の標準偏差で比べていたため、偶然のばらつきが混ざっていました。"
                "「短距離はゲート技術」「マイルは判断力」といった理由の説明は、このデータでは確かめていないので外しました。")},
        ]},
    ]
    return {
        "slug": "distance-jockey",
        "title": "距離で騎手の差は変わるか——長距離で大きいとは言えず、ほぼオッズどおり",
        "claim": "距離が長いほど騎手の影響が大きい",
        "verdict": "inconclusive",
        "summary": (f"偶然の分を除いた騎手間の勝率の差は、短距離 {pt(E['短距離']['win'])}・マイル {pt(E['マイル']['win'])}・中距離 {pt(E['中距離']['win'])}・長距離 {pt(E['長距離']['win'])}。"
                    "長距離とマイルは区別できず、人気の見込みを超える騎手の差はどの距離でもごく小さいです。"),
        "categories": ["jockey/distance"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": "2020〜2025年", "conditions": f"JRA 平地（障害を除く）。その距離帯で {MIN_N} 騎乗以上の騎手",
                    "sample": f"{sum(a['n_runs'].values()):,} 出走"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    old = json.loads(OUT.read_text(encoding="utf-8"))
    keep = next((b for b in old["body"] if b.get("type") == "chart" and b.get("title") == KEEP_CHART), None)
    col = article(analyze(frame(), args.boot), keep)
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
