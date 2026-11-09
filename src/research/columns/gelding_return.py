"""めぐコラム『去勢明けは買い？』を実データから集計し、frontend/content/columns/gelding-return.json を書く。

問い: 去勢明けの初戦・2 戦目…で、勝率・複勝率はどう推移するか。去勢前より走るようになるか。市場（オッズ）は織り込んでいるか。

データ: compare_series.load()（2016〜2025 年の JRA 平地。取消・除外を除く）。
定義（compare_series の『去勢明け初戦』と同じ）:
  去勢明け初戦 = 前走（JRA 平地の直前の出走）は牡、今走はセ。その後のセとしての出走を 2 戦目、3 戦目…と数える。
  去勢前 = 同じ馬の、去勢明け初戦より前の出走（直前・2 走前・3 走前）。
  ※ 前走が 2015 年以前・地方・障害のときは判定できない（その馬は対象外）。2016 年以降のデータで初めて見たときに既にセの馬も対象外。
比べ方:
  勝率はオッズから見込まれる勝率（1/単勝オッズ をレース内で合計 1 に直したもの）と比べる。
  複勝率は「人気から見込まれる複勝率」（全出走での、同じ人気・同じ頭数の複勝率）と比べる。
  参考に、同じ年齢の牡馬（去勢していない馬）の平均（年齢の構成をそろえた加重平均）を点線で出す。

使い方:
    python -m src.research.columns.gelding_return            # 集計して記事を書く
    python -m src.research.columns.gelding_return --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import YEARS, _p, _r, _rci, _stat, load, pairs
from src.research.columns.wind_venue import holm, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/gelding-return.json"
STEPS = ["去勢前\n3走前", "去勢前\n2走前", "去勢前\n直前", "去勢明け\n1戦目", "2戦目", "3戦目", "4戦目", "5戦目", "6戦目〜"]
AFTER = STEPS[3:]
FILTERS = {
    "去勢した年齢": {"全体": None, "2〜3歳": lambda d: d["g_age"] <= 3, "4歳以上": lambda d: d["g_age"] >= 4},
    "対象の馬": {"全体": None, "去勢後に5戦以上走った馬": lambda d: d["n_after_total"] >= 5},
}


def frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    d = (load() if df is None else df).copy()
    d = d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    # 人気から見込まれる複勝率: 全出走での、同じ人気・同じ頭数の複勝率
    d["exp_top3"] = d.groupby(["popularity", "starters"])["top3"].transform("mean")
    ev = (d["sex"] == "セ") & (d["prev_sex"] == "牡")
    first = d.loc[ev].groupby("horse_id").agg(gdate=("date", "min"), g_age=("age", "min"))
    d = d.merge(first, left_on="horse_id", right_index=True, how="left")
    d = d[d["gdate"].notna()].copy()
    after = d["date"] >= d["gdate"]
    d["k"] = np.nan
    d.loc[after, "k"] = d[after].groupby("horse_id").cumcount() + 1                         # 去勢明け k 戦目
    d.loc[~after, "k"] = -d[~after].groupby("horse_id").cumcount(ascending=False)          # 去勢前: −1 = 直前
    d["n_after_total"] = d.groupby("horse_id")["k"].transform("max")
    lab = {-3: STEPS[0], -2: STEPS[1], -1: STEPS[2], 1: STEPS[3], 2: STEPS[4], 3: STEPS[5], 4: STEPS[6], 5: STEPS[7]}
    d["step"] = d["k"].map(lambda k: lab.get(int(k)) if k < 6 else STEPS[8])
    return d[d["step"].notna()].copy()


def colt_baseline(d: pd.DataFrame, colts: pd.DataFrame, metric: str) -> list[float | None]:
    """各段階の年齢構成で重み付けした、同じ年齢の牡馬の平均。"""
    rate = colts.groupby("age")[metric].mean()
    out = []
    for s in STEPS:
        w = d.loc[d["step"] == s, "age"].value_counts(normalize=True)
        out.append(_r(float((w * rate.reindex(w.index)).sum())) if len(w) else None)
    return out


def step_series(d: pd.DataFrame, metric: str, exp: str, colts: pd.DataFrame, names: tuple[str, str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    act, ex, ps = [], [], []
    for s in STEPS:
        v = d.loc[d["step"] == s]
        ma, na, ca, va = _stat(v[metric])
        mb, nb, cb, vb = _stat(v[exp])
        act.append((ma, na, ca)); ex.append((mb, nb, cb))
        p = _p(va, vb, ma, mb)
        ps.append(None if p is None else float(f"{p:.3g}"))
    ser = [{"name": names[0], "values": [_r(m) for m, _, _ in act], "n": [n for _, n, _ in act], "ci": [_rci(c) for _, _, c in act]},
           {"name": names[1], "values": [_r(m) for m, _, _ in ex], "n": [n for _, n, _ in ex], "ci": [_rci(c) for _, _, c in ex]},
           {"name": names[2], "values": colt_baseline(d, colts, metric), "baseline": True}]
    return ser, {"a": names[0], "b": names[1], "p": ps}


def step_chart(d: pd.DataFrame, colts: pd.DataFrame, metric: str, exp: str, names: tuple[str, str, str], title: str, ylabel: str, caption: str) -> dict[str, Any]:
    ser, cmp = step_series(d, metric, exp, colts, names)
    variants = {}
    labels = list(FILTERS)
    for o1, f1 in FILTERS[labels[0]].items():
        for o2, f2 in FILTERS[labels[1]].items():
            if f1 is None and f2 is None:
                continue
            m = pd.Series(True, index=d.index)
            for f in (f1, f2):
                if f is not None:
                    m &= f(d)
            s, c = step_series(d[m], metric, exp, colts, names)
            variants[f"{o1}|{o2}"] = {"series": s, "compare": c}
    return {"type": "chart", "title": title, "xLabel": "去勢の前後の出走", "yLabel": ylabel, "xs": STEPS, "series": ser, "compare": cmp,
            "filters": [{"label": k, "options": list(v)} for k, v in FILTERS.items()], "variants": variants, "caption": caption}


def analyze(d: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    colts = df[df["sex"] == "牡"]
    a["sex"] = {s: (df.loc[df["sex"] == s, "win"].mean(), int((df["sex"] == s).sum())) for s in ("牡", "牝", "セ")}
    a["n_horses"] = int(d["horse_id"].nunique())
    a["g_age"] = d.drop_duplicates("horse_id")["g_age"].value_counts().sort_index().to_dict()
    a["tab"] = {s: {k: d.loc[d["step"] == s, k].mean() for k in ("win", "implied", "top3", "exp_top3", "ret")} | {"n": int((d["step"] == s).sum())} for s in STEPS}
    early = d[d["k"].between(1, 5)]
    a["early"] = {k: early[k].mean() for k in ("win", "implied", "top3", "exp_top3", "ret")} | {"n": len(early)}
    _, _, _, va = _stat(early["win"]); _, _, _, vb = _stat(early["implied"])
    a["early_p_win"] = _p(va, vb, early["win"].mean(), early["implied"].mean())
    _, _, _, va = _stat(early["top3"]); _, _, _, vb = _stat(early["exp_top3"])
    a["early_p_top3"] = _p(va, vb, early["top3"].mean(), early["exp_top3"].mean())
    a["early_age"] = {}
    for lab, f in FILTERS["去勢した年齢"].items():
        if f is None:
            continue
        w = early[f(early)]
        _, _, _, va = _stat(w["top3"]); _, _, _, vb = _stat(w["exp_top3"])
        a["early_age"][lab] = {k: w[k].mean() for k in ("win", "implied", "top3", "exp_top3")} | {"n": len(w), "p": _p(va, vb, w["top3"].mean(), w["exp_top3"].mean())}
    a["c_win"] = step_chart(d, colts, "win", "implied", ("実際の勝率", "オッズから見込まれる勝率", "同じ年齢の牡馬（平均）"),
                            "去勢の前後 N 戦目の勝率（実際とオッズの見込み）", "勝率",
                            "去勢明け初戦（前走は牡・今走はセ）を 1 戦目とし、同じ馬の前後の出走を並べたもの。点線は同じ年齢構成の牡馬の平均。＊は実際と見込みの差が有意（p<0.05）。去勢した年齢・去勢後の出走数で切り替えられる。")
    a["c_top3"] = step_chart(d, colts, "top3", "exp_top3", ("実際の複勝率", "人気から見込まれる複勝率", "同じ年齢の牡馬（平均）"),
                             "去勢の前後 N 戦目の複勝率（実際と人気の見込み）", "複勝率",
                             "人気から見込まれる複勝率は、全出走での同じ人気・同じ頭数の複勝率。点線は同じ年齢構成の牡馬の平均。＊は実際と見込みの差が有意（p<0.05）。")
    pp = {p.title: p.compute(df) for p in pairs(df) if p.slug == "gelding-return"}
    a["c_age"], a["c_pop"] = pp["セン馬の年齢別勝率"], pp["去勢明け初戦 vs それ以外 × 人気帯別 複勝率"]
    return a


def article(a: dict[str, Any]) -> dict[str, Any]:
    T, E = a["tab"], a["early"]
    pre, k1, k6 = T[STEPS[2]], T[STEPS[3]], T[STEPS[8]]
    best = max(AFTER, key=lambda s: T[s]["top3"])
    win_sig = [s.replace("\n", "") for s, p in zip(STEPS, a["c_win"]["compare"]["p"]) if s in AFTER and p is not None and p < 0.05]
    after_p = [p for s, p in zip(STEPS, a["c_top3"]["compare"]["p"]) if s in AFTER]
    top_sig = [(s.replace("\n", ""), T[s]["top3"] - T[s]["exp_top3"], ph) for s, p, ph in zip(AFTER, after_p, holm(after_p)) if p < 0.05]
    EA = a["early_age"]
    sw = a["sex"]
    rows = [[s.replace("\n", " "), f"{T[s]['n']:,}", pct(T[s]["win"]), pct(T[s]["implied"]), pct(T[s]["top3"]), pct(T[s]["exp_top3"]), pct(T[s]["ret"])] for s in STEPS]
    ages = a["g_age"]
    young = sum(v for k, v in ages.items() if k <= 3)
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "去勢明け初戦の複勝率", "value": pct(k1["top3"]), "sub": f"人気からの見込み {pct(k1['exp_top3'])}。去勢前の直前は {pct(pre['top3'])}"},
            {"label": f"いちばん高い{best.replace(chr(10), '')}の複勝率", "value": pct(T[best]["top3"]), "sub": f"見込み {pct(T[best]['exp_top3'])}"},
            {"label": "去勢明け 1〜5 戦目の単勝回収率", "value": pct(E["ret"]), "sub": f"{E['n']:,} 走。勝率 {pct(E['win'])}（見込み {pct(E['implied'])}）"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"去勢明け（前走は牡・今走はセ）の馬 {a['n_horses']:,} 頭について、去勢の前後の成績を 1 走ずつ並べました。"
            f"去勢の直前の走は複勝率 {pct(pre['top3'])} で、人気から見込まれる {pct(pre['exp_top3'])} を大きく下回っています。"
            "人気に応えられなくなってきた馬が去勢されている、ということです。\n\n"
            f"去勢明け初戦は人気も落ち（見込み {pct(k1['exp_top3'])}）、複勝率は {pct(k1['top3'])} とさらに下がります。そこから 2 戦目以降に上がり、{best.replace(chr(10), '')}で {pct(T[best]['top3'])} になります。"
            f"ただ、人気から見込まれる複勝率もほぼ同じように動いていて、1〜5 戦目をまとめると実際 {pct(E['top3'])} 対 見込み {pct(E['exp_top3'])}（{pfmt(a['early_p_top3'])}）、"
            f"勝率も {pct(E['win'])} 対 {pct(E['implied'])}（{pfmt(a['early_p_win'])}）です。"
            + ("段階ごとでは " + "、".join(f"{s}（{v * 100:+.1f}pt、補正後 {pfmt(ph)}）" for s, v, ph in top_sig) + " が見込みを下回りますが、去勢明けの 6 段階を同時に見ていることを考えると（Holm 法で補正）、偶然の範囲と区別しにくい差です。"
               if top_sig else "去勢明けのどの段階でも、見込みとの差は有意ではありません。")
            + f"去勢した年齢で分けても、1〜5 戦目の複勝率は 2〜3 歳で {pct(EA['2〜3歳']['top3'])}（見込み {pct(EA['2〜3歳']['exp_top3'])}、{pfmt(EA['2〜3歳']['p'])}）、"
            f"4 歳以上で {pct(EA['4歳以上']['top3'])}（見込み {pct(EA['4歳以上']['exp_top3'])}、{pfmt(EA['4歳以上']['p'])}）です。"
            + "\n\n「去勢明けは買い」というほどの上積みはデータからは見えず、市場は去勢明けの馬をおおむね正しく評価していると考えています。")},
        a["c_top3"],
        a["c_win"],
        {"type": "table", "columns": ["段階", "出走数", "勝率", "オッズから見込まれる勝率", "複勝率", "人気から見込まれる複勝率", "単勝回収率"], "rows": rows,
         "caption": "去勢明け 1 戦目が判定できた馬だけの集計。去勢前の走は 2016 年以降のデータにあるものだけ。"},
        {"type": "text", "text": (
            f"6 戦目以降も走り続けている馬は、成績がよく現役を続けられた馬に偏ります（生き残りの偏り）。グラフの「対象の馬」を「去勢後に5戦以上走った馬」に切り替えると、"
            "同じ顔ぶれの馬で 1〜5 戦目を比べられます。また去勢した年齢（2〜3歳 / 4歳以上）でも切り替えられます。")},
        {"type": "heading", "text": "セン馬全体の位置づけ"},
        {"type": "text", "text": (
            f"去勢明けに限らないセン馬全体の勝率は {pct(sw['セ'][0])} で、牡馬（{pct(sw['牡'][0])}）・牝馬（{pct(sw['牝'][0])}）より低めです。"
            "年齢別に見ると、2〜4 歳では牡馬より低く、5 歳以降は牡馬とほぼ同じになります。若いうちに去勢される馬は気性などに課題を抱えていることが多く、"
            "年齢が上がると牡馬の側も力の落ちた馬が残ることが、差が縮む理由と考えられます。")},
        a["c_age"],
        a["c_pop"],
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。\n\n"
                "・去勢明け初戦: 前走（JRA 平地の直前の出走）の性別が牡で、今回セになっている出走。去勢の日付そのものはデータにないので、性別の表記が替わった時点で判定しています。"
                "前走が 2015 年以前・地方・障害の馬や、2016 年以降に初めて出たときに既にセだった馬は判定できず、対象外です。\n\n"
                f"・去勢した年齢（去勢明け初戦の年齢）: 2〜3 歳 {young:,} 頭、4 歳以上 {a['n_horses'] - young:,} 頭。\n\n"
                "・人気から見込まれる複勝率: 全出走での、同じ人気・同じ頭数（実際に走った頭数）の複勝率。複勝オッズの記録がないため、単勝人気で代わりにしています。\n\n"
                "・同じ年齢の牡馬（点線）: 各段階の出走の年齢構成に合わせて、牡馬の年齢別の率を加重平均したもの。クラスや能力はそろえていないので、目安です。")},
        ]},
    ]
    return {
        "slug": "gelding-return",
        "title": "去勢明けは買い？——2戦目から上向くが、オッズも同じだけ動く",
        "claim": "去勢明けは買い",
        "verdict": "not_supported",
        "summary": (f"去勢明けの複勝率は初戦 {pct(k1['top3'])} から {best.replace(chr(10), '')}の {pct(T[best]['top3'])} へ上向きますが、人気からの見込みも同じように上がり、"
                    f"1〜5 戦目の勝率は {pct(E['win'])} と見込み（{pct(E['implied'])}）並みです。市場はおおむね織り込み済みです。"),
        "categories": ["horse/sex"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。前走は牡・今走はセの出走を「去勢明け初戦」とし、同じ馬の前後の出走を集計",
                    "sample": f"去勢明けの馬 {a['n_horses']:,} 頭（去勢明けの出走 {sum(T[s]['n'] for s in AFTER):,} 走）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = load()
    col = article(analyze(frame(df), df))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
