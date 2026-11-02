"""めぐコラム『初年度産駒と高齢出産』を実データから集計し、frontend/content/columns/first-crop-old-dam.json を書く。

問い: 母が何歳のときに産んだ馬か（母の出産時の年齢）で、勝率・複勝率は変わるか。種牡馬の初年度産駒は走らないのか。市場は織り込んでいるか。

データ:
  レース: compare_series.load()（2020〜2025 年の JRA 平地。取消・除外を除く）
  父・母: data/local/horse_pedigree_5gen/<生年>/<horse_id>.json（1 代目の父・母の id）。読むのに数分かかるので、
          data/local/research/columns/horse_parents.parquet にまとめて置く（無ければ作る。--rebuild で作り直す）。
定義:
  生年 = horse_id の先頭 4 桁。母の生年 = 母の id の先頭 4 桁（日本で生まれた馬の id は生年から始まる）。
  母の出産時の年齢 = 生年 − 母の生年。母が輸入馬（id が 000a… で生年が分からない）の馬は除く。
  初年度産駒 = 父の産駒（血統表のある馬）の中でいちばん早い生年の世代。2014 年以前の生まれは血統表がそろっていないので、
          いちばん早い産駒が 2016 年以降の父だけを対象にする（2 年分、産駒がいないことを確かめられる）。輸入種牡馬は日本での初年度。
比べ方: 出走ごとの勝率・複勝率（同じ馬が何度も走るので、信頼区間は馬単位のクラスタで計算）と、オッズから見込まれる勝率。

使い方:
    python -m src.research.columns.first_crop_old_dam            # 集計して記事を書く
    python -m src.research.columns.first_crop_old_dam --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import YEARS, _r, load
from src.research.columns.wind_venue import pct, pfmt

REPO = Path(__file__).resolve().parents[3]
PED_DIR = REPO / "data/local/horse_pedigree_5gen"
PARENTS = REPO / "data/local/research/columns/horse_parents.parquet"
OUT = REPO / "frontend/content/columns/first-crop-old-dam.json"
FIRST_FULL_YEAR = 2014           # これ以降の生まれは血統表がそろっている（各年 約 4,700 頭）
AGE_BINS = [0, 4] + list(range(5, 20)) + [99]
AGE_LABELS = ["3〜4歳"] + [f"{a}歳" for a in range(5, 20)] + ["20歳〜"]
CROPS = ["1年目\n（初年度）", "2年目", "3年目", "4年目〜"]
_YEAR_ID = r"^(19|20)\d{8}$"


def parents(rebuild: bool = False) -> pd.DataFrame:
    if PARENTS.exists() and not rebuild:
        return pd.read_parquet(PARENTS)
    rows = []
    for ydir in sorted(p for p in PED_DIR.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) >= FIRST_FULL_YEAR - 4):
        for f in ydir.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            a = {(x["generation"], x["position"]): x for x in d.get("ancestors", [])}
            s, m = a.get((1, 0), {}), a.get((1, 1), {})
            rows.append((d["horse_id"], s.get("horse_id"), s.get("name"), m.get("horse_id"), m.get("name")))
    p = pd.DataFrame(rows, columns=["horse_id", "sire_id", "sire", "dam_id", "dam"])
    PARENTS.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(PARENTS)
    return p


def frame(df: pd.DataFrame | None = None, ped: pd.DataFrame | None = None) -> pd.DataFrame:
    p = (parents() if ped is None else ped).copy()
    p["horse_id"] = p["horse_id"].astype(str)
    p["by"] = pd.to_numeric(p["horse_id"].str[:4], errors="coerce")
    dam = p["dam_id"].astype(str)
    p["dam_by"] = pd.to_numeric(dam.str[:4].where(dam.str.match(_YEAR_ID)), errors="coerce")
    p["dage"] = p["by"] - p["dam_by"]
    p.loc[~p["dage"].between(2, 30), "dage"] = np.nan
    first = p.groupby("sire_id")["by"].min().rename("sire_first")
    p = p.join(first, on="sire_id")
    p["crop"] = np.where(p["sire_first"] >= FIRST_FULL_YEAR + 2, p["by"] - p["sire_first"] + 1, np.nan)
    d = (load() if df is None else df).merge(p[["horse_id", "by", "dage", "crop", "sire_id", "sire_first"]], on="horse_id", how="left")
    d["dage_bin"] = pd.cut(d["dage"], AGE_BINS, labels=AGE_LABELS).astype(object)
    d["crop_bin"] = pd.cut(d["crop"], [0, 1, 2, 3, 99], labels=CROPS).astype(object)
    d["lq"] = np.log(d["odds"])
    return d


def clustered(v: pd.Series, g: pd.Series) -> tuple[float | None, int, list[float] | None, float]:
    """平均・標本数・95% 区間・平均の分散（同じ馬の出走をまとめたクラスタ頑健な分散）。"""
    m = v.notna()
    v, g = v[m].astype(float), g[m]
    n = len(v)
    if n == 0:
        return None, 0, None, 0.0
    mu = float(v.mean())
    s = (v - mu).groupby(g).sum()
    var = float((s ** 2).sum()) / (n * n)
    h = 1.96 * math.sqrt(var)
    return mu, n, [round(max(0.0, mu - h), 4), round(mu + h, 4)], var


def rate_series(d: pd.DataFrame, col: str, xs: list[str], metric: str, name: str) -> tuple[dict[str, Any], list[tuple[float | None, float]]]:
    vals, ns, cis, st = [], [], [], []
    for x in xs:
        w = d[d[col] == x]
        m, n, ci, var = clustered(w[metric], w["horse_id"])
        vals.append(_r(m)); ns.append(n); cis.append(ci); st.append((m, var))
    return {"name": name, "values": vals, "n": ns, "ci": cis}, st


def win_vs_implied(d: pd.DataFrame, col: str, xs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a, sa = rate_series(d, col, xs, "win", "実際の勝率")
    b, sb = rate_series(d, col, xs, "implied", "オッズから見込まれる勝率")
    ps = []
    for (ma, va), (mb, vb), x in zip(sa, sb, xs):
        w = d[d[col] == x]
        diff = w["win"] - w["implied"]                     # 同じ出走の差なので、差そのものの分散で検定する
        _, _, _, vd = clustered(diff, w["horse_id"])
        ps.append(None if ma is None or vd <= 0 else float(f"{math.erfc(abs(ma - mb) / math.sqrt(vd) / math.sqrt(2)):.3g}"))
    return [a, b], {"a": a["name"], "b": b["name"], "p": ps}


def fit(formula: str, d: pd.DataFrame):
    import statsmodels.formula.api as smf

    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["horse_id"])[0]})


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    z = d[d["dage"].notna()].copy()
    a["n_runs"], a["n_horses"] = len(z), int(z["horse_id"].nunique())
    a["share_known"] = d["dage"].notna().mean()
    a["by_range"] = (int(z["by"].min()), int(z["by"].max()))
    tab = z.groupby("dage_bin").agg(n=("win", "size"), h=("horse_id", "nunique"), win=("win", "mean"), top3=("top3", "mean"),
                                    implied=("implied", "mean"), ret=("ret", "mean"))
    a["tab"] = tab.reindex(AGE_LABELS)
    s3, _ = rate_series(z, "dage_bin", AGE_LABELS, "top3", "複勝率")
    sw, cw = win_vs_implied(z, "dage_bin", AGE_LABELS)
    a["c_top3"] = {"type": "chart", "title": "母の出産時の年齢別 複勝率", "xLabel": "母の出産時の年齢", "yLabel": "複勝率", "xs": AGE_LABELS,
                   "series": [s3, {"name": "全体の平均", "values": [_r(z["top3"].mean())] * len(AGE_LABELS), "baseline": True}],
                   "caption": "出走ごとの複勝率。縦線は 95% 信頼区間（同じ馬の出走をまとめて計算）。点線は全体の平均。"}
    a["c_win"] = {"type": "chart", "title": "母の出産時の年齢別 勝率（実際とオッズの見込み）", "xLabel": "母の出産時の年齢", "yLabel": "勝率", "xs": AGE_LABELS,
                  "series": sw, "compare": cw,
                  "caption": "オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。＊は実際と見込みの差が有意（p<0.05）。"}
    # 山なりか（2 次の項）と、オッズをそろえると残るか。馬単位のクラスタ頑健 SE
    q = fit("top3 ~ dage + I(dage**2)", z)
    a["quad"] = {k: (float(q.params[k]), float(q.pvalues[k])) for k in ("dage", "I(dage ** 2)")}
    a["peak"] = -q.params["dage"] / (2 * q.params["I(dage ** 2)"])
    for y in ("top3", "win"):
        f = fit(f"{y} ~ dage + I(dage**2) + lq + I(lq**2)", z)
        a[f"adj_{y}"] = {k: (float(f.params[k]), float(f.pvalues[k])) for k in ("dage", "I(dage ** 2)")}
        f2 = fit(f"{y} ~ C(dage_bin) + lq + I(lq**2)", z)
        names = [k for k in f2.params.index if "dage_bin" in k]
        R = np.zeros((len(names), len(f2.params)))
        for i, k in enumerate(names):
            R[i, list(f2.params.index).index(k)] = 1.0
        a[f"adj_{y}_joint"] = float(f2.wald_test(R, scalar=True).pvalue)
        # 母 10 歳のときの産駒と比べた、区分ごとの差（オッズをそろえたあと）
        f3 = fit(f"{y} ~ C(dage_bin, Treatment('10歳')) + lq + I(lq**2)", z.assign(dage_bin=z["dage_bin"].astype(str)))
        a[f"adj_{y}_bins"] = {k.split("[T.")[1][:-1]: (float(f3.params[k]), float(f3.pvalues[k])) for k in f3.params.index if "dage_bin" in k}
    # 初年度産駒
    c = d[d["crop_bin"].notna()].copy()
    a["crop_tab"] = c.groupby("crop_bin").agg(n=("win", "size"), h=("horse_id", "nunique"), s=("sire_id", "nunique"), win=("win", "mean"), top3=("top3", "mean"),
                                             implied=("implied", "mean"), ret=("ret", "mean"), dage=("dage", "mean")).reindex(CROPS)
    sc, cc = win_vs_implied(c, "crop_bin", CROPS)
    st3, _ = rate_series(c, "crop_bin", CROPS, "top3", "複勝率")
    a["c_crop"] = {"type": "chart", "title": "種牡馬の何年目の産駒か × 勝率（実際とオッズの見込み）", "xLabel": "父の何年目の産駒か", "yLabel": "勝率", "xs": CROPS,
                   "series": sc, "compare": cc,
                   "caption": f"いちばん早い産駒が {FIRST_FULL_YEAR + 2} 年以降に生まれた種牡馬の産駒だけ。＊は実際と見込みの差が有意（p<0.05）。"}
    a["crop_top3"] = st3
    # 同じ種牡馬の中で、初年度とそれ以降を比べる（3 世代以上の産駒が走っている種牡馬）
    many = c.groupby("sire_id")["crop"].transform("max") >= 3
    w = c[many].copy()
    w["first"] = (w["crop"] == 1).astype(float)
    a["n_sires_within"] = int(w["sire_id"].nunique())
    for y in ("top3", "win"):
        f = fit(f"{y} ~ first + C(sire_id) + C(age)", w)
        fo = fit(f"{y} ~ first + C(sire_id) + lq + I(lq**2)", w)
        a[f"crop_{y}"] = (float(f.params["first"]), float(f.bse["first"]), float(f.pvalues["first"]))
        a[f"crop_{y}_odds"] = (float(fo.params["first"]), float(fo.pvalues["first"]))
    a["crop_dage"] = (float(c.loc[c["crop"] == 1, "dage"].mean()), float(c.loc[c["crop"] > 1, "dage"].mean()))
    return a


def article(a: dict[str, Any]) -> dict[str, Any]:
    T = a["tab"]
    peak_bin = T["top3"].idxmax()
    young, old = T.loc["3〜4歳"], T.loc["20歳〜"]
    mid = T.loc[[f"{k}歳" for k in range(8, 13)]]
    mid_top3 = float((mid["top3"] * mid["n"]).sum() / mid["n"].sum())
    old2 = T.loc[["18歳", "19歳", "20歳〜"]]
    old_top3 = float((old2["top3"] * old2["n"]).sum() / old2["n"].sum())
    B3 = a["adj_top3_bins"]
    C = a["crop_tab"]
    c1, c2 = C.loc[CROPS[0]], C.loc[CROPS[1]]
    e3, se3, p3 = a["crop_top3"]
    ew, _, pw = a["crop_win"]
    rows = [[lab, f"{int(T.loc[lab, 'h']):,}", f"{int(T.loc[lab, 'n']):,}", pct(T.loc[lab, "top3"]), pct(T.loc[lab, "win"]), pct(T.loc[lab, "implied"]), pct(T.loc[lab, "ret"])]
            for lab in AGE_LABELS]
    crop_rows = [[k.replace("\n", ""), f"{int(C.loc[k, 's']):,}", f"{int(C.loc[k, 'h']):,}", f"{int(C.loc[k, 'n']):,}", pct(C.loc[k, "top3"]), pct(C.loc[k, "win"]),
                  pct(C.loc[k, "implied"]), f"{C.loc[k, 'dage']:.1f}歳"] for k in CROPS]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "母が 8〜12 歳のときの産駒の複勝率", "value": pct(mid_top3), "sub": f"いちばん高い区分は {peak_bin}（{pct(T.loc[peak_bin, 'top3'])}）"},
            {"label": "母が 18 歳以上のときの産駒の複勝率", "value": pct(old_top3), "sub": f"3〜4 歳のときは {pct(young['top3'])}"},
            {"label": "オッズをそろえた複勝率の差（母 3〜4 歳）", "value": f"{B3['3〜4歳'][0] * 100:+.1f}pt", "sub": f"母 10 歳のときの産駒と比べて（{pfmt(B3['3〜4歳'][1])}）。生の差は {(young['top3'] - T.loc['10歳', 'top3']) * 100:+.1f}pt"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"母が何歳のときに産んだ馬かで分けると、複勝率は山なりになります。母が若い（3〜5 歳）ときと高齢（18 歳以上）のときの産駒は低めで、"
            f"8〜12 歳のときの産駒が {pct(mid_top3)} といちばん高い水準です（18 歳以上は {pct(old_top3)}）。2 次式を当てはめると、いちばん高くなるのは母が {a['peak']:.0f} 歳前後のときです。\n\n"
            "ただし、オッズから見込まれる勝率も同じように山なりで、実際の勝率とほぼ重なります。市場は、若い母・高齢の母の産駒が走りにくいことの大部分をすでに織り込んでいます。\n\n"
            f"オッズをそろえても、小さな差は残ります（母の年齢の区分で変わるかの検定は 複勝率 {pfmt(a['adj_top3_joint'])}、勝率 {pfmt(a['adj_win_joint'])}）。"
            "母が 10 歳のときの産駒と比べた複勝率の差は、"
            + "、".join(f"{k} {B3[k][0] * 100:+.1f}pt（{pfmt(B3[k][1])}）" for k in ("3〜4歳", "5歳", "18歳", "19歳", "20歳〜"))
            + f"で、生の差（3〜4 歳で {(young['top3'] - T.loc['10歳', 'top3']) * 100:+.1f}pt）よりずっと小さくなります。"
            "母がごく若いときや 18〜19 歳のときの産駒は、人気よりやや割り引いてよい可能性がありますが、区分ごとの標本は少なく、17 区分を見ている中での差なので、強い材料とは言えません。")},
        a["c_top3"],
        a["c_win"],
        {"type": "table", "columns": ["母の出産時の年齢", "馬の数", "出走数", "複勝率", "勝率", "オッズから見込まれる勝率", "単勝回収率"], "rows": rows,
         "caption": "単勝回収率は高配当の数回で大きく動くので、区分ごとの値は参考程度。"},
        {"type": "heading", "text": "初年度産駒は走らない？"},
        {"type": "text", "text": (
            f"種牡馬ごとに、いちばん早く生まれた産駒の世代を「初年度」として、何年目の産駒かで比べました（いちばん早い産駒が {FIRST_FULL_YEAR + 2} 年以降に生まれた種牡馬だけ）。"
            f"初年度産駒の複勝率は {pct(c1['top3'])}、2 年目は {pct(c2['top3'])} で、勝率は {pct(c1['win'])}（見込み {pct(c1['implied'])}）でした。\n\n"
            f"同じ種牡馬の初年度とそれ以降を、年齢をそろえて比べると（3 世代以上が走っている {a['n_sires_within']} 頭の種牡馬）、初年度産駒の複勝率の差は "
            f"{e3 * 100:+.1f}pt（{pfmt(p3)}）、勝率は {ew * 100:+.1f}pt（{pfmt(pw)}）です。"
            + ("初年度産駒が走らないという傾向は見られません。" if p3 >= 0.05 and pw >= 0.05
               else "勝率はわずかに上回るものの、複勝率では差がなく、少なくとも初年度産駒が走らないという傾向は見られません。" if p3 >= 0.05 and ew > 0
               else "初年度産駒のほうが成績がよく、走らないという通説とは逆です。" if e3 > 0
               else "初年度産駒のほうが成績が低めです。")
            + f"オッズをそろえた勝率の差は {a['crop_win_odds'][0] * 100:+.1f}pt（{pfmt(a['crop_win_odds'][1])}）です。"
            f"なお母の平均年齢は初年度 {a['crop_dage'][0]:.1f} 歳、2 年目以降 {a['crop_dage'][1]:.1f} 歳とほぼ同じで、母の年齢の違いは初年度産駒の比較にはほとんど混ざっていません。"
            "一方、初年度は種牡馬の期待が高く、質の高い繁殖牝馬が集まりやすいと言われます。母の質（年齢以外）はこのデータではそろえていません。")},
        a["c_crop"],
        {"type": "table", "columns": ["何年目の産駒か", "種牡馬の数", "馬の数", "出走数", "複勝率", "勝率", "オッズから見込まれる勝率", "母の平均年齢"], "rows": crop_rows,
         "caption": "「4年目〜」は、いちばん早い産駒が 2016〜2019 年生まれの種牡馬の 4 世代目以降だけなので、種牡馬の顔ぶれが 1〜3 年目と違う。"},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地の出走（{a['n_horses']:,} 頭・{a['n_runs']:,} 走）。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。\n\n"
                f"・母の出産時の年齢: 馬の生年 − 母の生年（どちらも血統表の id の先頭 4 桁）。母が輸入馬で生年の分からない出走（全体の {pct(1 - a['share_known'], 0)}）は除いています。\n\n"
                "・信頼区間と検定: 同じ馬が何度も走るので、馬ごとにまとめたクラスタ頑健な標準誤差を使っています。\n\n"
                "・交絡: 母の年齢は、母の能力（よく走る仔を出す母ほど長く繁殖に残る）、父の格（若い母・高齢の母に付ける種牡馬）、"
                "初年度産駒かどうかなどと絡み合っています。ここで見ているのは「母が何歳のときの産駒か」で分けた成績で、加齢そのものの影響を取り出したものではありません。\n\n"
                f"・初年度産駒: 血統表がそろっているのは {FIRST_FULL_YEAR} 年以降の生まれなので、それより前から産駒のいる種牡馬は対象外です。輸入種牡馬は日本での初年度です。")},
        ]},
    ]
    return {
        "slug": "first-crop-old-dam",
        "title": "高齢出産の馬は走らない？——成績は山なり、ただし大半はオッズが織り込み済み",
        "claim": "初年度産駒・高齢出産馬は成績が低い",
        "verdict": "partial",
        "summary": (f"母が 8〜12 歳のときの産駒の複勝率は {pct(mid_top3)}、18 歳以上のときは {pct(old_top3)} と、高齢出産の馬は成績が低めです。"
                    "ただしオッズの見込みもほぼ同じだけ低く、差の大半は市場が織り込み済みです。初年度産駒が走らないという傾向は見られませんでした。"
                    if p3 >= 0.05 else
                    f"母が 8〜12 歳のときの産駒の複勝率は {pct(mid_top3)}、18 歳以上のときは {pct(old_top3)} と、高齢出産の馬は成績が低めですが、オッズの見込みもほぼ同じだけ低く、差の大半は市場が織り込み済みです。"),
        "categories": ["horse/birth"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場・芝ダート（障害除く）。母の生年が分かる馬（母が輸入馬の馬は除く）",
                    "sample": f"{a['n_horses']:,}頭・{a['n_runs']:,}出走"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="血統表から父・母の表を作り直す（数分かかる）")
    args = ap.parse_args()
    col = article(analyze(frame(ped=parents(args.rebuild))))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
