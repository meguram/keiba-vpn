"""めぐコラムで共通に使う、出走ごとの「枠」と「脚質」の列、条件別の比較グラフ・回帰。

脚質（予想の時点で分かるもの）:
  前走まで直近 3 走（JRA 平地。脚質の履歴のため集計の前年から読む）の初角の位置 (順位−1)/(頭数−1) の平均を、
  wind_venue と同じ境界で 前寄り（≤0.15）/ 好位（≤0.4）/ 中団（≤0.7）/ 後方 に分ける。その走の通過順は使わない（結果の漏れになる）。
  今走の初角の位置（style_now）は「結果としての脚質」で、予想には使えない。
枠:
  枠番（1〜8）と、頭数の影響を除いた馬番の相対位置 (馬番−1)/(登録頭数−1) を 内（<1/3）/ 中 / 外（≥2/3）に分けたもの。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, YEARS, _p, _r, _rci, _stat, jump_mask

PRE_STYLES = ["前寄り", "好位", "中団", "後方"]
PRE_CUTS = [-0.01, 0.15, 0.4, 0.7, 1.01]
NOW_STYLES = ["逃げ", "先行", "差し", "追込"]
NOW_CUTS = [-0.01, 1e-4, 0.25, 0.65, 1.01]
GATE_POS = ["内\n（前1/3）", "中", "外\n（後1/3）"]
WAKU = [f"{i}枠" for i in range(1, 9)]


def context() -> pd.DataFrame:
    """race_id・horse_id ごとの 枠番・馬番の相対位置・事前の脚質・今走の位置。"""
    cols = ["race_id", "date", "surface", "horse_id", "horse_number", "bracket_number", "field_size", "passing_order", "odds", "popularity",
            "race_name", "race_class", "distance"]
    years = [y for y in range(min(YEARS) - 1, max(YEARS) + 1) if (FLAT_DIR / str(y) / "race_result_flat.parquet").exists()]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    # 平地だけ（障害は surface が芝・ダートのものも除く）。脚質は平地の前走までの位置取りから作る
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r) & (r["odds"] > 0) & (r["popularity"] > 0)].copy()
    r["horse_id"] = r["horse_id"].astype(str)
    r = positions(r)
    r["gate_rel"] = (r["horse_number"] - 1) / (r["field_size"] - 1).clip(lower=1)
    r["gate_pos"] = pd.cut(r["gate_rel"], [-0.01, 1 / 3 - 1e-9, 2 / 3 - 1e-9, 1.01], labels=GATE_POS).astype(str)
    r["waku"] = r["bracket_number"].astype(int).astype(str) + "枠"
    return r[["race_id", "horse_id", "waku", "gate_rel", "gate_pos", "q_prev", "style_pre", "q_now", "style_now"]]


def positions(r: pd.DataFrame) -> pd.DataFrame:
    """出走（取消・除外を除いたもの）に、今走の初角の位置 q_now と、前走まで直近 3 走の平均 q_prev・その区分を付ける。"""
    r = r.copy()
    starters = r.groupby("race_id")["race_id"].transform("size")
    first = pd.to_numeric(r["passing_order"].astype(str).str.split("-").str[0], errors="coerce")
    r["q_now"] = ((first - 1) / (starters - 1)).where(first.between(1, 99) & (starters >= 4))
    r = r.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    r["q_prev"] = r.groupby("horse_id")["q_now"].transform(lambda s: s.shift().rolling(3, min_periods=1).mean())   # 今走は含めない
    r["style_pre"] = pd.cut(r["q_prev"], PRE_CUTS, labels=PRE_STYLES).astype(object).where(r["q_prev"].notna(), None)
    r["style_now"] = pd.cut(r["q_now"], NOW_CUTS, labels=NOW_STYLES).astype(object).where(r["q_now"].notna(), None)
    return r


def with_context(df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(context(), on=["race_id", "horse_id"], how="left")


def series(name: str, parts: list[pd.Series]) -> dict[str, Any]:
    st = [_stat(v) for v in parts]
    return {"name": name, "values": [_r(s[0]) for s in st], "n": [s[1] for s in st], "ci": [_rci(s[2]) for s in st]}


def pair_series(d: pd.DataFrame, cond: pd.Series, xcol: str, xs: list[str], metric: str, a: str, b: str
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """横軸 xs ごとに、cond の出走（a）とそれ以外（b）の metric を並べ、点ごとの差の p 値を付ける。"""
    sa, sb, ps = [], [], []
    for x in xs:
        m = d[xcol].astype(str) == x
        va, vb = d.loc[m & cond, metric], d.loc[m & ~cond, metric]
        sa.append(va); sb.append(vb)
        ma, _, _, xa = _stat(va)
        mb, _, _, xb = _stat(vb)
        p = _p(xa, xb, ma, mb)
        ps.append(None if p is None else float(f"{p:.3g}"))
    return [series(a, sa), series(b, sb)], {"a": a, "b": b, "p": ps}


def pair_chart(d: pd.DataFrame, cond: pd.Series, xcol: str, xs: list[str], metric: str, a: str, b: str, *, title: str, x_label: str,
               y_label: str, caption: str, filters: dict[str, dict[str, Callable[[pd.DataFrame], pd.Series] | None]] | None = None
               ) -> dict[str, Any]:
    """比較グラフ。filters は {ラベル: {選択肢: d -> 行の条件（先頭は None = 全体）}}。選択肢の組み合わせごとに variants を作る。"""
    ser, cmp = pair_series(d, cond, xcol, xs, metric, a, b)
    chart: dict[str, Any] = {"type": "chart", "title": title, "xLabel": x_label, "yLabel": y_label, "xs": xs, "series": ser, "compare": cmp}
    if filters:
        labels = list(filters)
        chart["filters"] = [{"label": k, "options": list(v)} for k, v in filters.items()]
        variants = {}
        for combo in _product([list(filters[k].items()) for k in labels]):
            if all(f is None for _, f in combo):
                continue
            m = pd.Series(True, index=d.index)
            for _, f in combo:
                if f is not None:
                    m &= f(d)
            s, c = pair_series(d[m], cond[m], xcol, xs, metric, a, b)
            variants["|".join(k for k, _ in combo)] = {"series": s, "compare": c}
        chart["variants"] = variants
    chart["caption"] = caption
    return chart


def _product(lists: list[list[Any]]) -> list[list[Any]]:
    out: list[list[Any]] = [[]]
    for lst in lists:
        out = [o + [x] for o in out for x in lst]
    return out


def ols_fit(formula: str, d: pd.DataFrame):
    """線形確率モデル（同じレースの出走は独立でないので、レース単位のクラスタ頑健 SE）。"""
    import statsmodels.formula.api as smf

    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["race_id"])[0]})


def joint_p(fit, prefix: str) -> float:
    """名前に prefix を含む係数（交互作用の項）がすべて 0 かの Wald 検定の p 値。"""
    names = [k for k in fit.params.index if prefix in k]
    if not names:
        return float("nan")
    R = np.zeros((len(names), len(fit.params)))
    for i, k in enumerate(names):
        R[i, list(fit.params.index).index(k)] = 1.0
    return float(fit.wald_test(R, scalar=True).pvalue)
