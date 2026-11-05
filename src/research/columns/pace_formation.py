"""めぐコラム『逃げ馬の数と人気馬の位置取りで、ペースはどう変わるか』を実データから集計し、frontend/content/columns/pace-formation.json を書く。\n\n問い: 出走馬の顔ぶれ（逃げ馬の頭数・前走で前にいた馬の頭数・上位人気馬の脚質）で、そのレースのラップ（前半3F・上がり3F・ペースの区分）は変わるか。\n      変わるなら、クラスや芝・ダートで違うか。集計に使っていない 2026 年でも同じか。\n\n隊列の手がかり（すべて、そのレースより前の走だけから作る。レース前に分かる）:\n  初角の位置 … passing_order の最初の数字（最初のコーナーの順位）。\n  逃げ馬 … 直近 3 走のうち 2 走以上で初角が 1 番手の馬。\n  前走3番手以内 … 前走の初角が 3 番手以内の馬（逃げ馬を含む）。\n  差し・追込 … 直近 3 走のすべてで、初角の位置 (順位−1)/(頭数−1) が 0.35 を超えた馬（プロジェクトの脚質の境界。tp64_pace_benefit と同じ）。\n  直近 3 走が JRA 平地でそろわない馬（デビューから間もない馬など）は判定できない。判定できる馬が出走馬の 7 割に満たないレースは除く。\nラップ（data/local/modeling/shared_cache/race_quality/race_quality.parquet。コース = 場×芝ダ×距離×馬場の標準からのずれ、秒）:\n  t3f_dev … 前半3F のずれ（マイナス = 標準より速い）、race_l3f_dev … 上がり3F のずれ（プラス = 遅い）、\n  bal … 上がり3F のずれ − 前半3F のずれ（大きいほど前傾 = ハイペース）、pace_cls … bal の上位 25% = ハイ・下位 25% = スロー。\n集計期間は 2021〜2025 年（直近 3 走を引くため 2020 年から読む）。2026 年は集計に使わず、結論を確かめるためだけに使う。\n\n使い方:\n    python -m src.research.columns.pace_formation            # 集計して記事を書く\n    python -m src.research.columns.pace_formation --dry-run  # 書かずに記事の JSON を出す\n    python -m src.research.columns.pace_formation --out PATH # 別の場所に書く（リポジトリが読み取り専用のホストで使う）\n"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import COLUMNS_DIR, FLAT_DIR, RACE_QUALITY_PATH, jump_mask
from src.research.columns.race_context import joint_p, series
from src.research.columns.wind_venue import pct, pfmt

OUT = COLUMNS_DIR / "pace-formation.json"
LOOKBACK_FROM = 2020          # 直近 3 走を引くため、集計の前年から読む
MAIN_YEARS = (2021, 2025)     # 集計期間
TEST_YEAR = 2026              # 確かめるためだけに使う年
REAR_MIN = 0.35               # 差し・追込の境界（初角の位置がこれより後ろ）
MIN_KNOWN = 0.7               # 判定できる馬が出走馬のこの割合以上のレースだけ使う
MIN_N = 30                    # グラフの 1 点に出す最小のレース数
CLASSES = ["未勝利", "1勝", "2勝", "3勝", "オープン・重賞"]
GRADE_CLASS = {"新馬": "未勝利", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝"}
PACES = {"high": "ハイ", "mid": "ミドル", "slow": "スロー"}
NIGE_BINS = ([-1, 0, 1, 99], ["0頭", "1頭", "2頭以上"])
PREV_BINS = ([-1, 1, 3, 5, 99], ["0〜1頭", "2〜3頭", "4〜5頭", "6頭以上"])
REAR_BINS = ([-1, 1, 2, 3], ["0〜1頭", "2頭", "3頭"])
SURF = {"全体": None, "芝": "芝", "ダート": "ダート"}
CLS_FILTER = {"全体": None, **{c: c for c in CLASSES}}


# ── 隊列の手がかり ──────────────────────────────────


def add_history(r: pd.DataFrame) -> pd.DataFrame:
    """出走ごとの表（horse_id・date・race_id・first・q を持つ）に、直近 3 走までの隊列の手がかりを付ける。今走は含めない。"""
    r = r.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    lead = (r["first"] == 1).where(r["first"].notna()).astype(float)
    rear = (r["q"] > REAR_MIN).where(r["q"].notna()).astype(float)
    last3 = lambda s: s.shift().rolling(3, min_periods=3).sum()          # noqa: E731
    r["lead3"] = lead.groupby(r["horse_id"]).transform(last3)           # 直近 3 走で初角 1 番手だった数
    r["rear3"] = rear.groupby(r["horse_id"]).transform(last3)           # 直近 3 走で差し・追込の位置だった数
    r["prev_first"] = r.groupby("horse_id")["first"].shift()
    r["is_nige"] = (r["lead3"] >= 2).where(r["lead3"].notna())
    r["is_rear"] = (r["rear3"] == 3).where(r["rear3"].notna())
    r["prev_front3"] = (r["prev_first"] <= 3).where(r["prev_first"].notna())
    r["known"] = r["lead3"].notna() & r["rear3"].notna()
    return r


def load_runs(last_year: int) -> pd.DataFrame:
    """JRA 平地の出走ごとの表（取消・除外を除く）。直近 3 走の隊列の手がかりを付ける。"""
    cols = ["race_id", "date", "venue", "surface", "distance", "grade", "race_name", "race_class", "horse_id", "passing_order", "odds", "popularity"]
    years = [y for y in range(LOOKBACK_FROM, last_year + 1) if (FLAT_DIR / str(y) / "race_result_flat.parquet").exists()]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~jump_mask(r) & (r["odds"] > 0) & (r["popularity"] > 0)].drop(columns=["race_name", "race_class", "odds"]).copy()
    r["horse_id"] = r["horse_id"].astype(str)
    r["starters"] = r.groupby("race_id")["race_id"].transform("size")
    first = pd.to_numeric(r["passing_order"].astype(str).str.split("-").str[0], errors="coerce")
    r["first"] = first.where(first.between(1, 99))
    r["q"] = ((r["first"] - 1) / (r["starters"] - 1)).where(r["first"].notna() & (r["starters"] >= 4))
    return add_history(r)


def race_table(runs: pd.DataFrame, first_year: int) -> pd.DataFrame:
    """レースごとの表: 逃げ馬の頭数・前走3番手以内の頭数・上位3人気の脚質・ラップのずれ。使えるレースだけ。"""
    d = runs[runs["date"].astype(str).str[:4].astype(int) >= first_year]
    count = lambda col: (lambda s: int((s == True).sum()))               # noqa: E712,E731
    r = d.groupby("race_id").agg(date=("date", "first"), venue=("venue", "first"), surface=("surface", "first"), distance=("distance", "first"),
                                 grade=("grade", "first"), starters=("starters", "first"), known=("known", "mean"),
                                 n_nige=("is_nige", count("is_nige")), n_prev=("prev_front3", count("prev_front3")))
    t = d[d["popularity"] <= 3].groupby("race_id").agg(top_n=("popularity", "size"), top_known=("is_rear", lambda s: bool(s.notna().all())),
                                                     top_rear=("is_rear", count("is_rear")))
    r = r.join(t)
    r["top_known"] = r["top_known"].fillna(False).astype(bool)
    r["top_n"] = r["top_n"].fillna(0).astype(int)
    r["year"] = r["date"].astype(str).str[:4].astype(int)
    r["cls"] = r["grade"].map(GRADE_CLASS).fillna("オープン・重賞")
    q = pd.read_parquet(RACE_QUALITY_PATH, columns=["race_id", "pace_cls", "t3f_dev", "race_l3f_dev", "bal", "going_group"]).drop_duplicates("race_id").set_index("race_id")
    r = r.join(q)
    r = r[(r["known"] >= MIN_KNOWN) & (r["starters"] >= 8) & r["pace_cls"].notna()].copy()
    r["key"] = r["venue"] + r["surface"] + r["distance"].astype(int).astype(str)
    r["nige_b"] = pd.cut(r["n_nige"], NIGE_BINS[0], labels=NIGE_BINS[1]).astype(str)
    r["prev_b"] = pd.cut(r["n_prev"], PREV_BINS[0], labels=PREV_BINS[1]).astype(str)
    ok = (r["top_n"] == 3) & r["top_known"]                                 # 上位3人気の3頭すべてが判定できるレース
    r["rear_b"] = pd.cut(r["top_rear"].where(ok), REAR_BINS[0], labels=REAR_BINS[1]).astype(str)
    r.loc[~ok, "rear_b"] = "不明"
    return r


# ── 集計 ──────────────────────────────────────────


def rate_series(d: pd.DataFrame, col: str, xs: list[str]) -> list[dict[str, Any]]:
    """横軸 xs（col の値）ごとの、ハイペース・スローペースになったレースの割合（1 点が MIN_N 未満なら出さない）。"""
    out = []
    for k in ("high", "slow"):
        s = series(f"{PACES[k]}ペースの割合", [(d.loc[d[col] == x, "pace_cls"] == k).astype(float) for x in xs])
        for i, n in enumerate(s["n"]):
            if n < MIN_N:
                s["values"][i], s["ci"][i] = None, None
        out.append(s)
    return out


def rate_chart(d: pd.DataFrame, col: str, xs: list[str], *, title: str, x_label: str, caption: str) -> dict[str, Any]:
    """クラス・芝ダートをプルダウンで切り替えられる、ハイ・スローの割合のグラフ。"""
    filters = [{"label": "クラス", "options": list(CLS_FILTER)}, {"label": "芝・ダート", "options": list(SURF)}]
    variants = {}
    for (cn, cv), (sn, sv) in product(CLS_FILTER.items(), SURF.items()):
        if cv is None and sv is None:
            continue
        w = d
        if cv is not None:
            w = w[w["cls"] == cv]
        if sv is not None:
            w = w[w["surface"] == sv]
        variants[f"{cn}|{sn}"] = {"series": rate_series(w, col, xs), "caption": f"{cn if cn != '全体' else '全クラス'}・{sn if sn != '全体' else '芝とダート'}。" + caption}
    return {"type": "chart", "title": title, "xLabel": x_label, "yLabel": "レースの割合", "xs": xs, "series": rate_series(d, col, xs),
            "filters": filters, "variants": variants, "caption": caption}


def lap_row(label: str, w: pd.DataFrame) -> list[str | int]:
    """表の 1 行: 条件・レース数・前半3F/上がり3F/前後半差（コース標準からのずれ、秒）・ハイ/スローの割合。"""
    sec = lambda v: f"{v:+.2f}"                                           # noqa: E731
    return [label, len(w), sec(w["t3f_dev"].mean()), sec(w["race_l3f_dev"].mean()), sec(w["bal"].mean()),
            pct((w["pace_cls"] == "high").mean(), 0), pct((w["pace_cls"] == "slow").mean(), 0)]


LAP_COLUMNS = ["条件", "レース数", "前半3Fのずれ（秒）", "上がり3Fのずれ（秒）", "前後半差（秒）", "ハイ", "スロー"]
FORMULA = ("{y} ~ C(nige_b, Treatment('0頭')) + C(prev_b, Treatment('2〜3頭')) + C(rear_b, Treatment('0〜1頭')) + starters + C(cls) + C(key)")


def fit(d: pd.DataFrame, y: str = "bal"):
    import statsmodels.formula.api as smf

    return smf.ols(FORMULA.format(y=y), d[d["rear_b"] != "不明"]).fit(cov_type="HC1")


def coef_row(label: str, name: str, f_main, f_test) -> list[str]:
    def one(f) -> tuple[str, str]:
        if name not in f.params:
            return "–", "–"
        return f"{f.params[name]:+.2f}", pfmt(f.pvalues[name]).replace("p=", "").replace("p<", "<")
    a, b = one(f_main), one(f_test)
    return [label, a[0], a[1], b[0], b[1]]


NIGE_T = "C(nige_b, Treatment('0頭'))[T.{}]"
PREV_T = "C(prev_b, Treatment('2〜3頭'))[T.{}]"
REAR_T = "C(rear_b, Treatment('0〜1頭'))[T.{}]"


def interaction_p(d: pd.DataFrame, which: str) -> float:
    """クラスで効き方が違うか: 隊列の変数 × クラスの交互作用がすべて 0 かの Wald 検定の p 値。"""
    import statsmodels.formula.api as smf

    z = d[d["rear_b"] != "不明"]
    terms = {"nige": "C(nige_b, Treatment('0頭'))", "prev": "C(prev_b, Treatment('2〜3頭'))"}
    other = [t for k, t in terms.items() if k != which]
    f = smf.ols(f"bal ~ {terms[which]} * C(cls) + {other[0]} + C(rear_b, Treatment('0〜1頭')) + starters + C(key)", z).fit(cov_type="HC1")
    return joint_p(f, ":")


def course_table(d: pd.DataFrame, min_races: int = 150) -> tuple[list[list[str | int]], dict[str, int]]:
    """主なコースごとの、逃げ馬・前走3番手以内の有無による前後半差の変わり方。向きの一致数も返す。"""
    rows, same = [], {"nige": 0, "prev": 0, "n": 0}
    sizes = d.groupby("key").size()
    for key in sizes[sizes >= min_races].sort_values(ascending=False).index:
        w = d[d["key"] == key]
        a, b = w[w["n_nige"] == 0], w[w["n_nige"] >= 1]
        c, e = w[w["n_prev"] <= 3], w[w["n_prev"] >= 4]
        dn, dp = b["bal"].mean() - a["bal"].mean(), e["bal"].mean() - c["bal"].mean()
        rows.append([key, len(w), f"{dn:+.2f}（{len(a)}/{len(b)}）", f"{dp:+.2f}（{len(c)}/{len(e)}）"])
        same["nige"] += int(dn > 0); same["prev"] += int(dp > 0); same["n"] += 1
    return rows, same


# ── ハロンごとのラップの推移 ─────────────────────────

LAP_MIN_COURSE = 100          # コースのプルダウンに出す最小のレース数
LAP_MIN_BIN = 20              # 1 本の線に出す最小のレース数
LAP_AXES = {                  # 分け方 → (列, 区分, 線の名前の前置き)
    "逃げ馬の頭数": ("nige_b", NIGE_BINS[1], "逃げ馬 "),
    "前走3番手以内の頭数": ("prev_b", PREV_BINS[1], "前走3番手以内 "),
    "上位3人気の差し・追込の頭数": ("rear_b", REAR_BINS[1], "上位3人気の差し・追込 "),
}


def parse_laps(raw: Any) -> list[float]:
    """lap_times は '[12.3, 11.4, …]' の文字列。正の有限の数だけを残す（読めなければ空）。"""
    try:
        v = json.loads(raw) if isinstance(raw, str) else list(raw)
        return [float(x) for x in v if isinstance(x, (int, float)) and np.isfinite(x) and x > 0]
    except (TypeError, ValueError):
        return []


def load_laps(first_year: int, last_year: int) -> pd.Series:
    """レースごとの 1 ハロンごとのラップ（秒、スタート→ゴール）。race_id をキーにした list の Series。"""
    parts = [pd.read_parquet(FLAT_DIR / str(y) / "race_result_lap_race.parquet", columns=["race_id", "lap_times"])
             for y in range(first_year, last_year + 1) if (FLAT_DIR / str(y) / "race_result_lap_race.parquet").exists()]
    d = pd.concat(parts, ignore_index=True).drop_duplicates("race_id")
    return pd.Series(d["lap_times"].map(parse_laps).to_numpy(), index=d["race_id"].to_numpy(), name="laps")


def attach_laps(d: pd.DataFrame, laps: pd.Series) -> dict[str, np.ndarray]:
    """d（レースの表）にラップを付け、コースごとの『レース × ハロン』の行列を返す。良馬場のレースだけを使い、ハロン数がコースの最頻値と違うレースは除く。
    d に前半3F・中盤・上がり3F の『コース平均とのラップ差の合計（秒）』（seg_first / seg_mid / seg_last）を足す（d を直接更新）。"""
    w = d.loc[d["going_group"] == "良", ["key"]].join(laps.rename("laps"), how="inner")             # 稍重・重・不良はラップの水準が大きくぶれるので除く
    w["nl"] = w["laps"].map(len)
    out: dict[str, np.ndarray] = {}
    for col in ("seg_first", "seg_mid", "seg_last"):
        d[col] = np.nan
    for key, g in w.groupby("key"):
        mode = g["nl"].mode().iloc[0]
        g = g[g["nl"] == mode]
        if mode < 6 or len(g) < LAP_MIN_COURSE:
            continue
        M = np.vstack(g["laps"].to_numpy())
        dev = M - M.mean(axis=0)
        out[key] = pd.DataFrame(M, index=g.index)
        d.loc[g.index, "seg_first"] = dev[:, :3].sum(axis=1)
        if mode >= 7:                                   # 1200m（6 ハロン）は前半3F と上がり3F だけで中盤がない
            d.loc[g.index, "seg_mid"] = dev[:, 3:-3].sum(axis=1)
        d.loc[g.index, "seg_last"] = dev[:, -3:].sum(axis=1)
    return out


def lap_series(name: str, dev: np.ndarray, width: int, offset: np.ndarray | None = None) -> dict[str, Any]:
    """ハロンごとの平均ラップ（秒）。dev はコース平均との差の行列で、offset（コースの平均ラップ）を足すと平均ラップそのものになる（省略は差のまま）。
    n は線全体のレース数。区間は平均 ± 1.96 × 標準誤差。足りない分（短いコース）は空欄。"""
    n = dev.shape[0]
    pad = width - dev.shape[1]
    base = np.zeros(dev.shape[1]) if offset is None else np.asarray(offset, dtype="float64")
    if n < LAP_MIN_BIN:
        return {"name": name, "values": [None] * width, "n": [n] * dev.shape[1] + [None] * pad, "ci": [None] * width}
    m = dev.mean(axis=0) + base
    se = dev.std(axis=0, ddof=1) / np.sqrt(n)
    return {"name": name, "values": [round(float(v), 3) for v in m] + [None] * pad, "n": [n] * dev.shape[1] + [None] * pad,
            "ci": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(m - 1.96 * se, m + 1.96 * se)] + [None] * pad}


VENUE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
ALL_VENUES = "全場"


def sd_sort_key(sd: str) -> tuple[int, int]:
    """『芝1600』『ダート1200』を 芝→ダート、距離の短い順に並べるキー。"""
    return (0 if sd.startswith("芝") else 1, int("".join(ch for ch in sd if ch.isdigit()) or 0))


def lap_units(mats: dict[str, pd.DataFrame]) -> dict[tuple[str, str], tuple[np.ndarray, pd.Index, np.ndarray]]:
    """プルダウンの選択肢ごとに（コース平均とのラップ差の行列, レースの index, 平均ラップ）を作る。キーは（会場, 芝/ダート+距離）。
    会場が『全場』のものは、同じ面・距離の全場を合わせたもの（差はコースごとの平均から測ってから合わせる）。"""
    unit: dict[tuple[str, str], tuple[np.ndarray, pd.Index, np.ndarray]] = {}
    groups: dict[str, list[tuple[str, str]]] = {}
    for key, m in mats.items():
        M = m.to_numpy()
        k2 = (key[:2], key[2:])
        unit[k2] = (M - M.mean(axis=0), m.index, M.mean(axis=0))
        groups.setdefault(k2[1], []).append(k2)
    for sd, ks in groups.items():
        if len(ks) < 2 or len({unit[k][0].shape[1] for k in ks}) != 1:
            continue
        dev = np.vstack([unit[k][0] for k in ks])
        if len(dev) >= 2 * LAP_MIN_COURSE:
            unit[(ALL_VENUES, sd)] = (dev, pd.Index(np.concatenate([unit[k][1].to_numpy() for k in ks])), np.vstack([unit[k][2] for k in ks]).mean(axis=0))
    vrank = {v: i for i, v in enumerate([ALL_VENUES] + VENUE_ORDER)}
    return dict(sorted(unit.items(), key=lambda kv: (vrank.get(kv[0][0], 99), sd_sort_key(kv[0][1]))))


def lap_xs(width: int) -> list[str]:
    """横軸の目盛り。最初の 1F にスタート、最後の 1F にゴールを付ける。"""
    xs = [f"{i}F" for i in range(1, width + 1)]
    xs[0], xs[-1] = "1F スタート", f"{width}F ゴール"
    return xs


def lap_chart(d: pd.DataFrame, mats: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """会場 → 芝・ダート+距離 の順にプルダウンで選び（cascade: 前の選択で存在する選択肢だけを出す）、全パターンの平均ラップ（秒）をボタンで選んで重ねて比べるグラフ。
    会場が『全場』のときは、コースごとの平均との差を合わせて全場の平均ラップに足したもの（会場の顔ぶれの違いを除く）。"""
    units = lap_units(mats)
    venues = list(dict.fromkeys(v for v, _ in units))
    sds = sorted({sd for _, sd in units}, key=sd_sort_key)
    variants: dict[str, Any] = {}
    for (venue, sd), (dev, idx, mean) in units.items():
        g = d.loc[idx]
        width = dev.shape[1]                                                    # コースの距離なりの長さ（余白は作らない）
        ser = [{**lap_series("全レース", dev, width, mean), "baseline": True}]
        for col, bins, pre in LAP_AXES.values():
            ser += [lap_series(pre + b, dev[(g[col] == b).to_numpy()], width, mean) for b in bins]
        mean_txt = " - ".join(f"{v:.1f}" for v in mean)
        variants[f"{venue}|{sd}"] = {"xs": lap_xs(width), "series": ser, "caption": (
            f"{venue} {sd}（{len(dev):,} レース）。全レースの平均ラップ（秒）: {mean_txt}。{LAP_MIN_BIN} レースに満たないパターンは線を出していない。縦線は 95% 信頼区間。良馬場のレースだけ。")}
    first = next(iter(variants.values()))
    initial = [f"{LAP_AXES['逃げ馬の頭数'][2]}{b}" for b in NIGE_BINS[1]]
    return {"type": "chart", "title": "ハロンごとの平均ラップ（パターン別）", "xLabel": "スタート → ゴール（1F = 200m。最後の 3 つが上がり3F）", "yLabel": "平均ラップ（秒）",
            "xs": first["xs"], "format": "num", "series": first["series"], "caption": first["caption"], "cascade": True, "fitAxis": True, "initial": initial,
            "filters": [{"label": "会場", "options": venues}, {"label": "芝・ダート／距離", "options": sds}], "variants": variants}


def segment_rows(d: pd.DataFrame) -> list[list[str | int]]:
    """分け方ごとの区分について、前半3F・中盤・上がり3F の『コース平均とのラップ差の合計（秒）』（コースをならした平均）。"""
    rows: list[list[str | int]] = []
    z = d[d["seg_first"].notna()]
    for an, (col, bins, pre) in LAP_AXES.items():
        for b in bins:
            w = z[z[col] == b]
            rows.append([pre + b, len(w), *(f"{w[c].mean():+.2f}" for c in ("seg_first", "seg_mid", "seg_last"))])
    return rows


def analyze() -> dict[str, Any]:
    runs = load_runs(TEST_YEAR)
    R = race_table(runs, MAIN_YEARS[0])
    A, T = R[R["year"].between(*MAIN_YEARS)], R[R["year"] == TEST_YEAR]
    a: dict[str, Any] = {"R": R, "A": A, "T": T, "n_all": len(R[R["year"] <= MAIN_YEARS[1]])}
    a["nige_chart"] = rate_chart(A, "nige_b", NIGE_BINS[1], title="逃げ馬の頭数別 ハイペース・スローペースになった割合", x_label="出走馬のうち逃げ馬の頭数（レース前に分かる）",
                                caption="逃げ馬=直近 3 走のうち 2 走以上で初角が 1 番手の馬。ハイ・スローは全レースの約 4 分の 1 ずつ。縦線は 95% 信頼区間。")
    a["prev_chart"] = rate_chart(A, "prev_b", PREV_BINS[1], title="前走3番手以内の馬の頭数別 ハイペース・スローペースになった割合", x_label="出走馬のうち、前走の初角が 3 番手以内だった馬の頭数",
                                caption="前走の初角が 3 番手以内の馬（逃げ馬を含む）の頭数。ハイ・スローは全レースの約 4 分の 1 ずつ。縦線は 95% 信頼区間。")
    AR = A[A["rear_b"] != "不明"]
    a["rear_chart"] = rate_chart(AR, "rear_b", REAR_BINS[1], title="上位3人気のうち差し・追込の馬の頭数別 ハイペース・スローペースになった割合", x_label="1〜3番人気のうち、直近 3 走とも差し・追込だった馬の頭数",
                                caption="差し・追込=直近 3 走のすべてで初角の位置が前から 35% より後ろ。縦線は 95% 信頼区間。")
    a["fit_main"], a["fit_test"] = fit(A), fit(T)
    a["fit_t3f"], a["fit_l3f"] = fit(A, "t3f_dev"), fit(A, "race_l3f_dev")
    a["p_cls_nige"], a["p_cls_prev"] = interaction_p(A, "nige"), interaction_p(A, "prev")
    a["course_rows"], a["course_same"] = course_table(A)
    AR_T = T[T["rear_b"] != "不明"]
    a["AR"], a["AR_T"] = AR, AR_T
    a["n_course"] = A["key"].nunique()
    laps = load_laps(MAIN_YEARS[0], MAIN_YEARS[1])
    A = A.copy()
    mats = attach_laps(A, laps)
    a["A_lap"], a["lap_chart"], a["seg_rows"] = A, lap_chart(A, mats), segment_rows(A)
    a["n_lap_courses"], a["n_lap_races"] = len(mats), int(A["seg_first"].notna().sum())
    return a


# ── 記事 ──────────────────────────────────────────


def bin_table(d: pd.DataFrame, col: str, xs: list[str]) -> list[list[str | int]]:
    return [lap_row(x, d[d[col] == x]) for x in xs]


def fit_coef(f, name: str) -> str:
    return f"{f.params[name]:+.2f}" if name in f.params else "–"


def sign_text(f, name: str) -> str:
    return f"{f.params[name]:+.2f}秒（{pfmt(f.pvalues[name])}）"


def article(a: dict[str, Any]) -> dict[str, Any]:
    A, T, AR, AR_T = a["A"], a["T"], a["AR"], a["AR_T"]
    fm, ft = a["fit_main"], a["fit_test"]
    y0, y1 = MAIN_YEARS
    n0, n2 = (A["nige_b"] == "0頭").sum(), (A["nige_b"] == "2頭以上").sum()
    hi = lambda w: (w["pace_cls"] == "high").mean()      # noqa: E731
    lo = lambda w: (w["pace_cls"] == "slow").mean()      # noqa: E731
    N0, N2 = A[A["nige_b"] == "0頭"], A[A["nige_b"] == "2頭以上"]
    P0, P6 = A[A["prev_b"] == "0〜1頭"], A[A["prev_b"] == "6頭以上"]
    TN0, TN2 = T[T["nige_b"] == "0頭"], T[T["nige_b"] == "2頭以上"]
    TP0, TP6 = T[T["prev_b"] == "0〜1頭"], T[T["prev_b"] == "6頭以上"]
    n_ = NIGE_T.format("2頭以上")
    pv0, pv6 = PREV_T.format("0〜1頭"), PREV_T.format("6頭以上")
    rr3 = REAR_T.format("3頭")
    rear_ps = {k: fm.pvalues[REAR_T.format(k)] for k in ("2頭", "3頭")}
    rear_sig = min(rear_ps.values()) < 0.05
    combos = [
        ("逃げ馬 0頭 かつ 前走3番手以内 0〜1頭", (A["n_nige"] == 0) & (A["n_prev"] <= 1)),
        ("逃げ馬 0頭", A["n_nige"] == 0),
        ("前走3番手以内 0〜1頭", A["n_prev"] <= 1),
        ("全レース", A["n_nige"] >= 0),
        ("前走3番手以内 6頭以上", A["n_prev"] >= 6),
        ("逃げ馬 2頭以上", A["n_nige"] >= 2),
        ("逃げ馬 2頭以上 かつ 前走3番手以内 6頭以上", (A["n_nige"] >= 2) & (A["n_prev"] >= 6)),
    ]
    combo_rows = [lap_row(lb, A[m]) for lb, m in combos]
    cls_rows, cls_diff = [], {}
    for c in CLASSES:
        w = A[A["cls"] == c]
        w0, w2, p0, p6 = w[w["nige_b"] == "0頭"], w[w["nige_b"] == "2頭以上"], w[w["prev_b"] == "0〜1頭"], w[w["prev_b"] == "6頭以上"]
        cls_diff[c] = w2["bal"].mean() - w0["bal"].mean()
        cls_rows.append([c, len(w), f"{len(w0)} / {len(w2)}", f"{cls_diff[c]:+.2f}", f"{pct(lo(w0), 0)} → {pct(lo(w2), 0)}",
                         f"{len(p0)} / {len(p6)}", f"{p6['bal'].mean() - p0['bal'].mean():+.2f}", f"{pct(lo(p0), 0)} → {pct(lo(p6), 0)}"])
    lo_c, hi_c = min(cls_diff, key=cls_diff.get), max(cls_diff, key=cls_diff.get)
    cls_big = (lo_c, f"{cls_diff[lo_c]:+.2f}", hi_c, f"{cls_diff[hi_c]:+.2f}")
    reg_rows = [
        coef_row("逃げ馬 1頭（0頭と比べて）", NIGE_T.format("1頭"), fm, ft),
        coef_row("逃げ馬 2頭以上（0頭と比べて）", n_, fm, ft),
        coef_row("前走3番手以内 0〜1頭（2〜3頭と比べて）", pv0, fm, ft),
        coef_row("前走3番手以内 4〜5頭（2〜3頭と比べて）", PREV_T.format("4〜5頭"), fm, ft),
        coef_row("前走3番手以内 6頭以上（2〜3頭と比べて）", pv6, fm, ft),
        coef_row("上位3人気の差し・追込 2頭（0〜1頭と比べて）", REAR_T.format("2頭"), fm, ft),
        coef_row("上位3人気の差し・追込 3頭（0〜1頭と比べて）", rr3, fm, ft),
    ]
    split_rows = [[lb, sign_text(a["fit_t3f"], nm), sign_text(a["fit_l3f"], nm), sign_text(fm, nm)]
                  for lb, nm in (("逃げ馬 2頭以上（0頭と比べて）", n_), ("前走3番手以内 0〜1頭（2〜3頭と比べて）", pv0), ("前走3番手以内 6頭以上（2〜3頭と比べて）", pv6))]
    cs_ = a["course_same"]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "逃げ馬 0頭 → 2頭以上 のハイペースの割合", "value": f"{pct(hi(N0), 0)} → {pct(hi(N2), 0)}", "sub": f"スローは {pct(lo(N0), 0)} → {pct(lo(N2), 0)}（{y0}〜{y1}年）"},
            {"label": "前走3番手以内 0〜1頭 → 6頭以上 のスローの割合", "value": f"{pct(lo(P0), 0)} → {pct(lo(P6), 0)}", "sub": f"ハイは {pct(hi(P0), 0)} → {pct(hi(P6), 0)}"},
            {"label": "上位3人気がすべて差し・追込（差し・追込 3頭）の前後半差", "value": f"{fm.params[rr3]:+.2f}秒", "sub": f"{pfmt(fm.pvalues[rr3])}。" + ("差あり" if rear_sig else "はっきりした差は出ない")},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            "出走馬の顔ぶれは、ペースをある程度は変えます。ただし変わり方は「逃げ馬の数」と「前走で前にいた馬の数」で出て、「人気馬が後ろから行く馬ばかり」では出ませんでした。\n\n"
            f"・逃げ馬（直近 3 走のうち 2 走以上で初角 1 番手）が 0 頭のレースは、2 頭以上のレースより前後半差が {N2['bal'].mean() - N0['bal'].mean():+.2f}秒 前傾（ハイペース寄り）でした。"
            f"ハイペースになる割合は {pct(hi(N0), 0)} → {pct(hi(N2), 0)}、スローペースは {pct(lo(N0), 0)} → {pct(lo(N2), 0)}。\n"
            f"・前走で 3 番手以内にいた馬が 0〜1 頭しかいないレースは、スローペースが {pct(lo(P0), 0)}（全体は {pct(lo(A), 0)}）。6 頭以上いると {pct(lo(P6), 0)} まで減り、ハイペースが増えます。\n"
            f"・上位 3 人気のうち直近 3 走とも差し・追込だった馬が 3 頭のレースは、他と比べて前後半差が {fm.params[rr3]:+.2f}秒（{pfmt(fm.pvalues[rr3])}）。"
            + ("はっきりした差があります。" if rear_sig else "向きはスロー寄りですが誤差の範囲で、ペースの目安にはなりません。") + "\n\n"
            f"頭数・クラス・コースをそろえても残る差です（{y0}〜{y1}年 {len(A):,} レースの回帰）。集計に使っていない {TEST_YEAR} 年（{len(T):,} レース）でも、逃げ馬 2頭以上は {sign_text(ft, n_)}、"
            f"前走3番手以内 6頭以上は {sign_text(ft, pv6)} と同じ向きでした。")},
        {"type": "heading", "text": "ハロンごとのラップの推移（コース別）"},
        {"type": "text", "text": (
            "ハイ・ミドル・スローの 3 区分ではなく、ラップがスタートからゴールまでどう推移するかを見ます。"
            "縦軸は平均ラップ（秒）そのものです。点線は全レースの平均で、ボタンで選んだパターン（最大 4 つ）をその上に重ねて比べられます。"
            "最初は逃げ馬の頭数 0 / 1 / 2頭以上が出ています。逃げ馬の頭数と前走3番手以内の頭数など、別の分け方のパターンも同じグラフに重ねられます。"
            "会場 → 芝・ダート／距離 の順にプルダウンで選びます。会場の「全場」は、同じ面・距離の全場を合わせたもので、レース数が多く線が安定します"
            "（会場ごとのラップの水準の違いは、コースごとの平均との差を合わせてから全場の平均に足して除いています）。"
            "個別のコースは 1 本あたりのレース数が少なく、線がぶれやすいので、縦線（95% 信頼区間）が重なるところは差と読まないでください。"
            "稍重・重・不良の馬場ではラップの水準が大きくぶれるため、このグラフと下の表は良馬場のレースだけで集計しています。"
            "下の表は全コースを合わせた、前半3F・中盤・上がり3F の、コース平均との差です。")},
        a["lap_chart"],
        {"type": "table", "columns": ["条件", "レース数", "前半3F（秒）", "中盤（秒）", "上がり3F（秒）"], "rows": a["seg_rows"],
         "caption": (f"{y0}〜{y1}年、{a['n_lap_courses']} コースの {a['n_lap_races']:,} レース。各区間のラップを、そのコースの平均との差で足し合わせたもの（マイナスは速い）。"
                     "中盤は前半3F と上がり3F を除いた区間。")},
        {"type": "heading", "text": "逃げ馬の頭数とペース"},
        {"type": "text", "text": "逃げ馬を「直近 3 走のうち 2 走以上で、最初のコーナーを 1 番手で通った馬」と決め、そのレースに何頭いるかで分けました。"},
        a["nige_chart"],
        {"type": "table", "columns": LAP_COLUMNS, "rows": bin_table(A, "nige_b", NIGE_BINS[1]),
         "caption": f"{y0}〜{y1}年。前半3F・上がり3F・前後半差は、コース（場×芝ダ×距離×馬場）の標準からのずれ。前後半差は上がり3Fのずれ − 前半3Fのずれで、大きいほど前傾（ハイペース）。"},
        {"type": "text", "text": (
            f"逃げ馬が多いと前半3F が速く（{N0['t3f_dev'].mean():+.2f}秒 → {N2['t3f_dev'].mean():+.2f}秒）、上がり3F は遅く（{N0['race_l3f_dev'].mean():+.2f}秒 → {N2['race_l3f_dev'].mean():+.2f}秒）なります。"
            "逃げ馬が 1 頭増えるごとに、前後半差はおよそ 0.2 秒ずつ前傾に寄ります。")},
        {"type": "heading", "text": "前走3番手以内の馬の頭数とペース"},
        {"type": "text", "text": "逃げ馬の定義に当てはまらなくても、前走で前にいた馬が多ければ隊列は前に寄ります。前走の最初のコーナーが 3 番手以内だった馬の頭数で分けました。"},
        a["prev_chart"],
        {"type": "table", "columns": LAP_COLUMNS, "rows": bin_table(A, "prev_b", PREV_BINS[1]), "caption": f"{y0}〜{y1}年。見方は上の表と同じ。"},
        {"type": "heading", "text": "上位3人気がすべて後ろ寄りの馬だったとき"},
        {"type": "text", "text": (
            "1〜3 番人気の 3 頭すべてが、直近 3 走とも差し・追込（初角の位置が前から 35% より後ろ）だった場合です。"
            "人気馬が後ろから行くなら前に行く馬が少なく、スローになるのでは、という見方を確かめます。"
            f"3 頭とも判定できるレースだけが対象で、3 頭すべてが差し・追込のレースは {int((AR['rear_b'] == '3頭').sum()):,} 件と少ないため、区間は広くなります。")},
        a["rear_chart"],
        {"type": "table", "columns": LAP_COLUMNS, "rows": bin_table(AR, "rear_b", REAR_BINS[1]), "caption": f"{y0}〜{y1}年。上位3人気の3頭すべての直近3走が判定できるレースだけ。"},
        {"type": "text", "text": (
            f"差し・追込が増えるほどスローがやや増える向き（スロー {pct(lo(AR[AR['rear_b'] == '0〜1頭']), 0)} → {pct(lo(AR[AR['rear_b'] == '3頭']), 0)}）ですが、頭数・クラス・コースをそろえると 2 頭で {pfmt(rear_ps['2頭'])}、3 頭で {pfmt(rear_ps['3頭'])} と誤差の範囲です。"
            f"{TEST_YEAR} 年の 3 頭（{int((AR_T['rear_b'] == '3頭').sum())} レース）は前後半差が {fit_coef(ft, rr3)}秒と逆向きですが、レース数が少なく結論は出せません。")},
        {"type": "heading", "text": "組み合わせると"},
        {"type": "table", "columns": LAP_COLUMNS, "rows": combo_rows, "caption": (f"{y0}〜{y1}年。前走3番手以内が 0〜1 頭のレースの {pct(len(A[(A['n_nige'] == 0) & (A['n_prev'] <= 1)]) / len(P0), 0)} は逃げ馬も 0 頭なので、重ねても数字はほぼ変わらない。"
                                                                    "逃げ馬 2頭以上と前走3番手以内 6頭以上が重なると、ハイが最も多くスローが最も少ない。")},
        {"type": "heading", "text": "ほかの条件をそろえても残るか"},
        {"type": "table", "columns": ["条件", f"{y0}〜{y1}年 前後半差の変化（秒）", "p値", f"{TEST_YEAR}年 前後半差の変化（秒）", "p値"], "rows": reg_rows,
         "caption": f"レース単位の回帰（出走頭数・クラス・コースをそろえる。標準誤差は HC1）。前後半差が大きいほどハイペース寄り。{TEST_YEAR} 年は集計に使っていない。"},
        {"type": "table", "columns": ["条件", "前半3Fのずれ（秒）", "上がり3Fのずれ（秒）", "前後半差（秒）"], "rows": split_rows,
         "caption": f"同じ回帰を前半3F・上がり3F・前後半差それぞれに当てた係数（{y0}〜{y1}年）。前半が速くなる分と、上がりが遅くなる分の両方が出る。"},
        {"type": "heading", "text": "クラス別"},
        {"type": "text", "text": (
            "向きはどのクラスでも同じで、スローになる割合は逃げ馬が多いほど・前走3番手以内の馬が多いほど減ります。"
            f"差の大きさはクラスで違い、{cls_big[0]}が最も小さく（{cls_big[1]}秒）、{cls_big[2]}で最も大きい（{cls_big[3]}秒）。"
            f"クラスごとに効き方が違うかの検定は、逃げ馬の頭数で {pfmt(a['p_cls_nige'])}、前走3番手以内の頭数で {pfmt(a['p_cls_prev'])}。")},
        {"type": "table", "columns": ["クラス", "レース数", "逃げ0頭 / 2頭以上（レース数）", "前後半差の差（秒）", "スローの割合 0頭 → 2頭以上", "前走3番手以内 0〜1頭 / 6頭以上（レース数）", "前後半差の差（秒）", "スローの割合 0〜1頭 → 6頭以上"], "rows": cls_rows,
         "caption": f"{y0}〜{y1}年。前後半差の差は左が「逃げ馬 2頭以上 − 0頭」、右が「前走3番手以内 6頭以上 − 0〜1頭」。レース数が少ないところは差がぶれやすい。"},
        {"type": "heading", "text": f"{TEST_YEAR}年でも同じか"},
        {"type": "table", "columns": LAP_COLUMNS, "rows": [lap_row("逃げ馬 0頭", TN0), lap_row("逃げ馬 2頭以上", TN2), lap_row("前走3番手以内 0〜1頭", TP0), lap_row("前走3番手以内 6頭以上", TP6),
                                                            lap_row(f"全レース", T)],
         "caption": f"{TEST_YEAR} 年（1〜10 月）。集計期間の数字を見たあとに、一度だけ当てた。"},
        {"type": "heading", "text": "予想への使い方"},
        {"type": "text", "text": (
            "出走表が出た時点で、逃げ馬が 2 頭以上、または前走 3 番手以内の馬が 6 頭以上なら、前傾（ハイペース寄り）を、"
            "逃げ馬が 0 頭で前走 3 番手以内も 0〜1 頭なら、スローを見込む材料になります。ただし、差は「ハイペースの割合が 2 割台から 3 割に増える」程度で、"
            "それだけでペースを決め打ちできるほどではありません。「上位人気がすべて差し・追込だからスロー」という見方は、このデータでは根拠が出ませんでした。")},
        {"type": "link", "href": "/column/pace-scenario", "label": "ペースは読めるか——2026年のデータで確かめる"},
        {"type": "link", "href": "/column/pace-advantage", "label": "ハイペースになるほど、1番人気は信頼できない"},
        {"type": "accordion", "title": f"コース別の向き（{cs_['n']} コース）", "blocks": [
            {"type": "table", "columns": ["コース", "レース数", "逃げ馬 0頭 → 1頭以上の前後半差（秒）（0頭/1頭以上のレース数）", "前走3番手以内 3頭以下 → 4頭以上の前後半差（秒）（レース数）"], "rows": a["course_rows"],
             "caption": f"{y0}〜{y1}年に150レース以上あるコース（ダートが中心）。符号が + の（隊列が前に寄るほど前傾になる）コースは、逃げ馬で {cs_['nige']}/{cs_['n']}、前走3番手以内で {cs_['prev']}/{cs_['n']}。"},
        ]},
        {"type": "accordion", "title": "データと方法", "blocks": [
            {"type": "text", "text": (
                "・対象は JRA の芝・ダート（障害除く）で、出走 8 頭以上、ラップの区分が作られているレース。直近 3 走が JRA 平地でそろう馬が出走馬の 7 割以上いるレースだけを使いました。\n\n"
                "・「初角の位置」は通過順位の最初の数字。逃げ馬は直近 3 走のうち 2 走以上で初角 1 番手の馬、差し・追込は直近 3 走のすべてで初角の位置 (順位−1)/(頭数−1) が 0.35 を超えた馬。"
                "どれもそのレースより前の走だけから作るので、出走表が出た時点で分かります。\n\n"
                "・ラップはコース（場×芝ダ×距離×馬場）の標準からのずれ。ハイ・スローは前後半差の上位・下位 25%。\n\n"
                f"・{TEST_YEAR} 年は集計の方針や区分を決めるのに使っていません。")},
        ]},
    ]
    return {
        "slug": "pace-formation",
        "title": "逃げ馬の数と前走の位置取りで、ペースはどう変わるか",
        "claim": "逃げ馬の頭数や先行馬の頭数、上位人気馬の脚質でペースは変わる",
        "verdict": "partial",
        "summary": (f"逃げ馬が 0 頭だとハイペースは {pct(hi(N0), 0)}、2 頭以上で {pct(hi(N2), 0)}。前走3番手以内の馬が 0〜1 頭だとスローが {pct(lo(P0), 0)}、6 頭以上で {pct(lo(P6), 0)}。"
                    f"クラスを問わず同じ向きで、{TEST_YEAR} 年でも再現。一方、上位 3 人気がすべて差し・追込でもペースは変わりませんでした。"),
        "categories": ["style/front-runner"],
        "published": "2026-11-05",
        "updated": "2026-11-05",
        "dataset": {"period": f"{y0}〜{y1}年（{TEST_YEAR}年は検証用。1〜10月）", "conditions": "JRA全競馬場・芝ダート（障害除く）。出走8頭以上でラップの区分があるレース",
                    "sample": f"{y0}〜{y1}年 {len(A):,} レース（検証 {TEST_YEAR}年 {len(T):,} レース）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    col = article(analyze())
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
