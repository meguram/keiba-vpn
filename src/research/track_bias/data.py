"""読み込み・掃除・市場の見込み q・位置取りと枠の値・1 レースの値 S。

- 平地（芝・ダ）、発走した馬（着順 ≥ 1 と競走中止。取消・除外を除く）だけ。頭数 N は発走した馬で数える。中止の馬は y = 0。
- y は 3 着内なら 1。3 着の同着は残りの枠を頭数で割る（y の和は 3）。
- 除くレース: 頭数 5 未満、走った馬に確定単勝オッズの欠けがある、通過順の欠けがある（直線の芝 1000m など角が無いレース）。
- 市場の見込み q: 確定単勝オッズの 1/odds を和 1 に直した勝率 → Harville で 3 着内の確率（レースで和 3）。
- 位置取り（前らしさ c = 0.5 − 相対位置）: 主は序盤（通過順の最初の数字。角が 2 つのレースは 3 角）、並べる版は 4 角（最後の数字）。
  4 角の位置は、好走する馬が前へ上がって来る分だけ着順の結果でもあるので主にしない。
- 枠（内らしさ g = 0.5 − 馬番の相対位置）、4 角の併走の群の中の外からの位置（案 B）、g から序盤の位置の分を除いた残り（案 C）。
- 1 レースの値 S_x = Σ_i (y_i − q_i) · x_i（y = 3 着内）。正なら市場の見込みより x の高い側（前・内）の馬が来た。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TABLES = REPO / "data/page_reference/tables"
FIRST_YEAR = 2016

# 1 レースの値の名前（x の名前）。front_* は前後、inner_* は内外（A・B・C の 3 案）
METRICS = ("front_early", "front_4c", "front_l3f", "front_combo", "inner_a", "inner_b", "inner_b_r", "inner_b_g", "inner_c")
# x の名前（馬ごとの列）
X_COL = {"front_early": "c_early", "front_4c": "c_4c", "front_l3f": "c_l3f_neg", "front_combo": "c_combo",
         "inner_a": "g_a", "inner_b": "g_b", "inner_b_r": "g_b_r", "inner_b_g": "g_b_g", "inner_c": "g_c"}
AUX = ("fr3_ex", "bk3_ex", "in3_ex", "out3_ex", "fav_front_ex", "long_back_ex")

DIST_BANDS = ((1400, "短"), (1800, "マイル"), (2200, "中"), (10**9, "長"))
COND_GROUP = {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}


def class_group(grade: str) -> str:
    g = str(grade or "")
    if g in ("新馬", "未勝利"):
        return "新馬・未勝利"
    if g == "1勝":
        return "1勝"
    if g in ("2勝", "3勝"):
        return "2・3勝"
    return "OP以上"


def dist_band(d: float) -> str:
    for hi, name in DIST_BANDS:
        if d <= hi:
            return name
    return "長"


def field_group(n: int) -> str:
    return "≤8" if n <= 8 else ("9〜13" if n <= 13 else "14+")


# ---------------------------------------------------------------- 通過順・角の並び

def first_last(s) -> tuple[float, float]:
    """passing_order（例 6-5-5-4）の最初と最後の数字。無ければ NaN。"""
    x = [int(v) for v in str(s or "").split("-") if v.strip().isdigit()]
    return (float(x[0]), float(x[-1])) if x else (np.nan, np.nan)


_TOKEN = re.compile(r"\(|\)|\d+")


def parse_order_text(text: str) -> dict[int, int]:
    """角の並びの文字列（例 `(*5,2)-(3,4)1`）から、馬番 → 併走の群の中の内からの位置（0 が最も内・単独も 0）。

    JRA の成績表の見方: 括弧は 1 馬身未満の併走の群で、括弧の中は内側の馬から書く。`*` は群の先頭（無視する）。
    """
    out: dict[int, int] = {}
    depth, k = 0, 0
    for t in _TOKEN.findall(str(text or "")):
        if t == "(":
            depth, k = 1, 0
        elif t == ")":
            depth = 0
        else:
            n = int(t)
            out[n] = k if depth else 0
            if depth:
                k += 1
    return out


def parse_order_groups(text: str) -> dict[int, tuple[int, int]]:
    """馬番 → (群の中の内からの位置, 群の頭数)。単独の馬は (0, 1)。"""
    pos = parse_order_text(text)
    out: dict[int, tuple[int, int]] = {}
    depth, members = 0, []
    for tk in _TOKEN.findall(str(text or "")):
        if tk == "(":
            depth, members = 1, []
        elif tk == ")":
            for h in members:
                out[h] = (pos[h], len(members))
            depth = 0
        elif depth:
            members.append(int(tk))
        else:
            out[int(tk)] = (0, 1)
    return out


def last_corner_lateral(cp, groups: bool = False):
    """corner_passing（JSON の配列）の最後の角（4 角があれば 4 角）の、馬番 → 群の中の内からの位置。"""
    try:
        arr = json.loads(cp) if isinstance(cp, str) else (cp or [])
    except (ValueError, TypeError):
        return {}
    def corner_no(e) -> int:
        try:
            return int(e.get("corner") or 0)
        except (TypeError, ValueError):
            return 0
    cs = [e for e in arr if isinstance(e, dict) and corner_no(e) >= 1 and e.get("order_text")]
    if not cs:
        return {}
    text = max(cs, key=corner_no)["order_text"]
    return parse_order_groups(text) if groups else parse_order_text(text)


# ---------------------------------------------------------------- 市場の見込み

def harville_top3(p: np.ndarray) -> np.ndarray:
    """勝率 p（和 1）から、Harville の 3 着内の確率（和 3。頭数 3 以下は全部 1）。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n <= 3:
        return np.ones(n)
    a = p / np.clip(1 - p, 1e-12, None)                     # p_j / (1 − p_j)
    p2 = p * (a.sum() - a)                                  # 2 着: Σ_{j≠i} p_j · p_i/(1 − p_j)
    m = (p[:, None] * p[None, :] / np.clip(1 - p[:, None], 1e-12, None)) / np.clip(1 - p[:, None] - p[None, :], 1e-12, None)
    np.fill_diagonal(m, 0.0)                                # m_jk = p_j p_k /(1−p_j) /(1−p_j−p_k)
    p3 = p * (m.sum() - m.sum(axis=1) - m.sum(axis=0))      # 3 着: j・k は i 以外
    return np.clip(p + p2 + p3, 0.0, 1.0)


# ---------------------------------------------------------------- 読み込み

_HAS_CORNER = r'"corner": [1-4]\b'
_FLAT_COLS = ["horse_id", "race_id", "date", "venue", "surface", "distance", "grade", "track_condition", "start_time", "round",
              "finish_position", "bracket_number", "horse_number", "horse_name", "passing_order", "last_3f", "odds", "popularity"]
_OPT_COLS = ["finish_position_raw", "finish_status", "corner_passing"]


def _read_year(y: int, tables: Path) -> pd.DataFrame:
    p = tables / str(y) / "race_result_flat.parquet"
    if not p.exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq
    have = set(pq.read_schema(p).names)
    r = pd.read_parquet(p, columns=[c for c in _FLAT_COLS + _OPT_COLS if c in have])
    for c in _FLAT_COLS + ["finish_position_raw", "finish_status"]:
        if c not in r.columns:
            r[c] = np.nan
    lap = tables / str(y) / "race_result_lap_race.parquet"
    cp = pd.Series(dtype=object)
    if lap.exists():                                         # 角の並びは lap の表のほうが揃っている（2024〜25 の flat はほぼ空）
        lr = pd.read_parquet(lap, columns=["race_id", "corner_passing"]).dropna()
        lr = lr[lr["corner_passing"].astype(str).str.contains(_HAS_CORNER)].drop_duplicates("race_id")
        cp = lr.set_index(lr["race_id"].astype(str))["corner_passing"]
    r["race_id"] = r["race_id"].astype(str)
    flat_cp = pd.Series(dtype=object)
    if "corner_passing" in r.columns:
        fr = r.drop_duplicates("race_id")
        fr = fr[fr["corner_passing"].astype(str).str.contains(_HAS_CORNER)]
        flat_cp = fr.set_index("race_id")["corner_passing"]
    cp = cp.combine_first(flat_cp) if len(flat_cp) else cp
    r = r.drop(columns=["corner_passing"], errors="ignore")
    r["corner_passing"] = r["race_id"].map(cp)
    return r


def load_horses(years=None, tables: Path | None = None) -> pd.DataFrame:
    """走った馬の表（掃除の後）。列は data の説明のとおり。"""
    tables = tables or TABLES
    if years is None:
        years = [int(p.name) for p in tables.iterdir() if p.is_dir() and p.name.isdigit()]
    parts = [_read_year(y, tables) for y in sorted(years)]
    r = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return clean(r)


def clean(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["surf"] = r["surface"].astype(str).str[0]
    r["fin"] = pd.to_numeric(r["finish_position"], errors="coerce")
    # 競走中止の馬は残す（発走して単勝が売れた馬なので、市場の見込みの正規化に入れ、y = 0）。
    # 中止の印は 中・dnf。2024〜25 の表は印が無いので、通過順のある着順 −1 の馬を中止とみなす（通過順の無い中止は取消・除外と分けられず落ちる）
    for c in ("finish_position_raw", "finish_status", "corner_passing"):
        if c not in r.columns:
            r[c] = np.nan
    raw = r["finish_position_raw"].astype(str)
    st = r["finish_status"].astype(str)
    no_mark = r["finish_position_raw"].isna() & r["finish_status"].isna()
    has_po = r["passing_order"].fillna("").astype(str).str.contains(r"\d")
    r["dnf"] = (r["fin"] < 1) & ((raw == "中") | (st == "dnf") | (no_mark & has_po))
    r = r[r["surf"].isin(["芝", "ダ"]) & ((r["fin"] >= 1) | r["dnf"])].copy()
    r.loc[r["dnf"], "fin"] = 99.0
    r["odds"] = pd.to_numeric(r["odds"], errors="coerce")
    r["horse_number"] = pd.to_numeric(r["horse_number"], errors="coerce")
    fl = r["passing_order"].map(first_last)
    r["pos_first"] = [a for a, _ in fl]
    r["pos_last"] = [b for _, b in fl]
    # 取り込みの誤り: 同じ日に同じ馬が 2 レースに出ている（中身の同じレースの写し）。race_id の後のレースを丸ごと落とす（pedigree_lab.data と同じ）
    r["race_id"] = r["race_id"].astype(str)
    r = r.sort_values(["race_id", "horse_number"], kind="stable")
    hid = r["horse_id"].astype(str)
    dup = r.assign(_h=hid).duplicated(["_h", "date"], keep="first") & r["horse_id"].notna()
    r = r[~r["race_id"].isin(r.loc[dup, "race_id"].unique())].copy()
    r["N"] = r.groupby("race_id")["fin"].transform("size")
    # 通過順の無い中止の馬（最初の角の前に止まった）は最後方に置く
    for c in ("pos_first", "pos_last"):
        r.loc[r["dnf"] & r[c].isna(), c] = r["N"] + 1
    bad = (r["odds"].isna() | (r["odds"] <= 0) | r["pos_first"].isna() | r["horse_number"].isna())
    bad_race = bad.groupby(r["race_id"]).transform("any")
    r = r[~bad_race & (r["N"] >= 5)].copy()
    r["date"] = r["date"].astype(str).str[:10]
    r["round"] = pd.to_numeric(r["round"], errors="coerce")
    r["popularity"] = pd.to_numeric(r["popularity"], errors="coerce")
    r["distance"] = pd.to_numeric(r["distance"], errors="coerce")
    for c in ("venue", "grade", "track_condition"):          # 区分の鍵の欠けは「?」（groupby が NaN の鍵を落とすため）
        r[c] = r[c].fillna("?").astype(str)
    return r.sort_values(["date", "venue", "race_id", "horse_number"]).reset_index(drop=True)


# ---------------------------------------------------------------- 馬ごとの値・1 レースの値

def top3_share(r: pd.DataFrame) -> pd.Series:
    """3 着内の y。3 着の同着は、残りの枠（3 − 3 着より上の頭数）を同着の頭数で割る（y の和をレースで 3 にそろえる）。"""
    fin = r["fin"]
    above = (fin < 3).groupby(r["race_id"]).transform("sum")
    m3 = (fin == 3).groupby(r["race_id"]).transform("sum")
    y = (fin < 3).astype(float)
    share = ((3 - above).clip(lower=0) / m3.replace(0, np.nan)).fillna(0)
    return y.where(fin != 3, share)


def resid_within(r: pd.DataFrame, ycol: str, xcol: str) -> pd.Series:
    """レースの中で ycol を xcol に回帰（切片つき）した残り。ycol の欠けはそのまま欠け。"""
    ok = r[ycol].notna()
    rid = r["race_id"]
    ym = r[ycol].where(ok).groupby(rid).transform("mean")
    xm = r[xcol].where(ok).groupby(rid).transform("mean")
    dy, dx = r[ycol] - ym, (r[xcol] - xm).where(ok)
    beta = (dx * dy).groupby(rid).transform("sum") / (dx * dx).groupby(rid).transform("sum").replace(0, np.nan)
    return dy - beta.fillna(0) * dx


def horse_values(r: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに q・y・c（前らしさ）・g（内らしさ）・脚質の名札を足す。"""
    r = r.copy()
    q = np.empty(len(r))
    lat = np.full(len(r), np.nan)
    gsz = np.full(len(r), np.nan)
    for _, idx in r.groupby("race_id", sort=False).indices.items():
        o = r["odds"].to_numpy()[idx]
        p = (1 / o) / (1 / o).sum()
        q[idx] = harville_top3(p)
        cp = r["corner_passing"].iloc[idx[0]]
        if isinstance(cp, str) and cp:
            m = last_corner_lateral(cp, groups=True)
            if m:
                hn = r["horse_number"].to_numpy()[idx].astype(int)
                lat[idx] = [m[int(h)][0] if int(h) in m else np.nan for h in hn]   # 並びに無い馬は欠け（そのレースの案 B は欠けになる）
                gsz[idx] = [m[int(h)][1] if int(h) in m else np.nan for h in hn]
    r["q"] = q
    r["y"] = top3_share(r)
    r["lat4"] = lat
    r["grp4"] = gsz
    g = r.groupby("race_id")
    n1 = (r["N"] - 1).clip(lower=1)
    r["f_early"] = (g["pos_first"].rank(method="average") - 1) / n1
    r["f_4c"] = (g["pos_last"].rank(method="average") - 1) / n1
    r["f_gate"] = (g["horse_number"].rank(method="average") - 1) / n1
    r["c_early"] = 0.5 - r["f_early"]
    r["c_4c"] = 0.5 - r["f_4c"]
    r["g_a"] = 0.5 - r["f_gate"]
    r["g_b"] = -(r["lat4"] - g["lat4"].transform("mean"))       # 4 角で群の外にいるほど負
    # 案 B の変種 g: 群の中の相対位置（内 +0.5〜外 −0.5）。単独の馬は内外が分からないので 0（中ほど）。
    # 単独の馬を最も内（lat 0）と数える元の B は、単独で走ることの多い逃げ・先行の馬を内側に数える
    rel = np.where(r["grp4"] >= 2, r["lat4"] / (r["grp4"] - 1).clip(lower=1) - 0.5, 0.0)
    r["g_b_g"] = np.where(r["lat4"].notna(), -rel, np.nan)
    # 上がり 3F の速さ（速いほど +）。中止などで無い馬は最も遅い側に置く。前有利の日は末脚の馬が来ないので、x は符号を反転
    l3 = pd.to_numeric(r["last_3f"], errors="coerce")
    l3 = l3.fillna(l3.groupby(r["race_id"]).transform("max") + 1)
    r["f_l3f"] = (l3.groupby(r["race_id"]).rank(method="average") - 1) / n1
    r["c_l3f_neg"] = -(0.5 - r["f_l3f"])
    r["c_combo"] = (r["c_early"] + r["c_l3f_neg"]) / 2       # 序盤の前らしさ と 末脚の遅さ の平均
    # 案 C: g_a から序盤の位置（c_early）の分を除いた残り（レースの中の最小二乗）。B の変種 B_r も同じ形で g_b から除く
    r["g_c"] = resid_within(r, "g_a", "c_early")
    r["g_b_r"] = resid_within(r, "g_b", "c_early")
    # 脚質の名札（序盤の位置から）: 逃げ＝序盤の先頭、先行＝前 1/3、差し＝〜2/3、追込＝残り
    r["run_style"] = np.select([g["pos_first"].transform("min") == r["pos_first"], r["f_early"] <= 1 / 3, r["f_early"] <= 2 / 3],
                           ["逃げ", "先行", "差し"], "追込")
    return r


def race_values(h: pd.DataFrame) -> pd.DataFrame:
    """1 レース 1 行: S（METRICS）と補助の値（AUX）と区分の鍵。"""
    e = h["y"] - h["q"]
    t = pd.DataFrame({"race_id": h["race_id"]})
    for k, x in X_COL.items():
        t[k] = e * h[x]
    t["fr3_ex"] = e * (h["f_early"] <= 1 / 3)
    t["bk3_ex"] = e * (h["f_early"] > 2 / 3)
    t["in3_ex"] = e * (h["f_gate"] <= 1 / 3)
    t["out3_ex"] = e * (h["f_gate"] > 2 / 3)
    pop = h["popularity"]
    t["fav_front_ex"] = e * ((pop <= 3) & (h["f_early"] <= 1 / 3))
    t["long_back_ex"] = e * ((pop >= 6) & (h["f_early"] > 2 / 3))
    R = t.groupby("race_id", sort=False).sum(min_count=1)
    nb = h.groupby("race_id", sort=False)["lat4"].apply(lambda s: s.notna().all())
    R.loc[~nb.reindex(R.index).fillna(False).astype(bool), ["inner_b", "inner_b_r", "inner_b_g"]] = np.nan
    first = h.groupby("race_id", sort=False).first()
    for c in ("date", "venue", "surf", "distance", "grade", "track_condition", "round", "start_time", "N"):
        R[c] = first[c]
    R["dist_band"] = R["distance"].map(dist_band)
    R["class_g"] = R["grade"].map(class_group)
    R["cond_g"] = R["track_condition"].map(COND_GROUP).fillna("良")
    R["field_g"] = R["N"].astype(int).map(field_group)
    R = R.reset_index()
    return add_course_info(R)


# ---------------------------------------------------------------- 開催の情報（仮柵・何日目・クッション値・含水率）

COURSE_POS = REPO / "data/page_reference/course_position"
CUSHION = REPO / "data/page_reference/cushion"


def _main_tree(p: Path) -> Path:
    """worktree（.claude/worktrees/<名前>）に無い置き場は、その worktree の本体の作業ツリーの同じ場所を読む（どちらも手元のファイル）。"""
    if p.exists() and any(p.iterdir()):
        return p
    parts = REPO.parts
    if ".claude" in parts and "worktrees" in parts:
        alt = Path(*parts[:parts.index(".claude")]) / p.relative_to(REPO)
        if alt.exists():
            return alt
    return p


def info_sources() -> dict:
    """開催の情報の元のファイルの hash（校正の記録に残す。本体の作業ツリーの表は cron で書き換わるため）。"""
    import hashlib
    out = {}
    for base, pat in ((COURSE_POS, "course_position_*.json"), (CUSHION, "cushion_20*.json")):
        for f in sorted(_main_tree(base).glob(pat)):
            out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
    return out


def course_info() -> pd.DataFrame:
    """日付 × 場 → 開催（回）・何日目・芝の仮柵（A・B・C・D）。"""
    rows = []
    for p in sorted(_main_tree(COURSE_POS).glob("course_position_*.json")):
        for d, xs in (json.loads(p.read_text(encoding="utf-8")).get("dates") or {}).items():
            for x in xs:
                rows.append({"date": d, "venue": x.get("venue"), "kai": x.get("kai"), "meet_day": x.get("day"), "rail": x.get("turf_course") or "?"})
    return pd.DataFrame(rows).drop_duplicates(["date", "venue"])


def cushion_info() -> pd.DataFrame:
    """日付 × 場 → クッション値・含水率（ゴール前・4 角）。その日の朝の測りの値（発表は発走前）。"""
    rows = []
    for p in sorted(_main_tree(CUSHION).glob("cushion_20*.json")):
        rows += json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return pd.DataFrame(columns=["date", "venue"])
    c = pd.DataFrame(rows)
    c = c[c["date"].astype(str).str.len() >= 10].rename(columns={"venue_name": "venue"})
    for k in ("cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"):
        c[k] = pd.to_numeric(c.get(k), errors="coerce")
    return c[["date", "venue", "cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"]].drop_duplicates(["date", "venue"], keep="last")


def meet_band(d) -> str:
    if pd.isna(d):
        return "?"
    d = int(d)
    return "1〜2日目" if d <= 2 else ("3〜4日目" if d <= 4 else ("5〜6日目" if d <= 6 else "7日目〜"))


def cushion_band(v) -> str:
    if pd.isna(v):
        return "?"
    return "〜8.9" if v < 9 else ("9〜9.9" if v < 10 else "10〜")


def moist_band(v) -> str:
    if pd.isna(v):
        return "?"
    return "〜4.9" if v < 5 else ("5〜9.9" if v < 10 else "10〜")


def add_course_info(R: pd.DataFrame) -> pd.DataFrame:
    """R に 開催・何日目・仮柵・クッション値・含水率と、その帯（基準線の区分の候補）を足す。"""
    R = R.drop(columns=[c for c in ("kai", "meet_day", "rail", "cushion_value", "turf_moisture_goal", "turf_moisture_4corner",
                                     "dirt_moisture_goal", "dirt_moisture_4corner") if c in R.columns])
    ci, cu = course_info(), cushion_info()
    if len(ci):
        R = R.merge(ci, on=["date", "venue"], how="left")
    else:
        R["kai"] = np.nan; R["meet_day"] = np.nan; R["rail"] = "?"
    if len(cu):
        R = R.merge(cu, on=["date", "venue"], how="left")
    for k in ("cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"):
        if k not in R.columns:
            R[k] = np.nan
    R["meet_band"] = R["meet_day"].map(meet_band)
    R["rail"] = np.where(R["surf"] == "芝", R["rail"].fillna("?").astype(str), "-")
    # 馬場の硬さ・含水の帯: 芝はクッション値、ダートはゴール前の含水率
    R["firm_band"] = np.where(R["surf"] == "芝", R["cushion_value"].map(cushion_band), R["dirt_moisture_goal"].map(moist_band))
    return R
