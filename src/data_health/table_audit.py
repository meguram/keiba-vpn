"""表の中身の監査（列の欠け・値の範囲・ずれ・整合・学習と推論の欠けの差）。

既存の data_health（checks・coverage・schema_check）は「ファイルがあるか」「スキーマに合うか」を見る。ここでは、手元の表
（``data/page_reference/tables/<年>/*_flat.parquet`` や学習の特徴量の表）の**中身**を毎回の検査として見る。
単発の調査スクリプト ``src/scripts/data/audit_column_drift_by_year.py``（年×列の分布のずれ）・``audit_payoff_odds.py``
（払戻とオッズの突き合わせの詳しい調査）とは役割を分け、ここは「月単位の急変・列ずれ・整合の検査を毎回同じ形で出す」ことに絞る。

- ``missing_profile``: 列 × 月ごとの欠け（None・NaN・空文字・空白）と 0 の割合。
- ``detect_jumps``: 前月・前年同月と比べて欠け（または 0）が急に増えた（既定 5 ポイント以上）月。
- ``check_result_rows``: 値の範囲（印の値 0・-3・9999 などは欠けの代わりとして別に数える）、列の取り違え・1 つずれの兆候
  （枠番と馬番の関係、人気とオッズの逆転、隣の列と同じ値）。未来のレース（枠順の発表前）は除く。
- ``check_result_races``: レース単位の整合（1 着の単勝オッズ×100＝単勝の払戻、頭数、レース内の属性、列ずれのレース、障害の surface）。
- ``compare_entries``: 結果の表と出馬表の馬の一致。
- ``check_horse_birth``: 馬の ID（先頭 4 桁＝生年）と馬齢の整合。
- ``compare_missing``: 学習と推論の各列の欠けの割合を並べ、差の大きい列を挙げる。

GCS は読まない（費用 0 円）。CLI は ``python -m src.data_health.table_audit``（docs/operations/data_audit_20261116.md）。
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

JUMP_PT = 5.0  # 欠けの急増とみなす差（ポイント）
MIN_ROWS = 200  # 月の行数がこれ未満なら急変の判定に使わない
SENTINELS = (0, -1, -3, 9999)  # 書き出し・取得元が「値なし」に使う印（取消・除外・未発表・置き場だけの列）
OBSTACLE_RE = r"障害|ジャンプ|\(JG|(?<!ヤング)JS|スプリングJ"   # 平地のヤングジョッキーズシリーズ（ヤングJS）は除く

# 期間の区切り（学習〜2023・開発 2024・テスト 2025・2026）
PERIODS: tuple[tuple[str, int, int], ...] = (
    ("学習(〜2023)", 0, 2023),
    ("開発(2024)", 2024, 2024),
    ("テスト(2025)", 2025, 2025),
    ("2026", 2026, 2026),
)


def period_of(year: int) -> str:
    for name, lo, hi in PERIODS:
        if lo <= year <= hi:
            return name
    return str(year)


def _cols(columns: Iterable[str] | None, df: pd.DataFrame) -> list[str]:
    return list(df.columns) if columns is None else list(columns)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """race_id・horse_id を文字列にそろえる（category・int の混在で groupby・join が黙って崩れないように）。"""
    out = df.copy()
    for c in ("race_id", "horse_id"):
        if c in out.columns:
            out[c] = out[c].astype("string")
    return out


# ── 欠けの集計 ────────────────────────────────────────────────────────────

def missing_mask(s: pd.Series) -> pd.Series:
    """None・NaN・空文字・空白だけの文字列を欠けとみなす。"""
    m = s.isna()
    if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        st = s.astype("string")
        m = m | st.str.strip().fillna("").eq("")
    return m.fillna(True).astype(bool)


def zero_mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return pd.Series(False, index=s.index)
    if pd.api.types.is_numeric_dtype(s):
        return s.eq(0).fillna(False).astype(bool)
    return pd.Series(False, index=s.index)


def month_key(dates: pd.Series) -> pd.Series:
    """日付（'2025-07-26'・'20250726'・datetime）を 'YYYY-MM' にする。読めなければ NA。"""
    if pd.api.types.is_datetime64_any_dtype(dates):
        d = dates
    else:
        st = dates.astype("string").str.replace("/", "-", regex=False)
        d = pd.to_datetime(st, errors="coerce", format="mixed")
    return d.dt.strftime("%Y-%m").astype("string")


def missing_profile(df: pd.DataFrame, *, date_col: str = "date", columns: Iterable[str] | None = None,
                    month: pd.Series | None = None) -> pd.DataFrame:
    """列 × 月の欠けと 0 の割合（%）。戻り値の列: month, column, rows, missing_pct, zero_pct。"""
    if month is None:
        month = month_key(df[date_col])
    cols = [c for c in _cols(columns, df) if c in df.columns and c != date_col]
    out = []
    g = month.fillna("NA")
    rows = g.value_counts()
    for c in cols:
        s = df[c]
        try:
            miss = missing_mask(s)
        except TypeError:  # 配列など
            miss = s.isna()
        zero = zero_mask(s)
        agg = pd.DataFrame({"m": miss.values, "z": zero.values, "g": g.values}).groupby("g").mean() * 100
        for mo, r in agg.iterrows():
            out.append((mo, c, int(rows.get(mo, 0)), float(r["m"]), float(r["z"])))
    return pd.DataFrame(out, columns=["month", "column", "rows", "missing_pct", "zero_pct"])


JUMP_COLUMNS = ["month", "column", "rows", "value", "prev", "prev_year", "delta"]


def detect_jumps(profile: pd.DataFrame, *, threshold: float = JUMP_PT, min_rows: int = MIN_ROWS,
                 metric: str = "missing_pct", before: str | None = None) -> pd.DataFrame:
    """前月（行数が足りる直前の月）より ``threshold`` 以上増えた月を返す。

    前年同月も同じ水準なら季節・番組の都合として拾わない。ただし前年同月そのものが前年の前月から急に増えていた
    （前年も同じ壊れ方をしていた）なら、前年同月では打ち消さない（再発を拾う）。2 年前の同じ月も同じ水準なら季節とみなす。``before``（'YYYY-MM'）より前の月だけを見る
    （まだ途中の当月を外す）。戻り値: month, column, rows, value, prev, prev_year, delta。
    """
    out = []
    p = profile[(profile["rows"] >= min_rows) & (profile["month"] != "NA")]
    if before:
        p = p[p["month"] < before]
    for col, g in p.groupby("column"):
        g = g.sort_values("month")
        vals = dict(zip(g["month"], g[metric]))
        rows = dict(zip(g["month"], g["rows"]))
        months = list(g["month"])
        idx = {m: i for i, m in enumerate(months)}
        for i, mo in enumerate(months):
            v = vals[mo]
            if i == 0:
                continue
            prev = vals[months[i - 1]]
            y, m = mo.split("-")
            py = f"{int(y) - 1:04d}-{m}"
            prev_year = vals.get(py)
            base = prev
            if prev_year is not None:
                j = idx[py]
                py_jumped = j > 0 and prev_year - vals[months[j - 1]] >= threshold
                p2 = vals.get(f"{int(y) - 2:04d}-{m}")
                seasonal = p2 is not None and abs(p2 - prev_year) < threshold   # 2 年前も同じ水準なら季節・番組の都合
                if not py_jumped or seasonal:
                    base = max(prev, prev_year)
            delta = v - base
            if delta >= threshold:
                out.append((mo, col, int(rows[mo]), round(v, 2), round(prev, 2),
                            None if prev_year is None else round(prev_year, 2), round(delta, 2)))
    return pd.DataFrame(out, columns=JUMP_COLUMNS)


def period_summary(profile: pd.DataFrame, metric: str = "missing_pct") -> pd.DataFrame:
    """月ごとの欠けを、行数で重みを付けて期間ごとにまとめる（列 × 期間）。"""
    p = profile[profile["month"] != "NA"].copy()
    p["period"] = p["month"].str[:4].astype(int).map(period_of)
    p["w"] = p[metric] * p["rows"]
    g = p.groupby(["column", "period"]).agg(w=("w", "sum"), rows=("rows", "sum"))
    g[metric] = (g["w"] / g["rows"]).round(2)
    return g[metric].unstack("period")


# ── 値の範囲・ずれ（行単位） ──────────────────────────────────────────────

# 結果・出馬表の値の範囲（両端を含む）。障害は距離・タイムが長く、上がり 3F は 1F の平均（13〜15 秒）が入るので別に扱う。
RANGES: dict[str, tuple[float, float]] = {
    "horse_number": (1, 18),
    "bracket_number": (1, 8),
    "jockey_weight": (48, 63),
    "distance": (1000, 4300),
    "odds": (1.0, 9999.0),
    "win_odds": (1.0, 9999.0),
    "last_3f": (30.0, 50.0),
    "weight": (300, 650),
    "weight_change": (-50, 50),
    "field_size": (1, 18),
    "round": (1, 12),
}
FLAT_ONLY = {"last_3f"}  # 障害の行は範囲の検査から外す列


def expected_bracket(horse_number: int, n: int) -> int | None:
    """JRA の枠の割り当て（頭数 n の馬番 → 枠番）。9〜16 頭は外の枠から 2 頭、17・18 頭は外の枠から 3 頭。"""
    if not (1 <= horse_number <= n <= 18):
        return None
    if n <= 8:
        return horse_number
    sizes = [1] * 8
    extra = n - 8
    for b in range(7, -1, -1):  # 外の枠から 2 頭にする
        if extra == 0:
            break
        sizes[b] += 1
        extra -= 1
    for b in range(7, -1, -1):  # 17・18 頭は外の枠から 3 頭
        if extra == 0:
            break
        sizes[b] += 1
        extra -= 1
    acc = 0
    for b, k in enumerate(sizes, start=1):
        acc += k
        if horse_number <= acc:
            return b
    return None


@dataclass
class Issue:
    check: str
    table: str
    column: str
    n_bad: int
    n_total: int
    examples: list[str]
    note: str = ""

    @property
    def pct(self) -> float:
        return 0.0 if not self.n_total else round(100.0 * self.n_bad / self.n_total, 3)

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "table": self.table, "column": self.column, "n_bad": self.n_bad,
                "n_total": self.n_total, "pct": self.pct, "examples": self.examples[:5], "note": self.note}


def _examples(df: pd.DataFrame, mask: pd.Series, cols: list[str], k: int = 5) -> list[str]:
    sub = df.loc[mask, list(dict.fromkeys(c for c in cols if c in df.columns))].head(k)
    return [", ".join(f"{c}={r[c]}" for c in sub.columns) for _, r in sub.iterrows()]


def _finished(df: pd.DataFrame) -> pd.Series:
    """着順が数字の行（取消・除外・中止・失格を除く）。着順の無い表（出馬表・オッズ）は全行。"""
    if "finish_position" not in df.columns:
        return pd.Series(True, index=df.index)
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    ok = fp.notna() & (fp > 0)
    if "finish_status" in df.columns:
        fs = df["finish_status"].astype("string").fillna("")
        ok &= fs.isin(["", "finished"])
    return ok


def _obstacle(df: pd.DataFrame) -> pd.Series:
    """障害レースの行（surface=障、またはレース名に障害・ジャンプ・JS）。"""
    ob = pd.Series(False, index=df.index)
    if "surface" in df.columns:
        ob |= df["surface"].astype("string").fillna("").eq("障")
    if "race_name" in df.columns:
        ob |= df["race_name"].astype("string").str.contains(OBSTACLE_RE, regex=True).fillna(False)
    return ob


def settled_rows(df: pd.DataFrame, today: _dt.date | None = None) -> pd.DataFrame:
    """検査の対象の行: 未来のレース（日付が今日より後）と、枠順の発表前（馬番 0・欠け）の行を除く。"""
    keep = pd.Series(True, index=df.index)
    if "date" in df.columns:
        d = pd.to_datetime(df["date"].astype("string"), errors="coerce", format="mixed")
        keep &= ~(d.dt.date > (today or _dt.date.today())).fillna(False)
    if "horse_number" in df.columns:
        keep &= pd.to_numeric(df["horse_number"], errors="coerce").fillna(0) > 0
    return df[keep]


def check_result_rows(df: pd.DataFrame, table: str = "race_result_flat", *, today: _dt.date | None = None) -> list[Issue]:
    """値の範囲と、列の取り違え・1 つずれの兆候（行単位）。"""
    df = settled_rows(_norm(df), today)
    out: list[Issue] = []
    n = len(df)
    key = ["race_id", "horse_number", "horse_id"]
    ob = _obstacle(df)
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        scope = ~ob if col in FLAT_ONLY else pd.Series(True, index=df.index)
        sent = v.isin(SENTINELS) & (lo > 0) & scope     # 印の値は欠けの代わり。範囲外とは別に数える
        bad = scope & v.notna() & ((v < lo) | (v > hi)) & ~sent
        out.append(Issue("range", table, col, int(bad.sum()), int((scope & v.notna()).sum()),
                         _examples(df, bad, key + [col]), f"{lo}〜{hi}" + ("（障害を除く）" if col in FLAT_ONLY else "")))
        if lo > 0:
            out.append(Issue("zero", table, col, int(sent.sum()), int((scope & v.notna()).sum()), _examples(df, sent, key + [col]),
                             f"印の値 {SENTINELS} が欠けの代わりに入っている（数値として使うと誤る）"))
    fin = _finished(df)
    if {"finish_position", "field_size"} <= set(df.columns):
        fp = pd.to_numeric(df["finish_position"], errors="coerce")
        fs = pd.to_numeric(df["field_size"], errors="coerce")
        bad = fin & fs.notna() & (fp > fs)
        out.append(Issue("range", table, "finish_position", int(bad.sum()), int(fin.sum()),
                         _examples(df, bad, key + ["finish_position", "field_size"]), "1〜頭数"))
    if "popularity" in df.columns and "field_size" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        fs = pd.to_numeric(df["field_size"], errors="coerce")
        sent = pop.isin(SENTINELS)
        has = pop.notna() & fs.notna() & ~sent
        bad = has & ((pop < 1) | (pop > fs))
        out.append(Issue("range", table, "popularity", int(bad.sum()), int(has.sum()),
                         _examples(df, bad, key + ["popularity", "field_size"]), "1〜頭数（印の値を除く）"))
        out.append(Issue("zero", table, "popularity", int(sent.sum()), int(pop.notna().sum()), _examples(df, sent, key + ["popularity"]),
                         f"印の値 {SENTINELS} が欠けの代わりに入っている"))
    if {"time_sec", "distance"} <= set(df.columns):
        t = pd.to_numeric(df["time_sec"], errors="coerce")
        dist = pd.to_numeric(df["distance"], errors="coerce")
        per200 = t / (dist / 200.0)
        bad = fin & t.notna() & (t > 0) & ((per200 < 10.5) | (per200 > 17.0))
        out.append(Issue("range", table, "time_sec", int(bad.sum()), int((fin & t.notna()).sum()),
                         _examples(df, bad, key + ["time_sec", "distance"]), "200m あたり 10.5〜17.0 秒"))
    if "date" in df.columns and "race_id" in df.columns:
        d = pd.to_datetime(df["date"].astype("string"), errors="coerce", format="mixed")
        yid = pd.to_numeric(df["race_id"].str[:4], errors="coerce")
        bad = d.isna() | (d.dt.year != yid)
        out.append(Issue("date", table, "date", int(bad.sum()), n,
                         _examples(df, bad, ["race_id", "date"]), "日付が読める・race_id の年と一致"))
    # 枠番と馬番の関係（頭数から決まる）
    if {"horse_number", "bracket_number", "race_id"} <= set(df.columns):
        hn = pd.to_numeric(df["horse_number"], errors="coerce")
        bn = pd.to_numeric(df["bracket_number"], errors="coerce")
        # 頭数は登録の頭数（取消を含む馬番の最大）。最大の馬番の取消行が抜けた表に備えて field_size とも比べる
        nmax = hn.groupby(df["race_id"]).transform("max")
        if "field_size" in df.columns:
            fsz = pd.to_numeric(df["field_size"], errors="coerce")
            nmax = np.fmax(nmax, fsz.where(fsz <= 18))
        exp = pd.Series([expected_bracket(int(h), int(k)) if pd.notna(h) and pd.notna(k) else None
                         for h, k in zip(hn, nmax)], index=df.index, dtype="float")
        has = bn.notna() & (bn > 0) & exp.notna()
        bad = has & bn.ne(exp)
        out.append(Issue("shift", table, "bracket_number", int(bad.sum()), int(has.sum()),
                         _examples(df, bad, ["race_id", "horse_number", "bracket_number"]),
                         "馬番と頭数から決まる枠番と一致"))
    # 人気とオッズ: 人気の順に並べたとき、オッズが逆転しているレース（取消・除外で人気が飛ぶのは可）
    ocol = "odds" if "odds" in df.columns else ("win_odds" if "win_odds" in df.columns else None)
    if ocol and "popularity" in df.columns:
        o = pd.to_numeric(df[ocol], errors="coerce")
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        has = o.notna() & (o > 0) & (o < 9999) & pop.notna() & (pop > 0) & ~pop.isin(SENTINELS)
        sub = pd.DataFrame({"rid": df["race_id"], "o": o, "p": pop})[has].sort_values(["rid", "p", "o"])
        prev_max = sub.groupby("rid")["o"].cummax().groupby(sub["rid"]).shift(1)
        badr = prev_max.notna() & (sub["o"] < prev_max - 1e-9)
        bad_races = sorted(set(sub.loc[badr, "rid"]))
        out.append(Issue("shift", table, f"popularity/{ocol}", len(bad_races), int(sub["rid"].nunique()), bad_races[:5],
                         "人気の順にオッズが逆転するレース数（同じオッズで人気が分かれるのは可）"))
    # 隣の列の値がそのまま入っている兆候（同じ値が大半の行で一致）
    for a, b in (("popularity", "horse_number"), ("bracket_number", "horse_number"),
                 ("finish_position", "horse_number"), ("finish_position", "popularity")):
        if {a, b} <= set(df.columns):
            va = pd.to_numeric(df[a], errors="coerce")
            vb = pd.to_numeric(df[b], errors="coerce")
            eq = (va == vb) & (va > 0) & ~va.isin(SENTINELS)
            fsz = pd.to_numeric(df.get("field_size", pd.Series(np.nan, index=df.index)), errors="coerce")
            if a == "bracket_number":
                eq &= fsz > 8
            per = eq.groupby(df["race_id"]).mean()
            cnt = df.groupby("race_id").size()
            sus = per[(per >= 0.99) & (cnt.reindex(per.index) >= 9)]
            out.append(Issue("shift", table, f"{a}=={b}", int(len(sus)), int(len(per)),
                             [str(x) for x in sus.index[:5]], "9 頭以上のレースで全行が一致（レース数）"))
    return out


def _win_payout(payoff: Any) -> list[tuple[str, float]]:
    if not isinstance(payoff, str) or not payoff.strip():
        return []
    try:
        p = json.loads(payoff)
    except ValueError:
        return []
    t = p.get("単勝") if isinstance(p, dict) else None
    items = t if isinstance(t, list) else ([t] if isinstance(t, dict) else [])
    res = []
    for it in items:
        try:
            nums = str(it.get("numbers", "")).split()
            pay = float(str(it.get("payout", "")).replace(",", ""))
        except (ValueError, AttributeError):
            continue
        for x in nums:
            res.append((x.strip(), pay))
    return res


def check_result_races(df: pd.DataFrame, table: str = "race_result_flat", *, today: _dt.date | None = None) -> list[Issue]:
    """レース単位の整合。"""
    df = settled_rows(_norm(df), today)
    out: list[Issue] = []
    rid = df["race_id"]
    g = df.groupby(rid)
    nr = g.ngroups
    # 同じレースの中で、レースの属性がそろっている
    for col in ("date", "distance", "surface", "track_condition", "weather", "race_class", "grade", "field_size",
                "venue", "start_time"):
        if col in df.columns:
            k = df[col].astype("string").fillna("<NA>").groupby(rid).nunique()
            bad = k[k > 1]
            out.append(Issue("race_const", table, col, int(len(bad)), nr, [str(x) for x in bad.index[:5]],
                             "レースの中で値が 1 つ"))
    # 頭数 = 出走馬の行数
    if "field_size" in df.columns:
        fs = pd.to_numeric(df["field_size"], errors="coerce").groupby(rid).first()
        rows = g.size()
        fin = _finished(df).groupby(rid).sum()
        bad_rows = fs.notna() & (fs != rows)
        bad_fin = fs.notna() & (fs < fin)
        out.append(Issue("race_count", table, "field_size==rows", int(bad_rows.sum()), nr,
                         [f"{r}: field_size={int(fs[r])} rows={rows[r]}" for r in bad_rows[bad_rows].index[:5]],
                         "取消・除外の行を含む表なら、差は取消・除外の分"))
        out.append(Issue("race_count", table, "field_size>=finished", int(bad_fin.sum()), nr,
                         [f"{r}: field_size={int(fs[r])} finished={int(fin[r])}" for r in bad_fin[bad_fin].index[:5]],
                         "完走の頭数が頭数を超えない"))
    # 1 着の確定の単勝オッズ×100 = 単勝の払戻（払戻の列が全部空の表＝速報の表は見ない）
    if {"payoff", "odds", "finish_position", "horse_number"} <= set(df.columns) and not missing_mask(df["payoff"]).all():
        first = df[pd.to_numeric(df["finish_position"], errors="coerce") == 1]
        pay = df.groupby(rid)["payoff"].first()
        n_chk = n_bad = n_nopay = 0
        ex = []
        for _, r in first.iterrows():
            wins = _win_payout(pay.get(r["race_id"]))
            if not wins:
                n_nopay += 1
                continue
            o = pd.to_numeric(r["odds"], errors="coerce")
            hn = pd.to_numeric(r["horse_number"], errors="coerce")
            if pd.isna(o) or o <= 0 or pd.isna(hn):
                continue
            hit = [p for num, p in wins if num == str(int(hn))]
            n_chk += 1
            if not hit:
                n_bad += 1
                ex.append(f"{r['race_id']}: 1着の馬番 {int(hn)} が単勝の払戻に無い")
                continue
            if len(wins) == 1 and abs(o * 100 - hit[0]) > 0.5:  # 同着は払戻が変わるので比べない
                n_bad += 1
                ex.append(f"{r['race_id']}: odds={o} payout={hit[0]}")
        out.append(Issue("payoff", table, "odds*100==単勝払戻", n_bad, n_chk, ex, "同着は馬番の一致だけ"))
        out.append(Issue("payoff", table, "単勝払戻なし", n_nopay, len(first), [], "1 着の行があるのに払戻が読めない"))
    # 列がずれたレース（2026-11-16 の監査）: 完走した馬の 3 頭以上で、馬体重が 300 未満・上がり 3F が 50 超（平地）・人気が頭数超え
    if {"weight", "last_3f", "popularity", "field_size"} <= set(df.columns):
        fin = _finished(df)
        w = pd.to_numeric(df["weight"], errors="coerce")
        l3 = pd.to_numeric(df["last_3f"], errors="coerce")
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        fs = pd.to_numeric(df["field_size"], errors="coerce")
        odd = fin & (((w > 0) & (w < 300)) | ((l3 > 50) & ~_obstacle(df)) | ((pop > fs) & ~pop.isin(SENTINELS)))
        cnt = odd.groupby(rid).sum()
        bad = cnt[cnt >= 3]
        out.append(Issue("shift", table, "列ずれのレース", int(len(bad)), nr, [str(x) for x in bad.index[:5]],
                         "完走 3 頭以上で 馬体重<300・上がり3F>50・人気>頭数（解析の列ずれ。障害に多い）"))
    # 障害レースの芝・ダの取り違え: レース名に「障害」があるのに surface が「障」でない
    if {"race_name", "surface"} <= set(df.columns):
        r1 = df.groupby(rid)[["race_name", "surface"]].first()
        ob = r1["race_name"].astype("string").str.contains(OBSTACLE_RE, regex=True).fillna(False)
        bad = r1[ob & r1["surface"].astype("string").fillna("").ne("障")]
        out.append(Issue("race_const", table, "障害の surface", int(len(bad)), int(ob.sum()),
                         [f"{i}: {r.race_name} surface={r.surface!r}" for i, r in bad.head(5).iterrows()],
                         "障害レースは surface=障"))
    return out


def compare_entries(result: pd.DataFrame, shutuba: pd.DataFrame, table: str = "result_vs_shutuba",
                    *, today: _dt.date | None = None) -> list[Issue]:
    """結果の表と出馬表の馬の一致（両方にあるレースだけ。未来・枠順前の行は除く）。"""
    result = _norm(result)
    shutuba = settled_rows(_norm(shutuba), today)
    rids = set(result["race_id"]) & set(shutuba["race_id"])
    a = result[result["race_id"].isin(rids)].drop_duplicates(["race_id", "horse_id"])
    b = shutuba[shutuba["race_id"].isin(rids)].drop_duplicates(["race_id", "horse_id"])
    ka = set(zip(a["race_id"], a["horse_id"]))
    kb = set(zip(b["race_id"], b["horse_id"]))
    only_r = ka - kb
    only_s = kb - ka
    ma = a.set_index(["race_id", "horse_id"])["horse_number"]
    mb = b.set_index(["race_id", "horse_id"])["horse_number"]
    j = ma.to_frame("r").join(mb.to_frame("s"), how="inner")
    hn_bad = j[pd.to_numeric(j["r"], errors="coerce") != pd.to_numeric(j["s"], errors="coerce")]
    races_r = set(result["race_id"])
    races_s = set(shutuba["race_id"])
    return [
        Issue("entries", table, "結果にだけある馬", len(only_r), len(ka), [f"{r} {h}" for r, h in sorted(only_r)[:5]]),
        Issue("entries", table, "出馬表にだけある馬", len(only_s), len(kb), [f"{r} {h}" for r, h in sorted(only_s)[:5]],
              "取消・除外の馬は結果に無いことがある"),
        Issue("entries", table, "馬番の不一致", len(hn_bad), len(j), [f"{i}" for i in hn_bad.index[:5]]),
        Issue("entries", table, "結果にだけあるレース", len(races_r - races_s), len(races_r),
              sorted(races_r - races_s)[:5]),
        Issue("entries", table, "出馬表にだけあるレース", len(races_s - races_r), len(races_s),
              sorted(races_s - races_r)[:5], "中止・結果の取り込み前のレースを含む"),
    ]


def check_horse_birth(df: pd.DataFrame, table: str = "race_result_flat", *, today: _dt.date | None = None) -> list[Issue]:
    """馬の ID の先頭 4 桁（生年）と、性齢の年齢・レースの年の整合（年ごとの表の中で見る）。"""
    if not {"horse_id", "sex_age", "date"} <= set(df.columns):
        return []
    df = settled_rows(_norm(df), today)
    hid = df["horse_id"]
    birth = pd.to_numeric(hid.str[:4], errors="coerce")
    age = pd.to_numeric(df["sex_age"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    year = pd.to_datetime(df["date"].astype("string"), errors="coerce", format="mixed").dt.year
    has = birth.notna() & age.notna() & year.notna() & hid.str.fullmatch(r"\d{10}").fillna(False)
    bad = has & ((year - birth) != age)
    badid = hid.notna() & ~hid.str.fullmatch(r"\d{10}|[0-9A-Za-z]{10}").fillna(False)
    # 同じ馬の性の一貫（牡・牝の入れ替わり。騸馬化の 牡→セ は数えない）
    sex = df["sex_age"].astype("string").str[0]
    s2 = pd.DataFrame({"h": hid, "s": sex}).dropna()
    s2 = s2[s2["s"] != "セ"]
    flip = s2.groupby("h")["s"].nunique()
    return [
        Issue("horse", table, "生年×馬齢", int(bad.sum()), int(has.sum()),
              _examples(df, bad, ["race_id", "horse_id", "sex_age", "date"]), "レースの年 − ID の生年 = 馬齢"),
        Issue("horse", table, "horse_id の形", int(badid.sum()), int(hid.notna().sum()), _examples(df, badid, ["race_id", "horse_id"]),
              "10 桁"),
        Issue("horse", table, "性の不一致", int((flip > 1).sum()), int(len(flip)), [str(x) for x in flip[flip > 1].index[:5]],
              "牡・牝が同じ馬で入れ替わる（セを除く）"),
    ]


# ── 学習と推論の欠けの比較 ───────────────────────────────────────────────

def column_missing(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.Series:
    cols = [c for c in _cols(columns, df) if c in df.columns]
    return pd.Series({c: float(missing_mask(df[c]).mean() * 100) if len(df) else math.nan for c in cols}, dtype="float64")


def compare_missing(train: pd.DataFrame | pd.Series, serve: pd.DataFrame | pd.Series, *,
                    threshold: float = 10.0, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """学習と推論の各列の欠け（%）を並べ、差が ``threshold`` 以上の列を差の大きい順に返す。

    学習・推論のどちらかにしか無い列は ``only`` に印を付けて含める（片方で列ごと無いのは 100% の欠け）。
    """
    cols_in = None if columns is None else list(columns)
    a = train if isinstance(train, pd.Series) else column_missing(train, cols_in)
    b = serve if isinstance(serve, pd.Series) else column_missing(serve, cols_in)
    cols = sorted(set(a.index) | set(b.index))
    if cols_in is not None:
        keep = set(cols_in)
        cols = [c for c in cols if c in keep]
    rows = []
    for c in cols:
        ta = a.get(c, math.nan)
        sb = b.get(c, math.nan)
        only = "train" if c not in b.index else ("serve" if c not in a.index else "")
        tv = 100.0 if math.isnan(ta) else ta
        sv = 100.0 if math.isnan(sb) else sb
        rows.append((c, round(ta, 2) if not math.isnan(ta) else None, round(sb, 2) if not math.isnan(sb) else None,
                     round(sv - tv, 2), only))
    out = pd.DataFrame(rows, columns=["column", "train_missing_pct", "serve_missing_pct", "delta", "only"])
    out = out[out["delta"].abs() >= threshold]
    return out.reindex(out["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ── CLI ──────────────────────────────────────────────────────────────────

DEFAULT_TABLES = ("race_result_flat", "race_shutuba_flat", "race_odds_flat")
ROW_CHECK_TABLES = {"race_result_flat", "race_shutuba_flat", "race_odds_flat", "race_result_on_time_flat"}


def audit_tables(tables_dir: str | Path, years: Iterable[int], tables: Iterable[str] = DEFAULT_TABLES, *,
                 today: _dt.date | None = None) -> dict[str, Any]:
    """手元の flat の表を年ごとに読んで、欠けの月次・急変・値の範囲・整合の検査をまとめる（GCS は読まない）。

    結果の表と出馬表の両方を読む年は、馬の一致（compare_entries）も見る。急変はまだ途中の当月を外して判定する。
    """
    root = Path(tables_dir)
    today = today or _dt.date.today()
    tables = list(tables)
    issues: list[dict[str, Any]] = []
    profiles: list[pd.DataFrame] = []
    by_year: dict[int, dict[str, pd.DataFrame]] = {}
    for t in tables:
        frames = []
        for y in years:
            p = root / str(y) / f"{t}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            found: list[Issue] = []
            if t in ROW_CHECK_TABLES:
                found = check_result_rows(d, t, today=today) + check_horse_birth(d, t, today=today)
                if t != "race_odds_flat":
                    found += check_result_races(d, t, today=today)
            if t in ("race_result_flat", "race_shutuba_flat"):
                by_year.setdefault(int(y), {})[t] = d
            issues += [{**i.as_dict(), "year": int(y)} for i in found]
            frames.append(d)
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        if "date" in d.columns:
            prof = missing_profile(d)
            prof["table"] = t
            profiles.append(prof)
    for y, tb in sorted(by_year.items()):
        if len(tb) == 2:
            issues += [{**i.as_dict(), "year": y} for i in compare_entries(tb["race_result_flat"], tb["race_shutuba_flat"], today=today)]
    cols = ["month", "column", "rows", "missing_pct", "zero_pct", "table"]
    prof = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=cols)
    jumps = []
    current = today.strftime("%Y-%m")
    for t, g in prof.groupby("table"):
        for metric in ("missing_pct", "zero_pct"):
            j = detect_jumps(g, metric=metric, before=current)
            j["table"] = t
            j["metric"] = metric
            jumps.append(j)
    jumps = [j for j in jumps if len(j)]
    jumps_df = pd.concat(jumps, ignore_index=True) if jumps else pd.DataFrame(columns=JUMP_COLUMNS + ["table", "metric"])
    return {"issues": issues, "profile": prof, "jumps": jumps_df}


def _fmt(v: Any) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v}"


def render_summary(res: dict[str, Any], *, limit: int = 60) -> str:
    """要約。印の値（zero）は後ろに回し、割合の大きい順（少数の本物の不良が埋もれないように表・検査ごとに 1 行）。"""
    lines = ["# 表の中身の監査（table_audit）", ""]
    bad = [i for i in res["issues"] if i["n_bad"] > 0]
    lines.append(f"## 検査で見つかったもの（{len(bad)} 件。表・検査・列・年ごと。印の値 zero は最後）")
    for i in sorted(bad, key=lambda x: (x["check"] == "zero", -x["pct"], x["table"], x["year"]))[:limit]:
        lines.append(f"- {i['table']} {i['year']} {i['check']}:{i['column']} {i['n_bad']}/{i['n_total']}（{i['pct']}%） {i['note']}"
                     f" 例: {'; '.join(i['examples'][:2])}")
    j = res["jumps"]
    lines += ["", f"## 欠け・0 の急増（前月より {JUMP_PT:g} ポイント以上。前年同月も同じ水準なら除く。当月は除く。{len(j)} 件）"]
    for _, r in j.sort_values("delta", ascending=False).head(limit).iterrows():
        what = "0" if r.get("metric") == "zero_pct" else "欠け"
        lines.append(f"- {r['table']} {r['column']} {r['month']}: {what} {r['value']}%（前月 {_fmt(r['prev'])} / 前年同月 {_fmt(r['prev_year'])}、+{r['delta']}）")
    return "\n".join(lines) + "\n"


def parse_years(text: str) -> list[int]:
    """'2016-2026' や '2020,2024-2025' を年の一覧にする（空の区切りは無視）。"""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.config.data_paths import ROOT

    this_year = _dt.date.today().year
    ap = argparse.ArgumentParser(description="手元の flat の表の中身の監査（欠けの急変・値の範囲・列ずれ・整合）。GCS は読まない。")
    ap.add_argument("--tables-dir", default=str(ROOT / "data/page_reference/tables"))
    ap.add_argument("--years", default=f"2016-{this_year}")
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES))
    ap.add_argument("--out", default="", help="出力先（issues.json・jumps.csv・profile.parquet・summary.md）。省略時は要約を表示するだけ")
    ap.add_argument("--train", default="", help="学習の表（parquet）。--serve と一緒に渡すと、列ごとの欠けの差を出す（flat の表の監査はしない）")
    ap.add_argument("--serve", default="", help="推論の入力（parquet の glob。例: data/local/inference/drift_features/2026/*.parquet）")
    ap.add_argument("--threshold", type=float, default=10.0, help="学習と推論の欠けの差（ポイント）がこれ以上の列を出す")
    a = ap.parse_args(argv)
    if bool(a.train) != bool(a.serve):
        ap.error("--train と --serve は両方を渡す")
    if a.train:
        import glob

        files = sorted(glob.glob(a.serve))
        if not files:
            ap.error(f"--serve に当たるファイルが無い: {a.serve}")
        serve = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        train = pd.read_parquet(a.train)
        diff = compare_missing(train, serve, threshold=a.threshold)   # 片方にしか無い列も only に印を付けて出す
        print(diff.to_string(index=False))
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            diff.to_csv(Path(a.out) / "train_vs_serve_missing.csv", index=False)
        return 0
    res = audit_tables(a.tables_dir, parse_years(a.years), [t for t in a.tables.split(",") if t])
    text = render_summary(res)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "issues.json").write_text(json.dumps(res["issues"], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        res["jumps"].to_csv(out / "jumps.csv", index=False)
        res["profile"].to_parquet(out / "profile.parquet", index=False)
        (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
