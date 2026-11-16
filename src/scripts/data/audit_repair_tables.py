"""データ監査（docs/operations/data_audit_20261116.md）の手元で直せる不備を、flat の表の作業用の写しに当てる。

本番の data/page_reference/tables は書き換えない。出力は <out>/<年>/ に、直した表は実体、それ以外は元へのシンボリックリンク。
  python -m src.scripts.data.audit_repair_tables --src data/page_reference/tables --out /home/jovyan/work/retrain_out/data_fix/tables --years 2016-2026

学習の表を直した表で作り直すときは、KEIBA_PAGE_REFERENCE_DIR と実行場所の data/page_reference/tables を、この出力に向ける
（docs/operations/data_audit_20261116.md の §5）。解析・書き出しのコードを直して本番の表を書き出し直したら、この補いは要らなくなる。

当てるもの（race_result_flat・race_result_race）:
- C3 障害の surface: レース名で判定（障害・ジャンプ・(JG・〜JS・スプリングJ。平地のヤングジョッキーズシリーズ「ヤングJS」は除く。
  src/research/columns/compare_series.py の JUMP_NAME_RE と同じ考え方）→「障」
- C4 round: race_id の末尾 2 桁
- C2 列ずれのレース: 完走 3 頭以上で 馬体重<300・上がり3F>50（平地）・人気>頭数 のレースは、odds・popularity・last_3f・weight・weight_change を欠損に
  （値は取り戻せない。着順・タイムは残す）
- C6 障害の完走馬の上がり 3F が 0 → 欠損
- オッズの列だけが別の数字のレース（1 着 1 頭で、単勝の払戻の馬番が 1 着の馬番と同じなのに 単勝オッズ×100 ≠ 払戻）→ そのレースの odds を欠損。
  払戻の人気が 1 着の人気と合わなければ popularity も欠損
- 馬体重が 1〜299 の行（1 頭だけずれた行。中止の馬など）→ weight・weight_change を欠損
記録: <out>/repair_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OB = re.compile(r"障害|ジャンプ|\(JG|(?<!ヤング)JS|スプリングJ")
SHIFT_COLS = ["odds", "popularity", "last_3f", "weight", "weight_change"]


def _empty(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("").isin(["", "None", "nan"])


def obstacle_mask(df: pd.DataFrame) -> pd.Series:
    return df["race_name"].astype("string").str.contains(OB.pattern, regex=True).fillna(False)


def shifted_races(df: pd.DataFrame) -> set[str]:
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    fin = fp.notna() & (fp > 0)
    w = pd.to_numeric(df["weight"], errors="coerce")
    l3 = pd.to_numeric(df["last_3f"], errors="coerce")
    pop = pd.to_numeric(df["popularity"], errors="coerce")
    fs = pd.to_numeric(df["field_size"], errors="coerce")
    ob = obstacle_mask(df)
    odd = fin & (((w > 0) & (w < 300)) | ((l3 > 50) & ~ob) | ((pop > fs) & (pop != 9999)))
    cnt = odd.groupby(df["race_id"].astype(str)).sum()
    return set(cnt[cnt >= 3].index)


def odds_broken_races(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    """オッズの列だけが別の数字のレース（2023 年 福島記念など）と、そのうち人気の列も払戻と合わないレース。

    1 着が 1 頭で、単勝の払戻の馬番が 1 着の馬番と同じなのに、確定の単勝オッズ×100 が払戻と合わないとき。払戻の側がずれている
    レース（単勝の馬番が 1 着と違う。2020 年 202005050104 など）は判定しない（正しいオッズを消さない）。
    """
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    first = df[fp == 1]
    n1 = first.groupby(first["race_id"].astype(str)).size()
    bad, bad_pop = set(), set()
    pay = df.groupby(df["race_id"].astype(str))["payoff"].first()
    for _, r in first.iterrows():
        rid = str(r["race_id"])
        if n1.get(rid, 0) != 1:
            continue
        try:
            t = json.loads(pay.get(rid) or "{}").get("単勝")
            t = t if isinstance(t, dict) else (t[0] if isinstance(t, list) and len(t) == 1 else None)
            p = float(str(t.get("payout", "")).replace(",", "")) if t else None
            num = str(t.get("numbers", "")).strip() if t else ""
            ppop = pd.to_numeric(str(t.get("popularity", "")).strip() or None, errors="coerce") if t else None
        except Exception:
            continue
        hn = pd.to_numeric(r["horse_number"], errors="coerce")
        if not p or pd.isna(hn) or num != str(int(hn)):
            continue
        o = pd.to_numeric(r["odds"], errors="coerce")
        if pd.notna(o) and o > 0 and abs(o * 100 - p) > 0.5:
            bad.add(rid)
            pop = pd.to_numeric(r.get("popularity"), errors="coerce")
            if ppop is not None and pd.notna(ppop) and pd.notna(pop) and int(pop) != int(ppop):
                bad_pop.add(rid)
    return bad, bad_pop


def repair_flat(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    rep = {}
    ob = obstacle_mask(df)
    before = (df["surface"].astype("string") != "障") & ob
    df.loc[ob, "surface"] = "障"
    rep["surface_to_obstacle_rows"] = int(before.sum())
    if "round" in df.columns:
        r = pd.to_numeric(df["race_id"].astype(str).str[-2:], errors="coerce").astype("Int64")
        rep["round_fixed_rows"] = int((pd.to_numeric(df["round"], errors="coerce") != r).sum())
        df["round"] = r
    if {"finish_position", "weight", "last_3f", "popularity", "field_size"} <= set(df.columns):
        bad = shifted_races(df)
        m = df["race_id"].astype(str).isin(bad)
        for c in SHIFT_COLS:
            if c in df.columns:
                if pd.api.types.is_integer_dtype(df[c]):
                    df[c] = df[c].astype("Int64")
                df.loc[m, c] = pd.NA if str(df[c].dtype) == "Int64" else np.nan
        rep["shifted_races"] = sorted(bad)
        if "payoff" in df.columns:
            ob_odds, ob_pop = odds_broken_races(df)
            ob_odds -= bad
            ob_pop -= bad
            df.loc[df["race_id"].astype(str).isin(ob_odds), "odds"] = np.nan
            mp = df["race_id"].astype(str).isin(ob_pop)
            if pd.api.types.is_integer_dtype(df["popularity"]):
                df["popularity"] = df["popularity"].astype("Int64")
            df.loc[mp, "popularity"] = pd.NA
            rep["odds_broken_races"] = sorted(ob_odds)
            rep["popularity_broken_races"] = sorted(ob_pop)
        w = pd.to_numeric(df["weight"], errors="coerce")
        mw = (w > 0) & (w < 300)
        for c in [c for c in ("weight", "weight_change") if c in df.columns]:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("Int64")
            df.loc[mw, c] = pd.NA
        rep["weight_lt300_rows_to_nan"] = int(mw.sum())
        fp = pd.to_numeric(df["finish_position"], errors="coerce")
        z = ob & (fp > 0) & (pd.to_numeric(df["last_3f"], errors="coerce") == 0)
        df.loc[z, "last_3f"] = np.nan
        rep["obstacle_l3_zero_to_nan"] = int(z.sum())
    return df, rep


def repair_race(df: pd.DataFrame, ob_races: set[str]) -> tuple[pd.DataFrame, dict]:
    """race 表は race_class が大きく欠けるので、flat で障害と判定したレースに合わせる。"""
    df = df.copy()
    ob = df["race_id"].astype(str).isin(ob_races)
    rep = {"surface_to_obstacle_races": int(((df["surface"].astype("string") != "障") & ob).sum())}
    df.loc[ob, "surface"] = "障"
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["race_id"].astype(str).str[-2:], errors="coerce").astype("Int64")
    return df, rep


def _link(target: Path, dst: Path) -> None:
    """dst を target へのシンボリックリンクにする（既存のリンク・壊れたリンクは張り直す。実体のファイルは消さない）。"""
    if os.path.lexists(dst):
        if not dst.is_symlink():
            return
        dst.unlink()
    os.symlink(target, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2016-2026")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    if out == src or src in out.parents or out in src.parents:
        ap.error("--out は --src（本番の表）と別の場所にする")
    report = {}
    years = set(range(lo, hi + 1))
    out.mkdir(parents=True, exist_ok=True)
    for sd in sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) not in years):
        _link(sd, out / sd.name)       # 範囲外の年は年のフォルダごと本番へのリンク（黙って空にしない）
    for y in range(lo, hi + 1):
        sd, od = src / str(y), out / str(y)
        if not sd.is_dir():
            continue
        if od.is_symlink():
            od.unlink()                 # 以前の版で年のフォルダごとリンクにしていた場合、その先（本番）に書かない
        od.mkdir(parents=True, exist_ok=True)
        if src in od.resolve().parents or od.resolve() == sd.resolve():
            ap.error(f"出力の年のフォルダが本番の表の下を指している: {od}")
        yr = {}
        ob_races: set[str] = set()
        for f in sorted(sd.iterdir(), key=lambda x: x.name != "race_result_flat.parquet"):
            dst = od / f.name
            if f.name in ("race_result_flat.parquet", "race_result_race.parquet") and dst.is_symlink():
                dst.unlink()            # 本番へのリンクの先に書き込まない
            if f.name == "race_result_flat.parquet":
                df, r = repair_flat(pd.read_parquet(f))
                df.to_parquet(dst, index=False)
                ob_races = set(df.loc[df["surface"] == "障", "race_id"].astype(str))
                yr["flat"] = r
            elif f.name == "race_result_race.parquet":
                df, r = repair_race(pd.read_parquet(f), ob_races)
                df.to_parquet(dst, index=False)
                yr["race"] = r
            else:
                _link(f, dst)
        report[str(y)] = yr
        print(y, json.dumps({k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in yr.items()}, ensure_ascii=False), flush=True)
    (out / "repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # 年の外のファイル（_chunks など）もリンク
    for f in src.iterdir():
        if not f.name[:4].isdigit():
            _link(f, out / f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
