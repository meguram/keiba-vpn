"""全出走について、「その走の直前時点」の走法軸の 10 点満点（クラス基準。axis_score10.py）を一覧にして保存する。

python -m src.research.race.build_axis_scores10 [--no-update-reference]

出力（どちらも GCS のミラー対象外。ローカルだけ）:
  data/local/running_profile/axis_scores10.parquet        … 各レースの「直前」時点の点（1 走 1 行。履歴が変わるので同じ馬でもレースごとに値が違う）
  data/local/running_profile/axis_scores10_latest.parquet … 最新レースが終わった時点の点（1 頭 1 行。全履歴を反映。次のレースの採点に使う値）

履歴の表の列: race_id, horse_id, date, venue, surface, distance, grade, finish_position,
  n_prior_runs（その走より前の、TS のある走数。多いほど確かな値）, n_pos_runs（同じく位置系の軸が測れた走数）,
  rel_ts・rel_lap・rel_pos（走数から見た確かさ 0〜1。TS／持続力・出脚・ギア／Hペース耐性・先行力・Cr・まくり）,
  n_complete（9 軸のうち点が出た数）, complete（9 軸すべて出た）,
  点の列: ts, sustain, debut, gear, endure, basic, front, cr, makuri, slope（無ければ NaN。0 や 5 は入れない）。
  基礎スピード・道悪・急坂には確かさの列を付けない（測れる走が条件で限られるため、走数でなく n_* を見る）。

方式（as-of。その走より前の走だけから作る）:
  - 軸（TS 以外）: rp.profile_table は「その走の直前のプロファイル」を、その軸が測れた走の行にだけ返す（測れなかった行と、
    測れた最初の走の行は NaN）。直前の値は「その走以降で最初に測れた行の値」と同じ（間に測れた走が無いので、使われる走が同じ）。
    ただし最初に測れた走の行は履歴が無いので不明のまま（後ろの値で埋めない）。最後に測れた走より後の行は、最新のプロファイル
    （profiles_as_of の最終日翌日時点。その軸の全走を反映）で埋める。
  - TS: その走より前の直近 10 走の上位 2 走平均（ピーク）。
  - 近似・将来情報が入る所（学習用の特徴量にするときは注意）:
    (a) 引き寄せ先の父の平均・全馬の平均が全期間の値（rp.profile_table の docstring どおり。父の走が 30 未満や父が不明のときの全馬平均も同じ）。
    引き寄せの強さ c は軸ごと（ax_* の父系は 2、道悪は 3、基礎スピードは 0 へ、急坂は引き寄せ先なし）。
    (b) 10 点への換算の基準（mu・sd・k・TS の節点）は、全期間の好走走で作る（各走の「直後」の値と着順を使う）。履歴の全行がこの基準で換算される。
  - 基準の更新: 既定で reference.json の score10 を、この表と同じ基準に置き換える（一時ファイルから置き換え）。built_from・分位・
    コースの地図は変えない（版も変えない）ので、すでに動いている別のプロセスは再起動するまで古い基準を持つ。
    --no-update-reference だと、画面（reference.json の基準）とこの表の点が合わなくなる。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.research.race import axis_score10 as s10
from src.research.race import running_map as rm
from src.research.race import running_profile as rp

OUT = rp.STORE_DIR / "axis_scores10.parquet"
OUT_LATEST = rp.STORE_DIR / "axis_scores10_latest.parquet"
RHO_TS, RHO_OTHER = 0.39, 0.20   # 1 走あたりの再現性（2026-11-11 の偶奇分割の実測。docs/ideas/running_axes_scale_1to10.md 課題6）。走数 n の確かさ = nρ/(1+(n-1)ρ)
MAP_SCORE_KEYS = ["ts", "sustain", "debut", "gear", "endure", "basic", "front", "cr", "makuri"]   # 完備の判定に使う 9 軸（急坂は中山・阪神だけなので除く）
LAP_KEYS = ["sustain", "debut", "gear"]                  # 実測ラップがあれば測れる軸（TS と同じ走数）
POS_KEYS = ["endure", "front", "cr", "makuri"]           # 通過位置が要る軸（古い走は位置が無く測れない）
_SENT = -1e18                                            # 「測れた最初の走（履歴なし）」の目印


def _rel(n: pd.Series, rho: float) -> pd.Series:
    n = n.clip(upper=s10.PEAK_LOOKBACK)   # 値に効くのは直近 10 走まで
    return (n * rho / (1 + (n - 1).clip(lower=0) * rho)).round(3)


def _score_cols(prof_or_pre: dict[str, pd.Series], peak: pd.Series, ref: dict) -> pd.DataFrame:
    out = {}
    for k, c, *_ in rm.AXES:
        out[k] = s10.score_series(k, peak if k == s10.TS_KEY else prof_or_pre[c], ref)
    out["slope"] = s10.score_series("slope", prof_or_pre["ax_slope"], ref)
    return pd.DataFrame(out)


def build_history(metrics: pd.DataFrame, table: pd.DataFrame, latest_prof: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """各走の直前時点の点。table = rp.profile_table の結果、latest_prof = 最終日翌日の rp.profiles_as_of（horse_id 索引）。"""
    keys = ["race_id", "horse_id", "date"]
    cols = [c for _, c, *_ in rm.AXES] + ["ax_slope"]
    t = table.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    ok = t[keys].merge(metrics[keys + cols].drop_duplicates(keys).set_index(keys).notna().reset_index(), on=keys, how="left")
    h = t["horse_id"]
    pre: dict[str, pd.Series] = {}
    for c in cols:
        valid = ok[c].fillna(False).astype(bool)
        v = t[c].where(valid).mask(valid & t[c].isna(), _SENT)      # 測れた最初の走（履歴なし）は目印。測れなかった行は NaN のまま
        filled = v.groupby(h).bfill()                                # その走以降で最初に測れた行の値（= その走の直前までの走で作った値）
        trailing = filled.isna()                                     # 後ろに測れた行が無い = 最後に測れた走より後の行
        p = filled.where(filled != _SENT)
        if c in latest_prof.columns:                                 # 最新のプロファイル（その軸の全走を反映）で埋める。測れた走が 1 つも無い馬は NaN のまま
            own = latest_prof[f"n_{c}"] > 0
            lv = latest_prof[c].where(own)
            p = p.where(~trailing, h.map(lv))
        pre[c] = p
    # TS のピーク（その走より前の走だけ）
    pk = t[keys].merge(s10.peak_ts_runs(metrics), on=keys, how="left")
    post = pk.groupby("horse_id")["peak_ts"].ffill()
    peak_pre = post.groupby(t["horse_id"]).shift(1)
    n_ts = pk["peak_ts"].notna().astype(int)
    t["n_prior_runs"] = n_ts.groupby(t["horse_id"]).cumsum() - n_ts
    pos_ok = ok["ax_endure"].fillna(False).astype(int)
    t["n_pos_runs"] = pos_ok.groupby(t["horse_id"]).cumsum() - pos_ok
    t["rel_ts"], t["rel_lap"], t["rel_pos"] = _rel(t["n_prior_runs"], RHO_TS), _rel(t["n_prior_runs"], RHO_OTHER), _rel(t["n_pos_runs"], RHO_OTHER)
    sc = _score_cols(pre, peak_pre, ref)
    t = pd.concat([t[keys + ["n_prior_runs", "n_pos_runs", "rel_ts", "rel_lap", "rel_pos"]], sc], axis=1)
    t["n_complete"] = t[MAP_SCORE_KEYS].notna().sum(axis=1)
    t["complete"] = t["n_complete"] == len(MAP_SCORE_KEYS)
    info = metrics[keys + ["venue", "surface", "distance", "finish_position"]].drop_duplicates(keys)
    t = t.merge(info, on=keys, how="left")
    grades = s10.load_grades(sorted(int(y) for y in t["date"].dt.year.unique()))
    t["grade"] = t["race_id"].astype(str).map(grades)
    first = ["race_id", "horse_id", "date", "venue", "surface", "distance", "grade", "finish_position",
             "n_prior_runs", "n_pos_runs", "rel_ts", "rel_lap", "rel_pos", "n_complete", "complete"]
    return t[first + [k for k, *_ in rm.AXES] + ["slope"]].sort_values(["date", "race_id", "horse_id"]).reset_index(drop=True)


def latest_profiles(metrics: pd.DataFrame, sires: pd.DataFrame, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """最新レースが終わった時点（データの最終日の翌日）の、馬ごとのプロファイル（horse_id 索引）。"""
    as_of = (metrics["date"].max() + pd.Timedelta(days=1)) if as_of is None else pd.Timestamp(as_of)
    ids = sorted(metrics.loc[metrics["date"] < as_of, "horse_id"].unique())
    cols = [c for _, c, *_ in rm.AXES] + ["ax_slope"]
    prof = rp.profiles_as_of(metrics, ids, as_of, sires, cols=cols, priors=rp.priors_as_of(metrics, as_of, sires, cols)).set_index("horse_id")
    return prof, as_of


def build_latest(metrics: pd.DataFrame, prof: pd.DataFrame, as_of: pd.Timestamp, ref: dict) -> pd.DataFrame:
    """最新レースが終わった時点の点。全履歴を反映（as_of より前の走だけ＝全走）。1 頭 1 行。
    自分の走が 1 つも無い軸（父の平均だけで決まる値）は、履歴の表と同じく NaN（0 や 5 を入れない）。"""
    ids = list(prof.index)
    past = metrics[metrics["date"] < as_of]
    out = pd.DataFrame({"horse_id": ids, "as_of": as_of, "last_run_date": pd.to_datetime(prof["last_run_date"]).to_numpy(), "n_runs": prof["n_runs"].to_numpy()})
    ts_n = past.dropna(subset=["ax_ts"]).groupby("horse_id").size()
    pos_n = past.dropna(subset=["ax_endure"]).groupby("horse_id").size()
    out["n_prior_runs"] = out["horse_id"].map(ts_n).fillna(0).astype(int)
    out["n_pos_runs"] = out["horse_id"].map(pos_n).fillna(0).astype(int)
    out["rel_ts"], out["rel_lap"], out["rel_pos"] = _rel(out["n_prior_runs"], RHO_TS), _rel(out["n_prior_runs"], RHO_OTHER), _rel(out["n_pos_runs"], RHO_OTHER)
    pk = s10.peak_ts_as_of(metrics, ids, as_of).reset_index(drop=True)
    vals = {c: prof[c].where(prof[f"n_{c}"] > 0).reset_index(drop=True) for _, c, *_ in rm.AXES}
    vals["ax_slope"] = prof["ax_slope"].where(prof["n_ax_slope"] > 0).reset_index(drop=True)
    sc = _score_cols(vals, pk, ref)
    out = pd.concat([out, sc], axis=1)
    out["n_complete"] = out[MAP_SCORE_KEYS].notna().sum(axis=1)
    out["complete"] = out["n_complete"] == len(MAP_SCORE_KEYS)
    try:   # 見やすさのための馬名（実測ラップの取れた馬だけ）
        e = pd.read_parquet(rp.REPO / "data/local/horse_lap/horse_lap_entries.parquet", columns=["horse_id", "horse_name"]).astype(str)
        out["horse_name"] = out["horse_id"].map(e.drop_duplicates("horse_id").set_index("horse_id")["horse_name"])
    except Exception:
        out["horse_name"] = None
    first = ["horse_id", "horse_name", "as_of", "last_run_date", "n_runs", "n_prior_runs", "n_pos_runs", "rel_ts", "rel_lap", "rel_pos", "n_complete", "complete"]
    return out[first + [k for k, *_ in rm.AXES] + ["slope"]]


def _write_reference_score10(ref: dict) -> None:
    p = rm.REFERENCE_DIR / "reference.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["score10"] = ref
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)   # 途中で落ちても reference.json を壊さない


def build(update_reference: bool = True) -> pd.DataFrame:
    metrics = rp.load_metrics()
    sires = rp._sire_map()
    cols = [c for _, c, *_ in rm.AXES] + ["ax_slope"]
    table = rp.profile_table(metrics, sires, cols=cols)
    ref = rm._build_score10(table, metrics)
    if ref is None:
        raise RuntimeError("10 点換算の基準を作れない（格付けが取れない）")
    prof, as_of = latest_profiles(metrics, sires)
    hist = build_history(metrics, table, prof, ref)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(OUT, index=False)
    build_latest(metrics, prof, as_of, ref).to_parquet(OUT_LATEST, index=False)
    if update_reference:
        _write_reference_score10(ref)
    return hist


def coverage(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(t["date"].dt.year)
    return pd.DataFrame({"走数": g.size(), "TSあり%": g["ts"].apply(lambda s: round(100 * s.notna().mean(), 1)),
                         "9軸そろう%": g["complete"].apply(lambda s: round(100 * s.mean(), 1)),
                         "位置系(Hペース耐性)あり%": g["endure"].apply(lambda s: round(100 * s.notna().mean(), 1))})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update-reference", action="store_true", help="reference.json の score10 を置き換えない（画面とこの表の点が合わなくなる）")
    a = ap.parse_args()
    t = build(not a.no_update_reference)
    lt = pd.read_parquet(OUT_LATEST)
    print(f"最新（{lt['as_of'].iloc[0].date()} 時点）: {len(lt)} 頭、9軸そろう {int(lt['complete'].sum())} 頭 → {OUT_LATEST}")
    print(f"{len(t)} 走 / {t['race_id'].nunique()} レース / {t['horse_id'].nunique()} 頭 → {OUT}")
    print(coverage(t).to_string())


if __name__ == "__main__":
    main()
