"""実測の馬別 200m ラップ（db.netkeiba.com）から、1 走ごとの走法軸を作る。

定義は docs/ideas/running_map_axes_fixed.md（2026-10-30 確定）。検証に使った host の ts_*.py と同じ式で、
一つのモジュールにまとめたもの。入力は horse_lap の JSONL（1 行 = 1 レース）。

出力列（1 走 1 行）:
  ts        TS（horse_ts.course_ts。レース水準補正のうえコース単位の偏差値）
  sustain   TS持続力（ピーク付近を続けた本数 / 窓の最大本数 の、コース偏差。大 = 持続）
  split     前後半比（案C: 後半3F ÷ 中盤、のコース偏差。大 = 前半型、小 = 後傾）
  debut     出脚（1F目のタイム。50 + 10z）
  gear      ギアチェンジ（窓内で加速した分の合計。50 + 10z）
  endure    消耗耐性（ペース・位置を揃えて、落ちなかったか。50 + 10z、高 = 落ちない）
  front     先行力（1000m/600m 通過の累積タイム順位。前を高く。50 + 10z）
  cr        Cr性能（3角 → 4角の累積タイム順位の押し上げ。50 + 10z）
  makuri    まくり特性（2角 → 4角の累積タイム順位の押し上げ。2角の区間が無いコースは欠測。50 + 10z）
  slope     急坂適性（中山・阪神のみ。高 = 坂で落ちない。50 + 10z）
  abn       異常走（ラスト3Fで 20 秒/200m 超）。印のみで値は消さない
欠測は NaN（0 にしない）。位置系の軸は、区間末の累積タイムのレース内順位から作る（horse_lap の position は横位置なので使わない）。

python -m src.research.race.run_axes --laps horse_lap_2026.jsonl ... --out run_axes.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.research.race import horse_ts as ht
from src.research.race import lap_correction as lc

AXIS_COLS = ["ts", "sustain", "split", "debut", "gear", "endure", "front", "cr", "makuri", "slope"]
BANDS = ("短", "1600", "1800+")
ABNORMAL_PACE = 20.0                       # ラスト3Fのどこかが 20 秒/200m を超えたら異常走
SLOPE_L1 = {("中山", "芝"), ("中山", "ダート"), ("阪神", "芝"), ("阪神", "ダート")}

# TS持続力（ts_sustain15 の「新 正規化」）
_F, _D_LONG, _D_SHORT, _T_GEAR, _DEC_DV, _FLOOR = 40, 12, 9, 20, 10, 45


# ── 読み込み ──────────────────────────────────────────────────────────────

def read_runs(paths: Iterable[Path | str]) -> pd.DataFrame:
    """horse_lap の JSONL から 1 走 1 行。完走（着順あり）・区間 6 本以上・全区間のタイムが正のものだけ。
    同じ race_id の行が複数あれば後の行を使う（日次の追記と手で回す取得が重なったとき、二重に数えない）。ただし中身のある行を
    中身の無い行（empty_ok）で置き換えない。壊れた行は飛ばす。"""
    recs: dict[str, dict] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("race_id"):
                    rid = str(r["race_id"])
                    if r.get("entries") or not (recs.get(rid) or {}).get("entries"):
                        recs[rid] = r
    rows = []
    for r in recs.values():
        ents = r.get("entries", [])
        for e in ents:
            s = e.get("sections", [])
            if not e.get("finish_order") or len(s) < 6:
                continue
            lt = np.array([x.get("lap_time") or np.nan for x in s], float)
            if not np.isfinite(lt).all() or (lt <= 0).any():
                continue
            dist = np.array([x["distance_m"] for x in s], float)
            seg = np.diff(np.concatenate([[0.0], dist]))
            if (seg <= 0).any():
                continue
            corners = [x.get("corner") or "" for x in s]
            i2 = corners.index("2C") if "2C" in corners else -1
            i3 = corners.index("3C") if "3C" in corners else -1
            i4 = corners.index("4C") if "4C" in corners else -1
            if not (0 <= i3 < i4):
                i3 = i4 = -1
            if not (0 <= i2 < i3):
                i2 = -1
            rows.append({
                "race_id": str(r["race_id"]), "horse_id": str(e["horse_id"]), "horse_name": e.get("horse_name", ""),
                "finish_order": int(e["finish_order"]), "nent": len(ents),
                "lt": lt, "dist": dist, "pc": lt / (seg / 200.0), "cum": np.cumsum(lt),
                "len1": float(dist[0]), "i2": i2, "i3": i3, "i4": i4,
            })
    return pd.DataFrame(rows)


# ── 共通の部品 ────────────────────────────────────────────────────────────

def band_of(distance: np.ndarray) -> np.ndarray:
    return np.where(distance <= 1400, "短", np.where(distance < 1800, "1600", "1800+"))


def course_dev(t: pd.DataFrame, col: str, min_n: int = 100) -> pd.Series:
    """コース（場・芝ダ・距離）の平均を引いた偏差。n が足りなければ 芝ダ+距離 → 芝ダ+距離帯(400m)の平均で引く。"""
    mu = pd.Series(np.nan, index=t.index)
    for keys in (ht.COURSE, ["surface", "distance"]):
        g = t.groupby(keys)[col]
        mu = mu.where(mu.notna(), g.transform("mean").where(g.transform("size") >= min_n))
    g = t.groupby(["surface", (t["distance"] // 400).rename("band")])[col]
    mu = mu.where(mu.notna(), g.transform("mean"))
    return t[col] - mu


def _band_std(x: pd.Series, band: np.ndarray, exclude: np.ndarray | None = None) -> pd.Series:
    """距離帯ごとの標準偏差で割る（exclude の走は std の計算から外す）。"""
    z = x.copy()
    ok = x.notna().to_numpy()
    for b in BANDS:
        mk = (band == b) & ok
        use = mk if exclude is None else mk & ~exclude
        sd = x[use].std() if use.sum() > 1 else np.nan
        z[mk] = x[mk] / sd
    return z


def _band_resid(y: pd.Series, X: np.ndarray, band: np.ndarray, exclude: np.ndarray) -> pd.Series:
    """距離帯ごとに y を X（定数項つき）で回帰した残差（係数は exclude を除いて推定し、全体に当てる）。"""
    out = pd.Series(np.nan, index=y.index)
    yv = y.to_numpy(float)
    for b in BANDS:
        mk = band == b
        Xb = np.column_stack([np.ones(mk.sum()), X[mk]])
        good = np.isfinite(Xb).all(1) & np.isfinite(yv[mk])
        fit = good & ~exclude[mk]
        if fit.sum() <= Xb.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(Xb[fit], yv[mk][fit], rcond=None)
        r = np.full(mk.sum(), np.nan)
        r[good] = yv[mk][good] - Xb[good] @ beta
        out[mk] = r
    return out


def _cum_rank_rel(t: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """各走の「区間 idx[i] の末までの累積タイム」を、同じレースの完走馬の中で順位にし、0（先頭）〜1（最後方）に直す。
    horse_lap の sections[].position は順位でなくコース横方向の位置なので、順位はこちらで作る（race_result_flat の
    通過順位と 0.94〜0.98 で一致）。idx が範囲外（-1 など）の走は NaN。同タイムは平均順位。"""
    v = np.round(np.array([c[j] if 0 <= j < len(c) else np.nan for c, j in zip(t["cum"], idx)], float), 2)   # 0.1 秒刻みの足し算の誤差で同タイムが割れないように
    s = pd.Series(v, index=t.index)
    g = s.groupby(t["race_id"])
    rank = g.rank(method="average")
    cnt = g.transform("count")
    return ((rank - 1) / np.maximum(cnt - 1, 1)).where(s.notna() & (cnt >= 2)).to_numpy(float)


# ── TS（レース水準補正つき） ─────────────────────────────────────────────

def with_ts(runs: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """runs に レース情報・水準のずれ（offset）を付け、horse_ts.course_ts で ts・mu_s・sd_s を足す。"""
    races = lc.race_table(years)[["race_id", "date", "venue", "surface", "distance", "track_condition"]]
    races = races.assign(race_id=races["race_id"].astype(str))
    off = ht.race_offsets(years)
    off = off.assign(race_id=off["race_id"].astype(str))
    h = runs.merge(races, on="race_id", how="inner").merge(off, on="race_id", how="left")
    h["peak"] = [float(x[-3:].min()) for x in h["lt"]]
    h = ht.course_ts(h)
    return h.reset_index(drop=True)


# ── 各軸 ──────────────────────────────────────────────────────────────────

def _sustain_one(lt: np.ndarray, ts: float, dist: float, lp: np.ndarray) -> float:
    """TS持続力の本数 / 窓の最大本数。lt = L5..L1 のラップTS、lp = L5..L1 の生ラップ（秒）。"""
    short = dist <= 1400
    lo = 2 if short else (1 if dist < 1800 else 0)
    d = _D_SHORT if short else _D_LONG
    mx = 3 if short else (4 if dist < 1800 else 5)
    if ts < _FLOOR:
        return 0.0
    ok = (lt >= ts - d) & (lt >= _F)
    ok[:lo] = False
    i0 = 2 + int(np.argmax(lt[2:]))
    if not ok[i0]:
        return 0.0
    l = r = i0
    while l - 1 >= lo and ok[l - 1]:
        l -= 1
    while r + 1 < 5 and ok[r + 1]:
        r += 1
    if np.all(np.diff(lp) >= 0.0) and lt[0] - ts >= _DEC_DV:      # 単調減速で L5 が TS 近く
        return 0.0
    blk = np.zeros(5, bool)
    blk[l:r + 1] = True
    bonus = 0.5 if (not blk[1] and lt[2] - lt[1] >= _T_GEAR) else 0.0
    return (blk.sum() + bonus) / mx


def _gear_one(lp: np.ndarray, short: bool) -> float:
    w = lp[2:] if short else lp
    return float(np.clip(w[:-1] - w[1:], 0, None).sum())


def _debut(t: pd.DataFrame) -> pd.Series:
    """出脚: 1F 目のタイムを、(場+芝ダ+距離+初区間長 → 芝ダ+距離帯+初区間長 → 芝ダ+初区間長) のグループ内で median/MAD 標準化し、
    レース平均 z の半分を引く。遅いほど z が正なので符号を反転して 50 + 10z。"""
    x1 = pd.Series([lt[0] for lt in t["lt"]], index=t.index)
    surf, l1 = t["surface"].astype(str), t["len1"].astype(int).astype(str)
    kA = t["venue"].astype(str) + surf + t["distance"].astype(int).astype(str) + "_" + l1
    kB = surf + (t["distance"] // 400).astype(int).astype(str) + "_" + l1
    kC = surf + "_" + l1
    nA, nB = kA.map(kA.value_counts()), kB.map(kB.value_counts())
    gkey = pd.Series(np.where(nA >= 100, "A" + kA, np.where(nB >= 100, "B" + kB, "C" + kC)), index=t.index)
    g = x1.groupby(gkey)
    med = g.transform("median")
    mad = (x1 - med).abs().groupby(gkey).transform("median") * 1.4826
    z = (x1 - med) / mad.replace(0, np.nan)
    z_race = z.groupby(t["race_id"]).transform("mean")
    return 50 + 10 * (-(z - 0.5 * z_race))


def _race_mean_excl(x: pd.Series, t: pd.DataFrame, exclude: np.ndarray) -> pd.Series:
    return x.where(~exclude).groupby(t["race_id"]).transform("mean")


def axes_from_frame(t: pd.DataFrame) -> pd.DataFrame:
    """ts・offset・mu_s・sd_s を持つ 1 走 1 行の表から全軸を作る。t は race_id, horse_id, date, venue, surface, distance,
    finish_order, nent, lt, dist, pc, cum, len1, i2, i3, i4, offset, mu_s, sd_s, ts を持つこと。"""
    t = t.reset_index(drop=True).copy()
    n = len(t)
    dist_m = t["distance"].to_numpy(float)
    band = band_of(dist_m)
    short = dist_m <= 1400
    bad = np.array([pc[-3:].max() > ABNORMAL_PACE for pc in t["pc"]])
    out = t[["race_id", "horse_id", "date", "venue", "surface", "distance", "track_condition", "finish_order", "ts"]].copy()
    out["band"] = band
    out["abn"] = bad

    LP = np.array([lt[-5:] for lt in t["lt"]], float)                     # L5..L1（秒）
    LT = 50 - 10 * ((LP - t["offset"].to_numpy()[:, None]) - t["mu_s"].to_numpy()[:, None]) / t["sd_s"].to_numpy()[:, None]
    ts = t["ts"].to_numpy(float)

    # TS持続力
    t["sustain_raw"] = [_sustain_one(LT[i], ts[i], dist_m[i], LP[i]) for i in range(n)]
    out["sustain"] = course_dev(t, "sustain_raw")

    # 前後半比（案C）
    t["split_raw"] = [pc[-3:].mean() / pc[1:-3].mean() for pc in t["pc"]]
    out["split"] = course_dev(t, "split_raw")

    # 出脚
    out["debut"] = _debut(t)

    # ギアチェンジ
    t["gear_raw"] = [_gear_one(LP[i], bool(short[i])) for i in range(n)]
    gd = course_dev(t, "gear_raw")
    gx = gd - gd.groupby(t["race_id"]).transform("mean")
    out["gear"] = 50 + 10 * _band_std(gx, band)

    # 通過順位（累積タイム順位）・ペース（消耗耐性の共変量、先行力・Cr・まくり）
    mark = np.where(short, 600.0, 1000.0)
    passt, j_mark = [], []
    for i in range(n):
        dist, cum = t["dist"].iat[i], t["cum"].iat[i]
        passt.append(float(np.interp(mark[i], np.concatenate([[0.0], dist]), np.concatenate([[0.0], cum]))))
        j_mark.append(int(np.argmin(np.abs(dist - mark[i]))))
    t["passt"] = passt
    t["relpos"] = _cum_rank_rel(t, np.array(j_mark))
    r2 = _cum_rank_rel(t, t["i2"].to_numpy(int))
    r3 = _cum_rank_rel(t, t["i3"].to_numpy(int))
    r4 = _cum_rank_rel(t, t["i4"].to_numpy(int))
    t["pass_med"] = t.groupby("race_id")["passt"].transform("median")
    t["own_rel"] = t["passt"] - t["pass_med"]
    t["rp"] = course_dev(t, "pass_med")

    # 消耗耐性（共変量はペースと自分の通過差だけ。位置取りの順位は入れない）
    mid = np.array([pc[1:-2].mean() for pc in t["pc"]])
    t["D21"] = (LP[:, 4] + LP[:, 3]) / 2 - mid
    resid = _band_resid(course_dev(t, "D21"), t[["rp", "own_rel"]].to_numpy(float), band, bad)
    r_e = resid - _race_mean_excl(resid, t, bad)
    out["endure"] = 50 - 10 * _band_std(r_e, band, bad)

    # 先行力 / Cr性能 / まくり特性（どれも累積タイム順位から）
    t["front_raw"] = -t["relpos"]
    out["front"] = 50 + 10 * _band_std(course_dev(t, "front_raw"), band)
    t["cr_raw"] = r3 - r4                       # 3C→4C で順位を上げた分（正 = 上げた）
    out["cr"] = 50 + 10 * _band_std(course_dev(t, "cr_raw"), band)
    t["mk_raw"] = r2 - r4                       # 2C→4C で順位を上げた分（2C の区間が無いコースは欠測）
    out["makuri"] = 50 + 10 * _band_std(course_dev(t, "mk_raw"), band)

    # 急坂適性（中山・阪神の L1 のみ）
    is_slope = np.array([(v, s) in SLOPE_L1 for v, s in zip(t["venue"], t["surface"])])
    sl = np.full(n, np.nan)
    for i in np.where(is_slope)[0]:
        pc = t["pc"].iat[i]
        if len(pc) >= 5:
            sl[i] = pc[-1] - 0.5 * (pc[-2] + pc[-3])
    t["sl_raw"] = sl
    x = t["sl_raw"].where(is_slope & ~bad)
    keys3, keys2 = [t["venue"], t["surface"], t["distance"]], [t["venue"], t["surface"]]
    mu = x.groupby(keys3).transform("mean").where(x.groupby(keys3).transform("count") >= 25, x.groupby(keys2).transform("mean"))
    d = (t["sl_raw"] - mu).where(is_slope)
    d = d - _race_mean_excl(d, t, bad)
    out["slope"] = 50 - 10 * _band_std(d, band, bad)

    return out


def default_years() -> list[int]:
    """TS のレース水準補正に使う結果の年。2020 年から今年まで（年が変わっても書き直さなくてよいように今年から決める）。"""
    from datetime import date

    return list(range(2020, date.today().year + 1))


def compute(paths: Iterable[Path | str], years: list[int] | None = None) -> pd.DataFrame:
    years = years or default_years()
    runs = read_runs(paths)
    return axes_from_frame(with_ts(runs, years))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", nargs="+", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=None, help="既定は 2020 年から今年まで")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = compute(a.laps, a.years)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df)} 走 / {df['race_id'].nunique()} レース → {a.out}")
    print(df[AXIS_COLS].describe().round(2).T.to_string())
    print("欠測率(%):", (df[AXIS_COLS].isna().mean() * 100).round(1).to_dict())


if __name__ == "__main__":
    main()
