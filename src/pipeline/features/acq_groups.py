"""新規データ候補の特徴量群（非 JRA 過去走 NJ、母側のきょうだい SIB）。どれも『その日より前の走』だけで作る（当日・未来は見ない）。

入力 runs: horse_id, race_id, date, is_jra, field_size, finish_position, time_sec（acq_history_extract の runs_all）
     keys: race_id, horse_id, date（特徴量を付ける行）
     ped : horse_id, sire_id, dam_id
市場の値（オッズ・人気）は使わない。馬・母・父はすべて ID でつなぐ。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

A_WIN, B_WIN = 0.3, 3.0      # 勝率の縮約（(勝+0.3)/(走+3)）
A_TOP3, B_TOP3 = 0.9, 3.0    # 複勝率の縮約


def _prep_runs(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r.dropna(subset=["date"])
    fs = pd.to_numeric(r["field_size"], errors="coerce")
    fp = pd.to_numeric(r["finish_position"], errors="coerce")
    ok = fs.gt(0) & fp.gt(0)
    r["win"] = (ok & fp.eq(1)).astype("int32")
    r["top3"] = (ok & fp.le(3)).astype("int32")
    r["valid"] = ok.astype("int32")
    r["pct"] = np.where(ok, (fp - 1) / np.maximum(fs - 1, 1), np.nan)
    r["pct0"] = np.nan_to_num(r["pct"], nan=0.0)
    return r.sort_values(["date"], kind="stable").reset_index(drop=True)


def cum_asof(table: pd.DataFrame, by: str, cols: list[str], keys: pd.DataFrame, key_by: str | None = None) -> pd.DataFrame:
    """table を by ごとに日付順に累積し、keys の各行について『keys.date より厳密に前』の最後の累積値を返す（行は keys と同順）。"""
    t = table.sort_values([by, "date"], kind="stable")[[by, "date"] + cols].copy()
    t[cols] = t.groupby(by, sort=False)[cols].cumsum()
    t = t.drop_duplicates([by, "date"], keep="last").sort_values("date", kind="stable")
    kb = key_by or by
    k = keys[[kb, "date"]].copy()
    k["_i"] = np.arange(len(k))
    k = k.rename(columns={kb: by}).sort_values("date", kind="stable")
    m = pd.merge_asof(k, t, on="date", by=by, allow_exact_matches=False)
    m = m.sort_values("_i")
    return m[cols].reset_index(drop=True).fillna(0.0)


def _rate(w, n, a, b):
    return (w + a) / (n + b)


def nonjra_features(runs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """NJ: 非 JRA（地方・海外）の過去走を含めた経歴。"""
    r = _prep_runs(runs)
    keys = keys.copy()
    keys["date"] = pd.to_datetime(keys["date"])
    cols = ["one", "valid", "win", "top3", "pct0"]
    r["one"] = 1
    allc = cum_asof(r, "horse_id", cols, keys)
    nj = r[r["is_jra"] == 0]
    njc = cum_asof(nj, "horse_id", cols, keys)
    jr = r[r["is_jra"] == 1]
    jc = cum_asof(jr, "horse_id", cols, keys)

    out = pd.DataFrame(index=range(len(keys)))
    out["nj_prior_n"] = njc["one"].to_numpy()
    out["nj_all_n"] = allc["one"].to_numpy()
    out["nj_jra_n"] = jc["one"].to_numpy()
    out["nj_share"] = np.where(allc["one"] > 0, njc["one"] / np.maximum(allc["one"], 1), np.nan)
    out["nj_prior_win_rate"] = np.where(njc["one"] > 0, _rate(njc["win"], njc["valid"], A_WIN, B_WIN), np.nan)
    out["nj_prior_top3_rate"] = np.where(njc["one"] > 0, _rate(njc["top3"], njc["valid"], A_TOP3, B_TOP3), np.nan)
    out["nj_prior_mean_pct"] = np.where(njc["valid"] > 0, njc["pct0"] / np.maximum(njc["valid"], 1), np.nan)
    out["nj_all_win_rate"] = np.where(allc["one"] > 0, _rate(allc["win"], allc["valid"], A_WIN, B_WIN), np.nan)
    out["nj_all_top3_rate"] = np.where(allc["one"] > 0, _rate(allc["top3"], allc["valid"], A_TOP3, B_TOP3), np.nan)
    out["nj_transfer_in"] = ((jc["one"] == 0) & (njc["one"] > 0)).astype("int8").to_numpy()

    # 直前の走（種別を問わない）と、直前の JRA 走・非 JRA 走からの日数
    def last_info(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
        s = sub.sort_values("date", kind="stable")[["horse_id", "date", "pct"]].rename(columns={"date": "_ld", "pct": f"{tag}_pct"})
        s["date"] = s["_ld"]
        k = keys[["horse_id", "date"]].copy()
        k["_i"] = np.arange(len(k))
        k = k.sort_values("date", kind="stable")
        s = s.drop_duplicates(["horse_id", "date"], keep="last")
        m = pd.merge_asof(k, s, on="date", by="horse_id", allow_exact_matches=False).sort_values("_i")
        return pd.DataFrame({f"{tag}_days": (m["date"] - m["_ld"]).dt.days.to_numpy(), f"{tag}_pct": m[f"{tag}_pct"].to_numpy()})

    la, lj, ln = last_info(r, "la"), last_info(jr, "lj"), last_info(nj, "ln")
    out["nj_days_last_nonjra"] = ln["ln_days"].to_numpy()
    out["nj_last_nonjra_pct"] = ln["ln_pct"].to_numpy()
    out["nj_last_is_nonjra"] = ((la["la_days"] == ln["ln_days"]) & la["la_days"].notna() & (la["la_days"] < lj["lj_days"].fillna(1e9))).astype("int8").to_numpy()
    out["nj_days_last_any"] = la["la_days"].to_numpy()
    out["nj_days_last_jra"] = lj["lj_days"].to_numpy()
    out["nj_gap_any_vs_jra"] = (lj["lj_days"] - la["la_days"]).to_numpy()
    out.index = keys.index
    return out.astype("float32")


def sibling_features(runs: pd.DataFrame, ped: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """SIB: 同じ母（全きょうだい）と、同じ母・同じ父（全きょうだい）の、その日より前の成績。自分自身は除く。"""
    r = _prep_runs(runs)
    r = r[r["is_jra"] == 1]  # きょうだいの比較は JRA の走に限る（地方は条件差が大きい）
    r = r.merge(ped[["horse_id", "sire_id", "dam_id"]], on="horse_id", how="inner")
    r = r[r["dam_id"].astype(str).str.len() > 0].copy()
    r["pair_id"] = r["dam_id"] + "|" + r["sire_id"]
    r["one"] = 1
    # 馬ごとの『初めて走った』『初めて勝った』フラグ（きょうだいの頭数を数えるため）
    r = r.sort_values(["horse_id", "date"], kind="stable")
    r["first_run"] = (r.groupby("horse_id").cumcount() == 0).astype("int32")
    r["win_cum_h"] = r.groupby("horse_id")["win"].cumsum()
    r["first_win"] = ((r["win"] == 1) & (r["win_cum_h"] == 1)).astype("int32")
    cols = ["one", "valid", "win", "top3", "pct0", "first_run", "first_win"]

    k = keys[["race_id", "horse_id", "date"]].copy()
    k["date"] = pd.to_datetime(k["date"])
    k = k.merge(ped[["horse_id", "sire_id", "dam_id"]], on="horse_id", how="left")
    k["dam_id"] = k["dam_id"].fillna("")
    k["pair_id"] = k["dam_id"] + "|" + k["sire_id"].fillna("")
    k.index = keys.index

    own = cum_asof(r, "horse_id", cols, k)
    out = pd.DataFrame(index=range(len(k)))
    for tag, gcol in (("sib", "dam_id"), ("fsib", "pair_id")):
        g = cum_asof(r, gcol, cols, k)
        g = g.to_numpy() - own.to_numpy()
        g = pd.DataFrame(np.maximum(g, 0), columns=cols)
        g.loc[(k["dam_id"] == "").to_numpy(), :] = 0.0  # 母が分からない馬はきょうだい 0 扱い（sib_no_dam で区別）
        n = g["one"].to_numpy()
        out[f"{tag}_n_ran"] = g["first_run"].to_numpy()
        out[f"{tag}_n_winners"] = g["first_win"].to_numpy()
        out[f"{tag}_runs"] = n
        out[f"{tag}_win_rate"] = np.where(n > 0, _rate(g["win"], g["valid"], A_WIN, B_WIN), np.nan)
        out[f"{tag}_top3_rate"] = np.where(n > 0, _rate(g["top3"], g["valid"], A_TOP3, B_TOP3), np.nan)
        out[f"{tag}_mean_pct"] = np.where(g["valid"] > 0, g["pct0"] / np.maximum(g["valid"], 1), np.nan)
        if gcol == "dam_id":
            out["sib_no_dam"] = (k["dam_id"].to_numpy() == "").astype("int8")
    # 母の産駒数（その馬より前の生まれ年の兄姉の頭数。走っていなくても数える）
    by = ped.assign(by=pd.to_numeric(ped["horse_id"].str[:4], errors="coerce"))
    dam_by = by[by["dam_id"].astype(str).str.len() > 0].groupby("dam_id")["by"].apply(lambda s: np.sort(s.to_numpy()))
    kb = pd.to_numeric(k["horse_id"].str[:4], errors="coerce").to_numpy()
    older = np.zeros(len(k), dtype="float32")
    arr = k["dam_id"].map(dam_by)
    for i, (a, y) in enumerate(zip(arr, kb)):
        if isinstance(a, np.ndarray) and y == y:
            older[i] = np.searchsorted(a, y, side="left")
    out["sib_n_older"] = older
    out["sib_ran_share"] = np.where(out["sib_n_older"] > 0, out["sib_n_ran"] / np.maximum(out["sib_n_older"], 1), np.nan)
    out.index = keys.index
    return out.astype("float32")


# ───────────────────────── 併せ馬（partner_notes）
_FW = str.maketrans("０１２３４５６７８９．", "0123456789.")
_RESULT = (("先着", 1), ("併入", 0), ("同入", 0), ("遅れ", -1))
_PN_MARGIN = re.compile(r"(\d+)秒(\d+)")
_PN_POS = re.compile(r"^(内|外|中)")


def parse_partner_note(note: str) -> list[dict]:
    """併せ馬メモ（『内クリニクラウン馬也と併入 / 外…』）→ 相手ごとの {pos, score, margin}。score は先着 +1・併入 0・遅れ −1、margin は秒（先着は正、遅れは負）。
    追走・先行だけで結果語が無いものは score=None。"""
    out = []
    for part in str(note or "").translate(_FW).split(" / "):
        part = part.strip()
        if not part:
            continue
        score = None
        for w, s in _RESULT:
            if w in part:
                score = s
                break
        mm = _PN_MARGIN.search(part)
        margin = None
        if score is not None:
            margin = 0.0 if score == 0 else score * (int(mm.group(1)) + int(mm.group(2)) / 10.0) if mm else None
        pos = _PN_POS.match(part)
        out.append({"pos": pos.group(1) if pos else "", "score": score, "margin": margin})
    return out


def partner_features(work: pd.DataFrame, keys: pd.DataFrame, window_days: int = 35) -> pd.DataFrame:
    """PT: レース前 window_days 日以内の追い切りのうち、併せ馬の相手と結果（先着・併入・遅れ）の要約。当日・未来の追い切りは使わない。
    work: horse_id, train_date, partner_notes"""
    w = work.copy()
    w["train_date"] = pd.to_datetime(w["train_date"], errors="coerce")
    w = w.dropna(subset=["train_date"])
    parsed = w["partner_notes"].map(parse_partner_note)
    w["n_partner"] = parsed.map(lambda p: sum(1 for x in p if x["score"] is not None))
    w["n_ahead"] = parsed.map(lambda p: sum(1 for x in p if x["score"] == 1))
    w["n_even"] = parsed.map(lambda p: sum(1 for x in p if x["score"] == 0))
    w["n_behind"] = parsed.map(lambda p: sum(1 for x in p if x["score"] == -1))
    w["margin_sum"] = parsed.map(lambda p: sum(x["margin"] for x in p if x["margin"] is not None))
    w["margin_n"] = parsed.map(lambda p: sum(1 for x in p if x["margin"] is not None))
    w["one"] = 1
    w["date"] = w["train_date"]
    cols = ["one", "n_partner", "n_ahead", "n_even", "n_behind", "margin_sum", "margin_n"]
    k = keys[["horse_id", "date"]].copy()
    k["date"] = pd.to_datetime(k["date"])
    hi = cum_asof(w, "horse_id", cols, k)  # date より厳密に前の累積
    lo = cum_asof(w, "horse_id", cols, k.assign(date=k["date"] - pd.Timedelta(days=window_days)))
    d = (hi.to_numpy() - lo.to_numpy()).clip(min=0)
    d = pd.DataFrame(d, columns=cols)
    out = pd.DataFrame(index=range(len(k)))
    out["pt_n_work"] = d["one"]
    out["pt_n_partner"] = d["n_partner"]
    out["pt_share_partner"] = np.where(d["one"] > 0, d["n_partner"] / np.maximum(d["one"], 1), np.nan)
    tot = d["n_ahead"] + d["n_even"] + d["n_behind"]
    out["pt_ahead_rate"] = np.where(tot > 0, d["n_ahead"] / np.maximum(tot, 1), np.nan)
    out["pt_behind_rate"] = np.where(tot > 0, d["n_behind"] / np.maximum(tot, 1), np.nan)
    out["pt_mean_score"] = np.where(tot > 0, (d["n_ahead"] - d["n_behind"]) / np.maximum(tot, 1), np.nan)
    out["pt_mean_margin"] = np.where(d["margin_n"] > 0, d["margin_sum"] / np.maximum(d["margin_n"], 1), np.nan)
    out.index = keys.index
    return out.astype("float32")
