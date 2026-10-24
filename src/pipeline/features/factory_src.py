"""特徴量ファクトリー G4: 未活用のデータ源（SmartRC・バロメーター・メグ指数・調教の中身・名前/ID の些細な特徴）から作る候補群。

入力 ``B`` は学習表（修正後）の行（RangeIndex、日付・race_id・馬番順）。必要列: race_id, horse_id, date, venue_code, surface, distance, track_condition,
finish_position, field_size, jockey_id, trainer_id, horse_number, bracket_number, horse_name, weight, jockey_weight, weight_change。
戻りは B と同じ index の数値列だけの表。どの群も「そのレースの日より前に分かる情報」だけで作る（履歴は horse_id ごとに日付順で 1 つ前までをずらして使う）。

市場の情報（オッズ・人気・est_pop・pop_*・odds_*・過去走の人気 h*_pop）は使わない。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
YEARS = list(range(2020, 2026))                      # 2026 年は学習に使わない
MARKET_PAT = re.compile(r"(^|_)(pop|est_pop|odds)($|_)|^pop_|^odds_|_pop$|_odds$")


def read_flat(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    import polars as pl

    parts = []
    for y in YEARS:
        f = TABLES / str(y) / f"{name}.parquet"
        if not f.exists():
            continue
        sch = pl.read_parquet_schema(f)
        cols = [c for c in columns if c in sch] if columns else None
        parts.append(pl.read_parquet(f, columns=cols))
    return pl.concat(parts, how="diagonal_relaxed").to_pandas()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")


def _code(s: pd.Series) -> pd.Series:
    """文字・区分の列を整数コードにする（欠損は NaN。教師なしの変換）。数値に読めるなら数値。"""
    n = _num(s)
    if n.notna().mean() > 0.9:
        return n.astype("float32")
    z = s.astype(str).str.strip().replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.Series(pd.factorize(z)[0], index=s.index).where(z.notna()).astype("float32")


def res_score(B: pd.DataFrame) -> np.ndarray:
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    fp = np.where(fp > 0, fp, np.nan)
    n = np.maximum(pd.to_numeric(B["field_size"], errors="coerce").to_numpy(dtype=float) - 1.0, 1.0)
    return 0.5 - (fp - 1.0) / n


def asof_rate(B: pd.DataFrame, key: pd.Series, value: np.ndarray, k: float, name: str) -> pd.DataFrame:
    """key ごとの value の平均を、その日より前の行だけで作る（経験ベイズで縮約）。key が欠損の行は NaN。"""
    from src.pipeline.models.stage_models import asof_group_mean

    kk = key.astype(str).where(key.notna() & (key.astype(str) != ""), "__na__")
    tmp = pd.DataFrame({"date": pd.to_datetime(B["date"]).to_numpy(), "k": kk.to_numpy(), "v": value}, index=pd.RangeIndex(len(B)))
    r = asof_group_mean(tmp, "k", "v", k=k, name=name)
    bad = (kk == "__na__").to_numpy()
    r.loc[bad, :] = np.nan
    r.index = B.index
    return r


def asof_rates(B: pd.DataFrame, key: pd.Series, name: str, k: float = 20.0) -> pd.DataFrame:
    res = res_score(B)
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    win = np.where(fp > 0, (fp == 1).astype(float), np.nan)
    t3 = np.where(fp > 0, (fp <= 3).astype(float), np.nan)
    a = asof_rate(B, key, win, k, f"{name}_win")
    b = asof_rate(B, key, t3, k, f"{name}_t3")
    c = asof_rate(B, key, res, k, f"{name}_res")
    out = pd.concat([a[f"{name}_win_mean"], b[f"{name}_t3_mean"], c[f"{name}_res_mean"], a[f"{name}_win_n"].rename(f"{name}_n")], axis=1)
    return out


# ───────────────────────── 馬ごとの履歴（1 つ前までをずらして使う）
def horse_order(B: pd.DataFrame) -> np.ndarray:
    return np.lexsort((B["race_id"].astype(str).to_numpy(), pd.to_datetime(B["date"]).to_numpy(), B["horse_id"].astype(str).to_numpy()))


def history_aggs(B: pd.DataFrame, V: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    """V: race_id・horse_id + cols（そのレース自身の値＝レース後に分かる値）。B の各行について、同じ馬の過去走の値から
    直近・指数平滑(0.5/0.25)・最良・直近 3 走平均・最悪・件数を作る。当該レース自身は含めない。"""
    m = B[["race_id", "horse_id"]].astype(str).merge(V.assign(race_id=V["race_id"].astype(str), horse_id=V["horse_id"].astype(str)).drop_duplicates(["race_id", "horse_id"]),
                                                    on=["race_id", "horse_id"], how="left")
    order = horse_order(B)
    hid = B["horse_id"].astype(str).to_numpy()
    n = len(B)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)
        last = np.full(n, np.nan); e5 = np.full(n, np.nan); e25 = np.full(n, np.nan); best = np.full(n, np.nan); m3 = np.full(n, np.nan)
        worst = np.full(n, np.nan); cnt = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and hid[order[j]] == hid[order[i]]:
                j += 1
            h5 = h25 = np.nan
            bst = wst = np.nan
            k = 0
            recent: list[float] = []
            for t in range(i, j):
                r = order[t]
                if k > 0:
                    last[r] = recent[-1] if recent else np.nan
                    e5[r], e25[r], best[r], worst[r] = h5, h25, bst, wst
                    if recent:
                        m3[r] = np.mean(recent[-3:])
                    cnt[r] = len(recent)
                v = x[r]
                if np.isfinite(v):
                    recent.append(v)
                    h5 = v if not np.isfinite(h5) else 0.5 * v + 0.5 * h5
                    h25 = v if not np.isfinite(h25) else 0.25 * v + 0.75 * h25
                    bst = v if not np.isfinite(bst) else max(bst, v)
                    wst = v if not np.isfinite(wst) else min(wst, v)
                k += 1
            i = j
        out.update({f"{prefix}{c}_last": last, f"{prefix}{c}_e50": e5, f"{prefix}{c}_e25": e25, f"{prefix}{c}_best": best, f"{prefix}{c}_m3": m3, f"{prefix}{c}_worst": worst})
        if c == cols[0]:
            out[f"{prefix}n"] = cnt
    return pd.DataFrame(out, index=B.index).astype("float32")


def add_field_relative(B: pd.DataFrame, F: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = F[cols].groupby(B["race_id"].to_numpy())
    z = (F[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    z.columns = [f"{c}_rz" for c in cols]
    return z.astype("float32")


# ───────────────────────── 群: SmartRC
SRC_COLS = ["race_id", "hcode", "weight", "owner", "producer", "ocode", "pcode", "blinker", "handicap", "jcode", "jcode_before", "tozai", "invite", "ustat", "interval",
            "rota_type", "diff_type", "rota_eval", "old_pr", "cr_value", "f_dirt_share", "f_1400_share", "cl_eval_cmb", "cl_eval_ja", "cl_eval_en", "cl_eval_eu", "diff_exp",
            "f_llcode", "f_slcode", "f_clcode", "fmf_llcode", "fmf_slcode", "fmf_clcode", "mf_llcode", "mf_slcode", "mf_clcode", "f_country", "fmf_country", "mf_country",
            "ten_has", "ten_has_rank", "agari_has", "agari_has_rank", "ten_pat", "agari_pat", "cnt_all", "cnt_r1", "cnt_r2", "cnt_r3"] + \
           [f"exp_{a}_{b}" for a in ("cur_course", "cur_short", "cur_long", "cur_same") for b in ("cnt", "wi3")] + \
           [f"exp_rota_{a}_{b}_wi{c}" for a in ("short", "long", "same") for b in ("na", "n5") for c in (3, 5)] + \
           [f"h{i}_{f}" for i in range(1, 6) for f in ("furlong3", "f3rank", "corner1", "corner2", "corner3", "corner4", "difftime", "tb_io", "tb_diff", "stime", "weight", "interval", "rank", "grade")]


def load_src(B: pd.DataFrame) -> pd.DataFrame:
    s = read_flat("smartrc_race_flat", SRC_COLS)
    bad = [c for c in s.columns if MARKET_PAT.search(c)]
    assert not bad, bad
    s = s.rename(columns={"hcode": "horse_id"})
    s["race_id"], s["horse_id"] = s["race_id"].astype(str), s["horse_id"].astype(str)
    s = s.drop_duplicates(["race_id", "horse_id"])
    m = B[["race_id", "horse_id"]].astype(str).merge(s, on=["race_id", "horse_id"], how="left")
    m.index = B.index
    return m


def src_equip(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=B.index)
    for c in ("blinker", "handicap", "invite", "ustat", "tozai"):
        out[f"sx_{c}"] = _num(S[c])
    jc, jb = S["jcode"].astype(str).str.strip(), S["jcode_before"].astype(str).str.strip()
    out["sx_jockey_changed"] = ((jb != "None") & (jb != "") & (jb != "nan") & (jb != jc)).astype("float32").where(S["jcode"].notna())
    out["sx_interval"] = _num(S["interval"])
    return out.astype("float32")


def src_owner(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    a = asof_rates(B, S["ocode"], "sx_owner", k=20.0)
    b = asof_rates(B, S["pcode"], "sx_breeder", k=20.0)
    return pd.concat([a, b], axis=1).astype("float32")


def src_rota(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    # old_pr（A〜E）は A の勝率 44%・E の勝率 0.7% と、他のどの列より桁違いに当たる（人気/予想印か結果を含む指標と見られる）。市場情報の疑いがあるので使わない
    cols = ["rota_type", "diff_type", "rota_eval", "cr_value", "f_dirt_share", "f_1400_share", "cl_eval_cmb", "diff_exp"]
    out = pd.DataFrame({f"sx_{c}": _code(S[c]) for c in cols}, index=B.index)
    for c in [x for x in S.columns if x.startswith("exp_")]:
        out[f"sx_{c}"] = _num(S[c])
    return out.astype("float32")


def src_line(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    cols = ["f_llcode", "f_slcode", "f_clcode", "fmf_llcode", "fmf_slcode", "fmf_clcode", "mf_llcode", "mf_slcode", "mf_clcode", "f_country", "fmf_country", "mf_country"]
    return pd.DataFrame({f"sx_{c}": _code(S[c]) for c in cols}, index=B.index).astype("float32")


def src_line_rates(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """父・母父の系統コード（大分類・小分類）ごとの as-of 成績（面別）。"""
    out = []
    surf = B["surface"].astype(str)
    for c in ("f_slcode", "mf_slcode", "f_clcode"):
        key = S[c].astype(str).where(S[c].notna()) + "|" + surf
        out.append(asof_rates(B, key, f"sx_{c}", k=40.0))
    return pd.concat(out, axis=1).astype("float32")


def src_ten_agari(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    cols = ["ten_has", "ten_has_rank", "agari_has", "agari_has_rank", "ten_pat", "agari_pat"]
    return pd.DataFrame({f"sx_{c}": _code(S[c]) for c in cols}, index=B.index).astype("float32")


def src_hist_misc(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """SmartRC の過去 5 走（h1 が直近）から、上がり順位・コーナー位置・着差・タイム・馬体重の平均と変化。"""
    out = {}

    def mat(f: str, idx=range(1, 6)) -> np.ndarray:
        cols = [_num(S[f"h{i}_{f}"]).to_numpy(dtype=float) if f"h{i}_{f}" in S.columns else np.full(len(S), np.nan) for i in idx]
        return np.column_stack(cols)

    for f in ("f3rank", "difftime", "tb_io", "tb_diff", "weight"):
        M = mat(f)
        out[f"sx_h_{f}_m3"] = np.nanmean(M[:, :3], axis=1)
        out[f"sx_h_{f}_m5"] = np.nanmean(M, axis=1)
        out[f"sx_h_{f}_last"] = M[:, 0]
    C = {i: mat(f"corner{i}", range(1, 4)) for i in (1, 2, 3, 4)}
    out["sx_h_corner1_m3"] = np.nanmean(C[1], axis=1)
    out["sx_h_corner4_m3"] = np.nanmean(C[4], axis=1)
    out["sx_h_corner_gain_m3"] = np.nanmean(C[1] - C[4], axis=1)
    out["sx_h_corner_gain_last"] = C[1][:, 0] - C[4][:, 0]
    st = mat("stime")
    out["sx_h_stime_last"] = st[:, 0]
    out["sx_h_stime_sd"] = np.nanstd(st, axis=1)
    iv = mat("interval")
    out["sx_h_interval_m3"] = np.nanmean(iv[:, :3], axis=1)
    return pd.DataFrame(out, index=B.index).astype("float32")


# ───────────────────────── 群: バロメーター・メグ指数・パドックの過去走
def baro_hist(B: pd.DataFrame) -> pd.DataFrame:
    v = read_flat("race_barometer_flat", ["race_id", "horse_id", "index_total", "index_start", "index_chase", "index_closing"])
    f = history_aggs(B, v, ["index_total", "index_start", "index_chase", "index_closing"], "bm_")
    return f


def megu_hist(B: pd.DataFrame) -> pd.DataFrame:
    v = read_flat("megu_index_flat", ["race_id", "horse_id", "megu_index", "adjusted_time_sec", "delta_pace_sec", "delta_track_sec", "delta_level_sec"])
    v["megu_index"] = pd.to_numeric(v["megu_index"], errors="coerce")
    f = history_aggs(B, v, ["megu_index", "delta_pace_sec", "delta_track_sec", "delta_level_sec"], "mg_")
    rel = add_field_relative(B, f, ["mg_megu_index_e50", "mg_megu_index_best", "mg_megu_index_last"])
    return pd.concat([f, rel], axis=1)


def paddock_hist(B: pd.DataFrame) -> pd.DataFrame:
    v = read_flat("race_paddock_flat", ["race_id", "horse_id", "paddock_rank"])
    v["pk_rank"] = _num(v["paddock_rank"])
    v["pk_top"] = (v["pk_rank"] <= 5).astype(float).where(v["pk_rank"].notna())
    return history_aggs(B, v, ["pk_top", "pk_rank"], "pk_")


# ───────────────────────── 群: 調教の中身（脚色・コース内位置・ランク・評価・併せ馬）
LEG = {"馬也": 0, "馬なり": 0, "一杯": 4, "G一杯": 4, "強め": 2, "G強": 2, "仕掛": 3, "末強": 2, "末一杯": 3, "末併": 1, "G馬": 0, "稍一杯": 3}


def trn_misc(B: pd.DataFrame) -> pd.DataFrame:
    import polars as pl

    p = ROOT / "data/local/modeling/training_pipeline/features/training_laps.parquet"
    t = pl.read_parquet(p, columns=["horse_id", "train_date", "race_date", "course", "kind", "rider", "position", "leg_color", "rank", "evaluation", "days_before"]).to_pandas()
    t = t[t["train_date"] < t["race_date"]]
    lc = t["leg_color"].astype(str).str.strip()
    t["hard"] = lc.map(LEG)
    t["hard"] = t["hard"].fillna(lc.str.contains("一杯").astype(float) * 4 + lc.str.contains("強").astype(float) * 2 + lc.str.contains("仕掛").astype(float) * 3)
    t["rk"] = t["rank"].astype(str).str.strip().map({"A": 4, "B": 3, "C": 2, "D": 1}).astype("float32")
    t["assistant"] = (t["rider"].astype(str).str.contains("助手|見習|調教師|厩務")).astype(float)
    t["pos"] = pd.to_numeric(t["position"], errors="coerce")
    t["last2w"] = (t["days_before"] <= 14).astype(float)
    t["race_date"] = t["race_date"].astype(str)
    g = t.groupby(["horse_id", "race_date"])
    last = t.sort_values("train_date").groupby(["horse_id", "race_date"]).tail(1).set_index(["horse_id", "race_date"])
    agg = pd.DataFrame({
        "tm_last_hard": last["hard"], "tm_last_rank": last["rk"], "tm_last_pos": last["pos"], "tm_last_assistant": last["assistant"],
        "tm_mean_hard": g["hard"].mean(), "tm_max_hard": g["hard"].max(), "tm_mean_rank": g["rk"].mean(), "tm_n": g.size(),
        "tm_n_hard": t[t["hard"] >= 3].groupby(["horse_id", "race_date"]).size(),
        "tm_assistant_share": g["assistant"].mean(),
    })
    key = pd.MultiIndex.from_arrays([B["horse_id"].astype(str), pd.to_datetime(B["date"]).dt.strftime("%Y-%m-%d")])
    out = agg.reindex(key)
    out.index = B.index
    out["tm_n_hard"] = out["tm_n_hard"].fillna(0).where(out["tm_n"].notna())
    return out.astype("float32")


# ───────────────────────── 群: 名前・ID・番号の些細な特徴
_KATA = re.compile(r"^[ァ-ヶー・]+$")


def trivia_name(B: pd.DataFrame) -> pd.DataFrame:
    nm = B["horse_name"].astype(str).str.replace(r"\s", "", regex=True)
    ln = nm.str.len()
    out = pd.DataFrame(index=B.index)
    out["tv_name_len"] = ln
    out["tv_name_kata"] = nm.map(lambda s: 1.0 if _KATA.match(s) else 0.0)
    out["tv_name_choon"] = nm.str.count("ー")
    out["tv_name_dakuten"] = nm.str.count(r"[ガ-ヂヅ-ドバ-ボヴ]")
    out["tv_name_small"] = nm.str.count(r"[ァィゥェォッャュョ]")
    out["tv_name_uniq"] = nm.map(lambda s: len(set(s)) / max(len(s), 1))
    out["tv_name_first_row"] = nm.str[:1].map(lambda c: "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン".find(c) // 5 if c else -1)
    out["tv_name_has_digit"] = nm.str.contains(r"[0-9０-９]").astype(float)
    return out.astype("float32")


def trivia_id(B: pd.DataFrame) -> pd.DataFrame:
    hid = B["horse_id"].astype(str)
    yr = pd.to_numeric(hid.str[:4], errors="coerce")
    seq = pd.to_numeric(hid.str[4:], errors="coerce")
    out = pd.DataFrame(index=B.index)
    out["tv_id_seq"] = seq
    out["tv_id_seq_pct"] = seq.groupby(yr).rank(pct=True)
    out["tv_id_kind"] = pd.to_numeric(hid.str[4:5], errors="coerce")
    return out.astype("float32")


def trivia_number(B: pd.DataFrame) -> pd.DataFrame:
    hn = pd.to_numeric(B["horse_number"], errors="coerce")
    br = pd.to_numeric(B["bracket_number"], errors="coerce")
    fs = pd.to_numeric(B["field_size"], errors="coerce")
    jw = pd.to_numeric(B["jockey_weight"], errors="coerce")
    wt = pd.to_numeric(B["weight"], errors="coerce")
    wc = pd.to_numeric(B["weight_change"], errors="coerce")
    rid = B["race_id"].astype(str)
    out = pd.DataFrame(index=B.index)
    out["tv_hn_odd"] = hn % 2
    out["tv_hn_eq_br"] = (hn == br).astype(float)
    out["tv_hn_last"] = (hn == fs).astype(float)
    out["tv_hn_first"] = (hn == 1).astype(float)
    out["tv_hn_pct"] = hn / fs
    out["tv_hn_from_end"] = fs - hn
    out["tv_hn_gap_br"] = hn - br
    out["tv_br_size"] = br.groupby(rid).transform("size")
    out["tv_jw_frac"] = (jw * 10 % 10)
    out["tv_wt_mod10"] = wt % 10
    out["tv_wc_abs"] = wc.abs()
    out["tv_wc_sign"] = np.sign(wc)
    out["tv_wt_rel_field"] = wt - wt.groupby(rid).transform("mean")
    out["tv_jw_rel_field"] = jw - jw.groupby(rid).transform("mean")
    out["tv_round"] = pd.to_numeric(rid.str[10:12], errors="coerce")
    out["tv_round_first"] = (out["tv_round"] == 1).astype(float)
    out["tv_round_last"] = (out["tv_round"] == 12).astype(float)
    d = pd.to_datetime(B["date"])
    out["tv_dom"] = d.dt.day
    out["tv_weekofmonth"] = (d.dt.day - 1) // 7
    out["tv_holiday_mon"] = (d.dt.weekday == 0).astype(float)
    return out.astype("float32")


def src_travel(B: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """所属（東西）と競馬場の東西から遠征かどうか、(所属 × 競馬場 × 面) の as-of 成績。"""
    t = _num(S["tozai"])
    east = B["venue_code"].astype(str).isin(["01", "02", "03", "04", "05", "06"]).to_numpy()
    travel = ((t == 1) & ~east) | ((t == 2) & east)
    out = pd.DataFrame({"sx_travel": travel.astype(float)}, index=B.index).where(t.notna())
    key = (t.astype(str) + "|" + B["venue_code"].astype(str) + "|" + B["surface"].astype(str)).where(t.notna())
    r = asof_rates(B, key, "sx_tvs", k=40.0)
    hok = B["venue_code"].astype(str).isin(["01", "02"]).to_numpy()
    out["sx_hokkaido_from_west"] = (hok & (t == 2)).astype(float)
    return pd.concat([out, r], axis=1).astype("float32")


GROUPS_SRC = {"SRC_travel": src_travel, 
    "SRC_equip": src_equip, "SRC_owner": src_owner, "SRC_rota": src_rota, "SRC_line": src_line, "SRC_line_rates": src_line_rates,
    "SRC_ten_agari": src_ten_agari, "SRC_hist_misc": src_hist_misc,
}
GROUPS_PLAIN = {"BARO_hist": baro_hist, "MEGU_hist": megu_hist, "PADDOCK_hist": paddock_hist, "TRN_misc": trn_misc,
                "TV_name": trivia_name, "TV_id": trivia_id, "TV_number": trivia_number}


def build_group(name: str, B: pd.DataFrame, S: pd.DataFrame | None = None) -> pd.DataFrame:
    if name in GROUPS_SRC:
        return GROUPS_SRC[name](B, S if S is not None else load_src(B))
    if name in GROUPS_PLAIN:
        return GROUPS_PLAIN[name](B)
    from src.pipeline.features import factory_embed, factory_ped, factory_text

    for mod in (factory_text, factory_ped, factory_embed):
        g = getattr(mod, "GROUPS_" + mod.__name__.rsplit("_", 1)[-1].upper(), {})
        if name in g:
            return g[name](B)
    raise KeyError(name)
