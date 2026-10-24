"""tp-28: 追い切りパターン別の経験的な成績表（勝率・連対率・複勝率・着順残差・単勝/複勝回収率）から作る馬ごとの特徴量。

TL_A（時計を標準化した値そのもの）や TL_B（調教師×パターンの縮約採点）と違い、『そのパターンの追い切りをした馬が、過去に実際にどれだけ走ったか』の素直な統計表を、
粗い粒度から細かい粒度まで階層で作り、経験ベイズで階層縮約する。すべて as-of（当該レース日より前に確定したレースだけ）。キーは horse_id / trainer_id。

- 追い切りの 3 つの時期: F=最終（レースの 1〜6 日前）、W1=1 週前（7〜13 日前）、W2=2 週前（14〜20 日前）。各時期の最も新しい追い切りを使う。
- パターン階層（F）: P1=種別×所属 → P2=+強度 → P3=+終い1F の時計階級 → P4=+4F の階級×終いの形 → P5=+日数帯 → P6=+乗り役×評価ランク。
- 回収率（ROI 系）は過去の確定払戻の統計（単勝・複勝）。現在の市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

SLOTS = {"F": (1, 6), "W1": (7, 13), "W2": (14, 20)}
Z_CUTS = (-1.0, -0.35, 0.35, 1.0)          # 標準化した時計（小さい = 速い）の 5 階級
ACCEL_CUT = 0.4
MIN_GLOBAL = 300                            # 全体の累積件数がこれ未満（2020 年の初め）は統計を出さない
METRICS = ["n", "win", "top2", "top3", "res", "rw_s", "rw_n", "rp_s", "rp_n"]     # 件数・勝・連対・複勝・着順残差の和・単勝回収の和/件数・複勝回収の和/件数


# ---------------------------------------------------------------- 払戻の解釈（過去の確定結果）

def _splits(s: str, k: int, ok) -> list[list[str]]:
    """数字の連結文字列 s を k 個に分ける全ての方法のうち、各片が ok(片) を満たすもの。"""
    out: list[list[str]] = []

    def rec(pos: int, left: int, acc: list[str]) -> None:
        if left == 0:
            if pos == len(s):
                out.append(list(acc))
            return
        for ln in range(1, 7):
            if pos + ln > len(s):
                break
            part = s[pos:pos + ln]
            if ok(part):
                acc.append(part)
                rec(pos + ln, left - 1, acc)
                acc.pop()

    rec(0, k, [])
    return out


def _is_payout(p: str) -> bool:
    return len(p) >= 3 and p[0] != "0" and p.endswith("0")


def parse_payoff(payoff: str | None, finish: dict[int, int]) -> dict[str, dict[int, float]]:
    """payoff（JSON 文字列）→ {'win': {馬番: 払戻/100}, 'place': {馬番: 払戻/100}}。曖昧・欠損は含めない。
    finish: 馬番 → 着順（複勝の馬番の連結文字列を、3 着以内の馬番の集合で分けるのに使う）。"""
    out: dict[str, dict[int, float]] = {"win": {}, "place": {}}
    if not payoff:
        return out
    try:
        d = json.loads(payoff)
    except Exception:
        return out
    top = {h for h, f in finish.items() if f is not None and 1 <= f <= 3}
    for key, name in (("単勝", "win"), ("複勝", "place")):
        v = d.get(key)
        pairs = [v] if isinstance(v, dict) else [x for x in (v or []) if isinstance(x, dict)] if isinstance(v, list) else []
        if len(pairs) > 1:                                        # 同着などで 1 頭ずつ別の項目になっている形
            m = {}
            for x in pairs:
                tk = [t for t in re.split(r"\D+", str(x.get("numbers", ""))) if t]
                pv = str(x.get("payout", "")).replace(",", "").replace(" ", "")
                if len(tk) != 1 or not _is_payout(pv):
                    m = {}
                    break
                m[int(tk[0])] = int(pv) / 100.0
            out[name] = m
            continue
        if not pairs:
            continue
        nums, pay = str(pairs[0].get("numbers", "")), str(pairs[0].get("payout", ""))
        if not nums or not pay:
            continue
        pay_digits = pay.replace(",", "").replace(" ", "")
        toks = [t for t in re.split(r"\D+", nums) if t]
        if name == "win":
            cands = {int(t) for t in toks}
            k = len(cands)
            sp = _splits(pay_digits, k, _is_payout)
            if len(toks) == k and len(sp) == 1:
                out["win"] = {int(t): int(p) / 100.0 for t, p in zip(toks, sp[0])}
            continue
        # 複勝: 3 着以内の馬番の集合と一致する分け方、払戻は 100 円以上で 10 円刻み
        k = len(top)
        if k == 0 or k > 5:
            continue
        nsp = _splits("".join(toks), k, lambda p: p[0] != "0" and 1 <= int(p) <= 18)
        nsp = [x for x in nsp if {int(t) for t in x} == top]
        if len(nsp) > 1:                                          # 同じ集合でも並びが違う分け方は、着順の並び（払戻の並び＝着順）に合うものを採る
            order = [h for h, f in sorted(finish.items(), key=lambda kv: kv[1]) if f <= 3]
            nsp = [x for x in nsp if [int(t) for t in x] == order]
        psp = _splits(pay_digits, k, _is_payout)
        if len(nsp) == 1 and len(psp) == 1:
            out["place"] = {int(t): int(p) / 100.0 for t, p in zip(nsp[0], psp[0])}
    return out


def runner_returns(results: pd.DataFrame, payoff_by_race: dict[str, str]) -> pd.DataFrame:
    """results: race_id・horse_number・finish_position。戻り値: 同じ並びの win_ret / place_ret（100 円あたりの払戻/100。ハズレは 0、解釈できないレースは NaN）。"""
    r = results[["race_id", "horse_number", "finish_position"]].copy()
    r["race_id"] = r["race_id"].astype(str)
    win = np.full(len(r), np.nan)
    place = np.full(len(r), np.nan)
    idx = r.groupby("race_id").indices
    hn = r["horse_number"].to_numpy()
    fp = r["finish_position"].to_numpy()
    for rid, ix in idx.items():
        fin = {int(hn[i]): int(fp[i]) for i in ix if np.isfinite(fp[i]) and fp[i] > 0}
        pr = parse_payoff(payoff_by_race.get(rid), fin)
        if pr["win"]:
            for i in ix:
                win[i] = pr["win"].get(int(hn[i]), 0.0)
        if pr["place"]:
            for i in ix:
                place[i] = pr["place"].get(int(hn[i]), 0.0)
    return pd.DataFrame({"win_ret": win, "place_ret": place}, index=results.index)


# ---------------------------------------------------------------- 追い切りのパターン

def _bin(z: pd.Series) -> pd.Series:
    b = np.digitize(z.to_numpy(dtype=float), Z_CUTS)
    return pd.Series(np.where(np.isfinite(z.to_numpy(dtype=float)), b.astype(str), "n"), index=z.index)


def _rider_class(r: pd.Series) -> pd.Series:
    r = r.fillna("").astype(str)
    return pd.Series(np.where(r == "助手", "a", np.where(r == "師", "t", np.where(r == "見習", "k", np.where(r == "", "b", "j")))), index=r.index)


def pick_sessions(t: pd.DataFrame) -> pd.DataFrame:
    """t: training_patterns.add_reference_z 済みの追い切り表。時期ごと・(horse_id, race_date)ごとに最も新しい追い切りを 1 本選び、パターン部品を付ける。"""
    t = t.copy()
    t["slot"] = np.select([t["days_before"].between(*SLOTS["F"]), t["days_before"].between(*SLOTS["W1"]), t["days_before"].between(*SLOTS["W2"])], ["F", "W1", "W2"], "")
    t = t[t["slot"] != ""]
    t = t.sort_values(["horse_id", "race_date", "slot", "days_before"], kind="stable")
    g = t.groupby(["horse_id", "race_date", "slot"], sort=False).head(1).copy()
    g["kind_c"] = g["kind"].astype(str) + "|" + g["center"].astype(str)
    g["int_c"] = g["intensity"].fillna(-1).astype(int).astype(str)
    g["t1b"] = _bin(g["t1z"])
    g["t4b"] = _bin(g["t4z"])
    g["shape"] = np.where(g["accel"].isna(), "n", np.where(g["accel"] <= -ACCEL_CUT, "A", np.where(g["accel"] >= ACCEL_CUT, "D", "F")))
    g["dband"] = np.where(g["days_before"] <= 2, "s", np.where(g["days_before"] <= 4, "m", "l"))
    g["rider_c"] = _rider_class(g["rider"])
    g["rank_c"] = g["rank"].fillna("").astype(str).str[:1]
    g["P1"] = g["kind_c"]
    g["P2"] = g["P1"] + "|" + g["int_c"]
    g["P3"] = g["P2"] + "|" + g["t1b"]
    g["P4"] = g["P3"] + "|" + g["t4b"] + "|" + g["shape"]
    g["P5"] = g["P4"] + "|" + g["dband"]
    g["P6"] = g["P5"] + "|" + g["rider_c"] + "|" + g["rank_c"]
    return g


def count_sessions(t: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    c = t[t["days_before"] <= days].groupby(["horse_id", "race_date"]).size().rename("n_sess").reset_index()
    return c


def slot_frame(sessions: pd.DataFrame, slot: str, cols: Iterable[str]) -> pd.DataFrame:
    s = sessions[sessions["slot"] == slot][["horse_id", "race_date", *cols]].copy()
    s.columns = ["horse_id", "race_date"] + [f"{slot}_{c}" for c in cols]
    return s


# ---------------------------------------------------------------- as-of 集計

def run_metrics(fin: np.ndarray, nfield: np.ndarray, win_ret: np.ndarray, place_ret: np.ndarray) -> np.ndarray:
    """出走ごとの集計用の指標行列（METRICS の並び）。"""
    f = fin.astype(float)
    n = np.maximum(nfield.astype(float) - 1.0, 1.0)
    res = 0.5 - (f - 1.0) / n
    wr_ok = np.isfinite(win_ret)
    pr_ok = np.isfinite(place_ret)
    return np.column_stack([
        np.ones(len(f)), (f == 1).astype(float), (f <= 2).astype(float), (f <= 3).astype(float), np.nan_to_num(res),
        np.where(wr_ok, win_ret, 0.0), wr_ok.astype(float), np.where(pr_ok, place_ret, 0.0), pr_ok.astype(float),
    ])


def _codes(s: pd.Series, valid: np.ndarray) -> np.ndarray:
    codes, _ = pd.factorize(s.where(valid))
    return codes.astype(np.int64)


class Shrunk:
    """階層縮約した統計。rate = (S + m·parent) / (n + m)。parent は一つ上の階層（最上位は全体の as-of 平均）。"""

    def __init__(self, sums: np.ndarray, parent: "Shrunk | None", glob: np.ndarray, m: float = 30.0, m_roi: float = 200.0):
        n = np.nan_to_num(sums[:, 0])
        self.n = n
        self.rates: dict[str, np.ndarray] = {}
        for j, nm in ((1, "win"), (2, "top2"), (3, "top3"), (4, "res")):
            p = parent.rates[nm] if parent is not None else glob[:, j]
            self.rates[nm] = (np.nan_to_num(sums[:, j]) + m * p) / (n + m)
        for sj, nj, nm in ((5, 6, "roi_w"), (7, 8, "roi_p")):
            nn = np.nan_to_num(sums[:, nj])
            p = parent.rates[nm] if parent is not None else glob[:, sj]
            self.rates[nm] = (np.nan_to_num(sums[:, sj]) + m_roi * p) / (nn + m_roi)
        self.glob = glob


def global_rates(day: np.ndarray, M: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """全体の as-of 平均（率）。行ごとに METRICS と同じ位置へ [n, win, top2, top3, res, roi_w, -, roi_p, -] の率を入れる（valid の行だけを集計に使う）。"""
    key_stat = np.where(valid, 0, -1)
    S = _asof_lookup(key_stat, np.zeros(len(day), dtype=np.int64), day, M)
    n = np.maximum(S[:, 0], 1.0)
    out = S.copy()
    for j in (1, 2, 3, 4):
        out[:, j] = S[:, j] / n
    out[:, 5] = S[:, 5] / np.maximum(S[:, 6], 1.0)
    out[:, 7] = S[:, 7] / np.maximum(S[:, 8], 1.0)
    return out


def build_features(runs: pd.DataFrame, laps: pd.DataFrame, returns: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """runs: race_id・horse_id・date・trainer_id・finish_position・field_size（結果は評価用の as-of 集計にだけ使う）。laps: 追い切り表（training_laps.parquet 相当）。
    returns: runner_returns の結果（runs と同じ index）。戻り値: 群名 → race_id・horse_id + 特徴量の DataFrame。"""
    from src.pipeline.features import training_patterns as tp

    runs = runs.reset_index(drop=True)
    t = tp.add_reference_z(tp.prepare(laps))
    sess = pick_sessions(t)
    d = runs[["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size"]].copy()
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["horse_id"] = d["horse_id"].astype(str)
    comp = ["P1", "P2", "P3", "P4", "P5", "P6", "t1z", "t4z", "accel", "int_c", "kind_c"]
    for slot in SLOTS:
        d = d.merge(slot_frame(sess, slot, comp), on=["horse_id", "race_date"], how="left")
    d = d.merge(count_sessions(t), on=["horse_id", "race_date"], how="left")
    assert len(d) == len(runs)
    day = pd.to_datetime(d["date"]).to_numpy("datetime64[D]").astype(np.int64)
    fin = d["finish_position"].to_numpy(dtype=float)
    wr = returns["win_ret"].to_numpy() if returns is not None else np.full(len(d), np.nan)
    pr = returns["place_ret"].to_numpy() if returns is not None else np.full(len(d), np.nan)
    M = run_metrics(fin, d["field_size"].to_numpy(), wr, pr)
    has_res = np.isfinite(fin) & (fin > 0)
    M[~has_res] = 0.0
    tid = d["trainer_id"].astype(str)
    hid = d["horse_id"]

    def level(prefix: str, lv: str, extra: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
        s = d[f"{prefix}_{lv}"]
        valid = s.notna().to_numpy() & has_res
        if extra is not None:
            s = s + "#" + extra
        return _codes(s, s.notna().to_numpy()), valid

    gl_cache: dict[bytes, np.ndarray] = {}

    def glob_for(valid: np.ndarray) -> np.ndarray:
        k = np.packbits(valid).tobytes()
        if k not in gl_cache:
            gl_cache[k] = global_rates(day, M, valid)
        return gl_cache[k]

    def stat(key: np.ndarray, valid: np.ndarray, parent: Shrunk | None, glob: np.ndarray, m: float = 30.0) -> Shrunk:
        """key: パターンのコード（無い行は -1）。valid: 集計に入れる行（パターンあり・結果あり）。結果の無い行にも事前の統計を出す。"""
        S = _asof_lookup(np.where(valid & (key >= 0), key, -1), key, day, M)
        return Shrunk(S, parent, glob, m=m)

    g_wp_f, g_wp_w, g_wp_fw, g_roi, g_horse, g_tr, g_diff = {}, {}, {}, {}, {}, {}, {}

    def put(store: dict, name: str, st: Shrunk, glob: np.ndarray, valid_feat: np.ndarray, roi: bool = False) -> None:
        valid_feat = valid_feat & (glob[:, 0] >= MIN_GLOBAL)
        base = {"win": glob[:, 1], "top3": glob[:, 3], "res": glob[:, 4]}
        for k_, nm in (("win", "win"), ("top3", "top3"), ("res", "res")):
            v = st.rates[k_] - base[nm]
            store[f"{name}_{nm}"] = np.where(valid_feat, v, np.nan)
        store[f"{name}_logn"] = np.where(valid_feat, np.log1p(st.n), np.nan)
        if roi:
            store[f"{name}_roiw"] = np.where(valid_feat, st.rates["roi_w"] - glob[:, 5], np.nan)
            store[f"{name}_roip"] = np.where(valid_feat, st.rates["roi_p"] - glob[:, 7], np.nan)

    # ---- F 時期の階層
    stats_f: dict[str, Shrunk] = {}
    vf = d["F_P1"].notna().to_numpy()
    glob_f = glob_for(vf & has_res)
    for lv, par in (("P1", None), ("P2", "P1"), ("P3", "P2"), ("P4", "P3"), ("P5", "P4"), ("P6", "P5")):
        key, valid = level("F", lv)
        stats_f[lv] = stat(key, valid, stats_f[par] if par else None, glob_f)
        put(g_wp_f, f"wf_{lv}", stats_f[lv], glob_f, d[f"F_{lv}"].notna().to_numpy(), roi=lv in ("P2", "P4", "P6"))
    # ROI は別群へ移す
    for k in [k for k in g_wp_f if k.endswith(("_roiw", "_roip"))]:
        g_roi[k] = g_wp_f.pop(k)

    # ---- W1 / W2 時期
    for slot, lvs in (("W1", ("P1", "P2", "P3")), ("W2", ("P1", "P2"))):
        vs = d[f"{slot}_P1"].notna().to_numpy()
        glob_s = glob_for(vs & has_res)
        sts: dict[str, Shrunk] = {}
        for lv, par in zip(lvs, (None,) + lvs[:-1]):
            key, valid = level(slot, lv)
            sts[lv] = stat(key, valid, sts[par] if par else None, glob_s)
            put(g_wp_w, f"w{slot[1]}_{lv}", sts[lv], glob_s, d[f"{slot}_{lv}"].notna().to_numpy())

    # ---- 2 つの時期の組み合わせ（F×W1）
    both = d["F_P2"].notna() & d["W1_P2"].notna()
    vb = both.to_numpy()
    glob_b = glob_for(vb & has_res)
    for nm, a, b in (("fw_P2P2", "F_P2", "W1_P2"), ("fw_P3P2", "F_P3", "W1_P2")):
        s = (d[a] + "##" + d[b]).where(both)
        key = _codes(s, vb)
        par = stats_f["P2"] if a == "F_P2" else stats_f["P3"]
        st = stat(key, vb & has_res, par, glob_f)
        put(g_wp_fw, nm, st, glob_b, vb)

    # ---- 馬×パターン、調教師×パターン
    p2 = stats_f["P2"]
    kh = _codes(hid + "#" + d["F_P2"].fillna(""), d["F_P2"].notna().to_numpy())
    st_h = stat(kh, vf & has_res, p2, glob_f, m=3.0)
    put(g_horse, "hp_P2", st_h, glob_f, vf)
    kh1 = _codes(hid + "#" + d["F_kind_c"].fillna(""), d["F_kind_c"].notna().to_numpy())
    st_h1 = stat(kh1, vf & has_res, stats_f["P1"], glob_f, m=3.0)
    put(g_horse, "hp_P1", st_h1, glob_f, vf)
    for lv in ("P2", "P4"):
        kt = _codes(tid + "#" + d[f"F_{lv}"].fillna(""), d[f"F_{lv}"].notna().to_numpy())
        st_t = stat(kt, vf & has_res, stats_f[lv], glob_f, m=10.0)
        put(g_tr, f"tp_{lv}", st_t, glob_f, vf)
    # 調教師全体（パターン無視）との差: 『その厩舎では珍しい型』
    ktr = _codes(tid, np.ones(len(d), dtype=bool))
    st_tr = stat(np.where(vf, ktr, -1), vf & has_res, None, glob_f, m=10.0)
    g_tr["tp_trainer_top3"] = np.where(vf & (glob_f[:, 0] >= MIN_GLOBAL), st_tr.rates["top3"] - glob_f[:, 3], np.nan)
    kt2 = _codes(tid + "#" + d["F_P2"].fillna(""), vf)
    S_tp = _asof_lookup(np.where(has_res & vf, kt2, -1), kt2, day, M)
    g_tr["tp_share"] = np.where(vf & (glob_f[:, 0] >= MIN_GLOBAL), S_tp[:, 0] / np.maximum(st_tr.n, 1.0), np.nan)

    # ---- 差分系: 時計の階級・終いの形の『上乗せ』、統計順位と時計順位のずれ
    g_diff["d_clock_res"] = g_wp_f["wf_P3_res"] - g_wp_f["wf_P2_res"]
    g_diff["d_shape_res"] = g_wp_f["wf_P4_res"] - g_wp_f["wf_P3_res"]
    g_diff["d_ctx_res"] = g_wp_f["wf_P6_res"] - g_wp_f["wf_P4_res"]
    g_diff["d_clock_top3"] = g_wp_f["wf_P3_top3"] - g_wp_f["wf_P2_top3"]
    g_diff["d_shape_top3"] = g_wp_f["wf_P4_top3"] - g_wp_f["wf_P3_top3"]
    rid = d["race_id"]
    for nm, arr in (("rk_P4_top3", g_wp_f["wf_P4_top3"]), ("rk_P6_res", g_wp_f["wf_P6_res"]), ("rk_P3_win", g_wp_f["wf_P3_win"])):
        g_diff[nm] = pd.Series(arr).groupby(rid.to_numpy()).rank(pct=True, ascending=False).to_numpy()
    clock_rank = pd.Series(d["F_t1z"].to_numpy(dtype=float)).groupby(rid.to_numpy()).rank(pct=True, ascending=True).to_numpy()
    g_diff["rk_gap_stat_vs_clock"] = g_diff["rk_P4_top3"] - clock_rank
    g_diff["n_sess_14"] = d["n_sess"].fillna(0).to_numpy()

    out = {}
    for nm, store in (("wp_f", g_wp_f), ("wp_w", g_wp_w), ("wp_fw", g_wp_fw), ("roi", g_roi), ("horse_pat", g_horse), ("trainer_pat", g_tr), ("diff", g_diff)):
        f = pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in store.items()})
        f.insert(0, "horse_id", d["horse_id"].to_numpy())
        f.insert(0, "race_id", d["race_id"].astype(str).to_numpy())
        out[nm] = f
    return out


def _asof_lookup(key_stat: np.ndarray, key_query: np.ndarray, day: np.ndarray, M: np.ndarray) -> np.ndarray:
    """key_stat の行（結果あり）だけを集計に使い、各行（key_query）に『その日より前』の同じキーの累積を返す。結果が無い行にも事前の統計を出せる。"""
    n, m = M.shape
    out = np.zeros((n, m))
    use = key_stat >= 0
    if not use.any():
        return out
    df = pd.DataFrame(M[use], columns=range(m))
    df["k"] = key_stat[use]
    df["d"] = day[use]
    g = df.groupby(["k", "d"], sort=True).sum()
    cum = g.groupby(level=0).cumsum()
    q = key_query >= 0
    qdf = pd.DataFrame({"k": key_query[q], "d": day[q]})
    # 各問い合わせ行について、その日より前（d' < d）の最後の累積行を引く
    cum = cum.reset_index().sort_values(["d", "k"])
    qdf["_i"] = np.flatnonzero(q)
    qdf = qdf.sort_values(["d", "k"])
    cum["d1"] = cum["d"] + 1                                     # d' + 1 <= d となる最後の累積（同日は含めない）
    merged = pd.merge_asof(qdf, cum.drop(columns=["d"]).rename(columns={"d1": "d"}), on="d", by="k", direction="backward", allow_exact_matches=True)
    vals = merged[list(range(m))].to_numpy()
    res = np.zeros((n, m))
    res[merged["_i"].to_numpy()] = np.nan_to_num(vals)
    return res
