"""◎ を固定した馬券のパターンの集計用に、A2（第 2 回の凍結 frozen_20261117_r2）の期間から「レース × 版」の配列を作る。

- 期間・◎・判定のオッズは bet_cycles の道具をそのまま使う（T-60 の再現 5 通り、凍結の分位の区切り、凍結のときのずれの元）。
- 設定 a2: A2 が参戦したレース（凍結の見送り条件: 未勝利・新馬 かつ q_r_max の分位 3 以上）。
  設定 all: 未勝利・新馬の全レース（q_r_max の見送りだけ外す）。◎ は同じ凍結の規則（候補が無ければ見送り、数を別に残す）。
- 判定に使う値（行ごと、馬の配列は Race と同じ予測の勝率の降順、18 頭に詰める）:
  予測の勝率・連対率・複勝率（確定オッズの offset を含む → 楽観）、T-60 の再現の単勝オッズ、
  モデルの組の確率（fit_blend_distribution＝MB60）、市場の組の確率（T-60 の再現の単勝から Harville）。
- 精算は確定の払戻（payoff.parquet・betting_2026・結果の表の払戻の列。枠連も同じ置き場から）。

費用 0 円（手元のファイルだけ。GCS を読まない）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import bet_cycles as bc

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "config/bet_cycles/frozen_20261117_r2.json"
R2_OUT = Path(os.environ.get("BET_CYCLES_R2_RECORDS") or "/home/jovyan/work/retrain_out/bet_cycles/r2")
CACHE = Path(os.environ.get("BET_PATTERNS_CACHE") or "/home/jovyan/work/retrain_out/bet_patterns")
NH = 18          # 1 レースの最大頭数
NF = 8           # 枠の数
MAIN2026 = ("2026-01-05", "2026-10-12")   # A2 の 2026 年の主の群（config/bet_cycles/prereg_2026_r2.json の main）と同じ窓
PRE2026 = ("2026-10-17", "2026-11-20")    # 本物の T-60 のオッズで ◎・人気を決める参考（凍結の fwd2026_pre。封印の終わりまで）
SETTINGS = ("a2", "all")


def load_frozen() -> dict:
    body = json.loads(FROZEN.read_text(encoding="utf-8"))
    if bc.strategy_hash({k: v for k, v in body.items() if k not in ("frozen_at", "hash")}) != body["hash"]:
        raise SystemExit("凍結ファイルのハッシュが合いません")
    want = body["pred_sha256"].get(str(bc.PRE_ODDS.relative_to(bc.ROOT)))
    cand = [bc.PRE_ODDS] + sorted(R2_OUT.glob("pre_odds_frozen_*.parquet"))
    bc.DRIFT_SRC = next((c for c in cand if c.exists() and bc._sha(c) == want), None)   # 再現のずれの元は凍結のときの中身（strategy_tables a2 と同じ）
    if bc.DRIFT_SRC is None:
        raise SystemExit("凍結のときの発走前オッズが見つかりません")
    return body


def period_inputs(body: dict, name: str) -> dict:
    """期間名（train・eval・test・main2026・pre2026）→ prepare_periods に渡す inputs。
    main2026 は凍結の fwd2026（T-60 の再現）を A2 の主の窓に、pre2026 は凍結の fwd2026_pre（本物の T-60）を 10/17〜11/20 に絞る。"""
    ps = body["inputs"]["periods"]
    if name in ("main2026", "pre2026"):
        spec = json.loads(json.dumps(ps["fwd2026" if name == "main2026" else "fwd2026_pre"]))
        spec["dates"] = list(MAIN2026 if name == "main2026" else PRE2026)
        return {"periods": {name: spec}}
    return {"periods": {name: ps[name]}}


def frames_of(years) -> dict[tuple[str, int], int]:
    from src.config.data_paths import TABLES_DIR

    out = {}
    for y in years:
        f = pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "bracket_number"])
        f = f.dropna()
        for rid, h, b in zip(f["race_id"].astype(str), pd.to_numeric(f["horse_number"], errors="coerce"), pd.to_numeric(f["bracket_number"], errors="coerce")):
            if h == h and b == b:
                out[(rid, int(h))] = int(b)
    return out


def wakuren_payoffs(race_ids: set[str], years) -> dict[str, dict[str, int]]:
    """枠連の払戻 {race_id: {"a-b": 払戻}}。払戻の表に無いレースは結果の表の払戻の列（枠連）で補う。発売が無いレースは入らない。"""
    from src.config.data_paths import TABLES_DIR

    out: dict[str, dict[str, int]] = {}
    for p in bc.PAYOFF_PATHS:
        if not Path(p).exists():
            continue
        t = pd.read_parquet(p, columns=["race_id", "bet_type", "combo", "payout"])
        t = t[(t["bet_type"] == "wakuren") & t["race_id"].astype(str).isin(race_ids)]
        for rid, c, v in zip(t["race_id"].astype(str), t["combo"].astype(str), t["payout"]):
            a, b = sorted(int(x) for x in c.split("-"))
            out.setdefault(rid, {})[f"{a}-{b}"] = int(v)
    for y in years:
        f = Path(TABLES_DIR) / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        t = pd.read_parquet(f, columns=["race_id", "payoff"]).drop_duplicates("race_id")
        for rid, raw in zip(t["race_id"].astype(str), t["payoff"]):
            if rid in out or rid not in race_ids:
                continue
            try:
                js = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):
                continue
            v = js.get("枠連")
            if v is None:
                continue
            ok = {}
            for e in (v if isinstance(v, list) else [v]):
                try:
                    a, b = sorted(int(x) for x in str(e["numbers"]).replace(" ", "").split("-"))
                    ps = str(e["payout"]).strip()
                    if not bc._PAYOUT_RE.match(ps) or not (1 <= a <= NF and 1 <= b <= NF):
                        raise ValueError
                    ok[f"{a}-{b}"] = int(ps.replace(",", ""))
                except (KeyError, ValueError, TypeError):
                    ok = {}
                    break
            if ok:
                out[rid] = ok
    return out


# ---------------------------------------------------------------- 組の確率の切り出し

def _slices(P2: np.ndarray, P3: np.ndarray, top2: np.ndarray, top3: np.ndarray, H: int, n: int, frame: np.ndarray) -> dict:
    """◎=H を含む組の確率（18 頭に詰めた配列）。P2[i,j]=i 1 着 j 2 着、P3[i,j,k]=i,j,k の順。"""
    o = {}
    um = np.zeros(NH); uhj = np.zeros(NH); ujh = np.zeros(NH)
    uhj[:n], ujh[:n] = P2[H, :], P2[:, H]
    um[:n] = uhj[:n] + ujh[:n]
    t1 = np.zeros((NH, NH)); t2 = np.zeros((NH, NH)); t3 = np.zeros((NH, NH))
    t1[:n, :n], t2[:n, :n], t3[:n, :n] = P3[H, :, :], P3[:, H, :], P3[:, :, H]
    trio = t1 + t1.T + t2 + t2.T + t3 + t3.T            # P({H,a,b})（a,b の順は問わない）
    o.update(um=um, uhj=uhj, ujh=ujh, t1=t1, t2=t2, t3=t3, trio=trio, wide=trio.sum(axis=1))
    o["win"] = float(P2[H].sum()) if n > 1 else 1.0
    o["fuku"] = float(top3[H] if n >= 8 else (top2[H] if n >= 5 else 0.0))
    U = P2 + P2.T
    F = np.zeros((n, NF))
    F[np.arange(n), np.clip(frame[:n] - 1, 0, NF - 1)] = 1.0
    W = F.T @ U @ F
    W[np.diag_indices(NF)] /= 2.0                        # 同じ枠の 2 頭は i<j の 1 回だけ
    o["waku"] = W[int(frame[H]) - 1].copy()
    return o


def _harville(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from src.pipeline.inference.exotic_probabilities import RankDistribution

    d = RankDistribution(win=q)
    return d._p2, d._p3, d.top2(), d.top3()


@dataclass
class Rows:
    """1 期間 × 1 設定の行（レース × 版のうち ◎ が置けたもの）。配列は先頭の次元が行。"""
    meta: pd.DataFrame          # race_id, ver, year, date, n, H, frame_H, ...
    arr: dict[str, np.ndarray]
    counts: dict


PAIR_KEYS = ("um", "uhj", "ujh", "wide")
TRI_KEYS = ("t1", "t2", "t3", "trio")


def build_rows(p: "bc.Period", body: dict, setting: str, frames: dict, waku: dict) -> Rows:
    spec = body["spec"]
    sk = spec["skip"] if setting == "a2" else {"in": spec["skip"]["in"]}
    k = max(p.k, 1)
    n0 = len(p.races) // k
    own_cache: dict[str, tuple] = {}
    metas, A = [], {}

    def put(name, v, shape, dtype=np.float32):
        A.setdefault(name, []).append(np.asarray(v, dtype=dtype).reshape(shape))

    cnt = {"race_versions": 0, "in_setting": 0, "no_honmei": 0, "k": k, "races": n0}
    for i, (r, f) in enumerate(zip(p.races, p.frecs)):
        cnt["race_versions"] += 1
        if bc._skip(f, sk):
            continue
        cnt["in_setting"] += 1
        H, _why = bc.pick_honmei(r, f, spec["honmei"])
        if H is None:
            cnt["no_honmei"] += 1
            continue
        nums = {int(h) for h in r.hn}
        paid = {int(x) for bt in bc.BET_TYPES for c in r.pay.get(bt, {}) for x in c.replace(">", "-").split("-")}
        if not paid <= nums:          # 払戻の組の馬が出走表に無い（オッズの欠けで取消とみなされた勝ち馬など）: 精算できないので買わず、数える
            cnt["pay_mismatch"] = cnt.get("pay_mismatch", 0) + 1
            bad = cnt.setdefault("pay_mismatch_races", [])
            if r.race_id not in bad:
                bad.append(r.race_id)
            continue
        n = len(r.pw)
        if n > NH:
            raise ValueError(f"{r.race_id} の頭数 {n} が {NH} を超えます")
        fr = np.array([frames.get((r.race_id, int(h)), 0) for h in r.hn])
        if (fr <= 0).any():
            fr = np.where(fr > 0, fr, 0)
        if r.race_id not in own_cache:
            from src.pipeline.inference.exotic_probabilities import fit_blend_distribution

            d, _ = fit_blend_distribution(r.pw, r.p2, r.p3)
            own_cache[r.race_id] = (d._p2, d._p3, d.top2(), d.top3())
        oP2, oP3, ot2, ot3 = own_cache[r.race_id]
        od = np.where(np.isfinite(r.odds) & (r.odds > 0), r.odds, np.nan)
        q = np.where(np.isfinite(od), 1.0 / od, 1e-4)
        q = q / q.sum()
        mP2, mP3, mt2, mt3 = _harville(q)
        so = _slices(oP2, oP3, ot2, ot3, H, n, np.maximum(fr, 1))
        sm = _slices(mP2, mP3, mt2, mt3, H, n, np.maximum(fr, 1))
        for key in PAIR_KEYS + ("waku",):
            put("o_" + key, so[key], (-1,))
            put("m_" + key, sm[key], (-1,))
        for key in TRI_KEYS:
            put("o_" + key, so[key], (NH, NH))
            put("m_" + key, sm[key], (NH, NH))
        for key in ("win", "fuku"):
            put("o_" + key, so[key], ())
            put("m_" + key, sm[key], ())
        # 馬ごとの値（18 頭に詰める）
        pad = lambda x, fill=np.nan: np.concatenate([np.asarray(x, float), np.full(NH - n, fill)])
        put("pw", pad(r.pw), (-1,)); put("p2", pad(r.p2), (-1,)); put("p3", pad(r.p3), (-1,))
        put("odds", pad(od), (-1,)); put("pop", pad(r.pop, 99), (-1,)); put("fin", pad(r.fin, 99), (-1,))
        put("frame", pad(fr, 0), (-1,), np.int8)
        valid = np.zeros(NH, bool); valid[:n] = True; valid[H] = False
        put("valid", valid, (-1,), bool)
        # 確定の払戻（◎ を含む組は密な配列、ボックス用に当たりの組の一覧）
        pay = r.pay
        idx = {int(h): j for j, h in enumerate(r.hn)}
        hH = int(r.hn[H])
        y = {kk: np.zeros(NH, np.int32) for kk in ("um", "uhj", "ujh", "wide")}
        yt = {kk: np.zeros((NH, NH), np.int32) for kk in TRI_KEYS}
        win_um, win_wide, win_trio = [], [], []
        for c, v in pay.get("umaren", {}).items():
            a, b = (idx.get(int(x), -1) for x in c.split("-"))
            if min(a, b) < 0:
                raise ValueError(f"{r.race_id} の馬連の当たり {c} が出走馬に無い")
            win_um.append((a, b, v))
            if H in (a, b):
                y["um"][b if a == H else a] += v
        for c, v in pay.get("wide", {}).items():
            a, b = (idx.get(int(x), -1) for x in c.split("-"))
            if min(a, b) < 0:
                raise ValueError(f"{r.race_id} のワイドの当たり {c} が出走馬に無い")
            win_wide.append((a, b, v))
            if H in (a, b):
                y["wide"][b if a == H else a] += v
        for c, v in pay.get("umatan", {}).items():
            a, b = (idx.get(int(x), -1) for x in c.split(">"))
            if min(a, b) < 0:
                raise ValueError(f"{r.race_id} の馬単の当たり {c} が出走馬に無い")
            if a == H:
                y["uhj"][b] += v
            elif b == H:
                y["ujh"][a] += v
        for c, v in pay.get("sanrenpuku", {}).items():
            a, b, cc = (idx.get(int(x), -1) for x in c.split("-"))
            if min(a, b, cc) < 0:
                raise ValueError(f"{r.race_id} の3連複の当たり {c} が出走馬に無い")
            win_trio.append((a, b, cc, v))
            if H in (a, b, cc):
                x, z = [t for t in (a, b, cc) if t != H]
                yt["trio"][x, z] += v
                yt["trio"][z, x] += v
        for c, v in pay.get("sanrentan", {}).items():
            a, b, cc = (idx.get(int(x), -1) for x in c.split(">"))
            if min(a, b, cc) < 0:
                raise ValueError(f"{r.race_id} の3連単の当たり {c} が出走馬に無い")
            if a == H:
                yt["t1"][b, cc] += v
            elif b == H:
                yt["t2"][a, cc] += v
            elif cc == H:
                yt["t3"][a, b] += v
        for kk in y:
            put("y_" + kk, y[kk], (-1,), np.int32)
        for kk in yt:
            put("y_" + kk, yt[kk], (NH, NH), np.int32)
        put("y_tan", pay.get("tansho", {}).get(str(hH), 0), (), np.int32)
        put("y_fuku", pay.get("fukusho", {}).get(str(hH), 0), (), np.int32)
        put("fuku_sold", int(n >= 5), (), np.int8)
        wk = waku.get(r.race_id)
        yw = np.zeros(NF, np.int32)
        fH = int(fr[H])
        if wk and fH > 0:
            for c, v in wk.items():
                a, b = (int(x) for x in c.split("-"))
                if fH in (a, b):
                    yw[(b if a == fH else a) - 1] += v
        put("y_waku", yw, (-1,), np.int32)
        put("waku_sold", int(bool(wk) and fH > 0 and (fr > 0).all()), (), np.int8)
        ww = np.full((3, 3), -1, np.int32)          # 枠連の当たりの組（枠 a, 枠 b, 払戻）。ボックス用（◎ の枠を含まない組も）
        if wk:
            if len(wk) > 3:
                raise ValueError(f"{r.race_id} の枠連の当たりの組が多すぎます（{len(wk)}）")
            for j, (c, v) in enumerate(wk.items()):
                a, b = (int(x) for x in c.split("-"))
                ww[j] = (a, b, v)
        put("w_waku", ww, (3, 3), np.int32)
        wu = np.full((2, 3), -1, np.int32); wwi = np.full((7, 3), -1, np.int32); wtr = np.full((3, 4), -1, np.int32)
        if len(win_um) > 2 or len(win_wide) > 7 or len(win_trio) > 3:      # 同着で当たりの欄が足りない（黙って切らない）
            raise ValueError(f"{r.race_id} の当たりの組が多すぎます（馬連 {len(win_um)}・ワイド {len(win_wide)}・3連複 {len(win_trio)}）")
        for j, t in enumerate(win_um[:2]):
            wu[j] = t
        for j, t in enumerate(win_wide[:7]):
            wwi[j] = t
        for j, t in enumerate(win_trio[:3]):
            wtr[j] = t
        put("w_um", wu, (2, 3), np.int32); put("w_wide", wwi, (7, 3), np.int32); put("w_trio", wtr, (3, 4), np.int32)
        metas.append({"race_id": r.race_id, "ver": i // max(n0, 1), "date": r.date, "year": int(r.race_id[:4]), "n": n, "H": H,
                      "H_num": hH, "H_fin": float(r.fin[H]), "H_pop": float(r.pop[H]), "H_odds": float(od[H]) if np.isfinite(od[H]) else np.nan})
    arr = {k_: np.stack(v) for k_, v in A.items()}
    meta = pd.DataFrame(metas)
    return Rows(meta, arr, cnt)


def save_rows(rows: Rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **rows.arr)
    rows.meta.to_parquet(path.with_suffix(".meta.parquet"), index=False)
    path.with_suffix(".counts.json").write_text(json.dumps(rows.counts, ensure_ascii=False), encoding="utf-8")


def load_rows(path: Path) -> Rows:
    z = np.load(path)
    return Rows(pd.read_parquet(path.with_suffix(".meta.parquet")), {k: z[k] for k in z.files},
                json.loads(path.with_suffix(".counts.json").read_text(encoding="utf-8")))


def build_period(name: str, body: dict | None = None) -> dict[str, Rows]:
    """期間 name の 2 つの設定の行を作る（台帳の申告は呼ぶ側で、この関数の前に行う）。"""
    body = body or load_frozen()
    per, _ = bc.prepare_periods(period_inputs(body, name), (name,), cuts=body["qcuts"])
    p = per[name]
    years = sorted({int(r.race_id[:4]) for r in p.races})
    ids = {r.race_id for r in p.races}
    frames = frames_of(years)
    waku = wakuren_payoffs(ids, years)
    out = {}
    for s in SETTINGS:
        rows = build_rows(p, body, s, frames, waku)
        rows.counts["drop"] = {k: int(v) for k, v in p.drop.items()}
        out[s] = rows
    return out
