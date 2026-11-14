"""重賞のルールベース予想家: HTML の材料（graded_report.json）とレースごとの JSON を作る（計算は run.py の各段の出力を読むだけ）。

- 重賞ごと: 主の集計（同じレース名 × 最新の回と同じコース、2016〜2024 年）と副の集計（同じコースの OP 以上の別レース）の、
  採用した型のパターンの [1-2-3-着外]・勝率・連対率・複勝率・縮小後の複勝率。最新の回と別のコースで行われた年（外した回）と理由。
- 検証のレース（2025・2026）: 各馬の印・点・着順と、当てはまったパターン（好成績・凡走）。
- 型の採否の一覧と、作れなかった切り口。
出力: <GRADED_OUT>/graded_report.json、data/local/graded/races/<race_id>.json（将来めぐ分析の画面に載せられる形。研究用で commit しない）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.graded import catalog as CAT
from src.research.graded import model as M
from src.research.graded import tally as TL
from src.research.graded.features import GRADED_OUT
from src.research.graded.run import PREREG, PREREG_ROI, REPO, _ctx, _load, _rows, load_payoff, with_returns

RACES_DIR = REPO / "data/local/graded/races"
TABLE_END = 2024
MAX_PATTERNS = 14          # 1 頭あたり HTML に載せるパターンの数（|点| の大きい順）
N_BOOT_TABLE = 1000        # 重賞ごとの表の回収率の区間（レース単位のブートストラップ、ポアソン(1) の重み）


class _Boot:
    """レース単位のブートストラップ（ポアソン(1) の重み）で、回収率の 95% 区間を出す。重みはレースごとに 1 回だけ作る。"""

    def __init__(self, race_ids, n_boot: int = N_BOOT_TABLE, seed: int = 0):
        rid = pd.Index(sorted(set(race_ids)))
        self.idx = rid
        self.W = np.random.default_rng(seed).poisson(1.0, size=(len(rid), n_boot)).astype(np.float32)

    def ci(self, race_ids: np.ndarray, n: np.ndarray, r: np.ndarray) -> list:
        if n.sum() <= 0:
            return [None, None, None]
        Wi = self.W[self.idx.get_indexer(race_ids)]
        nb = Wi.T @ n.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (Wi.T @ r.astype(np.float32)) / nb
        b = b[np.isfinite(b)]
        return [round(float(r.sum() / n.sum()), 3), round(float(np.quantile(b, 0.025)), 3), round(float(np.quantile(b, 0.975)), 3)]


def _roi_cells(x: pd.DataFrame, boot: _Boot) -> dict:
    """水準ごとの単勝・複勝の回収率 [点, 下限, 上限] と、払戻の解けた走数。x は base_frame の行に level を付けたもの。"""
    pr = x.groupby(["level", "race_id"], sort=False)[["np", "rt", "rf"]].sum().reset_index()
    out = {}
    for lv, g in pr.groupby("level", sort=False):
        rid, n = g["race_id"].to_numpy(), g["np"].to_numpy()
        out[lv] = {"np": int(n.sum()), "t": boot.ci(rid, n, g["rt"].to_numpy()), "f": boot.ci(rid, n, g["rf"].to_numpy())}
    return out


def roi_flag(pset: set, t: str, lv: str, x: float, kind: str, layer: str | None = None, prim: dict | None = None) -> str:
    """主に見る版の採否から、表の「回収率」の判定。prim（主に見る版）があれば、x を作った層が版の層に入り、
    型が点に使う型（束ねの代表）であるときだけ「回収率で残った」とする（印に効いていないパターンに付けない）。"""
    from src.research.graded.roi import LAYERS
    d = "好" if x > 0 else "凡" if x < 0 else ""
    in_use = prim is None or (layer in LAYERS[prim["params"]["layer"]] and t in prim["used"])
    if (t, lv, d) in pset and in_use:
        return "回収率で残った"
    if kind in ("好成績", "凡走"):
        return "成績だけ（回収率は足りない）"
    return ""


def _rec(n, w1, w2, w3):
    n, w1, w2, w3 = (int(round(v)) for v in (n, w1, w2, w3))
    return [w1, w2, w3, n - w1 - w2 - w3]


def _rates(n, w1, w2, w3):
    if n <= 0:
        return [None, None, None]
    return [round(w1 / n, 3), round((w1 + w2) / n, 3), round((w1 + w2 + w3) / n, 3)]


def _kind(layer, x, edhit, ed):
    if layer == "主" and x > 0 and edhit >= M.GOOD_EDHIT:
        return "好成績"
    if layer == "主" and x < 0 and ed >= M.MIN_MAIN_ED and edhit <= 1:
        return "凡走"
    return "参考"


def race_tables(d: pd.DataFrame, L: pd.DataFrame, pre: dict, types: list[str], pset: set | None = None, prim: dict | None = None) -> list[dict]:
    """重賞ごとの主・副のパターン表（2016〜2024 年、最新の回のコース）。pset（主に見る回収率の版で残ったパターン）があれば、
    主・副の単勝・複勝の回収率と 95% 区間、回収率の判定を足す。"""
    sh, lam = pre["shrink"], pre["score"]["lam"]
    b = TL.base_frame(d)
    b["year"] = pd.to_datetime(b["dt"]).dt.year
    b = b[b["year"] <= TABLE_END]
    g = d[d["grade"].isin(TL.GRADES)].drop_duplicates("race_id")
    latest = g.sort_values("dt").groupby("rkey").tail(1).set_index("rkey")
    out = []
    lev = L.reindex(b["row"])
    lev.index = b.index
    boot = _Boot(b["race_id"]) if pset is not None else None

    def agg_of(m, lv):
        x = b.loc[m].assign(level=lv[m].to_numpy())
        a = x.groupby("level").agg(n=("w1", "size"), w1=("w1", "sum"), w2=("w2", "sum"), w3=("w3", "sum"), e3=("e3", "sum"))
        a["ed"] = x.groupby("level")["race_id"].nunique()
        a["edhit"] = x[x["top3"] > 0].groupby("level")["race_id"].nunique()
        return a.fillna(0)
    glob_m = b["graded"].to_numpy()
    glob = {t: agg_of(glob_m, lev[t]).add_prefix("glob_") for t in types}
    for rkey, r in latest.iterrows():
        course = r["course"]
        eds = g[g["rkey"] == rkey].sort_values("dt")
        excluded = [{"year": int(e.year), "course": e.course, "reason": "別のコースで行われた回（改修による代替開催・距離の変更など）"}
                    for e in eds.itertuples() if e.course != course]
        same_c = (b["course"] == course).to_numpy()
        same_k = (b["rkey"] == rkey).to_numpy()
        op = b["op"].to_numpy()
        main_m, sub_m, sub4_m = op & same_k & same_c, op & same_c & ~same_k, same_c & ~same_k
        n_main_ed = int(b.loc[main_m, "race_id"].nunique())
        pats = []
        for t in types:
            lv = lev[t]
            parts = [agg_of(main_m, lv).add_prefix("main_"), agg_of(sub_m, lv).add_prefix("sub_"), agg_of(sub4_m, lv).add_prefix("sub4_"), glob[t]]
            a = pd.concat(parts, axis=1).fillna(0)
            a = a[(a["main_n"] > 0) & ~a.index.isin(list(M.NO_INFO))]
            if a.empty:
                continue
            a = a.rename_axis("level").reset_index()
            a["level"] = a["level"].astype(str)
            a["type"] = t
            a["all_main_n"] = float(main_m.sum())
            a["all_sub_n"] = float(sub_m.sum())
            S = M.shrink(a, sh["m_g"], sh["m_s"], sh["m_m"], lam)
            if boot is not None:
                rm = _roi_cells(b.loc[main_m].assign(level=lv[main_m].astype(str).to_numpy()), boot)
                rs = _roi_cells(b.loc[sub_m].assign(level=lv[sub_m].astype(str).to_numpy()), boot)
                rs4 = _roi_cells(b.loc[sub4_m].assign(level=lv[sub4_m].astype(str).to_numpy()), boot)
            for s in S.itertuples():
                kind = _kind(s.layer, s.x, s.main_edhit, s.main_ed)
                use4 = s.sub_kind == "3勝以上"
                sn, s1, s2, s3 = (s.sub4_n, s.sub4_w1, s.sub4_w2, s.sub4_w3) if use4 else (s.sub_n, s.sub_w1, s.sub_w2, s.sub_w3)
                pats.append({
                    "t": t, "lv": s.level, "k": kind, "x": round(float(s.x), 3), "ly": s.layer,
                    "m": _rec(s.main_n, s.main_w1, s.main_w2, s.main_w3), "mr": _rates(s.main_n, s.main_w1, s.main_w2, s.main_w3),
                    "med": [int(s.main_edhit), int(s.main_ed)],
                    "ps": round(float(s.r_m * s.main_e3 / s.main_n), 3) if s.main_n > 0 else None,
                    "s": _rec(sn, s1, s2, s3), "sr": _rates(sn, s1, s2, s3), "sk": s.sub_kind,
                })
                if boot is not None:
                    sr_ = (rs4 if use4 else rs).get(s.level, {})
                    pats[-1].update({"rm": rm.get(s.level), "rs": sr_ or None, "rk": roi_flag(pset, t, s.level, float(s.x), kind, s.layer, prim)})
        out.append({"rkey": rkey, "grade": r["grade"], "course": course, "venue": r["venue"], "surf": r["surf"], "dist": int(r["dist"]),
                    "latest": str(pd.Timestamp(r["dt"]).date()), "n_main_ed": n_main_ed, "n_main_runs": int(main_m.sum()),
                    "n_sub_runs": int(sub_m.sum()), "excluded": excluded, "patterns": pats})
    return out


def _asof_roi(p, layer: str) -> list | None:
    """as-of の件数の、その層の [払戻の解けた走数, 単勝の回収率, 複勝の回収率]。"""
    n = getattr(p, f"{layer}_np", 0) or 0
    if n <= 0:
        return None
    return [int(n), round(float(getattr(p, f"{layer}_rt")) / n, 3), round(float(getattr(p, f"{layer}_rf")) / n, 3)]


def eval_races(pre: dict, rows: pd.DataFrame, pay: dict, pre_roi: dict | None = None) -> list[dict]:
    """検証のレース（前日版）: 各馬の印・点・着順・該当パターン。pre_roi があれば、主に見る回収率の版の順位（rv）と、
    各パターンの as-of の回収率（主・副・全重賞）・回収率の判定を足す。"""
    from src.research.graded.run import EVAL, _yr
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", rows), EVAL)
    sh, sc = pre["shrink"], pre["score"]
    S = M.shrink(T, sh["m_g"], sh["m_s"], sh["m_m"], sc["lam"])
    used = pre["sets"]["preday"]["used"]
    S = S[S["type"].isin(used)]
    mk = pd.read_parquet(GRADED_OUT / "marks_前日版.parquet")
    v6 = GRADED_OUT / "marks_v6e本番.parquet"
    v6r = pd.read_parquet(v6)[["race_id", "hn", "rank"]].rename(columns={"rank": "v6_rank"}) if v6.exists() else None
    info = rows.drop_duplicates("race_id").set_index("race_id")
    pset, rvr, prim = set(), None, None
    if pre_roi is not None:
        prim = next(v for v in pre_roi["versions"] if v["primary"])
        pset = {tuple(p) for p in prim["patterns"]}
        pf = GRADED_OUT / f"marks_{prim['name']}.parquet"
        rvr = pd.read_parquet(pf)[["race_id", "hn", "rank"]].rename(columns={"rank": "rv"}) if pf.exists() else None
    out = []
    for rid, g in mk.groupby("race_id"):
        ri = info.loc[rid]
        g = g.merge(rows[["row", "horse_id", "fin", "jockey_name"]], on="row", how="left")
        if v6r is not None:
            g = g.merge(v6r[v6r["race_id"] == rid][["hn", "v6_rank"]], on="hn", how="left")
        if rvr is not None:
            g = g.merge(rvr[rvr["race_id"] == rid][["hn", "rv"]], on="hn", how="left")
        horses = []
        for h in g.sort_values("rank").itertuples():
            ps = S[S["row"] == h.row].copy()
            ps["ax"] = ps["x"].abs()
            ps = ps.sort_values("ax", ascending=False)
            pl = []
            for p in ps.head(MAX_PATTERNS).itertuples():
                pl.append({"t": p.type, "lv": p.level, "x": round(float(p.x), 3), "ly": p.layer, "k": _kind(p.layer, p.x, p.main_edhit, p.main_ed),
                           "m": _rec(p.main_n, p.main_w1, p.main_w2, p.main_w3), "med": [int(p.main_edhit), int(p.main_ed)],
                           "s": _rec(p.sub_n, p.sub_w1, p.sub_w2, p.sub_w3) if p.sub_kind == "OP以上" else _rec(p.sub4_n, p.sub4_w1, p.sub4_w2, p.sub4_w3),
                           "g": _rec(p.glob_n, p.glob_w1, p.glob_w2, p.glob_w3)})
                if pre_roi is not None:
                    sl = "sub" if p.sub_kind == "OP以上" else "sub4"
                    pl[-1].update({"rm": _asof_roi(p, "main"), "rs": _asof_roi(p, sl), "rg": _asof_roi(p, "glob"),
                                   "rk": roi_flag(pset, p.type, p.level, float(p.x), pl[-1]["k"], p.layer, prim)})
            horses.append({"hn": int(h.hn), "mk": h.mark, "sc": round(float(h.score), 3), "ng": int(h.n_good_main),
                           "fin": None if pd.isna(h.fin) else int(h.fin), "jk": h.jockey_name,
                           "v6": None if v6r is None or pd.isna(getattr(h, "v6_rank", np.nan)) else int(h.v6_rank),
                           "rv": None if rvr is None or pd.isna(getattr(h, "rv", np.nan)) else int(h.rv),
                           "n_good": int(sum(1 for p in pl if p["k"] == "好成績")), "n_bad": int(sum(1 for p in pl if p["k"] == "凡走")), "p": pl})
        top = horses[0]["hn"] if horses else None
        out.append({"race_id": rid, "date": str(pd.Timestamp(ri["dt"]).date()), "name": ri["race_name"], "rkey": ri["rkey"], "grade": ri["grade"],
                    "course": ri["course"], "horses": horses,
                    "pay": {"tansho": pay.get((rid, "tansho"), {}).get(str(top), 0), "fukusho": pay.get((rid, "fukusho"), {}).get(str(top), 0)},
                    "has_pay": (rid, "tansho") in pay})
    return out


def roi_summary(pre_roi: dict) -> dict:
    """HTML 用: 回収率で選ぶ版の一覧（パターンの数・使う型）と、主に見る版で残ったパターンの発見・確認の実際の回収率。"""
    U = pd.read_parquet(GRADED_OUT / "roi_units.parquet")
    prim = next(v for v in pre_roi["versions"] if v["primary"])
    lay = prim["params"]["layer"]
    keep = {tuple(p) for p in prim["patterns"]}
    U = U[U["layer_set"] == lay]
    k = list(zip(U["type"], U["level"].astype(str), U["dir"]))
    U = U[[x in keep for x in k]]
    pats = []
    for (t, lv, dr), g in U.groupby(["type", "level", "dir"]):
        e = {"t": t, "lv": lv, "dir": dr}
        for r in g.itertuples():
            # 区間は採否に使った片側の分位点（発見 5%・95%、確認 10%・90%）。両側で見ると 90%・80% の区間
            lo, hi = ("q050", "q950") if r.period == "disc" else ("q100", "q900")
            e[r.period] = {"n": int(r.n), "races": int(r.races), "y3": round(r.y3 / r.n, 3), "e3": round(r.e3 / r.n, 3),
                           **{k: [round(getattr(r, f"roi_ret_{c}"), 3), round(getattr(r, f"{lo}_ret_{c}"), 3), round(getattr(r, f"{hi}_ret_{c}"), 3)]
                              for k, c in (("t", "tan"), ("f", "fuku"))}}
        pats.append(e)
    return {"frozen_at": pre_roi["frozen_at"], "primary": pre_roi["primary"], "n_versions": pre_roi["n_versions"], "grid": pre_roi["grid"],
            "rules": pre_roi["rules"], "overall_roi": pre_roi["overall_roi"],
            "versions": [{"name": v["name"], "params": v["params"], "family": v.get("family"), "primary": v["primary"], "n_good": v["n_good"], "n_bad": v["n_bad"],
                          "used": v["used"]} for v in pre_roi["versions"]],
            "primary_patterns": pats}


def cmd_export() -> None:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    ev = json.loads((GRADED_OUT / "eval.json").read_text(encoding="utf-8"))
    round2 = ev.get("evaluation_round") == 2
    ph = GRADED_OUT / "eval_posthoc_v6e.json"
    if round2 and ph.exists():   # 評価の後に足した本番 v6e の 2026 年を含む比較（印・規則は凍結のまま）で、2025 年だけの比較を置き換える
        P = json.loads(ph.read_text(encoding="utf-8"))
        for name, per in P["versions"].items():
            if name not in ev["versions"]:
                continue
            for lab, e in per.items():
                ev["versions"][name].setdefault(lab, {}).update(e)
        # 事後の比較を足していない版の、2026 年を含む期間の v6e本番 との比較は 2025 年だけの値なので消す（期間の範囲を版でそろえる）
        for name, per in ev["versions"].items():
            if name in P["versions"]:
                continue
            for lab in ("2026", "2025-2026"):
                for k in ("base_v6e本番", "self_same_races_v6e本番", "diff_vs_v6e本番"):
                    per.get(lab, {}).pop(k, None)
        ev["baselines"]["v6e本番"] = P["paths"]
        ev["posthoc_v6e_note"] = P["note"]
    pre_roi = json.loads(PREREG_ROI.read_text(encoding="utf-8")) if round2 else None
    d, L = _load()
    if round2:
        d = with_returns(d)
    rows = _rows(d)
    pay = load_payoff()
    types_used = sorted(set(pre["sets"]["dayof"]["adopted"]))
    prim = next(v for v in pre_roi["versions"] if v["primary"]) if round2 else None
    pset = {tuple(p) for p in prim["patterns"]} if round2 else None
    rt = race_tables(d, L, pre, types_used, pset, prim)
    er = eval_races(pre, rows, pay, pre_roi)
    tinfo = [{"id": t.id, "name": t.name, "family": CAT.FAMILIES[t.family], "day_of": t.day_of, "start_year": t.start_year, "note": t.note,
              **{k: pre["types"].get(t.id, {}).get(k) for k in ("z_disc", "z_conf", "races_disc", "bh10_disc", "status")},
              "used_preday": t.id in pre["sets"]["preday"]["used"], "used_dayof": t.id in pre["sets"]["dayof"]["used"]}
             for t in CAT.TYPES]
    rep = {"prereg": {k: pre[k] for k in ("frozen_at", "periods", "thresholds", "shrink", "score", "sets")}, "eval": ev, "types": tinfo,
           "not_built": [{"id": a, "name": b, "reason": c} for a, b, c in CAT.NOT_BUILT], "races": rt, "eval_races": er}
    if round2:
        rep["roi"] = roi_summary(pre_roi)
        r1 = Path(os.environ.get("GRADED_OUT_R1", "/home/jovyan/work/graded_out")) / "eval.json"   # 第 1 回の評価（直す前のデータ）
        if not r1.exists():
            print(f"注意: 第 1 回の評価 {r1} が無いので、HTML の「第 1 回の評価」の節を出さない（GRADED_OUT_R1 で置き場を渡す）", flush=True)
        else:
            e1 = json.loads(r1.read_text(encoding="utf-8"))
            rep["eval_r1"] = {"evaluated_at": e1.get("evaluated_at"), "versions": {k: {p: {"all": x["all"]} for p, x in v.items()} for k, v in e1["versions"].items()}}
    (GRADED_OUT / "graded_report.json").write_text(json.dumps(rep, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    RACES_DIR.mkdir(parents=True, exist_ok=True)
    tn = {t.id: t.name for t in CAT.TYPES}
    for r in er:
        for h in r["horses"]:
            for p in h["p"]:
                p["name"] = tn.get(p["t"])
        Path(RACES_DIR / f"{r['race_id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print("export", len(rt), "races tables,", len(er), "eval races")
