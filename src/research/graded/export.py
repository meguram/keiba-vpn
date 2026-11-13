"""重賞のルールベース予想家: HTML の材料（graded_report.json）とレースごとの JSON を作る（計算は run.py の各段の出力を読むだけ）。

- 重賞ごと: 主の集計（同じレース名 × 最新の回と同じコース、2016〜2024 年）と副の集計（同じコースの OP 以上の別レース）の、
  採用した型のパターンの [1-2-3-着外]・勝率・連対率・複勝率・縮小後の複勝率。最新の回と別のコースで行われた年（外した回）と理由。
- 検証のレース（2025・2026）: 各馬の印・点・着順と、当てはまったパターン（好成績・凡走）。
- 型の採否の一覧と、作れなかった切り口。
出力: <GRADED_OUT>/graded_report.json、data/local/graded/races/<race_id>.json（将来めぐ分析の画面に載せられる形。研究用で commit しない）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.graded import catalog as CAT
from src.research.graded import model as M
from src.research.graded import tally as TL
from src.research.graded.features import GRADED_OUT
from src.research.graded.run import PREREG, REPO, _ctx, _load, _rows, load_payoff

RACES_DIR = REPO / "data/local/graded/races"
TABLE_END = 2024
MAX_PATTERNS = 14          # 1 頭あたり HTML に載せるパターンの数（|点| の大きい順）


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


def race_tables(d: pd.DataFrame, L: pd.DataFrame, pre: dict, types: list[str]) -> list[dict]:
    """重賞ごとの主・副のパターン表（2016〜2024 年、最新の回のコース）。"""
    sh, lam = pre["shrink"], pre["score"]["lam"]
    b = TL.base_frame(d)
    b["year"] = pd.to_datetime(b["dt"]).dt.year
    b = b[b["year"] <= TABLE_END]
    g = d[d["grade"].isin(TL.GRADES)].drop_duplicates("race_id")
    latest = g.sort_values("dt").groupby("rkey").tail(1).set_index("rkey")
    out = []
    lev = L.reindex(b["row"])
    lev.index = b.index

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
        out.append({"rkey": rkey, "grade": r["grade"], "course": course, "venue": r["venue"], "surf": r["surf"], "dist": int(r["dist"]),
                    "latest": str(pd.Timestamp(r["dt"]).date()), "n_main_ed": n_main_ed, "n_main_runs": int(main_m.sum()),
                    "n_sub_runs": int(sub_m.sum()), "excluded": excluded, "patterns": pats})
    return out


def eval_races(pre: dict, rows: pd.DataFrame, pay: dict) -> list[dict]:
    """検証のレース（前日版）: 各馬の印・点・着順・該当パターン。"""
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
    out = []
    for rid, g in mk.groupby("race_id"):
        ri = info.loc[rid]
        g = g.merge(rows[["row", "horse_id", "fin", "jockey_name"]], on="row", how="left")
        if v6r is not None:
            g = g.merge(v6r[v6r["race_id"] == rid][["hn", "v6_rank"]], on="hn", how="left")
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
            horses.append({"hn": int(h.hn), "mk": h.mark, "sc": round(float(h.score), 3), "ng": int(h.n_good_main),
                           "fin": None if pd.isna(h.fin) else int(h.fin), "jk": h.jockey_name,
                           "v6": None if v6r is None or pd.isna(getattr(h, "v6_rank", np.nan)) else int(h.v6_rank),
                           "n_good": int(sum(1 for p in pl if p["k"] == "好成績")), "n_bad": int(sum(1 for p in pl if p["k"] == "凡走")), "p": pl})
        top = horses[0]["hn"] if horses else None
        out.append({"race_id": rid, "date": str(pd.Timestamp(ri["dt"]).date()), "name": ri["race_name"], "rkey": ri["rkey"], "grade": ri["grade"],
                    "course": ri["course"], "horses": horses,
                    "pay": {"tansho": pay.get((rid, "tansho"), {}).get(str(top), 0), "fukusho": pay.get((rid, "fukusho"), {}).get(str(top), 0)},
                    "has_pay": (rid, "tansho") in pay})
    return out


def cmd_export() -> None:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    ev = json.loads((GRADED_OUT / "eval.json").read_text(encoding="utf-8"))
    d, L = _load()
    rows = _rows(d)
    pay = load_payoff()
    types_used = sorted(set(pre["sets"]["dayof"]["adopted"]))
    rt = race_tables(d, L, pre, types_used)
    er = eval_races(pre, rows, pay)
    tinfo = [{"id": t.id, "name": t.name, "family": CAT.FAMILIES[t.family], "day_of": t.day_of, "start_year": t.start_year, "note": t.note,
              **{k: pre["types"].get(t.id, {}).get(k) for k in ("z_disc", "z_conf", "races_disc", "bh10_disc", "status")},
              "used_preday": t.id in pre["sets"]["preday"]["used"], "used_dayof": t.id in pre["sets"]["dayof"]["used"]}
             for t in CAT.TYPES]
    rep = {"prereg": {k: pre[k] for k in ("frozen_at", "periods", "thresholds", "shrink", "score", "sets")}, "eval": ev, "types": tinfo,
           "not_built": [{"id": a, "name": b, "reason": c} for a, b, c in CAT.NOT_BUILT], "races": rt, "eval_races": er}
    (GRADED_OUT / "graded_report.json").write_text(json.dumps(rep, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    RACES_DIR.mkdir(parents=True, exist_ok=True)
    tn = {t.id: t.name for t in CAT.TYPES}
    for r in er:
        for h in r["horses"]:
            for p in h["p"]:
                p["name"] = tn.get(p["t"])
        Path(RACES_DIR / f"{r['race_id']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print("export", len(rt), "races tables,", len(er), "eval races")
