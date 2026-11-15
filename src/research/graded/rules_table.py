"""重賞のルールベース予想家: 各重賞で点に効いているルール（型 × 水準）を、1 行 = 重賞 × ルールの表にする材料（graded_rules.json）。

- 対象の型 = 前日版・当日版の点に使う型（束ねの代表、prereg.json の sets.*.used）と、主に見る回収率版の点に使う型（prereg_roi.json）。
- 対象の水準 = 各重賞の最新の回のコースで、点 x が 0 でないもの（主に件数の無い水準も、副・全重賞の層で点が付けば載せる）。
- 件数・率・回収率は、x を作った層（主・副・全重賞）の母集団の値。期間は 2016〜2024 年（export.race_tables と同じ凍結の値）。
- 採否: 前日版・当日版は型が used に入り x≠0 なら採用。回収率版は export.roi_flag で「回収率で残った」のときだけ採用（その他の水準は点 0）。
計算は集計を数え直すだけで、評価はやり直さない。出力: <GRADED_OUT>/graded_rules.json（列ごとの配列と辞書。HTML は
src/scripts/docs/build_graded_rules_table_html.py が作る）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.graded import catalog as CAT
from src.research.graded import model as M
from src.research.graded import tally as TL
from src.research.graded.export import TABLE_END, _kind, roi_flag
from src.research.graded.features import GRADED_OUT
from src.research.graded.run import PREREG, PREREG_ROI, _load, with_returns

SUMS = ["n", "w1", "w2", "w3", "e3", "np", "rt", "rf", "ed", "edhit"]
LAYER_NAME = {"主": "主", "副": "副", "全体": "全重賞"}   # shrink の「全体」は全重賞（b.graded）の集計
OUT_NAME = "graded_rules.json"


def _grp(x: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """keys ごとの走数・着数・期待・払戻と、該当馬のいた回・3着内が出た回の数（回はレース単位で数える）。"""
    a = x.groupby(keys, observed=True).agg(n=("w1", "size"), w1=("w1", "sum"), w2=("w2", "sum"), w3=("w3", "sum"), e3=("e3", "sum"),
                                           np=("np", "sum"), rt=("rt", "sum"), rf=("rf", "sum"))
    a["ed"] = x.groupby(keys, observed=True)["race_id"].nunique()
    a["edhit"] = x[x["top3"] > 0].groupby(keys, observed=True)["race_id"].nunique()
    return a.fillna(0.0)


def _sum_lv(g: pd.DataFrame | None) -> pd.DataFrame:
    if g is None or g.empty:
        return pd.DataFrame(columns=SUMS)
    return g.groupby(level="level")[SUMS].sum()


def rule_rows(d: pd.DataFrame, L: pd.DataFrame, pre: dict, pre_roi: dict) -> tuple[list[dict], list[dict]]:
    """(重賞の一覧, 行の一覧)。行は 重賞 × 型 × 水準 で x≠0 のもの。"""
    sh, lam = pre["shrink"], pre["score"]["lam"]
    prim = next(v for v in pre_roi["versions"] if v["primary"])
    pset = {tuple(p) for p in prim["patterns"]}
    used = {s: set(pre["sets"][s]["used"]) for s in ("preday", "dayof")}
    types = sorted(used["preday"] | used["dayof"] | set(prim["used"]))
    b = TL.base_frame(d)
    b = b[pd.to_datetime(b["dt"]).dt.year <= TABLE_END]
    if b["np"].isna().all():
        raise SystemExit("払戻の列が無い（with_returns を通した d を渡す）")
    g = d[d["grade"].isin(TL.GRADES)].drop_duplicates("race_id")
    latest = g.sort_values("dt").groupby("rkey").tail(1).set_index("rkey")
    lev = L.reindex(b["row"])
    lev.index = b.index
    races = []
    for rkey, r in latest.iterrows():
        c = r["course"]
        m_main = (b["course"] == c) & (b["rkey"] == rkey) & b["op"]
        races.append({"rkey": rkey, "grade": r["grade"], "course": c, "venue": r["venue"], "surf": r["surf"], "dist": int(r["dist"]),
                      "latest": str(pd.Timestamp(r["dt"]).date()), "n_main_ed": int(b.loc[m_main, "race_id"].nunique()),
                      "all_main_n": float(m_main.sum()), "all_sub_n": float(((b["course"] == c) & (b["rkey"] != rkey) & b["op"]).sum())})
    courses = {x["course"] for x in races}
    rows = []
    for t in types:
        x = b.assign(level=lev[t].astype("string").to_numpy())
        x = x[x["level"].notna()]
        G = _grp(x[x["graded"]], ["level"])                                    # 全重賞（型ごとに 1 回）
        C = _grp(x[x["course"].isin(courses)], ["course", "rkey", "op", "level"])
        for ri, r in enumerate(races):
            try:
                Cc = C.xs(r["course"], level="course")
            except KeyError:
                Cc = C.iloc[:0].droplevel("course")
            rk = Cc.index.get_level_values("rkey") == r["rkey"]
            op = Cc.index.get_level_values("op").to_numpy(bool)
            parts = {"main": _sum_lv(Cc[rk & op]), "sub": _sum_lv(Cc[~rk & op]), "sub4": _sum_lv(Cc[~rk]), "glob": G}
            idx = G.index.union(parts["main"].index).union(parts["sub4"].index)
            a = pd.concat([p.reindex(idx).add_prefix(f"{k}_") for k, p in parts.items()], axis=1).fillna(0.0)
            a = a.rename_axis("level").reset_index()
            a["level"] = a["level"].astype(str)
            a["type"] = t
            a["all_main_n"], a["all_sub_n"] = r["all_main_n"], r["all_sub_n"]
            S = M.shrink(a, sh["m_g"], sh["m_s"], sh["m_m"], lam)
            S = S[S["x"] != 0]
            for s in S.itertuples():
                ly = LAYER_NAME[s.layer]
                p = "main" if s.layer == "主" else "glob" if s.layer == "全体" else ("sub4" if s.sub_kind == "3勝以上" else "sub")
                v = {k: float(getattr(s, f"{p}_{k}")) for k in ("n", "w1", "w2", "w3", "np", "rt", "rf")}
                kind = _kind(s.layer, s.x, s.main_edhit, s.main_ed)
                rk_ = roi_flag(pset, t, s.level, float(s.x), kind, s.layer, prim)
                dr = "好" if s.x > 0 else "凡"
                flags = (1 if t in used["preday"] else 0) | (2 if t in used["dayof"] else 0) | (4 if rk_ == "回収率で残った" else 0)
                if not flags:
                    continue
                n = int(round(v["n"]))
                w = [int(round(v[k])) for k in ("w1", "w2", "w3")]
                rows.append({
                    "r": ri, "t": t, "lv": s.level, "dir": dr, "k": kind, "ly": ly,
                    "sub_kind": s.sub_kind if s.layer == "副" else "",
                    "rec": w + [n - sum(w)], "np": int(round(v["np"])),
                    "roi_t": round(v["rt"] / v["np"], 3) if v["np"] > 0 else None,
                    "roi_f": round(v["rf"] / v["np"], 3) if v["np"] > 0 else None,
                    "x": round(float(s.x), 4), "rm": round(float(s.r_m), 4),
                    "main": [int(round(getattr(s, f"main_{k}"))) for k in ("n", "w1", "w2", "w3")], "med": [int(s.main_edhit), int(s.main_ed)],
                    # 8 = パターンは回収率の基準を満たしたが、この行は層（全重賞）が回収率版の層に入らないので点に使わない
                    "flags": flags | (8 if (t, s.level, dr) in pset and not flags & 4 else 0), "roi_kept": rk_ == "回収率で残った",
                })
    return races, rows


def pack(races: list[dict], rows: list[dict], pre: dict, pre_roi: dict) -> dict:
    """行を列ごとの配列にし、繰り返す文字列（型・水準・区分・層）は辞書の番号にする。"""
    tn = {t.id: t for t in CAT.TYPES}
    used = sorted({r["t"] for r in rows})
    # 型の重みは成績版（前日版・当日版）の点に使う型だけ。z が無い型は使う型の最小の z（run.scored と同じ。今の凍結では該当なし）。
    # 回収率版は重みを付けない（roi.marks_for → M.score(s, used)）ので、回収率版だけで使う型は空欄
    sp = set(pre["sets"]["preday"]["used"]) | set(pre["sets"]["dayof"]["used"])
    zs = [pre["types"].get(t, {}).get("z_disc") for t in sp]
    zmin = min([z for z in zs if z is not None] or [1.0])
    tinfo = [{"id": t, "name": tn[t].name if t in tn else t, "family": CAT.FAMILIES[tn[t].family] if t in tn else "",
              "w": round(pre["types"].get(t, {}).get("z_disc") or zmin, 3) if t in sp else None} for t in used]
    ti = {t: i for i, t in enumerate(used)}
    lvs = sorted({r["lv"] for r in rows})
    li = {v: i for i, v in enumerate(lvs)}
    dic = {"k": ["好成績", "凡走", "参考"], "ly": ["主", "副", "全重賞"], "dir": ["好", "凡"], "sub_kind": ["", "OP以上", "3勝以上"]}
    col = {"r": [r["r"] for r in rows], "t": [ti[r["t"]] for r in rows], "lv": [li[r["lv"]] for r in rows]}
    for k, vs in dic.items():
        ix = {v: i for i, v in enumerate(vs)}
        col[k] = [ix[r[k]] for r in rows]
    for k in ("rec", "np", "roi_t", "roi_f", "x", "main", "flags"):   # 縮めた比は exp(x)、med は表に出さないので載せない
        col[k] = [r[k] for r in rows]
    prim = next(v for v in pre_roi["versions"] if v["primary"])
    return {"frozen_at": pre["frozen_at"], "frozen_at_roi": pre_roi["frozen_at"], "primary": prim["name"],
            "primary_patterns": prim["patterns"], "years": [2016, TABLE_END], "overall_roi": pre_roi["overall_roi"],
            "n_rows": len(rows), "races": races, "types": tinfo, "levels": lvs, "dict": dic, "cols": col}


def cmd_export_rules() -> None:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    pre_roi = json.loads(PREREG_ROI.read_text(encoding="utf-8"))
    d, L = _load()
    d = with_returns(d)
    races, rows = rule_rows(d, L, pre, pre_roi)
    out = GRADED_OUT / OUT_NAME
    out.write_text(json.dumps(pack(races, rows, pre, pre_roi), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("rules", len(races), "races,", len(rows), "rows ->", out, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)
