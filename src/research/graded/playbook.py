"""重賞ごとの「◎ の選び方・消しの選び方」の早見表（分析サイクル 20 回の最後の規則を、重賞ごとに 1 枚のカードにする）。

- 規則は <GRADED_OUT>/cycles_rules.json（python -m src.research.graded.run cycles の出力）のまま。評価はやり直さない。
- 規則を表（cycles.Table）に当て直して、2025・2026 年と 2018〜2024 年の成績を数える。2025・2026 年の値は、
  最後のサイクルの記録（cycles.json の per_rkey）と一致することを確かめる（ずれたら止める）。
- 達成の判定は、この早見表の基準（◎ の単勝の回収率 100% 超、消した馬の 3 着以内の率 5% 以下）で数え直す。
  サイクルのゴール（勝率 30%・単回収 120%）とは別。
計算は手元だけ（費用 0 円）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.graded import catalog as CAT
from src.research.graded import cycles as CY

ROI_OK = 1.0        # ◎ の単勝の回収率がこれを超えたら達成
KILL_OK = 0.05      # 消した馬の 3 着以内の率がこれ以下なら達成
SMALL_W = 0.05      # 効き（|重み| × 点のばらつき）が、一番効く項のこの割合未満なら「ほぼ効かない」と添える
TOL = 1e-5          # サイクルの記録との照合の許容


def scales(t: CY.Table) -> dict[str, float]:
    """特徴ごとの、レースの中での点のばらつき（レースの平均を引いた標準偏差）。特徴の尺度が違う（前日版の点は型の和、khit は件数、
    x は縮めた点）ので、効きの大きさ = |重み| × これ。◎ も消しもレースの中で比べるので、レースで全馬同じ点（馬場状態など）は効かない。"""
    cnt = np.diff(np.r_[t.starts, len(t.F)])
    mean = np.add.reduceat(t.F, t.starts, axis=0) / cnt[:, None]
    return {n: float(np.std(t.F[:, i] - mean[t.race_of, i])) for i, n in enumerate(t.names)}


def _eff(terms: dict[str, float], sc: dict[str, float] | None) -> dict[str, float]:
    return {k: abs(w) * (sc.get(k, 1.0) if sc else 1.0) for k, w in terms.items()}


def describe(terms: dict[str, float], khit_names: list[str] | None = None, kind: str = "honmei", sc: dict[str, float] | None = None) -> dict:
    """決め方の重みを日本語の文にする（kind = "honmei" は点の高い馬を ◎、"kill" は点の低い馬から消す）。{"filter": 絞り込みの文 or None, "terms": [{"text", "w", "small"}], "tie": 同点の決め方}。
    x_型 は、その型の中で馬が当たった水準の（縮めた）点。水準そのものの条件ではない。"""
    filt, items = None, []
    up, down = ("高い", "低い")
    end = "ほど上" if kind == "honmei" else "ほど消す"
    if kind == "kill":
        up, down = down, up    # 重みが正の項は、点が低いほど消す
    ef = _eff(terms, sc)
    top = max([v for k, v in ef.items() if k not in ("top3_pre", "top5_pre")] or [0.0])
    for k, w in sorted(terms.items(), key=lambda kv: -ef[kv[0]]):
        if k in ("top3_pre", "top5_pre") and w >= 50:
            filt = f"前日版の点の上位 {k[3]} 頭の中から選ぶ" if kind == "honmei" else f"前日版の点の上位 {k[3]} 頭より外の馬から先に消す"
            continue
        if k.startswith("x_"):
            tid = k[2:]
            t = CAT.BY_ID.get(tid)
            name = f"「{t.name}」" if t else f"型 {tid}"
            txt = f"{name}の点（{tid}）が{up if w > 0 else down}{end}"
        elif k == "score_pre":
            txt = f"前日版の点が{up if w > 0 else down}{end}"
        elif k == "score_day":
            txt = f"当日版の点が{up if w > 0 else down}{end}"
        elif k.startswith("khit_"):
            j = int(k[5:]) - 1
            nm = khit_names[j] if khit_names and j < len(khit_names) else k
            txt = f"消しの規則（{nm}）に当たった数が{('多い' if w > 0 else '少ない') if kind == 'honmei' else ('少ない' if w > 0 else '多い')}{end}"
        else:
            txt = f"{k} が{up if w > 0 else down}{end}"
        items.append({"text": txt, "w": float(w), "eff": float(ef[k]), "small": bool(top > 0 and ef[k] < SMALL_W * top), "key": k})
    tie = ("同点は前日版の点の高い馬 → 馬番の小さい馬を上にする" if kind == "honmei"
           else "同点は前日版の点の低い馬 → 馬番の大きい馬から消す")
    if "score_pre" not in terms:
        tie = "どの決め方にも前日版の点 × 0.001 を足してある（点の差がごく小さいときは前日版の点で決まる）。" + tie
    return {"filter": filt, "terms": items, "tie": tie}


def _rec(fin: np.ndarray) -> list[int]:
    f = np.asarray(fin, float)
    return [int((f == 1).sum()), int((f == 2).sum()), int((f == 3).sum()), int((f >= 4).sum())]


def honmei_stats(t: CY.Table, hrow: np.ndarray, races: np.ndarray) -> dict:
    """◎ の行（レースごと）のうち、races（レースの番号）の成績。勝率は全レース、回収率は払戻の解けたレースだけ（Best.report と同じ）。"""
    if len(races) == 0:
        return {"races": 0}
    h = hrow[races]
    pay = t.race["pay_ok"].to_numpy()[races]
    npay = int(pay.sum())
    return {"races": int(len(races)), "rec": _rec(t.r["fin"].to_numpy()[h]), "win": float(t.y1[h].mean()), "place": float(t.y3[h].mean()),
            "roi_t": float(t.rt[h][pay].sum() / npay) if npay else None, "roi_f": float(t.rf[h][pay].sum() / npay) if npay else None,
            "pay_races": npay}


def killed_rows(t: CY.Table, w: np.ndarray, k: int, i: int) -> np.ndarray:
    """レース i で消す行（cycles.killed と同じ: 点の低い順に k 頭、3 頭は残す、同点は並びの後ろから）。"""
    a = t.starts[i]
    b = t.starts[i + 1] if i + 1 < len(t.starts) else len(t.r)
    kk = max(0, min(k, (b - a) - 3))
    sc = t.F[a:b] @ w
    return (b - a - 1) - np.argsort(sc[::-1], kind="stable")[:kk] + a


def kill_stats(t: CY.Table, w: np.ndarray, k: int, races: np.ndarray) -> dict:
    if len(races) == 0:
        return {"races": 0}
    rows = np.concatenate([killed_rows(t, w, k, i) for i in races]).astype(int)
    fin = t.r["fin"].to_numpy()[rows]
    n = len(rows)
    return {"races": int(len(races)), "killed": int(n), "per_race": n / len(races), "rec": _rec(fin),
            "top3_rate": float((fin <= 3).mean()) if n else None, "precision": float((fin >= 4).mean()) if n else None}


def _val(r, c):
    v = r.get(c) if hasattr(r, "get") else None
    return None if v is None or (isinstance(v, float) and v != v) or v is pd.NA else v


def horse(t: CY.Table, h: int, pay_ok: bool) -> dict:
    """1 頭の実際の値（結果を見せるためだけ。オッズ・人気は決め方に使わない）。odds・pop は確定の単勝オッズ・人気（t.r の odds_final・pop_final。
    無ければ None）。tan・fuku は 100 円あたりの払戻（円）。払戻の解けないレースは None。"""
    r = t.r.iloc[int(h)]
    odds, pop, name = _val(r, "odds_final"), _val(r, "pop_final"), _val(r, "horse_name")
    return {"hn": int(r["hn"]), "name": None if name is None else str(name), "odds": None if odds is None else float(odds),
            "pop": None if pop is None else int(pop), "fin": float(r["fin"]),
            "tan": round(float(t.rt[h]) * 100) if pay_ok else None, "fuku": round(float(t.rf[h]) * 100) if pay_ok else None}


def recount(races: list[dict], who: str, kind: str) -> dict:
    """回ごとの実際の馬（card["races"]）から成績を数え直す（who = "A"・"B"・"base"、kind = "honmei"・"kill"）。カードの成績と同じ定義:
    ◎ の回収率は払戻の解けた回だけで割る。消しは消した全頭の着。"""
    if kind == "honmei":
        hs = [(x["base"] if who == "base" else x[who]["honmei"], x["pay_ok"]) for x in races]
        hs = [(h, p) for h, p in hs if h is not None]
        fin = [h["fin"] for h, _ in hs]
        paid = [h for h, p in hs if p]
        return {"races": len(hs), "rec": _rec(np.array(fin, float)) if hs else None,
                "roi_t": sum(h["tan"] for h in paid) / 100 / len(paid) if paid else None,
                "roi_f": sum(h["fuku"] for h in paid) / 100 / len(paid) if paid else None}
    ks = [h for x in races if x[who]["kill"] is not None for h in x[who]["kill"]]
    return {"killed": len(ks), "rec": _rec(np.array([h["fin"] for h in ks], float))}   # 0 頭でも kill_stats と同じ [0, 0, 0, 0]


def check_races(card: dict) -> None:
    """回ごとの実際の馬から数え直した成績が、カードの成績（2025・2026 年）と一致することを確かめる（ずれたら止める）。"""
    pairs = [("base", "honmei", card["base"]["eval"])] + [(m, kd, card[m][kd]["eval"]) for m in ("A", "B") for kd in ("honmei", "kill") if card[m][kd]]
    for who, kd, e in pairs:
        g = recount(card["races"], who, kd)
        if g["rec"] != e.get("rec"):
            raise ValueError(f"{card['rkey']}・{who}・{kd}: 実際の馬の着 {g['rec']} とカードの成績 {e.get('rec')} が違う")
        if kd == "kill":
            if g["killed"] != e["killed"]:
                raise ValueError(f"{card['rkey']}・{who}: 消した頭数が違う")
            continue
        for c in ("roi_t", "roi_f"):
            if (g[c] is None) != (e.get(c) is None) or (g[c] is not None and abs(g[c] - e[c]) > 1e-3):
                raise ValueError(f"{card['rkey']}・{who}・{c}: 実際の馬から数えた {g[c]} とカードの成績 {e.get(c)} が違う")


def race_head(t: CY.Table, i: int, dt=None) -> dict:
    """回の見出し: 日付・会場・頭数（t.r の field_size。無ければ表の頭数）・勝ち馬（同着は全頭）。"""
    a = t.starts[i]
    b = t.starts[i + 1] if i + 1 < len(t.starts) else len(t.r)
    pay = bool(t.race["pay_ok"].iloc[i])
    fs = _val(t.r.iloc[a], "field_size")
    return {"race_id": str(t.race["race_id"].iloc[i]), "date": str(dt[i])[:10] if dt is not None else "", "venue": str(_val(t.r.iloc[a], "venue") or ""),
            "field": int(fs) if fs is not None else int(b - a), "pay_ok": pay,
            "winners": [horse(t, h, pay) for h in range(a, b) if t.r["fin"].iloc[h] == 1]}


def level_of(levels: dict, row: int, tid: str) -> tuple[str, str] | None:
    return levels.get((int(row), tid))


def _check(what: str, got, want) -> None:
    if want is None or got is None:
        if want != got:
            raise ValueError(f"{what}: 当て直し {got} とサイクルの記録 {want} が違う")
        return
    if abs(float(got) - float(want)) > TOL:
        raise ValueError(f"{what}: 当て直し {got} とサイクルの記録 {want} が違う")


def global_stats(t: CY.Table, rules: dict, last_report: dict, khit_names: list[str] | None = None, sc: dict | None = None) -> dict:
    """全重賞に同じ決め方を使う版（A・B）の、2025・2026 年の値（サイクルの記録と照合する）と決め方の文。"""
    ev = np.flatnonzero(((t.race["year"] >= CY.EVAL[0]) & (t.race["year"] <= CY.EVAL[1])).to_numpy())
    out = {}
    for m in ("A", "B"):
        rh, rkl = rules[m]["honmei_global"], rules[m]["kill_global"]
        H = CY.honmei(t, CY._vec(t, rh["terms"])[:, None])[:, 0]
        h = honmei_stats(t, H, ev)
        k = kill_stats(t, CY._vec(t, rkl["terms"]), rkl["k"], ev)
        vh, vk = last_report[m]["honmei_global"]["val"], last_report[m]["kill_global"]["val"]
        _check(f"{m}・全体で 1 つの ◎ の勝率", h["win"], vh["win"])
        _check(f"{m}・全体で 1 つの ◎ の単回収", h["roi_t"], vh["roi_t"])
        _check(f"{m}・全体で 1 つの消しの的中率", k["precision"], vk["precision"])
        _check(f"{m}・全体で 1 つの消した頭数", k["killed"], vk["killed"])
        out[m] = {"honmei": {"rule": describe(rh["terms"], khit_names, "honmei", sc), "eval": h},
                  "kill": {"rule": describe(rkl["terms"], khit_names, "kill", sc), "k": int(rkl["k"]), "eval": k}}
    return out


def build_cards(t: CY.Table, rules: dict, last_report: dict, khit_names: list[str] | None = None, levels: dict | None = None,
                base_label: str = "前日版の点") -> dict:
    """重賞ごとのカードの値。rules = cycles_rules.json["rules"]、last_report = cycles.json の最後のサイクルの report。
    levels = {(row, 型): (水準, 層)}（例の馬の水準を書くため。無ければ書かない）。"""
    levels = levels or {}
    sc = scales(t)
    yr = t.race["year"].to_numpy()
    ev = (yr >= CY.EVAL[0]) & (yr <= CY.EVAL[1])
    dv = (yr >= CY.DEV[0]) & (yr <= CY.DEV[1])
    rk = t.race["rkey"].to_numpy()
    rkeys = sorted(set(rk[ev]))
    # ◎: A・B の重賞ごとの規則と基準（前日版の点）を列にして 1 回で当てる
    cols, idx = [], {}
    for m in ("A", "B"):
        for key, r in rules[m]["honmei_by_rkey"].items():
            idx[(m, key)] = len(cols)
            cols.append(CY._vec(t, r["terms"]))
    idx[("base", None)] = len(cols)
    cols.append(CY._vec(t, {"score_pre": 1}))
    H = CY.honmei(t, np.stack(cols, axis=1))
    per = {m: {"h": {x["rkey"]: x for x in last_report[m]["honmei_by_rkey"]["per_rkey"]},
               "k": {x["rkey"]: x for x in last_report[m]["kill_by_rkey"]["per_rkey"]}} for m in ("A", "B")}
    dt = t.r.groupby(t.race_of)["dt"].first().to_numpy() if "dt" in t.r.columns else None
    cards = []
    for key in rkeys:
        ri_e, ri_d = np.flatnonzero(ev & (rk == key)), np.flatnonzero(dv & (rk == key))
        last_i = ri_e[-1]
        rr = t.r.iloc[t.starts[last_i]]
        card = {"rkey": key, "grade": str(t.race["grade"].iloc[last_i]), "venue": str(rr.get("venue", "")), "course": str(rr.get("course", "")),
                "eval_dates": [str(dt[i])[:10] for i in ri_e] if dt is not None else [], "eval_races": int(len(ri_e)), "dev_races": int(len(ri_d))}
        b = idx[("base", None)]
        card["base"] = {"eval": honmei_stats(t, H[:, b], ri_e), "dev": honmei_stats(t, H[:, b], ri_d), "label": base_label}
        for m in ("A", "B"):
            r = rules[m]["honmei_by_rkey"].get(key)
            if key not in per[m]["h"] or key not in per[m]["k"]:
                raise ValueError(f"{m}・{key}: サイクルの記録に無い重賞")
            ph, pk = per[m]["h"][key], per[m]["k"][key]
            hc = None
            if r is not None:
                j = idx[(m, key)]
                e = honmei_stats(t, H[:, j], ri_e)
                _check(f"{m}・{key} の ◎ の勝率", e["win"], ph["win"])
                _check(f"{m}・{key} の ◎ の単回収", e["roi_t"], ph["roi_t"])
                ex = []
                ef = _eff(r["terms"], sc)
                tops = [x for x in sorted(r["terms"], key=lambda q: -ef[q]) if x.startswith("x_")][:3]
                for i in ri_e:
                    h = int(H[i, j])
                    lv = [{"type": q[2:], "name": CAT.BY_ID[q[2:]].name if q[2:] in CAT.BY_ID else q[2:], "level": v[0], "layer": v[1]}
                          for q in tops if (v := level_of(levels, t.r["row"].iloc[h], q[2:]))]
                    ex.append({"date": str(dt[i])[:10] if dt is not None else "", "hn": int(t.r["hn"].iloc[h]), "fin": float(t.r["fin"].iloc[h]),
                               "tan": float(t.rt[h]) * 100 if t.rt[h] > 0 else None, "pay_ok": bool(t.race["pay_ok"].iloc[i]), "levels": lv})
                same = int((H[ri_e, j] == H[ri_e, b]).sum())   # 前日版の点の ◎ と同じ馬だった回
                hc = {"rule": describe(r["terms"], khit_names, "honmei", sc), "label": r["label"], "eval": e | {"same_as_base": same}, "dev": honmei_stats(t, H[:, j], ri_d),
                      "default": bool(ph.get("default", False)), "examples": ex}
            kc = None
            kr = rules[m]["kill_by_rkey"].get(key)
            if kr is not None:
                w = CY._vec(t, kr["terms"])
                e = kill_stats(t, w, kr["k"], ri_e)
                _check(f"{m}・{key} の消しの的中率", e.get("precision"), pk["precision"])
                _check(f"{m}・{key} の消す頭数", kr["k"], pk["k"])
                kc = {"rule": describe(kr["terms"], khit_names, "kill", sc), "label": kr["label"], "k": int(kr["k"]), "eval": e,
                      "dev": kill_stats(t, w, kr["k"], ri_d), "default": bool(pk.get("default", False))}
            card[m] = {"honmei": hc, "kill": kc}
        races = []
        for i in ri_e:
            pay = bool(t.race["pay_ok"].iloc[i])

            def hs(h, i=i, pay=pay):
                if t.race_of[int(h)] != i:   # 選んだ行がこの回の馬であること
                    raise ValueError(f"{key}: 回 {i} の馬ではない行 {h}")
                return horse(t, h, pay)
            x = race_head(t, i, dt) | {"base": hs(H[i, b])}
            for m in ("A", "B"):
                r, kr = rules[m]["honmei_by_rkey"].get(key), rules[m]["kill_by_rkey"].get(key)
                x[m] = {"honmei": hs(H[i, idx[(m, key)]]) if r is not None else None,
                        "kill": [hs(h) for h in killed_rows(t, CY._vec(t, kr["terms"]), kr["k"], i)] if kr is not None else None}
            races.append(x)
        card["races"] = races
        check_races(card)
        card["flags"] = flags(card)
        cards.append(card)
    res = {"cards": cards, "summary": summary(cards)}
    if all(k in rules[m] for m in ("A", "B") for k in ("honmei_global", "kill_global")):
        res["global"] = global_stats(t, rules, last_report, khit_names, sc)
    return res


def _roi_ok(h: dict | None) -> bool:
    return bool(h and h["eval"].get("roi_t") is not None and h["eval"]["roi_t"] > ROI_OK)


def _kill_ok(k: dict | None) -> bool:
    return bool(k and k["eval"].get("top3_rate") is not None and k["eval"]["top3_rate"] <= KILL_OK)


def flags(card: dict) -> dict:
    """達成の印。B の既定（2018〜2024 年に回が無く、最初の候補が残っただけ）は B の達成に数えない。"""
    f = {}
    for m in ("A", "B"):
        h, k = card[m]["honmei"], card[m]["kill"]
        f[f"{m}_honmei"] = _roi_ok(h) and not (m == "B" and h["default"])
        f[f"{m}_kill"] = _kill_ok(k) and not (m == "B" and k["default"])
        f[f"{m}_default"] = bool((h and h["default"]) or (k and k["default"]))
    f["B_both"] = f["B_honmei"] and f["B_kill"]
    f["base_honmei"] = _roi_ok(card["base"])
    h = card["A"]["honmei"]   # A の決め方の 2018〜2024 年（A は同点を 2018〜2024 年の成績で決めるので、これも当てはめ）
    f["A_dev_honmei"] = bool(h and h["dev"].get("roi_t") is not None and h["dev"]["roi_t"] > ROI_OK)
    return f


def summary(cards: list[dict]) -> dict:
    s = {"n": len(cards)}
    for k in ("A_honmei", "B_honmei", "A_kill", "B_kill", "B_both", "base_honmei", "A_dev_honmei"):
        s[k] = sum(c["flags"][k] for c in cards)
    s["B_both_rkeys"] = [c["rkey"] for c in cards if c["flags"]["B_both"]]
    s["eval_races_1"] = sum(c["eval_races"] == 1 for c in cards)
    s["B_honmei_same_as_base"] = sum(c["flags"]["B_honmei"] and c["B"]["honmei"]["eval"]["same_as_base"] == c["eval_races"] for c in cards)
    s["A_dev_n"] = sum(bool(c["A"]["honmei"] and c["A"]["honmei"]["dev"].get("races")) for c in cards)
    return s
