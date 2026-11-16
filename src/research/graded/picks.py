"""めぐ分析の「データで予想」タブの元データ（重賞ごとの該当馬の記録と成績）。

- 出すのは、2018〜2024 年の中だけで選んで凍結した B の規則（src/research/graded/prereg_cycles.json の rules.B）と、
  比べる基準の前日版の ◎（前日版の点が最大の馬）だけ。A（2025・2026 年に合わせて選んだ規則）は、2025・2026 年の結果を使って
  選んでいるので出さない（キーも持たない）。
- 規則は凍結のハッシュ（sha256_rules）と照合し、違えば止める。点の元の集計は、各レースの前日までの as-of の値（tally._asof）。
- 確定の単勝オッズ・人気・馬名は結果を見せるためだけに使い、選び方には使わない。
- 出力: data/local/graded_picks/history.json（ホストの手元。git には入れない）。めぐ分析の事前計算
  （src/research/race_analysis/graded_picks.py）がこれを読み、レースごとの分析の JSON に、その重賞の分だけを入れる。

stats・period_of は標準ライブラリだけで動く（めぐ分析のモジュールからも使う）。表を作る build_history は重い。
"""

from __future__ import annotations

from typing import Any

DEV = (2018, 2024)        # B の規則を選んだ期間（当てはめ）
EVAL = (2025, 2026)       # 選んだ後に当てはめた期間（ただし候補の作り方を決める過程で何度か見ている）
DAYOF_TYPES = frozenset({"G06", "G07", "H07", "H10", "I09", "X03"})   # 当日に決まる値を使う型（catalog の day_of。前日には計算できない）
WHO = ("B", "base")       # 出す決め方（A は出さない）

NOTES = [
    "条件は 2024 年までのデータだけで決め、2025 年以降はその条件で選んだ結果です（2025・2026 年の結果を使って選び直していません）。",
    "ただし 2025・2026 年の重賞は、規則の候補の作り方を決める過程で何度か見ているので、まっさらな検証ではありません。",
    "2024 年までの成績は、条件を決めた期間そのものなので良く見えます。比べて読めるのは 2025 年以降の成績です。",
    "1 重賞あたり 2025・2026 年は 1〜2 回しかなく、単勝の回収率 100% 超は 1 回の高配当でも起きます。",
    "単勝オッズ・人気は確定の値で、結果を見せるためだけに使っています（◎ や消しの選び方には使っていません）。",
    "点は、型（前走の着順・騎手など）ごとに、その馬が当たった水準の過去の成績（そのレースの前日まで）を縮めた値です。",
]


def period_of(date: str, track_from: str | None = None) -> str:
    """回の期間: "dev"（2018〜2024 年、当てはめ）・"eval"（2025 年〜凍結の前）・"forward"（凍結の後。前向きの記録）。"""
    y = int(str(date)[:4])
    if y <= DEV[1]:
        return "dev"
    if track_from and str(date)[:10] >= track_from:
        return "forward"
    return "eval"


def _rec(fins: list[float]) -> list[int]:
    return [sum(f == 1 for f in fins), sum(f == 2 for f in fins), sum(f == 3 for f in fins), sum(f >= 4 for f in fins)]


def stats(races: list[dict], who: str, kind: str) -> dict[str, Any] | None:
    """回の記録から成績を数える（who = "B"・"base"、kind = "honmei"・"kill"）。早見表（playbook.honmei_stats・kill_stats）と同じ定義:
    勝率・複勝率は全回、回収率は払戻の解けた回だけで割る。消しは消した全頭の着。回が無ければ None。"""
    if kind == "honmei":
        hs = [(x["base"] if who == "base" else x[who]["honmei"], x["pay_ok"]) for x in races]
        hs = [(h, p) for h, p in hs if h is not None]
        if not hs:
            return None
        fins = [h["fin"] for h, _ in hs]
        paid = [h for h, p in hs if p]
        return {"races": len(hs), "rec": _rec(fins), "win": sum(f == 1 for f in fins) / len(fins), "place": sum(f <= 3 for f in fins) / len(fins),
                "roi_t": sum(h["tan"] for h in paid) / 100 / len(paid) if paid else None,
                "roi_f": sum(h["fuku"] for h in paid) / 100 / len(paid) if paid else None, "pay_races": len(paid)}
    ks = [(x, x[who]["kill"]) for x in races if x[who]["kill"] is not None]
    if not ks:
        return None
    fins = [h["fin"] for _, k in ks for h in k]
    return {"races": len(ks), "killed": len(fins), "rec": _rec(fins), "top3_rate": (sum(f <= 3 for f in fins) / len(fins)) if fins else None}


def needs_dayof(terms: dict) -> bool:
    """規則が当日に決まる値（馬体重・馬場など）を使うか。使うなら前日には ◎・消しを決められない。"""
    return any(k == "score_day" or (k.startswith("x_") and k[2:] in DAYOF_TYPES) for k in terms)


def build_history(t, prereg: dict, last_report: dict, khit_names: list[str] | None = None) -> dict:
    """表 t（cycles.Table。t.r に horse_name・odds_final・pop_final・field_size を足したもの）に、凍結した B の規則と前日版の ◎ を当てはめ、
    重賞ごとの回の記録を作る。prereg = prereg_cycles.json（ハッシュは呼ぶ側で照合する）、last_report = cycles.json の最後のサイクルの
    report（B の既定の印だけに使う）。2025・2026 年の B の値は、サイクルの記録と一致することを確かめる（ずれたら止める）。"""
    import numpy as np

    from src.research.graded import cycles as CY
    from src.research.graded import playbook as PB

    RB = prereg["rules"]["B"]
    sc = PB.scales(t)
    yr = t.race["year"].to_numpy()
    rk = t.race["rkey"].to_numpy()
    keys = sorted(RB["honmei_by_rkey"])
    cols = [CY._vec(t, RB["honmei_by_rkey"][k]["terms"]) for k in keys] + [CY._vec(t, {"score_pre": 1})]
    H = CY.honmei(t, np.stack(cols, axis=1))
    jb = len(keys)
    dt = t.r.groupby(t.race_of)["dt"].first().to_numpy()
    per_h = {x["rkey"]: x for x in last_report["B"]["honmei_by_rkey"]["per_rkey"]}
    per_k = {x["rkey"]: x for x in last_report["B"]["kill_by_rkey"]["per_rkey"]}
    out = {}
    for j, key in enumerate(keys):
        rh, rkl = RB["honmei_by_rkey"][key], RB["kill_by_rkey"].get(key)
        if key not in per_h or (rkl is not None and key not in per_k):
            raise ValueError(f"{key}: サイクルの記録に無い重賞")
        ri = np.flatnonzero((rk == key) & (yr >= DEV[0]))
        if len(ri) == 0:
            continue
        races = []
        wk = CY._vec(t, rkl["terms"]) if rkl is not None else None
        for i in ri:
            pay = bool(t.race["pay_ok"].iloc[i])

            def hs(h, i=i, pay=pay):
                if t.race_of[int(h)] != i:
                    raise ValueError(f"{key}: 回 {i} の馬ではない行 {h}")
                return PB.horse(t, h, pay)
            x = PB.race_head(t, i, dt) | {"base": hs(H[i, jb]),
                                          "B": {"honmei": hs(H[i, j]),
                                                "kill": [hs(h) for h in PB.killed_rows(t, wk, rkl["k"], i)] if wk is not None else None}}
            races.append(x)
        # 2025・2026 年の B の値を、サイクルの記録と照合する（評価はやり直さない）
        ev = [x for x in races if period_of(x["date"], prereg.get("track_from")) == "eval"]   # 凍結の後の回（前向きの記録）はサイクルの記録に無い
        sh = stats(ev, "B", "honmei")
        if sh is not None:
            PB._check(f"B・{key} の ◎ の勝率", sh["win"], per_h[key]["win"])
            PB._check(f"B・{key} の ◎ の単回収", sh["roi_t"] if sh["roi_t"] is not None else None, per_h[key]["roi_t"])
        if rkl is not None:
            sk = stats(ev, "B", "kill")
            if sk is not None and sk["killed"]:
                PB._check(f"B・{key} の消しの的中率", 1 - sk["top3_rate"], per_k[key]["precision"])
        last = t.r.iloc[t.starts[ri[-1]]]
        out[key] = {"rkey": key, "grade": str(t.race["grade"].iloc[ri[-1]]), "venue": str(last.get("venue", "")), "course": str(last.get("course", "")),
                    "honmei": {"rule": PB.describe(rh["terms"], khit_names, "honmei", sc), "label": rh["label"], "default": bool(per_h[key].get("default", False)),
                               "dayof": needs_dayof(rh["terms"])},
                    "kill": ({"rule": PB.describe(rkl["terms"], khit_names, "kill", sc), "label": rkl["label"], "k": int(rkl["k"]),
                              "default": bool(per_k[key].get("default", False)), "dayof": needs_dayof(rkl["terms"])} if rkl is not None else None),
                    "races": races}
    return {"rkeys": out, "frozen_at": prereg["frozen_at"], "sha256_rules": prereg["sha256_rules"], "track_from": prereg["track_from"],
            "periods": {"dev": list(DEV), "eval": list(EVAL)}, "notes": NOTES}
