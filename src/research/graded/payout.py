"""重賞のルールベース予想家: 馬ごとの単勝・複勝の払戻（回収率を「評価の物差し」にだけ使うための表）。

入力: 手元の race_result_race（各年）の payoff 列（JSON。単勝・複勝ほか）。2016〜2026 年の全レースを解く
（data/local/modeling/betting/payoff.parquet は 2020 年からしか無いため）。GCS は読まない。
出力: race_id・hn（馬番）・tan・fuku（100 円あたりの払戻を 1 で割った倍率。外れは 0）。

払戻・オッズは切り口・点の計算に使わない（catalog.FORBIDDEN_EXACT で検査する）。集計の物差し（回収率）にだけ使う。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.graded.features import TABLES, YEARS

# 払戻の頭打ち（大穴 1 頭に回収率が引っ張られないかを見る版）。単勝 50 倍・複勝 10 倍で打ち切る
CAP_TAN, CAP_FUKU = 50.0, 10.0


def parse_win_place(raw) -> tuple[dict[int, float], dict[int, float]] | None:
    """payoff の JSON → ({馬番: 単勝の倍率}, {馬番: 複勝の倍率})。同着は複数の馬に付く。解けなければ None。"""
    try:
        js = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    out: list[dict[int, float]] = []
    for jp in ("単勝", "複勝"):
        v = js.get(jp)
        m: dict[int, float] = {}
        if v is None:
            return None
        for e in (v if isinstance(v, list) else [v]):
            try:
                nums = [int(x) for x in str(e["numbers"]).replace(" ", "").split("-") if x != ""]
                pay = float(str(e["payout"]).replace(",", "").replace("円", ""))
            except (KeyError, ValueError, TypeError):
                return None
            if len(nums) != 1 or not 1 <= nums[0] <= 18:
                # 同着の単勝が「1315」「1,480150」のように 2 頭分つながった表記がある（2024 年に 3 レースなど）。
                # 馬番と払戻を確かに分けられないので、そのレースは払戻が解けなかったものとして回収率から外す
                return None
            m[nums[0]] = m.get(nums[0], 0.0) + pay / 100.0
        out.append(m)
    # 単勝の当たりは複勝の当たりに必ず入る。入らないのは、同着の 2 頭がつながった表記が 1〜18 に収まった場合
    # （例: 2025 年 202505050408 の「15」「200410」= 1 番 200 円と 5 番 410 円）。分けられないので解けない扱いにする
    if not set(out[0]) <= set(out[1]):
        return None
    return out[0], out[1]


def load_win_place(tables: Path = TABLES, years=YEARS) -> pd.DataFrame:
    """払戻のある馬（単勝・複勝の当たり）だけの表 race_id・hn・tan・fuku と、解けたレースの一覧（parsed=True の行で持つ）。"""
    rows, ok_races = [], []
    for y in years:
        f = tables / str(y) / "race_result_race.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f, columns=["race_id", "payoff"])
        for rid, raw in zip(r["race_id"].astype(str), r["payoff"]):
            p = parse_win_place(raw)
            if p is None:
                continue
            ok_races.append(rid)
            tan, fuku = p
            for h in set(tan) | set(fuku):
                rows.append((rid, h, tan.get(h, 0.0), fuku.get(h, 0.0)))
    hits = pd.DataFrame(rows, columns=["race_id", "hn", "tan", "fuku"])
    return hits, pd.Index(sorted(set(ok_races)), name="race_id")


def attach_returns(d: pd.DataFrame, hits: pd.DataFrame, ok_races: pd.Index) -> pd.DataFrame:
    """d（race_id・hn の行）に ret_tan・ret_fuku・ret_tan_c・ret_fuku_c（倍率。外れ 0）と pay_ok（払戻が解けたレース）を付ける。"""
    k = d[["race_id", "hn"]].copy()
    k["race_id"] = k["race_id"].astype(str)
    k["hn"] = pd.to_numeric(k["hn"], errors="coerce")
    h = hits.copy()
    h["hn"] = h["hn"].astype(float)
    m = k.merge(h, on=["race_id", "hn"], how="left")
    out = d.copy()
    out["ret_tan"] = m["tan"].fillna(0.0).to_numpy()
    out["ret_fuku"] = m["fuku"].fillna(0.0).to_numpy()
    out["ret_tan_c"] = out["ret_tan"].clip(upper=CAP_TAN)
    out["ret_fuku_c"] = out["ret_fuku"].clip(upper=CAP_FUKU)
    out["pay_ok"] = k["race_id"].isin(set(ok_races)).to_numpy()
    return out
