"""調教師 × 最終週の調教パターンの成績を、出走馬ごとに当てはめて見せる分析。

各馬について、レース週の平日のベスト 4F の追い切り（final_week_workout.pick_best_weekday）から『場所|パターン』を決め、
その馬の調教師の同じパターンの成績（2020〜2025、単勝 50 倍未満。表は src/scripts/data/trainer_pattern_table.py が作る凍結 JSON）を引く。
調教師 → 場所 → パターンの 3 段の成績を返す（色分けのオッズ分布つき）。

- 表は 2020〜2025 の結果を含むので、開催日が 2026-01-01 以降のレースにだけ使う（それ以前は未来の結果を含むため出さない）。
- 調教の記録はホストの data/local/horse_training だけにある。無い環境では出さない（not_applicable）。表は GCS を読まない。
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

from src.research.race_analysis.base import ModuleOutput, RaceContext

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
TABLE_PATH = REPO / "data/meta/trainer_pattern/trainer_pattern_2020_2025.json"
FIRST_DATE = "2026-01-01"   # 表が使った期間（〜2025）より後のレースにだけ当てはめる


@functools.lru_cache(maxsize=1)
def _table_cached(sig: tuple) -> dict | None:
    try:
        return json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_table() -> dict | None:
    try:
        st = TABLE_PATH.stat()
    except OSError:
        return None
    return _table_cached((st.st_mtime_ns, st.st_size))


def _training_path(horse_id: str) -> Path | None:
    from src.scraper.horse_training_local import horse_training_json_path

    try:
        return horse_training_json_path(REPO, horse_id)
    except ValueError:
        return None


def training_sig(entries: list[dict[str, Any]]) -> list[list[Any]]:
    """各馬の調教ファイルの更新（fingerprint 用）。調教が入る・変わると作り直す。stat だけで軽い。"""
    out: list[list[Any]] = []
    for e in entries:
        p = _training_path(str(e.get("horse_id") or ""))
        try:
            st = p.stat() if p else None
        except OSError:
            st = None
        out.append([e.get("horse_id"), st.st_mtime_ns if st else None])
    return out


def patterns_of(horse_ids: list[str], race_date: str) -> dict[str, str | None]:
    """horse_id → 『場所|パターン』（レース週の平日のベスト 4F。調教が無い馬は None）。レース日より前の調教だけ。"""
    import pandas as pd

    from src.pipeline.features import final_week_workout as W
    from src.pipeline.features import tp35_lap_groups as L
    from src.pipeline.features import training_laps as tl

    rows: list[dict[str, Any]] = []
    for hid in horse_ids:
        p = _training_path(hid)
        try:
            rec = json.loads(p.read_text(encoding="utf-8")) if p and p.is_file() else None
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict):
            rows += [r for r in tl.record_to_rows(dict(rec, horse_id=rec.get("horse_id") or hid)) if r["race_date"] == race_date]
    out: dict[str, str | None] = {h: None for h in horse_ids}
    if not rows:
        return out
    laps = tl.add_lap_features(pd.DataFrame(rows))
    bw = W.pick_best_weekday(L.prepare_laps(laps))
    for r in bw.itertuples(index=False):
        out[str(r.horse_id)] = f"{r.BW_place}|{r.BW_pat}"
    return out


def _sum_rows(rows: list[list[Any]]) -> list[Any]:
    """行 [n, 頭数, 1着, 2着, 3着, 単勝払戻計, 複勝払戻計, 区間別 5 本] を足し合わせる（頭数は足せないので None）。"""
    n = [sum(r[i] for r in rows) for i in (0, 2, 3, 4)]
    w, p = round(sum(r[5] for r in rows), 2), round(sum(r[6] for r in rows), 2)
    nb = len(rows[0][7])
    bins = [[round(sum(r[7 + k][j] for r in rows), 1) if k >= 3 else sum(r[7 + k][j] for r in rows) for j in range(nb)] for k in range(5)]
    return [n[0], None, n[1], n[2], n[3], w, p, *bins]


def levels_for(table: dict, trainer_id: str, pat: str | None) -> dict[str, Any] | None:
    """調教師・場所・パターンの 3 段の行。調教師が表に無ければ None。pat が None（調教なし）のときは調教師の段だけ。"""
    t = (table.get("trainers") or {}).get(str(trainer_id))
    if not t:
        return None
    allrows = list(t["p"].items())
    out: dict[str, Any] = {"trainer": _sum_rows([r for _, r in allrows]) if allrows else None, "place": None, "pattern": None}
    if pat:
        place = pat.split("|", 1)[0]
        out["place"] = _sum_rows([r for k, r in allrows if k.split("|", 1)[0] == place]) if any(k.split("|", 1)[0] == place for k, _ in allrows) else None
        out["pattern"] = t["p"].get(pat)
    return out


class TrainerPatternModule:
    id = "trainer_pattern"
    label = "調教師×調教"
    description = ("各馬のレース週のベスト追い切り（4F）のパターンと、その馬の調教師が同じパターンで出した過去の成績（2020〜2025、単勝50倍未満）。"
                   "調教師全体・同じ調教場所・同じパターンの 3 段で、着順内訳・勝率・複勝率・回収率と、オッズ帯別の分布（行にカーソルを合わせると詳細）を見られます。")

    def applicable(self, ctx: RaceContext) -> bool:
        if not ctx.date or ctx.date < FIRST_DATE or not ctx.entries or load_table() is None:
            return False
        return any(p is not None and p.is_file() for p in (_training_path(str(e.get("horse_id") or "")) for e in ctx.entries if e.get("horse_id")))

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        table = load_table() or {}
        ids = [str(e["horse_id"]) for e in ctx.entries if e.get("horse_id")]
        pats = patterns_of(ids, ctx.date or "")
        horses = []
        for e in ctx.entries:
            hid = str(e.get("horse_id") or "")
            pat = pats.get(hid)
            tid = str(e.get("trainer_id") or "")
            lv = levels_for(table, tid, pat) if tid else None
            horses.append({"key": e["key"], "horse_number": e.get("horse_number"), "horse_name": e.get("horse_name"), "trainer_id": tid or None,
                           "trainer_name": e.get("trainer_name"), "pat": pat, "levels": lv})
        figure = {"edges": table.get("edges"), "max_odds": table.get("max_odds"), "y0": table.get("y0"), "y1": table.get("y1"), "horses": horses}
        notes = ["回収率は 100 円ずつ買ったときの払戻の割合（100% が収支ゼロ）。単勝 50 倍以上の出走は集計から除いています。",
                 "パターンは、レース週の月〜金の追い切りのうち 4F が最も速い 1 本（坂路はラップ種別 A1〜X、ウッド・ポリは 4F の時計帯）。調教の記録が無い馬・出走が少ない組み合わせは参考程度です。"]
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=f"{table.get('y1')}-12-31", view="trainer_pattern", figure=figure)
