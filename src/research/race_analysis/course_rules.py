"""めぐ分析の「好走ルール（通常）」: 全レース版の凍結の第 3 版（049da502）の条件を、このレースの単位（コース × クラスの群 × 良馬場）について出す。

- 出すもの: 単位の買い（好走）の条件と消しの条件を 1 つずつ、読める文で。条件ごとに、訓練 2016〜2024 年・2025 年・2026 年の
  [1-2-3-着外]・単勝と複勝の回収率・件数と、同じ人気の馬の値（比べる相手）。
- 数字は data/meta/course/rule_stats_v3.json（src.research.course.v3 rule_stats が、凍結した条件を当て直して数えた値）から読むだけ。
  手で写さない。条件は選び直さない（凍結の sha を照合する）。
- どの出走馬が条件に当てはまるかは、まだ出さない（出馬表に当てはめる計算は別の段で足す）。
- 正直な表示: これまでの確かめでは、条件の ◎ はオッズに織り込み済みで、回収率は保証しない。2025・2026 年は何度か見た期間。
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.research.race_analysis.base import ModuleOutput, RaceContext

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
STATS = REPO / "data/meta/course/rule_stats_v3.json"
PREREG3 = REPO / "src/research/course/prereg_course_v3.json"

CLS = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 5, "G3": 6, "G2": 7, "G1": 8}   # folklore_build.CLS と同じ
GROUPS = (("新馬・未勝利", 0, 1), ("1勝・2勝", 2, 3), ("3勝・OP", 4, 5))                                   # course.run.GROUPS と同じ
VENUE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
# 内回り・外回り（芝）。folklore_build.IO_RULE と同じ表。京都芝 1400・1600・新潟芝 2000 は両方あるので race_keys.resolve_io の決まりで決める
IO_RULE = {
    ("06", 1200): "外", ("06", 1600): "外", ("06", 2200): "外", ("06", 1800): "内", ("06", 2000): "内", ("06", 2500): "内", ("06", 3600): "内",
    ("08", 1200): "内", ("08", 1800): "外", ("08", 2000): "内", ("08", 2200): "外", ("08", 2400): "外", ("08", 3000): "外", ("08", 3200): "外",
    ("09", 1200): "内", ("09", 1400): "内", ("09", 1600): "外", ("09", 1800): "外", ("09", 2000): "内", ("09", 2200): "内", ("09", 2400): "外",
    ("09", 2600): "外", ("09", 3000): "内", ("09", 3200): "外",
    ("04", 1000): "直", ("04", 1200): "内", ("04", 1400): "内", ("04", 1600): "外", ("04", 1800): "外", ("04", 2200): "内", ("04", 2400): "内",
}
AMBIG_IO = {("08", 1400), ("08", 1600), ("04", 2000)}

NOTES = [
    "参考です。回収率は保証しません。これらの条件を組み合わせて ◎ を選び、◎ の発走前の単勝オッズが 5 倍以下のときだけ ◎ の複勝を買う検証では、2026 年（1/1〜10/12、302 レース中 161 レースで購入）の回収率が 81.3%（同じ時の 1 番人気の複勝 87.4%）で、オッズに織り込み済みでした。事前登録の確かめでも、オッズを超える条件は 0 でした。下の表は条件を 1 つずつ当てた値で、その検証の数字とは別物です。",
    "条件は、2016〜2024 年の良馬場のレースだけで選び、2024 年末で凍結しました。2025・2026 年は、その条件を当てた結果です（何度か見た期間なので、まっさらな確かめではありません）。",
    "各条件の「同じ人気の馬」は、条件に当たった馬と同じ人気の馬の値です。条件の値がこれを上回らなければ、人気で選ぶのと変わりません。",
    "良馬場の条件です。当日の馬場が良でなければ、当てはまりません。",
    "着外は 4 着以下です。抽出の期間（2016〜24）は条件を選んだ期間そのものなので、良く見えます（消しの条件は、定義上その期間に 3 着以内が 0 です）。",
]


@lru_cache(maxsize=2)
def _load(sig: tuple) -> dict | None:
    try:
        S = json.loads(STATS.read_text(encoding="utf-8"))
        P = json.loads(PREREG3.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if S.get("sha256_cells") != P.get("sha256_cells"):
        logger.warning("条件ごとの成績が凍結の版と違うので使いません")
        return None
    return S


def load_stats() -> dict | None:
    try:
        st = STATS.stat()
    except OSError:
        return None
    return _load((st.st_mtime_ns, st.st_size))


_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")


def grade_from(grade: str | None, race_class: str | None) -> str | None:
    """出馬表の grade。特別レースは grade が空で race_class（「サラ系３歳以上 ２勝クラス」）にだけクラスがあるので、そこから読む。"""
    g = str(grade or "").strip()
    if g:
        return g
    rc = str(race_class or "").translate(_Z2H)
    for k, pat in (("新馬", "新馬"), ("未勝利", "未勝利"), ("1勝", "1勝"), ("2勝", "2勝"), ("3勝", "3勝"), ("OP", "オープン")):
        if pat in rc:
            return k
    return None


def group_of(grade: str | None) -> tuple[str | None, int | None]:
    """出馬表の grade（「新馬」「未勝利」「1勝」…）→ (単位の群, cls)。重賞・不明は (None, cls)。新馬は群 4 の分け方で「新馬」。"""
    c = CLS.get(str(grade or "").strip())
    if c is None:
        return None, None
    if c == 0:
        return "新馬", c
    for name, lo, hi in GROUPS:
        if lo <= c <= hi:
            return name, c
    return None, c


def unit_of(ctx: RaceContext, grade: str | None, course_type: str | None = None) -> str | None:
    """このレースの単位の鍵（例: 「京都芝1800外|1勝・2勝」）。frame の course（race_keys.course_key）と同じ作り。"""
    grp, cls = group_of(grade)
    if grp is None or not ctx.venue or not ctx.surface or not ctx.distance:
        return None
    surf = "ダ" if str(ctx.surface).startswith("ダ") else ("芝" if str(ctx.surface).startswith("芝") else None)
    if surf is None:
        return None
    vc = VENUE_CODE.get(str(ctx.venue))
    dist = int(ctx.distance)
    io = ""
    if surf == "芝" and vc:
        io = IO_RULE.get((vc, dist)) or ""
        if (vc, dist) in AMBIG_IO:
            if cls is not None and cls >= 5:
                # 学習の表では course_type がほぼ欠けていて「外」になっている。出馬表は標準のコースでも空文字なので、空文字は「外」と読む
                io = "内" if str(course_type or "").strip() == "内" else "外"
            else:
                io = "内外不明"
    return f"{ctx.venue}{surf}{dist}{io}|{grp}"


def figure_for(S: dict, unit: str, track: str | None, date: str | None = None) -> dict[str, Any]:
    R, V = S["rules"][unit], S["values"]

    def per(kind: str, i: int) -> dict:
        out = {}
        for p in ("train", "2025", "2026"):
            xs = ((V.get(p) or {}).get(unit) or {}).get(kind) or []
            out[p] = xs[i] if i < len(xs) else None
        return out
    rules = {kind: [r | {"stats": per(kind, i)} for i, r in enumerate(R[kind])] for kind in ("buy", "kill")}
    unit_stats = {p: ((V.get(p) or {}).get(unit) or {}).get("unit") for p in ("train", "2025", "2026")}
    return {"unit": unit, "layer": R.get("layer"), "n_buy": R.get("n_buy"), "n_kill": R.get("n_kill"), "n_buy_hit": R.get("n_buy_hit"),
            "rules": rules, "unit_stats": unit_stats,
            "periods": S.get("periods"), "data_until": S.get("data_until"), "cycle": S.get("cycle"),
            "track": track, "track_ok": track in (None, "", "良"),
            # 振り返りのとき: このレースの日が結果の最終日以前なら、2025・2026 年の値にこのレース自身とそれより後の結果が入る
            "includes_after": bool(date and S.get("data_until") and str(date)[:10] <= str(S["data_until"]) and str(date)[:4] >= "2025"),
            "unit_label": unit.replace("内外不明", "（内回り・外回りの区別なし）")}


class CourseRulesModule:
    id = "course_rules"
    label = "過去データの条件（通常）"
    description = "重賞以外。2016〜2024 年の良馬場のデータで決めた、コース × クラスの群ごとの、3 着内が多かった・少なかった条件と、その成績（参考）。"

    def applicable(self, ctx: RaceContext) -> bool:
        try:
            from src.research.race_analysis.graded_picks import rkey_of
            if ctx.race_name and ctx.date and rkey_of(ctx.race_name, ctx.date):
                return False          # 重賞（出馬表の grade が「OP」のままの G1 もある）は「好走データ予想」のタブで出す
            if str(ctx.grade or "") in ("G1", "G2", "G3") or "障害" in f"{ctx.race_class or ''}{ctx.race_name or ''}":
                return False          # 重賞・障害は単位の外（frame の単位は平地の新馬〜OP だけ）
            S = load_stats()
            u = unit_of(ctx, grade_from(ctx.grade, ctx.race_class), ctx.course_type)
            return bool(S and u and u in S["rules"])
        except Exception:   # 失敗しても、ほかのタブの事前計算を止めない
            logger.exception("好走ルール（通常）の判定に失敗: %s", ctx.race_id)
            return False

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        S = load_stats() or {}
        u = unit_of(ctx, grade_from(ctx.grade, ctx.race_class), ctx.course_type)
        fig = figure_for(S, u, ctx.track_condition, ctx.date) if (S and u in S.get("rules", {})) else None
        return ModuleOutput(columns=[], groups=[], cells={}, notes=list(NOTES), data_until=S.get("data_until"), view="course_rules", figure=fig)
