"""めぐ分析の「データで予想」タブ（重賞だけ）。凍結した B の規則と前日版の ◎ で選んだ該当馬の、これまでの記録・成績と、このレースの該当馬。

- 元データはホストの data/local/graded_picks/history.json（python -m src.research.graded.run picks_history が作る。研究コード
  src/research/graded/picks.py）。B は 2018〜2024 年の中だけで選んで凍結した規則。A（2025・2026 年に合わせて選んだ規則）は出さない。
- リーク防止: このレースより前の回だけを「これまで」に入れ、成績もその回だけで数える（開催日が前の回）。このレース自身の結果は、
  結果が入った後に「このレース」として別に出す。
- 重賞の判定はレース名（src/research/graded/race_keys.canonical_key）が記録の重賞にあるか。出馬表の grade は当てにしない。
- 履歴が変わっても指紋（precompute.fingerprint）は変えない。履歴を作り直したら patch_saved()（python -m src.research.race_analysis.graded_picks）で、
  保存済みの分析のこのタブだけを入れ直す（出馬表・GCS は読まない）。
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.research.race_analysis.base import ModuleOutput, RaceContext

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
PREREG_CYCLES = REPO / "src/research/graded/prereg_cycles.json"
ODDS_KEYS = ("odds", "pop", "tan", "fuku")   # 実オッズ・人気・確定払戻（公開モードで actual_odds_public=false の間、開発者以外には出さない）


def _pk():
    """研究コード（src/research/graded/picks.py）は使うときに読む（読めなくても、ほかのタブの事前計算を止めない）。"""
    from src.research.graded import picks
    return picks


def picks_dir() -> Path:
    return Path(os.environ.get("GRADED_PICKS_DIR") or REPO / "data/local/graded_picks")


@functools.lru_cache(maxsize=2)
def _history_cached(path: str, sig: tuple) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.exception("重賞の該当馬の記録を読めません: %s", path)
        return None


@functools.lru_cache(maxsize=2)
def _frozen_cached(sig: tuple) -> tuple[str | None, str | None]:
    try:
        P = json.loads(PREREG_CYCLES.read_text(encoding="utf-8"))
        return P.get("sha256_rules"), P.get("frozen_at")
    except (OSError, ValueError):
        return None, None


def load_history() -> dict | None:
    """記録（history.json）。凍結した規則（prereg_cycles.json）のハッシュ・凍結の日時と違えば、古い規則の記録なので使わない（None）。"""
    p = picks_dir() / "history.json"
    try:
        st, sp = p.stat(), PREREG_CYCLES.stat()
    except OSError:
        return None
    h = _history_cached(str(p), (st.st_mtime_ns, st.st_size))
    if not h:
        return None
    sha, fa = _frozen_cached((sp.st_mtime_ns, sp.st_size))
    if sha is None or h.get("sha256_rules") != sha or h.get("frozen_at") != fa:
        logger.warning("重賞の該当馬の記録が凍結の規則と違うので使いません（python -m src.research.graded.run picks_history で作り直す）")
        return None
    return h


def rkey_of(race_name: str | None, date: str | None) -> str | None:
    """記録にある重賞なら、その鍵（そろえたレース名）。"""
    h = load_history()
    if not h or not race_name or not date:
        return None
    from src.research.graded.race_keys import canonical_key
    try:
        mo = int(str(date)[5:7])
        k = canonical_key(race_name, mo)
    except (TypeError, ValueError):
        return None
    a = (h.get("aliases") or {}).get(k)      # 出馬表の略称（「マイルCS」など）→ 結果の名前の鍵（同じ月に見た略称だけ）
    if isinstance(a, dict) and mo in (a.get("months") or []):
        k = a["to"]
    return k if k in h["rkeys"] else None


def _stats_by_period(races: list[dict], track_from: str | None) -> dict[str, Any]:
    PK = _pk()
    out: dict[str, Any] = {}
    for per in ("dev", "eval", "forward"):
        rs = [x for x in races if PK.period_of(x["date"], track_from) == per]
        if rs:
            out[per] = {"B_honmei": PK.stats(rs, "B", "honmei"), "B_kill": PK.stats(rs, "B", "kill"), "base_honmei": PK.stats(rs, "base", "honmei")}
    return out


def load_picked(race_id: str, h: dict) -> dict | None:
    """前日の計算の該当馬（<GRADED_PICKS_DIR>/races/<race_id>.json）。記録と同じ凍結の規則で作ったものだけ使う。"""
    p = picks_dir() / "races" / f"{race_id}.json"
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if x.get("sha256_rules") != h.get("sha256_rules") or x.get("frozen_at") != h.get("frozen_at"):
        logger.warning("前日の計算の該当馬が凍結の規則と違うので使いません: %s", race_id)
        return None
    from src.research.graded.upcoming import entries_sig
    cur = entries_sig(race_id)
    if cur is not None and x.get("entries_sig") != cur:   # 取消・乗り替わりの後の古い計算は出さない（計算し直すまで「確定後に表示」）
        logger.info("前日の計算の該当馬が今の出馬表と違うので使いません: %s", race_id)
        return None
    v = h["rkeys"].get(x.get("rkey")) or {}
    hd, kd = (v.get("honmei") or {}).get("default"), (v.get("kill") or {}).get("default")
    if hd or kd:
        sk = dict(x.get("skipped") or {})
        if hd:
            sk["honmei"] = "default"
        if kd:
            sk["kill"] = "default"
        x = x | {"B": {"honmei": None if hd else x["B"]["honmei"], "kill": None if kd else x["B"]["kill"]}, "skipped": sk}
    return x


def _without_default(x: dict, v: dict) -> dict:
    """規則が既定（2018〜2024 年に回が無く、決められていない）なら、その B の ◎・消しは出さない（None）。"""
    hd, kd = v["honmei"]["default"], bool(v["kill"] and v["kill"]["default"])
    if not (hd or kd):
        return x
    return x | {"B": {"honmei": None if hd else x["B"]["honmei"], "kill": None if kd else x["B"]["kill"]}}


def figure_for(h: dict, rkey: str, race_id: str, date: str) -> dict[str, Any]:
    """画面に渡す値。これまで = 開催日がこのレースより前の回（新しい順）。このレース = 記録にあれば結果つき。"""
    PK = _pk()
    v = h["rkeys"][rkey]
    tf = h.get("track_from")
    races = [_without_default(x, v) for x in v["races"]]
    past = [x for x in races if x["date"] < str(date)[:10] and x["race_id"] != race_id]
    this = next((x for x in races if x["race_id"] == race_id), None)
    picked = load_picked(race_id, h) if this is None else None
    if this is not None:
        status = "result"
    elif picked is not None:
        status = "picked"    # 出馬表の確定後の前日の計算（upcoming.py）。当日の値を使う規則は picked["skipped"] に理由
    elif v["honmei"]["default"]:
        status = "default"
    elif v["honmei"]["dayof"] or (v["kill"] or {}).get("dayof"):
        status = "dayof"     # 当日の値（馬体重・馬場など）を使う規則は、前日には該当馬を出さない
    else:
        status = "pending"   # 出馬表の確定前（確定後に前日の計算で出す）
    return {"rkey": rkey, "grade": v["grade"], "course": v["course"], "rules": {"honmei": v["honmei"], "kill": v["kill"]},
            "this": {"status": status, "race": this, "period": PK.period_of(date, tf), "picked": picked},
            "past": sorted(past, key=lambda x: x["date"], reverse=True), "stats": _stats_by_period(past, tf),
            "periods": h.get("periods"), "track_from": tf, "frozen_at": h.get("frozen_at"), "data_until": h.get("data_until")}


class GradedPicksModule:
    id = "graded_picks"
    label = "データで予想"
    description = ("重賞だけ。2018〜2024 年のデータだけで決めて凍結した ◎ と消しの選び方（B）と、比べる基準の前日版の ◎ で選んだ馬の、"
                   "これまでの回の該当馬・成績・回収率と、このレースの該当馬。")

    def applicable(self, ctx: RaceContext) -> bool:
        try:
            return bool(ctx.date and ctx.date[:4] >= "2018" and rkey_of(ctx.race_name, ctx.date))
        except Exception:   # 記録が壊れていても、ほかのタブの事前計算を止めない（欠けた結果の扱いにもしない）
            logger.exception("データで予想の判定に失敗: %s", ctx.race_id)
            return False

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        try:
            h = load_history() or {}
            rk = rkey_of(ctx.race_name, ctx.date)
            fig = figure_for(h, rk, ctx.race_id, ctx.date or "") if rk else None
        except Exception:   # 失敗しても status=error（欠けた結果）にしない。ほかのタブの保存を止めないため
            logger.exception("データで予想の計算に失敗: %s", ctx.race_id)
            h, fig = {}, None
        notes = list(h.get("notes") or []) if fig else ["このレースの該当馬の記録を読めませんでした。"]
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=h.get("data_until"), view="graded_picks", figure=fig)


def redact_actual_odds(payload: dict[str, Any]) -> dict[str, Any]:
    """「データで予想」タブの実オッズ・人気・確定払戻を落とした写し（公開モードで actual_odds_public=false の間、開発者以外に返す）。"""
    def drop(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: drop(v) for k, v in o.items() if k not in ODDS_KEYS}
        if isinstance(o, list):
            return [drop(v) for v in o]
        return o
    mods = [m | {"figure": drop(m.get("figure"))} if m.get("id") == GradedPicksModule.id else m for m in payload.get("modules") or []]
    return payload | {"modules": mods}


def module_meta(m: "GradedPicksModule", ctx: RaceContext) -> dict[str, Any]:
    """service.analyze_race と同じ形のモジュールの値（タブ 1 つ分）。"""
    meta: dict[str, Any] = {"id": m.id, "label": m.label, "description": m.description}
    if not m.applicable(ctx):
        return meta | {"status": "not_applicable"}
    out = m.compute(ctx)
    return meta | {"status": "ok", "columns": [], "groups": [], "notes": out.notes, "data_until": out.data_until, "source": out.source,
                   "key_header": out.key_header, "view": out.view, "figure": out.figure}


def patch_saved(race_ids: list[str] | None = None) -> dict[str, int]:
    """保存済みの分析（data/local/race_analysis/<race_id>.json）の「データで予想」タブだけを作り直す（ほかのタブ・指紋はそのまま）。
    履歴（history.json）を作り直した後に呼ぶ。出馬表・GCS は読まない（保存済みの分析のレース名・日付だけを使う）。
    race_ids が None なら、保存済みのすべての分析のうち重賞のもの。"""
    from src.research.race_analysis import store

    ids = race_ids if race_ids is not None else sorted(p.stem for p in store.analysis_dir().glob("*.json"))
    m = GradedPicksModule()
    out = {"patched": 0, "skipped": 0, "error": 0}
    for rid in ids:
        pl = store.read(rid)
        if not pl or pl.get("status") != "ok":
            out["skipped"] += 1
            continue
        r = pl.get("race") or {}
        if not rkey_of(r.get("race_name"), r.get("date")) and not any(x.get("id") == m.id for x in pl.get("modules") or []):
            out["skipped"] += 1
            continue
        try:
            ctx = RaceContext(race_id=rid, date=r.get("date"), venue=r.get("venue"), surface=r.get("surface"), distance=r.get("distance"),
                              entries=pl.get("entries") or [], race_name=r.get("race_name"))
            meta = module_meta(m, ctx)
        except Exception:
            logger.exception("データで予想の計算に失敗: %s", rid)
            out["error"] += 1
            continue
        mods = [x for x in pl.get("modules") or [] if x.get("id") != m.id] + [meta]
        pl["modules"] = mods
        store.write(rid, pl)
        out["patched"] += 1
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="保存済みのめぐ分析に「データで予想」タブを入れ直す（履歴の作り直しの後に流す）")
    ap.add_argument("--races", help="race_id のカンマ区切り（省略時は保存済みのすべての重賞）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(patch_saved([x for x in a.races.split(",") if x] if a.races else None))
