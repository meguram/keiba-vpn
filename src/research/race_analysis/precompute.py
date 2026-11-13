"""分析結果の事前計算。出馬表・結果が更新された時点で計算して保存し、画面（API）は保存済みの結果を読むだけにする。

再計算の要否は fingerprint（入力の指紋）で決める: 出走馬と枠・騎手の構成、レース条件、集計に使う結果データの最終日（そのレースの開催日より前）、
種牡馬の対応表、分析ロジックの版。どれも変わらなければ書き直さない。過去のレースの指紋は変わらないので、一度計算すれば再計算されない。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.research.race_analysis import runs as runs_mod
from src.research.race_analysis import store, trainer_pattern
from src.research.race_analysis.service import _load_card, _normalize_entries, analyze_race

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "39"      # 39: 調教師×調教の各馬に調教の状況（集計の対象外の追い切り・未取得・記録なし）と最後の追い切りを付ける（2026-11-13）。38: 過去位置取りの脚質を位置取り帯と 1 対 1 の 6 段階（ハナ＋前から 20% ごと）にし、成績も同じ区分で。位置取り帯・帯の走数の列を外す（2026-11-12）。37: 走法マップの各軸に、クラス基準（5=2勝で安定・4=1勝・3=未勝利・G1 上位=10）の 10 点満点 scores を付ける（2026-11-11）。36: 騎手・枠・種牡馬×コースから連対率の列を外す（2026-11-11）。35: 騎手・枠・種牡馬×コースに単勝・複勝回収率を足し（50倍以上は除く）、過去位置取りに位置取り帯別の成績と直近5走の位置取りの幅を足す（2026-11-11）。34: 調教師×調教の説明文から記号（A1〜X）を外す（2026-11-11）。33: 調教師×最終週の調教パターンの成績を各馬に当てはめる分析を追加（2026-11-11）。32: 馬場特注ゾーンを、前のレースが少ない日（開催初日など）でも描けるよう格子の判定の下限・平滑化の幅・縮みを見直す（2026-11-09）。31: 血統の傾向から「よく走る距離」を外し、出脚・先行力を「前に行く度」に統合（2026-11-06）。30: 血統の傾向を、ダート適性・適性距離（m）・10点満点の走りの特性で見せる（2026-11-06）。29: 走法マップの各馬に、性別・父・母父だけから出す血統の傾向（10 観点の期待値と 80% 区間）を付ける（2026-11-06）。28: 走法マップの Best5 を、配分を調整した 1 パターンだけにする（2026-11-06）。27: 走法マップに Best5（2026-11-06）。26: 走法マップに道悪の地図（稍重以上・道悪適性入りの 6 軸）を足し、良馬場の地図は良馬場だけで測る（2026-11-05）。25: 走法マップに基礎スピードを足して 6 軸に・TS/TS持続力へ改名（2026-11-05）。24: 走法マップの軸に Hペース耐性を足して 5 軸に（2026-11-05）。23: 走法マップの各馬に着順を付ける（レース後に結果が入ったら作り直す。2026-11-05）。22: 走法マップの軸をギアチェンジ・TS・TS持続力・Cr性能の4つに固定（2026-11-05）。21: 過去位置取りの表示を直近5走・前走の見出し。20: 今日の馬場を前日まで・直前ほど重く。19: 走法マップを上位 N 方面（3〜6）の軸の無い地図に（2026-11-03）。分析ロジック・列定義を変えたら上げる（保存済みの結果が作り直される）


def _asof_cache() -> dict[str, str | None]:
    return {}


def day_results(storage: Any, race_id: str, race_date: Any = None) -> int:
    """同じ日・同じ場のこのレースより前と、同じ開催の前日で、結果（確定または速報）が入ったレースの数。
    走法マップの「今日の馬場」はそれらの結果で変わるので指紋に入れる。"""
    from src.research.race.running_map_day import window

    n = 0
    for rid, _, _ in window(race_id, race_date):      # 開催日が来ていない日のレースは問い合わせない
        for cat in ("race_result", "race_result_on_time"):
            try:
                card = storage.load(cat, rid)
            except Exception:
                card = None
            if card and card.get("entries"):
                n += 1
                break
    return n


def own_finish(storage: Any, race_id: str) -> list[list[Any]]:
    """このレース自身の結果（確定または速報）の着順。走法マップの着順マークに使うので指紋に入れる（結果が入る・確定で変わると作り直す）。"""
    from src.research.race_analysis.running_map_module import finish_positions

    for cat in ("race_result", "race_result_on_time"):
        try:
            card = storage.load(cat, race_id)
        except Exception:
            card = None
        fin = finish_positions(card)
        if fin:
            return sorted([h, p] for h, p in fin.items())
    return []


def fingerprint(card: dict, entries: list[dict[str, Any]], asof: dict[str, str | None], day: int = 0, finish: Any = None) -> str:
    date = card.get("date")
    if date not in asof:
        runs = runs_mod.load_runs()
        before = runs[runs["date"] < date] if date else runs
        asof[date] = before["date"].max() if len(before) else None
    try:
        st = runs_mod.SIRE_MAP_PATH.stat()
        sire_sig = [st.st_mtime_ns, st.st_size]
    except OSError:
        sire_sig = None
    # 走法の軸（run_axes）は日次の更新（update_run_axes）で全件が作り直される（全期間の統計に依存）。変わった日だけ窓の分析を作り直す
    try:
        from src.research.race.running_profile import AXES_PATH

        st = AXES_PATH.stat()
        axes_sig = [st.st_mtime_ns, st.st_size]
    except (OSError, ImportError):
        axes_sig = None
    doc = {
        "v": ANALYSIS_VERSION,
        "race": [card.get(k) for k in ("venue", "surface", "distance", "date", "track_condition")],
        "entries": [[e.get("horse_id"), e.get("horse_number"), e.get("bracket_number"), e.get("jockey_id")] for e in entries],
        "data_until": asof[date],
        "sire_map": sire_sig,
        "run_axes": axes_sig,
        "day_results": day,
        "finish": finish or [],
        "train": trainer_pattern.training_sig(entries) if (date or "") >= trainer_pattern.FIRST_DATE else None,
    }
    return hashlib.sha1(json.dumps(doc, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def degraded_reasons(payload: dict[str, Any]) -> list[str]:
    """計算結果に欠けがあれば理由を返す（分析コードの古さ・モデルの欠け・モジュールの失敗）。補助の計算は例外を握りつぶして地図だけ出すので、ここで拾う。"""
    out: list[str] = []
    for m in payload.get("modules") or []:
        if m.get("status") == "error":
            out.append(f"module:{m.get('id')}")
        out += [f"{m.get('id')}:{e}" for e in ((m.get("figure") or {}).get("build_errors") or [])]
    return out


def precompute_race(storage: Any, race_id: str, *, force: bool = False, asof: dict[str, str | None] | None = None) -> str:
    """1 レースぶんを計算して保存する。'written' / 'unchanged' / 'no_entries' / 'degraded' を返す。

    'degraded' = 補助の計算（Best5・血統など）が欠けた。保存済みの正常な結果があればそれを残し（欠けた結果で上書きしない）、
    無ければ欠けた結果を指紋なしで保存して次の周回でやり直す。"""
    asof = asof if asof is not None else _asof_cache()
    card, rows = _load_card(storage, race_id)
    entries, _ = _normalize_entries(rows)
    if not entries:
        return "no_entries"
    fp = fingerprint(card, entries, asof, day_results(storage, race_id, card.get("date")), own_finish(storage, race_id))
    old = store.read(race_id)
    if old and old.get("fingerprint") == fp and not force:
        return "unchanged"
    payload = analyze_race(storage, race_id)
    if payload.get("status") != "ok":
        return "no_entries"
    bad = degraded_reasons(payload)
    if bad:
        logger.error("分析の事前計算が欠けた結果になりました（ホストの分析コード・モデルを確認）: %s %s", race_id, bad)
        if old and not old.get("degraded"):
            return "degraded"
        payload["degraded"] = bad
        payload["fingerprint"] = "degraded"
    else:
        payload["fingerprint"] = fp
    payload["version"] = ANALYSIS_VERSION
    payload["computed_at"] = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S+09:00")   # ホストは UTC。画面は JST で見せる
    store.write(race_id, payload)
    return "degraded" if bad else "written"


def precompute_races(storage: Any, race_ids: list[str], *, force: bool = False) -> dict[str, int]:
    asof = _asof_cache()
    out = {"written": 0, "unchanged": 0, "no_entries": 0, "degraded": 0, "error": 0}
    for rid in race_ids:
        try:
            out[precompute_race(storage, rid, force=force, asof=asof)] += 1
        except Exception:
            logger.exception("分析の事前計算に失敗: %s", rid)
            out["error"] += 1
    return out


def refresh_for_races(storage: Any, race_ids: list[str], *, force: bool = False, extra_horse_ids: set[str] | None = None) -> dict[str, int]:
    """出馬表が更新されたレースの分析を作り直す。先に、その出走馬の父の対応表（差分のみ）を更新する。"""
    from src.scripts.data import build_horse_sire_map as sire

    ids: set[str] = set(extra_horse_ids or ())
    for rid in race_ids:
        try:
            _, rows = _load_card(storage, rid)
        except Exception:
            continue
        ids |= {str(r["horse_id"]) for r in rows if r.get("horse_id")}
    if ids:
        try:
            sire.build(horse_ids=ids, netkeiba_lines=sire.line_labels(fetch_max=10))      # 系統は smartrc の小系統（新しい種牡馬は最大 10 頭まで取得）
        except Exception:
            logger.exception("父の対応表の更新に失敗（分析は対応表の現状で続ける）")
    return precompute_races(storage, race_ids, force=force)
