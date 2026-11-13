"""参照用データを作った/更新したことを、VPS の事前ロード係(``keiba-cache-warm``)へ伝える目印を GCS に置く。

- 置き場: ``ops/cache_dirty/<YYYYMMDD(UTC)>/<キー>``（キーは race_id 12桁 / 開催日8桁 / ``h``+馬ID10桁 / ``_ALL``=全体）。中身は空に近い JSON。
- VPS 側は日付ごとの接頭辞を list するだけ（個別の読み出しはしない）ので GCP の read 課金はほぼ出ない。
- 書くのは「最善努力」: 失敗しても保存や配備を止めない。影モード(``KEIBA_FETCH_SHADOW=1``)・``GCS_BUCKET`` 未設定・
  開発機(``KEIBA_CACHE_MARKER=0``)では何もしない。同じキーは 60 秒に 1 回まで。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PREFIX = "ops/cache_dirty"
ALL_KEY = "_ALL"
_KEY_RE = re.compile(r"^(?:\d{12}|\d{8})$")
_HORSE_KEY_RE = re.compile(r"^h\d{10}$")      # 馬のページ（/api/horse/{id}/detail など）の取り直し用
# 馬のページの元になるカテゴリ（キーが馬 ID 10 桁の保存）
HORSE_CATEGORIES = frozenset({"horse_result", "horse_profile", "horse_race_history", "horse_pedigree_5gen"})
_DEDUPE_SEC = 60.0
_MAX_RACE_MARKS = 300      # 1 回の取り込みで置く race_id の目印の上限（超えたら開催日の目印だけにする。過去分の一括アップロード対策）
_last: dict[str, float] = {}
_lock = threading.Lock()
_io: list = []

# 画面の応答の元になるミラー対象（リポジトリルートからの相対パスの先頭）
_UI_PATH_PREFIXES = (
    "data/local/inference/", "data/local/race_analysis/", "data/calculated_data/tracking_difficulty/",
    "data/local/meta/entries_state/", "data/page_reference/race_lists/", "data/page_reference/race_day_schedule/",
)
_RACE_STEM = re.compile(r"^(\d{12})(?!\d)")
_DATE_SEG = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")


def enabled() -> bool:
    if os.environ.get("KEIBA_CACHE_MARKER", "").strip() == "0":
        return False
    if os.environ.get("KEIBA_FETCH_SHADOW", "").strip() == "1":
        return False
    return bool(os.environ.get("GCS_BUCKET", "").strip())


def marker_key(key: str, now: datetime | None = None) -> str:
    day = (now or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{PREFIX}/{day}/{key}"


def _accept(key: str) -> bool:
    if key == ALL_KEY or _HORSE_KEY_RE.match(key or ""):
        return True      # 馬 ID の先頭は生まれ年なので、年での間引きはしない
    if not _KEY_RE.match(key or ""):
        return False
    return int(key[:4]) >= datetime.now().year - 1      # 過去分の補填で目印が溢れないように


def _put(name: str, body: bytes) -> None:
    if not _io:
        from src.ops.gcs_io import default_io

        _io.append(default_io())      # クライアントは作り直さず使い回す
    _io[0].put_bytes(name, body, content_type="application/json")


def mark(key: str, reason: str = "") -> bool:
    """目印を 1 つ置く。置いたら True（無効・重複・失敗は False）。例外は出さない。"""
    try:
        if not enabled() or not _accept(key):
            return False
        now = time.monotonic()
        with _lock:
            if now - _last.get(key, -1e9) < _DEDUPE_SEC:
                return False
            _last[key] = now
        body = json.dumps({"key": key, "reason": reason[:80], "at": time.time()}, ensure_ascii=False).encode("utf-8")
        _put(marker_key(key), body)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("キャッシュ目印を置けません(%s): %s", key, exc)
        return False


# VPS の温め役（scripts/ops/vps/cache_warm.py）は、毎分この合図のメタデータ（世代）だけを見て、変わったときだけ目印を一覧する
# （GCP の外から毎分一覧すると月 約 265 円かかっていた。2026-11-13）。合図は push と配備のときだけ書く（保存ごとに書くと
# 最悪 月 13 万回・約 100 円になる）。保存ごとの目印は、温め役の安全網（30 分おきの一覧）で拾う。
BEACON_KEY = f"{PREFIX}/_beacon.json"


def touch_beacon(reason: str = "") -> bool:
    """合図を上書きする（1 プロセスで 60 秒に 1 回まで）。間引いた分は『書き残し』として覚え、次の呼び出しで書く
    （同じプロセスで 60 秒以内に続いた push・配備の目印が、安全網の 30 分まで温まらないのを防ぐ）。失敗しても例外は出さない。"""
    try:
        if not enabled():
            return False
        now = time.monotonic()
        with _lock:
            if now - _last.get("\0beacon", -1e9) < _DEDUPE_SEC:
                _last["\0beacon_pending"] = 1.0
                return False
            _last["\0beacon"] = now
            _last.pop("\0beacon_pending", None)
        _put(BEACON_KEY, json.dumps({"at": time.time(), "reason": reason[:80]}, ensure_ascii=False).encode("utf-8"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("キャッシュの合図を置けません（VPS の温めが最大 30 分遅れる）: %s", exc)
        return False


def mark_from_save(category: str, key: str) -> bool:
    """``HybridStorage.save`` から呼ぶ。画面に関わるカテゴリの race_id / 開催日キー、馬のカテゴリの馬 ID だけ目印にする。"""
    from src.utils.cache_epoch import UI_CATEGORIES

    if category in HORSE_CATEGORIES and re.fullmatch(r"\d{10}", key or ""):
        return mark("h" + key, f"save:{category}")
    if category not in UI_CATEGORIES:
        return False
    return mark(key, f"save:{category}")


def mark_all(reason: str = "") -> bool:
    """配備・手動クリアなど『全体を取り直したい』ときの目印。置いたら合図も上書きする。"""
    ok = mark(ALL_KEY, reason)
    if ok:
        touch_beacon(f"all:{reason}")
    return ok


def keys_from_paths(rels: list[str] | tuple[str, ...]) -> tuple[list[str], list[str]]:
    """ミラーに上げたファイル（リポジトリルートからの相対パス）から (race_id のキー, 開催日のキー) を取り出す。画面に関係しない場所は無視。"""
    races: set[str] = set()
    dates: set[str] = set()
    for rel in rels:
        if not rel.startswith(_UI_PATH_PREFIXES):
            continue
        parts = rel.split("/")
        m = _RACE_STEM.match(parts[-1])
        if m:
            races.add(m.group(1))
        else:
            stem = parts[-1].split(".", 1)[0]      # race_lists/20261227.json のように、ファイル名が開催日のもの
            if re.fullmatch(r"\d{8}", stem):
                dates.add(stem)
        for seg in parts[:-1]:
            d = _DATE_SEG.match(seg)
            if d:
                dates.add("".join(d.groups()))
    return sorted(races), sorted(dates)


def mark_from_paths(rels: list[str] | tuple[str, ...]) -> int:
    """``gcs_mirror.push`` が上げたファイルから目印を置く。置いた数を返す。"""
    if not enabled():
        return 0
    races, dates = keys_from_paths(rels)
    if len(races) > _MAX_RACE_MARKS:
        races = []
    n = 0
    for k in races + dates:
        n += 1 if mark(k, "mirror") else 0
    if n or _last.get("\0beacon_pending"):
        touch_beacon("mirror")
    return n
