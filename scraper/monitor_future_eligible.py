"""
モニター (/monitor)・データビューア等用の開催日フィルタ。

netkeiba トップのレース一覧では、重賞・L・OP・特別などに Icon_GradeType が付き、
未勝利・1勝クラス等には付かない。未来で「開催未定なのに仮の一覧だけ載る」日を抑え、
特別登録・重賞等が一覧に現れた日だけ未来欄に出す。

データビューアはさらに、開催なし・JRA 0件の日をプルダウンから除外する。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from scraper.missing_races import is_jra_race_id

# 古い race_lists（list_grade_icon 無し）向け。Icon 付きレース名に寄せた保守的パターン。
_NAME_GRADE_HINT = re.compile(
    r"[(（]\s*G[123]\s*[)）]|[(（]\s*L\s*[)）]|[(（]\s*OP\s*[)）]|"
    r"ステークス|ダービー|"
    r"記念|"
    r"[SsＳｓ][（(]|"  # バレンタインS( など
    r"特別\s*[（(]|"
    r".+(賞|杯)$"
)


def jst_today() -> date:
    return datetime.now(ZoneInfo("Asia/Tokyo")).date()


def _compact_to_date(s: str) -> date | None:
    if not s or len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _is_placeholder_race_title(name: str | None) -> bool:
    """パース失敗等でレース名が「3R」「11R」だけの行（実質データなし）。"""
    if not name or not isinstance(name, str):
        return True
    return bool(re.fullmatch(r"\d{1,2}R", name.strip()))


def race_has_list_grade_signal(race: dict[str, Any]) -> bool:
    """トップ一覧相当の「掲載グレード」があるか（Icon またはレース名フォールバック）。"""
    name = race.get("race_name") or ""
    if _is_placeholder_race_title(name):
        return False
    if race.get("list_grade_icon"):
        return True
    if not isinstance(name, str):
        return False
    return bool(_NAME_GRADE_HINT.search(name))


def include_date_in_monitor_summary(
    date_compact: str,
    raw_races: list[dict[str, Any]] | None,
    meta: dict[str, Any] | None,
) -> bool:
    """
    モニターの日付一覧・欠損ヒートマップに含めるか。

    開催なし・JRA0件の日は過去未来を問わず除外（空の race_lists プレースホルダ対策）。
    未来はさらに、重賞級が一覧に載った日だけ。
    過去・当日は JRA が1件でもあれば対象（通常カード）。
    """
    meta = meta or {}
    if meta.get("note") == "no_race_scheduled":
        return False

    races = raw_races or []
    jra = [
        r for r in races
        if r.get("race_id") and is_jra_race_id(str(r["race_id"]))
    ]
    if not jra:
        return False

    rd = _compact_to_date(date_compact)
    if rd is None:
        return True
    if rd > jst_today():
        graded = any(race_has_list_grade_signal(r) for r in jra)
        # 中央の通常開催は土日が中心。未来の平日は重賞級が載っている日だけ（祝日・特例も拾う）。
        if rd.weekday() < 5 and not graded:
            return False
        return graded
    return True


def include_date_in_data_viewer_race_list(
    date_compact: str,
    raw_races: list[dict[str, Any]] | None,
    meta: dict[str, Any] | None,
) -> bool:
    """
    /data-viewer の開催日セレクト用。

    - no_race_scheduled または JRA レース 0 件の日は除外。
    - 未来はモニターと同様、重賞級が一覧に載った日だけ（race_has_list_grade_signal）。
    - 過去・当日は、極端に少ない／多い件数や「クラス戦だけの断片」を除外し、
      ある程度のカードでは重賞・OP・特別級の掲載が1件以上ある日だけとする。
    """
    meta = meta or {}
    if meta.get("note") == "no_race_scheduled":
        return False

    races = raw_races or []
    n_all = len(races)
    if n_all > 36:
        return False

    jra = [
        r for r in races
        if r.get("race_id") and is_jra_race_id(str(r["race_id"]))
    ]
    jra_named = [r for r in jra if not _is_placeholder_race_title(r.get("race_name"))]
    if not jra_named:
        return False

    any_jra_grade = any(race_has_list_grade_signal(r) for r in jra_named)
    if 0 < n_all < 4:
        return any_jra_grade

    rd = _compact_to_date(date_compact)
    if rd is None:
        return True
    if rd > jst_today():
        if rd.weekday() < 5 and not any_jra_grade:
            return False
        return any_jra_grade

    # 過去・当日: 本命の中央開催カードには通常1件以上は重賞・OP・特別が載る。
    # 未勝利だけの誤取得断片（同一条件で多頭数）はここで落とす。
    if len(jra_named) >= 4 and not any_jra_grade:
        return False
    return True
