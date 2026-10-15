"""着順セルの正規化。元の文字列(raw)を捨てず、状態(status)と数値(position)に分けて持つ。

従来は 取/除/中/失 も、読み取れなかった文字列も、すべて finish_position=-1 にまとめていたため、
「正規の例外(競走中止など)」と「解析の失敗」を後から区別できなかった。

  finished        … 数字（着順）。注記が付く形（"5(降)" など）も先頭の数字を着順とする
  scratched       … 取（出走取消）
  excluded        … 除（競走除外）
  dnf             … 中（競走中止）
  disqualified    … 失（失格）
  unparsed        … 上のどれでもない（語彙に無い文字列・空欄）。**スクレイピング側で見直すべきデータ**

position は従来どおり、着順が付かないときは -1（下流の既存処理が `> 0` で絞っているため互換を保つ）。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

FINISHED = "finished"
UNPARSED = "unparsed"
STATUS_TOKENS: dict[str, str] = {"取": "scratched", "除": "excluded", "中": "dnf", "失": "disqualified"}
NON_FINISH_STATUSES = frozenset(STATUS_TOKENS.values())
KNOWN_STATUSES = frozenset({FINISHED, UNPARSED, *NON_FINISH_STATUSES})


def normalize_finish(raw: Any) -> tuple[int, str]:
    """(position, status)。position は着順が付かないとき -1。"""
    text = unicodedata.normalize("NFKC", "" if raw is None else str(raw)).strip()
    if re.fullmatch(r"\d{1,2}", text):
        return int(text), FINISHED
    if text in STATUS_TOKENS:
        return -1, STATUS_TOKENS[text]
    m = re.match(r"^(\d{1,2})\D", text)
    if m:
        return int(m.group(1)), FINISHED
    return -1, UNPARSED


def finish_anomalies(entries: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    """1 レース分の entries から、解析の失敗を疑う手がかりを返す（馬番のリスト）。

      unparsed        … 着順の文字列を解釈できなかった馬
      timed_unranked  … 着順が付いていないのにタイム(time_sec>0)がある馬（完走したのに着順を読み損ねた疑い）
      rank_gap        … 着順に欠番がある（最大着順 > 着順の付いた頭数。同着は頭数も同じ着順を持つので欠番にならない）
    """
    rows = list(entries)
    out: dict[str, list[int]] = {"unparsed": [], "timed_unranked": [], "rank_gap": []}
    ranked = [int(e.get("finish_position") or 0) for e in rows if int(e.get("finish_position") or 0) > 0]
    for e in rows:
        pos = int(e.get("finish_position") or 0)
        hn = int(e.get("horse_number") or 0)
        if e.get("finish_status") == UNPARSED:
            out["unparsed"].append(hn)
        if pos <= 0 and float(e.get("time_sec") or 0) > 0:
            out["timed_unranked"].append(hn)
    if ranked and max(ranked) > len(ranked):
        out["rank_gap"] = [max(ranked)]
    return out
