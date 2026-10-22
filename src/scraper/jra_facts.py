"""JRA 公式サイトの『事実データ（テキスト）』の取得・解析（設計と部品。既定は無効）。

対象はテキストの事実だけ（出馬表・結果・払戻・オッズの表）。画像・映像・音声・記事・PDF は取らない（JRA の『ご利用に際して』は、
映像・音声・画像・記事の二次利用に許諾を求める。事実データの個別の扱いは記載がなく、法的には未確定）。
実アクセスは KEIBA_JRA_PUBLIC_FETCH=1 のときだけ。承認が出るまで cron には載せない。低頻度・単一接続・キャッシュ。

HTML 構造は実ページで未検証。表の見出しの語から列を対応づける（表記の揺れに強い）方式にしてあり、読めない表は保存せず None を返す。
実ページとの突き合わせは、取得の承認後に行う。保存の出どころは ``_meta.source = "jra"``（src/utils/provenance.py）。
保存カテゴリ案: race_shutuba_jra / race_result_jra / race_odds_jra（CATEGORY_MAP とスキーマへの登録は承認後）。
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
import urllib.request
from typing import Any

from src.utils.provenance import stamp

ENV_FLAG = "KEIBA_JRA_PUBLIC_FETCH"
MIN_INTERVAL_SEC = float(os.environ.get("KEIBA_JRA_MIN_INTERVAL_SEC", "30"))   # 同一プロセスの最短間隔（単一接続・低頻度）
MAX_REQUESTS_PER_DAY = int(os.environ.get("KEIBA_JRA_MAX_REQUESTS_PER_DAY", "300"))
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain")           # 画像・映像・PDF などは取らない
USER_AGENT = "meguai-keiba/1.0 (+personal research; contact via site owner)"


class JraFetchDisabled(RuntimeError):
    pass


class JraFetchRefused(RuntimeError):
    pass


_lock = threading.Lock()
_last_at = 0.0
_day_count: dict[str, int] = {}


def fetch_enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip() == "1"


def fetch_text(url: str, *, opener=None, now=time.time, sleep=time.sleep, timeout: float = 20.0) -> str:
    """テキストの HTML だけを取る。無効・上限超過・テキスト以外は拒否。単一接続（ロック）・最短間隔つき。"""
    global _last_at
    if not fetch_enabled():
        raise JraFetchDisabled(f"{ENV_FLAG}=1 でないので取得しません（利用規約の確認と承認が済むまで無効）")
    if not re.match(r"^https://www\.jra\.go\.jp/", url):
        raise JraFetchRefused("JRA 公式（https://www.jra.go.jp/）以外は取りません")
    with _lock:
        day = time.strftime("%Y%m%d", time.localtime(now()))
        if _day_count.get(day, 0) >= MAX_REQUESTS_PER_DAY:
            raise JraFetchRefused("1 日の取得上限に達しました")
        wait = MIN_INTERVAL_SEC - (now() - _last_at)
        if wait > 0:
            sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        opener = opener or urllib.request.urlopen
        with opener(req, timeout=timeout) as r:
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_CONTENT_TYPES:
                raise JraFetchRefused(f"テキスト以外（{ctype}）は取りません")
            raw = r.read(2_000_000)
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", r.headers.get("Content-Type") or "")
            if m:
                charset = m.group(1)
        _last_at = now()
        _day_count[day] = _day_count.get(day, 0) + 1
    return raw.decode(charset, errors="replace")


# ── 表の見出し → 標準キー（表記の揺れに強くする）──

_HEADER_KEYS: dict[str, tuple[str, ...]] = {
    "bracket_number": ("枠", "枠番"),
    "horse_number": ("馬番", "番", "馬 番"),
    "horse_name": ("馬名", "馬"),
    "sex_age": ("性齢", "性別年齢", "性年齢"),
    "jockey_weight": ("斤量", "負担重量"),
    "jockey_name": ("騎手", "騎手名"),
    "trainer_name": ("調教師", "厩舎", "調教師名"),
    "weight": ("馬体重", "体重"),
    "finish_position": ("着順", "着"),
    "finish_time": ("タイム", "走破タイム"),
    "margin": ("着差",),
    "last_3f": ("上り", "上がり", "上がり3F", "上り3F"),
    "popularity": ("人気", "人気順"),
    "win_odds": ("単勝", "単勝オッズ", "オッズ"),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).strip()


def _num(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", unicodedata.normalize("NFKC", text or ""))
    return float(m.group()) if m else None


def _tables(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if len(rows) < 2:
            continue
        head = [_norm(c.get_text()) for c in rows[0].find_all(["th", "td"])]
        body = [[_norm(c.get_text()) for c in r.find_all(["th", "td"])] for r in rows[1:]]
        yield head, body


def _column_map(head: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, h in enumerate(head):
        for key, labels in _HEADER_KEYS.items():
            if key not in out and h in tuple(_norm(x) for x in labels):
                out[key] = i
    return out


def parse_table(html: str, required: tuple[str, ...]) -> list[dict[str, Any]] | None:
    """見出しから列を対応づけて行を返す。必須の列がそろう表が無ければ None（保存しない）。"""
    for head, body in _tables(html):
        cmap = _column_map(head)
        if not all(k in cmap for k in required):
            continue
        rows = []
        for cells in body:
            if len(cells) <= max(cmap.values()):
                continue
            row: dict[str, Any] = {}
            for key, i in cmap.items():
                v = cells[i]
                if key in ("horse_number", "bracket_number", "finish_position", "popularity"):
                    n = _num(v)
                    row[key] = int(n) if n is not None else (v or None)
                elif key in ("jockey_weight", "win_odds", "last_3f"):
                    row[key] = _num(v)
                elif key == "weight":
                    w = re.match(r"(\d+)\(?([+-]?\d+)?\)?", unicodedata.normalize("NFKC", v))
                    row["weight"] = int(w.group(1)) if w else None
                    row["weight_change"] = int(w.group(2)) if w and w.group(2) else None
                else:
                    row[key] = v
            if row.get("horse_number") not in (None, ""):
                rows.append(row)
        if rows:
            return rows
    return None


def parse_shutuba(html: str) -> list[dict[str, Any]] | None:
    return parse_table(html, ("horse_number", "horse_name"))


def parse_result(html: str) -> list[dict[str, Any]] | None:
    return parse_table(html, ("finish_position", "horse_number"))


def parse_win_odds(html: str) -> list[dict[str, Any]] | None:
    return parse_table(html, ("horse_number", "win_odds"))


def to_storage_doc(race_id: str, kind: str, rows: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """保存用の文書。出どころ jra を付ける。kind: shutuba|result|odds。"""
    key = {"shutuba": "entries", "result": "entries", "odds": "entries"}[kind]
    return stamp({"race_id": race_id, key: rows, **(meta or {})}, "jra")
