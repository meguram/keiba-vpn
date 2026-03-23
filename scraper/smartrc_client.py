"""
SmartRC API クライアント

smartrc.jp の smartrc.php プロキシを介して JSON データを取得する。
認証情報は不要。X-Requested-With ヘッダーのみで全データ取得可能。

使用エンドポイント:
  - days/view         : 開催日カレンダー
  - races/view        : 日付+場所別レース一覧
  - runners/view      : 出走馬データ (評価・テン1F・推定人気・過去5走等)
  - horses/view       : 馬情報 (血統・系統・色分け)
  - fullresults/view  : 全戦績

rcode フォーマット: YYYYMMDDPPKKNNRR (16桁)
  YYYYMMDD = 開催日
  PP       = 場所コード (01=札幌 ... 10=小倉)
  KK       = 回次
  NN       = 日次
  RR       = レース番号
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("scraper.smartrc")

JRA_PLACE_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}

_API_BASE = "https://www.smartrc.jp/v3/smartrc.php"


def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def netkeiba_to_rcode(race_id: str, date: str) -> str:
    """
    netkeiba race_id (YYYYPPKKNNRR 12桁) + 日付 (YYYYMMDD)
    → SmartRC rcode (YYYYMMDDPPKKNNRR 16桁)
    """
    place = race_id[4:6]
    kai = race_id[6:8]
    nichi = race_id[8:10]
    race_num = race_id[10:12]
    return f"{date}{place}{kai}{nichi}{race_num}"


def rcode_to_netkeiba(rcode: str) -> str:
    """SmartRC rcode (16桁) → netkeiba race_id (12桁)"""
    year = rcode[:4]
    place = rcode[8:10]
    kai = rcode[10:12]
    nichi = rcode[12:14]
    race_num = rcode[14:16]
    return f"{year}{place}{kai}{nichi}{race_num}"


class SmartRCClient:
    """SmartRC JSON API クライアント (認証不要)。"""

    def __init__(self, interval: float = 0.5):
        _load_env()
        self._interval = interval
        self._last_request_time: float = 0

    @property
    def available(self) -> bool:
        try:
            data = self._request("days")
            return data.get("success", False) or "data" in data
        except Exception:
            return False

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, operation: str, body: dict | None = None) -> dict:
        """POST リクエストを送信して JSON を返す。"""
        self._throttle()
        url = f"{_API_BASE}/{operation}/view"
        headers = {"X-Requested-With": "XMLHttpRequest"}
        resp = requests.post(
            url,
            data=json.dumps(body or {}),
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("SmartRC [%d]: %s %s", resp.status_code, operation, body)
            return {"success": False, "data": []}
        try:
            return resp.json()
        except Exception:
            logger.warning("SmartRC JSON パースエラー: %s", operation)
            return {"success": False, "data": []}

    # ── days/view ─────────────────────────────────────

    def get_days(self) -> list[dict]:
        """
        開催日カレンダーを取得する。

        Returns:
            [{'rdate': '20260315', 'place': '05', 'name': '金鯱賞', 'class': 'B'}, ...]
        """
        raw = self._request("days").get("data", [])
        result = []
        for d in raw:
            rdate = d.get("rdate", "")
            races_str = d.get("races", "")
            if not races_str:
                continue
            for race in races_str.split(","):
                parts = race.split(":")
                if len(parts) >= 3:
                    result.append({
                        "rdate": rdate,
                        "place": parts[0].strip(),
                        "name": parts[1].strip(),
                        "class": parts[2].strip(),
                    })
        return result

    # ── races/view ────────────────────────────────────

    def get_races(self, rdate: str, place: str) -> list[dict]:
        """
        指定日+場所のレース一覧を取得する。

        Returns:
            rcode, 距離, トラック, 出走頭数, 発走時刻等を含むリスト。
        """
        return self._request("races", {"rdate": rdate, "place": place}).get("data", [])

    # ── runners/view ──────────────────────────────────

    def get_runners(self, rcode: str) -> list[dict]:
        """
        出走馬データを取得する。
        評価、テン1F、推定人気、CR、過去5走サマリ(h1_〜h5_)等を含む。
        """
        return self._request("runners", {"rcode": rcode}).get("data", [])

    # ── horses/view ───────────────────────────────────

    def get_horse(self, hcode: str) -> dict | None:
        """
        馬情報を取得する。血統・系統・色分け情報を含む。

        Returns:
            馬情報の dict。見つからない場合は None。
        """
        data = self._request("horses", {"hcode": hcode}).get("data", [])
        return data[0] if data else None

    # ── fullresults/view ──────────────────────────────

    def get_fullresults(self, hcode: str) -> list[dict]:
        """
        1頭の全戦績を取得する。
        着順、タイム、上り3F、テン1F、コーナー通過順、着差等を含む。
        """
        return self._request("fullresults", {"hcode": hcode}).get("data", [])

    # ── 高レベル: 1レースの全データ取得 ───────────────

    def scrape_race(self, rcode: str) -> dict[str, Any]:
        """
        1レースの runners + 各馬の horse + fullresults をまとめて取得する。

        Returns:
            {
                "rcode": "...",
                "race_id": "...",
                "source": "smartrc",
                "runners": [...],       # runners/view の生データ
                "horses": {...},        # hcode → horses/view データ
                "fullresults": {...},   # hcode → fullresults/view データ
            }
        """
        race_id = rcode_to_netkeiba(rcode)
        runners = self.get_runners(rcode)

        if not runners:
            logger.info("SmartRC runners 空: rcode=%s", rcode)
            return {
                "rcode": rcode,
                "race_id": race_id,
                "source": "smartrc",
                "runners": [],
                "horses": {},
                "fullresults": {},
            }

        horses: dict[str, Any] = {}
        fullresults: dict[str, list] = {}

        for r in runners:
            hcode = r.get("hcode", "")
            if not hcode or hcode in horses:
                continue
            try:
                horse = self.get_horse(hcode)
                if horse:
                    horses[hcode] = horse
            except Exception as e:
                logger.warning("horses 取得失敗 [%s]: %s", hcode, e)

            try:
                results = self.get_fullresults(hcode)
                if results:
                    fullresults[hcode] = results
            except Exception as e:
                logger.warning("fullresults 取得失敗 [%s]: %s", hcode, e)

        logger.info(
            "SmartRC race %s: %d runners, %d horses, %d fullresults entries",
            race_id, len(runners), len(horses),
            sum(len(v) for v in fullresults.values()),
        )

        return {
            "rcode": rcode,
            "race_id": race_id,
            "source": "smartrc",
            "runners": runners,
            "horses": horses,
            "fullresults": fullresults,
        }

    def scrape_date(self, date: str) -> dict[str, Any]:
        """
        指定日の全レースの rcode 一覧を取得する。

        Returns:
            {
                "date": "20260315",
                "source": "smartrc",
                "venues": [...],   # 各場のレース一覧
                "rcodes": [...],   # 全 rcode リスト
            }
        """
        days = self.get_days()
        places = [d["place"] for d in days if d["rdate"] == date
                  and d["place"] in JRA_PLACE_CODES]

        venues: list[dict] = []
        rcodes: list[str] = []

        for place in places:
            races = self.get_races(date, place)
            venue_races = []
            for race in races:
                rcode = race.get("rcode", "")
                if rcode:
                    rcodes.append(rcode)
                    venue_races.append(race)
            venues.append({"place": place, "races": venue_races})

        logger.info("SmartRC date %s: %d venues, %d races", date, len(venues), len(rcodes))
        return {
            "date": date,
            "source": "smartrc",
            "venues": venues,
            "rcodes": rcodes,
        }
