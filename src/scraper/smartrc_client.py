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
import random
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("scraper.smartrc")

JRA_PLACE_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}

_API_BASE = "https://www.smartrc.jp/v3/smartrc.php"

# 制限を受けたときの退避。403/429/5xx/通信エラーは待って再試行し、それでも続けば取得を止めて一定時間は一切リクエストしない
_RETRY_BACKOFF_SEC = (30, 120, 300)
_COOLDOWN_SEC = 3 * 3600
_STATE_PATH = Path(os.environ.get("KEIBA_SMARTRC_STATE") or Path(__file__).resolve().parents[2] / "data" / "local" / "meta" / "smartrc_restriction.json")


class SmartRCRestricted(RuntimeError):
    """SmartRC がアクセスを制限している（403/429/5xx が続く）。呼び出し側は取得を中断する。"""


def _load_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
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
    """SmartRC JSON API クライアント (認証不要)。

    アクセスは直列で、1 リクエストごとに interval（既定 1 秒）＋ジッタを空ける。制限（403/429/5xx）を受けたら待って再試行し、
    続けば SmartRCRestricted を送出して cooldown（既定 3 時間）は取得しない。同じ馬の horses/fullresults は reuse_ttl の間は使い回す。"""

    def __init__(self, interval: float = 1.0, jitter: float = 0.4, backoff: tuple[float, ...] = _RETRY_BACKOFF_SEC,
                 cooldown: float = _COOLDOWN_SEC, state_path: Path | None = None, reuse_ttl: float = 12 * 3600):
        _load_env()
        self._interval = interval
        self._jitter = jitter
        self._backoff = backoff
        self._cooldown = cooldown
        self._state_path = state_path or _STATE_PATH
        self._reuse_ttl = reuse_ttl                     # 同じ馬の horses/fullresults を使い回す秒数（馬は 1 日に 2 度走らない）
        self._horse_cache: dict[str, tuple[float, Any, list]] = {}
        self._last_request_time: float = 0

    @property
    def available(self) -> bool:
        try:
            data = self._request("days")
            return data.get("success", False) or "data" in data
        except Exception:
            return False

    def _throttle(self):
        wait = self._interval + random.uniform(0, self._jitter) - (time.time() - self._last_request_time)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    # ── アクセス制限の退避 ─────────────────────────────

    def blocked_until(self) -> float:
        """制限の退避中ならその終了時刻（epoch 秒）。退避中でなければ 0。"""
        try:
            until = float(json.loads(self._state_path.read_text(encoding="utf-8")).get("until", 0))
        except (OSError, ValueError):
            return 0.0
        return until if until > time.time() else 0.0

    def _mark_blocked(self, status: int | None, operation: str) -> None:
        state = {"until": time.time() + self._cooldown, "status": status, "operation": operation,
                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        except OSError as e:                              # 書けなくても、この実行内では止まる
            logger.warning("SmartRC 制限の記録を書けません: %s", e)
        self._blocked_in_memory = state["until"]

    def _request(self, operation: str, body: dict | None = None) -> dict:
        """POST リクエストを送信して JSON を返す。

        403/429/5xx と通信エラーは待って再試行し（Retry-After があれば従う）、それでも続けば SmartRCRestricted を送出して
        cooldown の間は一切リクエストしない（cron の次の起動も、記録を見て止まる）。404 などは空を返す。"""
        url = f"{_API_BASE}/{operation}/view"
        headers = {"X-Requested-With": "XMLHttpRequest"}
        for attempt in range(len(self._backoff) + 1):
            until = max(self.blocked_until(), getattr(self, "_blocked_in_memory", 0.0))
            if until > time.time():
                raise SmartRCRestricted(f"SmartRC はアクセス制限の退避中（あと {int(until - time.time())} 秒）")
            self._throttle()
            status, resp = None, None
            try:
                resp = requests.post(url, data=json.dumps(body or {}), headers=headers, timeout=30)
                status = resp.status_code
            except requests.RequestException as e:
                logger.warning("SmartRC 通信エラー: %s %s", operation, e)
            if status == 200:
                try:
                    return resp.json()
                except Exception:
                    logger.warning("SmartRC JSON パースエラー: %s", operation)
                    return {"success": False, "data": []}
            if status is not None and status < 500 and status not in (403, 429):
                logger.warning("SmartRC [%d]: %s %s", status, operation, body)
                return {"success": False, "data": []}
            if attempt >= len(self._backoff):
                self._mark_blocked(status, operation)
                raise SmartRCRestricted(f"SmartRC が {status} を返し続けた（{operation}）。{int(self._cooldown)} 秒は取得しない")
            wait = self._backoff[attempt]
            try:
                wait = min(max(wait, float(resp.headers.get("Retry-After", 0))), 900) if resp is not None else wait
            except (TypeError, ValueError):
                pass
            logger.warning("SmartRC [%s] %s: %d 秒待って再試行（%d/%d）", status, operation, wait, attempt + 1, len(self._backoff))
            time.sleep(wait)
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
            hit = self._horse_cache.get(hcode)
            if hit and time.time() - hit[0] <= self._reuse_ttl:
                if hit[1]:
                    horses[hcode] = hit[1]
                if hit[2]:
                    fullresults[hcode] = hit[2]
                continue
            horse, results = None, []
            try:
                horse = self.get_horse(hcode)
                if horse:
                    horses[hcode] = horse
            except SmartRCRestricted:
                raise
            except Exception as e:
                logger.warning("horses 取得失敗 [%s]: %s", hcode, e)

            try:
                results = self.get_fullresults(hcode)
                if results:
                    fullresults[hcode] = results
            except SmartRCRestricted:
                raise
            except Exception as e:
                logger.warning("fullresults 取得失敗 [%s]: %s", hcode, e)
            if horse or results:
                self._horse_cache[hcode] = (time.time(), horse, results)

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
