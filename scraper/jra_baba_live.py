"""
JRA公式 馬場情報ライブスクレイパー

https://www.jra.go.jp/keiba/baba/ からクッション値・含水率・馬場状態を
リアルタイムに取得し、既存のクッション値JSONフォーマットにマージ保存する。

データ取得方法:
  1. _data_cushion.html (静的HTML/Shift_JIS) → 全開催場×全日のクッション値
  2. 各場のページ (index.html等) → ヘッダー情報 (開催回・日・日付)
  3. Playwright → _moist_data JS変数 → 全日程の含水率
  4. 使用コース情報

馬場情報の更新タイミング (JRA年間スケジュール連動):
  前日: 09:00〜11:30 頃に前日発表
  当日: 05:30〜08:30 頃に朝計測

スケジュールは JRA公式カレンダー API から自動取得し、
変則開催 (祝日月曜・火曜 etc.) も含めた全開催日を正確にカバーする。

Usage:
  python -m scraper.jra_baba_live               # 最新データ取得 (1回)
  python -m scraper.jra_baba_live --watch       # スケジュール連動ポーリング (推奨)
  python -m scraper.jra_baba_live --check       # 更新チェックのみ
  python -m scraper.jra_baba_live --schedule    # 直近のポーリング予定を表示
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("scraper.jra_baba_live")

BASE_URL = "https://www.jra.go.jp/keiba/baba"
VENUE_PAGES = ["index.html", "index2.html", "index3.html"]

VENUE_NAME_TO_CODE = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

POLL_INTERVAL_SEC = 300  # 5分間隔


class JRABabaLiveScraper:
    """JRA公式馬場情報ページからライブデータを取得する。"""

    def __init__(self, output_dir: str = "data/jra_baba"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.output_dir / ".last_cushion_hash"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        })

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def scrape(self) -> list[dict]:
        """全開催場のデータを取得して保存する。"""
        cushion_map = self._fetch_cushion_data()
        if not cushion_map:
            logger.info("クッション値データなし (開催なしの可能性)")
            return []

        venue_info = self._fetch_venue_info()
        moisture_map = self._fetch_moisture_data()

        all_records = self._merge_data(cushion_map, venue_info, moisture_map)

        if all_records:
            self._save_records(all_records)
            logger.info("合計 %d レコード保存完了", len(all_records))
        else:
            logger.info("マージ結果: 0 レコード")

        return all_records

    def has_new_data(self) -> bool:
        """
        _data_cushion.html を軽量取得して前回と比較。
        変更があれば True を返す。
        """
        url = f"{BASE_URL}/_data_cushion.html"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.debug("軽量チェック失敗: %s", e)
            return False

        current_hash = hashlib.md5(resp.content).hexdigest()
        prev_hash = ""
        if self._state_file.exists():
            prev_hash = self._state_file.read_text().strip()

        if current_hash != prev_hash:
            self._state_file.write_text(current_hash)
            logger.info("更新検知: ハッシュ変更 %s → %s",
                        prev_hash[:8] or "(初回)", current_hash[:8])
            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    #  Step 1: クッション値の取得
    # ═══════════════════════════════════════════════════════════════

    def _fetch_cushion_data(self) -> dict[str, list[dict]]:
        url = f"{BASE_URL}/_data_cushion.html"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.error("クッション値HTML取得失敗: %s", e)
            return {}

        resp.encoding = "shift_jis"
        soup = BeautifulSoup(resp.text, "html.parser")
        result: dict[str, list[dict]] = {}

        for rc_div in soup.select("[id^='rc']"):
            venue_name = rc_div.get("title", "")
            if not venue_name:
                continue
            entries = []
            for unit in rc_div.select(".unit"):
                time_el = unit.select_one(".time")
                cushion_el = unit.select_one(".cushion")
                if not time_el or not cushion_el:
                    continue
                time_text = time_el.get_text(strip=True)
                try:
                    cv = float(cushion_el.get_text(strip=True))
                except (ValueError, TypeError):
                    continue
                entries.append({"time": time_text, "cushion": cv})
            if entries:
                result[venue_name] = entries
                logger.info("  クッション値: %s → %d エントリ", venue_name, len(entries))
        return result

    # ═══════════════════════════════════════════════════════════════
    #  Step 2: 開催情報の取得
    # ═══════════════════════════════════════════════════════════════

    def _fetch_venue_info(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for vp in VENUE_PAGES:
            url = f"{BASE_URL}/{vp}"
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                logger.debug("ページ取得失敗: %s => %s", url, e)
                continue
            resp.encoding = resp.apparent_encoding or "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            h2 = soup.select_one(".contents_header h2")
            if not h2:
                continue
            header_text = h2.get_text(strip=True)
            info = self._parse_header(header_text)
            if not info:
                continue
            course_pos = ""
            for dl in soup.select(".data_line_list.wide"):
                text = dl.get_text()
                if "コース" in text:
                    m = re.search(r"([A-D])コース", text)
                    if m:
                        course_pos = m.group(1)
                    break
            info["course_position"] = course_pos
            result[info["venue_name"]] = info
            logger.info("  開催情報: %s 第%d回第%d日 %s %sコース",
                        info["venue_name"], info["kai"], info["race_day"],
                        info["date"], course_pos)
        return result

    # ═══════════════════════════════════════════════════════════════
    #  Step 3: 含水率の取得 (Playwright)
    # ═══════════════════════════════════════════════════════════════

    def _fetch_moisture_data(self) -> dict[str, list[dict]]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("playwright未インストール。含水率はスキップ。")
            return {}

        result: dict[str, list[dict]] = {}
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                for vp in VENUE_PAGES:
                    url = f"{BASE_URL}/{vp}"
                    try:
                        page.goto(url, timeout=20000)
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        continue
                    h2 = page.query_selector(".contents_header h2")
                    if not h2:
                        continue
                    header_text = h2.inner_text().strip()
                    info = self._parse_header(header_text)
                    if not info:
                        continue
                    moist = page.evaluate("""
                        () => {
                            if (typeof _moist_data === 'undefined' || !_moist_data.length)
                                return null;
                            return _moist_data.map(m => ({
                                time: m.time,
                                name: m.name || '',
                                turf_goal: m.turf && m.turf[0] ? m.turf[0].g : null,
                                turf_4c:   m.turf && m.turf[0] ? m.turf[0]['4c'] : null,
                                dirt_goal: m.dirt && m.dirt[0] ? m.dirt[0].g : null,
                                dirt_4c:   m.dirt && m.dirt[0] ? m.dirt[0]['4c'] : null,
                            }));
                        }
                    """)
                    if moist:
                        result[info["venue_name"]] = moist
                        logger.info("  含水率: %s → %d エントリ", info["venue_name"], len(moist))
                browser.close()
        except Exception as e:
            logger.error("Playwright含水率取得エラー: %s", e)
        return result

    # ═══════════════════════════════════════════════════════════════
    #  データ結合
    # ═══════════════════════════════════════════════════════════════

    def _merge_data(
        self,
        cushion_map: dict[str, list[dict]],
        venue_info: dict[str, dict],
        moisture_map: dict[str, list[dict]],
    ) -> list[dict]:
        records = []
        for venue_name, cushion_entries in cushion_map.items():
            info = venue_info.get(venue_name, {})
            venue_code = VENUE_NAME_TO_CODE.get(venue_name, "")
            year = info.get("year", datetime.date.today().year)
            kai = info.get("kai", 0)
            race_day_num = info.get("race_day", 0)
            course_pos = info.get("course_position", "")

            moisture_entries = moisture_map.get(venue_name, [])
            moisture_by_date: dict[str, dict] = {}
            for me in moisture_entries:
                date_key = self._time_to_date(me.get("time", ""), year)
                if date_key:
                    moisture_by_date[date_key] = me

            for ce in cushion_entries:
                time_text = ce["time"]
                date_str = self._time_to_date(time_text, year)
                if not date_str:
                    continue
                weekday = self._extract_weekday(time_text)
                is_race = weekday in ("土曜日", "日曜日", "月曜日")
                me = moisture_by_date.get(date_str, {})
                record = {
                    "date": date_str,
                    "weekday": weekday,
                    "year": year,
                    "venue_code": venue_code,
                    "venue_name": venue_name,
                    "kai": kai,
                    "race_day": race_day_num if is_race else 0,
                    "is_race_day": is_race,
                    "course_position": course_pos,
                    "cushion_value": ce["cushion"],
                    "turf_moisture_goal": self._safe_float(me.get("turf_goal")),
                    "turf_moisture_4corner": self._safe_float(me.get("turf_4c")),
                    "dirt_moisture_goal": self._safe_float(me.get("dirt_goal")),
                    "dirt_moisture_4corner": self._safe_float(me.get("dirt_4c")),
                    "measurement_time": time_text,
                    "scraped_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                records.append(record)
        records.sort(key=lambda x: (x["date"], x["venue_code"]))
        return records

    # ═══════════════════════════════════════════════════════════════
    #  ユーティリティ
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_header(text: str) -> dict | None:
        m = re.search(
            r'第(\d+)回\s*(.+?)競馬\s*第(\d+)日.*?(\d{4})年\s*(\d+)月\s*(\d+)日.*?（(.+?)）',
            text,
        )
        if not m:
            return None
        venue_name = m.group(2).strip()
        if venue_name not in VENUE_NAME_TO_CODE:
            return None
        year, month, day = int(m.group(4)), int(m.group(5)), int(m.group(6))
        weekday = m.group(7).strip()
        if not weekday.endswith("日"):
            weekday += "日"
        return {
            "kai": int(m.group(1)),
            "race_day": int(m.group(3)),
            "venue_name": venue_name,
            "venue_code": VENUE_NAME_TO_CODE[venue_name],
            "year": year,
            "date": f"{year}-{month:02d}-{day:02d}",
            "weekday": weekday,
        }

    @staticmethod
    def _time_to_date(time_text: str, year: int) -> str:
        m = re.search(r"(\d+)月\s*(\d+)日", time_text)
        if not m:
            return ""
        return f"{year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    @staticmethod
    def _extract_weekday(time_text: str) -> str:
        m = re.search(r"（(.+?)）", time_text)
        if m:
            wd = m.group(1).strip()
            return wd if wd.endswith("日") else wd + "日"
        return ""

    @staticmethod
    def _safe_float(val) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ═══════════════════════════════════════════════════════════════
    #  保存
    # ═══════════════════════════════════════════════════════════════

    def _save_records(self, records: list[dict]):
        all_path = self.output_dir / "cushion_values.json"
        existing: list[dict] = []
        if all_path.exists():
            try:
                existing = json.loads(all_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        existing_idx: dict[str, int] = {}
        for i, r in enumerate(existing):
            key = f"{r.get('date', '')}_{r.get('venue_code', '')}"
            existing_idx[key] = i

        added = 0
        updated = 0
        for r in records:
            key = f"{r['date']}_{r['venue_code']}"
            if key in existing_idx:
                existing[existing_idx[key]] = r
                updated += 1
            else:
                existing.append(r)
                existing_idx[key] = len(existing) - 1
                added += 1

        existing.sort(key=lambda x: (x.get("date", ""), x.get("venue_code", "")))
        all_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("保存: %s (計%d件, 新規%d, 更新%d)",
                     all_path, len(existing), added, updated)

        years_seen = {r.get("year") for r in records if r.get("year")}
        for year in years_seen:
            year_path = self.output_dir / f"cushion_{year}.json"
            year_records = [r for r in existing if r.get("year") == year]
            year_path.write_text(
                json.dumps(year_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("  年度別: %s (%d件)", year_path, len(year_records))


# ═══════════════════════════════════════════════════════════════════
#  スケジュール連動ポーリングデーモン
# ═══════════════════════════════════════════════════════════════════

def _load_poll_schedule(output_dir: str = "data/jra_baba") -> list[dict]:
    """カレンダーJSONからpoll_scheduleをロード。なければ取得。"""
    from scraper.jra_calendar import JRACalendarScraper
    scraper = JRACalendarScraper(output_dir=output_dir)
    year = datetime.date.today().year
    data = scraper.get_schedule(year)
    return data.get("poll_schedule", [])


def _get_today_entry(schedule: list[dict]) -> dict | None:
    today = datetime.date.today().isoformat()
    for entry in schedule:
        if entry["date"] == today:
            return entry
    return None


def _get_poll_windows(entry: dict) -> list[tuple[str, str]]:
    """スケジュールエントリからポーリング窓のリストを返す。"""
    windows = []
    t = entry.get("type", "")
    pw = entry.get("poll_window")
    if pw:
        windows.append((pw[0], pw[1]))
    pdw = entry.get("pre_day_window")
    if pdw:
        windows.append((pdw[0], pdw[1]))
    return windows


def _in_any_window(windows: list[tuple[str, str]]) -> bool:
    now_hm = datetime.datetime.now().strftime("%H:%M")
    for start, end in windows:
        if start <= now_hm <= end:
            return True
    return False


def _next_poll_time(schedule: list[dict]) -> tuple[str, str, int]:
    """
    次にポーリングすべき時刻を計算。
    Returns: (date_str, window_start, seconds_until)
    """
    now = datetime.datetime.now()
    today = now.date()

    for days_ahead in range(60):
        check_date = today + datetime.timedelta(days=days_ahead)
        check_str = check_date.isoformat()

        for entry in schedule:
            if entry["date"] != check_str:
                continue

            windows = _get_poll_windows(entry)
            for start, end in windows:
                h, m = map(int, start.split(":"))
                window_start = datetime.datetime.combine(check_date, datetime.time(h, m))

                if days_ahead == 0:
                    eh, em = map(int, end.split(":"))
                    window_end = datetime.datetime.combine(check_date, datetime.time(eh, em))
                    if now <= window_end:
                        if now >= window_start:
                            return check_str, start, 0
                        else:
                            return check_str, start, int((window_start - now).total_seconds())
                else:
                    wait = int((window_start - now).total_seconds())
                    if wait > 0:
                        return check_str, start, wait

    return "", "", 3600


def start_watch_daemon(output_dir: str = "data/jra_baba"):
    """
    JRA年間スケジュール連動の更新検知ポーリングデーモン。

    1. 年間カレンダーから全開催日 + 前日発表日を取得
    2. 各日の適切な時間帯 (当日朝 / 前日発表) にポーリング
    3. 5分間隔で _data_cushion.html のハッシュを監視
    4. 変更検知時にフルスクレイプ (含水率含む) を実行
    5. 変則開催 (祝日月曜・火曜 etc.) も完全カバー
    """
    logger.info("=" * 60)
    logger.info("  JRA馬場情報 スケジュール連動ポーリングデーモン")
    logger.info("=" * 60)

    schedule = _load_poll_schedule(output_dir)
    if not schedule:
        logger.error("ポーリングスケジュール取得失敗")
        return

    # 直近の監視日を表示
    today = datetime.date.today().isoformat()
    upcoming = [e for e in schedule if e["date"] >= today][:10]
    logger.info("直近の監視日 (%d日中の先頭10日):", len([e for e in schedule if e["date"] >= today]))
    for e in upcoming:
        windows = _get_poll_windows(e)
        w_str = " / ".join(f"{s}〜{e}" for s, e in windows)
        venues = ", ".join(e.get("venues", []))
        logger.info("  %s [%s] %s %s", e["date"], e["type"], w_str, venues)

    logger.info("")
    logger.info("ポーリング間隔: %d秒", POLL_INTERVAL_SEC)
    logger.info("停止するには Ctrl+C")
    logger.info("")

    scraper = JRABabaLiveScraper(output_dir=output_dir)

    # 年初にスケジュールを更新するための最終チェック年
    last_schedule_year = datetime.date.today().year

    try:
        while True:
            # 年が変わったらスケジュールを再取得
            current_year = datetime.date.today().year
            if current_year != last_schedule_year:
                logger.info("年跨ぎ → %d年のスケジュール取得", current_year)
                schedule = _load_poll_schedule(output_dir)
                last_schedule_year = current_year

            entry = _get_today_entry(schedule)
            if entry:
                windows = _get_poll_windows(entry)
                if _in_any_window(windows):
                    logger.debug("ポーリング窓内 [%s %s] → チェック中...",
                                 entry["date"], entry["type"])
                    if scraper.has_new_data():
                        venues = ", ".join(entry.get("venues", []))
                        logger.info("★ 新データ検知 [%s %s] %s → フルスクレイプ開始",
                                    entry["date"], entry["type"], venues)
                        try:
                            records = scraper.scrape()
                            logger.info("★ フルスクレイプ完了: %d レコード", len(records))
                        except Exception as e:
                            logger.error("フルスクレイプ失敗: %s", e, exc_info=True)
                    else:
                        logger.debug("変更なし")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

            # 窓外 → 次のポーリング時刻まで待機
            next_date, next_start, wait = _next_poll_time(schedule)
            if wait > 0:
                next_time = datetime.datetime.now() + datetime.timedelta(seconds=wait)
                logger.info("待機中 → 次回: %s %s (%s, あと %s)",
                            next_date, next_start,
                            next_time.strftime("%m/%d %H:%M"),
                            _format_duration(wait))
                time.sleep(min(wait, 1800))
            else:
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("デーモン停止")


def _format_duration(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60}分"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}時間{m}分"


def run_cron_job(output_dir: str = "data/jra_baba") -> int:
    """
    cron から呼ばれる軽量エントリポイント。

    1. カレンダーを参照して今日が監視日か判定
    2. 現在時刻が窓内か判定
    3. 窓内なら has_new_data() → 変更あればフルスクレイプ
    4. 対象外なら即終了 (exit 0)

    Returns: 取得レコード数 (0 = 変更なし or 対象外)
    """
    schedule = _load_poll_schedule(output_dir)
    entry = _get_today_entry(schedule)

    if not entry:
        logger.debug("今日は監視対象外")
        return 0

    windows = _get_poll_windows(entry)
    if not _in_any_window(windows):
        logger.debug("窓外: %s", windows)
        return 0

    scraper = JRABabaLiveScraper(output_dir=output_dir)

    if not scraper.has_new_data():
        logger.info("cron: %s [%s] 変更なし", entry["date"], entry["type"])
        return 0

    venues = ", ".join(entry.get("venues", []))
    logger.info("cron: %s [%s] %s → 新データ検知 → スクレイプ開始",
                entry["date"], entry["type"], venues)

    records = scraper.scrape()
    logger.info("cron: スクレイプ完了 %d レコード", len(records))
    return len(records)


def print_upcoming_schedule(output_dir: str = "data/jra_baba", days: int = 14):
    """直近のポーリングスケジュールを表示。"""
    schedule = _load_poll_schedule(output_dir)
    today = datetime.date.today()
    cutoff = (today + datetime.timedelta(days=days)).isoformat()
    today_str = today.isoformat()

    upcoming = [e for e in schedule if today_str <= e["date"] <= cutoff]

    print(f"\n=== 馬場情報ポーリング予定 (今後{days}日間) ===")
    if not upcoming:
        print("  (予定なし)")
        return

    type_labels = {
        "race_day": "当日朝",
        "pre_day": "前日発表",
        "both": "当日+前日",
    }

    for e in upcoming:
        d = e["date"]
        t = type_labels.get(e.get("type", ""), e.get("type", ""))
        windows = _get_poll_windows(e)
        w_str = " / ".join(f"{s}〜{end}" for s, end in windows)
        venues = ", ".join(e.get("venues", []))

        is_today = d == today_str
        marker = " ← 今日" if is_today else ""

        # 曜日を計算
        y, m, dd = map(int, d.split("-"))
        wd = ["月", "火", "水", "木", "金", "土", "日"][datetime.date(y, m, dd).weekday()]

        print(f"  {d} ({wd}) [{t:6s}] {w_str:20s} {venues}{marker}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="JRA 馬場情報ライブスクレイパー")
    parser.add_argument("--cron", action="store_true",
                        help="cron用: カレンダー参照→窓内なら更新検知→スクレイプ")
    parser.add_argument("--watch", action="store_true",
                        help="スケジュール連動ポーリングモード (常駐)")
    parser.add_argument("--check", action="store_true",
                        help="更新チェックのみ (スクレイプしない)")
    parser.add_argument("--schedule", action="store_true",
                        help="直近のポーリング予定を表示")
    parser.add_argument("--schedule-days", type=int, default=14,
                        help="--schedule の表示日数 (default: 14)")
    args = parser.parse_args()

    if args.cron:
        run_cron_job()
    elif args.watch:
        start_watch_daemon()
    elif args.check:
        scraper = JRABabaLiveScraper()
        if scraper.has_new_data():
            print("新データあり")
            sys.exit(0)
        else:
            print("変更なし")
            sys.exit(1)
    elif args.schedule:
        print_upcoming_schedule(days=args.schedule_days)
    else:
        scraper = JRABabaLiveScraper()
        records = scraper.scrape()
        if records:
            print(json.dumps(records[:5], ensure_ascii=False, indent=2))
            if len(records) > 5:
                print(f"  ... (計 {len(records)} レコード)")
        else:
            print("取得データなし")


if __name__ == "__main__":
    main()
