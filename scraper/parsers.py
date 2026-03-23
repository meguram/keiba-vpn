"""
netkeiba.com ページパーサー

各ページタイプのHTML構造を解析し、構造化データ(dict)を返す。
SelectorChain による多段フォールバックで構造変更に対応する。

対応ページ:
  1. RaceResultParser       - 過去レース結果 (db.netkeiba.com/race/{id}/)
  2. RaceCardParser         - 出馬表 (race.netkeiba.com/race/shutuba.html?race_id={id})
  3. HorseParser            - 馬情報 (db.netkeiba.com/horse/{id}/)
  4. RaceListParser         - 日付別レース一覧 (db.netkeiba.com/race/list/{date}/)
  5. SpeedIndexParser       - タイム指数 (race.netkeiba.com/race/speed.html?race_id={id}) [要ログイン]
  6. ShutubaPastParser      - 馬柱・調教 (race.netkeiba.com/race/shutuba_past.html?race_id={id})
  7. OddsParser             - オッズ (race.netkeiba.com/odds/index.html?race_id={id})
  8. PaddockParser          - パドック評価 (race.netkeiba.com/race/paddock.html?race_id={id}) [要ログイン]
  9. BarometerParser        - 偏差値バロメーター (race.netkeiba.com/race/barometer.html?race_id={id}) [要ログイン]
 10. HorseLapParser         - 馬別ラップ (race.netkeiba.com/race/oikiri.html?race_id={id})
 11. TrainingParser         - 馬別調教タイム (db.netkeiba.com/horse/training.html?id={id}) [要ログイン]
 12. TrainerCommentParser   - 厩舎コメント (race.netkeiba.com/race/comment.html?race_id={id}) [要ログイン]
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from scraper.selectors import (
    SelectorChain, safe_text, safe_attr, extract_id_from_url,
    extract_numbers, parse_weight_change,
)

logger = logging.getLogger("scraper.parsers")


# ═══════════════════════════════════════════════════════
# 1. 過去レース結果パーサー
#    URL: https://db.netkeiba.com/race/{race_id}/
# ═══════════════════════════════════════════════════════
class RaceResultParser:
    """
    過去のレース結果ページをパースする。

    出力:
    {
      "race_id": "202505020811",
      "race_name": "ヴィクトリアマイル",
      "date": "2025-05-18",
      "venue": "東京",
      "round": 11,
      "surface": "芝",
      "distance": 1600,
      "direction": "左",
      "weather": "曇",
      "track_condition": "良",
      "grade": "G1",
      "entries": [ {...}, ... ],
      "payoff": {...},
      "lap_times": [...]
    }
    """

    _RESULT_TABLE = SelectorChain("result_table", [
        "table.race_table_01",
        "table[class*='race_table']",
        "table[summary*='レース結果']",
        "#contents_liquid table",
    ])
    _RACE_NAME = SelectorChain("race_name", [
        "dl.racedata h1",
        ".racedata h1",
        "h1[class*='RaceName']",
        "#main h1",
    ])
    _RACE_DATA = SelectorChain("race_data", [
        "dl.racedata dd",
        ".racedata dd",
        "div.RaceData01",
        "p[class*='RaceData']",
    ])
    _RACE_DATE = SelectorChain("race_date", [
        "div.race_head_inner p[class*='smalltxt']",
        "#main .smalltxt",
        "p.smalltxt",
        "div.data_intro p",
    ])

    _GRADE_MAP = {"(G1)": "G1", "(G2)": "G2", "(G3)": "G3", "(Ｇ１)": "G1", "(Ｇ２)": "G2", "(Ｇ３)": "G3",
                  "(GI)": "G1", "(GII)": "G2", "(GIII)": "G3",
                  "(L)": "L", "オープン": "OP", "3勝": "3勝", "2勝": "2勝", "1勝": "1勝", "未勝利": "未勝利", "新馬": "新馬"}

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id}

        result.update(self._parse_race_info(soup))
        result["entries"] = self._parse_entries(soup)
        result["field_size"] = len(result["entries"])
        result["payoff"] = self._parse_payoff(soup)
        result["lap_times"] = self._parse_lap(soup)
        result["pace"] = self._parse_pace(soup, result["lap_times"])

        return result

    def _parse_race_info(self, soup: BeautifulSoup) -> dict:
        info: dict[str, Any] = {}

        name_tag = self._RACE_NAME.select_one(soup)
        raw_name = safe_text(name_tag)
        info["race_name"] = re.sub(r"\(G[I1-3]+\)|\(Ｇ[１-３]\)", "", raw_name).strip()

        grade = "その他"
        for key, val in self._GRADE_MAP.items():
            if key in raw_name or key in str(soup):
                grade = val
                break
        info["grade"] = grade

        data_tag = self._RACE_DATA.select_one(soup)
        data_text = safe_text(data_tag) if data_tag else ""

        # 「芝左1600m」「ダ右1200m」等のパターン
        m = re.search(r"(芝|ダート?|障)(左|右|直)?\s*(\d{3,5})\s*m", data_text)
        if m:
            surface_raw = m.group(1)
            info["surface"] = "ダート" if surface_raw.startswith("ダ") else surface_raw
            info["direction"] = m.group(2) or ""
            info["distance"] = int(m.group(3))
        else:
            info["surface"], info["direction"], info["distance"] = "", "", 0

        m_w = re.search(r"天候\s*[:：]\s*(\S+)", data_text)
        info["weather"] = m_w.group(1) if m_w else ""

        m_t = re.search(r"(?:芝|ダート?|障)\s*[:：]\s*(\S+)", data_text)
        info["track_condition"] = m_t.group(1) if m_t else ""

        m_start = re.search(r"発走\s*[:：]\s*(\d{1,2}:\d{2})", data_text)
        info["start_time"] = m_start.group(1) if m_start else ""

        date_tag = self._RACE_DATE.select_one(soup)
        date_text = safe_text(date_tag) if date_tag else ""
        m_d = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_text)
        if m_d:
            info["date"] = f"{m_d.group(1)}-{int(m_d.group(2)):02d}-{int(m_d.group(3)):02d}"
        else:
            info["date"] = ""

        m_v = re.search(r"(?:\d+回)?(\S+?)(?:\d+日目|$)", date_text)
        info["venue"] = m_v.group(1) if m_v else ""

        m_r = re.search(r"(\d+)\s*R", str(soup.find("span", class_="race_num") or soup.title or ""))
        info["round"] = int(m_r.group(1)) if m_r else 0

        return info

    def _parse_entries(self, soup: BeautifulSoup) -> list[dict]:
        table = self._RESULT_TABLE.select_one(soup)
        if not table:
            return []

        rows = table.select("tr")[1:]  # ヘッダ行スキップ
        entries = []
        for row in rows:
            cols = row.select("td")
            if len(cols) < 13:
                continue

            pos_raw = safe_text(cols[0])
            if pos_raw in ("取", "除", "中", "失"):
                finish_position = -1
            else:
                try:
                    finish_position = int(pos_raw)
                except ValueError:
                    finish_position = -1

            horse_link = cols[3].select_one("a[href*='/horse/']")
            horse_name = safe_text(horse_link) if horse_link else safe_text(cols[3])
            horse_id = extract_id_from_url(safe_attr(horse_link, "href")) if horse_link else ""

            jockey_link = cols[6].select_one("a[href*='/jockey/']")
            jockey_name = safe_text(jockey_link) if jockey_link else safe_text(cols[6])
            jockey_id = extract_id_from_url(safe_attr(jockey_link, "href"), r"/(\w+)/?$") if jockey_link else ""

            trainer_link = None
            for c in cols[12:]:
                t = c.select_one("a[href*='/trainer/']")
                if t:
                    trainer_link = t
                    break
            trainer_name = safe_text(trainer_link) if trainer_link else ""
            trainer_id = extract_id_from_url(safe_attr(trainer_link, "href"), r"/(\w+)/?$") if trainer_link else ""

            weight_text = safe_text(cols[14]) if len(cols) > 14 else ""
            weight, weight_change = parse_weight_change(weight_text)

            finish_time_str = safe_text(cols[7])
            time_sec = self._time_to_sec(finish_time_str)

            entry = {
                "finish_position": finish_position,
                "bracket_number": int(safe_text(cols[1]) or 0),
                "horse_number": int(safe_text(cols[2]) or 0),
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": safe_text(cols[4]),
                "jockey_weight": self._to_float(safe_text(cols[5])),
                "jockey_name": jockey_name,
                "jockey_id": jockey_id,
                "finish_time": finish_time_str,
                "time_sec": time_sec,
                "margin": safe_text(cols[8]) if len(cols) > 8 else "",
                "passing_order": safe_text(cols[10]) if len(cols) > 10 else "",
                "last_3f": self._to_float(safe_text(cols[11])) if len(cols) > 11 else 0,
                "odds": self._to_float(safe_text(cols[12])) if len(cols) > 12 else 0,
                "popularity": int(safe_text(cols[13]) or 0) if len(cols) > 13 else 0,
                "weight": weight,
                "weight_change": weight_change,
                "trainer_name": trainer_name,
                "trainer_id": trainer_id,
            }
            entries.append(entry)

        return entries

    def _parse_payoff(self, soup: BeautifulSoup) -> dict:
        payoff = {}
        pay_tables = soup.select("table.pay_table_01, table[class*='pay_table']")
        for table in pay_tables:
            for row in table.select("tr"):
                th = row.select_one("th")
                tds = row.select("td")
                if th and len(tds) >= 2:
                    bet_type = safe_text(th)
                    payoff[bet_type] = {
                        "numbers": safe_text(tds[0]),
                        "payout": safe_text(tds[1]),
                    }
        return payoff

    def _parse_lap(self, soup: BeautifulSoup) -> list[float]:
        for cell in soup.select("td, span"):
            text = safe_text(cell)
            if re.match(r"\d+\.\d\s*-\s*\d+\.\d", text):
                parent_th = cell.find_parent("tr")
                is_pace_row = False
                if parent_th:
                    th = parent_th.select_one("th")
                    if th and "ペース" in safe_text(th):
                        is_pace_row = True
                if not is_pace_row:
                    return [float(x) for x in re.findall(r"\d+\.\d", text)]
        return []

    @staticmethod
    def _parse_pace(soup: BeautifulSoup, lap_times: list[float]) -> dict[str, Any]:
        """ペース行から前半/後半3Fと、ラップからT1F/L1Fを算出する。"""
        pace: dict[str, Any] = {}

        for cell in soup.select("td, span"):
            text = safe_text(cell)
            m = re.search(r"\((\d+\.\d)\s*-\s*(\d+\.\d)\)", text)
            if m:
                parent_tr = cell.find_parent("tr")
                if parent_tr:
                    th = parent_tr.select_one("th")
                    if th and "ペース" in safe_text(th):
                        pace["first_half_3f"] = float(m.group(1))
                        pace["second_half_3f"] = float(m.group(2))
                        break

        if lap_times:
            pace["t1f"] = lap_times[0]
            if len(lap_times) >= 3:
                pace["t3f"] = round(sum(lap_times[:3]), 1)
            pace["l1f"] = lap_times[-1]
            if len(lap_times) >= 3:
                pace["l3f"] = round(sum(lap_times[-3:]), 1)

        return pace

    @staticmethod
    def _time_to_sec(s: str) -> float:
        """'1:31.9' → 91.9, '59.3' → 59.3"""
        m = re.match(r"(\d+):(\d+\.\d+)", s.strip())
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m2 = re.match(r"(\d+\.\d+)", s.strip())
        if m2:
            return float(m2.group(1))
        return 0.0

    @staticmethod
    def _to_float(s: str) -> float:
        nums = extract_numbers(s)
        return float(nums[0]) if nums else 0.0


# ═══════════════════════════════════════════════════════
# 2. 出馬表 (レースカード) パーサー
#    URL: https://race.netkeiba.com/race/shutuba.html?race_id={id}
# ═══════════════════════════════════════════════════════
class RaceCardParser:
    """
    出馬表ページをパースする。

    出力:
    {
      "race_id": "202506010101",
      "race_name": "3歳未勝利",
      "date": "2025-01-05",
      "venue": "中山",
      "round": 1,
      "surface": "ダート",
      "distance": 1200,
      "direction": "右",
      "weather": "晴",
      "track_condition": "良",
      "start_time": "10:05",
      "field_size": 16,
      "grade": "未勝利",
      "race_class": "3歳未勝利",
      "weight_rule": "馬齢",
      "course_type": "",
      "entries": [ {...}, ... ]
    }
    """

    _GRADE_MAP = {
        "(G1)": "G1", "(G2)": "G2", "(G3)": "G3",
        "(Ｇ１)": "G1", "(Ｇ２)": "G2", "(Ｇ３)": "G3",
        "(GI)": "G1", "(GII)": "G2", "(GIII)": "G3",
        "(JpnI)": "JpnI", "(JpnII)": "JpnII", "(JpnIII)": "JpnIII",
        "(Jpn1)": "JpnI", "(Jpn2)": "JpnII", "(Jpn3)": "JpnIII",
        "(L)": "L", "リステッド": "L",
        "オープン": "OP", "OPEN": "OP",
        "3勝": "3勝", "3勝クラス": "3勝",
        "2勝": "2勝", "2勝クラス": "2勝",
        "1勝": "1勝", "1勝クラス": "1勝",
        "未勝利": "未勝利", "新馬": "新馬",
    }

    _ENTRY_TABLE = SelectorChain("shutuba_table", [
        "table.Shutuba_Table",
        "table.ShutubaTable",
        "table.RaceTable01",
        "table[class*='shutuba']",
        "table[class*='Shutuba']",
        "#page table",
    ])
    _ENTRY_ROW = SelectorChain("entry_row", [
        "tr.HorseList",
        "tr[class*='HorseList']",
        "tbody tr",
    ])
    _RACE_NAME = SelectorChain("card_race_name", [
        "h1.RaceName",
        "div.RaceName",
        ".RaceName",
        "h1[class*='RaceName']",
    ])
    _RACE_DATA1 = SelectorChain("card_race_data1", [
        "div.RaceData01",
        "span.RaceData01",
        "div[class*='RaceData']",
    ])
    _RACE_DATA2 = SelectorChain("card_race_data2", [
        "div.RaceData02",
        "span.RaceData02",
    ])

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id}

        result.update(self._parse_card_info(soup))
        result["entries"] = self._parse_entries(soup)

        return result

    def _parse_card_info(self, soup: BeautifulSoup) -> dict:
        info: dict[str, Any] = {}

        name_tag = self._RACE_NAME.select_one(soup)
        info["race_name"] = safe_text(name_tag)

        # RaceData01: "10:05発走 / ダ1200m(右) / 天候:晴 / 馬場:良"
        data1_tag = self._RACE_DATA1.select_one(soup)
        data1 = data1_tag.get_text(" ", strip=True) if data1_tag else ""

        m = re.search(r"(芝|ダ|ダート|障)\s*(\d{3,5})\s*m", data1)
        if m:
            surface_raw = m.group(1)
            info["surface"] = "ダート" if surface_raw.startswith("ダ") else surface_raw
            info["distance"] = int(m.group(2))
        else:
            info["surface"], info["distance"] = "", 0

        m_dir = re.search(r"\d+m\s*\(?\s*(左|右|直)\s*\)?", data1)
        info["direction"] = m_dir.group(1) if m_dir else ""

        m_w = re.search(r"天候\s*[:：]\s*([^\s/]+)", data1)
        info["weather"] = m_w.group(1) if m_w else ""

        m_t = re.search(r"馬場\s*[:：]\s*([^\s/]+)", data1)
        info["track_condition"] = m_t.group(1) if m_t else ""

        m_start = re.search(r"(\d{1,2}:\d{2})\s*発走", data1)
        info["start_time"] = m_start.group(1) if m_start else ""

        # RaceData02: "1回中山1日目 サラ系３歳 未勝利 [指] 馬齢 16頭 ..."
        data2_tag = self._RACE_DATA2.select_one(soup)
        data2 = data2_tag.get_text(" ", strip=True) if data2_tag else ""

        m_v = re.search(r"\d+回\s*(\S+?)\s*\d+日目", data2)
        info["venue"] = m_v.group(1) if m_v else ""

        m_heads = re.search(r"(\d+)\s*頭", data2)
        info["field_size"] = int(m_heads.group(1)) if m_heads else 0

        title = safe_text(soup.title) if soup.title else ""
        m_d = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
        if m_d:
            info["date"] = f"{m_d.group(1)}-{int(m_d.group(2)):02d}-{int(m_d.group(3)):02d}"
        else:
            info["date"] = ""

        m_r = re.search(r"(\d+)\s*R", title)
        info["round"] = int(m_r.group(1)) if m_r else 0

        raw_name = info.get("race_name", "")
        combined = raw_name + " " + data2
        grade = ""
        for key, val in self._GRADE_MAP.items():
            if key in combined:
                grade = val
                break
        info["grade"] = grade

        info["race_class"] = ""
        m_cls = re.search(r"(サラ系?\d歳(?:以上)?\s*(?:新馬|未勝利|\d勝クラス|オープン)?)", data2)
        if m_cls:
            info["race_class"] = m_cls.group(1).strip()

        info["weight_rule"] = ""
        for wr in ("定量", "馬齢", "ハンデ", "別定"):
            if wr in data2:
                info["weight_rule"] = wr
                break

        m_ct = re.search(r"[/／]\s*(内|外|内2周|外2周)", data1)
        if not m_ct:
            m_ct = re.search(r"\((内|外|内2周|外2周)\)", data1)
        info["course_type"] = m_ct.group(1) if m_ct else ""

        return info

    def _parse_entries(self, soup: BeautifulSoup) -> list[dict]:
        table = self._ENTRY_TABLE.select_one(soup)
        if not table:
            return []

        rows = self._ENTRY_ROW.select(table)
        entries = []

        for row in rows:
            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            horse_name = safe_text(horse_link)
            horse_id = extract_id_from_url(safe_attr(horse_link, "href"))

            jockey_link = row.select_one("a[href*='/jockey/']")
            jockey_name = safe_text(jockey_link) if jockey_link else ""
            jockey_id = extract_id_from_url(safe_attr(jockey_link, "href")) if jockey_link else ""

            trainer_tag = row.select_one("td.Trainer")
            trainer_link = row.select_one("a[href*='/trainer/']")
            trainer_name = safe_text(trainer_link) if trainer_link else (safe_text(trainer_tag) if trainer_tag else "")
            trainer_id = extract_id_from_url(safe_attr(trainer_link, "href"), r"/(\w+)/?$") if trainer_link else ""

            # td[0]=Waku td[1]=Umaban td[2]=CheckMark td[3]=HorseInfo
            # td[4]=Barei td[5]=斤量 td[6]=Jockey td[7]=Trainer td[8]=Weight
            cols = row.select("td")

            waku_tag = row.select_one("td[class*='Waku']")
            bracket = int(extract_numbers(safe_text(waku_tag))[0]) if waku_tag and extract_numbers(safe_text(waku_tag)) else 0

            uma_tag = row.select_one("td[class*='Umaban']")
            horse_number = int(extract_numbers(safe_text(uma_tag))[0]) if uma_tag and extract_numbers(safe_text(uma_tag)) else 0

            barei_tag = row.select_one("td.Barei")
            sex_age = safe_text(barei_tag) if barei_tag else ""

            # 斤量: Barei の次のtd (index 5)
            jockey_weight = 0.0
            if len(cols) > 5:
                nums = extract_numbers(safe_text(cols[5]))
                jockey_weight = float(nums[0]) if nums else 0.0

            weight_tag = row.select_one("td.Weight, td[class*='Weight']")
            weight, weight_change = parse_weight_change(safe_text(weight_tag)) if weight_tag else (0, 0)

            odds_tag = row.select_one("td.Odds span, td.Popular span, td[class*='Odds']")
            odds = 0.0
            if odds_tag:
                nums = extract_numbers(safe_text(odds_tag))
                odds = float(nums[0]) if nums else 0.0

            pop_tag = row.select_one("td.Popular span.OddsPeople, span[class*='OddsPeople']")
            popularity = 0
            if pop_tag:
                nums = extract_numbers(safe_text(pop_tag))
                popularity = int(float(nums[0])) if nums else 0

            sire = ""
            dam_sire = ""
            pedigree_area = row.select_one("span.Pedigree, td.Pedigree, div.Pedigree")
            if pedigree_area:
                ped_links = pedigree_area.select("a")
                if len(ped_links) >= 1:
                    sire = safe_text(ped_links[0])
                if len(ped_links) >= 2:
                    dam_sire = safe_text(ped_links[1])

            entry = {
                "horse_number": horse_number,
                "bracket_number": bracket,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": sex_age,
                "jockey_weight": jockey_weight,
                "jockey_name": jockey_name,
                "jockey_id": jockey_id,
                "trainer_name": trainer_name,
                "trainer_id": trainer_id,
                "weight": weight,
                "weight_change": weight_change,
                "odds": odds,
                "popularity": popularity,
                "sire": sire,
                "dam_sire": dam_sire,
            }
            entries.append(entry)

        return entries


# ═══════════════════════════════════════════════════════
# 3. 馬情報パーサー
#    URL: https://db.netkeiba.com/horse/{horse_id}/
# ═══════════════════════════════════════════════════════
class HorseParser:
    """
    馬の詳細情報ページをパースする。

    プロフィールページ: https://db.netkeiba.com/horse/{horse_id}/
    戦績ページ:         https://db.netkeiba.com/horse/result/{horse_id}/

    出力:
    {
      "horse_id": "2021105354",
      "horse_name": "アスコリピチェーノ",
      "name_en": "Ascoli Piceno",
      "status": "現役",
      "sex": "牝",
      "age": 5,
      "color": "黒鹿毛",
      "birthday": "2021年2月24日",
      "trainer": "黒岩陽一",
      "owner": "サンデーレーシング",
      "breeder": "ノーザンファーム",
      "birthplace": "安平町",
      "total_earnings": 39699,
      "career": "11戦6勝",
      "career_record": [6, 2, 0, 3],
      "major_wins": ["25'ヴィクトリアマイル(G1)", ...],
      "sire": "ダイワメジャー",
      "dam": "...",
      "dam_sire": "...",
      "race_history": [...]
    }
    """

    _PROFILE_TABLE = SelectorChain("horse_profile", [
        "table.db_prof_table",
        "div.db_prof_area table",
        "table[class*='prof']",
        "#db_main_box table",
    ])
    _HORSE_NAME = SelectorChain("horse_name", [
        "div.horse_title h1",
        "div.db_head_name h1",
        "h1[class*='horse']",
        "#db_main_box h1",
    ])
    _STATUS_TEXT = SelectorChain("horse_status", [
        "div.horse_title p.txt_01",
        "div.db_head_name p.txt_01",
        "p[class*='txt_01']",
    ])
    _RACE_TABLE = SelectorChain("horse_race_table", [
        "table.db_h_race_results",
        "table[class*='race_results']",
        "table.nk_tb_common",
    ])

    def parse(self, html: str, horse_id: str = "",
              result_html: str | None = None,
              ped_html: str | None = None) -> dict[str, Any]:
        """
        プロフィールHTMLをパースする。
        result_html が指定された場合は戦績もパースする。
        ped_html が指定された場合は血統専用ページからパースする。
        """
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"horse_id": horse_id}

        result.update(self._parse_profile(soup))

        if ped_html:
            ped_soup = BeautifulSoup(ped_html, "html.parser")
            result.update(self._parse_pedigree(ped_soup))
        elif not result.get("sire"):
            result.update(self._parse_pedigree(soup))

        if result_html:
            result_soup = BeautifulSoup(result_html, "html.parser")
            result["race_history"] = self._parse_race_history(result_soup)
        else:
            result["race_history"] = []

        return result

    def _parse_profile(self, soup: BeautifulSoup) -> dict:
        info: dict[str, Any] = {}

        name_tag = self._HORSE_NAME.select_one(soup)
        info["horse_name"] = safe_text(name_tag)

        en_link = soup.select_one("p.eng_name, a[href*='en.netkeiba.com']")
        info["name_en"] = safe_text(en_link) if en_link else ""

        status_tag = self._STATUS_TEXT.select_one(soup)
        status_text = safe_text(status_tag) if status_tag else ""

        m = re.match(r"(現役|引退|抹消)\s*(牡|牝|セ)\s*(\d+)\s*歳\s*(\S+)?",
                     status_text.replace("　", " "))
        if m:
            info["status"] = m.group(1)
            info["sex"] = m.group(2)
            info["age"] = int(m.group(3))
            info["color"] = m.group(4) or ""

        table = self._PROFILE_TABLE.select_one(soup)
        if table:
            for row in table.select("tr"):
                th = safe_text(row.select_one("th"))
                td = row.select_one("td")
                if not td:
                    continue
                val = safe_text(td)
                link = td.select_one("a")
                link_text = safe_text(link) if link else val

                if "生年月日" in th:
                    info["birthday"] = val
                elif "調教師" in th:
                    info["trainer"] = link_text
                elif "馬主" in th:
                    info["owner"] = link_text
                elif "生産者" in th:
                    info["breeder"] = link_text
                elif "産地" in th:
                    info["birthplace"] = val
                elif "獲得賞金" in th and "中央" in th:
                    info["total_earnings"] = self._parse_earnings(val)
                elif "通算成績" in th:
                    info["career"] = val
                    m2 = re.search(r"\[(\d+)-(\d+)-(\d+)-(\d+)\]", val)
                    if m2:
                        info["career_record"] = [int(m2.group(i)) for i in range(1, 5)]
                elif "主な勝鞍" in th:
                    info["major_wins"] = [safe_text(a) for a in td.select("a")]

        return info

    def _parse_pedigree(self, soup: BeautifulSoup) -> dict:
        """血統テーブル (blood_table) から父/母/母父を取得"""
        ped: dict[str, str] = {"sire": "", "dam": "", "dam_sire": ""}

        blood_table = soup.select_one(
            "table.blood_table, table[class*='blood'], "
            "table[summary*='血統']"
        )
        if blood_table:
            big_cells: list[tuple[int, str]] = []
            for td in blood_table.select("td[rowspan]"):
                rs = int(td.get("rowspan", 1))
                link = td.select_one("a")
                name = safe_text(link) if link else ""
                if name:
                    big_cells.append((rs, name))

            if big_cells:
                big_cells_sorted = sorted(big_cells, key=lambda x: x[0], reverse=True)
                max_rs = big_cells_sorted[0][0]
                parents = [c for c in big_cells if c[0] == max_rs]
                if len(parents) >= 1:
                    ped["sire"] = parents[0][1]
                if len(parents) >= 2:
                    ped["dam"] = parents[1][1]
                grandparents = [c for c in big_cells if c[0] == max_rs // 2]
                if len(grandparents) >= 3:
                    ped["dam_sire"] = grandparents[2][1]
            else:
                rows = blood_table.select("tr")
                all_links = []
                for r in rows:
                    link = r.select_one("a")
                    if link:
                        all_links.append(safe_text(link))
                seen: set[str] = set()
                unique: list[str] = []
                for name in all_links:
                    if name and name not in seen:
                        seen.add(name)
                        unique.append(name)
                if len(unique) >= 1:
                    ped["sire"] = unique[0]
                if len(unique) >= 3:
                    ped["dam"] = unique[2]
                elif len(unique) >= 2:
                    ped["dam"] = unique[1]
                if len(unique) >= 4:
                    ped["dam_sire"] = unique[3]

        if not ped["sire"]:
            for row in soup.select("table.db_prof_table tr, table[class*='prof'] tr"):
                th = safe_text(row.select_one("th"))
                td = row.select_one("td")
                if not td:
                    continue
                if "母父" in th or "BMS" in th.upper():
                    link = td.select_one("a")
                    ped["dam_sire"] = safe_text(link) if link else safe_text(td)
                elif "父" in th and "母" not in th:
                    link = td.select_one("a")
                    ped["sire"] = safe_text(link) if link else safe_text(td)
                elif "母" in th:
                    link = td.select_one("a")
                    ped["dam"] = safe_text(link) if link else safe_text(td)

        return ped

    def _parse_race_history(self, soup: BeautifulSoup) -> list[dict]:
        """戦績ページ (/horse/result/{id}/) をパース"""
        table = self._RACE_TABLE.select_one(soup)
        if not table:
            return []

        rows = table.select("tr")[1:]
        history = []
        for row in rows:
            cols = row.select("td")
            if len(cols) < 20:
                continue

            race_link = cols[4].select_one("a[href*='/race/']") if len(cols) > 4 else None

            pos_raw = safe_text(cols[11]) if len(cols) > 11 else ""
            try:
                finish = int(pos_raw)
            except ValueError:
                finish = -1

            wt, wt_chg = parse_weight_change(safe_text(cols[24])) if len(cols) > 24 else (0, 0)

            dist_text = safe_text(cols[14]) if len(cols) > 14 else ""
            m_dist = re.search(r"(芝|ダ|障)(\d{3,5})", dist_text)

            finish_time_str = safe_text(cols[18]) if len(cols) > 18 else ""
            time_sec = self._time_to_sec(finish_time_str)

            rec = {
                "date": safe_text(cols[0]),
                "venue": safe_text(cols[1]),
                "weather": safe_text(cols[2]),
                "race_round": int(safe_text(cols[3]) or 0),
                "race_name": safe_text(race_link) if race_link else safe_text(cols[4]),
                "race_id": extract_id_from_url(safe_attr(race_link, "href")) if race_link else "",
                "field_size": int(safe_text(cols[6]) or 0),
                "bracket_number": int(safe_text(cols[7]) or 0),
                "horse_number": int(safe_text(cols[8]) or 0),
                "odds": self._to_float(safe_text(cols[9])),
                "popularity": int(safe_text(cols[10]) or 0),
                "finish_position": finish,
                "jockey_name": safe_text(cols[12]),
                "jockey_weight": self._to_float(safe_text(cols[13])),
                "surface": m_dist.group(1) if m_dist else "",
                "distance": int(m_dist.group(2)) if m_dist else 0,
                "track_condition": safe_text(cols[16]) if len(cols) > 16 else "",
                "finish_time": finish_time_str,
                "time_sec": time_sec,
                "margin": safe_text(cols[19]) if len(cols) > 19 else "",
                "passing_order": safe_text(cols[21]) if len(cols) > 21 else "",
                "last_3f": self._to_float(safe_text(cols[23])) if len(cols) > 23 else 0,
                "weight": wt,
                "weight_change": wt_chg,
                "winner": safe_text(cols[27]) if len(cols) > 27 else "",
            }
            history.append(rec)

        return history

    @staticmethod
    def _time_to_sec(s: str) -> float:
        """'1:31.9' → 91.9, '59.3' → 59.3"""
        m = re.match(r"(\d+):(\d+\.\d+)", s.strip())
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m2 = re.match(r"(\d+\.\d+)", s.strip())
        if m2:
            return float(m2.group(1))
        return 0.0

    @staticmethod
    def _parse_earnings(text: str) -> int:
        """'3億9,699万円' → 39699 (万円単位)"""
        text = text.replace(",", "").replace(" ", "")
        total = 0
        m_oku = re.search(r"(\d+)億", text)
        if m_oku:
            total += int(m_oku.group(1)) * 10000
        m_man = re.search(r"(\d+)万", text)
        if m_man:
            total += int(m_man.group(1))
        if total == 0:
            nums = extract_numbers(text)
            total = int(float(nums[0])) if nums else 0
        return total

    @staticmethod
    def _to_float(s: str) -> float:
        nums = extract_numbers(s)
        return float(nums[0]) if nums else 0.0


# ═══════════════════════════════════════════════════════
# 4. 日付別レース一覧パーサー
#    参照: db 一覧 / netkeiba トップのレース一覧断片
# ═══════════════════════════════════════════════════════
class RaceListParser:
    """
    指定日のレース一覧からrace_idを収集する。

    対応URL:
      - https://db.netkeiba.com/race/list/{YYYYMMDD}/  (過去・確定データ)
      - https://race.netkeiba.com/top/race_list.html?kaisai_date={YYYYMMDD} (レガシー)
      - https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=...&current_group=...
        （トップ https://race.netkeiba.com/top/?kaisai_date= が AJAX で読む断片・race_id 確定）

    出力:
    {
      "date": "20250518",
      "races": [
        {"race_id": "202505020811", "round": 11, "race_name": "ヴィクトリアマイル"},
        ...
      ]
    }
    """

    VENUE_CODE = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }

    def parse(self, html: str, date: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        races = []

        seen = set()
        for link in soup.select("a[href*='/race/']"):
            href = safe_attr(link, "href")

            # /race/{12桁ID}/ パターン (db.netkeiba.com)
            m = re.search(r"/race/(\d{12})/?", href)
            if not m:
                # race_id=パラメータ (race.netkeiba.com)
                m = re.search(r"race_id=(\d{12})", href)
            if not m:
                continue

            race_id = m.group(1)
            if race_id in seen:
                continue

            # race_id: YYYYPPCCDDRRNN (PP=venue, CC=kai, DD=day, RR=round)
            venue_code = race_id[4:6] if len(race_id) == 12 else ""
            if venue_code not in self.VENUE_CODE:
                continue

            seen.add(race_id)

            text = safe_text(link)
            race_round = int(race_id[10:12]) if len(race_id) == 12 else 0
            venue = self.VENUE_CODE[venue_code]

            races.append({
                "race_id": race_id,
                "round": race_round,
                "venue": venue,
                "race_name": text,
            })

        races.sort(key=lambda r: r["race_id"])

        return {"date": date, "races": races}


# ═══════════════════════════════════════════════════════
# 5. タイム指数パーサー [要ログイン]
#    URL: https://race.netkeiba.com/race/speed.html?race_id={id}
# ═══════════════════════════════════════════════════════
class SpeedIndexParser:
    """
    タイム指数ページをパースする。

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 1,
          "horse_name": "ロードレイジング",
          "horse_id": "2023100380",
          "speed_max": 0,
          "speed_avg": 0,
          "speed_distance": 0,
          "speed_course": 0,
          "speed_recent": [0, 0, 0],
          "odds": 426.0,
          "popularity": 16
        }, ...
      ]
    }
    """

    _TABLE = SelectorChain("speed_table", [
        "table.SpeedIndex_Table",
        "table[class*='SpeedIndex']",
        "table.RaceTable01.ShutubaTable",
    ])

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = self._TABLE.select_one(soup)
        if not table:
            return result

        for row in table.select("tr.HorseList"):
            if not row.select("td"):
                continue

            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            entry = {
                "horse_number": self._get_int(row, "td.UmaBan, td[class*='Umaban']"),
                "horse_name": safe_text(horse_link),
                "horse_id": extract_id_from_url(safe_attr(horse_link, "href")),
                "speed_max": self._extract_speed(row, "td.sk__max_index"),
                "speed_avg": self._extract_speed(row, "td.sk__average_index"),
                "speed_distance": self._extract_speed(row, "td.sk__max_distance_index"),
                "speed_course": self._extract_speed(row, "td.sk__max_course_index"),
                "speed_recent": [
                    self._extract_speed(row, "td.sk__index1"),
                    self._extract_speed(row, "td.sk__index2"),
                    self._extract_speed(row, "td.sk__index3"),
                ],
                "odds": self._get_float(row, "td.sk__odds, td.Odds"),
                "popularity": self._get_int(row, "td.sk__ninki, td.Ninki"),
            }
            result["entries"].append(entry)

        return result

    @staticmethod
    def _extract_speed(row, selector: str) -> int:
        """
        タイム指数セルからソート用hidden値を除いた表示値を抽出する。
        <td><span class="Sort_Function_Data_Hidden">1086</span>86</td> → 86
        """
        td = row.select_one(selector)
        if not td:
            return 0
        hidden = td.select_one("span.Sort_Function_Data_Hidden")
        if hidden:
            hidden.decompose()
        text = td.get_text(strip=True)
        if text in ("-", "未", "***", ""):
            return 0
        nums = extract_numbers(text)
        return int(float(nums[0])) if nums else 0

    @staticmethod
    def _get_int(row, selector: str) -> int:
        el = row.select_one(selector)
        if not el:
            return 0
        nums = extract_numbers(safe_text(el))
        return int(float(nums[0])) if nums else 0

    @staticmethod
    def _get_float(row, selector: str) -> float:
        el = row.select_one(selector)
        if not el:
            return 0.0
        nums = extract_numbers(safe_text(el))
        return float(nums[0]) if nums else 0.0


# ═══════════════════════════════════════════════════════
# 6. 馬柱 + 調教タイム パーサー
#    URL: https://race.netkeiba.com/race/shutuba_past.html?race_id={id}
# ═══════════════════════════════════════════════════════
class ShutubaPastParser:
    """
    馬柱 (過去5走) + 調教タイムをパースする。

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 1,
          "horse_name": "...",
          "horse_id": "...",
          "past_races": [
            {"date": "2026.02.18", "venue": "大井", "race": "雲取賞", "surface_dist": "ダ1800", ...}, ...
          ]
        }, ...
      ],
      "training": [
        {
          "horse_number": 1,
          "horse_name": "...",
          "date": "3/13",
          "course": "栗坂",
          "condition": "良",
          "rider": "...",
          "time": "52.2-38.0-25.2-12.6",
          "rank": "2",
          "impression": "一杯"
        }, ...
      ]
    }
    """

    _PAST_TABLE = SelectorChain("past_table", [
        "table.Shutuba_Past5_Table",
        "table[class*='Shutuba_Past']",
        "table.Shutuba_Table",
    ])
    _TRAINING_TABLE = SelectorChain("training_table", [
        "table.OikiriTable",
        "table.Stable_Time",
        "table[class*='Oikiri']",
    ])

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")

        return {
            "race_id": race_id,
            "entries": self._parse_past(soup),
            "training": self._parse_training(soup),
        }

    def _parse_past(self, soup: BeautifulSoup) -> list[dict]:
        table = self._PAST_TABLE.select_one(soup)
        if not table:
            return []

        entries = []
        rows = table.select("tr.HorseList")

        i = 0
        while i < len(rows):
            row = rows[i]
            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                i += 1
                continue

            waku_tds = row.select("td[class*='Waku']")
            bracket = 0
            horse_num = 0
            if len(waku_tds) >= 2:
                bracket_nums = extract_numbers(safe_text(waku_tds[0]))
                bracket = int(bracket_nums[0]) if bracket_nums else 0
                uma_nums = extract_numbers(safe_text(waku_tds[1]))
                horse_num = int(uma_nums[0]) if uma_nums else 0
            elif len(waku_tds) == 1:
                cls_str = " ".join(waku_tds[0].get("class", []))
                waku_match = re.search(r"Waku(\d)", cls_str)
                if waku_match:
                    bracket = int(waku_match.group(1))
                nums = extract_numbers(safe_text(waku_tds[0]))
                horse_num = int(nums[0]) if nums else 0

            entry = {
                "bracket_number": bracket,
                "horse_number": horse_num,
                "horse_name": safe_text(horse_link),
                "horse_id": extract_id_from_url(safe_attr(horse_link, "href")),
                "past_races": [],
            }

            past_cols = row.select("td.Past")
            for pc in past_cols:
                text = pc.get_text(" ", strip=True)
                if not text or text == "-":
                    continue

                # "2026.02.18 大井13 雲取賞 JpnIII ダ1800 ..."
                past_rec = self._parse_past_cell(text)
                if past_rec:
                    rank_cls = [c for c in (pc.get("class") or []) if c.startswith("Ranking_")]
                    if rank_cls:
                        nums = extract_numbers(rank_cls[0])
                        past_rec["finish_position"] = int(nums[0]) if nums else 0
                    entry["past_races"].append(past_rec)

            entries.append(entry)
            i += 1

        return entries

    @staticmethod
    def _parse_past_cell(text: str) -> dict | None:
        """過去走セルのテキストをパース"""
        m_date = re.search(r"(\d{4}\.\d{2}\.\d{2})", text)
        if not m_date:
            return None

        m_dist = re.search(r"(芝|ダ|障)(\d{3,5})", text)
        m_time = re.search(r"(\d:\d{2}\.\d)", text)

        return {
            "date": m_date.group(1),
            "surface": m_dist.group(1) if m_dist else "",
            "distance": int(m_dist.group(2)) if m_dist else 0,
            "finish_time": m_time.group(1) if m_time else "",
            "raw": text[:80],
        }

    def _parse_training(self, soup: BeautifulSoup) -> list[dict]:
        table = self._TRAINING_TABLE.select_one(soup)
        if not table:
            return []

        trainings = []
        rows = table.select("tr")

        for row in rows[1:]:
            cols = row.select("td")
            if len(cols) < 8:
                continue

            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            uma_tag = row.select_one("td[class*='Umaban'], td[class*='Waku']")
            uma_nums = extract_numbers(safe_text(uma_tag)) if uma_tag else []

            training = {
                "horse_number": int(uma_nums[0]) if uma_nums else 0,
                "horse_name": safe_text(horse_link),
                "date": safe_text(cols[3]) if len(cols) > 3 else "",
                "course": safe_text(cols[4]) if len(cols) > 4 else "",
                "condition": safe_text(cols[5]) if len(cols) > 5 else "",
                "rider": safe_text(cols[6]) if len(cols) > 6 else "",
                "time": safe_text(cols[7]) if len(cols) > 7 else "",
                "rank": safe_text(cols[8]) if len(cols) > 8 else "",
                "impression": safe_text(cols[9]) if len(cols) > 9 else "",
            }
            trainings.append(training)

        return trainings


# ═══════════════════════════════════════════════════════
# 7. オッズパーサー (JSON API ベース)
#    API: https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={id}&type=1|2
#
#    オッズページは JavaScript 動的レンダリングのため HTML パースでは取得不可。
#    netkeiba の内部 JSON API を直接呼び出す。
# ═══════════════════════════════════════════════════════
class OddsParser:
    """
    単勝・複勝オッズを JSON API から取得する。

    API レスポンス (type=1: 単勝):
      {"status":"result","data":{"odds":{"1":{"01":["60.8","","12"],...}}}}
      各馬: [win_odds, _, popularity]

    API レスポンス (type=2: 複勝):
      {"status":"result","data":{"odds":{"2":{"01":["8.9","12.8","12"],...}}}}
      各馬: [place_min, place_max, popularity]

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 1,
          "win_odds": 12.3,
          "place_odds_min": 3.1,
          "place_odds_max": 5.2,
          "popularity": 5
        }, ...
      ]
    }
    """

    ODDS_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={type}"

    WIN_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1"
    PLACE_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=2"
    UMAREN_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=4"
    WIDE_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=5"
    UMATAN_API = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=6"

    def parse_from_api(self, session, race_id: str) -> dict[str, Any]:
        """HTTP session を使って JSON API からオッズを取得・パースする。"""
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        win_data: dict[str, list] = {}
        place_data: dict[str, list] = {}

        try:
            resp = session.get(self.WIN_API.format(race_id=race_id), timeout=15)
            if resp.status_code == 200:
                d = resp.json()
                if d.get("status") == "result":
                    win_data = d.get("data", {}).get("odds", {}).get("1", {})
        except Exception:
            pass

        try:
            resp = session.get(self.PLACE_API.format(race_id=race_id), timeout=15)
            if resp.status_code == 200:
                d = resp.json()
                if d.get("status") == "result":
                    place_data = d.get("data", {}).get("odds", {}).get("2", {})
        except Exception:
            pass

        all_numbers = sorted(set(list(win_data.keys()) + list(place_data.keys())))

        for num_str in all_numbers:
            horse_number = int(num_str) if num_str.isdigit() else 0
            if horse_number == 0:
                continue

            win_arr = win_data.get(num_str, [])
            place_arr = place_data.get(num_str, [])

            win_odds = self._safe_float(win_arr[0]) if len(win_arr) > 0 else 0.0
            popularity = self._safe_int(win_arr[2]) if len(win_arr) > 2 else 0

            place_min = self._safe_float(place_arr[0]) if len(place_arr) > 0 else 0.0
            place_max = self._safe_float(place_arr[1]) if len(place_arr) > 1 else 0.0

            result["entries"].append({
                "horse_number": horse_number,
                "win_odds": win_odds,
                "place_odds_min": place_min,
                "place_odds_max": place_max,
                "popularity": popularity,
            })

        result["entries"].sort(key=lambda e: e["horse_number"])
        return result

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        """HTML フォールバック (出馬表にオッズが埋め込まれているケース)。"""
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = soup.select_one(
            "table.RaceOdds_HorseList_Table, table[class*='Odds'], "
            "table#odds_tan_table, div.TanFukuBlock table"
        )
        if not table:
            return result

        for row in table.select("tr"):
            cols = row.select("td")
            if len(cols) < 3:
                continue

            ninki_td = row.select_one("td.Ninki")
            uma_td = row.select_one("td.UmaBan, td[class*='Umaban']")
            if not ninki_td and not uma_td:
                continue

            uma_nums = extract_numbers(safe_text(uma_td)) if uma_td else []
            horse_number = int(uma_nums[0]) if uma_nums else 0
            if horse_number == 0:
                pop_nums = extract_numbers(safe_text(ninki_td)) if ninki_td else []
                if pop_nums:
                    horse_number = 0

            odds_spans = row.select("td.Odds span, td[class*='Odds'] span")
            win_odds = 0.0
            place_min = 0.0
            place_max = 0.0
            popularity = 0

            if odds_spans:
                win_text = safe_text(odds_spans[0])
                win_nums = re.findall(r"[\d.]+", win_text)
                win_odds = float(win_nums[0]) if win_nums else 0.0

            pop_el = row.select_one("td.Ninki span, td.Ninki")
            if pop_el:
                pn = extract_numbers(safe_text(pop_el))
                popularity = int(float(pn[0])) if pn else 0

            if horse_number > 0 or popularity > 0:
                result["entries"].append({
                    "horse_number": horse_number,
                    "win_odds": win_odds,
                    "place_odds_min": place_min,
                    "place_odds_max": place_max,
                    "popularity": popularity,
                })

        return result

    def parse_pair_odds_from_api(self, session, race_id: str) -> dict[str, Any]:
        """
        2連系オッズ (馬連/ワイド/馬単) を JSON API から取得・パースする。

        API レスポンス:
          type=4 (馬連): キー "4", 値 {"NNMM": [odds, "", popularity]}
          type=5 (ワイド): キー "5", 値 {"NNMM": [min_odds, max_odds, popularity]}
          type=6 (馬単): キー "6", 値 {"NNMM": [odds, "", popularity]}
          NN=1着馬番(2桁), MM=2着馬番(2桁)
          馬連/ワイド: NN < MM (非順序)、馬単: NN→MM (順序あり)

        出力:
        {
          "race_id": "...",
          "umaren": [{"pair": [1,2], "odds": 12.3, "popularity": 5}, ...],
          "wide":   [{"pair": [1,2], "odds_min": 3.1, "odds_max": 5.2, "popularity": 5}, ...],
          "umatan": [{"pair": [1,2], "odds": 25.1, "popularity": 10}, ...]
        }
        """
        result: dict[str, Any] = {
            "race_id": race_id, "umaren": [], "wide": [], "umatan": [],
        }

        API_MAP = [
            (self.UMAREN_API, "4", "umaren"),
            (self.WIDE_API,   "5", "wide"),
            (self.UMATAN_API, "6", "umatan"),
        ]

        for url_tpl, resp_key, out_key in API_MAP:
            try:
                resp = session.get(url_tpl.format(race_id=race_id), timeout=15)
                if resp.status_code != 200:
                    continue
                d = resp.json()
                if d.get("status") != "result":
                    continue
                raw = d.get("data", {}).get("odds", {}).get(resp_key, {})

                for combo_key, arr in raw.items():
                    if not isinstance(arr, list) or len(arr) < 1:
                        continue
                    if len(combo_key) != 4 or not combo_key.isdigit():
                        continue
                    h1 = int(combo_key[:2])
                    h2 = int(combo_key[2:])
                    if h1 == 0 or h2 == 0:
                        continue

                    pop = self._safe_int(arr[2]) if len(arr) > 2 else 0

                    if out_key == "wide":
                        result[out_key].append({
                            "pair": [h1, h2],
                            "odds_min": self._safe_float(arr[0]),
                            "odds_max": self._safe_float(arr[1]) if len(arr) > 1 else 0.0,
                            "popularity": pop,
                        })
                    else:
                        result[out_key].append({
                            "pair": [h1, h2],
                            "odds": self._safe_float(arr[0]),
                            "popularity": pop,
                        })
            except Exception:
                pass

        for key in ("umaren", "wide", "umatan"):
            result[key].sort(key=lambda e: (e["pair"][0], e["pair"][1]))

        return result

    @staticmethod
    def _safe_float(v) -> float:
        try:
            return float(str(v).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(v) -> int:
        try:
            return int(float(str(v).replace(",", "")))
        except (ValueError, TypeError):
            return 0


# ═══════════════════════════════════════════════════════
# 8. パドック評価パーサー [要ログイン]
#    URL: https://race.netkeiba.com/race/paddock.html?race_id={id}
# ═══════════════════════════════════════════════════════
class PaddockParser:
    """
    パドック評価ページをパースする。

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 1,
          "horse_name": "...",
          "horse_id": "...",
          "paddock_rank": "A",
          "paddock_comment": "...",
        }, ...
      ]
    }
    """

    _TABLE = SelectorChain("paddock_table", [
        "table.Paddock_Table",
        "table[class*='Paddock']",
        "div.PaddockBlock table",
        "table.RaceTable01",
    ])

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = self._TABLE.select_one(soup)
        if not table:
            for row in soup.select("tr.HorseList, div.Paddock_HorseList"):
                entry = self._parse_row_fallback(row)
                if entry:
                    result["entries"].append(entry)
            return result

        for row in table.select("tr"):
            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            cols = row.select("td")
            uma_tag = row.select_one("td[class*='Umaban'], td:first-child")
            uma_nums = extract_numbers(safe_text(uma_tag)) if uma_tag else []

            rank = ""
            comment = ""
            for td in cols:
                cls = " ".join(td.get("class", []))
                text = safe_text(td)
                if "Rank" in cls or "Hyouka" in cls or "評価" in cls:
                    rank = text
                if "Comment" in cls or len(text) > 10:
                    if not rank or text != rank:
                        comment = comment or text

            result["entries"].append({
                "horse_number": int(uma_nums[0]) if uma_nums else 0,
                "horse_name": safe_text(horse_link),
                "horse_id": extract_id_from_url(safe_attr(horse_link, "href")),
                "paddock_rank": rank,
                "paddock_comment": comment[:200],
            })

        return result

    def _parse_row_fallback(self, element) -> dict | None:
        horse_link = element.select_one("a[href*='/horse/']")
        if not horse_link:
            return None

        text = element.get_text(" ", strip=True)
        uma_nums = extract_numbers(safe_text(element.select_one("td, span")))

        rank_match = re.search(r"[A-E][+-]?|[◎○▲△×]", text)
        rank = rank_match.group(0) if rank_match else ""

        return {
            "horse_number": int(uma_nums[0]) if uma_nums else 0,
            "horse_name": safe_text(horse_link),
            "horse_id": extract_id_from_url(safe_attr(horse_link, "href")),
            "paddock_rank": rank,
            "paddock_comment": "",
        }


# ═══════════════════════════════════════════════════════
# 9. 偏差値バロメーター (走行データ) パーサー [要ログイン]
#    AJAX API: db.netkeiba.com/race/ajax_race_result_horse_laptime.html?id={race_id}&credit=1
#    Page URL: db.netkeiba.com/race/barometer/{race_id}/
#
#    バロメーターページは JS 動的レンダリングのため、
#    AJAX API で取得した HTML を直接パースする。
#    LapSummary_Table にタイム指数 (全体/スタート/追走/上がり) が含まれる。
# ═══════════════════════════════════════════════════════
class BarometerParser:
    """
    走行データ (タイム指数・ラップ) をパースする。

    AJAX API のレスポンス HTML に含まれる LapSummary_Table から:
    - 着順, 馬番, 馬名
    - タイム指数: 全体 (TotalIndex), スタート (StartIndex), 追走, 上がり

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 16,
          "horse_name": "セイウンハーデス",
          "horse_id": "...",
          "finish_order": 1,
          "index_total": 113,
          "index_start": 102,
          "index_chase": 109,
          "index_closing": 100
        }, ...
      ]
    }
    """

    AJAX_URL = "https://db.netkeiba.com/race/ajax_race_result_horse_laptime.html?id={race_id}&credit=1"

    def parse_from_api(self, session, race_id: str) -> dict[str, Any]:
        """HTTP session を使って AJAX API からバロメーターデータを取得する。"""
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        try:
            resp = session.get(
                self.AJAX_URL.format(race_id=race_id),
                timeout=15,
                headers={"Referer": f"https://db.netkeiba.com/race/barometer/{race_id}/"},
            )
            resp.encoding = "euc-jp"
            if resp.status_code != 200 or len(resp.text) < 100:
                return result
            return self.parse(resp.text, race_id=race_id)
        except Exception:
            return result

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        """AJAX レスポンス HTML をパースする。"""
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = soup.select_one("table.LapSummary_Table")
        if not table:
            return result

        for row in table.select("tr"):
            horse_info = row.select_one("td.Horse_Info")
            if not horse_info:
                continue

            horse_link = horse_info.select_one("a[href*='/horse/']")
            horse_name = safe_text(horse_link) if horse_link else safe_text(horse_info)
            horse_id = extract_id_from_url(safe_attr(horse_link, "href")) if horse_link else ""

            num_td = row.select_one("td.Num")
            num_nums = extract_numbers(safe_text(num_td)) if num_td else []
            horse_number = int(num_nums[0]) if num_nums else 0

            order_td = row.select_one("td.Result_Num")
            order_nums = extract_numbers(safe_text(order_td)) if order_td else []
            finish_order = int(order_nums[0]) if order_nums else 0

            index_cells = row.select("td.IndexMasterCell")
            indices = []
            for cell in index_cells:
                text = safe_text(cell)
                nums = extract_numbers(text)
                indices.append(int(float(nums[0])) if nums else 0)

            result["entries"].append({
                "horse_number": horse_number,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "finish_order": finish_order,
                "index_total": indices[0] if len(indices) > 0 else 0,
                "index_start": indices[1] if len(indices) > 1 else 0,
                "index_chase": indices[2] if len(indices) > 2 else 0,
                "index_closing": indices[3] if len(indices) > 3 else 0,
            })

        return result


# ═══════════════════════════════════════════════════════
# 10. 追い切り (調教詳細) パーサー
#     URL: https://race.netkeiba.com/race/oikiri.html?race_id={id}
# ═══════════════════════════════════════════════════════
class OikiriParser:
    """
    追い切り（調教）詳細ページをパースする。

    出力:
    {
      "race_id": "202606020611",
      "entries": [
        {
          "horse_number": 1,
          "horse_name": "...",
          "horse_id": "...",
          "training_date": "3/13",
          "course": "栗坂",
          "condition": "良",
          "rider": "助手",
          "lap_times": "52.2-38.0-25.2-12.6",
          "impression": "一杯",
          "evaluation": "A",
          "comment": "..."
        }, ...
      ]
    }
    """

    _TABLE = SelectorChain("oikiri_table", [
        "table.OikiriTable",
        "table[class*='Oikiri']",
        "table.Stable_Time",
        "table.RaceTable01",
    ])

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = self._TABLE.select_one(soup)
        if not table:
            return result

        for row in table.select("tr"):
            horse_link = row.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            cols = row.select("td")
            uma_tag = row.select_one("td[class*='Umaban'], td[class*='Waku']")
            uma_nums = extract_numbers(safe_text(uma_tag)) if uma_tag else []

            def col_text(idx: int) -> str:
                return safe_text(cols[idx]) if idx < len(cols) else ""

            eval_tag = row.select_one("td[class*='Rank'], td[class*='Hyouka'], span[class*='Rank']")
            evaluation = safe_text(eval_tag) if eval_tag else ""

            comment_tag = row.select_one("td[class*='Comment'], td.Tanpyo")
            comment = safe_text(comment_tag) if comment_tag else ""

            training_date = ""
            course = ""
            condition = ""
            rider = ""
            lap_times = ""
            impression = ""

            for i, td in enumerate(cols):
                text = safe_text(td)
                cls = " ".join(td.get("class", []))
                if re.match(r"\d+/\d+", text):
                    training_date = text
                elif "course" in cls.lower() or text in ("栗坂", "栗CW", "美坂", "美ウッド", "栗P", "美P"):
                    course = text
                elif text in ("良", "稍重", "稍", "重", "不良"):
                    condition = text
                elif re.match(r"\d+\.\d+-", text):
                    lap_times = text
                elif text in ("馬なり", "一杯", "強め", "仕掛け", "末強め", "G前仕掛け"):
                    impression = text

            result["entries"].append({
                "horse_number": int(uma_nums[0]) if uma_nums else 0,
                "horse_name": safe_text(horse_link),
                "horse_id": extract_id_from_url(safe_attr(horse_link, "href")),
                "training_date": training_date,
                "course": course,
                "condition": condition,
                "rider": rider,
                "lap_times": lap_times,
                "impression": impression,
                "evaluation": evaluation,
                "comment": comment[:200],
            })

        return result


# ═══════════════════════════════════════════════════════
# 11. 馬別調教タイムパーサー
#     URL: https://db.netkeiba.com/horse/training.html?id={horse_id}
# ═══════════════════════════════════════════════════════
class TrainingParser:
    """
    馬の調教タイムページ (1ページ分) をパースする。

    ページ構造:
      - レース出走ごとに <table class="race_table_01"> が1つ存在
      - <caption> にレース情報 (日付, 会場, レース名, 着順)
      - 各テーブル内に複数の調教行
      - ページネーション: ?page=N (10レース/ページ)

    出力:
    {
      "horse_id": "2023106216",
      "total_items": 18,
      "current_page": 1,
      "max_page": 2,
      "entries": [
        {
          "race_info": "2026/03/16 中京1R 3歳未勝利 結果 ： 3着",
          "date": "2026-03-11",
          "day_of_week": "水",
          "course": "栗坂",
          "track_condition": "良",
          "rider": "助手",
          "time_raw": "- 53.1 38.4 24.9 12.6",
          "lap_times": [53.1, 38.4, 24.9, 12.6],
          "position": "",
          "leg_color": "一杯",
          "evaluation": "動き上々",
          "rank": "B",
          "comment": ""
        },
        ...
      ]
    }
    """

    def parse(self, html: str, horse_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {
            "horse_id": horse_id,
            "total_items": 0,
            "current_page": 1,
            "max_page": 1,
            "entries": [],
        }

        self._parse_pager(soup, result)

        tables = soup.find_all("table", class_="race_table_01")
        if not tables:
            return result

        for table in tables:
            race_info = ""
            caption = table.find("caption")
            if caption:
                race_info = caption.get_text(separator=" ", strip=True)

            rows = table.find_all("tr")
            current_comment = ""

            for row in rows:
                tds = row.find_all("td")
                if not tds:
                    continue

                first_text = tds[0].get_text(strip=True)

                if first_text.startswith("[短評]") or first_text.startswith("【短評】"):
                    current_comment = (
                        first_text.replace("[短評]", "").replace("【短評】", "").strip()
                    )
                    continue

                # colspan付き行は短評/メモ行
                if tds[0].get("colspan"):
                    current_comment = first_text[:200]
                    continue

                date_match = re.match(r"(\d{4})/(\d{2})/(\d{2})\((.)\)", first_text)
                if not date_match:
                    continue

                entry = self._parse_training_row(
                    tds, date_match, current_comment, race_info,
                )
                if entry:
                    result["entries"].append(entry)
                    current_comment = ""

        return result

    def _parse_pager(self, soup: BeautifulSoup, result: dict) -> None:
        pager = soup.find("ul", class_="pager")
        if not pager:
            return

        pager_text = pager.get_text(strip=True)
        m = re.search(r"(\d+)件中(\d+)~(\d+)件目", pager_text)
        if m:
            total = int(m.group(1))
            start = int(m.group(2))
            result["total_items"] = total

            per_page = 10
            result["current_page"] = (start - 1) // per_page + 1
            result["max_page"] = (total + per_page - 1) // per_page

    def _parse_training_row(
        self,
        tds: list,
        date_match: re.Match,
        comment: str,
        race_info: str = "",
    ) -> dict[str, Any] | None:
        if len(tds) < 8:
            return None

        year, month, day, dow = (
            date_match.group(1), date_match.group(2),
            date_match.group(3), date_match.group(4),
        )
        date_str = f"{year}-{month}-{day}"

        course = safe_text(tds[1])
        track_cond = safe_text(tds[2])
        rider = safe_text(tds[3])

        time_raw, lap_times = self._parse_training_time(tds[4])

        position = safe_text(tds[5]) if len(tds) > 5 else ""
        leg_color = safe_text(tds[6]) if len(tds) > 6 else ""
        eval_text = safe_text(tds[7]) if len(tds) > 7 else ""
        rank = safe_text(tds[8]) if len(tds) > 8 else ""

        return {
            "race_info": race_info,
            "date": date_str,
            "day_of_week": dow,
            "course": course,
            "track_condition": track_cond,
            "rider": rider,
            "time_raw": time_raw,
            "lap_times": lap_times,
            "position": position,
            "leg_color": leg_color,
            "evaluation": eval_text,
            "rank": rank,
            "comment": comment,
        }

    @staticmethod
    def _parse_training_time(td) -> tuple[str, list[float]]:
        """
        <ul class="TrainingTimeDataList"> 内の <li> 要素から
        各ラップタイムを抽出する。
        先頭の '-' は全体タイム省略マーカーなので除外。
        """
        time_list = td.find("ul", class_="TrainingTimeDataList")
        if time_list:
            items = time_list.find_all("li")
            times: list[float] = []
            parts: list[str] = []
            for li in items:
                txt = li.get_text(strip=True)
                parts.append(txt)
                try:
                    times.append(float(txt))
                except ValueError:
                    pass
            return " ".join(parts), times

        raw = td.get_text(separator=" ", strip=True)
        times = []
        for tok in raw.split():
            try:
                times.append(float(tok))
            except ValueError:
                pass
        return raw, times


# ═══════════════════════════════════════════════════════
# 12. TrainerCommentParser — 厩舎コメント
# ═══════════════════════════════════════════════════════

class TrainerCommentParser:
    """
    厩舎コメントページをパースする。

    URL: race.netkeiba.com/race/comment.html?race_id={id}
    要ログイン (プレミアム)

    出力:
    {
      "race_id": "202507020211",
      "entries": [
        {
          "bracket_number": 1,
          "horse_number": 1,
          "horse_name": "アスクドゥポルテ",
          "horse_id": "2020101025",
          "comment": "真面目な性格で...",
          "evaluation": "B",
          "trainer_name": "梅田師",
          "questioner": "堀尾"
        }, ...
      ]
    }
    """

    _TABLE = SelectorChain("comment_table", [
        "table.Stable_Comment",  # 2021年以降の新構造
        "table.Comment_Table",   # 旧構造
        "table[class*='Comment']",
        "div.CommentBlock table",
        "table.RaceTable01",
        "table.Shutuba_Table",
    ])

    _RE_TRAINER = re.compile(r"〈([^〉]+師)〉\s*$")
    _RE_QUESTIONER = re.compile(r"〈([^〉]+)〉")

    def parse(self, html: str, race_id: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {"race_id": race_id, "entries": []}

        table = self._TABLE.select_one(soup)
        rows = table.select("tr") if table else []

        if not rows:
            rows = soup.select("tr.HorseList, div.CommentList tr, div.RaceTable tr")

        for row in rows:
            entry = self._parse_row(row)
            if entry:
                result["entries"].append(entry)

        if not result["entries"]:
            result["entries"] = self._fallback_div_parse(soup)

        logger.info("厩舎コメント: %s — %d件", race_id, len(result["entries"]))
        return result

    def _parse_row(self, row) -> dict[str, Any] | None:
        horse_link = row.select_one("a[href*='/horse/']")
        if not horse_link:
            return None

        horse_name = safe_text(horse_link)
        horse_id = extract_id_from_url(safe_attr(horse_link, "href"))

        cols = row.select("td")
        if len(cols) < 3:
            return None

        bracket = 0
        horse_num = 0
        comment = ""
        evaluation = ""

        for i, td in enumerate(cols):
            cls_list = td.get("class", [])
            cls = " ".join(cls_list)
            text = safe_text(td)

            # 枠番（Waku1, Waku2, ... Waku8、または1番目のカラム）
            if any(c.startswith("Waku") and len(c) > 4 and c[4:].isdigit() for c in cls_list):
                nums = extract_numbers(text)
                if nums:
                    bracket = int(nums[0])
            elif i == 0 and not bracket:
                nums = extract_numbers(text)
                if nums and 1 <= int(nums[0]) <= 8:
                    bracket = int(nums[0])
            # 馬番（class="Waku"のみ、Umaban、または2番目のカラム）
            elif "Waku" in cls_list and not any(c.startswith("Waku") and len(c) > 4 for c in cls_list):
                nums = extract_numbers(text)
                if nums and not horse_num:
                    horse_num = int(nums[0])
            elif "Umaban" in cls or (i == 1 and not horse_num):
                nums = extract_numbers(text)
                if nums:
                    horse_num = int(nums[0])
            # コメント（txt_l, Comment, comment など）
            elif any(c in cls.lower() for c in ("comment", "txt_l", "text")):
                comment = text
            # 評価（Hyouka, Rank など）
            elif "Hyouka" in cls or "Rank" in cls or "評価" in text:
                if len(text) <= 3:
                    evaluation = text

        if not comment:
            for td in cols:
                text = safe_text(td)
                if len(text) > 15 and text != horse_name:
                    comment = text
                    break

        if not bracket and cols:
            nums = extract_numbers(safe_text(cols[0]))
            if nums and int(nums[0]) <= 8:
                bracket = int(nums[0])

        if not horse_num and len(cols) > 1:
            nums = extract_numbers(safe_text(cols[1]))
            if nums and int(nums[0]) <= 18:
                horse_num = int(nums[0])

        trainer_name = ""
        questioner = ""
        if comment:
            m_trainer = self._RE_TRAINER.search(comment)
            if m_trainer:
                trainer_name = m_trainer.group(1)
            m_all = self._RE_QUESTIONER.findall(comment)
            if m_all:
                for name in m_all:
                    if not name.endswith("師"):
                        questioner = name
                        break

        if not evaluation:
            for td in cols:
                cls = " ".join(td.get("class", []))
                text = safe_text(td).strip()
                if text and len(text) <= 3 and text not in (horse_name, str(bracket), str(horse_num)):
                    if any(c in cls.lower() for c in ("rank", "hyou", "eval")):
                        evaluation = text

        return {
            "bracket_number": bracket,
            "horse_number": horse_num,
            "horse_name": horse_name,
            "horse_id": horse_id,
            "comment": comment,
            "evaluation": evaluation,
            "trainer_name": trainer_name,
            "questioner": questioner,
        }

    def _fallback_div_parse(self, soup) -> list[dict[str, Any]]:
        """div ベースのフォールバック構造に対応。"""
        entries = []
        for block in soup.select("div.CommentItem, div.Comment_HorseList, li.Comment_Item"):
            horse_link = block.select_one("a[href*='/horse/']")
            if not horse_link:
                continue

            horse_name = safe_text(horse_link)
            horse_id = extract_id_from_url(safe_attr(horse_link, "href"))

            text_el = block.select_one(".Comment_Text, .CommentText, p")
            comment = safe_text(text_el) if text_el else ""

            num_el = block.select_one(".Umaban, .HorseNum")
            nums = extract_numbers(safe_text(num_el)) if num_el else []

            trainer_name = ""
            questioner = ""
            if comment:
                m = self._RE_TRAINER.search(comment)
                if m:
                    trainer_name = m.group(1)
                m_all = self._RE_QUESTIONER.findall(comment)
                for name in m_all:
                    if not name.endswith("師"):
                        questioner = name
                        break

            entries.append({
                "bracket_number": 0,
                "horse_number": int(nums[0]) if nums else 0,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "comment": comment,
                "evaluation": "",
                "trainer_name": trainer_name,
                "questioner": questioner,
            })
        return entries
