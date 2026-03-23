"""
JRA公式 クッション値・含水率 スクレイパー

https://www.jra.go.jp/keiba/baba/archive/ からPDFを取得し、
クッション値・含水率データを構造化JSONに変換・保存する。
PDF はローカルに置かず一時ファイルにDLしてパース後に削除する（GCS/ライブ運用と重複しないキャッシュを避ける）。

対応年度: 2020〜2026 (クッション値は2020年9月11日から公表)
PDF形式:
  - Format A (2025〜): 1テーブル/ページ, 日別行
  - Format B (2020〜2024): 開催日ペアごとにクッション値テーブル+含水率テーブル

Usage:
  python -m scraper.jra_cushion --years 2024 2025 2026
  python -m scraper.jra_cushion --all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("scraper.jra_cushion")

BASE_URL = "https://www.jra.go.jp/keiba/baba/archive"

VENUE_MAP = {
    "sapporo": {"code": "01", "name": "札幌"},
    "hakodate": {"code": "02", "name": "函館"},
    "fukushima": {"code": "03", "name": "福島"},
    "niigata": {"code": "04", "name": "新潟"},
    "tokyo": {"code": "05", "name": "東京"},
    "nakayama": {"code": "06", "name": "中山"},
    "chukyo": {"code": "07", "name": "中京"},
    "kyoto": {"code": "08", "name": "京都"},
    "hanshin": {"code": "09", "name": "阪神"},
    "kokura": {"code": "10", "name": "小倉"},
}

VENUE_CODE_TO_NAME = {v["code"]: v["name"] for v in VENUE_MAP.values()}
VENUE_NAME_TO_CODE = {v["name"]: v["code"] for v in VENUE_MAP.values()}


class JRACushionScraper:
    """JRA公式PDFからクッション値・含水率を取得・保存する。"""

    def __init__(self, output_dir: str = "data/jra_baba"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; keiba-ml/1.0)",
        })

    # ── Step 1: HTMLページからPDFリンクを収集 ──

    def discover_pdf_links(self, year: int) -> list[dict]:
        """年度ページをスクレイプしてPDFリンク一覧を返す。"""
        if year == 2026:
            url = f"{BASE_URL}/"
        else:
            url = f"{BASE_URL}/{year}.html"

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("ページ取得失敗: %s => %s", url, e)
            return []

        html = resp.text
        pattern = rf'{year}pdf/(\w+?)(\d{{2}})\.pdf'
        matches = re.findall(pattern, html)

        links = []
        seen = set()
        for venue_en, kai_str in matches:
            key = f"{venue_en}{kai_str}"
            if key in seen:
                continue
            seen.add(key)
            venue_info = VENUE_MAP.get(venue_en)
            if not venue_info:
                continue
            pdf_url = f"{BASE_URL}/{year}pdf/{venue_en}{kai_str}.pdf"
            links.append({
                "year": year,
                "venue_en": venue_en,
                "venue_code": venue_info["code"],
                "venue_name": venue_info["name"],
                "kai": int(kai_str),
                "url": pdf_url,
                "filename": f"{year}_{venue_en}{kai_str}.pdf",
            })
        logger.info("%d年: %d個のPDFリンク発見", year, len(links))
        return links

    # ── Step 2: PDFダウンロード（一時ファイルのみ・呼び出し側で削除） ──

    def download_pdf(self, link: dict, force: bool = False) -> Path | None:
        """PDF を一時ファイルに保存してパスを返す。force は互換用（未使用）。"""
        del force  # ローカルキャッシュ廃止のため常に取得
        path: Path | None = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="jra_cushion_")
            os.close(fd)
            path = Path(tmp)
            resp = self.session.get(link["url"], timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            logger.info("  DL: %s (%d bytes)", link["filename"], len(resp.content))
            return path
        except Exception as e:
            logger.error("  DL失敗: %s => %s", link["url"], e)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

    # ── Step 3: PDFパース ──

    def parse_pdf(self, path: Path, link: dict) -> list[dict]:
        """PDFを解析し、日別のクッション値・含水率レコードを返す。"""
        try:
            import fitz
        except ImportError:
            logger.error("pymupdf が必要です: pip install pymupdf")
            return []

        doc = fitz.open(str(path))
        all_records: list[dict] = []

        for page in doc:
            table_finder = page.find_tables()
            tables = list(table_finder.tables) if table_finder else []
            if not tables:
                continue

            if self._is_format_a(tables):
                records = self._parse_format_a(tables, link)
            else:
                records = self._parse_format_b(tables, link, page)
            all_records.extend(records)

        doc.close()
        return all_records

    @staticmethod
    def _is_format_a(tables) -> bool:
        """Format A: 最初のテーブルに9列以上 (開催日次, 月日, 曜日, コース, ...)"""
        for table in tables:
            if table.col_count >= 9:
                df = table.to_pandas()
                cols = [str(c) for c in df.columns]
                joined = " ".join(cols)
                if "開催日次" in joined or "測定月日" in joined:
                    return True
        return False

    def _parse_format_a(self, tables, link: dict) -> list[dict]:
        """Format A (2025〜): 1つの大テーブル, 日別行。"""
        records = []
        target_table = None
        for table in tables:
            if table.col_count >= 9:
                df = table.to_pandas()
                cols_joined = " ".join(str(c) for c in df.columns)
                if "開催日次" in cols_joined or "測定月日" in cols_joined:
                    target_table = df
                    break

        if target_table is None:
            return records

        df = target_table
        for _, row in df.iterrows():
            vals = [str(v).strip() if v is not None else "" for v in row.values]
            if len(vals) < 9:
                continue

            race_day_str = vals[0]
            date_str = vals[1]
            weekday = vals[2]
            course_pos = vals[3]
            cushion_val = vals[5]
            moisture_turf = vals[7]
            moisture_dirt = vals[8]

            if not date_str or "月" not in date_str:
                continue
            if weekday not in ("金曜日", "土曜日", "日曜日", "月曜日", "火曜日"):
                continue

            try:
                cv = float(cushion_val)
            except (ValueError, TypeError):
                continue

            full_date = self._resolve_date(date_str, link["year"])
            is_race_day = "第" in race_day_str
            race_day_num = 0
            if is_race_day:
                m = re.search(r"(\d+)", race_day_str)
                if m:
                    race_day_num = int(m.group(1))

            turf_goal, turf_4c = self._split_moisture(moisture_turf)
            dirt_goal, dirt_4c = self._split_moisture(moisture_dirt)

            records.append({
                "date": full_date,
                "weekday": weekday,
                "year": link["year"],
                "venue_code": link["venue_code"],
                "venue_name": link["venue_name"],
                "kai": link["kai"],
                "race_day": race_day_num,
                "is_race_day": is_race_day,
                "course_position": course_pos,
                "cushion_value": cv,
                "turf_moisture_goal": turf_goal,
                "turf_moisture_4corner": turf_4c,
                "dirt_moisture_goal": dirt_goal,
                "dirt_moisture_4corner": dirt_4c,
            })

        return records

    def _parse_format_b(self, tables, link: dict, page) -> list[dict]:
        """Format B (2020〜2024): 開催日ペアごとにクッション値+含水率のテーブルペア。"""
        records = []
        text = page.get_text()

        date_ranges = self._extract_date_ranges(text)

        i = 0
        pair_idx = 0
        while i < len(tables):
            table = tables[i]
            df = table.to_pandas()
            cols_text = " ".join(str(v) for v in df.values.flatten() if v)

            if "クッション値" in cols_text:
                cushion_df = df
                moisture_df = None
                if i + 1 < len(tables):
                    next_df = tables[i + 1].to_pandas()
                    next_text = " ".join(str(v) for v in next_df.values.flatten() if v)
                    if "含水率" in next_text or "ゴール前" in next_text:
                        moisture_df = next_df
                        i += 1

                day_records = self._extract_format_b_pair(
                    cushion_df, moisture_df, link, pair_idx, date_ranges,
                )
                records.extend(day_records)
                pair_idx += 1

            elif "含水率" in cols_text or "ゴール前" in cols_text:
                cushion_df = None
                if i + 1 < len(tables):
                    next_df = tables[i + 1].to_pandas()
                    next_text = " ".join(str(v) for v in next_df.values.flatten() if v)
                    if "クッション値" in next_text:
                        cushion_df = next_df
                        i += 1

                if cushion_df is not None:
                    day_records = self._extract_format_b_pair(
                        cushion_df, df, link, pair_idx, date_ranges,
                    )
                    records.extend(day_records)
                    pair_idx += 1
            i += 1

        return records

    def _extract_format_b_pair(
        self, cushion_df, moisture_df, link: dict,
        pair_idx: int, date_ranges: list,
    ) -> list[dict]:
        records = []

        day_cols = []
        if cushion_df is not None:
            for c in cushion_df.columns:
                cs = str(c)
                if "曜日" in cs:
                    day_cols.append(cs)
            if not day_cols:
                for c in cushion_df.columns:
                    cs = str(c)
                    if cs.startswith("Col") and cs != "Col0":
                        day_cols.append(cs)

            for _, row in cushion_df.iterrows():
                vals = [str(v).strip() if v else "" for v in row.values]
                if "クッション値" not in " ".join(vals):
                    continue
                for j, dc in enumerate(day_cols):
                    try:
                        cv = float(row[dc])
                    except (ValueError, TypeError):
                        continue
                    day_label = dc.replace("曜日", "")
                    weekday_map = {"金": "金曜日", "土": "土曜日", "日": "日曜日", "月": "月曜日"}
                    weekday = weekday_map.get(day_label, dc)

                    rec = {
                        "year": link["year"],
                        "venue_code": link["venue_code"],
                        "venue_name": link["venue_name"],
                        "kai": link["kai"],
                        "pair_index": pair_idx,
                        "weekday": weekday,
                        "cushion_value": cv,
                        "turf_moisture_goal": None,
                        "turf_moisture_4corner": None,
                        "dirt_moisture_goal": None,
                        "dirt_moisture_4corner": None,
                        "is_race_day": weekday != "金曜日",
                        "race_day": 0,
                        "date": "",
                        "course_position": "",
                    }
                    records.append(rec)

        if moisture_df is not None and records:
            day_cols_m = [c for c in moisture_df.columns if "曜日" in str(c)]
            if not day_cols_m:
                day_cols_m = [c for c in moisture_df.columns
                              if str(c).startswith("Col") and str(c) != "Col0"]
                if "場所" in moisture_df.columns:
                    day_cols_m = [c for c in moisture_df.columns
                                  if c not in ("Col0", "場所") and str(c) != "None"]

            for _, row in moisture_df.iterrows():
                vals = [str(v).strip() if v else "" for v in row.values]
                row_text = " ".join(vals)

                section = ""
                location = ""
                if "芝" in row_text and "含水率" in row_text:
                    section = "turf"
                elif "ダート" in row_text and "含水率" in row_text:
                    section = "dirt"

                for v in vals:
                    if "ゴール前" in v:
                        location = "goal"
                        break
                    if "コーナー" in v:
                        location = "4corner"
                        break

                if not location:
                    continue
                if not section:
                    if len(records) > 0:
                        last_section = ""
                        for r in reversed(records):
                            if r.get("_last_section"):
                                last_section = r["_last_section"]
                                break
                        section = "dirt" if last_section == "turf" else "turf"

                for j, dc in enumerate(day_cols_m):
                    if j >= len(records):
                        break
                    try:
                        val = float(str(row[dc]).strip())
                    except (ValueError, TypeError):
                        continue
                    rec_idx = j
                    while rec_idx < len(records):
                        key = f"{section}_moisture_{location}"
                        if key in records[rec_idx] and records[rec_idx][key] is None:
                            records[rec_idx][key] = val
                            records[rec_idx]["_last_section"] = section
                            break
                        rec_idx += len(day_cols_m)

        if date_ranges and pair_idx < len(date_ranges):
            dr = date_ranges[pair_idx]
            import datetime
            try:
                days_in_pair = []
                d = datetime.date(dr["year"], dr["start_month"], dr["start_day"])
                end_d = datetime.date(dr["year"], dr["end_month"], dr["end_day"])
                while d <= end_d:
                    days_in_pair.append(d.strftime("%Y-%m-%d"))
                    d += datetime.timedelta(days=1)

                for i, rec in enumerate(records):
                    if i < len(days_in_pair):
                        rec["date"] = days_in_pair[i]
            except (ValueError, OverflowError):
                pass

        for r in records:
            r.pop("_last_section", None)

        return records

    # ── ユーティリティ ──

    @staticmethod
    def _resolve_date(date_str: str, year: int) -> str:
        m = re.search(r'(\d+)月\s*(\d+)日', date_str)
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            return f"{year}-{month:02d}-{day:02d}"
        return ""

    @staticmethod
    def _extract_date_ranges(text: str) -> list[dict]:
        """テキストから開催日の日付範囲を抽出する。
        例: '第１日・第２日（2024年7月19日～21日）' → {year:2024, start_month:7, start_day:19, end_month:7, end_day:21}
        """
        ranges = []
        pattern = r'[（(](\d{4})年(\d+)月(\d+)日[〜～](?:(\d+)月)?(\d+)日[）)]'
        for m in re.finditer(pattern, text):
            year_val = int(m.group(1))
            start_month = int(m.group(2))
            start_day = int(m.group(3))
            end_month = int(m.group(4)) if m.group(4) else start_month
            end_day = int(m.group(5))
            ranges.append({
                "year": year_val,
                "start_month": start_month,
                "start_day": start_day,
                "end_month": end_month,
                "end_day": end_day,
            })
        return ranges

    @staticmethod
    def _split_moisture(val: str) -> tuple[float | None, float | None]:
        parts = val.split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        if len(parts) == 1:
            try:
                return float(parts[0]), None
            except ValueError:
                pass
        return None, None

    # ── Step 4: 保存 ──

    def save_records(self, records: list[dict], year: int | None = None):
        if not records:
            return

        all_path = self.output_dir / "cushion_values.json"
        existing: list[dict] = []
        if all_path.exists():
            try:
                existing = json.loads(all_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        existing_keys = set()
        for r in existing:
            key = f"{r.get('date')}_{r.get('venue_code')}_{r.get('weekday')}"
            existing_keys.add(key)

        added = 0
        for r in records:
            key = f"{r.get('date')}_{r.get('venue_code')}_{r.get('weekday')}"
            if key not in existing_keys:
                existing.append(r)
                existing_keys.add(key)
                added += 1

        existing.sort(key=lambda x: (x.get("date", ""), x.get("venue_code", "")))
        all_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("保存: %s (計%d件, 新規%d件)", all_path, len(existing), added)

        if year:
            year_path = self.output_dir / f"cushion_{year}.json"
            year_records = [r for r in existing if r.get("year") == year]
            year_path.write_text(
                json.dumps(year_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("  年度別: %s (%d件)", year_path, len(year_records))

    # ── メイン処理 ──

    def scrape(
        self,
        years: list[int] | None = None,
        force_download: bool = False,
    ) -> dict[str, Any]:
        if years is None:
            years = list(range(2020, 2027))

        total_records: list[dict] = []
        stats = {"years": {}, "total_pdfs": 0, "total_records": 0}

        for year in years:
            logger.info("=" * 50)
            logger.info("  %d年のクッション値・含水率データ取得", year)
            logger.info("=" * 50)

            links = self.discover_pdf_links(year)
            year_records: list[dict] = []

            for link in links:
                time.sleep(0.5)

                pdf_path = self.download_pdf(link, force=force_download)
                if not pdf_path:
                    continue
                try:
                    records = self.parse_pdf(pdf_path, link)
                finally:
                    try:
                        pdf_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                logger.info("    %s 第%d回: %d レコード",
                            link["venue_name"], link["kai"], len(records))
                year_records.extend(records)

            total_records.extend(year_records)
            stats["years"][str(year)] = {
                "pdfs": len(links),
                "records": len(year_records),
            }

            self.save_records(year_records, year)
            logger.info("  %d年: %d PDF, %d レコード", year, len(links), len(year_records))

        stats["total_pdfs"] = sum(s["pdfs"] for s in stats["years"].values())
        stats["total_records"] = len(total_records)

        self.save_records(total_records)
        logger.info("完了: 計 %d レコード", len(total_records))
        return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="JRA クッション値・含水率スクレイパー")
    parser.add_argument("--years", nargs="+", type=int, help="対象年度 (例: 2024 2025)")
    parser.add_argument("--all", action="store_true", help="全年度 (2020-2026)")
    parser.add_argument("--force", action="store_true", help="PDFを再ダウンロード")
    args = parser.parse_args()

    years = args.years
    if args.all:
        years = list(range(2020, 2027))
    if not years:
        years = [2025, 2026]

    scraper = JRACushionScraper()
    stats = scraper.scrape(years=years, force_download=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
