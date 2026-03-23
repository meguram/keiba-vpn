"""
ページ構造モニター

netkeiba.com の HTML ページ構造に変化がないかを定期的にチェックし、
差分が検出された場合にパーサーへの影響を判定して再パースを実行する。

フロー:
  1. 各カテゴリの最新 HTML をフェッチ (または HtmlArchive から取得)
  2. SelectorChain 群を試行し、構造フィンガープリントを生成
  3. 前回のフィンガープリントと比較
  4. 差分があれば影響判定 → 再パース実行

フィンガープリント JSON:
  data/meta/structure/{category}.json

構造バージョン管理:
  data/meta/structure/versions.json
  破壊的構造変更 (CRITICAL) が検出されるとバージョンを bump。
  HybridStorage.is_fresh() は scraped_at と versions.json の
  changed_at を比較し、データが現在のページ構造に準拠しているかを判定する。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger("scraper.structure_monitor")

FINGERPRINT_DIR = Path("data/meta/structure")
VERSIONS_PATH = FINGERPRINT_DIR / "versions.json"

CATEGORY_PROBES: dict[str, dict[str, Any]] = {
    "race_result": {
        "sample_url": "https://db.netkeiba.com/race/{race_id}/",
        "id_type": "race",
        "selectors": {
            "result_table": [
                "table.race_table_01",
                "table[class*='race_table']",
                "table[summary*='レース結果']",
                "#contents_liquid table",
            ],
            "race_name": [
                "dl.racedata h1",
                ".racedata h1",
                "h1[class*='RaceName']",
                "#main h1",
            ],
            "race_data": [
                "dl.racedata dd",
                ".racedata dd",
                "div.RaceData01",
                "p[class*='RaceData']",
            ],
            "pay_table": [
                "table.pay_table_01",
                "table[class*='pay_table']",
            ],
        },
        "row_selector": "table.race_table_01 tr",
        "column_check": "table.race_table_01 tr:first-child td, table.race_table_01 tr:first-child th",
    },
    "race_shutuba": {
        "sample_url": "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "shutuba_table": [
                "table.Shutuba_Table",
                "table.ShutubaTable",
                "table.RaceTable01",
                "table[class*='shutuba']",
                "table[class*='Shutuba']",
            ],
            "entry_row": [
                "tr.HorseList",
                "tr[class*='HorseList']",
            ],
            "race_name": [
                "h1.RaceName",
                "div.RaceName",
                ".RaceName",
            ],
            "race_data1": [
                "div.RaceData01",
                "span.RaceData01",
            ],
        },
        "row_selector": "tr.HorseList",
        "column_check": "tr.HorseList td",
    },
    "race_index": {
        "sample_url": "https://race.netkeiba.com/race/speed.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "speed_table": [
                "table.SpeedIndex_Table",
                "table[class*='SpeedIndex']",
                "table.RaceTable01.ShutubaTable",
            ],
        },
        "row_selector": "tr.HorseList",
        "column_check": "tr.HorseList td",
    },
    "race_shutuba_past": {
        "sample_url": "https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "past_table": [
                "table.Shutuba_Past5_Table",
                "table[class*='Shutuba_Past']",
                "table.Shutuba_Table",
            ],
        },
        "row_selector": "tr.HorseList",
        "column_check": "tr.HorseList td",
    },
    "race_odds": {
        "sample_url": "https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "odds_table": [
                "table.RaceOdds_HorseList_Table",
                "table[class*='Odds']",
                "table#Ninki",
            ],
        },
        "optional_selectors": ["odds_table"],
        "row_selector": "table.RaceOdds_HorseList_Table tr",
        "column_check": None,
        "note": "オッズは JS 動的レンダリング。実データは JSON API (api_get_jra_odds.html) から取得",
    },
    "race_paddock": {
        "sample_url": "https://race.netkeiba.com/race/paddock.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "paddock_table": [
                "table.Paddock_Table",
                "table[class*='Paddock']",
                "div.PaddockBlock table",
                "table.RaceTable01",
            ],
        },
        "row_selector": "tr.HorseList, div.Paddock_HorseList",
        "column_check": None,
    },
    "race_barometer": {
        "sample_url": "https://db.netkeiba.com/race/ajax_race_result_horse_laptime.html?id={race_id}&credit=1",
        "id_type": "race",
        "encoding": "euc-jp",
        "selectors": {
            "lap_summary_table": [
                "table.LapSummary_Table",
            ],
            "index_cells": [
                "td.IndexMasterCell",
            ],
            "horse_info": [
                "td.Horse_Info.Horse_Link",
                "td.Horse_Info",
            ],
        },
        "optional_selectors": ["index_cells"],
        "row_selector": "table.LapSummary_Table tr",
        "column_check": None,
        "delayed_publish": True,
        "min_age_days": 14,
        "not_published_marker": "公開予定",
        "note": "バロメーターは AJAX API で取得。データはレース翌週金曜18時頃に公開",
    },
    "race_oikiri": {
        "sample_url": "https://race.netkeiba.com/race/oikiri.html?race_id={race_id}",
        "id_type": "race",
        "selectors": {
            "oikiri_table": [
                "table.OikiriTable",
                "table[class*='Oikiri']",
                "table.Stable_Time",
                "table.RaceTable01",
            ],
        },
        "row_selector": None,
        "column_check": None,
    },
    "horse_profile": {
        "sample_url": "https://db.netkeiba.com/horse/{horse_id}/",
        "id_type": "horse",
        "archive_category": "horse_profile",
        "selectors": {
            "horse_name": [
                "div.horse_title h1",
                "div.db_head_name h1",
                "h1[class*='horse']",
                "#db_main_box h1",
            ],
            "prof_table": [
                "table.db_prof_table",
                "table[class*='prof']",
            ],
        },
        "row_selector": None,
        "column_check": None,
    },
    "horse_ped": {
        "sample_url": "https://db.netkeiba.com/horse/ped/{horse_id}/",
        "id_type": "horse",
        "archive_category": "horse_ped",
        "selectors": {
            "blood_table": [
                "table.blood_table",
                "table[class*='blood']",
                "table[summary*='血統']",
            ],
        },
        "row_selector": "table.blood_table tr",
        "column_check": None,
    },
    "horse_result_html": {
        "sample_url": "https://db.netkeiba.com/horse/result/{horse_id}/",
        "id_type": "horse",
        "archive_category": "horse_result_html",
        "selectors": {
            "race_table": [
                "table.db_h_race_results",
                "table[class*='race_results']",
                "table.nk_tb_common",
            ],
        },
        "row_selector": "table.db_h_race_results tr",
        "column_check": "table.db_h_race_results tr:nth-child(2) td",
    },
}


SMARTRC_API_PROBES: dict[str, dict[str, Any]] = {
    "smartrc_days": {
        "endpoint": "days",
        "description": "開催日カレンダー",
        "required_keys": ["rdate", "races"],
    },
    "smartrc_runners": {
        "endpoint": "runners",
        "description": "出走馬データ (評価/テン1F/推定人気/CR/過去5走)",
        "required_keys": ["rcode", "uno", "hname", "hcode"],
    },
    "smartrc_horses": {
        "endpoint": "horses",
        "description": "馬情報 (血統・系統・色分け)",
        "required_keys": ["hcode", "hname"],
    },
    "smartrc_fullresults": {
        "endpoint": "fullresults",
        "description": "全戦績 (着順/タイム/上り3F/テン1F等)",
        "required_keys": ["rcode", "hcode", "rank"],
    },
}


@dataclass
class SelectorProbeResult:
    selector: str
    matched: bool
    index: int
    element_count: int = 0
    optional: bool = False


@dataclass
class StructureFingerprint:
    category: str
    timestamp: str
    sample_id: str
    selector_results: dict[str, SelectorProbeResult] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    column_classes: list[str] = field(default_factory=list)


@dataclass
class StructureDiff:
    category: str
    broken_selectors: list[str] = field(default_factory=list)
    fallback_selectors: list[str] = field(default_factory=list)
    column_change: bool = False
    row_count_changed: bool = False
    old_fingerprint: dict | None = None
    new_fingerprint: dict | None = None

    @property
    def has_breaking_change(self) -> bool:
        return len(self.broken_selectors) > 0 or self.column_change

    @property
    def has_any_change(self) -> bool:
        return (self.has_breaking_change
                or len(self.fallback_selectors) > 0
                or self.row_count_changed)

    @property
    def severity(self) -> str:
        if self.has_breaking_change:
            return "CRITICAL"
        if self.fallback_selectors:
            return "WARNING"
        return "OK"


class StructureMonitor:
    """
    netkeiba ページ構造のモニタリングと差分検知。
    """

    def __init__(self, base_dir: str = "."):
        self._base_dir = Path(base_dir)
        self._fp_dir = self._base_dir / FINGERPRINT_DIR
        self._fp_dir.mkdir(parents=True, exist_ok=True)

    # ── 構造バージョン管理 ─────────────────────────────

    def load_versions(self) -> dict[str, dict]:
        """versions.json を読み込む。{category: {version, changed_at_unix, changed_at}}"""
        vp = self._base_dir / VERSIONS_PATH
        if not vp.exists():
            return {}
        try:
            return json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_versions(self, versions: dict[str, dict]):
        vp = self._base_dir / VERSIONS_PATH
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")

    def bump_version(self, category: str):
        """指定カテゴリの構造バージョンを bump する。"""
        versions = self.load_versions()
        now = time.time()
        cur = versions.get(category, {"version": 0, "changed_at_unix": 0, "changed_at": ""})
        cur["version"] = cur.get("version", 0) + 1
        cur["changed_at_unix"] = now
        cur["changed_at"] = datetime.fromtimestamp(now).isoformat()
        versions[category] = cur
        self.save_versions(versions)
        logger.info("[%s] 構造バージョン bump → v%d (%s)", category, cur["version"], cur["changed_at"])

    def confirm_version(self, category: str):
        """構造変更なしを確認。初回なら version=1 で登録する。"""
        versions = self.load_versions()
        if category not in versions:
            now = time.time()
            versions[category] = {
                "version": 1,
                "changed_at_unix": now,
                "changed_at": datetime.fromtimestamp(now).isoformat(),
            }
            self.save_versions(versions)
            logger.info("[%s] 構造バージョン初期化 → v1", category)

    def get_version(self, category: str) -> dict | None:
        """指定カテゴリの構造バージョン情報を返す。"""
        return self.load_versions().get(category)

    # ── フィンガープリント ─────────────────────────────

    def probe_html(self, html: str, category: str, sample_id: str) -> StructureFingerprint:
        """HTML を解析して構造フィンガープリントを生成する。"""
        soup = BeautifulSoup(html, "html.parser")
        probe_cfg = CATEGORY_PROBES.get(category, {})
        optional_names = set(probe_cfg.get("optional_selectors", []))
        fp = StructureFingerprint(
            category=category,
            timestamp=datetime.now().isoformat(),
            sample_id=sample_id,
        )

        for name, selectors in probe_cfg.get("selectors", {}).items():
            is_optional = name in optional_names
            for i, sel in enumerate(selectors):
                try:
                    results = soup.select(sel)
                except Exception:
                    results = []
                if results:
                    fp.selector_results[name] = SelectorProbeResult(
                        selector=sel, matched=True, index=i,
                        element_count=len(results), optional=is_optional,
                    )
                    break
            else:
                fp.selector_results[name] = SelectorProbeResult(
                    selector=selectors[0] if selectors else "",
                    matched=False, index=-1, optional=is_optional,
                )

        row_sel = probe_cfg.get("row_selector")
        if row_sel:
            try:
                fp.row_count = len(soup.select(row_sel))
            except Exception:
                pass

        col_sel = probe_cfg.get("column_check")
        if col_sel:
            try:
                cols = soup.select(col_sel)
                fp.column_count = len(cols)
                fp.column_classes = []
                for c in cols:
                    cls = " ".join(c.get("class", []))
                    fp.column_classes.append(cls)
            except Exception:
                pass

        return fp

    def save_fingerprint(self, fp: StructureFingerprint):
        """フィンガープリントをファイルに保存する。"""
        path = self._fp_dir / f"{fp.category}.json"
        data = asdict(fp)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("フィンガープリント保存: %s", path)

    def load_fingerprint(self, category: str) -> dict | None:
        """保存済みフィンガープリントを読み出す。"""
        path = self._fp_dir / f"{category}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def compare(self, old: dict, new: StructureFingerprint) -> StructureDiff:
        """前回と今回のフィンガープリントを比較して差分を返す。"""
        diff = StructureDiff(
            category=new.category,
            old_fingerprint=old,
            new_fingerprint=asdict(new),
        )

        old_selectors = old.get("selector_results", {})
        for name, new_probe in new.selector_results.items():
            old_probe = old_selectors.get(name, {})

            if not new_probe.matched:
                if new_probe.optional:
                    continue
                if old_probe.get("matched", False):
                    diff.broken_selectors.append(
                        f"{name}: 全セレクタ失敗 (前回は '{old_probe.get('selector')}' で取得)"
                    )
                continue

            if old_probe.get("matched") and new_probe.index > 0 and old_probe.get("index", 0) == 0:
                diff.fallback_selectors.append(
                    f"{name}: primary '{old_probe.get('selector')}' → fallback '{new_probe.selector}'"
                )

        old_col = old.get("column_count", 0)
        new_col = new.column_count
        if old_col > 0 and new_col > 0 and old_col != new_col:
            diff.column_change = True

        old_row = old.get("row_count", 0)
        if old_row > 0 and new.row_count > 0 and abs(old_row - new.row_count) > 2:
            diff.row_count_changed = True

        return diff

    def check_category(
        self,
        category: str,
        html: str,
        sample_id: str,
    ) -> StructureDiff | None:
        """1カテゴリの構造チェックを実行する。"""
        new_fp = self.probe_html(html, category, sample_id)
        old_fp = self.load_fingerprint(category)

        if old_fp is None:
            logger.info("[%s] 初回チェック — フィンガープリントを保存", category)
            self.save_fingerprint(new_fp)
            self.confirm_version(category)
            return None

        diff = self.compare(old_fp, new_fp)
        self.save_fingerprint(new_fp)

        if diff.has_breaking_change:
            self.bump_version(category)
        else:
            self.confirm_version(category)

        return diff

    # ── SmartRC JSON API 構造チェック ──────────────────

    def probe_smartrc_api(self, category: str, records: list[dict]) -> StructureFingerprint:
        """SmartRC JSON レスポンスのスキーマフィンガープリントを生成する。"""
        probe_cfg = SMARTRC_API_PROBES.get(category, {})
        fp = StructureFingerprint(
            category=category,
            timestamp=datetime.now().isoformat(),
            sample_id=f"smartrc_{probe_cfg.get('endpoint', '')}",
        )

        if not records:
            return fp

        sample = records[0]
        actual_keys = set(sample.keys())
        required_keys = probe_cfg.get("required_keys", [])

        for key in required_keys:
            found = key in actual_keys
            fp.selector_results[key] = SelectorProbeResult(
                selector=f"json_key:{key}",
                matched=found,
                index=0 if found else -1,
                element_count=1 if found else 0,
            )

        fp.row_count = len(records)

        expected_urlist_fields = probe_cfg.get("urlist_fields", 0)
        if expected_urlist_fields:
            urlist = sample.get("urlist", "")
            if urlist:
                items = urlist.split(",")
                if items:
                    field_count = len(items[0].split(":"))
                    fp.column_count = field_count

        return fp

    def check_smartrc_category(
        self,
        category: str,
        records: list[dict],
    ) -> StructureDiff | None:
        """SmartRC API の1カテゴリの構造チェックを実行する。"""
        new_fp = self.probe_smartrc_api(category, records)
        old_fp = self.load_fingerprint(category)

        if old_fp is None:
            logger.info("[%s] SmartRC 初回チェック — フィンガープリントを保存", category)
            self.save_fingerprint(new_fp)
            self.confirm_version(category)
            return None

        diff = self.compare(old_fp, new_fp)
        self.save_fingerprint(new_fp)

        if diff.has_breaking_change:
            self.bump_version(category)
        else:
            self.confirm_version(category)

        return diff

    # ── 一括チェック ─────────────────────────────

    def check_all(
        self,
        runner=None,
        sample_race_id: str = "",
        sample_horse_id: str = "",
        old_sample_race_id: str = "",
    ) -> list[StructureDiff]:
        """
        全カテゴリのページ構造をチェックする。
        netkeiba (HTML) と SmartRC (JSON API) の両方を対象とする。

        runner が渡されれば HTTP fetch、なければ HtmlArchive から最新を取得。
        delayed_publish カテゴリは old_sample_race_id を優先使用する。
        """
        if not sample_race_id or not sample_horse_id:
            raise ValueError("sample_race_id と sample_horse_id を指定してください")

        from scraper.html_archive import HtmlArchive
        archive = HtmlArchive()

        diffs: list[StructureDiff] = []

        # --- netkeiba HTML カテゴリ ---
        for category, cfg in CATEGORY_PROBES.items():
            id_type = cfg.get("id_type", "race")
            is_delayed = cfg.get("delayed_publish", False)

            if is_delayed and old_sample_race_id:
                effective_race_id = old_sample_race_id
            else:
                effective_race_id = sample_race_id

            sample_id = sample_horse_id if id_type == "horse" else effective_race_id
            archive_cat = cfg.get("archive_category", category)

            html = None
            if runner:
                url = cfg["sample_url"].format(
                    race_id=sample_id, horse_id=sample_id,
                )
                try:
                    encoding = cfg.get("encoding")
                    if encoding:
                        resp = runner.client._session.get(url, timeout=15)
                        resp.encoding = encoding
                        html = resp.text
                    else:
                        html = runner.client.fetch(url, use_cache=False)
                except Exception as e:
                    logger.warning("[%s] フェッチ失敗: %s", category, e)

            if not html:
                html = archive.load(archive_cat, sample_id)

            if not html:
                logger.warning("[%s] HTML が取得できませんでした", category)
                continue

            # 遅延公開: ページが「未公開」表示のみで実データがない場合
            not_published = cfg.get("not_published_marker")
            if not_published and not_published in html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                primary_sel = list(cfg.get("selectors", {}).values())
                has_primary = False
                for sel_chain in primary_sel:
                    for sel in sel_chain:
                        if soup.select_one(sel):
                            has_primary = True
                            break
                    if has_primary:
                        break
                if not has_primary:
                    logger.info("[%s] データ未公開 (sample=%s) — 構造チェックをスキップ",
                                category, sample_id)
                    pending_fp = {
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        "sample_id": sample_id,
                        "selector_results": {},
                        "row_count": 0,
                        "column_count": 0,
                        "column_classes": [],
                        "not_published": True,
                        "note": "データ未公開のためチェックスキップ",
                    }
                    fp_path = self._fp_dir / f"{category}.json"
                    fp_path.write_text(
                        json.dumps(pending_fp, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info("未公開フィンガープリント保存: %s", fp_path)
                    continue

            diff = self.check_category(category, html, sample_id)
            if diff:
                diffs.append(diff)

        # --- SmartRC JSON API カテゴリ ---
        smartrc_diffs = self._check_smartrc_apis()
        diffs.extend(smartrc_diffs)

        return diffs

    def _check_smartrc_apis(self) -> list[StructureDiff]:
        """SmartRC の全 API エンドポイントの構造をチェックする。"""
        diffs: list[StructureDiff] = []
        try:
            from scraper.smartrc_client import SmartRCClient
            client = SmartRCClient()
        except Exception as e:
            logger.warning("SmartRC クライアント初期化失敗: %s", e)
            return diffs

        days_raw = client._request("days").get("data", [])
        fetch_map: dict[str, Any] = {
            "smartrc_days": lambda: days_raw,
        }

        sample_rcode = ""
        if days_raw:
            rdate = days_raw[0].get("rdate", "")
            races_str = days_raw[0].get("races", "")
            if races_str:
                place = races_str.split(":")[0].strip()
                races = client.get_races(rdate, place)
                if races:
                    sample_rcode = races[0].get("rcode", "")

        if sample_rcode:
            fetch_map["smartrc_runners"] = lambda: client.get_runners(sample_rcode)
            runners = client.get_runners(sample_rcode)
            if runners:
                hcode = runners[0].get("hcode", "")
                if hcode:
                    fetch_map["smartrc_horses"] = lambda: [client.get_horse(hcode)]
                    fetch_map["smartrc_fullresults"] = lambda: client.get_fullresults(hcode)

        for category, fetcher in fetch_map.items():
            try:
                records = fetcher()
                if not records:
                    logger.warning("[%s] SmartRC レスポンスが空", category)
                    continue
                diff = self.check_smartrc_category(category, records)
                if diff:
                    diffs.append(diff)
            except Exception as e:
                logger.warning("[%s] SmartRC チェック失敗: %s", category, e)

        return diffs

    def generate_report(self, diffs: list[StructureDiff]) -> str:
        """差分チェック結果のレポートを生成する。"""
        html_diffs = [d for d in diffs if d.category not in SMARTRC_API_PROBES]
        api_diffs = [d for d in diffs if d.category in SMARTRC_API_PROBES]

        lines = [
            f"# 構造チェックレポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            "",
        ]

        critical = [d for d in diffs if d.severity == "CRITICAL"]
        warning = [d for d in diffs if d.severity == "WARNING"]
        ok = [d for d in diffs if d.severity == "OK"]

        lines.append(f"CRITICAL: {len(critical)}  WARNING: {len(warning)}  OK: {len(ok)}")
        lines.append(f"  netkeiba HTML: {len(html_diffs)}件  SmartRC API: {len(api_diffs)}件")
        lines.append("")

        if critical:
            lines.append("## CRITICAL — セレクタ破損またはカラム変更")
            for d in critical:
                lines.append(f"\n### {d.category}")
                for b in d.broken_selectors:
                    lines.append(f"  - 🔴 {b}")
                if d.column_change:
                    old_col = d.old_fingerprint.get("column_count", "?") if d.old_fingerprint else "?"
                    new_col = d.new_fingerprint.get("column_count", "?") if d.new_fingerprint else "?"
                    lines.append(f"  - 🔴 カラム数変更: {old_col} → {new_col}")
            lines.append("")

        if warning:
            lines.append("## WARNING — フォールバックセレクタ使用")
            for d in warning:
                lines.append(f"\n### {d.category}")
                for f in d.fallback_selectors:
                    lines.append(f"  - 🟡 {f}")
            lines.append("")

        return "\n".join(lines)


def run_daily_check(
    sample_race_id: str = "",
    sample_horse_id: str = "",
    auto_reparse: bool = True,
    notify: bool = True,
) -> dict:
    """
    毎日の構造チェックを実行する。

    1. 全カテゴリのページ構造をフェッチ＆比較
    2. 差分レポートを生成
    3. CRITICAL な変更があれば再パースを実行
    4. レポートをファイルに保存
    """
    from scraper.run import ScraperRunner

    logger.info("=== 構造チェック開始 ===")
    monitor = StructureMonitor()

    if not sample_race_id or not sample_horse_id:
        storage_mod = __import__("scraper.storage", fromlist=["HybridStorage"])
        storage = storage_mod.HybridStorage(".")
        dates = storage.list_keys("race_lists")
        if dates:
            latest = sorted(dates, reverse=True)[0]
            race_list = storage.load("race_lists", latest)
            if race_list and race_list.get("races"):
                sample_race_id = race_list["races"][0].get("race_id", "")

        if sample_race_id and not sample_horse_id:
            card = storage.load("race_shutuba", sample_race_id)
            if card:
                for entry in card.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid:
                        sample_horse_id = hid
                        break

    if not sample_race_id:
        logger.error("サンプル race_id が特定できません")
        return {"status": "error", "message": "sample race_id not found"}

    # 遅延公開カテゴリ用に十分古いサンプルレースIDを取得
    old_sample_race_id = ""
    try:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=21)).strftime("%Y%m%d")
        old_dates = [d for d in sorted(dates, reverse=True) if d <= cutoff]
        if old_dates:
            old_list = storage.load("race_lists", old_dates[0])
            if old_list and old_list.get("races"):
                old_sample_race_id = old_list["races"][0].get("race_id", "")
                logger.info("遅延公開カテゴリ用サンプル: race=%s (date=%s)",
                            old_sample_race_id, old_dates[0])
    except Exception as e:
        logger.warning("遅延公開サンプル取得失敗: %s", e)

    logger.info("サンプル: race=%s, horse=%s", sample_race_id, sample_horse_id)

    runner = None
    try:
        runner = ScraperRunner(interval=1.5, cache=False, auto_login=True)
    except Exception as e:
        logger.warning("ScraperRunner 初期化失敗 (アーカイブモード): %s", e)

    diffs = monitor.check_all(
        runner=runner,
        sample_race_id=sample_race_id,
        sample_horse_id=sample_horse_id or "2023101473",
        old_sample_race_id=old_sample_race_id,
    )

    report = monitor.generate_report(diffs)
    report_path = Path("data/meta/structure/report.md")
    report_path.write_text(report, encoding="utf-8")
    logger.info("レポート保存: %s", report_path)

    result: dict[str, Any] = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "sample_race_id": sample_race_id,
        "sample_horse_id": sample_horse_id,
        "total_categories": len(CATEGORY_PROBES) + len(SMARTRC_API_PROBES),
        "checked": len(diffs),
        "critical": 0,
        "warning": 0,
        "reparsed_categories": [],
        "report_path": str(report_path),
    }

    critical_cats = []
    for d in diffs:
        if d.severity == "CRITICAL":
            result["critical"] += 1
            critical_cats.append(d.category)
        elif d.severity == "WARNING":
            result["warning"] += 1

    if critical_cats and auto_reparse and runner:
        logger.info("=== CRITICAL 検出 — 再パース開始 ===")
        for cat in critical_cats:
            logger.info("再パース: %s", cat)
            try:
                count = runner.reparse_all(cat)
                result["reparsed_categories"].append({"category": cat, "count": count})
                logger.info("再パース完了: %s → %d件", cat, count)
            except Exception as e:
                logger.error("再パース失敗 [%s]: %s", cat, e)
                result["reparsed_categories"].append({"category": cat, "error": str(e)})

    if runner:
        runner.close()

    severity = "CRITICAL" if result["critical"] > 0 else "WARNING" if result["warning"] > 0 else "OK"
    result["severity"] = severity

    summary_path = Path("data/meta/structure/last_check.json")
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== 構造チェック完了 [%s] ===", severity)
    return result
