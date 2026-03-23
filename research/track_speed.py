"""馬場速度レベリングシステム (Track Speed Rating System)
=====================================================

2着馬走破タイムをベースに、コース条件ごとの統計量から
日付×競馬場×芝ダートの馬場速度を定量化する。

手法:
  1. 2020-2024年データで条件別ベースタイム(mean/std)を構築
  2. 各レースの2着タイムからz-scoreを算出
  3. 出走馬の能力偏差(HSR: Horse Speed Rating)を反復推定し、
     フィールド強度を補正 → 馬の質に起因する高時計バイアスを除去
  4. 残差を日単位で集約 → Track Speed Index (TSI)

TSI の解釈:
  負値 = 高速馬場（タイムが出やすい）
  正値 = 低速馬場（タイムが出にくい）
  0 付近 = 標準的な馬場
"""

import json
import statistics
import time as _time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JRA_VENUES = {
    "東京", "中山", "阪神", "京都", "中京",
    "小倉", "新潟", "福島", "札幌", "函館",
}

GRADE_MAP = {
    "G1": "G", "G2": "G", "G3": "G",
    "L": "OP", "OP": "OP", "オープン": "OP",
    "3勝": "3勝", "1600万下": "3勝", "1600万": "3勝",
    "2勝": "2勝", "1000万下": "2勝", "1000万": "2勝",
    "1勝": "1勝", "500万下": "1勝", "500万": "1勝",
    "未勝利": "未勝利",
    "新馬": "新馬",
}

SPEED_LABELS = [
    (-float("inf"), -1.5, "超高速", 5),
    (-1.5,         -0.5, "高速",   4),
    (-0.5,          0.5, "標準",   3),
    (0.5,           1.5, "低速",   2),
    (1.5,   float("inf"), "超低速", 1),
]


def _classify_speed(tsi: float) -> tuple[str, int]:
    for lo, hi, label, level in SPEED_LABELS:
        if lo <= tsi < hi:
            return label, level
    return "標準", 3


def _grade_group(grade: str) -> str:
    if not grade:
        return "unknown"
    for prefix, group in GRADE_MAP.items():
        if grade == prefix or grade.startswith(prefix):
            return group
    if "障" in grade:
        return "障害"
    return "other"


def _baseline_key(venue, surface, distance, grade_group, track_condition):
    return f"{venue}_{surface}_{distance}_{grade_group}_{track_condition}"


class TrackSpeedAnalyzer:
    """馬場速度レベリングエンジン

    反復的推定:
      Iteration 0: 条件別ベースラインから raw z-score を算出
      Iteration 1+: 馬の能力指数(HSR)を更新 → フィールド補正 → TSI 更新
      3回の反復で収束
    """

    def __init__(self, storage):
        self.storage = storage
        self._races: list[dict] = []
        self._baselines: dict = {}
        self._fallback1: dict = {}
        self._fallback2: dict = {}
        self._horse_perfs: dict[str, list] = defaultdict(list)
        self._hsr: dict[str, float] = {}
        self._race_z: dict[int, float] = {}
        self._race_field_adj: dict[int, float] = {}
        self._race_tsi: dict[int, float] = {}
        self._daily: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        base_years=range(2020, 2025),
        all_years=range(2020, 2027),
        n_iterations: int = 3,
        progress_cb=None,
    ):
        """全パイプライン実行"""
        t0 = _time.time()

        self._load_all_races(all_years, progress_cb)
        if progress_cb:
            progress_cb(f"レース読込完了: {len(self._races)}件 ({_time.time()-t0:.1f}s)")

        self._build_baselines(base_years)
        if progress_cb:
            progress_cb(f"ベースライン構築完了: {len(self._baselines)}グループ")

        self._compute_raw_z()
        if progress_cb:
            progress_cb(f"raw z-score算出: {len(self._race_z)}レース")

        for i in range(n_iterations):
            self._compute_hsr()
            self._compute_field_adjustment()
            self._compute_tsi()
            if progress_cb:
                n_hsr = len(self._hsr)
                progress_cb(
                    f"反復{i+1}/{n_iterations} 完了 (HSR: {n_hsr}頭)"
                )

        self._aggregate_daily()
        if progress_cb:
            elapsed = _time.time() - t0
            progress_cb(
                f"全計算完了: {len(self._daily)}日 ({elapsed:.1f}s)"
            )

        return self._daily

    def get_baselines_summary(self) -> dict:
        return {
            k: {
                "mean": round(v["mean"], 2),
                "std": round(v["std"], 3),
                "count": v["count"],
            }
            for k, v in sorted(self._baselines.items())
        }

    def get_stats(self) -> dict:
        cal = getattr(self, "_calibration", {})
        return {
            "total_races_loaded": len(self._races),
            "races_with_z": len(self._race_z),
            "horses_with_hsr": len(self._hsr),
            "baseline_groups": len(self._baselines),
            "days_computed": len(self._daily),
            "calibration_turf": round(cal.get("turf_offset", 0), 3),
            "calibration_dirt": round(cal.get("dirt_offset", 0), 3),
        }

    def save(self, output_dir: str) -> dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "track_speed_baselines.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "description": "2着馬走破タイム ベースライン統計量",
                        "base_years": "2020-2024",
                        "key_format": "venue_surface_distance_gradeGroup_trackCondition",
                    },
                    "baselines": self.get_baselines_summary(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        daily_summary = {}
        for date, venues in self._daily.items():
            daily_summary[date] = {}
            for venue, surfaces in venues.items():
                daily_summary[date][venue] = {}
                for surface, data in surfaces.items():
                    daily_summary[date][venue][surface] = {
                        "tsi": data["tsi"],
                        "level": data["level"],
                        "label": data["label"],
                        "sample_size": data["sample_size"],
                        "track_conditions": data["track_conditions"],
                    }

        with open(out / "track_speed_daily.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "description": "日別馬場速度指数 (Track Speed Index)",
                        "tsi_interpretation": "TSI < 0 → 高速馬場, TSI > 0 → 低速馬場",
                        "levels": {
                            "5": "超高速 (TSI ≤ -1.5)",
                            "4": "高速 (-1.5 < TSI ≤ -0.5)",
                            "3": "標準 (-0.5 < TSI < 0.5)",
                            "2": "低速 (0.5 ≤ TSI < 1.5)",
                            "1": "超低速 (TSI ≥ 1.5)",
                        },
                        "field_strength_correction": (
                            "出走馬の能力偏差(HSR)を反復推定し、"
                            "フィールド強度中央値で各レースのraw z-scoreを補正。"
                            "高レベルメンバーによる高時計バイアスを除去済み。"
                        ),
                        "computed_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "stats": self.get_stats(),
                    },
                    "daily": daily_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(out / "track_speed_races.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {"description": "レース別馬場速度詳細"},
                    "daily": self._daily,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "baselines": str(out / "track_speed_baselines.json"),
            "daily": str(out / "track_speed_daily.json"),
            "races": str(out / "track_speed_races.json"),
        }

    # ------------------------------------------------------------------
    # Phase 1: Data Loading
    # ------------------------------------------------------------------

    def _load_all_races(self, years, progress_cb=None):
        self._races = []
        available_years = self.storage.list_years("race_result")
        target_years = sorted(
            str(y) for y in years if str(y) in available_years
        )

        for year in target_years:
            race_ids = self.storage.list_keys("race_result", year)
            if progress_cb:
                progress_cb(f"  {year}年: {len(race_ids)}件 読込中...")

            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(
                    lambda rid: self.storage.load("race_result", rid),
                    race_ids,
                ))

            loaded = 0
            for data in results:
                rec = self._extract_race(data)
                if rec:
                    rec["idx"] = len(self._races)
                    self._races.append(rec)
                    loaded += 1

            if progress_cb:
                progress_cb(f"    → {loaded}/{len(race_ids)}件 抽出")

    def _extract_race(self, data: dict | None) -> dict | None:
        if not data:
            return None

        venue = data.get("venue", "")
        surface = data.get("surface", "")
        distance = data.get("distance", 0)

        if venue not in JRA_VENUES:
            return None

        # surface/distance が欠損している場合、race_shutuba から補完を試みる
        if not surface or not distance:
            race_id = data.get("race_id", "")
            if race_id:
                shutuba = self.storage.load("race_shutuba", race_id)
                if shutuba:
                    surface = surface or shutuba.get("surface", "")
                    distance = distance or shutuba.get("distance", 0)

        # それでも欠損なら、走破タイムからの推定を試みる
        if not surface or not distance:
            surface, distance = self._infer_surface_distance(data)

        if surface not in ("芝", "ダート") or not distance:
            return None

        grade = _grade_group(data.get("grade", ""))
        if grade in ("障害", "unknown", "other"):
            return None

        entries = data.get("entries", [])
        entry_2nd = None
        for e in entries:
            if e.get("finish_position") == 2 and e.get("time_sec"):
                entry_2nd = e
                break
        if not entry_2nd:
            return None

        horse_entries = []
        for e in entries:
            fp = e.get("finish_position", -1)
            if fp > 0 and e.get("horse_id") and e.get("time_sec"):
                horse_entries.append(
                    {
                        "horse_id": e["horse_id"],
                        "finish_position": fp,
                        "time_sec": e["time_sec"],
                        "odds": e.get("odds"),
                        "horse_name": e.get("horse_name", ""),
                    }
                )

        date_str = data.get("date", "")
        date_compact = date_str.replace("-", "") if date_str else ""

        return {
            "race_id": data.get("race_id", ""),
            "date": date_compact,
            "venue": venue,
            "surface": surface,
            "distance": distance,
            "track_condition": data.get("track_condition", ""),
            "grade": data.get("grade", ""),
            "grade_group": grade,
            "time_2nd": entry_2nd["time_sec"],
            "field_size": data.get("field_size", len(entries)),
            "entries": horse_entries,
            "race_name": data.get("race_name", ""),
            "round": data.get("round", 0),
        }

    @staticmethod
    def _infer_surface_distance(data: dict) -> tuple[str, int]:
        """走破タイムとラップタイムから surface/distance を推定する。

        JRA の標準的な距離は 1000, 1200, 1400, 1600, 1800, 2000, 2200,
        2400, 2500, 2600, 3000, 3200, 3400, 3600 m。
        ラップの本数 × 200m で距離を推定し、芝/ダートは勝ちタイムの
        速度帯で判定する。
        """
        laps = data.get("lap_times", [])
        if laps:
            distance = len(laps) * 200
            VALID_DISTANCES = [
                1000, 1200, 1400, 1600, 1800, 2000, 2200,
                2400, 2500, 2600, 3000, 3200, 3400, 3600,
            ]
            best = min(VALID_DISTANCES, key=lambda d: abs(d - distance))
            distance = best
        else:
            entries = data.get("entries", [])
            winner_time = None
            for e in entries:
                if e.get("finish_position") == 1 and e.get("time_sec"):
                    winner_time = e["time_sec"]
                    break
            if not winner_time:
                return "", 0
            # 芝 1200m ≈ 68s, 1600m ≈ 93s, 2000m ≈ 118s, 2400m ≈ 143s
            # ダートは芝+3-5s per 400m
            SPEED_MAP = [
                (60, 1000), (70, 1200), (84, 1400), (96, 1600),
                (110, 1800), (122, 2000), (135, 2200), (146, 2400),
                (155, 2500), (164, 2600), (185, 3000), (197, 3200),
            ]
            distance = min(SPEED_MAP, key=lambda t: abs(t[0] - winner_time))[1]

        # 芝 vs ダートの判定: 同距離帯でダートは芝より 2-4% 遅い
        entries = data.get("entries", [])
        winner_time = None
        for e in entries:
            if e.get("finish_position") == 1 and e.get("time_sec"):
                winner_time = e["time_sec"]
                break

        if winner_time and distance:
            speed_per_m = winner_time / distance  # sec/m
            # 芝の典型: 0.058 sec/m,  ダートの典型: 0.062 sec/m
            surface = "ダート" if speed_per_m > 0.060 else "芝"
        else:
            surface = "芝"

        return surface, distance

    # ------------------------------------------------------------------
    # Phase 2: Baseline Statistics
    # ------------------------------------------------------------------

    def _build_baselines(self, base_years):
        year_set = set(str(y) for y in base_years)
        groups = defaultdict(list)
        fb1_groups = defaultdict(list)
        fb2_groups = defaultdict(list)

        for r in self._races:
            year = r["race_id"][:4] if r["race_id"] else r["date"][:4]
            if year not in year_set:
                continue

            key = _baseline_key(
                r["venue"], r["surface"], r["distance"],
                r["grade_group"], r["track_condition"],
            )
            groups[key].append(r["time_2nd"])

            fb1 = f"ALL_{r['surface']}_{r['distance']}_{r['grade_group']}_{r['track_condition']}"
            fb1_groups[fb1].append(r["time_2nd"])

            fb2 = f"ALL_{r['surface']}_{r['distance']}_{r['grade_group']}_ALL"
            fb2_groups[fb2].append(r["time_2nd"])

        MIN_N = 5

        def _min_std(key: str) -> float:
            """距離に応じた最低std。少サンプルの過小推定を防止"""
            parts = key.split("_")
            try:
                dist = int(parts[2]) if len(parts) > 2 else 1600
            except ValueError:
                dist = 1600
            return 0.4 + dist / 4000

        def _stats(times, key=""):
            raw_std = statistics.stdev(times) if len(times) > 1 else 1.0
            floor = _min_std(key)
            return {
                "mean": statistics.mean(times),
                "std": max(raw_std, floor),
                "count": len(times),
            }

        self._baselines = {
            k: _stats(v, k) for k, v in groups.items() if len(v) >= MIN_N
        }
        self._fallback1 = {
            k: _stats(v, k) for k, v in fb1_groups.items() if len(v) >= MIN_N
        }
        self._fallback2 = {
            k: _stats(v, k) for k, v in fb2_groups.items() if len(v) >= MIN_N
        }

    def _get_baseline(self, race: dict) -> dict | None:
        key = _baseline_key(
            race["venue"], race["surface"], race["distance"],
            race["grade_group"], race["track_condition"],
        )
        if key in self._baselines:
            return self._baselines[key]

        fb1 = f"ALL_{race['surface']}_{race['distance']}_{race['grade_group']}_{race['track_condition']}"
        if fb1 in self._fallback1:
            return self._fallback1[fb1]

        fb2 = f"ALL_{race['surface']}_{race['distance']}_{race['grade_group']}_ALL"
        if fb2 in self._fallback2:
            return self._fallback2[fb2]

        return None

    # ------------------------------------------------------------------
    # Phase 3: Z-scores & Iterative HSR / TSI
    # ------------------------------------------------------------------

    def _compute_raw_z(self):
        """各レースの2着タイムと各出走馬の個別タイムのz-scoreを算出"""
        self._race_z = {}
        self._horse_perfs = defaultdict(list)

        for r in self._races:
            bl = self._get_baseline(r)
            if not bl:
                continue
            z = (r["time_2nd"] - bl["mean"]) / bl["std"]
            self._race_z[r["idx"]] = z

            for e in r["entries"]:
                hz = (e["time_sec"] - bl["mean"]) / bl["std"]
                self._horse_perfs[e["horse_id"]].append((r["idx"], hz))

    def _compute_hsr(self):
        """Horse Speed Rating: 各馬の能力を「TSI補正済みz-scoreの中央値」で推定"""
        self._hsr = {}
        for horse_id, perfs in self._horse_perfs.items():
            if len(perfs) < 2:
                continue
            adjusted = [hz - self._race_tsi.get(ridx, 0.0) for ridx, hz in perfs]
            self._hsr[horse_id] = statistics.median(adjusted)

    def _compute_field_adjustment(self):
        """各レースのフィールド強度 = 出走馬HSRの中央値"""
        self._race_field_adj = {}
        for r in self._races:
            if r["idx"] not in self._race_z:
                continue
            hsrs = [
                self._hsr[e["horse_id"]]
                for e in r["entries"]
                if e["horse_id"] in self._hsr
            ]
            self._race_field_adj[r["idx"]] = (
                statistics.median(hsrs) if len(hsrs) >= 3 else 0.0
            )

    def _compute_tsi(self):
        """TSI = raw_z - field_adjustment"""
        self._race_tsi = {}
        for idx, z in self._race_z.items():
            self._race_tsi[idx] = z - self._race_field_adj.get(idx, 0.0)

    # ------------------------------------------------------------------
    # Phase 4: Daily Aggregation
    # ------------------------------------------------------------------

    def _aggregate_daily(self):
        # 芝/ダート別にTSIの全体平均を算出してゼロ中心に校正する
        # (ベースライン推定バイアスを除去)
        turf_tsi = []
        dirt_tsi = []
        for r in self._races:
            tsi = self._race_tsi.get(r["idx"])
            if tsi is None:
                continue
            if r["surface"] == "芝":
                turf_tsi.append(tsi)
            else:
                dirt_tsi.append(tsi)

        turf_offset = statistics.mean(turf_tsi) if turf_tsi else 0.0
        dirt_offset = statistics.mean(dirt_tsi) if dirt_tsi else 0.0
        self._calibration = {"turf_offset": turf_offset, "dirt_offset": dirt_offset}

        daily_raw: dict[str, dict[str, dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for r in self._races:
            if r["idx"] not in self._race_tsi:
                continue
            date = r["date"]
            if not date or len(date) != 8:
                continue

            offset = turf_offset if r["surface"] == "芝" else dirt_offset
            calibrated = self._race_tsi[r["idx"]] - offset

            bl = self._get_baseline(r)
            detail = {
                "race_id": r["race_id"],
                "round": r["round"],
                "race_name": r["race_name"],
                "distance": r["distance"],
                "grade": r["grade"],
                "track_condition": r["track_condition"],
                "field_size": r["field_size"],
                "time_2nd": round(r["time_2nd"], 1),
                "baseline_mean": round(bl["mean"], 2) if bl else None,
                "baseline_std": round(bl["std"], 3) if bl else None,
                "raw_z": round(self._race_z.get(r["idx"], 0), 3),
                "field_adj": round(self._race_field_adj.get(r["idx"], 0), 3),
                "tsi": round(calibrated, 3),
            }
            daily_raw[date][r["venue"]][r["surface"]].append(detail)

        self._daily = {}
        for date in sorted(daily_raw.keys()):
            self._daily[date] = {}
            for venue in sorted(daily_raw[date].keys()):
                self._daily[date][venue] = {}
                for surface in sorted(daily_raw[date][venue].keys()):
                    races = daily_raw[date][venue][surface]
                    tsi_vals = [d["tsi"] for d in races]
                    avg_tsi = statistics.mean(tsi_vals) if tsi_vals else 0.0
                    label, level = _classify_speed(avg_tsi)

                    self._daily[date][venue][surface] = {
                        "tsi": round(avg_tsi, 3),
                        "level": level,
                        "label": label,
                        "sample_size": len(races),
                        "track_conditions": sorted(
                            set(d["track_condition"] for d in races)
                        ),
                        "races": sorted(races, key=lambda d: d["round"]),
                    }


# ==================================================================
# Entry point
# ==================================================================

def run_analysis(base_dir: str = ".", progress_cb=None):
    """分析の実行エントリーポイント"""
    import sys
    sys.path.insert(0, base_dir)
    from scraper.storage import HybridStorage

    storage = HybridStorage(base_dir)
    analyzer = TrackSpeedAnalyzer(storage)

    analyzer.run(progress_cb=progress_cb)
    files = analyzer.save(str(Path(base_dir) / "data" / "analysis"))
    stats = analyzer.get_stats()

    return {"files": files, "stats": stats, "daily": analyzer._daily}


if __name__ == "__main__":
    def _print(msg):
        print(f"[TrackSpeed] {msg}")

    result = run_analysis(".", progress_cb=_print)
    print(f"\nStats: {result['stats']}")
    print("Files:")
    for k, v in result["files"].items():
        print(f"  {k}: {v}")
