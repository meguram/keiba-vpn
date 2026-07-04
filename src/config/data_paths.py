"""Centralized data path definitions for the keiba-vpn project.

Calculated / aggregated artifacts (UI, inference outputs, research summaries):
  ``data/calculated_data/`` — primary read path for API pages (override: ``KEIBA_CALCULATED_DATA_DIR``)

Portable web/UI bundle (legacy mirror; synced into calculated_data via sync script):
  ``data/page_reference/``  — or ``KEIBA_PAGE_REFERENCE_DIR`` if overridden

Operational / pipeline (stay on the host, not required for read-only UI):
  ``data/local/`` — scrape queue meta, pedigree shards, features, L2 access logs, etc.
  ``data/cache/`` — HybridStorage disk L2
  ``data/queue/`` — job queue
"""
from __future__ import annotations

import os
from pathlib import Path

# ── GCS パス SSoT（AREA-06 §3）────────────────────────────────────────────
GCS_BASE = "chuou/data/preprocessed/netkeiba/pc"
GCS_OTHERS = "chuou/data/others"


def _gcs_bucket() -> str:
    return os.environ.get("GCS_BUCKET", "")


def race_path(category: str, race_id: str) -> str:
    """レース単位 GCS パス（gs:// 付き）。"""
    year = race_id[:4]
    return f"gs://{_gcs_bucket()}/{GCS_BASE}/{category}/{year}/{race_id}.json"


def horse_path(category: str, horse_id: str) -> str:
    """馬単位 GCS パス（gs:// 付き）。prefix = horse_id 先頭4桁。"""
    prefix = horse_id[:4]
    return f"gs://{_gcs_bucket()}/{GCS_BASE}/{category}/{prefix}/{horse_id}.json"


def others_path(category: str, key: str) -> str:
    """others 配下 GCS パス（gs:// 付き）。"""
    return f"gs://{_gcs_bucket()}/{GCS_OTHERS}/{category}/{key}.json"


def gcs_blob_path(category: str, key: str, id_type: str) -> str:
    """バケット内相対 blob パス（HybridStorage 用）。"""
    if id_type == "other":
        return f"{GCS_OTHERS}/{category}/{key}.json"
    if id_type == "horse":
        prefix = key[:4]
        return f"{GCS_BASE}/{category}/{prefix}/{key}.json"
    year = key[:4] if len(key) >= 4 else "unknown"
    return f"{GCS_BASE}/{category}/{year}/{key}.json"


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "data" / "local"
CALCULATED_DATA_DIR = ROOT / "data" / "calculated_data"


def calculated_data_root() -> Path:
    """Root for post-processed / model / aggregation artifacts served to the API."""
    override = os.environ.get("KEIBA_CALCULATED_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return CALCULATED_DATA_DIR


def page_reference_root() -> Path:
    """Page-serving data root (portable). Override with ``KEIBA_PAGE_REFERENCE_DIR``."""
    override = os.environ.get("KEIBA_PAGE_REFERENCE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "data" / "page_reference"


# Module-level alias (re-evaluated only at import; use *\_root() if env may change)
CALCULATED = calculated_data_root()
PAGE_REF = page_reference_root()


def resolve_existing_path(*candidates: Path) -> Path:
    """Return the first path that exists; if none exist, return the first candidate."""
    if not candidates:
        raise ValueError("resolve_existing_path requires at least one path")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# -- Calculated-data layout (preferred for API) --------------------------------
CALC_KNOWLEDGE_DIR = CALCULATED / "knowledge"
CALC_PREDICTIONS_JSON = CALCULATED / "predictions" / "predictions.json"
CALC_RACE_LISTS_DIR = CALCULATED / "race_lists"
CALC_ART_DIR = CALCULATED / "note_aptitude_race"
CALC_ART_DIR_L3 = CALCULATED / "note_aptitude_race_l3"
CALC_IDX_DIR = CALCULATED / "pedigree_race_index"
CALC_PED_SIMILARITY_DIR = CALCULATED / "pedigree_map"
CALC_BLOODLINE_VECTOR_DIR = CALCULATED / "bloodline_vector" / "v2_l2"
CALC_TABLES_DIR = CALCULATED / "tables"
CALC_RACE_PERFORMANCE_DIR = CALCULATED / "race_performance"
CALC_TRACK_SPEED_RACES_DIR = CALCULATED / "track_speed"
CALC_JRA_BABA_DIR = CALCULATED / "cushion"
CALC_BLOODLINE_DIR = CALCULATED / "bloodline"
CALC_COURSE_BLOODLINE_DIR = CALCULATED / "course_bloodline"
CALC_TRACKING_DIFFICULTY_DIR = CALCULATED / "tracking_difficulty"
CALC_GROWTH_CURVE_DIR = CALCULATED / "growth_curve"
CALC_HORSE_NAME_INDEX = CALC_KNOWLEDGE_DIR / "horse_name_index.json"

# -- Page-reference: calendar / lists (UI date picker, race list APIs) --------
RACE_LISTS_DIR = resolve_existing_path(CALC_RACE_LISTS_DIR, PAGE_REF / "race_lists")
LEGACY_RACE_LISTS_DIR = LOCAL / "race_lists"

# -- Page-reference: person profiles (jockey / trainer pages) -------------------
PERSON_DIR = resolve_existing_path(CALCULATED / "meta" / "person", PAGE_REF / "meta" / "person")
LEGACY_PERSON_DIR = LOCAL / "meta" / "person"

# -- Page-reference: modeling manifests & eval summaries (UI / docs) ----------
MODELING_META_DIR = resolve_existing_path(
    CALCULATED / "meta" / "modeling", PAGE_REF / "meta" / "modeling"
)
LEGACY_MODELING_META_DIR = LOCAL / "meta" / "modeling"

# -- Page-reference artifacts (bloodline, tables, performance, etc.) ----------
ART_DIR = resolve_existing_path(CALC_ART_DIR, PAGE_REF / "note_aptitude_race")
ART_DIR_L3 = resolve_existing_path(CALC_ART_DIR_L3, PAGE_REF / "note_aptitude_race_l3")
IDX_DIR = resolve_existing_path(CALC_IDX_DIR, PAGE_REF / "pedigree_race_index")
PED_SIMILARITY_DIR = resolve_existing_path(CALC_PED_SIMILARITY_DIR, PAGE_REF / "pedigree_map")
BLOODLINE_VECTOR_DIR = resolve_existing_path(
    CALC_BLOODLINE_VECTOR_DIR, PAGE_REF / "bloodline_vector" / "v2_l2"
)
TABLES_DIR = resolve_existing_path(CALC_TABLES_DIR, PAGE_REF / "tables")
RACE_PERFORMANCE_DIR = resolve_existing_path(CALC_RACE_PERFORMANCE_DIR, PAGE_REF / "race_performance")

# -- Knowledge / baselines ------------------------------------------------------
KNOWLEDGE_DIR = resolve_existing_path(CALC_KNOWLEDGE_DIR, PAGE_REF / "knowledge", LOCAL / "knowledge")
TRACK_SPEED_BASELINES = KNOWLEDGE_DIR / "track_speed_baselines.parquet"
TRACK_SPEED_PACE_BASELINES = KNOWLEDGE_DIR / "track_speed_pace_baselines.parquet"
COURSE_PROFILES_JSON = KNOWLEDGE_DIR / "course_profiles.json"
SIRE_APTITUDE_NOTE_JSON = KNOWLEDGE_DIR / "sire_aptitude_note.json"
MYOSTATIN_GENES_JSON = KNOWLEDGE_DIR / "myostatin_genes.json"
PERFORMANCE_GENES_JSON = KNOWLEDGE_DIR / "performance_genes.json"
HORSE_NAME_INDEX_JSON = resolve_existing_path(
    CALC_HORSE_NAME_INDEX,
    KNOWLEDGE_DIR / "horse_name_index.json",
    ROOT / "data" / "knowledge" / "horse_name_index.json",
    LOCAL / "knowledge" / "horse_name_index.json",
)

# -- JRA baba / cushion ---------------------------------------------------------
JRA_BABA_DIR = resolve_existing_path(
    CALC_JRA_BABA_DIR, PAGE_REF / "cushion", ROOT / "data" / "jra_baba", LOCAL / "jra_baba"
)
CUSHION_VALUES_JSON = resolve_existing_path(
    CALC_JRA_BABA_DIR / "cushion_values.json",
    PAGE_REF / "cushion" / "cushion_values.json",
    ROOT / "data" / "jra_baba" / "cushion_values.json",
    LOCAL / "jra_baba" / "cushion_values.json",
)

# -- Track speed --------------------------------------------------------------
TRACK_SPEED_RACES_DIR = resolve_existing_path(CALC_TRACK_SPEED_RACES_DIR, PAGE_REF / "track_speed")

# -- Predictions (dashboard) --------------------------------------------------
PREDICTIONS_JSON = resolve_existing_path(
    CALC_PREDICTIONS_JSON,
    ROOT / "data" / "processed" / "predictions.json",
)

# -- Inference artifacts (precomputed, served read-only to UI) ----------------
TRACKING_DIFFICULTY_DIR = CALC_TRACKING_DIFFICULTY_DIR
GROWTH_CURVE_DIR = CALC_GROWTH_CURVE_DIR

# -- Research artifacts (pipeline / research scripts only) --------------------
PED_DIR = LOCAL / "horse_pedigree_5gen"
PED_10GEN_DIR = LOCAL / "horse_pedigree_10gen"
PED_10GEN_RESEARCH_DIR = LOCAL / "research" / "pedigree_10gen"
PED_10GEN_3VIEW_DIR = LOCAL / "research" / "pedigree_10gen_3view"
BLOODLINE_DIR = resolve_existing_path(CALC_BLOODLINE_DIR, ROOT / "data" / "research" / "bloodline")
HORSE_APTITUDE_DIR = LOCAL / "research" / "horse_aptitude"
COURSE_BLOODLINE_DIR = resolve_existing_path(
    CALC_COURSE_BLOODLINE_DIR, ROOT / "data" / "research" / "course_bloodline"
)

# -- Meta / operational (not in portable bundle) ------------------------------
META_DIR = LOCAL / "meta"
META_MODELING_DIR = MODELING_META_DIR  # backward-compat name for scripts
META_PERSON_DIR = PERSON_DIR
META_STRUCTURE_DIR = META_DIR / "structure"
META_LOGS_DIR = META_DIR / "logs"
GLOBAL_SLOTS_DIR = META_DIR / "netkeiba_global_slots"

# -- Features (pipeline / ML) -------------------------------------------------
FEATURES_DIR = LOCAL / "features"
FEATURES_INFO_DIR = LOCAL / "features_info"

# -- ML / modeling (training parquet — pipeline, not portable UI) -------------
ML_DIR = LOCAL / "ml"
ML_WAREHOUSE_DIR = ML_DIR / "warehouse"
MODELING_DIR = LOCAL / "modeling"

# -- Analysis (pipeline / research scripts only) ------------------------------
ANALYSIS_DIR = LOCAL / "analysis"
PEDIGREE_ANALYSIS_DIR = ANALYSIS_DIR / "pedigree"

# Categories stored under page_reference via HybridStorage (local_only)
PAGE_REF_STORAGE_CATEGORIES: frozenset[str] = frozenset({"race_lists"})


def page_ref_category_dir(category: str) -> Path:
    """Directory for a HybridStorage ``local_only`` category."""
    calc = calculated_data_root() / category
    if calc.exists():
        return calc
    return page_reference_root() / category


def legacy_local_category_dir(category: str) -> Path:
    """Pre-migration location for ``local_only`` categories."""
    return LOCAL / category


def person_profile_path(ptype: str, person_id: str) -> Path:
    """Jockey/trainer JSON served to person pages."""
    name = f"{ptype}_{person_id}.json"
    for base in (PERSON_DIR, LEGACY_PERSON_DIR):
        p = base / name
        if p.is_file():
            return p
    return PERSON_DIR / name
