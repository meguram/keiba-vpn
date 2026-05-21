"""Centralized data path definitions for the keiba-vpn project.

Portable web/UI bundle (copy to another machine → pages work without GCS):
  ``data/page_reference/``  — or ``KEIBA_PAGE_REFERENCE_DIR`` if overridden.

Operational / pipeline (stay on the host, not required for read-only UI):
  ``data/local/`` — scrape queue meta, pedigree shards, features, L2 access logs, etc.
  ``data/cache/`` — HybridStorage disk L2
  ``data/queue/`` — job queue
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "data" / "local"


def page_reference_root() -> Path:
    """Page-serving data root (portable). Override with ``KEIBA_PAGE_REFERENCE_DIR``."""
    override = os.environ.get("KEIBA_PAGE_REFERENCE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "data" / "page_reference"


# Module-level alias (re-evaluated only at import; use page_reference_root() if env may change)
PAGE_REF = page_reference_root()

# -- Page-reference: calendar / lists (UI date picker, race list APIs) --------
RACE_LISTS_DIR = PAGE_REF / "race_lists"
LEGACY_RACE_LISTS_DIR = LOCAL / "race_lists"

# -- Page-reference: person profiles (jockey / trainer pages) -------------------
PERSON_DIR = PAGE_REF / "meta" / "person"
LEGACY_PERSON_DIR = LOCAL / "meta" / "person"

# -- Page-reference: modeling manifests & eval summaries (UI / docs) ------------
MODELING_META_DIR = PAGE_REF / "meta" / "modeling"
LEGACY_MODELING_META_DIR = LOCAL / "meta" / "modeling"

# -- Page-reference artifacts (bloodline, tables, performance, etc.) ----------
ART_DIR = PAGE_REF / "note_aptitude_race"
ART_DIR_L3 = PAGE_REF / "note_aptitude_race_l3"
IDX_DIR = PAGE_REF / "pedigree_race_index"
PED_SIMILARITY_DIR = PAGE_REF / "pedigree_map"
BLOODLINE_VECTOR_DIR = PAGE_REF / "bloodline_vector" / "v2_l2"
TABLES_DIR = PAGE_REF / "tables"
RACE_PERFORMANCE_DIR = PAGE_REF / "race_performance"

# -- Knowledge / baselines (page-reference) -----------------------------------
KNOWLEDGE_DIR = PAGE_REF / "knowledge"
TRACK_SPEED_BASELINES = KNOWLEDGE_DIR / "track_speed_baselines.parquet"
TRACK_SPEED_PACE_BASELINES = KNOWLEDGE_DIR / "track_speed_pace_baselines.parquet"
COURSE_PROFILES_JSON = KNOWLEDGE_DIR / "course_profiles.json"
SIRE_APTITUDE_NOTE_JSON = KNOWLEDGE_DIR / "sire_aptitude_note.json"
MYOSTATIN_GENES_JSON = KNOWLEDGE_DIR / "myostatin_genes.json"
PERFORMANCE_GENES_JSON = KNOWLEDGE_DIR / "performance_genes.json"

# -- JRA baba (page-reference) ------------------------------------------------
JRA_BABA_DIR = PAGE_REF / "cushion"
CUSHION_VALUES_JSON = JRA_BABA_DIR / "cushion_values.json"

# -- Track speed (page-reference) ---------------------------------------------
TRACK_SPEED_RACES_DIR = PAGE_REF / "track_speed"

# -- Research artifacts (pipeline / research scripts only) --------------------
PED_DIR = LOCAL / "horse_pedigree_5gen"
PED_10GEN_DIR = LOCAL / "horse_pedigree_10gen"
PED_10GEN_RESEARCH_DIR = LOCAL / "research" / "pedigree_10gen"
PED_10GEN_3VIEW_DIR = LOCAL / "research" / "pedigree_10gen_3view"
BLOODLINE_DIR = LOCAL / "research" / "bloodline"
HORSE_APTITUDE_DIR = LOCAL / "research" / "horse_aptitude"
COURSE_BLOODLINE_DIR = LOCAL / "research" / "course_bloodline"

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
    return page_reference_root() / category


def legacy_local_category_dir(category: str) -> Path:
    """Pre-migration location for ``local_only`` categories."""
    return LOCAL / category


def resolve_existing_path(primary: Path, legacy: Path) -> Path:
    """Return ``primary`` if it exists, else ``legacy`` if it exists, else ``primary``."""
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


def person_profile_path(ptype: str, person_id: str) -> Path:
    """Jockey/trainer JSON served to person pages."""
    name = f"{ptype}_{person_id}.json"
    primary = PERSON_DIR / name
    if primary.is_file():
        return primary
    legacy = LEGACY_PERSON_DIR / name
    return legacy if legacy.is_file() else primary
