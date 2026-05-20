"""Centralized data path definitions for the keiba-vpn project.

All locally-managed data lives under data/local/ (operational/pipeline) or
data/page_reference/ (data directly served to web pages via API endpoints).
data/cache/ (HybridStorage L2 cache) and data/queue/ (job queue) are separate.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "data" / "local"
PAGE_REF = ROOT / "data" / "page_reference"

# -- Page-reference artifacts (served to web pages) ---------------------------
ART_DIR         = PAGE_REF / "note_aptitude_race"
ART_DIR_L3      = PAGE_REF / "note_aptitude_race_l3"
IDX_DIR         = PAGE_REF / "pedigree_race_index"
PED_SIMILARITY_DIR     = PAGE_REF / "pedigree_map"
BLOODLINE_VECTOR_DIR = PAGE_REF / "bloodline_vector" / "v2_l2"
TABLES_DIR      = PAGE_REF / "tables"
RACE_PERFORMANCE_DIR = PAGE_REF / "race_performance"

# -- Knowledge / baselines (page-reference) -----------------------------------
KNOWLEDGE_DIR   = PAGE_REF / "knowledge"
TRACK_SPEED_BASELINES    = KNOWLEDGE_DIR / "track_speed_baselines.parquet"
TRACK_SPEED_PACE_BASELINES = KNOWLEDGE_DIR / "track_speed_pace_baselines.parquet"
COURSE_PROFILES_JSON     = KNOWLEDGE_DIR / "course_profiles.json"
SIRE_APTITUDE_NOTE_JSON  = KNOWLEDGE_DIR / "sire_aptitude_note.json"
MYOSTATIN_GENES_JSON     = KNOWLEDGE_DIR / "myostatin_genes.json"
PERFORMANCE_GENES_JSON   = KNOWLEDGE_DIR / "performance_genes.json"

# -- JRA baba (page-reference) ------------------------------------------------
JRA_BABA_DIR    = PAGE_REF / "cushion"
CUSHION_VALUES_JSON = JRA_BABA_DIR / "cushion_values.json"

# -- Track speed (page-reference) ---------------------------------------------
TRACK_SPEED_RACES_DIR = PAGE_REF / "track_speed"

# -- Research artifacts (pipeline / research scripts only) --------------------
PED_DIR         = LOCAL / "horse_pedigree_5gen"
PED_10GEN_DIR   = LOCAL / "horse_pedigree_10gen"
PED_10GEN_RESEARCH_DIR = LOCAL / "research" / "pedigree_10gen"
PED_10GEN_3VIEW_DIR    = LOCAL / "research" / "pedigree_10gen_3view"
BLOODLINE_DIR   = LOCAL / "research" / "bloodline"
HORSE_APTITUDE_DIR = LOCAL / "research" / "horse_aptitude"
COURSE_BLOODLINE_DIR = LOCAL / "research" / "course_bloodline"
RACE_LISTS_DIR  = LOCAL / "race_lists"

# -- Meta / operational -------------------------------------------------------
META_DIR        = LOCAL / "meta"
META_MODELING_DIR = META_DIR / "modeling"
META_PERSON_DIR = META_DIR / "person"
META_STRUCTURE_DIR = META_DIR / "structure"
META_LOGS_DIR   = META_DIR / "logs"
GLOBAL_SLOTS_DIR = META_DIR / "netkeiba_global_slots"

# -- Features (pipeline / ML) -------------------------------------------------
FEATURES_DIR    = LOCAL / "features"
FEATURES_INFO_DIR = LOCAL / "features_info"

# -- ML / modeling ------------------------------------------------------------
ML_DIR          = LOCAL / "ml"
ML_WAREHOUSE_DIR = ML_DIR / "warehouse"
MODELING_DIR    = LOCAL / "modeling"

# -- Analysis (pipeline / research scripts only) ------------------------------
ANALYSIS_DIR    = LOCAL / "analysis"
PEDIGREE_ANALYSIS_DIR = ANALYSIS_DIR / "pedigree"
