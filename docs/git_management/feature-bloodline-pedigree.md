# feature/bloodline-pedigree

**対象領域**: 血統・種牡馬クラスタ
**分岐元**: `main` @ `e7e9e17`（2026-09-30）
**対象エンドポイント数**: 65（`src/api/app.py`, FastAPI legacy, :8000）

このブランチは2026-09-30時点で `main` から分岐した時点では差分を持たない（`main` と同一コミット）。今後、下表のエンドポイント群に関する変更はこのブランチで行い、レビュー後に `main` へPRでマージすること。

## 対象エンドポイント

| Method | Path | Handler |
|---|---|---|
| GET | `/api/admin/bloodline-cluster/artifact-status` | `api_admin_bl_artifact_status` (L10894) |
| GET | `/api/admin/bloodline-cluster/job-status` | `api_admin_bl_job_status` (L10987) |
| POST | `/api/admin/bloodline-cluster/rebuild/{target}` | `api_admin_bl_rebuild` (L10937) |
| POST | `/api/admin/bloodline-cluster/reload` | `api_admin_bl_reload` (L10993) |
| GET | `/api/bloodline-cluster/clusters` | `api_bloodline_cluster_clusters` (L10708) |
| GET | `/api/bloodline-cluster/horse-aptitude` | `api_bloodline_cluster_horse_aptitude` (L11278) |
| GET | `/api/bloodline-cluster/horse-name-suggest` | `api_bloodline_cluster_horse_name_suggest` (L11304) |
| GET | `/api/bloodline-cluster/lookup` | `api_bloodline_cluster_lookup` (L10684) |
| GET | `/api/bloodline-cluster/lookup-by-id` | `api_bloodline_cluster_lookup_id` (L10691) |
| GET | `/api/bloodline-cluster/meta` | `api_bloodline_cluster_meta` (L10669) |
| POST | `/api/bloodline-cluster/reload` | `api_bloodline_cluster_reload` (L10723) |
| GET | `/api/bloodline-cluster/sire-best-conditions` | `api_bloodline_cluster_sire_best_conditions` (L11201) |
| GET | `/api/bloodline-cluster/sire-heatmap` | `api_bloodline_cluster_sire_heatmap` (L11169) |
| GET | `/api/bloodline-cluster/sire-info` | `api_bloodline_cluster_sire_info` (L10698) |
| GET | `/api/bloodline-cluster/sire-presence-horses` | `api_bloodline_cluster_sire_presence_horses` (L11228) |
| GET | `/api/bloodline-cluster/sire-presence-stats` | `api_bloodline_cluster_sire_presence_stats` (L11065) |
| GET | `/api/bloodline-cluster/sire-summary-card` | `api_bloodline_cluster_sire_summary_card` (L11191) |
| GET | `/api/bloodline-cluster/stats` | `api_bloodline_cluster_stats` (L11020) |
| GET | `/api/bloodline-cluster/suggest` | `api_bloodline_cluster_suggest` (L10715) |
| GET | `/api/bloodline-cluster/tags` | `api_bloodline_cluster_tags` (L11005) |
| POST | `/api/bloodline/analyze` | `api_bloodline_analyze` (L10437) |
| GET | `/api/bloodline/data/{analysis_type}` | `api_bloodline_data` (L10555) |
| GET | `/api/bloodline/status` | `api_bloodline_status` (L10503) |
| GET | `/api/bloodline/surfaces` | `api_bloodline_surfaces` (L10544) |
| POST | `/api/course-bloodline/analyze` | `api_course_bloodline_analyze` (L11375) |
| GET | `/api/course-bloodline/data/{analysis_type}` | `api_course_bl_data` (L11448) |
| GET | `/api/course-bloodline/status` | `api_course_bl_status` (L11429) |
| GET | `/api/course-bloodline/surfaces` | `api_course_bl_surfaces` (L11440) |
| GET | `/api/course-profiles` | `api_course_profiles` (L11357) |
| GET | `/api/pedigree-map` | `api_pedigree_map` (L9288) |
| GET | `/api/pedigree-map/cluster-hierarchy` | `api_pedigree_map_cluster_hierarchy` (L9334) |
| GET | `/api/pedigree-map/condition-ranking` | `api_pedigree_map_condition_ranking` (L11100) |
| GET | `/api/pedigree-map/progeny-under-condition` | `api_pedigree_map_progeny_under_condition` (L11133) |
| GET | `/api/pedigree-map/tags` | `api_pedigree_map_tags` (L9318) |
| GET | `/api/pedigree-map/tags-full` | `api_pedigree_map_tags_full` (L9347) |
| GET | `/api/pedigree-race-stats/lineage-meta` | `api_pedigree_race_stats_lineage_meta` (L13314) |
| GET | `/api/pedigree-race-stats/meta` | `api_pedigree_race_stats_meta` (L12972) |
| GET | `/api/pedigree-race-stats/query` | `api_pedigree_race_stats_query` (L12999) |
| POST | `/api/pedigree/batch-race-ensure-5gen` | `api_pedigree_batch_race_ensure_5gen` (L10372) |
| GET | `/api/pedigree/note-aptitude` | `api_pedigree_note_aptitude` (L10002) |
| GET | `/api/pedigree/note-aptitude/table` | `api_pedigree_note_aptitude_table` (L10053) |
| POST | `/api/pedigree/race-ensure-5gen` | `api_pedigree_race_ensure_5gen` (L10295) |
| POST | `/api/pedigree/race-ensure-5gen/cancel` | `api_pedigree_race_ensure_5gen_cancel` (L10345) |
| GET | `/api/pedigree/race-ensure-5gen/status` | `api_pedigree_race_ensure_5gen_status` (L10326) |
| GET | `/api/pedigree/race-note-3d` | `api_pedigree_race_note_3d` (L10077) |
| GET | `/api/pedigree/race-note-3d-compare` | `api_pedigree_race_note_3d_compare` (L10131) |
| GET | `/api/pedigree/race-note-3d-v2` | `api_pedigree_race_note_3d_v2` (L10183) |
| POST | `/api/pedigree/rebuild-sire-factor-stats` | `api_rebuild_sire_factor_stats` (L10221) |
| POST | `/api/pedigree/tune-weights` | `api_pedigree_tune_weights` (L10265) |
| GET | `/api/pedigree/week-races` | `api_pedigree_week_races` (L10207) |
| GET | `/api/stallion-sire-tree` | `api_stallion_sire_tree` (L9368) |
| GET | `/api/stallion-sire-tree/bms-stats/{horse_id}` | `api_stallion_sire_tree_bms_stats` (L9923) |
| GET | `/api/stallion-sire-tree/l1-groups` | `api_stallion_sire_tree_l1_groups` (L9785) |
| GET | `/api/stallion-sire-tree/node/{horse_id}` | `api_stallion_sire_tree_node` (L9894) |
| POST | `/api/stallion-sire-tree/rebuild` | `api_stallion_sire_tree_rebuild` (L9742) |
| GET | `/api/stallion-sire-tree/rebuild/status` | `api_stallion_sire_tree_rebuild_status` (L9760) |
| GET | `/api/stallion-sire-tree/roots` | `api_stallion_sire_tree_roots` (L9846) |
| GET | `/api/stallion-sire-tree/search` | `api_stallion_sire_tree_search` (L9979) |
| GET | `/bloodline` | `bloodline_page` (L10428) |
| GET | `/bloodline-cluster` | `bloodline_cluster_page` (L10660) |
| GET | `/bloodline-vector` | `bloodline_vector_page` (L9237) |
| GET | `/course-bloodline` | `course_bloodline_page_redirect` (L11346) |
| GET | `/note-aptitude-race` | `note_aptitude_race_page` (L9274) |
| GET | `/pedigree-map` | `pedigree_map_page` (L9265) |
| GET | `/pedigree-race-stats` | `pedigree_race_stats_page` (L12964) |
