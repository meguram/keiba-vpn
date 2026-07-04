# DEC-014: DBスキーマ統一: alembic と AREA 仕様書の整合

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-01, AREA-03, AREA-06 |
| **矛盾ID** | S-2-A/B/C/D/E |

---

### コンテキスト

scrape_runs テーブルの target_type VARCHAR(50 vs 30)、target_id NULL許可 vs NOT NULL・VARCHAR(50 vs 20)、status の 'FAILURE' vs 'FAILED'、VARCHAR(20 vs 10) などの乖離がある。またAREA-01 の SQL サンプルに SQL 予約語 class とテーブル名 results（実: race_results）が誤記されている。AREA-06 の horse_stats_snapshot DDL に win_rate_going が欠落している。

---

### 決定事項

alembic/versions/001_initial_schema.py を正として仕様書を修正する（実装→仕様書方向の修正）。具体的には: (1) scrape_runs: target_type VARCHAR(30)、target_id VARCHAR(20) NOT NULL、status CHECK('SUCCESS','FAILED','RETRY') VARCHAR(10)。(2) AREA-01 §3-3 の SQL サンプルの class → race_class、results → race_results に修正。(3) AREA-06 §5-2 の horse_stats_snapshot DDL に win_rate_going を追加。

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| 実装→仕様書修正（採用） | alembic が migration 済みのため変更コスト最小 | |
| 仕様書→実装修正 | 仕様書を正にできる | migration 追加が必要・現行データに影響 |

---

### 影響範囲

- docs/decisions/AREA-01-app-requirements.md §3-3 の SQL 修正
- docs/decisions/AREA-03-backend.md §2-6 の scrape_runs 定義修正
- docs/decisions/AREA-06-data.md §5-2 に win_rate_going 追加

---

### 備考

target_id の NULL 許可を実装が撤廃した理由: バッチ全体スキャンなど ID 不要ケースは target_id="batch" 等の識別子で代替可能と判断。
