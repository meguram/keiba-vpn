# DEC-015: APIエンドポイントパス統一: /races/ 複数形に統一

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-02, AREA-03 |
| **矛盾ID** | S-3-A/B |

---

### コンテキスト

AREA-02 §7 が GET /api/v1/race/{id}/tracking-difficulty（単数形）と記載しているが、実装の flask_app.py は /api/v1/races/<race_id>/tracking-difficulty（複数形）を使用。他の全レース系エンドポイントは複数形で統一されており、AREA-02 が誤記。また src/api/auth.py の PUBLIC_API_PREFIXES に /api/v1/ を含まない旧パス体系が残存している。

---

### 決定事項

(1) AREA-02 §7 の /api/v1/race/{id}/tracking-difficulty を /api/v1/races/{race_id}/tracking-difficulty に修正。(2) src/api/auth.py の PUBLIC_API_PREFIXES を /api/v1/ ベースに全面更新。

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| 実装（複数形）を正とする（採用） | 実装変更不要・全エンドポイント一貫 | |
| 仕様書（単数形）を正とする | 意味的明確 | 実装変更コスト大 |

---

### 影響範囲

- docs/decisions/AREA-02-frontend.md §7 の /api/v1/race/{id}/ を /api/v1/races/{race_id}/ に修正
- src/api/auth.py の PUBLIC_API_PREFIXES を /api/v1/ ベースに更新

---

### 備考

フロントエンドの lib/api.ts がエンドポイントを呼ぶ際も /races/ 複数形を使うよう確認が必要（現在は AREA-02 の誤記を参照している可能性）。
