# DEC-TEMPLATE: 設計決定テンプレート

このテンプレートを使って設計決定（Decision）を記録する。

---

## DEC-XXX: 〈決定タイトル〉

| 項目 | 内容 |
|------|------|
| **日付** | YYYY-MM-DD |
| **ステータス** | proposed / accepted / deprecated / superseded |
| **担当** | — |
| **関連 AREA** | AREA-01〜09 のどれ |
| **関連 DEC** | DEC-XXX（依存がある場合） |

---

### コンテキスト

なぜこの決定が必要になったか。現状と問題点を説明する。

---

### 決定事項

何を決めたか。具体的に記述する。

```python
# 決定の内容をコードで示す場合（任意）
```

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| A（採用） | ... | ... |
| B | ... | ... |

---

### 影響範囲

- 変更が必要なファイル
- テストへの影響
- cron への影響（重要: 現行 cron を壊さないこと）

---

### 備考

追加情報・参考リンク等。

---

## 記録済み設計決定

| DEC | 内容 | ステータス |
|-----|------|-----------|
| DEC-001 | GCS + HybridStorage を唯一のデータ正本とする | accepted |
| DEC-002 | スクレイプは ScrapeJobQueue 経由に統一（cron は直接スクレイプしない） | accepted |
| DEC-003 | テンポラルリーク防止のため `as_of_race_id` でスナップショット管理 | accepted |
| DEC-004 | PostgreSQL を採用しない（GCS + Parquet で代替） | accepted |
| DEC-005 | Redis を採用しない（ローカルディスクキャッシュで代替） | accepted |
| DEC-006 | 大衆指標（オッズ・人気）は学習特徴量から除外 | accepted |
| DEC-007 | スキーマ検証を保存時必須とし、不合格時は GCS に書かない | accepted |
| DEC-008 | N/A マーカー: 取得不可データはスタブ JSON + ローカルインデックスで管理 | accepted |
| DEC-009 | `_ensure_race_list_date` はキュー全体待機をせずローカル/GCS をポーリング | accepted |
| DEC-010 | cron の管理は `setup_all_cron.sh` 一本化。手動編集禁止 | accepted |
| DEC-011 | race_barometer は SLA 6（翌週金曜）。2020-2022 は N/A 扱い | accepted |
| DEC-012 | アーカイブ移行: リファクタリング前のコードを `archive/` に移動して保持 | accepted |
