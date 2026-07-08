# DEC-018: フロントエンド: D3.js 依存追加と未実装ページ対応

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | ~~accepted~~ → **superseded（一部廃止）** |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-02 |
| **矛盾ID** | S-6-A/B/C |
| **後続変更** | 2026-07-08: `/pedigree-map`・`/bloodline-cluster` 削除済み（下記参照） |

---

### コンテキスト

AREA-02 §5-1 は D3.js v7 を必須として血統マップ（`/pedigree-map`）に使用すると定義しているが、`frontend/package.json` に `d3` が未記載で `npm install` 後に使えない状態。また `/bloodline`（14分析タイプ）と `/ai-sla` の2ページが未実装で 404 になる。Next.js の API リライト先ポートはフロントエンドのデフォルトが 5000（Flask）に設定されており、FastAPI（8000）との混在時に混乱が生じる可能性がある。

---

### 決定事項

1. `frontend/package.json` に `d3@^7` を追加する。
2. `/bloodline` は空の `page.tsx`（"under construction" 表示）を作成し URL を有効化する。
3. `/ai-sla` は AREA-04 の Cron SLA 表をそのまま表示するシンプルな静的ページとして実装する。
4. `frontend/next.config.js` の `KEIBA_API_URL` デフォルトは `http://127.0.0.1:5000`（Flask）を維持する（DEC-013 と整合）。

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| D3追加 + placeholder（採用） | 依存欠落解消・ルーティング有効化 | 血統マップ完全実装は別タスク |
| 現状維持 | 工数ゼロ | 血統マップが実行時エラー・ページが 404 |

---

### 影響範囲

- `frontend/package.json` に `"d3": "^7.0.0"` 追加
- `frontend/app/bloodline/page.tsx` 新規作成（placeholder）
- `frontend/app/ai-sla/page.tsx` 新規作成（Cron SLA 静的表示）

---

### 備考

血統マップの完全な D3.js フォースグラフ実装は Phase 3 以降の作業。本 DEC は依存関係欠落とルーティングの404を解消することのみを対象とする。

---

### 後続変更（2026-07-08）

本 DEC が対象とした2ページは、その後 UI として不要と判断され削除された。

| ページ | 対応 |
|--------|------|
| `/pedigree-map`（血統構造マップ） | `frontend/app/pedigree-map/` ディレクトリごと削除。D3.js フォースグラフ実装は未着手のまま廃止。 |
| `/bloodline-cluster`（メタクラスタ判定） | `frontend/app/bloodline-cluster/` ディレクトリごと削除。 |

> この変更により本 DEC の決定事項「2. `/bloodline` は空の `page.tsx` を作成し URL を有効化する」のうち `/pedigree-map` と `/bloodline-cluster` に関する部分は実質無効となった。D3.js 依存の追加（決定事項1）および `/ai-sla` の静的ページ実装（決定事項3）は引き続き有効。
