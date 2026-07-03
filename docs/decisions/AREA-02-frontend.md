# AREA-02 — フロントエンド要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-001(技術選定), DEC-005(FE要件), DEC-007(ADR-005), DEC-010(FE実装)

---

## 1. 技術選定

| 項目 | 決定内容 | 根拠 |
|---|---|---|
| フレームワーク | TypeScript + Next.js 14（App Router） | SSR/ISR/CSR 混在対応、DEC-001 から一貫 |
| ホスティング | Vercel Hobby（¥0/月） | CDN 配信・自動デプロイ、ADR-005 |
| スタイリング | Tailwind CSS + Recharts（グラフ） | 開発速度・バンドルサイズ |
| 商用化時 | Vercel Pro（¥2,500/月）へ移行 | 商用 SLA が必要になったタイミング |

---

## 2. レンダリング戦略

| ページ / データ | 方式 | 理由 |
|---|---|---|
| レース一覧 | ISR（revalidate: 300s） | 更新頻度低、キャッシュ効果大 |
| AI 予測結果ページ | ISR（revalidate: 3,600s） | 推論は 1 日 1〜数回 |
| 当日出馬表・当日レース | CSR | 直前変更（オッズ・馬体重）への対応 |
| 静的ページ（ルール説明等） | SSG | ビルド時生成で最速 |

---

## 3. UX 設計要件（FE-01〜12）

| ID | 要件 | 優先度 |
|---|---|---|
| FE-01 | モバイルファースト: ブレークポイント 375px / 768px / 1,280px | 高 |
| FE-02 | LCP ≤2.5s（3G 環境）, Lighthouse Performance ≥85（モバイル） | 高 |
| FE-03 | デフォルト表示: 今日のレース一覧（開催場・レース番号・発走時刻・予測 TOP3）| 高 |
| FE-04 | `data_status` バナー出し分け（fresh: 非表示 / stale: 黄警告 / unavailable: 赤）| 高 |
| FE-05 | 発走 T-10 分未満でカウントダウンを赤字表示 | 中 |
| FE-06 | 予測カード: 推奨馬券種別 + 上位 3 頭カード形式、推奨馬ハイライト | 中 |
| FE-07 | SHAP 特徴量説明: 折りたたみ表示（デフォルト非表示）| 中 |
| FE-08 | Skeleton UI（データ読み込み中の視覚フィードバック） | 中 |
| FE-09 | 仮想スクロール（react-window）: レース一覧・馬柱テーブル | 中 |
| FE-10 | win_probability vs confidence_score のツールチップ説明 | 中 |
| FE-11 | PWA: ホーム画面追加 + オフラインキャッシュ | 中（Phase 2） |
| FE-12 | opentelemetry 分散トレーシング連携 | 低（Phase 3） |

---

## 4. API 通信設計

- Next.js Server Components から VPS Flask API へは HTTPS 経由
- WireGuard VPN 経由での管理ページアクセス
- API エラー時: `data_status: unavailable` バナー + 前回キャッシュデータを表示

---

## 5. パフォーマンス最適化

| 施策 | 内容 |
|---|---|
| gzip 圧縮 | Nginx で有効化、転送量 -60% |
| 画像最適化 | Next.js Image コンポーネント使用 |
| バンドル分割 | dynamic import で必要なコンポーネントのみ遅延ロード |
| CDN キャッシュ | Vercel Edge Cache（ISR 生成済みページ） |
| 仮想スクロール | react-window で大量行テーブルを仮想化 |

---

## 6. フロントエンド実装フェーズ

| Phase | タスク |
|---|---|
| Phase 1 | レース一覧・出馬表・予測スコア表示の基本 UI |
| Phase 2 | 予測カード再設計, Skeleton UI, ISR 最適化, PWA |
| Phase 3 | Grafana/Lighthouse CI, opentelemetry フロント連携 |
