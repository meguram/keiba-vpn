# AREA-01: アプリケーション機能要件

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. ユーザー種別

| 種別 | 認証 | 閲覧可能機能 |
|------|------|-------------|
| **ゲスト** | 不要 | レース一覧・予測 TOP3 のみ |
| **ログイン済み** | セッション認証 | 全頭予測・分析・管理 UI |
| **管理者** | ログイン済み + 内部 IP | 管理 API・手動スクレイプ操作 |

---

## 2. 機能要件 (F-*)

### F-01: レース一覧表示

- 当日・翌日のレース一覧を表示する
- 開催場・レース番号・発走時刻・距離・コース種別を表示する
- データソース: `race_lists` + `race_shutuba`（GCS）
- 発走カウントダウンをリアルタイム更新する

### F-02: レース詳細・出馬表

- 出走馬・騎手・調教師・斤量・馬齢・戦績サマリを表示する
- データソース: `race_shutuba` + `race_shutuba_past`（GCS）

### F-03: AI 予測表示

- 着順予測（LightGBM LambdaRank スコア）を表示する
- 予測対象モデル:
  - `keiba_lgbm`: 複勝圏確率・着順ランク
  - `tracking_difficulty`: 追走難度スコア
  - `final_odds`: 単勝・複勝オッズ予測
  - `pace_predictor`: ペース予測
- ゲスト: TOP3 のみ表示。ログイン済み: 全頭表示

### F-04: オッズ表示

- 単勝・複勝・馬連・ワイド・馬単・3連複・3連単を表示する
- データソース: `race_odds` + `race_pair_odds`（GCS）
- 当日レース: SLA 3（T-15）取得データ

### F-05: バリューベット判定

- 予測確率 vs. オッズから期待値を計算し、高期待値馬を強調表示する
- 条件: `expected_win_roi ≥ 100` または `expected_show_roi ≥ 100`

### F-06: 馬情報表示

- 馬プロフィール（血統・馬齢・性別）を表示する
- 直近成績履歴（`horse_result`）を表示する
- 調教情報（`horse_training`）を表示する
- データソース: `horse_result` + `horse_pedigree_5gen`（GCS）

### F-07: 血統分析

- 種牡馬統計・コース適性を表示する
- 血統ベクトル類似度・ROI 分析
- データソース: `research/pedigree/` による分析結果

### F-08: スクレイピング監視（管理者向け）

- `/monitor` ページで GCS データ存在マトリクスを表示する
- 日付 × レース × カテゴリの存在状況（✓/✗/−）を視覚化する
  - `✓`: データあり
  - `✗`: 未取得
  - `−`: 取得不可（N/A マーカー）
- カラム選択で欠損データの一括スクレイプを UI からトリガできる

### F-09: キュー管理（管理者向け）

- `/queue-status` でジョブキュー状態をリアルタイム表示する
- ジョブの再キュー・失敗リキュー・一時停止操作が可能
- スキーマ検証失敗件数 (`schema_validation_failures`) を表示する

### F-10: cron ジョブ管理（管理者向け）

- `/cron-jobs` で現在登録されている cron を一覧表示する
- 各ジョブの最終実行時刻・ログを表示する

### F-11: 馬券最適化（Phase 3）

- `BettingOptimizer`: Kelly 基準による馬券種別・賭け金推奨
- `CompositeOptimizer`: パラメータ最適化（グリッドサーチ）
- 対応馬券種: 単勝 / 複勝 / 馬連 / ワイド / 馬単

---

## 3. 分析機能要件 (AN-*)

### AN-01: 種牡馬分析

- コース・距離・馬場状態別の種牡馬成績統計
- 種牡馬系統グラフ（血統ツリー可視化）
- サイア・クラスタリング（L2/L3 クラスタ）

### AN-02: コース・馬場分析

- コース別トラック速度指数
- 馬場バイアス（外枠・内枠有利度）
- レース質指数（8 軸 NNLS モデル）

### AN-03: 騎手・調教師統計

- コース別・距離別成績集計
- データソース: `jt_race_features`（Parquet）
- 更新: 毎日 JST 05:30（`build_jockey_trainer_stats`）

### AN-04: 馬適性プロファイル

- コース・距離・馬場適性スコア
- myostatin 遺伝子型（CC/CT/TT）と距離適性の相関
- 成長曲線予測（`growth_curve_service.py`）

---

## 4. データ鮮度インジケータ

- API レスポンスに `data_status` フィールドを付与する
- 値: `fresh` / `stale` / `unavailable` / `computing`
- `unavailable`: N/A マーカーが付いたデータ（2020-2022 の `race_barometer` 等）

---

## 5. スクレイピング要件（取得対象）

`docs/requirements/data/scrape_process.md` が単一の真実の源泉（SSoT）。
主要カテゴリを以下に要約する。

### レース系

| カテゴリ | 内容 | SLA |
|----------|------|-----|
| `race_lists` | 開催日のレース ID 一覧 | SLA 0 |
| `race_shutuba` | 出馬表（出走馬・騎手・斤量等） | SLA 1 |
| `race_shutuba_past` | 馬柱（過去成績詳細） | SLA 1 |
| `race_oikiri` | 追い切り情報 | SLA 1 |
| `race_odds` | 単勝/複勝/枠連オッズ | SLA 1/3 |
| `race_pair_odds` | 馬連/ワイド/馬単オッズ | SLA 1/3 |
| `race_result_on_time` | 速報結果（発走+15分） | SLA 4 |
| `race_result` | 確定結果（DB 版） | SLA 6 |
| `race_result_lap` | ラップタイム | SLA 6 |
| `race_index` | タイム指数 | SLA 6 |
| `race_barometer` | 走行指数 (AJAX API) | SLA 6（2023〜） |
| `race_paddock` | パドック情報 | SLA 3（要ログイン） |
| `race_trainer_comment` | 調教師コメント | SLA 1（要ログイン） |
| `smartrc_race` | SmartRC 独自指標 | SLA 1/3 |

### 馬系

| カテゴリ | 内容 | 更新 |
|----------|------|------|
| `horse_result` | 馬プロフィール + 戦績 | `weekly-update` |
| `horse_pedigree_5gen` | 5 代血統 JSON | `weekly-update` |
| `horse_training` | 調教データ | SLA 1（要ログイン） |

### JRA 系

| カテゴリ | 内容 |
|----------|------|
| `jra_cushion` | クッション値（PDF/ライブ） |
| `jra_baba_live` | 馬場情報ライブ（`jra-baba-morning`） |

---

## 6. N/A マーカー仕様

歴史的に取得不可能なデータは空データとして保存し、未取得と区別する。

| 条件 | 処理 |
|------|------|
| race_barometer が存在しない AND 開催日が 30 日以上前 | N/A スタブ JSON を GCS に保存 |
| 2020-01-01 〜 2022-12-31 の race_barometer | すべて N/A（データ提供前） |

- GCS パス: `chuou/data/preprocessed/netkeiba/pc/race_barometer/{YYYY}/{race_id}.json`（`{"_not_available": true, ...}`）
- ローカルインデックス: `data/local/meta/not_available/{YYYY}/{YYYYMMDD}.json`
- UI 表示: `−`（ダッシュ）
