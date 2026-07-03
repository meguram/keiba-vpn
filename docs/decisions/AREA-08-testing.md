# AREA-08 — テスト要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-009(CI ゲート), DEC-010(CI lint), 新規追加

---

## 1. テスト戦略方針

| 原則 | 内容 |
|---|---|
| テストピラミッド | Unit > Integration > E2E の順で比重を置く |
| CI 必須 | 全テストが通過しなければ main へのマージ禁止 |
| 環境分離 | dev/stg はローカル PC、prod は VPS（詳細 → AREA-09） |
| テンポラルリーク防止 | バリデーション日 strict カットオフをテストで保証 |

---

## 2. 単体テスト（Unit Tests）

### 対象コンポーネント

| コンポーネント | ツール | 主要テスト項目 |
|---|---|---|
| Flask API エンドポイント | pytest + unittest.mock | レスポンス形式, ステータスコード, 認証ガード |
| ModelRegistry | pytest | ロード, hot-reload, RWLock スレッド安全性 |
| Pandera バリデーション | pytest | 境界値（距離 800/3600m, 頭数 2/18, 馬体重 380/620kg） |
| ETL 変換ロジック | pytest | データ変換・正規化の正確性 |
| GCS パス生成 | pytest | `test_gcs_paths.py` でパス形式を全件検証 |
| Circuit Breaker | pytest | OPEN/HALF-OPEN/CLOSED 状態遷移 |
| キャッシュ TTL | pytest + fakeredis | TTL 設定値, auto-expire 動作 |
| Next.js コンポーネント | Vitest + React Testing Library | props rendering, data_status バナー |

### カバレッジ目標

| レイヤー | 目標カバレッジ |
|---|---|
| バックエンド（API + ロジック） | ≥80% |
| 推論バッチ（ModelRegistry, ETL guard） | ≥90% |
| フロントエンド（主要コンポーネント） | ≥70% |

---

## 3. 統合テスト（Integration Tests）

| テスト | 内容 |
|---|---|
| ETL → Redis フラグ | ETL 完了後 `etl:complete:{date}` が Redis に存在することを確認 |
| 推論バッチ → Redis + PostgreSQL | バッチ実行後 predictions が Redis と DB の両方に書き込まれること |
| Flask → Redis キャッシュ | API 呼び出しで L2 キャッシュがヒットすること（fakeredis または実 Redis） |
| Circuit Breaker → GCS フォールバック | OPEN 時に GCS データが返却されること |
| 4 層キャッシュ連鎖 | L2 miss → L3 hit のパス確認 |
| Flask-Login セッション | ログイン/ログアウト/`@login_required` 動作確認 |

---

## 4. E2E テスト

| ツール | Playwright（TypeScript） |
|---|---|
| 実行環境 | dev/stg ローカル PC（→ AREA-09）|
| 主要シナリオ | |

| シナリオ | 確認内容 |
|---|---|
| ゲストユーザー閲覧 | `/api/v1/races/today` で TOP3 のみ表示 |
| ログイン → 全頭表示 | ログイン後に予測 TOP3 以降が表示される |
| ETL stale バナー | `data_status: stale` 時に黄色バナーが表示される |
| T-10分カウントダウン | 発走 10 分前に赤字カウントダウンが表示される |
| Circuit OPEN バナー | フォールバック時に「データ更新中」バナーが出る |
| PWA ホーム追加 | manifest.json が正しく返却される |

---

## 5. AI / ML テスト

| テスト | 内容 | ツール |
|---|---|---|
| テンポラルリーク検証 | テスト特徴量にバリデーション日以降のデータが混入していないこと | pytest + pandera |
| モデル精度回帰 | Logloss ≤2.2, Calibration ≤0.05, ROI ≥-15% を保持 | pytest + mlflow または手動ログ |
| 推論ベンチマーク | 現実的な 18 頭特徴量で P99 ≤200ms | pytest benchmark |
| モデルサイズ CI ゲート | `assert model_size_mb < 50` | pytest（CI 必須）|
| SHAP 値の符号検証 | 主要特徴量の寄与方向が期待値と一致すること | pytest |

---

## 6. CI ゲート（GitHub Actions / CI パイプライン）

```yaml
# 全 PR に必須チェック
- lint:
    - ruff（Python）: Gunicorn 内 predict() 呼び出し禁止ルール含む
    - ESLint / TypeScript（Next.js）
    - GCS パスハードコード禁止チェック（gcs_paths.py SSoT 検証）
- unit-tests:
    - pytest（バックエンド）
    - vitest（フロントエンド）
- integration-tests:
    - pytest（ETL→Redis, キャッシュ連鎖）
- model-gates:           # main ブランチへの push 時のみ
    - assert model_size_mb < 50
    - 推論ベンチマーク（18 頭, P99 ≤200ms）
```

### ブロッキング条件

| 条件 | ブロック対象 |
|---|---|
| lint 失敗 | 全 PR |
| unit test 失敗 | 全 PR |
| GCS パスハードコード検出 | 全 PR |
| モデルサイズ超過（>50MB） | main マージ |
| 推論 P99 超過（>200ms） | main マージ |
| テンポラルリーク検出 | Cloud Run Jobs 学習ジョブ |

---

## 7. テストデータ管理

| データ種別 | 管理方法 |
|---|---|
| ユニットテスト用フィクスチャ | `tests/fixtures/` に静的ファイルとして管理 |
| 現実的推論ベンチマーク | `realistic_18horses_features()` ファクトリ関数 |
| E2E テスト用シードデータ | dev/stg 環境の PostgreSQL にシードスクリプトで投入 |
| 本番データの使用 | テストへの流用禁止（個人情報保護） |

---

## 8. 非機能要件（テスト系）

| ID | 要件 |
|---|---|
| NFR-TEST-01 | CI 実行時間 ≤10 分（unit + lint）|
| NFR-TEST-02 | E2E テスト実行時間 ≤15 分 |
| NFR-TEST-03 | テスト環境は prod データを使用しない |
| NFR-TEST-04 | モデル学習テストは Cloud Run Jobs で実行（ローカル学習禁止は prod のみ、dev は許可）|
