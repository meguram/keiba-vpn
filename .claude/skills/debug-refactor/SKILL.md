---
name: debug-refactor
description: Find and fix bugs/errors, or refactor code, in the keiba-vpn repo following this project's specific conventions (temporal-leak safety, HybridStorage singleton usage, Flask-as-primary-API, module layout in AGENTS.md). Use when the user asks to debug a failing test, fix an error/bug, investigate unexpected behavior, or refactor/clean up existing code in this repo.
---

このリポジトリ（keiba-vpn）でバグ修正・エラー調査・リファクタリングを行うための手順です。
一般的な良いプラクティスに加え、このプロジェクト特有の落とし穴を必ず確認してください。

## 0. 最初にやること

1. `AGENTS.md` を確認済みでなければ読む（レイアウト表・命名規則・作業の指針）。
2. 問題を再現するテストがあるか探す: `grep -rln "<関連キーワード>" tests/`
3. まだ再現できないなら、修正前に**再現テスト（またはピンポイントの手動確認）を先に用意する**。

## 1. テストの一括実行

```bash
make test              # CIと同じ順序・除外設定。DB/Redis未起動でも大半は動く
make test-frontend     # frontend/ の lint + build
make test-all          # 上記両方
```

個別ファイルだけ動かす場合:
```bash
python3 -m pytest tests/pipeline/test_xxx.py -v --tb=short
```

DB接続が必要なテストだけ失敗する場合は `make db-up && make db-migrate` を先に実行する
（`DATABASE_URL` / `REDIS_URL` は `.env` を参照）。

## 2. このプロジェクト特有の落とし穴（必ずチェック）

修正・リファクタリング対象のコードが以下に該当しないか確認する。当てはまる場合は原則を破らないこと。

| チェック項目 | 原則 | 違反した場合の症状 |
|---|---|---|
| **テンポラルリーク** | `as_of_race_id` より未来の情報を特徴量・学習データに含めない。`get_snapshot(race_id, as_of=race_id)` は `window_end=as_of_race_id` を必須で渡す | CIの `python -m src.scripts.ci.check_no_shuffle` や temporal leak テストで検知される。本番投入後に発覚すると重大 |
| **`train_test_split(shuffle=True)`** | 時系列データは絶対にランダムシャッフルしない | `check_no_shuffle` がブロック。学習・評価の前提が崩れる |
| **HybridStorage のインスタンス化** | `src/api/v1/services.py` / `delegates.py` / `flask_app.py` で新しいコードを書くとき、リクエストハンドラ内で `HybridStorage()` を直接 `new` **しない**。既知の問題（`docs/html/design/FULLSTACK_ARCHITECTURE.html` §4 P1）として、Flask側は現在リクエスト毎に再生成しておりL1メモリキャッシュが機能していない。新規コードでこれを悪化させないこと。可能なら `src/api/app.py` の `_get_storage()` と同様のプロセス単位シングルトンパターンを使う | L1キャッシュが効かず、GCS呼び出し・レイテンシが不要に増える |
| **Flask (`/api/v1`) が仕様上の正** | DEC-013により、新規APIエンドポイントは `src/api/flask_app.py` / `src/api/v1/` 側に実装する。`src/api/app.py`（FastAPI, レガシー）には追加しない | 段階的廃止予定のレイヤーに機能が増え、移行コストが増す |
| **GCS保存前スキーマ検証** | `HybridStorage.save` は `schemas.validate` を必ず実行する（`KEIBA_SCHEMA_STRICT` 既定1）。保存データの形を変える変更は `data/requirements/data/schemas/json/` のスキーマ側も更新する | 保存が `SchemaValidationError` で失敗し、キューが `failure_reason=schema_validation` になる |
| **モジュール境界** | `AGENTS.md` のレイアウト表に沿う。`src/scraper/` `src/pipeline/` `src/research/` `src/api/` `src/scripts/` の役割を混在させない | 既存の命名・分割規則から逸脱し、後続の変更が追跡しにくくなる |
| **馬券戦略との整合** | モデリング関連の変更（評価指標・CIゲート等）を行う場合は `docs/decisions/DEC-026-modeling-betting-strategy-alignment.html` を確認。キャリブレーション・市場対比評価の方針と矛盾しないか確認する | モデルの精度指標だけを見て、馬券適用時のEV/Kelly計算に悪影響を与える変更を通してしまう |

## 3. バグ修正の進め方

1. **再現**: 上記の再現テスト、または `python3 -c "..."` での最小再現を用意する。
2. **原因特定**: 該当モジュールを `AGENTS.md` のレイアウト表から特定し、`Read` で該当ファイルを読む。巨大な `data/` の丸読みは避け、`Grep`/`Glob` で絞り込む。
3. **修正**: 依頼された範囲だけを直す。関連しない周辺のリファクタリングは同じ変更に混ぜない（別途リファクタリングとして提案する）。
4. **検証**: 再現テストが通ることを確認 → `make test` で全体に regression が無いか確認。
5. **後片付け**: 一時的なデバッグ用の `print`/コメントを削除する。

## 4. リファクタリングの進め方

1. **スコープを明確にする**: 「何を」「なぜ」変えるかを一文で言えるか確認する。言えなければ着手しない。
2. **既存の命名・分割に合わせる**: 新しい抽象化・パターンを持ち込む前に、同じ役割の既存コードがどう書かれているかを確認する（例: 同じ `src/pipeline/features/` 内の他ファイル）。
3. **参照整合性を確認する**: 関数・クラス・ファイルをリネーム/移動する場合は、呼び出し元をすべて `grep -rn` で洗い出してから変更する。テスト・ドキュメント（`docs/html/`, `docs/decisions/`）内のコード例も対象に含める。
4. **小さく分割する**: 挙動を変えないリファクタリングと、挙動を変える修正は別のコミットに分ける（同じPR内でも diff を分けて説明する）。
5. **検証**: `make test` で既存のテストが全て通ることを確認する。振る舞いを変えないリファクタリングでテストが落ちた場合は、リファクタリングそのものに問題がある可能性が高い。

## 5. 既知の未解決課題（このSkillの対象になりやすい）

以下は調査済みで、今後の修正候補として記録されている問題。関連する変更を依頼された場合は参照する。

- `docs/html/design/FULLSTACK_ARCHITECTURE.html` §4: Flask側 HybridStorage/PredictionCache のシングルトン化未対応、L2ディスクキャッシュの週次アクセスゲート、Redis `race:entries`/`race:results` 未実装
- `docs/decisions/DEC-026-modeling-betting-strategy-alignment.html`: モデリング評価指標にキャリブレーション・市場対比評価が未導入
- `tests/utils/test_race_list_for_date.py` 他4件: `make test` 実行時にDB無しでも失敗する既知の未解決テスト（本Skill作成時点で原因未特定。着手する場合はまずこの5件の失敗原因を切り分けること）
