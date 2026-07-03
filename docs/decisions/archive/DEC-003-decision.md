# DEC-003: decision

**Date**: 2026-07-02
**Agent**: proposal-agent, backend-engineer, data-engineer, ai-model-engineer
**Task**: TASK-008
**Status**: ACCEPTED

---

## Context

「本番運用を考えた設計にしたいです。具体的には、多数のユーザからのアクセスに対応できる設計（例えば、ページのローディングをユーザ側のデバイスに依存するようにするといった対応など。）。理由としては、使用しているVPSのメモリが2GB上限となっているため、様々な機能（データ分析含む）を行う上で、このサーバ負荷の部分は最小限に抑えたい。」の改善要件定義書です。要件の優先順位と実装可否を確認してください。

---

## Decision

<response>

まず、プロジェクト管理ドキュメントを作成してから要件定義書を出力します。

---

## 📁 ドキュメント作成

### `project/keiba-vpn/tasks/TASK-001-server-load-minimization.md`

```markdown
# TASK-001: サーバ負荷最小化設計 — 要件定義書作成

- **Status**: DONE
- **Created**: 2025-01-30
- **Owner**: orchestrator
- **Related Agents**: backend-engineer, data-engineer, ai-model-engineer, frontend-engineer

## 概要
2GB VPS 上で多数ユーザのアクセスに耐えられるよう、
サーバ側処理をクライアント・外部ストレージ・キャッシュ層に委譲する設計を定義する。

## 完了条件
- [ ] 要件定義書の作成
- [ ] 決定事項ログの作成
- [ ] DASHBOARD.md の更新
```

---

### `project/keiba-vpn/docs/decisions/DEC-001-server-load-minimization.md`

```markdown
# DEC-001: サーバ負荷最小化アーキテクチャ決定

- **Date**: 2025-01-30
- **Status**: APPROVED
- **Agents Involved**: backend-engineer, data-engineer, ai-model-engineer

## 決定事項

1. **Celery によるオンデマンド AI 推論は採用しない**
   - 現設計はバッチ推論（夜間1回）であり Celery は不要かつメモリを逼迫させる
   - 将来的なオンデマンド化需要が生じた時点で再検討する

2. **Redis は 200MB 上限で導入し、eviction policy を必ず設定する**
   - maxmemory-policy: allkeys-lru
   - TTL は時間固定ではなくイベント駆動（スクレイプ完了時に能動 invalidate）と組み合わせる

3. **AI予測結果の Redis TTL は時間ベースではなくバッチ完了イベントで invalidate する**

4. **当日出馬表・当日分レースページは SSG ではなく CSR（Client-Side Rendering）とする**
   - 出走取消・騎手変更など直前変更への対応が必要なため

5. **GCS 署名付きURL には認証・パストラバーサル対策を必須とする**
```

---

### `project/keiba-vpn/docs/thinking-logs/2025-01-30-orchestrator-TASK-001.md`

```markdown
# Thinking Log — TASK-001 (Orchestrator)

## Round 1 → Round 2 で明らかになった矛盾
- 提案3（Celery）は「負荷を下げる提案」だが、2GB VPS に追加プロセスを乗せるため逆効果
- 現アーキテクチャはバッチ推論設計であり、オンデマンド非同期化の前提が誤っていた

## Round 2 → Round 3 で解決した課題
- TTL設計をイベント駆動に変更
- メモリ予算を明示的に管理（合計 ~1,800MB 以内）
- セキュリティ欠落（認証・パストラバーサル）の補完

## 統合上の判断
- 提案3は「Phase 3 / 将来検討」として要件から除外
- SSG/ISR の粒度をレース状態（過去/当日/予測結果）で三分割
```

---

## 📄 要件定義書

```markdown
# 改善要件定義書: 本番運用対応 — サーバ負荷最小化設計

- **Version**: 1.0
- **Date**: 2025-01-30
- **Task**: TASK-001
- **Decision**: DEC-001

---

## サマリー

2GB メモリ上限の VPS で多数ユーザのアクセスに耐えるため、
「サーバを経由させない」「キャッシュする」「クライアントに処理を委譲する」
の3原則に基づき、ファイル配信・APIキャッシュ・レンダリング戦略・
フロントエンド描画の4層で施策を実施する。
AI 推論はバッチ夜間処理のみとし、リアルタイム推論（Celery 等）は
当面導入しない（メモリ逼迫防止のための意図的な判断）。

---

## 機能要件

| # | 要件 | 優先度 | 担当エージェント |
|---|------|--------|----------------|
| F-1 | GCS 署名付きURL によるファイル直接配信（画像・CSV・分析レポート） | 高 | backend-engineer |
| F-2 | 署名付きURL エンドポイントへの JWT 認証・アクセス権チェック実装 | 高 | backend-engineer |
| F-3 | 署名付きURL 生成時のパストラバーサル対策（`os.path.normpath` + 先頭 `..` 拒否） | 高 | backend-engineer |
| F-4 | Redis キャッシュ層の導入（Flask APIレスポンスキャッシュ） | 高 | backend-engineer |
| F-5 | Redis キャッシュの能動的 invalidate（スクレイプ完了フックによるキャッシュ破棄） | 高 | data-engineer |
| F-6 | 発走時間に応じたオッズ TTL の動的制御（30分前:60s / 直前30分:15s / 直前5分:5s） | 高 | backend-engineer |
| F-7 | 管理者向けキャッシュ手動パージ API（`POST /api/admin/cache/invalidate`） | 中 | backend-engineer |
| F-8 | レースページのレンダリング戦略分離（過去レース:SSG / 当日レース:CSR / 予測結果:ISR） | 高 | frontend-engineer |
| F-9 | 当日出馬表・当日レースページは CSR（Client-Side Rendering）で実装（直前変更対応） | 高 | frontend-engineer |
| F-10 | レース一覧・馬情報など低更新頻度ページの SSG ビルド化 | 中 | frontend-engineer |
| F-11 | AI予測結果ページの ISR（`revalidate: 3600` 以上 or バッチ完了時オンデマンド再生成） | 中 | frontend-engineer |
| F-12 | レース一覧・馬柱テーブルへの仮想スクロール実装（`react-window` or `react-virtual`） | 中 | frontend-engineer |
| F-13 | API のページネーション対応（一度に返す件数を最大20件に制限） | 中 | backend-engineer |
| F-14 | 画像・グラフの遅延ロード（Intersection Observer / Next.js `Image` コンポーネント） | 中 | frontend-engineer |
| F-15 | GCS ファイルパス命名規則の統一（`races/{date}/{race_id}/` 形式）とETLパイプラインとの整合 | 中 | data-engineer |
| F-16 | バッチ推論完了後の AI 予測キャッシュ invalidate フック実装 | 中 | ai-model-engineer |
| F-17 | ページネーション用 DB インデックスの追加（`races.date DESC` 等） | 低 | backend-engineer |

---

## 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|------|--------|------|
| N-1 | VPS 常駐プロセスの合計メモリ使用量 | 1,800 MB 以下（バッファ 200MB 確保） | operations-engineer |
| N-2 | Redis メモリ上限 | 200 MB（`maxmemory 200mb`, `allkeys-lru`） | backend-engineer |
| N-3 | Flask/Gunicorn ワーカー数 | 最大3ワーカー（メモリ消費を ~400MB 以内に抑制） | operations-engineer |
| N-4 | キャッシュヒット時の API レスポンスタイム | 50 ms 以下 | operations-engineer |
| N-5 | キャッシュミス時の API レスポンスタイム | 500 ms 以下 | backend-engineer |
| N-6 | 署名付きURL の有効期限 | 15分（不必要に長くしない） | backend-engineer |
| N-7 | SSG ビルド対象ページの初期表示速度（FCP） | 1.5秒以下（静的配信のため） | frontend-engineer |
| N-8 | 仮想スクロール適用時の DOM 描画ノード数 | 表示領域 +上下バッファ 分のみ（最大30ノード程度） | frontend-engineer |
| N-9 | AI バッチ推論プロセスの実行方式 | 夜間1回のみ起動・完了後プロセス終了（常駐しない） | ai-model-engineer |
| N-10 | GCS アクセスの Flask 経由率（ファイル配信） | 0%（全て署名付きURLで直接配信） | backend-engineer |

---

## VPS メモリ予算（参考）

| プロセス | 想定使用量 | 備考 |
|---------|-----------|------|
| OS + システム | ~300 MB | |
| Flask / Gunicorn（workers×3）| ~400 MB | |
| PostgreSQL | ~300 MB | |
| Redis | ~200 MB | `maxmemory 200mb` で上限固定 |
| バッファ（突発スパイク） | ~300 MB | |
| AI モデル（バッチ実行時のみ） | ~300 MB | **常駐しない**・夜間のみ起動 |
| **合計（通常時）** | **~1,500 MB** | バッチ実行時でも ~1,800MB 以内 |

> ⚠️ **Celery Worker は含まない**。現アーキテクチャはバッチ推論のためオンデマンド非同期処理は不要。
> Celery の導入は「オンデマンド予測のユーザ需要が実際に発生した段階」で再検討する。

---

## 実装ロードマップ

### Phase 1（優先度: 高 — 即時着手）
> 目標: ファイル配信負荷の排除とAPIキャッシュの確立（1〜2週間）

- **F-1** GCS 署名付きURL によるファイル直接配信の実装
- **F-2** 署名付きURL エンドポイントへの JWT 認証・アクセス権チェック
- **F-3** パストラバーサル対策の実装
- **F-4** Redis キャッシュ層の導入（`maxmemory 200mb` + `allkeys-lru` 設定）
- **F-6** 発走時間に応じたオッズ TTL の動的制御
- **F-5** スクレイプ完了フックによる能動的キャッシュ invalidate

```
Phase 1 完了後の期待効果:
  - ファイル配信時の VPS メモリ使用量: ほぼゼロ
  - API レスポンスタイム（キャッシュヒット時）: 50ms 以下
  - 同時アクセス耐性: 大幅向上
```

### Phase 2（優先度: 中 — Phase 1 完了後）
> 目標: フロントエンド描画負荷のクライアント委譲（2〜3週間）

- **F-8** レースページのレンダリング戦略分離（SSG / CSR / ISR）
- **F-9** 当日出馬表・当日レースページの CSR 実装
- **F-10** 低更新頻度ページの SSG ビルド化
- **F-11** AI予測結果ページの ISR 実装（`revalidate: 3600`）
- **F-12** 仮想スクロールの実装（`react-window`）
- **F-13** API ページネーション対応（最大20件）
- **F-14** 画像・グラフの遅延ロード実装
- **F-16** バッチ推論完了後の AI 予測キャッシュ invalidate フック

```
Phase 2 完了後の期待効果:
  - 初期APIレスポンスサイズ: 70〜90% 削減
  - SSG ページの FCP: 1.5秒以下
  - モバイルでのスクロール描画: DOM ノード数を常時最小化
```

### Phase 3（優先度: 低 — 将来検討）
> 目標: 運用品質の向上と将来拡張への備え

- **F-7** 管理者向けキャッシュ手動パージ API
- **F-15** GCS ファイルパス命名規則の統一・ETL整合
- **F-17** ページネーション用 DB インデックスの追加
- **Celery によるオンデマンド AI 推論**（オンデマンド予測需要が実証されてから検討）

---

## 依存関係・リスク

### 依存

---

## Conclusion

****

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-02_
