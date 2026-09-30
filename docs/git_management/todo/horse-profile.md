# TODO: feature/horse-profile

**対象領域**: 馬プロフィール・馬名検索・関係者統計
**関連ドキュメント**: [../feature-horse-profile.md](../feature-horse-profile.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/api/horse/{horse_id}/detail`: 馬詳細情報（netkeiba + SmartRC + 統計）を集約して返す
- `/api/horse/{horse_id}/recent_races`: horse_resultのrace_historyから直近N走（新しい順）
- `/api/horse/{horse_id}/race_performance_history`: 戦績のうちrace_performance生成済みのものだけを返す
- `/api/person/{ptype}/{person_id}/stats`: 騎手・調教師の成績情報
- `/api/horse-names/index-meta`: 馬名インデックスの参照用メタ（パス・頭数・生成時刻）
- `/api/horse-names/search`: 馬名検索・候補返却
- ページURLは無し（`/api/*`のみで構成される機能領域）

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
