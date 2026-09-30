# TODO: feature/myostatin

**対象領域**: ミオスタチン遺伝子解析
**関連ドキュメント**: [../feature-myostatin.md](../feature-myostatin.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/myostatin`: ページ表示
- `/api/myostatin`: ミオスタチン遺伝子型ナレッジベース（`MYOSTATIN_GENES_JSON`）の内容を返す
- `POST /api/myostatin/predict`: 父・母父（+距離）を指定し、`MyostatinLookup`で種牡馬情報・産駒予測・特徴量を返す
- `POST /api/myostatin/recalculate`: 全ての未確定馬のミオスタチン遺伝子型を血統から再計算

## 既知の課題

（無し）

## TODO（手動追記用）

- [ ]

## メモ
