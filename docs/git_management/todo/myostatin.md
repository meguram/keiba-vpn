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

## 現状の実装（2026-10-01時点）

- `/myostatin`: ページ表示
- `/api/myostatin`: ミオスタチン遺伝子型ナレッジベース（`MYOSTATIN_GENES_JSON`）の内容を返す
- `POST /api/myostatin/predict`: 父・母父（+距離）を指定し、`MyostatinLookup`で種牡馬情報・産駒予測・特徴量を返す。
  2026-10-01に`MyostatinLookup.predict_offspring_with_basis()`を追加し、レスポンスに
  `confidence`（総合信頼度: confirmed/highly_likely/estimated/inferred/population_default）と
  `basis`（父・母父それぞれの根拠テキスト＋母由来は常に集団平均併用である旨の注記、日本語リスト）を追加。
  既存の`sire_info`/`dam_sire_info`/`offspring`/`features`はキー追加のみで変更なし（ML特徴量生成側の
  `offspring_features`/`sire_allele_features`/`dam_sire_allele_features`は非変更、パイプライン影響なし）。
- `POST /api/myostatin/recalculate`: 全ての未確定馬のミオスタチン遺伝子型を血統から再計算

## 目標（推測）

ユーザが「この馬（配合）は短距離向きか長距離向きか」をミオスタチン遺伝子型という科学的根拠
付きの追加軸で判断できることが目標と推測される。血統・戦績だけでは見えない適性を補完する
位置づけと推測される。

## このラインまで実装できたらブランチを消してよい

- 主要な父・母父についてミオスタチン遺伝子型情報がナレッジベースに揃っている
  （「情報無し」表示になる主要種牡馬が実質無い）
- 未確定馬の遺伝子型再計算が定期的に走り、新規馬が長期間「不明」のままにならない
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- 下記TODOの「`recalculate`の定期実行有無の確認・スケジュール化」は、VPSなら既存のOS crontab
  方式に乗せればよい。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら同方式を
  概ね継続できるが、Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run Jobs
  での再設計が必要になる（採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 主要種牡馬のうちミオスタチン遺伝子型情報が「不明」になっている割合を計測する
      — 2026-10-01対応: 本作業環境には`MYOSTATIN_GENES_JSON`（`data/calculated_data/knowledge/myostatin_genes.json`）
      が存在しなかった（`.gitignore`対象でgit追跡外のローカルデータのため、この環境では未同期だった模様）。
      git履歴に残る直前の正式版（コミット`3af97d2`時点、2026-03-25更新、229頭収録）を復元して計測。
      結果: KB登録229頭中、両アレルとも完全不明(`"??"`)は52頭(22.7%)、片アレルのみ判明(`"?T"`/`"C?"`)は
      83頭(36.2%)、両アレル判明(CC/CT/TT)は94頭(41.0%)。さらに「主要種牡馬」をKB内で他の登録馬の父として
      1回以上参照されている馬（＝日本の主要血統を広げている実績馬、62頭）に絞ると、KB自体に未登録が24頭(38.7%)、
      登録済みだが完全不明(`"??"`)は0頭、片アレルのみ判明が9頭(14.5%)、両アレル判明が29頭(46.8%)で、
      「不明相当」合計は24/62=38.7%。すなわち主要血統の直系の種牡馬自体は比較的よく判明しているが、
      その父（海外の輸入血統の祖先: Nureyev, Orpen, Pivotal, Danzig等）はKB未登録のまま多く残っている。
      なおこの計測のためにKBファイルを本環境のローカルパスに復元した（git管理外・既存の正式データのコピーのみで
      新規データの創作や本番環境への書き込みは行っていない）。
- [x] 予測（`/api/myostatin/predict`）の根拠・信頼度をユーザに分かりやすく提示できているか確認する
      — 2026-10-01対応: ローカルサーバーを一時起動し実際に`curl`で呼び出して確認。改善前は`sire_info`/
      `dam_sire_info`/`offspring`/`features`のみを返し、(1) 父・母父がKB未登録の場合でも`sire_info: null`
      以外に警告が無く、`offspring`/`features`は集団平均値にフォールバックした数値をKB確定値と同じ見た目で
      返していた、(2) 個々の信頼度(`confidence`: confirmed/highly_likely/estimated/inferred)は
      `sire_info`/`dam_sire_info`内に埋まっており予測全体としての総合信頼度が無かった、(3) 母自身の遺伝子型が
      常に未検査で集団平均併用である旨がレスポンス上に一切出ていなかった、という3点で「根拠・信頼度の分かりやすい
      提示」は不足と判断。`src/research/genes/myostatin.py`に`MyostatinLookup.predict_offspring_with_basis()`
      を追加（父・母父の既知/未知・信頼度から総合信頼度を算出し、根拠を日本語の文章リストで生成）し、
      `src/api/app.py`の`POST /api/myostatin/predict`レスポンスに`confidence`・`basis`を追加キーとして
      付与する実装を行った。既存キー（`sire_info`/`dam_sire_info`/`offspring`/`features`）やML特徴量生成側
      （`offspring_features`等、`src/pipeline/features/feature_builder.py`が参照）の戻り値は変更していない。
      実サーバーで父=ディープインパクト/母父=ロードカナロア、父がKB未登録のケース、母父省略のケースの3パターンを
      `curl`で確認し、意図通り`confidence`と`basis`が返ることを確認済み。

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `POST /api/myostatin/recalculate`（未確定馬の再計算）が定期実行されているか確認し、
      無ければOS crontab方式でスケジュール化する

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記のスケジュール化は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実装が前提になる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
