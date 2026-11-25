# 本番を公開モード（KEIBA_PUBLIC_MODE=public）にする案（2026-11-25、scope-fix。plan だけ。未実施）

## 結論

- 今回の漏れの直しは、private の経路（`tier.py` の許可リスト）で行った。本番の設定は変えていない。
- 根本の案は、本番を `mode=public` にして、出し分けを `config/public_fields.json` の分類（A・B・C・O）1 本にまとめること。
- 会員（有料）の見え方も変わるので、統括役とユーザーの判断を待つ。

## public にすると変わること

| 対象 | 今（private） | public にした後 |
|---|---|---|
| 有料会員の予測 | 全項目（実オッズ・生の期待値を含む） | A 分類と想定オッズだけ。実オッズ・生の期待値は `actual_odds_public` が true になるまで出ない |
| 有料会員のレース詳細 | 全項目（指数・パドック・厩舎コメント・馬柱など） | B の事実だけ。C（取得元に固有）は出ない |
| `/api/horse/`・`/api/v1/horse/`・出走馬カード | 誰でも | 開発者だけ（`paths_public_mode.owner_only`。403） |
| 予測成績 | 有料は仮想購入まで | `odds_derived_public` の許可で決まる |

- 出走馬カードが開発者だけになる。匿名・無料に事実だけのカードを出すと決めたこと（11/25）と食い違う。public にする前に、`owner_only` からカードを外し、`redact_entry_card` を通す形に直す。
- 公開モードの匿名の印（`_public_mark`）と予測の数値の扱いは、今回 private と同じにした（`entry_A_anonymous` は空）。

## 進め方の案

1. stg だけで `KEIBA_PUBLIC_MODE=public` にする（`scripts/gcp/deploy_cloud_run_api.sh` の ENV_VARS と `config/edge_render.json` の env を同時に直す。`tests/ops/test_edge_render.py` が食い違いを確かめる）。
2. stg で、4 つの区分（匿名・無料・有料・開発者）の見え方を `check_anon_scope` と画面で確かめる。有料会員の画面で空欄になる所（C の項目）を「公開準備中」の表示にする。
3. ユーザーが stg を見て決めてから、本番へ。nginx の API のキャッシュを purge する（`scripts/ops/vps/purge_api_cache.sh`。最悪 約 2 円）。edge のフォルダも作り直す。

## 費用

- 設定の切り替えだけで、要求の数・GCS の読み出しは増えない（出し分けは応答の後で行う）。purge は 1 回 約 2 円まで。
