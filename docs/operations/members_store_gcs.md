# 会員名簿の正本を GCS に移す（KEIBA_MEMBERS_STORE=gcs）

`login_scale_review_20261117.md` の「名簿の書き込みに穴」への対策。

## 今の作り（切り替え前）と問題

- 正本はホストの `data/local/auth/members.json`（root だけが読める）。
- ホストの root の cron（`scripts/cron/gcs_mirror_push_auth.sh`、5 分おき）が GCS の `ops/mirror/data/local/auth/members.json` へ上げ、Cloud Run はそれを取り込む。
- Cloud Run の上の書き込みは、そのインスタンスの手元のファイルにしか残らない。ほかのインスタンス・GCS には伝わらず、次の取り込みで消える。該当する書き込みは次の 2 つ。
  - 管理 API: `/api/v1/admin/users/*`
  - 決済: 申し込み・Webhook（`src/billing/service.py`）

## 切り替えた後の作り

- 正本は GCS の `ops/mirror/data/local/auth/members.json`。
- 書き込みは、どこからでも `manage_users._transact` を通る。
  - GCS の名簿を世代つきで読み、変えてから「読んだ世代のままのときだけ書く」（`if_generation_match`）。
  - 他が先に書いていたら、読み直してやり直す（最大 6 回）。同時に書いても失わない。
  - 書けたら、そのマシンの手元の写しと目録（`ops/mirror/_index/auth.recent.json`）も更新する。ほかのインスタンスは、目録の取り込みかログインの照合の前の取り直しで拾う。
- ホストの root の cron（`push --only-auth`）は、名簿を上げずに GCS から手元へ取り込む。
  - GCS の名簿が壊れていれば取り込まない。上書きの前に `.bak` を残す。
  - GCS にまだ名簿が無いときだけ、手元の名簿（正しく読めて 1 人以上）を置く（移行の初回）。
- 正本を初めて置くのは、ホストの cron だけ。GCS に名簿が無い間は、`_transact` は書かずに止まる（手元に写しが無い Cloud Run が、空の名簿を正本にしないように）。
- 既定（`KEIBA_MEMBERS_STORE` が未設定）は、今までどおり手元のファイルだけ。

## 費用

- 書き込み 1 回 = 読み出し 1 回（Class B）＋ 条件つきの書き込み 1 回（Class A）＋ 目録の更新（読み 1・書き 1）。月 1 万回の書き込みでも、Class A 2 万回 × 0.8 円/千回 ＋ Class B 2 万回 × 0.06 円/千回 ≒ **約 17 円**。
- ホストの cron は、5 分おきに名簿 1 件のメタデータ（md5）だけを取り（Class B。一覧はしない）、手元の名簿の md5 と比べて、変わったときだけ中身を読む。
  - メタデータの取得は、最悪で 288 回/日 × 30 日 = 8,640 回 × 0.06 円/千回 ≒ **約 0.5 円/月**。
  - 中身を読むのは変わったときだけ。最悪で 5 分ごとに毎回変わったとしても、会員 1,000 人（0.3 MB）× 8,640 回 ≒ 2.5 GiB で約 45 円/月、会員 1 万人（2.7 MB）だと約 23 GiB で約 410 円/月。
  - ふつうは名簿が変わるのは 1 日数回なので、月 1 円未満。毎回変わるほど書かれるようになったら、ホストの写しをやめる（下の「残り」）。
- Cloud Run の中の読み書きは、同じ東京リージョンなので転送料は 0。

## 切り替えの手順（ユーザーの操作。順番を守る）

1. **ホスト**: `.env` に `KEIBA_MEMBERS_STORE=gcs` を足す。root の cron も `.env` を読む（`gcs_mirror.main` が `load_project_dotenv` を呼ぶ）。
   - 次の cron（5 分以内）で、ログに `GCS から取り込んだ` か `変更なし` が出ることを確かめる。
   - 確認の例: `sudo /opt/conda/bin/python3 -m src.ops.gcs_mirror push --only-auth`
2. **GitHub の Variables** に `MEMBERS_STORE=gcs` を足す。
   - stg（`deploy-stg.yml`）と本番の昇格（`promote.yml`）の配備が、毎回 Cloud Run にこれを付ける。手で付けても、次の CI の配備で外れることはない。
   - stg と本番は同じバケット（同じ名簿）を使うので、変数は 1 つで共通にする。
   - stg の配備（stg へ push）で rstg に効く。管理画面で試験用の会員を 1 人作ってすぐ消し、ホストの `manage_users list` で消えたことを確かめる。**これは本番の名簿への書き込みになる**。
3. **本番**: 次の master への昇格で効く（ユーザーの判断）。
4. 決済（`KEIBA_BILLING`）を有効にするのは、3 の後。

- 権限: Cloud Run の SA（keiba-api）には `roles/storage.objectUser` が付いている（`scripts/gcp/setup_project.sh`）。追加は要らない。

## 戻し方

1. GitHub の Variables から `MEMBERS_STORE` を消し、stg・本番を配備し直す（または前のリビジョンへ戻す）。
2. ホストの `.env` から `KEIBA_MEMBERS_STORE` を消す。root の cron が、また手元の名簿を GCS へ上げるようになる。
   - ホストの手元の写しは、cron が GCS から取り込んで最新になっている。そのため、戻しても Cloud Run 側で書いた分は失われない（戻す前に 1 回 cron が回っていること）。

## 確かめたこと（2026-11-17）

- テスト `tests/api/test_members_gcs_store.py`:
  - 書き込みが GCS・手元の写し・目録に入る。
  - 書く直前に別の書き手が割り込んでも、両方の変更が残る。
  - 5 つのスレッドが同時に会員を作っても、全員が残る。
  - 移行の初回は、手元の名簿を元にする。
  - 壊れた GCS の名簿は上書きしない。
  - 入力の誤りでは何も書かない。
  - ホストの cron が、取り込みに切り替わる。
  - 既定では GCS に触れない。
- 本物の GCS の試験用のキー（本番の名簿ではない場所。確認後に削除）で、次を確かめた。
  - 作成・変更が世代つきで書ける。
  - 古い世代での書き込みは拒否される（412）。
  - 手元の写しで照合できる。

## 残り（会員が増えたら）

- ホストへの依存をなくす。会員の管理の CLI を Cloud Run Jobs か管理 API だけにして、ホストの手元の名簿は使わない。
