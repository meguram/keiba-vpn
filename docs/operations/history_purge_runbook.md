# 公開リポジトリの履歴から実データ・重み・第三者記事を消す手順

対象: `meguram/keiba-vpn`（公開）。除去するパスは `config/history_purge_paths.txt`、書き換え後に先端へ戻す整理済みの版は `config/history_purge_restore.txt`。
理由は `docs/operations/netkeiba_purge_report.md`、`security_hygiene.md`、メモリの `project_netkeiba_tos`。
**強制 push を含む。ユーザーの合図があるまで実行しない。**

## 実行の前提（すべて満たしてから）
1. 稼働中の fork が、すべてコミット・push を終えている（`ListAgents` で稼働なし）。書き換えのあとに古い履歴の上へ push されると、除去した内容が戻る。
2. 手元のコミットを push 済み（`git status` で未 push が無い）。
3. `docs/other_predictor/` の 3 ページ（中身は差し替え済みの空ページ）の登録を、`src/scripts/docs/build_docs_book.py` の `PARTS` から外し、`python -m src.scripts.docs.build_docs_book` で `book.html` を再生成して push 済み（除去パスに含まれるため）。`pytest tests/scripts/test_build_docs_book.py` が通ること。
4. 他の clone・他のセッション（例: keiba-vpn-28）に、書き換えの予定を伝えてある。
5. GitHub のブランチ保護（Settings → Branches）で、`develop`・`main` への強制 push が禁止されていれば、一時的に許可する（終わったら戻す）。

## 保険の確認（履歴除去の前に必須）
履歴を書き換えると、除去したファイルの元の値は git から復元できない。**実体が 3 か所に揃っていること**を、ファイルごとの sha256 で確認する。
```bash
export KEIBA_GCS_BUCKET=<バケット名>        # (b) の確認に必要
bash scripts/ops/purge_preflight.sh backup  # (c) リポジトリ外へ bundle と tar.gz を作る
bash scripts/ops/purge_preflight.sh check   # 全部 OK で終了コード 0
```
| 場所 | 内容 | 確認する人 |
|---|---|---|
| (a) ホストの実ファイル | `config/fitted_artifacts.txt` の 91 ファイル（外部資料の抜粋 3 件を含む）が、手元の版と sha256 一致 | check が自動で |
| (b) GCS `models/fitted_artifacts/` | `model_publish publish fitted_artifacts` 済みで manifest が一致 | check が自動で（読み取り専用の鍵） |
| (c) リポジトリ外バックアップ | `git bundle`（全ブランチ）と対象ファイルの tar.gz | check が自動で |

**ホスト版が正**: 学習済みの重みは、ホストが最新。開発 PC の複製と違う場合（現状 `models/final_odds_bundle.json`）は、ホストで pack と publish を行い、バックアップもホストの版から作る。
`NG` が 1 つでもある間は、次の手順に進まない。詳細と現状の結果表は `docs/operations/fitted_artifacts.md`。

## 手順
### 1. バックアップ（リポジトリの外に）
```bash
mkdir -p ~/keiba_private/backup
git clone --mirror https://github.com/meguram/keiba-vpn.git /tmp/purge_backup.git
git -C /tmp/purge_backup.git bundle create ~/keiba_private/backup/keiba-vpn_$(date +%Y%m%d).bundle --all
git bundle verify ~/keiba_private/backup/keiba-vpn_$(date +%Y%m%d).bundle
```
手元だけにある未 push の作業があれば、その clone も別に退避する。

### 2. 書き換え（作業用の mirror の上だけ。push はしない）
```bash
CLONE_FROM=https://github.com/meguram/keiba-vpn.git bash scripts/ops/history_purge_rewrite.sh /tmp/purge_real
```
出力で次を確認する。
- 「残っている除去対象: 0」
- 先端のツリー差分は「消えた 0（`PARTS` を外したあとなら）/ 増えた 0 / 内容が変わった 0」
- 書き換え後のサイズ（リハーサルでは .git 132M → 24M）

`master`・`stg` は、空の初期コミットだけのブランチなので、`--prune-empty` で消える。GitHub 側には手を付けない（下の push で対象にしない）。

### 3. 書き換え後の動作確認
```bash
git clone -q /tmp/purge_real/mirror.git /tmp/purge_check && cd /tmp/purge_check && git checkout -q develop
python3 -m pytest tests -q -p no:cacheprovider --ignore=tests/scraper/manual --ignore=tests/research/manual
bash scripts/dev/secret_scan.sh --tracked
```

### 4. 強制 push（ブランチごと、リース付き）
書き換え前の各ブランチの先頭は `/tmp/purge_real/heads_before.txt`。`--force-with-lease` に、その値を渡す。
```bash
cd /tmp/purge_real/mirror.git
while read -r br old; do
  case "$br" in master|stg) continue;; esac
  git push --force-with-lease="refs/heads/$br:$old" https://github.com/meguram/keiba-vpn.git "refs/heads/$br:refs/heads/$br"
done < /tmp/purge_real/heads_before.txt
```
- `--mirror` は使わない（GitHub の PR 参照などで失敗・混乱する）。
- タグは無い（`git tag` が 0 件）。ブランチは `develop`・`main`・`feature/*` が対象。
- 失敗したブランチは、誰かが先に push していないかを確認してやり直す。

### 5. 手元・他の clone の取り直し
```bash
git fetch origin
git checkout develop
git reset --hard origin/develop       # 未 push の作業があれば先に退避する
git branch -r | head                   # 取り直し
```
または再 clone。**稼働中の fork・他セッションは、書き換えの前に止める。** 書き換えたあとに古い clone から push すると、除去した内容が戻る。
ホストは `git pull` しない運用なので、影響しない。

### 6. GitHub 側に残るもの
- 旧コミットの SHA は、GitHub 上でしばらく URL から閲覧できることがある（キャッシュ・到達不能オブジェクト）。
- PR の参照（`refs/pull/*`）が旧コミットを指す。マージ済み・閉じた PR の参照は残る。
- fork がある場合、fork 側には旧履歴が残る（`gh api repos/meguram/keiba-vpn/forks` で確認）。
- 対応: GitHub サポートへ、キャッシュ済みビューと到達不能オブジェクトの削除（ガベージコレクション）を依頼する。

依頼文の案:
> リポジトリ meguram/keiba-vpn で、公開すべきでないデータが含まれたコミットを、履歴の書き換え（強制 push）で除去しました。旧コミットの SHA を指す、キャッシュされたビュー、PR の参照（refs/pull/*）、到達不能オブジェクトを削除（ガベージコレクション）していただけないでしょうか。対象の旧ブランチ先頭の SHA は `heads_before.txt` のとおりです。

### 7. ロールバック（書き換えを取り消す）
バックアップから元に戻す。
```bash
git clone -q ~/keiba_private/backup/keiba-vpn_YYYYMMDD.bundle /tmp/purge_restore
cd /tmp/purge_restore && for b in develop main; do git push --force https://github.com/meguram/keiba-vpn.git "refs/remotes/origin/$b:refs/heads/$b"; done
```
（`feature/*` も同様。）ただし、一度公開された内容を取り戻すことは、通常は避けたい。ロールバックは、書き換えの手順に誤りがあった場合に限る。

### 8. 終わったら
- 保護を戻す。`security_hygiene.md` と `netkeiba_purge_report.md` に実施日を追記する。
- `bash scripts/dev/secret_scan.sh --tracked`、`git log --all --name-only | grep -F -f ...` で除去対象が無いことを再確認する。
