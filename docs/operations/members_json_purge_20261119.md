# config/members.json を git から外す（2026-11-19）

公開リポジトリ（meguram/keiba-vpn）で、`config/members.json` が 2026-10-21 の `5985e6ba` から追跡されていた。
ユーザーの指示「履歴から外してください。git では管理しない方がいいです。」への対応。秘密の値はこの文書に書かない。

## 結論

- 漏れたのは、動作確認用の `test`（ID・パスワードとも `test`）のハッシュ 1 件だけ。このパスワードは、もともと docs・スクリプトに公開で書いてある。
- 実会員の名簿（`data/local/auth/members.json`・GCS の正本）は、どのブランチの履歴にも一度も入っていない（`git log --all` で確認）。
- したがって、**実会員のパスワードを変える必要はない**。
- 段 1（追跡を外す）は通常の commit で行った。段 2（履歴から消す）は、効果が小さく手間が大きい（下の「段 2 の評価」）。実行するかはユーザーが決める。

## 中身の確かめ方（値は読まずに、形だけを見た）

| 項目 | 結果 |
|---|---|
| 件数 | 1 件（ID は平文の `test`） |
| 方式 | ソルト付き PBKDF2-SHA256・20 万回。ペッパー無し |
| 有効になる条件 | `requires_env: KEIBA_TEST_USER`（環境変数が 1 のときだけ） |
| パスワード | `test` と一致（ハッシュを計算して照合。値は出力していない） |
| 履歴の版 | 全ブランチで同じ 1 版だけ（`5985e6ba` で足したまま、変更なし） |

- 実会員の方式は、PBKDF2-SHA256 60 万回のあとにペッパー（`KEIBA_MEMBERS_PEPPER` または `DEV_SECRET_KEY` から導く）で HMAC をかけ、ID も HMAC で持つ（`src/api/auth.py`）。名簿だけが漏れても、ペッパーが無ければ照合もできない（`DEV_SECRET_KEY` か `KEIBA_MEMBERS_PEPPER` が設定されている前提。どちらも無い dev では、コードの既定値から導かれる）。
- 履歴の `.env` 系は `*.example` だけで、`DEV_SECRET_KEY`・`DEV_PASSWORD` には例の値しか入っていない（値は表示せずに形だけ確認）。`KEIBA_MEMBERS_PEPPER=` を足した commit は無い。

## 段 1: 追跡を外す（develop に通常の push）

### 読む所（洗い出し）

| 場所 | 読み方 | ファイルが無いとき |
|---|---|---|
| `src/api/auth.py` `_load_members` | `config/members.json` と `MEMBERS_LOCAL_FILE`（`data/local/auth/members.json`）を合わせて読む | `exists` で見てから読む。読めなければ空。実会員の照合は変わらない |
| `src/scripts/ops/manage_users.py` | `config/members.json` は読むだけ（test を変えさせない） | 影響なし |
| `deploy/docker/Dockerfile.cloudrun`・`.serving`・`.gcp-jobs` | `COPY config/` | CI の checkout に無いので、イメージに入らない＝ test が使えない（下） |
| `deploy/docker/*.dockerignore`・`Dockerfile.cloudrun.gcloudignore` | jobs はもともと除外。ほかも今回除外に足した（ホストから手で配備しても入らない） | 影響なし |
| `src/ops/release_publish.py`・`release_manifest.py` | 名簿は入れない・比べない | 影響なし |
| `src/ops/gcs_mirror.py` | 束・目録は `data/local/auth` だけ | 影響なし |
| `scripts/ops/vps/probe_stg_api.sh` | rstg に test/test でログインして確かめる（手動の道具） | 401 のあと匿名で確かめる（止まらない） |
| `tests/api/test_tiered_access.py` | 実ファイルで test/test を照合していた | テストの中で名簿を作るように直した |

### 環境ごとの影響

- **本番の Cloud Run**: 会員は `KEIBA_MEMBERS_STORE=gcs` で GCS の正本を読む（GitHub の Variables に `MEMBERS_STORE=gcs` が 2026-11-17 から設定済み。`deploy-stg.yml`・`promote.yml` が毎回付ける）。test は `KEIBA_TEST_USER` が無いので、もともと無効。**影響なし**。
- **stg（rstg）**: `KEIBA_TEST_USER=1` が付くが、イメージに `config/members.json` が無くなるので、**test/test でログインできなくなる**。自動の配備・CI・監視は test/test を使っていない。手で確かめるときは、管理画面か `manage_users` で試験用の会員を作る。
- **ホストの API（Flask :23110 など）**: 会員は `data/local/auth/members.json`（cron が GCS から取り込む）。`.env` は `KEIBA_TEST_USER=0`。**影響なし**。
- **VPS**: nginx の中継だけで、名簿を持たない。

Cloud Run の実際の環境変数は、gcloud の認証があるマシンで次を流して確かめる。
本番の通信（percent > 0）のリビジョンに `KEIBA_MEMBERS_STORE=gcs` があり、`KEIBA_TEST_USER` が無いこと。

```bash
gcloud run services describe keiba-api --region asia-northeast1 \
  --format='value(status.traffic[].revisionName,status.traffic[].percent,status.traffic[].tag)'
gcloud run revisions describe <上の revisionName> --region asia-northeast1 \
  --format='value(spec.containers[0].env)' | tr ';' '\n' | grep -E 'MEMBERS_STORE|TEST_USER'
```

### 共有の作業ツリーの早送り

使い捨ての repo で試した結果:

- 手元の実物が**変わっていない**とき、追跡を外す commit を早送りで取り込むと、実物は**消える**。
- 手元の実物を**変えている**とき、早送りは止まる（`Your local changes ... would be overwritten`）。

共有の作業ツリーの実物は git の版と同じなので、早送りで消える。ホストは `KEIBA_TEST_USER=0` なので、消えてもログインは変わらない。
手元で test を使い続けたいときだけ、早送りの前に写しを取り、後で戻す。

```bash
cd /home/jovyan/work/keiba-vpn
install -m 600 config/members.json /home/jovyan/work/progress/members.json.bak   # 写し（git の外）
bash /home/jovyan/work/progress/pull_shared.sh                                 # 早送り
[ -e config/members.json ] || install -m 644 /home/jovyan/work/progress/members.json.bak config/members.json
git status --short config/members.json                                         # 何も出ない（ignore されている）
```

ほかの worktree・clone も、develop を取り込むと同じく消える。どれもホストの本番の処理には使わないので、影響はない。

## 段 2: 履歴から消す（準備まで。実行はユーザー）

### 段 2 の評価

- 消えるのは、公開済みの test/test のハッシュだけ。安全上の効き目は小さい。
- 手間と危険は大きい。
  - GitHub のブランチ 23 本（手元の worktree は 33）の SHA がすべて変わる。
  - 書き換えた後、各作業ツリーを付け直す必要がある。
- GitHub の上に、すぐには消えない参照が残る。
  - 書き換え前の全ブランチの commit は、GitHub のガベージコレクションまで、SHA の URL で見え続ける。
  - マージ済みの PR #1・#2 の `refs/pull/*/head` は書き換えられない。PR #3 の「force-pushed」の履歴・Actions の実行画面にも、古い SHA が残る。
  - 完全に消すには、GitHub サポートへの依頼（Sensitive data removal）か、リポジトリの作り直しが要る。
- フォークは 0（2026-11-19 時点）。誰かの clone・キャッシュにある分は取り返せない。
- 推奨: 段 1 で十分とし、段 2 は「GitHub サポートに頼むか」と合わせてユーザーが判断する。行うなら下の手順で。

### 実行の窓

- 開催日（11/21〜23）と、金曜の本番の昇格（11/20 の昼）の前後は避ける。**推奨は 11/24（月）以降の平日の昼**。
- 並行して動いているセッションの push を止めてもらってから行う（書き換えの最中に push されると、その commit が古い履歴の上に乗る）。

### 手順

1. 準備
   - git-filter-repo を入れる。このマシンの git は 2.34 なので、版を固定する: `pip install git-filter-repo==2.38.0`（2026-11-19 に、この版で 2. まで動くことを確かめた）。
   - 統括役が各セッションに「push を止める」と伝える。
   - PR #3（`feature/latency`、下書き）の担当に、手元の未 push の commit を push してもらう。
2. 書き換えと確認（GitHub は変えない）
   - `bash scripts/ops/purge_members_json_history.sh ~/members-purge-20261124`
   - 書き換え前の丸ごとの写し（`backup.git` と tar.gz）を作り、その写しから書き換える。
   - `refs/pull/*` を落としてから、`git filter-repo --invert-paths --path config/members.json` を流す。
   - 履歴にパス・中身（blob）が 1 件でも残れば止まる。
   - 書き換え前後の先頭（`before_heads.txt`・`new_heads.txt`）と、filter-repo の対応表（`filter-repo-maps/`。GitHub サポートへの依頼に使う）を作業場所に残す。
3. force push
   - 同じコマンドに `--push` を付け、`PURGE` と打つ。写しは、その時点の GitHub から取り直す。
   - 先に workflow（`ci.yml`・`deploy-stg.yml`・`promote.yml`）を止め、3 本とも止まったことを確かめる。止めないと、master への push で本番の配備（`promote.yml`）が予定外に走る。
   - 実行中・待ちの run がある、または写しの後に GitHub のブランチが動いていれば、push せずに止まる。
   - 全ブランチ・全タグ（今は 0 本）を、写しの時点の SHA を条件に（`--force-with-lease`）force push し、GitHub の先頭が書き換え後と一致するかを確かめる。
4. GitHub の後始末
   - PR #3 は、同じリポジトリのブランチなので、新しい SHA に付け替わる。マージ済みの PR #1・#2 は、古い commit を指したまま残る。
   - 古い commit を消し切るなら、GitHub サポートに依頼する。依頼には、対象のリポジトリ・ファイルのパス・`5985e6ba`・PR #1〜#3・`filter-repo-maps/` の対応表を添える。
   - Actions のキャッシュ（1 件）を消す: `gh cache delete --all -R meguram/keiba-vpn`。
5. 各作業ツリーの付け直し（必ず、workflow を戻す前に）
   - **ホストの共有の作業ツリー**（`/home/jovyan/work/keiba-vpn`。root 所有。root の cron が動く）
     - `pull_shared.sh` と `git_pull_hourly.sh` は、早送りできないと何もしない。止まりはするが、壊れはしない。
     - 付け直しは手で、`pull_shared.sh` と同じく sudo で行い、最後に持ち主を戻す。
       1. `T=$(date -u +%Y-%m-%dT%H:%M:%S)`、`sudo git -c gc.auto=0 fetch origin`。
       2. 記録されていない変更を控える: `git diff > /home/jovyan/work/progress/shared_tree.patch`。
       3. `sudo git -c gc.auto=0 reset --keep origin/develop`。
       4. `find .git -user root -newermt "$T" -print0 | sudo xargs -0 -r chown jovyan:users`。
     - 段 1 を取り込んだ後なら、古い先頭と新しい先頭のツリーは同じなので、`--keep` はぶつからない。ぶつかって止まったときは、控えを当て直す。
     - 段 1 をまだ取り込んでいないときは、追跡中の `config/members.json` が消える。先に「共有の作業ツリーの早送り」の手順で写しを取る。`data/`・`.env` など ignore されたものには触れない。
     - `promote_oneshot.sh` が作る `refs/promote/ui-*` と、手元のブランチは古い履歴を指したまま残る（GitHub には送られない）。昇格の記録として要らなくなったら `git update-ref -d` で消す。残すと、手元の `git log --all` には今後も members.json が出る。
   - **worktree（33 本）・各自の clone**
     - push 済みの作業: 消して作り直す。
     - 未 push の commit: 新しい develop に `git rebase --onto origin/develop <古い分岐点>` で載せ直す。
     - 古い履歴のまま push すると、`config/members.json` の入った commit が戻る。push の前に、`git log origin/develop..HEAD -- config/members.json` が空であることを確かめる。
   - VPS・学習PC に clone があれば、同じく作り直す。
6. CI・配備
   - workflow を戻す: `for w in ci.yml deploy-stg.yml promote.yml; do gh workflow enable $w -R meguram/keiba-vpn; done`。
   - 止めていた間の push は、戻しても動き直さない。次の通常の push（develop → stg）で、CI と stg の配備が緑になることを確かめる。
   - Artifact Registry のイメージ・`live-` タグは commit の SHA に依らないので、影響はない。
   - `release_manifest` はファイルの中身で比べるので、SHA が変わっても影響はない。
7. 戻し方
   - push の後に問題があれば、`backup.git` から戻す: `git -C backup.git push --force https://github.com/meguram/keiba-vpn.git 'refs/heads/*:refs/heads/*' 'refs/tags/*:refs/tags/*'`（`--mirror` は使わない。`refs/pull` が拒否され、写しの後にできたブランチも消える）。
8. 写しの始末
   - `backup.git` と tar.gz は、消したいファイルを含む。作業場所は `chmod 700`（スクリプトが付ける）。リポジトリの中・共有の場所には置かない。
   - 付け直しと CI が緑になって 1 週間、問題が無ければ消す。

## 会員のパスワード（手当て）

- 実会員のハッシュは公開されていないので、パスワードの変更は求めない。会員への案内も要らない。
- test/test は、公開のリポジトリにパスワードごと書いてある。
  - **公開の環境で `KEIBA_TEST_USER` を付けない**（`tiered_access.md`）。
  - 段 1 で CI のイメージから test が消えたので、rstg でもログインできない。
- 今後、名簿が漏れた疑いが出たら、次の 2 つで対応する。
  1. ペッパー（`KEIBA_MEMBERS_PEPPER`）を変える。全員の照合が通らなくなる。
  2. 各会員のパスワードを、`manage_users`（`user_management.md`）で再設定する。
     - 新しいパスワードは、会員ごとに個別の連絡で伝える。
     - 秘密の値は docs・commit・チャットに書かない。
