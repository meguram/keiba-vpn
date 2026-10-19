# stg の画面配信: 本番ビルド運用（next build + next start）

> 2026-10-19 に、stg（ホスト PC）の Next.js を `next dev` から本番ビルドにする仕組みを入れた（初回の試行でビルドが 6 分以上終わらず stg が約 9 分止まったため、同日に無停止の切替・並列制限・負荷ガードを足した）。dev（この PC）は従来どおり `next dev`。

## なぜ
`next dev` は開発用の巨大なバンドル（`main-app.js` 約 1.75MB、1 ページ 3〜5MB）を配信し、初回アクセス時にコンパイルもするため、トンネル越しの初回表示が 4〜12 秒かかっていた。本番ビルドは同じページの転送が約 0.3MB（dev PC 実測 4.68MB → 0.27MB）、HTML 応答は約 80ms → 15ms。

## 仕組み（ビルドは動いているサーバを止めない）
- 切り替えは `scripts/server/service_start.profiles.sh`（stg 節）と `scripts/server/server_watchdog.sh`。stg は既定で `FRONTEND_NPM_SCRIPT=start`、`NODE_ENV=production`。
- **ビルドは別ディレクトリ `frontend/.next-stg-build` に作る**（`NEXT_DIST_DIR`。`frontend/next.config.js` が上書きに対応）。動いている `next start`（`.next-stg`）には触れない。失敗しても稼働中のものは何も変わらない。
- **切り替え**（`--switch-frontend`）: next を止める → `.next-stg` を `.next-stg.old` に退避 → `.next-stg-build` を `.next-stg` に入れ替え → `next start` → 起動確認 → `.old` を削除。起動に失敗したら `.old` に戻して再起動する（失敗した出力は `.next-stg.failed`）。ダウンタイムは数秒（dev PC 実測 3〜4 秒）。
- **ビルドの負荷を抑える**: `nice -n 19`（あれば `ionice -c3`）、`NODE_OPTIONS=--max-old-space-size=4096`、ワーカー数は `experimental.cpus=2`（`NEXT_BUILD_CPUS` で変更。stg 以外は Next の既定）。初回の試行はコア数ぶん（約 25 プロセス）が走り、ホスト（24 コア・他セッションで load 約 15）が load 47 まで上がって進まなかった。dev PC では 30 秒（既定）→ 31〜41 秒（cpus=2・nice 19）。
- **高負荷ガード**: 自動のビルドは、load1 が `コア数×0.8`（24 コアで約 19）を超えている間は実行しない（ログに理由）。旧ビルドがあればそれで起動・継続、無ければ待機（次の周期で再試行）。閾値は `KEIBA_BUILD_MAX_LOAD=<数値>` で変更、`KEIBA_FORCE_BUILD=1` で無視（明示実行用）。
- **ビルドの要否**（`scripts/server/frontend_build.sh`）: `BUILD_ID` が無い／入力（`app/ components/ lib/ styles/ public/ next.config.js package.json` と `config/ports.env`）が BUILD_ID より新しい／ビルド時に固定される環境（`KEIBA_ENV` / `KEIBA_API_URL` / `KEIBA_FASTAPI_URL` / `NODE_ENV` / `NEXT_PUBLIC_*`。前回は `.build-env`）が違う。
- **watchdog（cron）が自動でビルドするのは「next が落ちていて、使える本番ビルドが無い」ときだけ**（その場でビルド→配置→起動）。使える旧ビルドがあるときは、ソースが新しくても旧ビルドで即起動する（復旧を優先）。ソース更新の反映は、下の手順（ビルド→切替）を明示的に実行する。
- **ロックと目印**: `logs/.frontend-build-stg.lock`（flock。ビルド/切替/起動中は他の watchdog 実行がスキップ）。`logs/.frontend-build-failed-stg`（ビルド失敗の目印。10 分は自動再試行しない）。`logs/.frontend-want-stg`（起動したいのにできていない目印。PID ファイルが無くても watchdog が監視・再試行を続ける）。**PID ファイルには目印を書かない**（cron の watchdog は `.frontend.pid`、`--profile stg` 付きは `.frontend-stg.pid` と別ファイルになり、復旧判定が壊れるため）。
- **PATH**: watchdog は `/opt/conda/bin` と `/usr/local/bin` を PATH の末尾に補う（cron / `sudo -n` で `npm: command not found` になる問題の対策）。
- **停止（ss / netstat / lsof に依存しない。`/proc` だけで動く）**: ホストの初回の切替では、親（sh -c / npm）だけが止まり、子の `next-server`（プロセス名 `next-server`、親が 1 の孤児、`-p` を含まない）が旧ビルドのままポート 23200 を掴み続け、新しいサーバは起動に失敗して、旧ビルドの画面が出続けた（新ページが 404）。そのため `stop_frontend`（`frontend_stop_all`）は次のとおり止める。
  1. PID ファイルのプロセス（コマンドラインに npm / next / node を含むもののみ）と、その**子孫すべて**（`/proc/*/stat` の ppid をたどる）。
  2. ポートを LISTEN しているプロセス（`/proc/net/tcp{,6}` の LISTEN の inode → `/proc/*/fd` の `socket:[inode]` で持ち主を特定）で、**cwd が `frontend/` のもの**と、その子孫。dev の別プロファイル（別ポート）や、別ディレクトリ起動のプロセスは止めない。
  3. TERM → 最大 10 秒、ポートが空くまで待つ（`KEIBA_STOP_WAIT_SEC`）→ 残っていれば KILL → 最大 5 秒待つ。**それでも空かなければ失敗を返し、切替は入れ替えずに中止**（旧ビルドと `.next-stg-build` を残す）。
- **起動後の確認**: 起動した `next start` が期待したビルドで動いているかを確認する。(a) `/_next/static/<BUILD_ID>/_buildManifest.js` が 200（別ビルドのサーバなら 404）、(b) ポートの持ち主が、いま起動した PID の子孫であること（旧プロセスが残っていればここで検出）。不一致なら失敗扱いで、切替は元のディレクトリに戻す。成功時のログに『新プロセスの PID・リスナー PID・BUILD_ID』が出る（`切替完了: 新プロセス PID=… リスナーPID=… BUILD_ID=…`）。
- ビルド時に固定される値（`rewrites` の転送先・`NEXT_PUBLIC_*`）はビルド時に渡る。ホストの stg では `NEXT_PUBLIC_*` は未設定（コードの既定値で動作）。設定を足したら自動で『環境が違う』判定になる。
- `next build` は tracked の `frontend/next-env.d.ts` を書き換える（`.next-dev` → `.next-stg…` の参照）。ホストでは `M` のままでよい（反映時にこのファイルは触らない）。

## 反映の手順（ホストで実行する順番）
1. **コード反映**: `sudo -n git fetch -q origin` → `sudo -n git checkout <hash> -- <変更ファイル>`（pull の post-merge フックは `.next-stg` を消し、動いている next が 500 になるので使わない）。API 側を変えたときは `restart_server.sh`（キューが空のこと）。反映前に dev PC で `cd frontend && npx tsc --noEmit` を通す（型エラー・lint エラーがあるとビルドが失敗する）。
2. **ビルド（旧サーバ稼働のまま）**: 負荷を確認（`uptime`。load1 が約 19 未満が目安）して、
   `sudo -n bash scripts/server/frontend_build.sh stg`
   （内部で nice 19。約 1〜数分。ssh が切れないよう `sudo -n nohup bash scripts/server/frontend_build.sh stg > /tmp/fe_build.out 2>&1 &` で投げて `tail logs/watchdog.log` で見てもよい）。高負荷で見送られたら下がるのを待つ（急ぐときだけ `KEIBA_FORCE_BUILD=1`）。`ビルド成功` と出れば `frontend/.next-stg-build` ができている。
3. **切替**（成功後。ダウンタイム数秒）: `sudo -n bash scripts/server/server_watchdog.sh --profile stg --switch-frontend`（`切替完了` と出る）。
4. **確認**: `tail logs/watchdog.log` の `切替完了: 新プロセス PID=… リスナーPID=… BUILD_ID=…`、`cat frontend/.next-stg/BUILD_ID` と一致、`curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:23200/_next/static/$(cat frontend/.next-stg/BUILD_ID)/_buildManifest.js` が 200、`ps -eo pid,etime,comm | awk '$3=="next-server"'`（1 本、起動したばかり）、公開 URL（`/weekly-predictions` `/races` `/race/<id>`、追加したページが 404 でないこと）。
5. **ロールバック**:
   - 切替直後に不具合: 前の出力は切替成功で削除されるため、前のコミットに戻して 1〜3 をやり直す（約 2 分）。
   - 本番ビルド運用そのものをやめて `next dev` に戻す: ホストの `.env.stg` に `KEIBA_STG_NEXT_MODE=dev` を 1 行追記 → Next の PID を番号で確認して `sudo -n kill <PID…>` → `sudo -n bash scripts/server/server_watchdog.sh --profile stg`。`.next-stg` の本番ビルドは dev 起動時に自動で消える。本番ビルドに戻すときはその行を消す。

**dev → 本番ビルドの初回**: `.next-stg` には `next dev` 由来のキャッシュが入っている（`BUILD_ID` が無い）。手順 2（別ディレクトリに作る間、dev のまま配信が続く）→ 3 で、dev の出力は `.next-stg.old` に退避され、成功確認後に削除される。

## リスクと注意
- 反映のたびに「ビルド → 切替」が要る（`next dev` のホットリロードは無い）。ソースが変わっても、切替するまで画面は旧ビルドのまま。
- ビルド中はホストの CPU を使う（nice 19・並列 2 で抑えているが、load が高いときは数分かかる）。ビルドが終わらない・負荷が上がる場合は `ps` でビルドの PID を確認して止める（`.next-stg-build` は捨ててよい）。
- ビルドが失敗したら稼働中のものは変わらない。10 分は自動再試行されない。ログ: `logs/frontend_build_<日時>.log`。
- 手動で `npm run build` を並行して走らせない（flock は watchdog のものだけを見る）。
- 動作確認の実績は dev PC のみ（`KEIBA_ENV=stg` で build → switch、稼働中ビルド→切替の無停止検証＝ビルド中の失敗 0・切替の約 3.6 秒のみ、高負荷ガード、ビルド失敗時に稼働中が不変、旧ビルド無しでの自動ビルド）。ホストでの再試行は、ログを見ながら実施する。

## ビルドは /tmp の作業コピーで行う（ホストの 9p 問題）

ホストの repo（`/home/jovyan/work`）は Windows ドライブ（9p）上にあり、小ファイルの作成が `/tmp`（overlay）の約 300 倍遅い（2,000 ファイルで 11 秒 対 0.04 秒）。ここで直接 `next build` すると、`.next-stg-build` への大量の書き込みで「Collecting page data」から 7 分以上進まなかった（2026-10-19、2 回とも。負荷の高い時間帯でさらに悪化）。`/tmp` 上では同じビルドが 23 秒で完走する。

そのため `server_watchdog.sh` のビルドは、ソース（約 1.4MB）を `/tmp/keiba-vpn-fe-build<suffix>/` にコピーし、`node_modules` は symlink のまま、そこでビルドする。成功したら、成果物（キャッシュを除く約 6MB・約 240 ファイル、9p への書き込みは約 3 秒）だけを `frontend/.next-stg-build` に置く。作業コピーのビルドキャッシュは次回の高速化に使う。`/tmp` は消えてもよい（次回はキャッシュ無しでビルドされるだけ）。作業場所は `FRONTEND_BUILD_WORK_DIR` で変えられる。
