# stg の画面配信: 本番ビルド運用（next build + next start）

> 2026-10-19 に、stg（ホスト PC）の Next.js を `next dev` から本番ビルドに切り替えた。dev（この PC）は従来どおり `next dev`。

## なぜ
`next dev` は開発用の巨大なバンドル（`main-app.js` 約 1.75MB、1 ページ 3〜5MB）を配信し、初回アクセス時にコンパイルもするため、トンネル越しの初回表示が 4〜12 秒かかっていた。本番ビルドは同じページの転送が約 0.3MB（dev PC 実測 4.68MB → 0.27MB）、HTML 応答は約 80ms → 15ms。

## 仕組み
- 切り替えは `scripts/server/service_start.profiles.sh`（stg 節）と `scripts/server/server_watchdog.sh`。stg は既定で `FRONTEND_NPM_SCRIPT=start`、`NODE_ENV=production`。
- watchdog が Next を起動するとき、**必要なら先に `npm run build`** を実行する（判定は `scripts/server/frontend_build.sh`）。再ビルドする条件:
  1. `frontend/.next-stg/BUILD_ID` が無い（未ビルド、または開発モードの残骸 → 残骸は消してからビルド）
  2. `frontend/` の入力（`app/ components/ lib/ styles/ public/ next.config.js package.json` など、`config/ports.env` を含む）が BUILD_ID より新しい
  3. ビルド時に固定される環境が前回と違う（`KEIBA_ENV` / `KEIBA_API_URL` / `KEIBA_FASTAPI_URL` / `NODE_ENV` / `NEXT_PUBLIC_*`。前回の値は `.next-stg/.build-env`）
- ビルドのログは `logs/frontend_build_<日時>.log`、watchdog の記録は `logs/watchdog.log`。ビルドは約 30〜60 秒（dev PC で 30 秒）。
- **ロック**: ビルドから起動確認までは `logs/.frontend-build-stg.lock`（flock）を握る。cron の watchdog（3 分おき）が途中で割り込んでも、「別の watchdog がビルド/起動中」とログに出して何もしない。
- ビルド失敗時: 旧ビルド（BUILD_ID あり）があればそれで起動、無ければ**起動せずエラー**（`next dev` へは自動で戻さない）。型エラー・lint エラーのあるコミットを反映するとビルドが失敗するので、反映前に `cd frontend && npx tsc --noEmit` を通すこと。
- ビルド時に固定される値（`rewrites` の転送先・`NEXT_PUBLIC_*`）はビルド時に渡る。ホストの stg では `NEXT_PUBLIC_*` は未設定（コードの既定値で動作。確認: `.env` / `.env.stg` / `frontend/.env*` に無い）。設定を足したら自動で再ビルドされる。
- `next build` は tracked の `frontend/next-env.d.ts` を書き換える（`.next-dev` → `.next-stg` の参照）。ホストでは `M` のままでよい（反映時にこのファイルは触らない）。

## 反映の手順（ホスト）
1. 変更をホストに入れる（`sudo -n git checkout <hash> -- <ファイル>`、または pull。**pull の post-merge フックは `.next-stg` を消す**ので、動いている next が 500 を返す → watchdog が 5xx を検知して即再ビルド・再起動する。ダウンタイムは約 1 分）。
2. API 側を変えたときは `restart_server.sh`（キューが空のこと）。
3. Next を止める: `next-server` / `node …next` / `sh -c next start` の PID を番号で確認して `sudo -n kill <PID…>`（`pkill -f` は自分の ssh コマンド行に一致するので使わない）。
4. watchdog を実行: `sudo -n bash scripts/server/server_watchdog.sh --profile stg`（ビルドを含むと 1 分前後かかるので、ssh のタイムアウトに注意。`sudo -n nohup … &` で投げてもよい。何もしなくても cron の watchdog が 3 分以内に起動する）。
5. 確認: `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:23200/`、`ps -eo pid,etime,comm | awk '$3=="next-server"'`、`tail logs/watchdog.log`（「本番ビルドを実行」「ビルド成功」「起動完了」）。

ソースを変えない再起動（設定のみ・クラッシュ復旧）は、ビルド済みなので 10 秒前後で戻る。

## ロールバック（next dev に戻す）
1. ホストの `.env.stg` に `KEIBA_STG_NEXT_MODE=dev` を 1 行追記する（cron の watchdog にも効く）。
2. Next を止めて（上の手順 3）、watchdog を実行（上の 4）。`.next-stg` の本番ビルドは dev 起動時に自動で消える。
3. 本番ビルド運用に戻すときは `.env.stg` のその行を消す。

## リスクと注意
- 反映のたびに再ビルドが要る（`next dev` のホットリロードは無い）。
- ホストの負荷が高いとき（load 20 超）はビルドが数分かかる。その間 stg の画面は止まる（API は動く）。
- ビルド中は `.next-stg` を触らない。手動で `npm run build` を並行して走らせない（watchdog のロックは手動実行を知らない）。
- 動作確認の実績は dev PC のみ（`KEIBA_ENV=stg` で build → `next start -p 23299`、/ ・/weekly-predictions・/races・/race/<id>・/api プロキシが動作）。ホストでの初回は、ログを見ながら実施する。
