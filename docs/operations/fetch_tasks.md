# このPCのデータ取得の一覧（fetch_status）

このPCで**実行中・待機中**のデータ取得を 1 画面に出す。終わったもの・止まっているものは出さない。

```bash
bash scripts/dev/fetch_status.sh                 # 1 回だけ
watch -n 5 bash scripts/dev/fetch_status.sh      # 見張る
```

- 読むのはローカルの ps・`/proc`・`/tmp/keiba_fetch_tasks/`・`logs/`・`data/local/meta/` だけ。netkeiba・GCS にはアクセスしないので、費用は 0 円。
- 🚨 の行（補填の中断印・アクセス保留）は、取得が無くても出す。これは「終わった」のではなく「止められている」状態だから。

## 仕組み（src/scraper/fetch_registry.py）

### 1. 表示するかどうか: ps の走査で決める
心拍ファイルの無い取得も漏らさない。たとえば次のようなもの:
- 古いコードのまま動いている取得
- NetkeibaClient を通らない取得（JRA・気象庁・smartrc）

次のどれかに当たるプロセスを拾う。
- **取得の入口の python**: 引数が `FETCH_MODULE_PATTERN` に当たるもの。ただし `--report`・`--dry-run` で起動したものは除く。
- **取得を起動する sh**: sh の中身（コメントを除く）に、取得の入口の起動があるもの。中身で判定するので、cron の sh も、作業用の待ち受けの sh も、自動で拾える。
- **心拍ファイルに載ったプロセス**: パターンに無くても、登録したもの・待機中のもの・直近 10 分に GET があるものは拾う。

拾ったプロセスは、起動元の sh と子の python を親子関係（ppid）でまとめ、1 つのタスクとして出す。

### 2. 状態の判定
- **待機中**: 次のどれか。
  - 心拍の state が waiting（取得スロット待ち、`backfill_guard.wait_for_quiet` で他の取得の終了を待っている、など）
  - sh だけが生きていて、取得の子プロセスが無く、子が `sleep` か何も無い（再開の `sleep`、前の取得の終了待ちなど）。
    子が取得でない作業（取得の後の push・集計など）をしていれば、取得は終わっているので出さない。
  - `mark` で待機中として登録したもの
- 登録（`mark`）した sh が取得の子を起動したら、子に合わせて「実行中」になる（登録し直さなくてよい）。
- **実行中**: 待機中に当たらないもの。心拍の最後の GET から 10 分以上たっていれば「止まっている?」を添える。

### 3. 心拍ファイル（補足の情報）
- 置き場所は `/tmp/keiba_fetch_tasks/<pid>-<uid>.json`。環境変数 `KEIBA_FETCH_REGISTRY_DIR` で変えられる。
  - ディレクトリは 1777 にする（cron の root と jovyan の両方が書くため）。
  - ファイル名にユーザーを入れているのは、1777 のディレクトリでは他のユーザーのファイルを上書きできないから。
  - 一時ファイルは O_EXCL で作る。死んだプロセスの自分の分は、次に書くときに消す。
  - PID の使い回しは、`/proc/<pid>/stat` の開始時刻と照合して見分ける。
- NetkeibaClient の GET（JSON API の直接の GET を含む）は、自動で数える。
- 次の待ちの間は、自動で「待機中」と書く。
  - 取得スロットの待ちが 5 秒を超えたとき
  - `backfill_guard.wait_for_quiet` の待ち
  - 補填の譲り（`NightGuard._pace`）
- 常駐のプロセス（`src.scraper.scheduler`・`src.ops.fetch_agent`）は入口のパターンに入れない。取っている間だけ、心拍で出る。書き込みは 5 秒に 1 回まで。
- 失敗しても取得は止めない（例外を出さない）。
- pytest の中では、`KEIBA_FETCH_REGISTRY_DIR` を指定したときだけ書く。

### 4. 補填の譲りの判定とは別物
- `backfill_guard.OTHER_FETCH_PATTERN`（補填が譲る相手）とは別に持っている。
- `fetch_status`・`fetch_registry` のプロセスは、譲りのパターンに当たらない（テストで確かめている）。

## 新しい取得を足すときの決まり

1. **入口のモジュールを `fetch_registry.FETCH_MODULE_PATTERN` に足す。**
   - sh から `-m` で起動するモジュールは、`tests/scraper/test_fetch_registry.py::test_every_module_launched_from_scripts_is_classified` が分類を強制する。
   - 取得でないモジュールは、同じテストの `NON_FETCH_MODULES` に足す。
2. **長い取得は進捗を書く**と、進捗と終了見込みが出る。
   ```python
   from src.scraper import fetch_registry
   fetch_registry.progress(done, total, unit="頭", label="◯◯の取得")      # 何件かごとに呼んでよい（書き込みは間引く）
   fetch_registry.set_state("waiting", "他の取得の終了待ち")              # 自前の待ちのループに入るとき
   fetch_registry.set_state("running")                                    # 待ちが終わったとき（GET が出ても自動で戻る）
   ```
3. **sh の待ち受け**（前の取得の終了を待ってから起動する、など）は、自分を登録すると「待っている相手」が出る。
   ```bash
   python -m src.scraper.fetch_registry mark --pid $$ --state waiting --label "◯◯の取り直し" --waiting-for "PID 1234 の終了"
   ```
   - sh の中身に取得の入口があれば、登録しなくても「待機中」として出る（待っている相手は出ない）。
   - 登録は、そのプロセスが終われば表示から自動で消える。
4. **待ち受けや表示のための補助の sh・モジュールの名前に、`backfill_` などの `OTHER_FETCH_PATTERN` に当たる語を入れない。**
   当たると、補填がそれを他の取得とみなして待ち続ける。

## 既知の取得の詳しい行

`src/scripts/scraping/fetch_status.py::_detail`。実行中のときだけ出す。

| 取得 | 出すもの |
|---|---|
| `src.scraper.backfill --phase past` | 段階ごとの残りの件数・終了見込み（`docs/operations/backfill_2016.md`） |
| `backfill_horse_training_recent` | 今回の実行の残りの頭数と終了見込み |
| `backfill_horse_training_past` | 進捗と終了見込み。`--redo-truncated` のときは、打ち切った馬の取り直しの進捗。分母はログの「打ち切った馬の取り直し: 対象…」の行から取る |
