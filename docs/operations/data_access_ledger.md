# データの見た回数の台帳（D1）と起点年の照合（B7）

2025 年（test）・2026 年（封印）の結果を「誰が・何の目的で・何回見たか」を 1 つの台帳に残す。評価の道具は、結果を読む前に台帳を確かめる。
計画とレビューの記録は `/home/jovyan/work/progress/ledger_plan_20261118.md`（2026-11-18、統括役の承認済み）。

## 要点

- 道具: `src/ops/data_ledger.py`。決まり: `config/data_access/policy.json`。
- 台帳は追記だけの JSONL。書く先は次の 2 か所で、`guard` が両方に書く。
  - 共有の写し（実行時の正）: `/home/jovyan/work/progress/data_access/ledger.jsonl`
  - git の台帳: `config/data_access/ledger.jsonl`（`.gitattributes` の `merge=union`）
- 判定の鍵は「系統（family）＋版（凍結の hash）＋期間（日付の重なり）＋目的」。出力先・環境変数は鍵に入れない。出力先を変えても 2 回目と分かる。
- 既定は警告＋記録（warn）。`enforce_families` に入れた系統だけ拒否する。`KEIBA_LEDGER_MODE=enforce` は enforce に上げるだけで、warn には下げられない。
- 費用 0 円（ホストのファイルと git だけ）。

## 期間と目的

| 期間 | 区間 | 扱い |
|---|---|---|
| 開発 | 〜2021-12-31 | 記録しない |
| 開発の検証 | 2022〜2024 年 | 記録だけ |
| test | 2025 年 | 下の決まり |
| 封印 | 2026-01-01〜11-20 | 下の決まり |
| 前向き | 2026-11-21〜 | `forward`・`eval_only` は何度でも、`select` は拒否 |

目的は 6 つ。

- `select`: 選ぶ。test・封印・前向きでは拒否する。
- `confirm`: 確かめ（採否の判定）。
  - 同じ対象（family＋版）で 2 回目なら拒否する。
  - 同じ系統の別の版がすでに見ている場合や、同じ対象で `eval_only` の後の場合は、`eval_only` に自動で落とす（採否に使えない）。
- `eval_only`: 評価だけ。理由の文が必須で、見た回数に数える。
- `reproduce`: 凍結した道具が過去の記録を再現するかの確かめ。理由の文が必須。
- `forward`: 前向きの記録。
- `fit`: 2025 年を学習に入れる。記録だけ。

`--blind`（着順を読まない）も期間に触れた回数として数える。道具が書いた例外の行（結果の置き場なし）だけは数えない（落ちた後の再実行を許す）。

## 道具への組み込み

```python
from src.ops import data_ledger as L
with L.guard(tool="src/…/x.py:evaluate", family="course", version=frozen_sha[:12],
             period=("2025-01-01", "2026-11-20"), purpose="confirm",
             frozen_paths=[FROZEN, Path(__file__)], meta={"n_hypotheses": 22}) as tok:
    df = load_results(...)
    tok.check_dates(df["date"])            # 申告した期間からはみ出せば記録（enforce では止める）
    res["ledger"] = tok.summary()          # この期間を何回見たかを結果に残す
    tok.finish(result_path=out)
```

- 既存の印（`EVALUATED*.json` など）は残す。台帳は印に重ねて使う。
- 道具の外で見たときは申告する: `python -m src.ops.data_ledger declare --family … --version … --from … --to … --purpose eval_only --reason …`。
- 一覧: `python -m src.ops.data_ledger show --from 2025-01-01 --to 2025-12-31`。

## これまでに見た回数（過去の分、2026-11-18 時点）

`config/data_access/backfill_20261118.json`（29 件。git の commit・印・申告から起こした）。回数の分からない行は「以上」の下限。

| 期間 | 見た回数（下限） | 内訳 |
|---|---|---|
| 2025 年 | 19 回以上 | 確かめ 17・選ぶ 2（全レース版 第 3 版の試行錯誤、推奨馬券 第 2 回の区切りの選定） |
| 2026 年（〜11/20） | 19 回以上 | 確かめ 16・評価だけ 3（11/17 の 237 レースの申告を含む） |

重賞（graded）は 5 回、全レース版（course）は 3 回、推奨馬券（bet_cycles）は test 2025 を 2 回見た。どちらの年も手つかずではない。

## enforce の系統（2026-11-18 から）

`bet_cycles`・`rules_bet_opt`・`prereg_rules`・`upset_types` は拒否（enforce）。統括役の承認（11/18）。

- 本番・当日の経路から拒否の処理が呼ばれないことを確かめた（`tests/ops/test_data_ledger.py` の `test_guard_is_only_in_research_cli_entrypoints`・`test_importing_bet_cycles_does_not_load_ledger`）。
  - guard を呼ぶのは研究の CLI の入口だけ: `bet_cycles.py` の `cmd_final`・`cmd_prereg_run`、`bet_opt_search.py` の `cmd_final`、`prereg_rules.py`・`upset_types.py` の `run`、`bet_forward_ledger.py`・`bet_age2_premium.py`・`bet_r2_combine.py`・`bet_r2_walkthrough.py` の `main`。
  - 第 2 段の推奨（`src/pipeline/inference/stage2_bets.py`）は `bet_cycles` の補助の関数（`normalize_probs`・`play`・`_skip` など）だけを使い、guard のある関数を呼ばない。`bet_cycles` を import しても台帳は読み込まれない。
  - 候補の影のランナー（`candidate_runner.py`・`scripts/cron/candidate_runner.sh`）・API（`src/api/`）・`src/pipeline/`・`src/ops/jobs/` に台帳の呼び出しは無い。cron から上の CLI を呼ぶ所も無い。
  - guard を呼んでよい入口はテストの `_GUARD_ENTRY` に固定した（bet_cycles の `cmd_final`・`cmd_prereg_run`、bet_age2_premium の `run`、bet_r2_walkthrough の `build`、ほかは `main`・`cmd_run`・`cmd_final`）。guard を呼ぶファイルを足したら、そこに入口を書く。
  - `src/api`・`src/pipeline`・`src/ops`・`scripts/cron`・`scripts/ops`・`scripts/server` に、台帳と guard を持つ CLI の名前が出てこないこともテストで確かめる。
  - import しても台帳が読み込まれないことは補助の確かめで、根拠の主は上の静的な確かめ。

enforce の系統で止まりやすい場面:

- `confirm`・`reproduce` は、凍結のファイルと道具を commit 済みの clean な木で流す。共有の作業ツリーで別のセッションの未コミットの変更があると、`frozen_dirty` で止まる。
- 道具は jovyan で流す。sudo で流したときは、元の利用者（`SUDO_USER`）の共有の置き場を使う。
- `check_dates` のはみ出しで拒否されると、その実行は aborted として 1 回と数える。申告する期間は、読む表の範囲と合わせてから流す。
- `bet_forward_ledger.py`（前向きの集計、`forward`）は、期間が 11/21 より前に掛かると拒否される。前向きの集計は 11/21 からの期間で流す。
  - forward の行が積もると、同じ前向きの期間への bet_cycles の `confirm` は eval_only に落ちる（前向きの判定は forward のまま集計する）。

## 段取り

1. 2026-11-18: 本体・決まり・テスト・CI の検査（一覧だけ）・過去の分。全部 warn。
2. 各担当の道具への組み込み。graded-handicapper（§13・§14）は担当が入れる。bet-optimizer・rules-bet-opt・model-improve-cycles・longshot-model は相談のうえで入れる。
3. 組み込みを確かめた系統から、`enforce_families` に足す（統括役に知らせてから）。
4. 全系統の後に、`python -m src.scripts.ci.check_ledger_guard --strict` を CI で失敗にする。

CI では、git の台帳が追記だけか（`check-append-only`）と、台帳を呼ばない評価の道具の一覧（`check_ledger_guard`）を見る。一覧は手掛かりで、確実ではない。

## 起点年の照合（B7）

- 事象: 起点年は `KEIBA_FIRST_YEAR` が無いと 2020 年になる（`src/pipeline/models/training_years.py`）。2016 年からの表でも、付け忘れると分割・履歴の年が変わる。
  - 2026-11-07〜08 には、2020 年起点で学習した v4・v5・v6e の推論が 2016 年からの履歴を読んだ（`docs/operations/frozen_tracking.md`）。
- 今の状態: 本番 v6e は 2020〜2025 年で学習し、推論も 2020 年起点で合っている。候補 v6e16 は `config/candidate_runtime.json` で 2016 を当てていて、こちらも合っている。
- やったこと:
  - `first_year_stamp()` を足した。`EnsembleBundle.save` が `bundle.json` に起点年を書く。
  - `bundle_first_year()`・`check_first_year()` を足した。照合して、食い違えば止める（strict）か警告する。
- 残り:
  - 学習の道具（`fit_final_v6.py`・`model_cycles_final.py`）の `fit_info.json` に、実際の学習の行の最小の年を書き、照合する。担当（model-improve-cycles）と相談のうえで入れる。
    - `rebuild_base_2016.py` は実行中だけ `TRAINING_YEARS` を差し替え、`FIRST_YEAR` は 2020 のままにする。そのため、設定の値でなくデータの年を記録する。
  - 推論の経路（`EnsembleBundle.load`・影のランナー）への照合は、11/23（月）の開催の後に入れる（統括役の条件）。stg で 1 回通すのは週明け。

### 追い切りの欠け方の年ごとの差（2026-11-18、手元の表・0 円）

`retrain_out/base_2016/tp40/groups/CY.parquet`（ウッドの集計 `wc_*`、2016〜2025 年）の空の割合。

| 年 | wc_c1_n（本数） | wc_c1_d1（前回との差） |
|---|---|---|
| 2016 | 91.0% | 92.7% |
| 2017 | 80.5% | 83.9% |
| 2018 | 65.7% | 71.4% |
| 2019 | 39.2% | 49.3% |
| 2020 | 9.9% | 25.0% |
| 2021〜2024 | 9.6%→6.3% | 24.8%→19.4% |
| 2025 | 6.0% | 18.9% |

- 事象: 2016〜2019 年は欠けが 39〜91% で、2020 年以降（6〜10%）と桁が違う。2025 年（test）は 6.0% で、最も埋まっている。
  - `_d1` が `_n` より約 13〜16pt 多く欠けるのは、前回の追い切りが無い馬では差が作れないため（定義どおり）。
- 根拠: 古い年の追い切りは、後の年も走った馬だけが取り直されていた。そのため値の有無が「その馬が後も走ったか」で決まる（生き残りの漏れ、data-audit の D1。値の有無だけで勝ちの AUC 0.56〜0.66）。2020 年以降は取得の作りが変わり、欠けは取得の遅れなどだけになる。
- 対応: model-improve-cycles の r04 は、2016〜2019 年の行の `wc_*`・`tl_*` を欠けにそろえて学習している（`blank_before 2020`、`docs/operations/retrain_trials/model_cycles_20261116.md`）。
  - 起点年を比べるときは、この欠けにそろえた扱いを前提にする。そろえない表（`base_2016` の CY 群のまま）と比べると、起点年の差に欠けの差が混ざる。
- 起点年への含み:
  - 2016 年起点の学習では、「追い切りが空」の行の大半が 2016〜2019 年の行になる。
  - 2026 年の推論では追い切りが 72〜99% 欠ける期間があった（data-audit の B1）。その行の予測は、古い年の行で学んだ「空のときの振る舞い」に寄る。
  - 2020 年起点（本番 v6e）では、空の行は 6〜10% で、取得の遅れによるものに近い。
  - 起点年を変える比較では、追い切りの欠けの扱いも同じにそろえる（起点年だけの差にならない）。担当（model-improve-cycles）に渡す。
