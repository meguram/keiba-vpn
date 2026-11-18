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
  - 2025 年の追い切りの埋まり方と 2016〜2019 年の欠け方の差を、手元の表で数えて、この節に書く。
