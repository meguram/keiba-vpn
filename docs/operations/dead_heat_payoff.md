# 同着（ダッドヒート）と払戻の扱い

着順に同着があるレースで、券種ごとに『払戻が出る組合せ』を正しく求めるための共通関数と、払戻データの修正の記録（2026-10-24）。

## 共通関数
`src/pipeline/betting/winning_combos.py`（テスト `tests/pipeline/test_winning_combos.py`）
- `finish_groups({馬番: 着順})` → 着順の同着グループ（着順 0 以下の取消・除外・中止は含めない）
- `winning_combos(groups, 券種, n_runners=, brackets=)` → 払戻が出る組合せの集合
- `normalize_payoff(保存形式の払戻)` / `hit_payout(券種, 買い目, 払戻の組ごとの値)` / `diff_expected(...)`（保存値と期待集合の差）

## 同着のルール（実データで確認）
| 券種 | 3 着同着（A, B） | 2 着同着（A, B） | 1 着同着（A, B） |
|---|---|---|---|
| 単勝 | 1 着のみ | 同左 | A・B の 2 頭 |
| 複勝 | 1・2・A・B の 4 頭 | 1・A・B | A・B・3 着 |
| 馬連 | 1-2 | 1-A・1-B | A-B |
| ワイド | **5 組**（1-2・1-A・1-B・2-A・2-B。**A-B は当たらない**） | 3 組（A-B も当たる） | 3 組 |
| 馬単 | 1→2 | 1→A・1→B | A→B・B→A |
| 3 連複 | {1,2,A}・{1,2,B} | {1,A,B} | {A,B,3 着} |
| 3 連単 | 1→2→A・1→2→B | 1→A→B・1→B→A | A→B→3・B→A→3 |

ワイドは『3 連複で当たる組の中の 2 頭』になる（3 着を A・B が分け合うと 3 着以内の枠は 1 つだけなので A-B は当たらない。2 着を分け合えば枠を 2 つ占めるので A-B も当たる）。netkeiba の払戻表も 2024〜2025 年の 3 着同着 18 レースで 5 組だった。当初『6 組のはず』として「欠け」と数えた 18 件は、この規則の取り違えによる誤検出で、データは正しかった。
払戻の金額は券ごとに公式の値があり（同着では按分で半額になる等）、こちらでは計算しない。保存された払戻の組ごとの値を使う。

## 修正した内容
- **連結された払戻の焼き込み**: 2024-01〜2026-03 の `race_result` は複勝・ワイドが 1 つに連結されて保存されていた（5,901 レース）。`payoff_fix` で複数行に直した値を、着順から求めた期待集合と一致することを確認した 5,882 レースだけ保存へ焼き込んだ（`src/scripts/maintenance/bake_payoff_fix.py`、退避 `data/local/backup/payoff_fix_<日時>/`、レポート `data/local/meta/payoff_bake_report.json`）。曖昧だった 19 レースは db.netkeiba から再取得した。
- **精算の同着対応**: `prediction_results.settle_candidate` を `winning_combos` に置き換えた。直した不具合は 3 つ: 3 連複が同着で常に外れ扱い、2 着同着の馬連が外れ扱い、3 着同着のワイド A-B が当たり（概算払戻）扱い。`parse_payoff` に 3 連単を追加（順序を保つ）。
- **洗い出し**: `python -m src.scripts.data.tie_payoff_audit`（結果 `data/local/meta/tie_payoff/`）。年別の同着レース数と、券種別の不整合を出す。

- **平坦表の払戻が空だった本当の原因**: 『払戻なし 55 レース』は取得の失敗ではなく、`export_tables` が JSON 列の長さ 1,024 バイト超を列ごと落としていたためだった。同着や券種の行が多いレースは払戻の JSON が 1,044〜1,100 バイトになり、保存本体にはあるのに `race_result_flat.payoff` だけ空になった。`FLAT_MAX_PAYOFF_BYTES = 8192` を足して払戻だけ上限を広げ（`export_tables.py`）、出力済みの表は `src/scripts/maintenance/fill_flat_payoff.py` で保存本体から穴埋めした（2024 年 5・2025 年 1・2026 年 9 レース。2020〜2023 年は前の作業で補填済み）。以後の `export_tables --force` でも空にならない。
- **平坦表の再出力**: 2024〜2026 年の `race_result_flat` を保存本体から作り直した（退避 `data/local/backup/race_result_flat_dead_heat_<日時>/`）。旧い出力は 2025 年の単勝オッズが追走指数のまま・`round` が 0 など古い値を含んでいたが、再出力で正しい値になった。列は `course_type`・`finish_position_raw`・`finish_status` が増えた。
- **betting の払戻表**: `data/local/modeling/betting/payoff.parquet`（2020〜2025）と `betting_2026/payoff.parquet` を平坦表から作り直した（退避 `data/local/backup/betting_payoff_deadheat*_<日時>/`）。オッズ表（`odds_win_place`・`odds_pair`）は作り直しても同一だったので置き換えていない。

## 評価での除外
- `exotic_method_eval.py`（的中確率の方式比較）は、着順が 1,2,3 と重複なく並ばない同着レースを除く（2022〜2024 年で 50 レース。確率モデルは同着を扱わないため）。
- 馬券の回収率評価（`portfolio_target_eval.py`）は、実際の払戻の組を使うので同着レースも含める。

## 洗い出しの結果（再実行 `tie_payoff_audit`）
同着レース数（1〜3 着の同着）: 2020 年 30・2021 年 21・2022 年 12・2023 年 14・2024 年 22・2025 年 28・2026 年 16（4 着以下の同着は別）。修正後の不整合は 2020 年 `202005050104` の 1 レースだけ（下記）。

## 未解決
- 2020 年 `202005050104`: 払戻（単勝 3 番、複勝 3・5・6）と、結果表の着順（5 番が 1 着、3 番は着順なし）が netkeiba 上で食い違う（降着・失格の疑い）。払戻は公式値なので精算には使えるが、着順ラベル（`rank_tbl`）の 3 番の 10 着は推定値。ラベルを使う学習・評価では除外を検討する。
