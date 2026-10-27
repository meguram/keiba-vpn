# 想定めぐ指数: 階層縮約モデル（HEM）への置き換えと、旧版への戻し方

画面・API（`/api/race/{id}/predictions` の `expected_megu`）の値を、旧式（直近5走の加重平均＋条件補正）から階層縮約モデル（HEM、決定ログ §75、`src/scripts/data/hem_megu.py`）に置き換えた。応答のキー・単位（1点＝0.1秒）は同じ。`value`・`range`・`adjust` が HEM になり、`base_value`・`change`・`reasons` は従来どおり。

## 仕組み

- 推論: `src/pipeline/inference/hem_megu_infer.py`（学習時と同じ関数で項を作る）→ `expected_megu.attach_expected_megu` が出走馬ぜんぶをまとめて計算。出走馬の履歴だけを使う（低メモリ）。
- 係数: `config/expected_megu/hem.json`（学習済みの係数なので git 管理しない。`config/fitted_artifacts.txt` に登録）。2020〜2021 年でセル平均、2021 年の次走で係数を当てはめ、幅（±）は 2022〜2024 年の残差の標準偏差（走数の区分ごと）。2025 年の被覆率（±1 標準偏差の中に入る割合）は約 74%。
- 騎手の累積表: `data/local/modeling/hem/jockey_table.parquet`（`KEIBA_HEM_STATE` で変更可）。無ければ騎手の項は 0 で動く。

## 更新

- 新しいレースが増えたら騎手の累積表だけ作り直す（係数は変えない）: `python -m src.scripts.data.hem_megu state`
- 係数ごと作り直す: `HEM_EXPORT=<出力先> python -m src.scripts.data.hem_megu export`（`hem.json` と累積表。2022〜2025 年の精度比較も出る）。

## 旧版へ戻す（ロールバック）

1. `.env` に `KEIBA_EXPECTED_MEGU=legacy` を足して API を再起動（`PORT=23220 PYTHON=… bash scripts/server/restart_server.sh`。キューが空のとき）。
2. または `config/expected_megu/hem.json` を退避（無いと自動で旧版）。旧版の係数 `config/expected_megu/model.json` はそのまま残してある。

## 確認

2025 年（全 41,192 走）: RMSE 11.92（旧 12.67）、相関 0.721（旧 0.700）、レース内相関 0.490（旧 0.466）。出力の平均は同程度（37.3 と 37.0）、標準偏差は縮約で小さい（12.3 と 15.1）。
