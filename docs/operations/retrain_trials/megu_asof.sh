#!/bin/bash
# めぐ指数 v3a を年ごとの as-of で作る: Y=2022..2025 は 2016〜(Y-1) で当てはめた基準、2016〜2021 は 2016〜2021 の基準。
set -e
R=/home/jovyan/work/retrain_out; cd /home/jovyan/work/keiba-vpn
export KEIBA_ENV=stg PYTHONPATH=.
run() { # $1=基準の年 $2=計算する年
  M=$R/megu_v3a_ref$1; [ -f $M/par.csv ] || nice -n 10 python3 -m src.pipeline.megu_index.v3a_build build --frames-dir $R/megu_frames --ref-years $1 --out-dir $M --beta-weight 0.05 --pace-cap 3
  nice -n 10 python3 -m src.pipeline.megu_index.recompute --out-dir $R/megu_asof --years $2 --v3a-model-dir $M
}
run 2016-2021 2016-2021
run 2016-2021 2022
run 2016-2022 2023
run 2016-2023 2024
run 2016-2024 2025
echo DONE
