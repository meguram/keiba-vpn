#!/usr/bin/env bash
# stg（学習PC）: 2020 年以降のすべてのデータポイントが、スキーマに適合した状態で揃っているかを合否で判定する。
#
#   bash docs/todos/verify/stg_data_complete.sh                 # 全件検証（download 量が大きい。下記）
#   BUDGET=5000 MAX_RUNS=100 bash docs/todos/verify/stg_data_complete.sh   # 1 回 5000 件ずつ、全件になるまで繰り返す
#   SINCE=2024-01-01 bash docs/todos/verify/stg_data_complete.sh           # 期間を絞った確認（※ 2020 年より後だと「完全」とは判定されない）
#
# 判定（python -m src.data_health --require-complete）:
#   対象: 2020-01-01 〜 実行日の前日の JRA レース全件 × 重要度 required/recommended のカテゴリ、出走馬、派生データ、インフラ
#   OK  : 不足・スキーマ不適合・未検証が 0、race_lists に不完全な日が無い、検証が全件（DATA_HEALTH_VALIDATE=full）
# 結果: data/local/meta/data_health/stg/{latest.html,race_keys.csv,race_ids/,scrape_plan.json}
#       未達なら race_ids/*.txt（揃っていない race_id）と scrape_plan.json（再取得の計画）を確認する。
#
# 注意: GCP の外（学習PC）から全件 download すると送信料金が発生する（約 $0.12/GB）。初回は数十〜数百 GB ではなく
#       数 GB〜数十 GB 規模の見込みだが未実測。台帳（data/local/meta/data_health/stg/ledger）に結果を残すので、
#       2 回目以降は更新分だけ。BUDGET を指定すると途中で止めても続きから再開できる。
source "$(dirname "$0")/_lib.sh"
TITLE="stg: 2020 年以降の全データポイントの完全性"
export KEIBA_ENV=stg
SINCE="${SINCE:-2020-01-01}"
BUDGET="${BUDGET:-0}"            # 0 = 無制限
MAX_RUNS="${MAX_RUNS:-1}"

echo "$TITLE  (KEIBA_ENV=stg since=$SINCE budget=$BUDGET)"
export DATA_HEALTH_VALIDATE=full DATA_HEALTH_VALIDATE_BUDGET="$BUDGET"

if [ "$(env_get GCS_BUCKET)" = "" ]; then
  fail ".env / .env.stg に GCS_BUCKET が無い（stg は GCS に接続できる必要がある）"; finish; exit $?
fi

rc=1
for run in $(seq 1 "$MAX_RUNS"); do
  echo "---- 実行 $run/$MAX_RUNS"
  python3 -m src.data_health --since "$SINCE" --require-complete --quiet
  rc=$?
  [ "$rc" -ne 3 ] && break               # 完全(0) か、想定外の失敗(2 など)
  # 未達の理由が「検証が途中」だけなら続行、それ以外(不足・不適合)が残っていれば繰り返しても解消しない
  python3 - <<'PY' || break
import json, sys
c = json.load(open("data/local/meta/data_health/stg/latest.json", encoding="utf-8"))["completeness"]
only_progress = all(r["code"] in ("validate_incomplete", "unvalidated") for r in c["reasons"])
sys.exit(0 if only_progress else 1)
PY
done

python3 - <<'PY'
import json
r = json.load(open("data/local/meta/data_health/stg/latest.json", encoding="utf-8"))
c, rc = r["completeness"], r["race_coverage"]
print(f"  対象 {rc['universe']['races']} レース / 範囲 {c['scope']} / 重要度 {','.join(c['levels'])}")
for x in c["reasons"][:20]:
    print(f"  - {x['message']}")
PY

if [ "$rc" -eq 0 ]; then pass "2020 年以降の全データが揃い、スキーマに適合している"
elif [ "$rc" -eq 3 ]; then fail "未達の項目あり（data/local/meta/data_health/stg/latest.html、race_ids/*.txt、scrape_plan.json を確認）"
else fail "チェック自体が失敗（終了コード $rc）"; fi
finish
