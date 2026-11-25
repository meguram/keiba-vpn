#!/usr/bin/env bash
# stg の版（rstg）だけで使う動作確認用の test/test の 1 件を作り、Secret Manager の KEIBA_STG_TEST_USER に登録する。
# 値（salt・ハッシュ）は画面・引数・ファイルに出さず、パイプで put_secret.sh に渡す。パスワードは公開済みの test（実会員とは無関係）。
#
#   bash scripts/ops/stg_test_user_secret.sh        # ホストか Cloud Shell で（gcloud の認証が要る）
#
# 読むのは src/api/auth.py::_stg_test_user（KEIBA_TEST_USER=1 のときだけ）。載せるのは scripts/gcp/deploy_cloud_run_api.sh（API_TEST_USER=1＝stg のときだけ）。
# 会員名簿の正本（GCS とホストの写し）は本番と共有なので、test はそこに入れない。
# 費用: 有効な版が 7 個目になると月 約 9 円（6 個までは無料）。読み出しは版の起動ごとで月 約 0 円。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
python3 -c '
import hashlib, json, os, sys
salt = os.urandom(16)
it = 200000
sys.stdout.write(json.dumps({"id": "test", "role": "member", "algo": "pbkdf2_sha256", "iterations": it,
                             "salt": salt.hex(), "hash": hashlib.pbkdf2_hmac("sha256", b"test", salt, it).hex(),
                             "requires_env": "KEIBA_TEST_USER"}))
' | bash scripts/gcp/put_secret.sh KEIBA_STG_TEST_USER
bash scripts/gcp/put_secret.sh KEIBA_STG_TEST_USER --retire >/dev/null     # 古い版を無効にする（有効な版を増やさない）
