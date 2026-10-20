#!/bin/bash
# v5 の shadow を有効にする（.env への追記のみ）。bundle_v5_2025 が完成していること。実行前に .env を退避。
set -e
R=/home/jovyan/work/keiba-vpn
B=$R/data/local/modeling/training_pipeline/final_v5/bundle_v5_2025
[ -f $B/fit_info.json ] || { echo "bundle_v5_2025 が未完成"; exit 2; }
grep -q "^KEIBA_V5_SHADOW=" $R/.env && { echo "既に KEIBA_V5_SHADOW がある。変更しない"; exit 0; }
TS=$(date +%Y%m%d_%H%M%S)
sudo -n mkdir -p $R/data/local/meta/cron_backups && sudo -n cp -a $R/.env $R/data/local/meta/cron_backups/env_before_v5_shadow_$TS
{ echo ""; echo "# v5 shadow（v4 と並走。出力 predictions_v5/。追加: $(date +%F)）"; echo "KEIBA_V5_SHADOW=1"; echo "KEIBA_V5_BUNDLE=$B"; echo "KEIBA_V5_STAGE=$R/data/local/modeling/training_pipeline/final_stage/v4_fixed"; echo "KEIBA_V5_TL_REF=$R/data/local/modeling/training_pipeline_v4/tp35/tl_reference_v2.parquet"; } | sudo -n tee -a $R/.env >/dev/null
echo "追記した（次回のランナー起動から有効）。退避: $R/data/local/meta/cron_backups/env_before_v5_shadow_$TS"
echo "戻す: .env から KEIBA_V5_ の行を消す（または退避したファイルを戻す）"
