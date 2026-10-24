#!/usr/bin/env bash
# ホスト(root)の crontab から、不要な外部アクセスを減らす（2026-10-22 の cron 監査。根拠: docs/operations/cron_access_audit.md）。
#   sudo bash scripts/ops/cron_prune_apply.sh --dry-run         # 変更前後の行を表示するだけ
#   sudo bash scripts/ops/cron_prune_apply.sh                   # バックアップを作って適用
#   sudo bash scripts/ops/cron_prune_apply.sh --with-research   # 研究専用の cron(sporthorse の誕生日取得)も止める
#   bash scripts/ops/cron_prune_restore.sh [バックアップ]       # 元に戻す
# 変更の内容: ①未処理が 0 件で完了済みのバックフィル(2022〜2025 の fast、2024・2025 の full)を止める
#             ②2026 のバックフィルを毎日→週 1 回に減らし、③採用ゼロの horse-remarks(馬の備考・指数。1日 500〜900 件)を止める。行は消さず、コメントにして理由を残す。
# テスト用: --crontab-file F --out O （F を読んで O に書く。crontab には触れない）
set -euo pipefail
DRY=0; RESEARCH=0; IN=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1; shift;; --with-research) RESEARCH=1; shift;;
  --crontab-file) IN="$2"; shift 2;; --out) OUT="$2"; shift 2;;
  *) echo "不明な引数: $1" >&2; exit 2;; esac; done
BACKUP_DIR="${CRON_BACKUP_DIR:-/var/backups/keiba-cron}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [ -n "$IN" ]; then cp "$IN" "$TMP/before.txt"; else crontab -l > "$TMP/before.txt" 2>/dev/null || { echo "crontab を読めません(root で実行)" >&2; exit 1; }; fi

RESEARCH="$RESEARCH" python3 - "$TMP/before.txt" "$TMP/after.txt" <<'PY'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]
STAMP = "2026-10-22"
COMMENT_OUT = {
    "backfill-2025-fast": "未処理 0 件で完了済み", "backfill-2024-fast": "未処理 0 件で完了済み",
    "backfill-2023-fast": "未処理 0 件で完了済み", "backfill-2022-fast": "未処理 0 件で完了済み",
    "backfill-2025-full": "未処理 0 件で完了済み", "backfill-2024-full": "未処理 0 件で完了済み",
    "horse-remarks-daily": "現行モデルで採用ゼロ。1日 500〜900 件の取得に見合わない",
}
if os.environ.get("RESEARCH") == "1":
    COMMENT_OUT["sporthorse-births-nightly"] = "研究専用・不採用の結論済み"
RESCHEDULE = {  # 名前: (新しい 5 フィールド, 理由)
    "backfill-2026-fast": ("0 15 * * 0", "新しい開催日の補填のみ。毎日→週1"),
    "backfill-2026-full": ("50 22 * * 0", "新しい開催日の補填のみ。毎日→週1"),
}
def job_of(line):
    m = re.search(r"cron_run\.sh\s+([A-Za-z0-9_.-]+)\s+--", line)
    return m.group(1) if m else None
out, changed = [], []
for line in open(src, encoding="utf-8").read().splitlines():
    name = job_of(line)
    if line.lstrip().startswith("#") or name is None:
        out.append(line); continue
    if name in COMMENT_OUT:
        new = f"# PRUNED {STAMP}: {COMMENT_OUT[name]} | {line}"
        out.append(new); changed.append(("停止", name, line, new)); continue
    if name in RESCHEDULE:
        fields = line.split(None, 5)
        if len(fields) >= 6 and fields[:5] != RESCHEDULE[name][0].split():
            new_sched = RESCHEDULE[name][0]
            note = f"# RESCHEDULED {STAMP}: {name} {' '.join(fields[:5])} -> {new_sched} ({RESCHEDULE[name][1]})"
            new = f"{new_sched} {fields[5]}"
            out.append(note); out.append(new); changed.append(("頻度減", name, line, new)); continue
    out.append(line)
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
for kind, name, old, new in changed:
    print(f"[{kind}] {name}\n  前: {old[:150]}\n  後: {new[:150]}")
print(f"変更 {len(changed)} 行")
PY

if [ -n "$OUT" ]; then cp "$TMP/after.txt" "$OUT"; echo "出力: $OUT"; exit 0; fi
[ "$DRY" = 1 ] && { echo "(dry-run) 適用しません"; exit 0; }
mkdir -p "$BACKUP_DIR"; chmod 700 "$BACKUP_DIR"
B="$BACKUP_DIR/crontab_$(date +%Y%m%d_%H%M%S).txt"; cp "$TMP/before.txt" "$B"; chmod 600 "$B"
crontab "$TMP/after.txt"
echo "適用しました。バックアップ: $B（戻すとき: bash scripts/ops/cron_prune_restore.sh $B）"
