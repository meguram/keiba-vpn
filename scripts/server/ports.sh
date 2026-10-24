# shellcheck shell=bash
# config/ports.env を読み込み、未設定の KEIBA_PORT_* だけ export する（環境変数が優先）。
# 使い方: source "$ROOT/scripts/server/ports.sh"; load_keiba_ports "$ROOT"

load_keiba_ports() {
  local root="${1:?root}" file line key val
  file="$root/config/ports.env"
  [[ -f "$file" ]] || { echo "[ports] 警告: $file がありません" >&2; return 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; val="${line#*=}"
    key="${key//[[:space:]]/}"; val="${val//[[:space:]]/}"
    [[ "$key" =~ ^KEIBA_PORT_[A-Z0-9_]+$ ]] || continue
    if [[ -z "${!key:-}" ]]; then
      export "$key=$val"
    fi
  done <"$file"
}

# keiba_port <dev|stg|prod> <frontend|flask|fastapi|mlflow>
keiba_port() {
  local var="KEIBA_PORT_${1^^}_${2^^}"
  printf '%s' "${!var:-}"
}
