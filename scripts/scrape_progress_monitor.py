"""
スクレイピング進捗モニター（2プロセス同時表示）
  python scripts/scrape_progress_monitor.py            # 10秒ごとに更新
  python scripts/scrape_progress_monitor.py --once     # 1回表示して終了
  python scripts/scrape_progress_monitor.py --interval 5
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT           = Path(__file__).resolve().parents[1]
PED_DIR         = _ROOT / "data" / "local" / "horse_pedigree_5gen"
NOT_FOUND       = _ROOT / "data" / "meta" / "ancestors_not_found.json"
ANCESTORS_STATUS = _ROOT / "data" / "meta" / "ancestors_upward_status.json"
PATCH_SEX_STATUS = _ROOT / "data" / "meta" / "patch_sex_status.json"

BAR_WIDTH = 44


# ── ユーティリティ ───────────────────────────────────────────────
def bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + "?" * width + "]"
    pct  = min(done / total, 1.0)
    fill = int(pct * width)
    return "[" + "█" * fill + "░" * (width - fill) + f"] {pct*100:5.1f}%"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 86400 * 30:
        return "---"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_finish(eta_sec: float) -> str:
    if eta_sec <= 0:
        return "---"
    finish = time.localtime(time.time() + eta_sec)
    return time.strftime("%m/%d %H:%M", finish)


def is_running(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def _read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# ── ① sex 補完モニター ───────────────────────────────────────────
def read_patch_sex_status() -> dict:
    return _read_json(PATCH_SEX_STATUS)


def render_patch_sex(st: dict, running: bool) -> str:
    done     = st.get("done", 0)
    total    = st.get("total", 46811)
    updated  = st.get("updated", 0)
    failed   = st.get("failed", 0)
    rate     = st.get("rate_per_hour", 0.0)
    eta_sec  = st.get("eta_sec", -1)
    elapsed  = st.get("elapsed_sec", 0)
    status_ico = "🟢 稼働中" if running else ("✅ 完了" if done >= total and total > 0 else "🔴 停止")

    lines = [
        f"  ┌{'─'*58}┐",
        f"  │  ① sex 補完           {time.strftime('%H:%M:%S')}  {status_ico:<12}│",
        f"  ├{'─'*58}┤",
        f"  │  処理済み              : {done:>8,} / {total:,}{'':<14}│",
        f"  │  更新成功              : {updated:>8,} 件{'':<22}│",
        f"  │  失敗                  : {failed:>8,} 件{'':<22}│",
        f"  ├{'─'*58}┤",
        f"  │  進捗  {bar(done, total):<49}│",
    ]
    rate_str    = f"{rate:>7,.0f} 件/時" if rate > 0 else "  計測中...  "
    eta_str     = fmt_eta(eta_sec)      if rate > 0 else "---       "
    finish_str  = fmt_finish(eta_sec)   if rate > 0 else "---            "
    elapsed_str = f"{elapsed/60:.1f} 分" if rate > 0 else "---"
    lines += [
        f"  │  速度  ≈ {rate_str:<48}│",
        f"  │  ETA   ≈ {eta_str:<10}  (完了予測: {finish_str:<14}){'':<4}│",
        f"  │  経過    {elapsed_str:<48}│",
        f"  └{'─'*58}┘",
    ]
    return "\n".join(lines)


# ── ② 祖先血統スクレイパーモニター ──────────────────────────────
def read_ancestors_status() -> dict:
    st = _read_json(ANCESTORS_STATUS)
    if st:
        return st
    # フォールバック: ファイル数カウント
    cur = sum(1 for _ in PED_DIR.rglob("*.json"))
    nf  = _read_json(NOT_FOUND).get("count", 0)
    return {
        "total_existing":     cur,
        "not_found_this_run": nf,
        "scraped_this_run":   0,
        "total_target":       78824,
        "remaining":          max(78824 - cur - nf, 0),
        "rate_per_hour":      0,
        "eta_sec":            -1,
        "elapsed_min":        0,
        "running":            True,
        "timestamp":          "---",
    }


def render_ancestors(st: dict, running: bool) -> str:
    total      = st.get("total_target", 78824)
    existing   = st.get("total_existing", 0)
    not_found  = st.get("not_found_this_run", 0)
    scraped    = st.get("scraped_this_run", 0)
    remaining  = st.get("remaining", max(total - existing, 0))
    rate       = st.get("rate_per_hour", 0.0)
    eta_sec    = st.get("eta_sec", -1)
    elapsed    = st.get("elapsed_min", 0)
    ts         = st.get("timestamp", "---")
    status_ico = "🟢 稼働中" if running else "🔴 停止"

    lines = [
        f"  ┌{'─'*58}┐",
        f"  │  ② 祖先血統スクレイパー {time.strftime('%H:%M:%S')}  {status_ico:<12}│",
        f"  │  最終更新: {ts:<44}│",
        f"  ├{'─'*58}┤",
        f"  │  ローカルJSON取得済み  : {existing:>8,} 件{'':<22}│",
        f"  │  404/データなし        : {not_found:>8,} 件{'':<22}│",
        f"  │  今回取得分            : {scraped:>8,} 件{'':<22}│",
        f"  │  全ノード数（上限）    : {total:>8,} 件{'':<22}│",
        f"  ├{'─'*58}┤",
        f"  │  進捗  {bar(existing, total):<49}│",
        f"  │  残り  ≈ {remaining:>8,} 件{'':<35}│",
    ]
    rate_str    = f"{rate:>7,.0f} 件/時" if rate > 0 else "  計測中...  "
    eta_str     = fmt_eta(eta_sec)     if rate > 0 else "---       "
    finish_str  = fmt_finish(eta_sec)  if rate > 0 else "---            "
    elapsed_str = f"{elapsed:.1f} 分"  if rate > 0 else "---"
    lines += [
        f"  │  速度  ≈ {rate_str:<48}│",
        f"  │  ETA   ≈ {eta_str:<10}  (完了予測: {finish_str:<14}){'':<4}│",
        f"  │  経過    {elapsed_str:<48}│",
        f"  └{'─'*58}┘",
    ]
    return "\n".join(lines)


# ── メイン描画 ───────────────────────────────────────────────────
def render_all() -> str:
    patch_st   = read_patch_sex_status()
    anc_st     = read_ancestors_status()
    patch_run  = is_running("patch_sex")
    anc_run    = is_running("scrape_ancestors_upward")

    parts = [
        "",
        render_patch_sex(patch_st, patch_run),
        "",
        render_ancestors(anc_st, anc_run),
        "",
        "  ※ Ctrl+C で終了  /  各スクレイパーは別プロセスで稼働継続中",
    ]
    return "\n".join(parts)


DISPLAY_FILE = _ROOT / "data" / "meta" / "scrape_monitor.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--no-file",
        action="store_true",
        help="ファイル書き出しを無効にしてターミナルのみに出力",
    )
    args = ap.parse_args()

    import os

    try:
        while True:
            output = render_all()

            # ── ① ファイルに上書き保存（Cursor で開いて確認用） ──────
            if not args.no_file:
                try:
                    DISPLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
                    DISPLAY_FILE.write_text(output + "\n", encoding="utf-8")
                except Exception:
                    pass

            # ── ② ターミナルに出力 ───────────────────────────────────
            if not args.once:
                os.system("clear")          # bash で確実に画面をクリア
            print(output)

            if args.once:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n  モニター終了")


if __name__ == "__main__":
    main()
