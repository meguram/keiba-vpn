"""決済を有効にする前の点検。python -m src.billing.preflight [--env-file .env]  （未達があれば終了コード 1）。秘密の値は表示しない。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.billing import service

ROOT = Path(__file__).resolve().parents[2]
LEGAL = ROOT / "frontend" / "app" / "legal" / "[doc]" / "page.tsx"


def _env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("\"'")
    except OSError:
        pass
    return out


def checks(env: dict[str, str]) -> list[tuple[str, bool, str]]:
    cfg = service.config()
    key = env.get("STRIPE_SECRET_KEY", "")
    live = key.startswith(("sk_live_", "rk_live_"))
    feat = (ROOT / "frontend" / "lib" / "features.ts").read_text(encoding="utf-8")
    legal = LEGAL.read_text(encoding="utf-8") if LEGAL.exists() else ""
    return [
        ("データの利用許諾の確認（config/billing.json license_confirmed）", bool(cfg.get("license_confirmed")), "netkeiba 等の許諾を確認してから true に"),
        ("STRIPE_SECRET_KEY", bool(key), f"{'本番' if live else 'テスト'}キー" if key else "未設定（制限付きキー rk_ を推奨）"),
        ("STRIPE_WEBHOOK_SECRET（whsec_）", env.get("STRIPE_WEBHOOK_SECRET", "").startswith("whsec_"), "Stripe のエンドポイント登録後に表示される署名シークレット"),
        ("STRIPE_PRICE_ID（price_）", env.get("STRIPE_PRICE_ID", "").startswith("price_"), "月額の価格 ID"),
        ("KEIBA_BILLING=1", env.get("KEIBA_BILLING", "") in ("1", "true", "yes", "on"), "最後に入れる（入れると API が有効になる）"),
        ("法的表記の【要記入】が残っていない", bool(legal) and "【要記入】" not in legal, "frontend/app/legal/[doc]/page.tsx"),
        ("URL が本番のドメイン", all(str(cfg.get(k, "")).startswith("https://") for k in ("success_url", "cancel_url", "portal_return_url")), "config/billing.json"),
        ("UI の公開（features.ts の billing）", any("billing:" in ln and "enabled: true" in ln for ln in feat.splitlines()), "enabled を true に（最後）"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    a = ap.parse_args()
    env = {**_env_file(Path(a.env_file)), **{k: v for k, v in os.environ.items() if k.startswith(("STRIPE_", "KEIBA_BILLING"))}}
    bad = 0
    for name, ok, hint in checks(env):
        print(f"[{'OK' if ok else '未'}] {name}" + ("" if ok else f" — {hint}"))
        bad += not ok
    print("\n全部そろいました" if not bad else f"\n未達 {bad} 件（この間、決済は無効のまま）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
