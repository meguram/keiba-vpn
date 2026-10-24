"""取得プログラムが名乗る User-Agent（固定・識別可能）。

方針（docs/operations/scraping_policy.md）: ブラウザを装わない。取得元が運営上の連絡を取れるよう、
プロジェクトの識別トークンと連絡先（環境変数 KEIBA_SCRAPER_CONTACT。メールまたは URL。リポジトリには書かない）を含める。
制限を受けても User-Agent を切り替えて続行しない。
"""

from __future__ import annotations

import logging
import os

TOKEN = "keiba-research-bot/1.0"
_warned = False


def contact(env: dict[str, str] | None = None) -> str:
    e = os.environ if env is None else env
    return (e.get("KEIBA_SCRAPER_CONTACT") or "").strip()


def user_agent(env: dict[str, str] | None = None) -> str:
    """``Mozilla/5.0 (compatible; keiba-research-bot/1.0; +<連絡先>)``。連絡先が未設定なら識別トークンのみ。
    先頭を ``Mozilla/5.0 (compatible; …)`` にするのは、検索ボットなどが一般に使う書式で、素の文字列だけの UA を弾くサーバーがあるため。"""
    c = contact(env)
    return f"Mozilla/5.0 (compatible; {TOKEN}; +{c})" if c else f"Mozilla/5.0 (compatible; {TOKEN})"


def warn_if_no_contact(logger: logging.Logger) -> None:
    """連絡先が未設定のとき、プロセスごとに 1 回だけ警告する。"""
    global _warned
    if _warned or contact():
        return
    _warned = True
    logger.warning("KEIBA_SCRAPER_CONTACT が未設定です。User-Agent に連絡先が入りません（.env に連絡先のメールまたは URL を設定してください）")
