"""
弾力的セレクタチェーン

HTML構造が変更されても動作し続けるために、
各要素に対して複数のCSSセレクタ候補を順に試す仕組みを提供する。

構造変更を検知した場合は WARNING ログを出力し、
フォールバックセレクタで処理を継続する。
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger("scraper.selectors")


class SelectorChain:
    """
    複数のCSSセレクタを優先順位で試し、最初にマッチした結果を返す。

    primary が失敗すると WARNING を出してフォールバックに移行する。
    すべてのセレクタが失敗した場合は None を返す。

    Usage:
        chain = SelectorChain("result_table", [
            "table.race_table_01",
            "table[class*='race_table']",
            "table[summary*='レース結果']",
            "table.nk_tb_common",
        ])
        table = chain.select_one(soup)
    """

    def __init__(self, name: str, selectors: list[str]):
        self.name = name
        self.selectors = selectors
        self._hit_cache: str | None = None

    def select_one(self, soup: BeautifulSoup | Tag) -> Tag | None:
        if self._hit_cache:
            result = soup.select_one(self._hit_cache)
            if result:
                return result
            self._hit_cache = None

        for i, sel in enumerate(self.selectors):
            result = soup.select_one(sel)
            if result:
                if i > 0:
                    logger.warning(
                        "⚠️ HTML構造変更検知 [%s]: primary '%s' 失敗 → fallback '%s' で取得",
                        self.name, self.selectors[0], sel,
                    )
                self._hit_cache = sel
                return result

        logger.error("❌ 全セレクタ失敗 [%s]: %s", self.name, self.selectors)
        return None

    def select(self, soup: BeautifulSoup | Tag) -> list[Tag]:
        if self._hit_cache:
            results = soup.select(self._hit_cache)
            if results:
                return results
            self._hit_cache = None

        for i, sel in enumerate(self.selectors):
            results = soup.select(sel)
            if results:
                if i > 0:
                    logger.warning(
                        "⚠️ HTML構造変更検知 [%s]: primary '%s' 失敗 → fallback '%s'",
                        self.name, self.selectors[0], sel,
                    )
                self._hit_cache = sel
                return results

        logger.error("❌ 全セレクタ失敗 [%s]: %s", self.name, self.selectors)
        return []


def safe_text(tag: Tag | None) -> str:
    """タグのテキストを安全に取得 (None対応, strip済み)"""
    if tag is None:
        return ""
    return tag.get_text(strip=True)


def safe_attr(tag: Tag | None, attr: str) -> str:
    """タグの属性を安全に取得"""
    if tag is None:
        return ""
    return tag.get(attr, "")


def extract_id_from_url(url: str, pattern: str = r"/(\w+)/?$") -> str:
    """URLからIDを抽出"""
    m = re.search(pattern, url)
    return m.group(1) if m else ""


def extract_numbers(text: str) -> list[str]:
    """テキストから数値を全て抽出"""
    return re.findall(r"-?\d+\.?\d*", text)


def parse_weight_change(text: str) -> tuple[int, int]:
    """'480(+4)' → (480, 4)"""
    m = re.match(r"(\d+)\s*\(([+\-]?\d+)\)", text.replace(" ", ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = extract_numbers(text)
    if nums:
        return int(float(nums[0])), 0
    return 0, 0
