"""
`sires` テーブルへの UPSERT ユーティリティ。

血統スクレイピング時に呼ばれ、種牡馬レコード（sire_id, sire_name, sire_line）を
INSERT OR UPDATE する。sire_line が取得できていない場合は既存値を保持する。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import Sire

logger = logging.getLogger(__name__)


def upsert_sire(
    session: Session,
    sire_id: str,
    sire_name: str,
    sire_line: str | None = None,
) -> None:
    """1頭分の種牡馬レコードを sires テーブルへ UPSERT する。

    - 新規の場合は INSERT
    - 既存の場合:
        - sire_name が空でなければ上書き
        - sire_line は None / 空文字でなければ上書き（既存値を空で上書きしない）
    """
    if not sire_id:
        return

    sire_name = (sire_name or "").strip()
    sire_line_clean = (sire_line or "").strip() or None

    existing: Sire | None = session.get(Sire, sire_id)
    if existing is None:
        session.add(Sire(
            sire_id=sire_id,
            sire_name=sire_name,
            sire_line=sire_line_clean,
        ))
        logger.debug("sires INSERT: %s %s (%s)", sire_id, sire_name, sire_line_clean)
    else:
        changed = False
        if sire_name and existing.sire_name != sire_name:
            existing.sire_name = sire_name
            changed = True
        if sire_line_clean and existing.sire_line != sire_line_clean:
            existing.sire_line = sire_line_clean
            changed = True
        if changed:
            logger.debug("sires UPDATE: %s %s (%s)", sire_id, sire_name, sire_line_clean)


def upsert_sires_from_ancestors(
    session: Session,
    ancestors: list[dict[str, Any]],
) -> int:
    """祖先リスト（parse_blood_table_5gen 相当の dict リスト）から sires を一括 UPSERT する。

    Parameters
    ----------
    ancestors:
        各要素は {horse_id, name, sire_line?, generation, position, sex, ...}
        sex が '牡' または horse_id が '/horse/sire/' 形式の場合に種牡馬と判定。
    """
    count = 0
    for anc in ancestors:
        horse_id = (anc.get("horse_id") or "").strip()
        name = (anc.get("name") or "").strip()
        sex = anc.get("sex", "")
        sire_line = anc.get("sire_line", "")

        if not horse_id or not name:
            continue
        if sex and sex != "牡":
            continue

        upsert_sire(session, horse_id, name, sire_line)
        count += 1

    return count


def upsert_sires_from_ped_html(html: str, session: Session) -> int:
    """blood_table HTML を解析して sires テーブルへ UPSERT する。

    セッション管理（flush/commit/rollback）は呼び出し元が行う。

    Returns
    -------
    int
        UPSERT した種牡馬数。
    """
    from src.scraper.sire_line_extractor import extract_sires_from_ped_html

    ancestors = extract_sires_from_ped_html(html)
    return upsert_sires_from_ancestors(session, ancestors)


def safe_upsert_sires_from_html(html: str) -> int:
    """DB 接続有無にかかわらず呼べる安全ラッパー。

    scraping ループ内から呼ばれる想定。DB 未接続・エラー時は WARNING ログを出して 0 を返す。
    """
    try:
        from src.db.session import get_session, init_engine
        init_engine()
        with get_session() as sess:
            n = upsert_sires_from_ped_html(html, sess)
            sess.commit()
        return n
    except Exception as exc:
        logger.warning("sires UPSERT スキップ (DB 未接続等): %s", exc)
        return 0
