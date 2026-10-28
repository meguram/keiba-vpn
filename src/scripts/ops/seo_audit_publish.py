"""SEO 点検を実行して GCS の ``ops/seo/audit.json`` に置く（監視ポータルの『SEO』が読む）。CI か開発 PC で実行する。

    python -m src.scripts.ops.seo_audit_publish            # GCS に置く（GCS_BUCKET と認証が要る）
    python -m src.scripts.ops.seo_audit_publish --print    # 置かずに要約だけ表示

``site_public/`` が要るので、監視ポータルのイメージ（Cloud Run）では動かない。
"""

from __future__ import annotations

import sys

from src.ops import seo_audit


def main(argv: list[str]) -> int:
    if "--print" in argv:
        d = seo_audit.audit()
        print(f"score={d['score']} counts={d['counts']} placeholders={d['placeholders']} pages={len(d['pages'])}")
        return 0
    from src.ops.gcs_io import default_io

    d = seo_audit.publish(default_io())
    if d.get("status") != "ok":
        print(f"SEO 点検に失敗: {d.get('error')}", file=sys.stderr)
        return 1
    print(f"公開しました: {seo_audit.OBJECT} score={d['score']} counts={d['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
