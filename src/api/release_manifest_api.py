"""この環境の実効リリース・マニフェストを返す（stg / prod の挙動が揃っているかの照合用）。

トークン（環境変数 KEIBA_MANIFEST_TOKEN、ヘッダ X-Keiba-Manifest-Token）が一致したときだけ返す。未設定なら無効（404）。
返す中身は src/ops/release_manifest.py の allowlist のみ（ハッシュ・版名・挙動フラグの値。秘密・URL・バケット名は含まない）。
"""

from __future__ import annotations

import asyncio
import hmac
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _authorized(request: Request) -> bool:
    expected = (os.environ.get("KEIBA_MANIFEST_TOKEN") or "").strip()
    if not expected:
        return False
    got = (request.headers.get("x-keiba-manifest-token") or "").strip()
    return bool(got) and hmac.compare_digest(got.encode(), expected.encode())


@router.get("/api/ops/release-manifest", response_class=JSONResponse)
async def release_manifest(request: Request, detail: int = 0):
    if not _authorized(request):
        return JSONResponse({"status": "not_found"}, status_code=404)
    from src.ops.release_manifest import build_manifest

    data = await asyncio.to_thread(lambda: build_manifest(detail=bool(detail)))
    return JSONResponse(data, headers={"Cache-Control": "no-store"})
