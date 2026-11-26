"""予測の API を「AI の予測値・印・サンプル買い目」だけにする切り替え（KEIBA_PRED_SLIM。既定 1）。

ユーザーの決定（2026-11-26）: めぐ予想は AI 予測値と買い目推奨だけでよい。表示は計算済みの値を出す。
1 のとき、/api/race/{id}/predictions は要求のときの付け足し（位置取り・ペースの整合・追走難度 v2・ラップ予想・ペースの型・
調教の被覆・想定めぐ指数）を計算しない。起動直後の 1 本目が 10〜11 秒、温まった台でも 1 本 約 2 秒かかっていた（stg の計測）。
0 は API の付け足しの計算だけを戻す（画面の欄は 2026-11-26 に消したので、0 にしても画面には出ない。障害の切り分け用）。
本番の昇格の後に問題が無ければ、付け足しのコードごと消す。
ホストの tracking_v2・lap_forecast の保存は v6e の特徴量と候補のランナーが読むので、この切り替えでは止めない。
"""
from __future__ import annotations

import os

# 切り替えで応答から無くなる機能（匿名の応答の access.member_only の表示から外す。src/api/tier.py の LOCKED_GROUPS の名前）
DROPPED_GROUPS = ("展開・ラップ・追走難度", "めぐ指数・成長度・クラス基準")


def enabled() -> bool:
    return (os.environ.get("KEIBA_PRED_SLIM") or "1").strip() != "0"
