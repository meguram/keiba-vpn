"""候補の影のランナー（src/pipeline/inference/candidate_runner.py）の出力を API で使う（試行中）。

- 予測の表示: ``predictions_v6c/<日>/<rid>.parquet`` があれば、v6e の当日の予測より先にそれを使う（新しい候補の第 1 段）。
- 推奨馬券: 未勝利・新馬のレースで ``bets_stage2/<日>/<rid>.json``（凍結した買い方 70ba29fa7b55 の判定）があれば、既定方策 target と同じ形の応答で返す。
  それ以外のレース（重賞・条件戦）は今の target のまま（ユーザーの判断 2026-11-18。レース一覧の推奨度・予測の印も同じ）
  （strategy="target"・sub_strategy="stage2_r2"。サーバの target の分岐＝見送りの隠し方・一覧・精算・印をそのまま通す）。
- ファイルが無いレースは今の表示のまま（v6e・target）。止めの目印があれば、ファイルがあっても今の表示に戻す:
  - ``bets_stage2/_disabled.json``: 予測の表示も推奨馬券も戻す
  - ``predictions_v6c/_disabled.json``: 予測の表示だけ戻す（推奨馬券は第 2 段のまま）
  目印の中身が ``{"disabled": false}`` なら無効（ミラーは削除を伝えないので、戻すときは消さずに上書きする。
  ``python -m src.pipeline.inference.candidate_runner --set-disabled on|off`` が目印とその日のファイルの書き直しをまとめて行う）。
- 実オッズ由来の値（T-60 の単勝オッズ・人気・q・r と、それを offset にした第 2 段の勝率・買いの線）は開発者（view="full"）にだけ出す
  （config/public_fields.json の actual_odds_public）。会員には ◎ の馬番と「試行中」だけ。
- 窓より古い日の取り寄せ（lazy_fetch）はしない（ファイルの無い日の GCS への問い合わせを増やさない）。
- 開始日（KEIBA_STAGE2_START、既定 20261121）より前の日のファイルは読まない（リハーサルの名残を出さない）。
いずれもホストの書いたファイルがミラーで届くだけで、Cloud Run の環境変数・配備は要らない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SUB = "stage2_r2"
NOTICE = ("試行中: 第 2 段のモデル（市場とのずれ）と凍結した買い方（第 2 回）の推奨です。回収率は確かめ中で、"
          "2025 年の検証は回収率 107%（区間 94〜120%）でした。判定は発走 60 分前のオッズによるもので、直前のオッズとは異なります。")
EVIDENCE = ("未勝利・新馬で、2 番人気以下・勝率 5% 以上の馬のうち、発走 60 分前の単勝オッズ（市場）よりモデルが高く見る度合い"
            "（r=log(モデルの勝率/市場の勝率)）が大きい馬がいるレースだけ、その馬の単勝を 100 円買います。")


def _inf() -> Path:
    from src.api.local_predictions import inference_dir

    return inference_dir()


def start_day() -> str:
    return (os.environ.get("KEIBA_STAGE2_START") or "20261121").strip()


def _ok_day(day: str) -> bool:
    return bool(day) and len(day) == 8 and day.isdigit() and day >= start_day()


def _mark_on(p: Path) -> bool:
    """止めの目印が有効か。ファイルがあり、中身が {"disabled": false} でなければ有効（読めない・空も有効＝安全側）。"""
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8") or "{}").get("disabled", True))
    except (OSError, ValueError, AttributeError):
        return True


def disabled(inf: Path | None = None) -> bool:
    return _mark_on((inf or _inf()) / "bets_stage2" / "_disabled.json")


def v6c_disabled(inf: Path | None = None) -> bool:
    inf = inf or _inf()
    return disabled(inf) or _mark_on(inf / "predictions_v6c" / "_disabled.json")


def _local(p: Path, day: str) -> bool:
    """ローカルにあるか。"""
    # 窓（14 日）より古い日の取り寄せ（lazy_fetch）はしない: ファイルの無い日にもレースごとに GCS へ問い合わせることになるため。
    # 古い日の試行の表示は、取り込んだインスタンスにだけ残る（過去のレースなので、今の表示に戻っても運用に影響しない）
    return p.exists()


def v6c_path(race_id: str, day: str, inf: Path | None = None) -> Path | None:
    """新しい候補の第 1 段の予測（あれば）。"""
    inf = inf or _inf()
    if not _ok_day(day) or v6c_disabled(inf):
        return None
    p = inf / "predictions_v6c" / day / f"{race_id}.parquet"
    return p if _local(p, day) else None


def load_decision(race_id: str, day: str, inf: Path | None = None) -> dict | None:
    inf = inf or _inf()
    if not _ok_day(day) or disabled(inf):
        return None
    p = inf / "bets_stage2" / day / f"{race_id}.json"
    if not _local(p, day):
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("strategy") == SUB else None


def target_payload(d: dict, names: dict[int, str], view: str, cap: int | None) -> dict[str, Any]:
    """判定 → target と同じ形の応答。買うレースは推奨度 1（見送りは 0）。est_multiple は 1（精算の条件『確定オッズ ≥ 想定』で
    買い目を落とさない。凍結の買い方は T-60 の判定で必ず買う）。実オッズ由来の値（T-60 のオッズ・人気・q・r・全頭の一覧）は view="full"（開発者）だけ。"""
    from src.utils import paid_zone as PZ

    full = view == "full"
    buy = d.get("status") == "buy"
    hm = d.get("honmei") or {}
    bets = []
    for i, b in enumerate(d.get("bets") or [] if buy else []):
        combo = [int(x) for x in b.get("combo") or []]
        pw = float(hm.get("p_win") or 0.0)
        item = {
            "bet_type": b.get("bet_type", "tansho"), "bet_label": "単勝", "combo": combo, "combo_label": "-".join(str(x) for x in combo),
            "legs": [{"horse_number": h, "horse_name": names.get(h, ""), "mark": "◎"} for h in combo],
            "yen": int(b.get("yen") or 100), "est_multiple": 1.0, "est_payout_yen": 0,
            # 第 2 段の勝率は T-60 のオッズを offset にした値なので、勝率と買いの線（1/勝率）も開発者だけ（統括役の判断・安全側）
            "hit_rate_pct": round(pw * 100, 1) if full else None, "ev_line_multiple": (round(1.0 / pw, 1) if full and pw > 0 else None), "priority": i + 1,
        }
        if full and hm.get("odds"):
            item["odds"] = float(hm["odds"])          # 確定払戻が出る前の概算払戻に使う（attach_bet_results）
        bets.append(item)
    level = 1 if buy and bets else 0
    keep = ("frozen_hash", "status", "skip_reason", "skip_label", "c_class", "field", "as_of", "model")
    stage2: dict[str, Any] = {k: d.get(k) for k in keep}
    if buy:
        stage2["honmei"] = {k: hm.get(k) for k in (("horse_number", "p_win", "q", "r", "pop", "odds") if full else ("horse_number",))}
    if full:
        stage2.update({"r_max": d.get("r_max"), "q_r_max": d.get("q_r_max"), "odds_t60_at": d.get("odds_t60_at"), "horses": d.get("horses") or []})
    out = {
        "strategy": "target", "sub_strategy": SUB, "trial": True, "view": view, "race_id": d.get("race_id"),
        "status": "bet" if level else "skip", "level": level, "skip_reason": None if level else (d.get("skip_label") or "見送り"),
        "bets": bets, "total_yen": sum(b["yen"] for b in bets), "budget_yen": sum(b["yen"] for b in bets), "target_multiple": 1.0,
        "max_total_bet": cap, "odds_basis": "own", "race_tier": PZ.tier_info(level), "legend": PZ.legend(),
        "disclaimer": PZ.disclaimer(), "evidence": EVIDENCE, "note": NOTICE, "stage2": stage2,
    }
    if level:                                       # 印は ◎ だけ（単勝 1 点の推奨。相手の ○▲ は付けない）
        out["horse_marks"] = {str(bets[0]["combo"][0]): "honmei"}
    return out


SCOPE_GRADES = ("未勝利", "新馬")                 # 凍結した買い方の対象（skip.in.c_class）。class_level と同じく grade の完全一致


def in_scope(card: dict, d: dict | None = None) -> bool:
    """第 2 段の判定を画面に出すレースか（未勝利・新馬だけ）。出馬表の grade を正とし、無ければ判定ファイルの c_class。"""
    g = str((card or {}).get("grade") or "").strip()
    if not g and d:
        g = str(d.get("c_class") or "").strip()
    return g in SCOPE_GRADES


def optimize_payload(race_id: str, view: str, cap: int | None, storage: Any) -> dict[str, Any] | None:
    """第 2 段の判定があるレースだけ応答を返す（無ければ None → 呼び出し側は今の target）。"""
    from src.api.local_predictions import _day_of

    try:
        card = storage.load("race_shutuba", race_id) or {}
    except Exception:  # noqa: BLE001
        return None
    d = load_decision(race_id, _day_of(card))
    if d is None:
        return None
    if not in_scope(card, d):                     # 未勝利・新馬以外は今の target のまま（ユーザーの判断 2026-11-18）
        return None
    names: dict[int, str] = {}
    for e in card.get("entries") or card.get("horses") or []:
        try:
            names[int(e.get("horse_number") or 0)] = str(e.get("horse_name") or "")
        except (TypeError, ValueError):
            continue
    return target_payload(d, names, view, cap)
