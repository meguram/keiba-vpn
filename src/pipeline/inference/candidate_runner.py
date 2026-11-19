"""候補の影のランナー（開催日の常駐）: 新しい候補の束（第 1 段）＋第 2 段＋凍結した買い方の判定を、本番の推論とは別のプロセスで動かす。

本番の推論（raceday-weight-runner、v6e と §77）には一切触らない。新しい候補は 2016 年起点の設定一式
（KEIBA_FIRST_YEAR=2016 など。import の時点で決まる値を含む）が要るので、同じプロセスには入れられない。

流れ（レースごと）:
  1. 本体の weight-runner が推論を終えたら（events/<日>.jsonl の status=ok）、または T-30 を過ぎたら、
     同じレースを新しい候補の束で推論する → predictions_v6c/<日>/<rid>.parquet と、第 2 段の入力 X（手元だけ）。
  2. T-60 の単勝オッズの手元の写し（odds_t60/<日>/<rid>.json、raceday-runner が書く）が届いたら、
     第 2 段 → predictions_stage2/<日>/<rid>.parquet、判定 → bets_stage2/<日>/<rid>.json。
  3. 対象は未勝利・新馬のレースだけ（凍結した買い方の対象。統括役・ユーザーの判断 2026-11-18）。ほかのレースは何もしない（今の表示のまま）。
  4. 規則による見送り（頭数・ずれが小さい・候補なし）は bets_stage2 に書く。システムの原因（T-60 の写しが無い・推論の失敗・上限・前段の
     保存値が無い・オッズの崩れ）のときは**ファイルを書かない**（API は今の target のまま）。見張りの記録 launch_watch に system_skip を残す。
  発走を過ぎたレース・判定済みのレースは書かない（再起動しても書き直さない）。どのファイルも 1 レース 1 回だけ書く。

出力は 3 つの木（publish_dir の下）。ホストの gcs-mirror-push が 5 分おきに上げ、API はファイルがあれば新しい表示にする。
止めの目印 ``bets_stage2/_disabled.json`` があると、API は今の表示（v6e・target）に戻る（ランナーは書き続けてよい）。

設定は ``config/candidate_runtime.json``（環境変数の一式・出力先・期限・メモリと GCS の上限）。
起動: ``python -m src.pipeline.inference.candidate_runner [--day YYYYMMDD] [--once]``（過去の日のリハーサルは src/scripts/data/candidate_replay.py）。
止めの目印: ``python -m src.pipeline.inference.candidate_runner --set-disabled on|off [--v6c-only] --day YYYYMMDD``（目印と、その日の判定ファイルの
書き直し＝push で画面のキャッシュの目印も立つ。ミラーは削除を伝えないので、戻すときも off で上書きする）。
既定はオフ: 設定の enabled が true か、環境変数 KEIBA_CANDIDATE_RUNNER=1 か、目印 data/local/meta/candidate_runner.on があるときだけ動く
（追跡されるファイルを書き換えずに有効にする。共有の作業ツリーの毎時の pull を止めないため）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config/candidate_runtime.json"
_JST = timezone(timedelta(hours=9))
logger = logging.getLogger("candidate_runner")

TREES = ("predictions_v6c", "predictions_stage2", "bets_stage2")
DISABLED_MARK = "_disabled.json"


# ── 設定 ─────────────────────────────────────────────────────────────────


def load_config(path: Path | str | None = None) -> dict:
    p = Path(path or os.environ.get("KEIBA_CANDIDATE_CONFIG") or CONFIG_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


ENABLE_MARK = ROOT / "data/local/meta/candidate_runner.on"
SCOPE_GRADES = ("未勝利", "新馬")
SYSTEM_REASONS = ("no_prediction", "no_odds_t60", "stage2_error", "odds_broken", "deadline", "no_stage_inputs")


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("enabled")) or os.environ.get("KEIBA_CANDIDATE_RUNNER", "").strip() == "1" or ENABLE_MARK.exists()


def apply_env(cfg: dict) -> dict[str, str]:
    """設定の環境変数を、このプロセスにだけ当てる（.env より優先）。src.pipeline を import する前に呼ぶこと。"""
    env = {k: str(v) for k, v in (cfg.get("env") or {}).items()}
    os.environ.update(env)
    return env


# ── 判断（純粋関数） ───────────────────────────────────────────────────────


def plan_actions(now: datetime, races: list[dict], st: dict[str, dict], *, deadline_min: int, predict_after_min: int) -> dict[str, list[str]]:
    """いま何をするか。races: [{race_id, post_time}]。st: race_id → {main_done, predicted, odds, decided, failed}。

    返り値: predict（第 1 段を流す）・stage2（第 2 段と判定）・expire（期限の見送りを書く）の race_id の一覧。
    - 発走を過ぎたレースは何もしない（判定が無ければ、その旨はログだけ）。
    - predict: 本体の推論が済んだ（main_done）か、発走の predict_after_min 分前を過ぎた。未推論・未失敗のものだけ。
    - stage2: 推論済み・オッズあり・未判定・期限（発走の deadline_min 分前）より前。
    - expire: 未判定で、期限を過ぎた（システムの原因の見送り＝ファイルを書かない）。
    """
    out: dict[str, list[str]] = {"predict": [], "stage2": [], "expire": []}
    for r in races:
        rid, post = r["race_id"], r.get("post_time")
        if post is None:
            continue
        s = st.get(rid) or {}
        if s.get("decided") or now >= post:
            continue
        if not s.get("predicted") and not s.get("failed") and (s.get("main_done") or now >= post - timedelta(minutes=predict_after_min)):
            out["predict"].append(rid)
        late = now >= post - timedelta(minutes=deadline_min)
        if s.get("predicted") and s.get("odds") and not late:      # 期限（T-20）を過ぎてから判定しない（push と目録の遅れで発走後にしか見えない）
            out["stage2"].append(rid)
        elif late:
            out["expire"].append(rid)
    return out


def expire_reason(s: dict) -> str:
    if s.get("failed") or not s.get("predicted"):
        return "no_prediction"
    if not s.get("odds"):
        return "no_odds_t60"
    return "deadline"


def runners_and_odds(X_hn, odds_entries: list[dict]) -> tuple[list[int], dict[int, float]]:
    """X の馬番のうち、T-60 の単勝オッズ（> 0）がある馬を出走馬とする（取消・除外はオッズが付かない）。"""
    od = {}
    for e in odds_entries or []:
        try:
            hn, o = int(e.get("horse_number") or 0), float(e.get("win_odds") or 0)
        except (TypeError, ValueError):
            continue
        if hn > 0 and o > 0:
            od[hn] = o
    return [int(h) for h in X_hn if int(h) in od], od


# ── 書き出し（1 回だけ・一時ファイル → rename） ─────────────────────────────


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_parquet(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _append_watch(watch_dir: Path, day: str, rec: dict) -> None:
    """見張りの記録（ホストの手元だけ。GCS には上げない）。"""
    try:
        watch_dir.mkdir(parents=True, exist_ok=True)
        with (watch_dir / f"{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(_JST).isoformat(timespec="seconds"), **rec}, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("見張りの記録を書けない: %s", exc)


# ── 本体の events ─────────────────────────────────────────────────────────


# 推論の入力の取り込みの欠けの目印（群ごとに、列ごとの欠けの割合の平均）。産地（attr_）は欠けを 99 で表すので数えない
GAP_GROUPS = {
    "na_oikiri_eval": {"prefix": ("oikiri_",)},                  # race_oikiri（当日の追い切りの評価）。推論の時点で欠けやすい
    "na_tl": {"prefix": ("tl_",)},                               # 馬別の調教ページ
    "na_wc": {"prefix": ("wc_",)},
    "na_weight": {"exact": ("weight", "weight_change"), "prefix": ("d_wt_",)},   # 馬体重（発表前・取り込み漏れ）
}


def input_gaps(X) -> dict:
    """推論の入力の取り込みの欠け（群ごとに、列ごとの欠けの割合の平均）。学習との差が大きくなる日の目印として、判定の前に記録する。"""
    out = {}
    try:
        names = [c for c in X.columns if isinstance(c, str)]
        for k, g in GAP_GROUPS.items():
            cols = [c for c in names if c in g.get("exact", ()) or c.startswith(g.get("prefix", ()) or ("\0",))]
            if cols:
                out[k] = round(float(X[cols].isna().mean().mean()), 3)
    except Exception as exc:  # noqa: BLE001  記録だけ。推論は止めない
        logger.debug("入力の欠けを数えられない: %s", exc)
        out["gap_error"] = str(exc)[:80]
    return out


def main_done_ids(events_path: Path) -> set[str]:
    """本体の weight-runner の推論が済んだレース（status=ok）。"""
    out: set[str] = set()
    try:
        for ln in events_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("status") == "ok" and e.get("race_id"):
                out.add(str(e["race_id"]))
    except OSError:
        pass
    return out


# ── 守り（メモリ・GCS） ─────────────────────────────────────────────────────


def protect_process(cfg: dict) -> None:
    """メモリが足りなくなったとき、本体より先にこのプロセスが落ちるようにする（oom_score_adj を上げる。権限は要らない）。"""
    try:
        Path("/proc/self/oom_score_adj").write_text(str(int(cfg.get("oom_score_adj", 1000))))
    except OSError as exc:
        logger.warning("oom_score_adj を書けない: %s", exc)


def rss_gb() -> float:
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def start_rss_watchdog(hard_gb: float, on_kill=None, every_sec: float = 5.0) -> None:
    """RSS が hard_gb を超えたら、すぐにこのプロセスを終わらせる（本体の推論を巻き込む前に。systemd-run はこのホストで使えない）。"""
    if not hard_gb:
        return
    import threading

    def _run():
        while True:
            if rss_gb() > hard_gb:
                logger.critical("RSS %.1fGB が固い上限 %.1fGB を超えた。終わる（本体には影響なし。以後のレースは今の表示）", rss_gb(), hard_gb)
                try:
                    if on_kill:
                        on_kill()
                finally:
                    os._exit(9)
            time.sleep(every_sec)

    threading.Thread(target=_run, name="rss-watchdog", daemon=True).start()


class StaleInputs(RuntimeError):
    """材料が古い・足りない（止める。ファイルを書かないので、画面は今の表示のまま）。"""


class GcsMeter:
    """GCS の読み出し（Blob.download_as_bytes・download_to_filename・一覧）を数え、1 日の上限を超えたら以後の推論を止める（GCP の外から GCS を読む量に上限）。"""

    def __init__(self, cap_mb: float):
        self.cap = cap_mb * 1e6
        self.reads = 0
        self.bytes = 0
        self.lists = 0
        from collections import Counter

        self.by_kind: Counter = Counter()                  # 読み出しの種類（置き場の名前の数字を N に伏せた形）ごとの回数。見積もりと原因探しの材料
        try:
            from google.cloud.storage import blob as _b

            orig = _b.Blob.download_as_bytes
            meter = self

            def _bytes(blob_self, *a, **k):
                s = orig(blob_self, *a, **k)
                meter.reads += 1
                meter.note_kind(blob_self)
                meter.bytes += len(s) if s else 0
                return s

            _b.Blob.download_as_bytes = _bytes
            orig_f = _b.Blob.download_to_filename

            def _file(blob_self, filename, *a, **k):
                out = orig_f(blob_self, filename, *a, **k)
                meter.reads += 1
                meter.note_kind(blob_self)
                try:
                    meter.bytes += os.path.getsize(filename)
                except OSError:
                    pass
                return out

            _b.Blob.download_to_filename = _file
            from google.cloud.storage import client as _c

            orig_l = _c.Client.list_blobs

            def _list(client_self, *a, **k):
                meter.lists += 1
                return orig_l(client_self, *a, **k)

            _c.Client.list_blobs = _list
        except Exception as exc:  # noqa: BLE001
            logger.warning("GCS の読み出しを数えられない: %s", exc)

    def note_kind(self, blob) -> None:
        self.by_kind[re.sub(r"\d+", "N", str(getattr(blob, "name", "") or ""))] += 1     # 置き場の名前の数字（日付・id）を伏せた形

    def summary(self) -> dict:
        top = dict(self.by_kind.most_common(12))
        rest = sum(self.by_kind.values()) - sum(top.values())
        return {"gcs_mb": round(self.bytes / 1e6, 2), "gcs_reads": self.reads, "gcs_lists": self.lists,
                "gcs_by_kind": {**top, **({"(その他)": rest} if rest else {})}}

    @property
    def over(self) -> bool:
        return self.bytes > self.cap


def preflight(cfg: dict, day: str) -> list[str]:
    """warm の前に、影の作業場所に要るものがそろっているか（無いと黙って空の列になる・全レースで失敗する）。足りないものの一覧を返す。"""
    env = cfg.get("env") or {}
    inf = Path(env.get("KEIBA_INFERENCE_DIR") or "")
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    need = {
        "PX の状態（state/px_v5/svd.npz）": inf / "state/px_v5/svd.npz",
        "前段の保存値（tracking_v2/<日>）": inf / "tracking_v2" / iso,
        "ラップ予想の保存値（lap_forecast/<日>）": inf / "lap_forecast" / iso,
        "第 2 段のモデル": Path(env.get("KEIBA_STAGE2_MODEL_DIR") or "") / "meta.json",
        "束": Path(env.get("KEIBA_V6_BUNDLE") or "") / "bundle.json",
        "RB の状態": Path(env.get("KEIBA_V6_RB_STATE") or ""),
    }
    rd = cfg.get("redirect_2016") or {}
    if rd.get("layer_a"):
        need["2016 年起点の layer_a"] = Path(rd["layer_a"])
    return [f"{k}: {v}" for k, v in need.items() if not v.exists()]


# ── 本体 ──────────────────────────────────────────────────────────────────


class Runner:
    def __init__(self, cfg: dict, day: str, *, replay: bool = False):
        import pandas as pd  # noqa: F401  （早めに失敗させる）

        self.cfg, self.day, self.replay = cfg, day, replay
        self.iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        self.publish = Path(cfg["publish_dir"])
        self.main_inf = Path(cfg["main_inference_dir"])
        self.watch_dir = Path(cfg.get("watch_dir") or self.main_inf / "launch_watch")
        self.st: dict[str, dict] = {}
        self.X: dict[str, Any] = {}
        self.card: dict[str, Any] = {}
        self.meter = GcsMeter(float(cfg.get("gcs_cap_mb_per_day", 50)))
        self.main_done_fn = None            # (now) -> set[race_id]。リハーサルでは時刻で絞る
        self.odds_fn = None                 # (race_id, now) -> dict | None。リハーサルでは手元の pre_odds から
        self.sleep_fn = time.sleep
        self.now_fn = lambda: datetime.now(_JST)
        self.post: dict[str, datetime] = {}
        self.degraded = False               # メモリの上限を超えた後は推論しない（期限の見送りだけ書き続ける）

    # 起動時の準備（重い）
    def warm(self) -> None:
        from src.pipeline.inference import stage2_bets as sb
        from src.pipeline.inference import weight_triggered_predict as wtp
        from src.pipeline.inference.stage2_market import Stage2Model
        from src.pipeline.inference.v3_serving import V3Serving, v6_paths
        from src.pipeline.models import training_years as ty
        from src.scraper.storage import HybridStorage

        want_fy = int(self.cfg["env"].get("KEIBA_FIRST_YEAR", ty.FIRST_YEAR))
        if ty.FIRST_YEAR != want_fy:
            raise SystemExit(f"KEIBA_FIRST_YEAR が効いていない（{ty.FIRST_YEAR} ≠ {want_fy}）。設定の当て方を確かめる")
        t0 = time.time()
        self.storage = getattr(self, "storage", None) or HybridStorage()
        self.frozen = sb.load_frozen()
        self.s2 = Stage2Model()
        rd = self.cfg.get("redirect_2016") or {}
        if rd.get("enabled", True) and rd.get("rebuild_base"):
            # 血統・馬主・産地の表と血統の集計のキャッシュを 2016 年起点の土台に向ける（本番の表は 2020〜 の馬だけで、学習とずれる。
            # v6_parity --rebuild-2016 と同じ。REBUILD_BASE は rebuild_groups_2016 の import の時点で読まれる）
            os.environ["REBUILD_BASE"] = str(rd["rebuild_base"])
            from src.scripts.data import rebuild_groups_2016 as rg

            rg.redirect_serving()
            logger.info("2016 年起点の表に付け替えた: REBUILD_BASE=%s", rg.BASE)
        layer_a = rd.get("layer_a") or None
        self.ctx = wtp.InferenceContext(layer_a=Path(layer_a) if layer_a else None)
        self.ctx.warm()
        p = v6_paths()
        if getattr(self.ctx, "v6", None) is not None and Path(str(self.ctx.v6.bundle_dir)) == Path(str(p["bundle"])):
            self.v6, info = self.ctx.v6, "ctx.v6 を使い回す"          # InferenceContext.warm が同じ束で作った v6（二重に warm しない）
        else:
            self.v6 = V3Serving(p["bundle"], p["stage"], tl_reference=p["tl_reference"])
            self.v6.rb_state = p["rb_state"]
            info = self.v6.warm()
        self.raw_full = self.ctx.raw_hist
        self.hist_full = self.v6.history.copy() if getattr(self.v6, "history", None) is not None else None
        self.storage_writes = 0
        _orig_save = self.storage.save

        def _no_save(category, key, data, *a, **k):     # 読み出し専用（影のランナーは GCS に書かない。§77 も作らない）
            self.storage_writes += 1
            logger.error("影のランナーが保存しようとした（書かない）: %s/%s", category, key)
            return False

        self.storage.save = _no_save                      # type: ignore[method-assign]
        self._orig_save = _orig_save
        missing = [c for c in self.s2.feature_cols if c not in self.v6.feature_cols]
        if missing:
            raise SystemExit(f"束の列と第 2 段の列が合わない: {len(missing)} 列 {missing[:8]}")
        self.bundle_name = Path(p["bundle"]).name
        stale = self.freshness(p)
        if stale and not self.replay:
            raise StaleInputs(stale)
        logger.info("warm 済み %.0f 秒: FIRST_YEAR=%s 束=%s stage=%s rb=%s tl_ref=%s 第2段=%s 凍結=%s RSS=%.1fGB info=%s",
                    time.time() - t0, ty.FIRST_YEAR, p["bundle"], p["stage"], p["rb_state"], p["tl_reference"], self.s2.path,
                    self.frozen["hash"], rss_gb(), info)
        _append_watch(self.watch_dir, self.day, {"event": "warm", "sec": round(time.time() - t0), "first_year": ty.FIRST_YEAR,
                                                  "bundle": str(p["bundle"]), "rb_state": str(p["rb_state"]), "rss_gb": round(rss_gb(), 1)})

    def freshness(self, p: dict) -> list[str]:
        """材料の鮮度: RB の状態の cut が、履歴の表の最終日（その日より前の最後の開催日）以降か。古いと日曜に土曜の結果が入らない。"""
        out = []
        try:
            cut = json.loads((Path(p["rb_state"]) / "params.json").read_text(encoding="utf-8")).get("cut")
        except (OSError, ValueError):
            cut = None
        last = str(getattr(self.ctx, "history_last_date", "") or "")[:10]
        if not cut or (last and str(cut) < last):
            out.append(f"RB の状態が古い（cut={cut}・履歴の最終日={last}）")
        try:   # cut は結果の表で決まるので、めぐ指数の遅れはここで見る（止めない。本体の v6 も同じ表を読む。警告と記録だけ）
            import pandas as pd

            h = pd.read_parquet(Path(p["rb_state"]) / "hist.parquet", columns=["date", "rb_mg_x"])
            mg_last = str(pd.to_datetime(h.loc[h["rb_mg_x"].notna(), "date"]).max().date())
            if cut and mg_last < str(cut)[:10]:
                logger.warning("RB の状態にめぐ指数の最近の走が無い（めぐ指数のある最終日=%s・cut=%s）", mg_last, cut)
            _append_watch(self.watch_dir, self.day, {"event": "rb_state", "cut": cut, "megu_last": mg_last, "history_last": last})
        except Exception as exc:  # noqa: BLE001
            logger.warning("RB の状態のめぐ指数の最終日を読めない: %s", exc)
        return out

    def stage_inputs_ok(self, rid: str) -> bool:
        """前段 S3 の保存値（tracking_v2/<日>/<rid>.json）があるか。無いと展開・位置取りの列が黙って空になる。"""
        inf = Path((self.cfg.get("env") or {}).get("KEIBA_INFERENCE_DIR") or "")
        return (inf / "tracking_v2" / self.iso / f"{rid}.json").exists()

    def schedule(self, *, refresh: bool = False) -> list[dict]:
        """その日の発走時刻表。1 回作ったら使い回す（race_day_schedule が無い日は、全レースの出馬表を読んで作るため重い）。
        refresh=True は、時刻表の文書（race_day_schedule・1 件）がある日だけ読み直す。無い日は前のまま（出馬表を読み直さない）。"""
        if not refresh and getattr(self, "_sched", None) is not None:
            return [dict(r) for r in self._sched]                 # 写しを返す（呼び出し側が post_time を書き換えても、使い回しの表は変えない）
        if refresh:
            snap = self.storage.load("race_day_schedule", self.day)
            if not (isinstance(snap, dict) and snap.get("slots")):
                if not getattr(self, "_refresh_skip_noted", False):
                    self._refresh_skip_noted = True
                    logger.warning("%s: race_day_schedule が無いので、発走時刻の読み直しをしない（変更は拾えない）", self.day)
                    _append_watch(self.watch_dir, self.day, {"event": "schedule_refresh_skipped"})
                return [dict(r) for r in (self._sched or [])]
            from src.scraper.race_day_schedule import schedule_payload_to_runtime_list

            rows = schedule_payload_to_runtime_list(snap)        # 読んだ文書をそのまま使う（2 回読まない）
        else:
            from src.scraper.auto_scrape_queue import _fetch_race_schedule_storage

            rows = _fetch_race_schedule_storage(self.storage, self.day)
        out = []
        for r in rows:
            pt = r.get("post_time")
            if pt is not None and pt.tzinfo is None:
                pt = pt.replace(tzinfo=_JST)
            out.append({**r, "post_time": pt})
        self._sched = out
        return [dict(r) for r in out]

    def _out(self, tree: str, rid: str, ext: str) -> Path:
        return self.publish / tree / self.day / f"{rid}.{ext}"

    def restore_state(self, races: list[dict]) -> None:
        """再起動のとき: 判定済みのレースは飛ばす。推論済み（predictions_v6c がある）なら X を読み直す。"""
        xdir = Path(self.cfg["x_dir"]) / self.day
        for r in races:
            rid = r["race_id"]
            s = self.st.setdefault(rid, {})
            if self._out("bets_stage2", rid, "json").exists():
                s["decided"] = True
            xp = xdir / f"{rid}.parquet"
            if self._out("predictions_v6c", rid, "parquet").exists() and xp.exists():
                import pandas as pd

                self.X[rid] = pd.read_parquet(xp)
                s["predicted"] = True

    def predict(self, races_by_id: dict[str, dict], rids: list[str]) -> None:
        import polars as pl

        from src.pipeline.inference import weight_triggered_predict as wtp

        for rid in rids:
            s = self.st.setdefault(rid, {})
            t0 = time.time()
            try:
                if self.degraded:
                    raise RuntimeError("メモリの上限を超えた後なので推論しない")
                if not self.replay and not self.stage_inputs_ok(rid):
                    s["no_stage_inputs"] = True
                    raise RuntimeError("前段 S3 の保存値（tracking_v2）が無い")
                if self.meter.over:
                    raise RuntimeError(f"GCS の読み出しが 1 日の上限を超えた（{self.meter.bytes / 1e6:.0f}MB）")
                card = wtp.fill_card_meta(wtp.card_frame(self.storage, [rid]), [races_by_id[rid]])
                card = card.filter(pl.col("horse_number").fill_null(0) > 0)
                if card.height == 0:
                    raise RuntimeError("馬番のある出馬表が無い")
                kw = {}
                if self.replay:              # 過去の日の当て直し: その日（その R の発走）より後の結果を履歴に入れない（backfill_predictions と同じ）
                    import datetime as _dt

                    cut = _dt.date.fromisoformat(self.iso) if self.raw_full.schema["date"] == pl.Date else self.iso
                    self.ctx.raw_hist = self.raw_full.filter(pl.col("date") < cut)
                    if self.hist_full is not None:
                        h, st_ = self.hist_full, str(card["start_time"][0])
                        kw["history"] = h[(h["date"].astype(str) < self.iso) | ((h["date"].astype(str) == self.iso) & (h["start_time"].astype(str) < st_)
                                                                             & (h["race_id"].astype(str) != rid))]
                F, _ = self.ctx.build(card)
                X = self.v6.features(card, F, self.storage, **kw)
                P = self.v6.predict(X)
                _write_parquet(Path(self.cfg["x_dir"]) / self.day / f"{rid}.parquet", X)      # X を先に（v6c だけ残って X が無い状態を作らない）
                if not self._out("predictions_v6c", rid, "parquet").exists():      # 1 回だけ
                    _write_parquet(self._out("predictions_v6c", rid, "parquet"), P)
                self.X[rid] = X
                self.card[rid] = card
                s["predicted"] = True
                rec = {"event": "v6c", "race_id": rid, "sec": round(time.time() - t0, 1), "rows": int(len(X)), "rss_gb": round(rss_gb(), 1),
                       "gcs_mb": round(self.meter.bytes / 1e6, 1), **input_gaps(X)}
                logger.info("第 1 段（候補）: %s", rec)
                _append_watch(self.watch_dir, self.day, rec)
            except Exception as exc:  # noqa: BLE001
                s["failed"] = True
                logger.exception("第 1 段（候補）に失敗: %s", rid)
                _append_watch(self.watch_dir, self.day, {"event": "v6c_error", "race_id": rid, "error": str(exc)[:300]})

    def _grade(self, rid: str) -> str:
        c = self.card.get(rid)
        if c is not None and "grade" in c.columns:
            v = c["grade"].drop_nulls()
            if len(v):
                return str(v[0])
        try:
            sh = self.storage.load("race_shutuba", rid) or {}
            return str(sh.get("grade") or (sh.get("race_info") or {}).get("grade") or "")
        except Exception:  # noqa: BLE001
            return ""

    def stage2(self, rid: str, odds_doc: dict) -> None:
        import numpy as np

        from src.pipeline.inference import stage2_bets as sb
        from src.pipeline.inference.stage2_market import OddsBroken

        s = self.st[rid]
        t0 = time.time()
        X = self.X[rid]
        hn_all = X["horse_number"].astype(int).to_numpy()
        runners, od = runners_and_odds(hn_all, odds_doc.get("entries") or [])
        if str(odds_doc.get("phase") or "") != "t60":      # 凍結の買い方は T-60 の offset の版だけ（確定・直前のオッズで判定しない）
            return self._system_skip(rid, "no_odds_t60", f"オッズの phase が t60 でない: {odds_doc.get('phase')}")
        base = {"odds_t60_at": odds_doc.get("snapshot_at"), "odds_phase": "t60", "model": f"{self.bundle_name}+{self.s2.name}"}
        try:
            Xr = X[X["horse_number"].astype(int).isin(runners)].reset_index(drop=True)
            odds = np.array([od[int(h)] for h in Xr["horse_number"].astype(int)], dtype=float)
            P = self.s2.predict_race(Xr, odds)
            P.insert(0, "horse_number", Xr["horse_number"].astype(int).to_numpy())
            P.insert(0, "horse_id", Xr["horse_id"].astype(str).to_numpy())
            P.insert(0, "race_id", rid)
            P["pop_t60"] = sb.popularity_from_q(P["q"].to_numpy())
            P["odds_t60_at"] = odds_doc.get("snapshot_at")
            P["odds_phase"] = "t60"                         # q・offset・人気はすべてこの T-60 のオッズから（stage2_market.predict_race）
            P["model"] = self.s2.name
            if not self._out("predictions_stage2", rid, "parquet").exists():      # 1 回だけ
                _write_parquet(self._out("predictions_stage2", rid, "parquet"), P)
            d = sb.decide(race_id=rid, horse_numbers=P["horse_number"], p_win=P["p_win"], p_top2=P["p_top2"], p_top3=P["p_top3"],
                          q=P["q"], r=P["r"], pop=P["pop_t60"], odds=P["odds"], grade=self._grade(rid), frozen=self.frozen)
        except OddsBroken as exc:
            return self._system_skip(rid, "odds_broken", str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("第 2 段・判定に失敗: %s", rid)
            return self._system_skip(rid, "stage2_error", str(exc)[:300])
        self._decide(rid, {**base, **d}, t0)

    def _safe(self, fn, rid: str, *a) -> None:
        """1 レースの失敗でその日の残りを止めない（失敗したレースはシステムの原因の見送り＝今の表示）。"""
        try:
            fn(rid, *a)
        except Exception as exc:  # noqa: BLE001
            logger.exception("レースの処理に失敗: %s", rid)
            self._system_skip(rid, "stage2_error", str(exc))

    def _system_skip(self, rid: str, reason: str, detail: str = "") -> None:
        """システムの原因の見送り: 判定ファイルを書かない（API は今の target のまま）。見張りの記録だけ残す。"""
        st = self.st.setdefault(rid, {})
        if st.get("decided"):
            return
        st["decided"] = True
        rec = {"event": "system_skip", "race_id": rid, "reason": reason, "detail": detail[:300]}
        logger.warning("システムの原因で判定を出さない（今の表示のまま）: %s", rec)
        _append_watch(self.watch_dir, self.day, rec)

    def expire(self, rid: str) -> None:
        st = self.st.get(rid) or {}
        self._system_skip(rid, "no_stage_inputs" if st.get("no_stage_inputs") else expire_reason(st))

    def _decide(self, rid: str, d: dict, t0: float) -> None:
        d = {**d, "as_of": datetime.now(_JST).isoformat(timespec="seconds"), "day": self.day, "trial": True}
        p = self._out("bets_stage2", rid, "json")
        if p.exists():                       # 1 回だけ（再起動・二重起動でも書き直さない）
            self.st[rid]["decided"] = True
            return
        post = self.post.get(rid)
        if post is not None and self.now_fn() >= post:      # 発走後は書かない（推論が長引いたとき）
            self.st[rid]["decided"] = True
            logger.warning("発走後なので判定を書かない: %s（%s）", rid, d.get("status"))
            _append_watch(self.watch_dir, self.day, {"event": "late_no_write", "race_id": rid, "status": d.get("status"), "skip_reason": d.get("skip_reason")})
            return
        _write_json(p, d)
        self.st[rid]["decided"] = True
        rec = {"event": "decision", "race_id": rid, "status": d.get("status"), "skip_reason": d.get("skip_reason"),
               "honmei": (d.get("honmei") or {}).get("horse_number"), "r_max": d.get("r_max"), "sec": round(time.time() - t0, 2)}
        logger.info("判定: %s", rec)
        _append_watch(self.watch_dir, self.day, rec)

    def loop(self, *, once: bool = False, now_fn=None) -> None:
        now_fn = now_fn or (lambda: datetime.now(_JST))
        self.now_fn = now_fn
        races = sorted([r for r in self.schedule() if r.get("post_time")], key=lambda r: (r["post_time"], r["race_id"]))
        if not races:
            logger.info("%s: 発走時刻表が無い（開催なし）", self.day)
            return
        allr = len(races)
        races = [r for r in races if self._grade(r["race_id"]) in SCOPE_GRADES]      # 未勝利・新馬だけ（ほかは今の表示のまま）
        logger.info("%s: 対象 %d / %d レース（未勝利・新馬）", self.day, len(races), allr)
        _append_watch(self.watch_dir, self.day, {"event": "scope", "races": len(races), "all": allr})
        if not races:
            return
        by_id = {r["race_id"]: r for r in races}
        self.restore_state(races)
        from src.pipeline.inference import odds_t60_copy as oc

        odds_root = Path(self.cfg.get("odds_t60_dir") or self.main_inf / "odds_t60")
        ev_path = self.main_inf / "events" / f"{self.day}.jsonl"
        poll = float(self.cfg.get("poll_sec", 30))
        rss_cap = float(self.cfg.get("rss_cap_gb", 0) or 0)
        main_fn = self.main_done_fn or (lambda _now: main_done_ids(ev_path))
        odds_fn = self.odds_fn or (lambda rid, _now: oc.read_copy(self.day, rid, root=odds_root))
        self.post = {r["race_id"]: r["post_time"] for r in races}
        dl, pa = int(self.cfg.get("deadline_min", 20)), int(self.cfg.get("predict_after_min", 30))
        refresh_sec = float(self.cfg.get("schedule_refresh_sec", 600))
        last_refresh = time.monotonic()
        while True:
            if not self.replay and time.monotonic() - last_refresh >= refresh_sec:   # 発走の繰り下げなどを拾う（時刻表 1 件の読み出し）
                last_refresh = time.monotonic()
                try:
                    for r in self.schedule(refresh=True):
                        if r.get("race_id") in by_id and r.get("post_time") and r["post_time"] != by_id[r["race_id"]]["post_time"]:
                            logger.info("発走時刻の変更: %s %s → %s", r["race_id"], by_id[r["race_id"]]["post_time"], r["post_time"])
                            by_id[r["race_id"]]["post_time"] = r["post_time"]
                            self.post[r["race_id"]] = r["post_time"]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("時刻表の読み直しに失敗（前のまま続ける）: %s", exc)
            now = now_fn()
            main_ok = main_fn(now)
            for rid in by_id:
                s = self.st.setdefault(rid, {})
                s["main_done"] = rid in main_ok
                if not s.get("odds"):
                    doc = odds_fn(rid, now)
                    if doc:
                        s["odds"] = doc
                        _append_watch(self.watch_dir, self.day, {"event": "odds_t60_seen", "race_id": rid, "snapshot_at": doc.get("snapshot_at"),
                                                                  "sim_now": now.isoformat(timespec="seconds")})
            for rid in plan_actions(now, races, self.st, deadline_min=dl, predict_after_min=pa)["stage2"]:   # 軽い第 2 段を先に
                self._safe(self.stage2, rid, self.st[rid]["odds"])
            todo = plan_actions(now, races, self.st, deadline_min=dl, predict_after_min=pa)["predict"]
            self.predict(by_id, todo[:1])                  # 重い推論は発走の早い 1 レースだけ。済んだらすぐ次の周へ（第 2 段を待たせない）
            acts = plan_actions(now_fn(), races, self.st, deadline_min=dl, predict_after_min=pa)   # 推論の後の時刻で決め直す
            for rid in acts["stage2"]:
                self._safe(self.stage2, rid, self.st[rid]["odds"])
            for rid in acts["expire"]:
                self._safe(self.expire, rid)
            if rss_cap and not self.degraded and rss_gb() > rss_cap:
                self.degraded = True
                logger.error("メモリが上限を超えた（%.1fGB > %.1fGB）。以後は推論せず、期限の見送りだけ書く（本体には影響なし）", rss_gb(), rss_cap)
                _append_watch(self.watch_dir, self.day, {"event": "degraded_rss", "rss_gb": round(rss_gb(), 1)})
            pending = [r for r in races if not self.st[r["race_id"]].get("decided") and now < r["post_time"]]
            if once or not pending:
                logger.info("%s: 終わり（未判定 %d レース・保存の試み %d 件・GCS 読み出し %.1fMB/%d 回・一覧 %d 回）", self.day, len(pending),
                            getattr(self, "storage_writes", 0), self.meter.bytes / 1e6, self.meter.reads, self.meter.lists)
                _append_watch(self.watch_dir, self.day, {"event": "end", "pending": len(pending), "storage_writes": getattr(self, "storage_writes", 0),
                                                          **self.meter.summary()})
                return
            if len(todo) <= 1:                             # まだ推論を待つレースがあれば眠らない
                self.sleep_fn(poll)


def set_disabled(cfg: dict, on: bool, day: str, *, v6c_only: bool = False, dev_only: bool = False) -> dict:
    """止めの目印を書き（on: {"disabled": true}・off: {"disabled": false}。ミラーは削除を伝えないので消さない）、その日の判定ファイルに
    toggled_at を足して書き直す（push で中身が変わる → レースごとに画面のキャッシュの目印が立ち、prerender・VPS のキャッシュが取り直される）。"""
    pub = Path(cfg["publish_dir"])
    tree = "predictions_v6c" if v6c_only else "bets_stage2"
    mark = "_dev_only.json" if dev_only else DISABLED_MARK
    now = datetime.now(_JST).isoformat(timespec="seconds")
    _write_json(pub / tree / mark, {"disabled": bool(on), "at": now, "scope": "dev_only" if dev_only else ("v6c" if v6c_only else "all")})
    n = 0
    for p in sorted((pub / "bets_stage2" / day).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["toggled_at"] = now
            _write_json(p, d)
            n += 1
        except (OSError, ValueError) as exc:
            logger.warning("判定ファイルを書き直せない: %s %s", p, exc)
    _append_watch(Path(cfg.get("watch_dir") or pub / "launch_watch"), day, {"event": "set_disabled", "on": bool(on), "scope": tree, "touched": n})
    return {"mark": str(pub / tree / mark), "disabled": bool(on), "touched": n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", default=datetime.now(_JST).strftime("%Y%m%d"))
    ap.add_argument("--config", default="")
    ap.add_argument("--once", action="store_true", help="1 周だけ回して終わる（確認用）")
    ap.add_argument("--set-disabled", choices=["on", "off"], help="止めの目印を書いて終わる（推論はしない）")
    ap.add_argument("--v6c-only", action="store_true", help="--set-disabled を予測の表示（predictions_v6c）だけに")
    ap.add_argument("--dev-only", action="store_true", help="--set-disabled on で、第 2 段の推奨を開発者（view=full）だけに出す（会員には今の target）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(a.config or None)
    if a.set_disabled:
        print(json.dumps(set_disabled(cfg, a.set_disabled == "on", a.day, v6c_only=a.v6c_only, dev_only=a.dev_only), ensure_ascii=False))
        return 0
    if not enabled(cfg):
        logger.info("候補の影のランナーは無効（config の enabled か KEIBA_CANDIDATE_RUNNER=1 で有効）")
        return 0
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    apply_env(cfg)                           # .env より後に当てる（FIRST_YEAR などは import の前に決まっている必要がある）
    if "src.pipeline.models.training_years" in sys.modules:
        raise SystemExit("training_years が先に import された（KEIBA_FIRST_YEAR が効かない）")
    protect_process(cfg)
    r = Runner(cfg, a.day)
    from src.scraper.storage import HybridStorage

    r.storage = HybridStorage()
    if not [x for x in r.schedule() if x.get("post_time")]:          # 開催の無い日は重い warm をせずに終わる
        logger.info("%s: 発走時刻表が無い（開催なし）。終わる", a.day)
        return 0
    miss = preflight(cfg, a.day)
    if miss:
        for m in miss:
            logger.error("影の作業場所に足りないもの: %s", m)
        _append_watch(Path(cfg.get("watch_dir") or Path(cfg["main_inference_dir"]) / "launch_watch"), a.day, {"event": "preflight_missing", "missing": miss})
        if any(m.startswith(("PX", "第 2 段", "束", "RB")) for m in miss):
            return 3                          # これらが無いと全レースで失敗する。期限の見送りも書かない（API は今の表示のまま）
    start_rss_watchdog(float(cfg.get("rss_hard_gb", 0) or 0),
                       on_kill=lambda: _append_watch(Path(cfg.get("watch_dir") or Path(cfg["main_inference_dir"]) / "launch_watch"), a.day,
                                                     {"event": "killed_rss", "rss_gb": round(rss_gb(), 1), **r.meter.summary()}))
    try:
        r.warm()
    except StaleInputs as exc:
        logger.error("材料が古い・足りないので止める（画面は今の表示のまま）: %s", exc)
        _append_watch(r.watch_dir, a.day, {"event": "stale_stop", "detail": str(exc), **r.meter.summary()})
        return 4
    try:
        r.loop(once=a.once)
    except BaseException as exc:                     # 例外・中断で終わったときも、GCS の読み出しの内訳を残す
        _append_watch(r.watch_dir, a.day, {"event": "abort", "error": str(exc)[:200], **r.meter.summary()})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
