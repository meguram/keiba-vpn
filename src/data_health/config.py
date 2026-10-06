"""データ存在チェックの設定（環境変数 ``DATA_HEALTH_*`` と CLI）。

優先順位: CLI 引数 > 環境変数 > 既定値。環境ごとの値は ``.env.stg`` / ``.env.prod`` に書けば
``KEIBA_ENV`` に応じて自動で切り替わる（同一 ``.env`` に書く場合は ``DATA_HEALTH_LEVELS_STG`` のように環境名を付ける）。

  DATA_HEALTH_ENV            評価する要件プロファイル（dev|stg|prod）。既定は KEIBA_ENV
  DATA_HEALTH_SINCE / _UNTIL 評価期間（YYYY-MM-DD）
  DATA_HEALTH_FAIL_ON        終了コード 2 にする重大度（fail|warn|never）
  DATA_HEALTH_OUT_DIR        結果の保存ルート（直下に <env>/ を作る）
  DATA_HEALTH_LEVELS         重要度の上書き  例: race_barometer=required,C07=optional,infra.redis=optional
  DATA_HEALTH_LEVELS_<ENV>   同上（その環境のときだけ有効。<ENV> は DEV|STG|PROD。汎用より優先）
  DATA_HEALTH_SKIP           評価しない ID（カンマ区切り）  例: infra.redis,C07
  DATA_HEALTH_VALIDATE       健全性（スキーマ適合）の検証範囲  off|sample|full（既定 full）
                             full は範囲内の全データを検証。結果は台帳に残り、更新されたものだけ再検証する
  DATA_HEALTH_VALIDATE_BUDGET  full のとき 1 回の実行で download する最大件数（既定 5000、0 = 無制限。dev は 0）
  DATA_HEALTH_VALIDATE_WORKERS 検証の並列数（既定 8）
  DATA_HEALTH_COMPLETE_LEVELS  「完全性」(--require-complete) に含める重要度  例: required,recommended,optional（既定 required,recommended）
  DATA_HEALTH_SCHEMA_SAMPLE  sample モードの検証件数（カテゴリごとの最新 N 件。既定 20）
  DATA_HEALTH_HORSE_PAST_DAYS / _FUTURE_DAYS / _RACES_MAX   馬の評価窓
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from src.data_health import spec as S

LEVEL_VALUES = ("required", "recommended", "optional", "skip")
FAIL_ON_VALUES = ("fail", "warn", "never")
PREFIX = "DATA_HEALTH_"


@dataclass
class Settings:
    env: str                                   # 評価する要件プロファイル
    since: date | None = None
    until: date | None = None
    out_dir: str | None = None
    fail_on: str = "fail"
    levels: dict[str, str] = field(default_factory=dict)
    horse_past_days: int = S.HORSE_WINDOW_PAST_DAYS
    horse_future_days: int = S.HORSE_WINDOW_FUTURE_DAYS
    horse_races_max: int = S.HORSE_RACES_MAX
    schema_sample: int = 20
    validate: str = "full"                  # off | sample | full（健全性＝スキーマ適合の検証範囲）
    validate_budget: int = 5000             # full のとき 1 回の実行で download する最大件数（0 = 無制限）
    validate_workers: int = 8
    complete_levels: tuple[str, ...] = ("required", "recommended")   # 「完全性」の判定に含める重要度
    warnings: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)   # 設定名 → 値の出所（env / cli）


def known_ids() -> set[str]:
    from src.data_health.checks import INFRA_CHECK_IDS

    ids = {c.name for c in S.RACE_CATEGORIES} | {c.name for c in S.HORSE_CATEGORIES} | {a.id for a in S.ARTIFACTS}
    return ids | {f"gcs.{g['id']}" for g in S.GCS_OBJECTS} | set(INFRA_CHECK_IDS)


def _get(environ: Mapping[str, str], name: str) -> str:
    return (environ.get(PREFIX + name) or "").strip()


def _date(name: str, text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{PREFIX}{name}={text!r} は YYYY-MM-DD で指定してください") from None


def _int(name: str, text: str, default: int, lo: int, hi: int) -> int:
    if not text:
        return default
    try:
        v = int(text)
    except ValueError:
        raise ValueError(f"{PREFIX}{name}={text!r} は整数で指定してください") from None
    if not lo <= v <= hi:
        raise ValueError(f"{PREFIX}{name}={v} は {lo}〜{hi} の範囲で指定してください")
    return v


def parse_levels(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" not in part:
            raise ValueError(f"重要度の指定 {part!r} は ID=レベル の形式で書いてください（例: race_barometer=required）")
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip().lower()
        if v not in LEVEL_VALUES:
            raise ValueError(f"{k} のレベル {v!r} は {'/'.join(LEVEL_VALUES)} のいずれかです")
        out[k] = v
    return out


def load_settings(environ: Mapping[str, str] | None = None, *, profile: str | None = None) -> Settings:
    """環境変数から設定を読む。``profile`` 省略時は DATA_HEALTH_ENV → KEIBA_ENV の順。"""
    environ = os.environ if environ is None else environ
    if profile is None:
        profile = _get(environ, "ENV").lower()
        if not profile:
            from src.config.deployment import keiba_env

            profile = keiba_env()
    if profile not in S.ENVS:
        raise ValueError(f"環境 {profile!r} は {'/'.join(S.ENVS)} のいずれかです")
    st = Settings(env=profile)
    if _get(environ, "SINCE"):
        st.since, st.sources["since"] = _date("SINCE", _get(environ, "SINCE")), "env"
    if _get(environ, "UNTIL"):
        st.until, st.sources["until"] = _date("UNTIL", _get(environ, "UNTIL")), "env"
    fo = _get(environ, "FAIL_ON").lower()
    if fo:
        if fo not in FAIL_ON_VALUES:
            raise ValueError(f"{PREFIX}FAIL_ON={fo!r} は {'/'.join(FAIL_ON_VALUES)} のいずれかです")
        st.fail_on, st.sources["fail_on"] = fo, "env"
    if _get(environ, "OUT_DIR"):
        st.out_dir, st.sources["out_dir"] = _get(environ, "OUT_DIR"), "env"
    st.horse_past_days = _int("HORSE_PAST_DAYS", _get(environ, "HORSE_PAST_DAYS"), st.horse_past_days, 0, 365)
    st.horse_future_days = _int("HORSE_FUTURE_DAYS", _get(environ, "HORSE_FUTURE_DAYS"), st.horse_future_days, 0, 60)
    st.horse_races_max = _int("HORSE_RACES_MAX", _get(environ, "HORSE_RACES_MAX"), st.horse_races_max, 1, 2000)

    mode = _get(environ, "VALIDATE").lower()
    if mode:
        if mode not in ("off", "sample", "full"):
            raise ValueError(f"{PREFIX}VALIDATE={mode!r} は off/sample/full のいずれかです")
        st.validate, st.sources["validate"] = mode, "env"
    budget_default = 0 if profile == "dev" else st.validate_budget      # dev はローカルなので無制限
    st.validate_budget = _int("VALIDATE_BUDGET", _get(environ, "VALIDATE_BUDGET"), budget_default, 0, 10_000_000)
    st.validate_workers = _int("VALIDATE_WORKERS", _get(environ, "VALIDATE_WORKERS"), st.validate_workers, 1, 64)
    cl = tuple(x.strip().lower() for x in _get(environ, "COMPLETE_LEVELS").split(",") if x.strip())
    if cl:
        bad = [x for x in cl if x not in ("required", "recommended", "optional")]
        if bad:
            raise ValueError(f"{PREFIX}COMPLETE_LEVELS={bad} は required/recommended/optional のいずれかです")
        st.complete_levels = cl
    st.schema_sample = _int("SCHEMA_SAMPLE", _get(environ, "SCHEMA_SAMPLE"), st.schema_sample, 0, 500)

    levels = parse_levels(_get(environ, "LEVELS"))
    levels.update(parse_levels(_get(environ, f"LEVELS_{profile.upper()}")))
    for sid in filter(None, (p.strip() for p in _get(environ, "SKIP").split(","))):
        levels[sid] = "skip"
    unknown = sorted(set(levels) - known_ids())
    if unknown:
        st.warnings.append("未知の ID を無視します（綴りを確認）: " + ", ".join(unknown))
        levels = {k: v for k, v in levels.items() if k in known_ids()}
    st.levels = levels
    return st
