"""リリース・マニフェスト: 『その環境が今どんな挙動をするか』を決める要素を 1 つの JSON にまとめる。

stg と prod（移行後は Cloud Run / Jobs も）で、特徴量・モデル・config・フラグ・フロントが揃っているかを
``compare`` で機械的に比べるための材料。標準ライブラリのみ（VPS・Cloud Run・ホストのどこでも動く）。秘密は含めない。

  python -m src.ops.release_manifest print                    # この環境の実効マニフェスト
  python -m src.ops.release_manifest print --detail           # ファイル単位のハッシュつき
  python -m src.ops.release_manifest write-frontend-version --frontend-dir frontend --env prod --out frontend/public/version.json
  python -m src.ops.release_manifest suggest-flags            # この環境の挙動フラグから config/behavior_flags.json の案を出す
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[2]

# 挙動（予測・期待値・印・推奨の出力）を決めるコードのディレクトリ。ここの *.py のハッシュが『コードの実効版』になる
# （ホストは git の HEAD と実ファイルがずれた継ぎはぎ運用なので、commit ではなくファイル内容で比べる）。
CODE_DIRS = (
    "src/api", "src/pipeline/inference", "src/pipeline/models", "src/pipeline/features",
    "src/pipeline/betting", "src/utils", "src/config",
)
# 挙動を決める config。運用専用（メンテ窓・スクレイプ防御・費用・ポート・会員名簿・監視）は比べない。
CONFIG_GLOBS = ("config/*.json", "config/*.yaml", "config/*/*.json", "config/*/*/*.json")
CONFIG_EXCLUDE = (
    "config/anti_scraping", "config/maintenance", "config/cost_assumptions", "config/members",
    "config/entries_schedule", "config/drift_", "config/frozen_tracking", "config/gcp_job_map",
    "config/behavior_flags",   # フラグ表自体は flags 側で比べる
)
FRONTEND_EXCLUDE_DIRS = {"node_modules", "out", ".turbo", ".git", "dist", "coverage", ".cache"}   # ほか ".next*"（.next-dev / .next-stg など）はすべて除外
FRONTEND_EXCLUDE_FILES = {"version.json", "next-env.d.ts", "tsconfig.tsbuildinfo"}
FRONTEND_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".css", ".json", ".svg", ".html", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".woff", ".woff2"}

# 値を返してよい挙動フラグ（allowlist）。秘密・認証・URL・パスの値は入れない。パス系は _PATH_FLAGS で末尾名だけにする。
BEHAVIOR_FLAGS = (
    "KEIBA_PROB_CALIBRATION", "KEIBA_PACE_TYPE", "KEIBA_PACE_COHERENCE", "KEIBA_PACE_COHERENCE_V2",
    "KEIBA_LAP_FORECAST", "KEIBA_TRACKING_V2", "KEIBA_TRACKING_V2_RECOMPUTE", "KEIBA_GROWTH_INDEX", "KEIBA_GROWTH_AXIS",
    "KEIBA_V3_AUX", "KEIBA_V3_SERVING", "KEIBA_V5_SHADOW", "KEIBA_DEFAULT_MODEL_VERSION", "KEIBA_MODEL_STRICT_VERSIONS",
    "KEIBA_TRAINING_COVERAGE", "KEIBA_INFERENCE_ENGINE", "KEIBA_FEATURE_BUILDER", "KEIBA_PREDICTION_RESULTS",
    "KEIBA_WEIGHT_TRIGGER_MODE", "KEIBA_PRE_RACE_PREDICT_ENABLED", "KEIBA_ODDS_LATE_SNAPSHOTS", "KEIBA_ODDS_TRIFECTA",
    "KEIBA_LEGACY_API", "KEIBA_EVE_PRECOMPUTE_TRACKING", "KEIBA_EVE_PRECOMPUTE_FINAL_ODDS",
    # 環境の識別・見た目（意図した差）
    "KEIBA_ENV", "KEIBA_DEPLOYMENT_LABEL", "KEIBA_PROFILE", "KEIBA_STG_MOCK_PREDICTIONS", "KEIBA_TEST_USER",
    # パス（末尾名だけ記録）
    "KEIBA_V3_BUNDLE", "KEIBA_V3_STAGE", "KEIBA_V5_BUNDLE", "KEIBA_V5_STAGE", "KEIBA_V5_TL_REF", "KEIBA_KESI_CONFIG",
    "KEIBA_MODEL_STORE",
)
_PATH_FLAGS = {"KEIBA_V3_BUNDLE", "KEIBA_V3_STAGE", "KEIBA_V5_BUNDLE", "KEIBA_V5_STAGE", "KEIBA_V5_TL_REF", "KEIBA_KESI_CONFIG"}
_HASHED_FLAGS = {"KEIBA_MODEL_STORE"}   # バケット名などは出さず、同一かどうかだけ分かる短いハッシュにする

# 環境ごとに違って当然の項目（比較から外す）。ドット区切りの完全一致。
INTENTIONAL_DIFF = frozenset({
    "env", "host", "generated_at", "component", "flags.KEIBA_ENV", "flags.KEIBA_DEPLOYMENT_LABEL", "flags.KEIBA_PROFILE",
    "flags.KEIBA_STG_MOCK_PREDICTIONS", "flags.KEIBA_TEST_USER", "frontend.env", "frontend.public_env.NEXT_PUBLIC_ENV_LABEL", "frontend.built_at", "frontend.release",
    "frontend.build_id", "frontend.commit", "frontend.short", "code.commit", "flag_sources", "flag_drift",
})

_CACHE: dict[str, tuple[float, Any]] = {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(files: dict[str, str]) -> str:
    return _sha("\n".join(f"{k}\0{v}" for k, v in sorted(files.items())).encode())[:16]


def _cached(key: str, ttl: float, fn):
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def code_files(root: Path = ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in CODE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts or p.is_symlink():
                continue
            out[p.relative_to(root).as_posix()] = _file_sha(p)
    return out


def config_files(root: Path = ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for pat in CONFIG_GLOBS:
        for p in sorted(root.glob(pat)):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or p.is_symlink() or rel.endswith(".local.json") or any(rel.startswith(x) for x in CONFIG_EXCLUDE):
                continue
            out[rel] = _file_sha(p)[:16]
    return out


def frontend_files(frontend_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not frontend_dir.is_dir():
        return out
    for p in sorted(frontend_dir.rglob("*")):
        if not p.is_file() or p.is_symlink() or p.name in FRONTEND_EXCLUDE_FILES or p.suffix not in FRONTEND_SUFFIXES:
            continue
        rel = p.relative_to(frontend_dir)
        if any(part in FRONTEND_EXCLUDE_DIRS or part.startswith(".next") for part in rel.parts):
            continue
        out[rel.as_posix()] = _file_sha(p)[:16]
    return out


def public_env(env: dict[str, str], env_file: Path | None = None) -> dict[str, str]:
    """ビルド時に埋め込まれる NEXT_PUBLIC_*（ブラウザに配られる値なので記録してよい）。env_file は KEY=VALUE 形式。"""
    vals: dict[str, str] = {}
    if env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("NEXT_PUBLIC_") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip("'\"")
    vals.update({k: v for k, v in env.items() if k.startswith("NEXT_PUBLIC_")})
    return {k: v[:80] for k, v in sorted(vals.items())}


def frontend_digest(frontend_dir: Path) -> str:
    return _tree_digest(frontend_files(frontend_dir))


def bundle_fingerprint(path: Path | None) -> dict[str, Any]:
    """バンドルの同一性。manifest.json（model_registry 形式）があればその files の sha256、無ければ名前+サイズ。"""
    if path is None:
        return {"name": None, "digest": None}
    p = Path(path)
    if not p.exists():
        return {"name": p.name, "digest": None, "missing": True}
    mf = p / "manifest.json"
    if mf.is_file():
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            files = {k: (v or {}).get("sha256", "") for k, v in (m.get("files") or {}).items()}
            return {"name": p.name, "version": m.get("version"), "n_features": m.get("n_features"), "digest": _tree_digest(files), "basis": "manifest"}
        except (OSError, ValueError):
            pass
    files = {f.name: str(f.stat().st_size) for f in sorted(p.iterdir()) if f.is_file()} if p.is_dir() else {p.name: str(p.stat().st_size)}
    return {"name": p.name, "digest": _tree_digest(files), "basis": "name+size"}


def _bundle_paths(env: dict[str, str], root: Path) -> dict[str, Path | None]:
    work = root / "data/local/modeling/training_pipeline"
    return {
        "v3": Path(env.get("KEIBA_V3_BUNDLE") or work / "final_v3/bundle_2025"),
        "v5": Path(env.get("KEIBA_V5_BUNDLE") or work / "final_v5/bundle_v5_2025"),
    }


def flag_value(name: str, env: dict[str, str]) -> str | None:
    v = env.get(name)
    if v is None or v == "":
        return None
    v = v.strip()
    if name in _PATH_FLAGS:
        return Path(v).name
    if name in _HASHED_FLAGS:
        return "sha:" + _sha(v.encode())[:8]
    return v[:80]


def load_flag_table(root: Path = ROOT) -> dict[str, Any]:
    p = root / "config/behavior_flags.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"defaults": {}, "env_overrides": {}, "env_overridable": []}


def _git_commit(root: Path) -> str | None:
    c = os.environ.get("KEIBA_GIT_COMMIT")
    if c:
        return c[:40]
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest(*, root: Path = ROOT, env: dict[str, str] | None = None, component: str = "api", detail: bool = False,
                   frontend_dir: Path | None = None) -> dict[str, Any]:
    e = dict(os.environ if env is None else env)
    cf = _cached(f"code:{root}", 300, lambda: code_files(root))
    cfg = _cached(f"cfg:{root}", 60, lambda: config_files(root))
    table = load_flag_table(root)
    defaults = {k: str(v) for k, v in (table.get("defaults") or {}).items()}
    envname = (e.get("KEIBA_ENV") or e.get("APP_ENV") or "prod").strip().lower() or "prod"
    envname = {"staging": "stg", "production": "prod", "development": "dev", "local": "dev"}.get(envname, envname)
    overrides = {k: str(v) for k, v in ((table.get("env_overrides") or {}).get(envname) or {}).items()}
    overridable = set(table.get("env_overridable") or []) | set((table.get("env_overrides") or {}).get("__all__", []))

    flags: dict[str, str | None] = {}
    sources: dict[str, str] = {}
    drift: list[str] = []
    for name in BEHAVIOR_FLAGS:
        explicit = e.get(name)
        wanted = overrides.get(name, defaults.get(name))
        if explicit not in (None, ""):
            flags[name] = flag_value(name, e)
            sources[name] = "env"
            if wanted is not None and explicit.strip() != wanted and name not in overridable:
                drift.append(name)
        elif wanted is not None:
            flags[name] = flag_value(name, {name: wanted})
            sources[name] = "file"
        else:
            flags[name] = None
            sources[name] = "unset"
    fdir = frontend_dir if frontend_dir is not None else root / "frontend"
    m: dict[str, Any] = {
        "schema": SCHEMA, "component": component, "env": envname, "host": socket.gethostname(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {
            "commit": _git_commit(root), "n_files": len(cf), "tree": _tree_digest(cf),
            "dirs": {d: _tree_digest({k: v for k, v in cf.items() if k.startswith(d + "/")}) for d in CODE_DIRS},
        },
        "models": {k: bundle_fingerprint(v) for k, v in _bundle_paths(e, root).items()},
        "configs": {"digest": _tree_digest(cfg), "n_files": len(cfg), "files": cfg},
        "flags": flags, "flag_sources": sources, "flag_drift": sorted(drift),
    }
    if fdir.is_dir():
        m["frontend"] = {"source_digest": _cached(f"fe:{fdir}", 120, lambda: frontend_digest(fdir)), "public_env": public_env(e)}
    if detail:
        m["code"]["files"] = cf
    return m


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    if not d and prefix:
        out[prefix] = {}
    return out


def overridable_paths(root: Path = ROOT) -> set[str]:
    """フラグ表で『環境ごとに違ってよい』と宣言した項目（flags.<名前>）。"""
    t = load_flag_table(root)
    names = set(t.get("env_overridable") or [])
    for envname, ov in (t.get("env_overrides") or {}).items():
        if isinstance(ov, dict):
            names |= set(ov)
    return {f"flags.{n}" for n in names}


def compare(a: dict, b: dict, *, allow: set[str] | frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """a と b の差を返す。``intentional`` が True の差は『環境ごとに違って当然』（INTENTIONAL_DIFF・allow）。"""
    fa, fb = flatten(a), flatten(b)
    ok = set(INTENTIONAL_DIFF) | set(allow)
    diffs: list[dict[str, Any]] = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va == vb:
            continue
        intentional = k in ok or any(k.startswith(x + ".") for x in ok)
        diffs.append({"path": k, "a": va, "b": vb, "intentional": intentional})
    return diffs


def blocking(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """意図しない差のうち、判断に使うもの（内訳の files は要約のため除く）。"""
    return [d for d in diffs if not d["intentional"] and ".files." not in d["path"] and not d["path"].endswith(".n_files")]


def write_frontend_version(frontend_dir: Path, env: str, out: Path, *, extra: dict | None = None, env_file: Path | None = None) -> dict:
    cur: dict[str, Any] = {}
    if out.is_file():
        try:
            cur = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            cur = {}
    cur.update({"env": env, "source_digest": frontend_digest(frontend_dir), "built_at": cur.get("built_at") or datetime.now().astimezone().isoformat(timespec="seconds")})
    cur["public_env"] = public_env(dict(os.environ), env_file)
    cur.update(extra or {})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cur, ensure_ascii=False) + "\n", encoding="utf-8")
    return cur


def suggest_flags(env: dict[str, str] | None = None) -> dict[str, Any]:
    """この環境で設定済みの挙動フラグから config/behavior_flags.json の案を作る（環境の識別用フラグは除く）。"""
    e = dict(os.environ if env is None else env)
    skip = {"KEIBA_ENV", "KEIBA_DEPLOYMENT_LABEL", "KEIBA_PROFILE", "KEIBA_STG_MOCK_PREDICTIONS", "KEIBA_TEST_USER", "KEIBA_MODEL_STORE"} | _PATH_FLAGS
    defaults = {n: e[n].strip() for n in BEHAVIOR_FLAGS if n not in skip and e.get(n, "").strip()}
    return {"version": 1, "defaults": defaults, "env_overrides": {"stg": {}, "prod": {}}, "env_overridable": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("print")
    p.add_argument("--detail", action="store_true")
    p.add_argument("--component", default="api")
    w = sub.add_parser("write-frontend-version")
    w.add_argument("--frontend-dir", default="frontend")
    w.add_argument("--env", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--env-file", help="ビルド時に読み込む env ファイル（NEXT_PUBLIC_* を記録する）")
    sub.add_parser("suggest-flags")
    a = ap.parse_args(argv)
    if a.cmd == "print":
        print(json.dumps(build_manifest(detail=a.detail, component=a.component), ensure_ascii=False, indent=1))
    elif a.cmd == "write-frontend-version":
        print(json.dumps(write_frontend_version(Path(a.frontend_dir), a.env, Path(a.out), env_file=Path(a.env_file) if a.env_file else None), ensure_ascii=False))
    else:
        print(json.dumps(suggest_flags(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
