"""スキーマ（schema_defs.json）の確認: 全環境で同一か・実データが適合しているか（健全性）・未定義カテゴリ。

定義は git 管理の ``src/scraper/schema_defs.json`` が正本。fingerprint が環境間で一致していれば同じ定義で検証されている。

健全性の検証（``mode``）:
  off     … 検証しない（存在のみ）
  sample  … カテゴリごとの最新 N 件だけ検証（既定。download は N × カテゴリ数）
  full    … 範囲内の全データを検証。**台帳（ledger）**に結果を残し、ファイルの更新時刻とカテゴリ定義のハッシュが
            変わらない限り再検証しない。1 回の download 件数は ``budget`` で制限し、複数回の実行で全件に到達する
            （新しい race_id から順に検証）。GCS の送信課金は未検証分の download 量にだけ比例する。
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.data_health.keytable import HARVEST_CATEGORIES, KeyTable, extract_meta
from src.data_health.spec import RACE_CATEGORIES
from src.scraper import schemas

logger = logging.getLogger("data_health.schema")
MAX_FAILED_KEYS = 5
MODES = ("off", "sample", "full")
JRA_PLACES = {f"{i:02d}" for i in range(1, 11)}


def undefined_categories() -> list[dict[str, str]]:
    """CATEGORY_MAP のうちスキーマが無いもの（理由付き）。理由も無い = 定義漏れ。"""
    from src.scraper.storage import HybridStorage

    out = []
    for cat in sorted(HybridStorage.CATEGORY_MAP):
        if cat in schemas.SCHEMAS:
            continue
        r = schemas.NO_SCHEMA_REASON.get(cat)
        out.append({"category": cat, "status": r["status"] if r else "missing", "note": r["note"] if r else "スキーマも理由も未登録"})
    return out


def issue_codes(rep: dict[str, Any]) -> list[str]:
    """validate の結果を短いコードにする（台帳に保存する用）。例: top_missing:race_id / entry_type:odds"""
    out = [f"top_missing:{f}" for f in rep["top_missing"]]
    out += [f"top_type:{e['field']}" for e in rep["top_type_errors"]]
    out += [f"top_constraint:{e['field']}" for e in rep["top_constraint_errors"]]
    for grp, d in (rep.get("entry_issues") or {}).items():
        kind = grp.split("_")[0]                # missing / type / constraint
        out += [f"entry_{kind}:{f}" for f in d]
    for lkey, sub in (rep.get("multi_list_issues") or {}).items():
        out.append(f"list_issue:{lkey}")
    return out or ["invalid"]


# ── 台帳 ─────────────────────────────────────────────────────────────────

class Ledger:
    """カテゴリごとの検証結果。``ok`` = {key: 更新時刻}、``bad`` = {key: {ts, issues}}。"""

    def __init__(self, category: str, directory: Path | None):
        self.category, self.directory = category, directory
        self.fp = schemas.category_fingerprint(category)
        self.ok: dict[str, float] = {}
        self.bad: dict[str, dict[str, Any]] = {}
        self.na: dict[str, float] = {}          # 「取得を試みたが存在しない」スタブ（_meta.not_available）。健全でも不足でもない
        if directory:
            try:
                d = json.loads((directory / f"{category}.json").read_text(encoding="utf-8"))
                if d.get("fp") == self.fp:                       # 定義が変わったカテゴリは全件再検証
                    self.ok, self.bad, self.na = d.get("ok", {}), d.get("bad", {}), d.get("na", {})
            except (OSError, ValueError):
                pass

    def state(self, key: str, ts: float) -> tuple[str, list[str]] | None:
        """台帳にあり、ファイルが更新されていなければ ("ok"|"bad", issues)。"""
        if key in self.ok and self.ok[key] == ts:
            return "ok", []
        if key in self.na and self.na[key] == ts:
            return "na", []
        b = self.bad.get(key)
        if b and b["ts"] == ts:
            return "bad", b["issues"]
        return None

    def record(self, key: str, ts: float, ok: bool, issues: list[str], examples: list[str] | None = None,
               na: bool = False) -> None:
        self.ok.pop(key, None)
        self.bad.pop(key, None)
        self.na.pop(key, None)
        if na:
            self.na[key] = ts
        elif ok:
            self.ok[key] = ts
        else:
            self.bad[key] = {"ts": ts, "issues": issues, "examples": examples or []}

    def prune(self, keys: set[str]) -> None:
        self.ok = {k: v for k, v in self.ok.items() if k in keys}
        self.bad = {k: v for k, v in self.bad.items() if k in keys}
        self.na = {k: v for k, v in self.na.items() if k in keys}

    def save(self) -> None:
        if not self.directory:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"category": self.category, "fp": self.fp, "ok": self.ok, "bad": self.bad, "na": self.na}
        (self.directory / f"{self.category}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                                                            encoding="utf-8")


EXAMPLES_PER_RECORD = 3


def _validate_one(storage: Any, category: str, key: str) -> tuple[bool, list[str], dict[str, Any], list[str], bool]:
    """(適合, 問題コード, キーテーブル用のメタ, 違反の中身, 取得不可スタブか)。download した JSON から日付なども取り出す。"""
    try:
        data = storage.load(category, key, bypass_cache=True)
    except Exception as e:                      # 1 件の失敗で全体を止めない
        logger.warning("load 失敗 %s/%s: %s", category, key, e)
        return False, ["unreadable"], {}, [f"読み込み失敗: {type(e).__name__}"], False
    if not isinstance(data, dict):
        return False, ["unreadable"], {}, ["読み込めない（存在するが JSON として取得できない）"], False
    if (data.get("_meta") or {}).get("not_available"):          # スクレイパーが保存した「存在しない」スタブ
        return True, [], {}, [], True
    rep = schemas.validate(category, data)
    meta = extract_meta(data) if category in HARVEST_CATEGORIES else {}
    examples = [schemas.describe_violation(v) for v in rep.get("violations", [])[:EXAMPLES_PER_RECORD]]
    return bool(rep["passed"]), ([] if rep["passed"] else issue_codes(rep)), meta, examples, False


def refresh_ledger(storage: Any, ledger: Ledger, keys: dict[str, float], todo: list[str], *, limit: int | None, workers: int,
                   keytable: KeyTable | None = None, on_progress: Any = None) -> int:
    """todo のうち limit 件（None = 全件）を download して検証し、台帳へ記録する。検証した件数を返す。"""
    batch = todo if limit is None else todo[:max(limit, 0)]
    if not batch:
        return 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (key, (ok, issues, meta, examples, na)) in enumerate(
                zip(batch, pool.map(lambda k: _validate_one(storage, ledger.category, k), batch)), 1):
            ledger.record(key, keys[key], ok, issues, examples, na)
            if keytable is not None and meta:
                keytable.update(key, meta, "data")
            if on_progress is not None and i % 100 == 0:
                on_progress(i)
    return len(batch)


# ── 検証の実行 ───────────────────────────────────────────────────────────

def validate_scope(env: str, storage: Any, present: dict[str, dict[str, dict[str, float]]], *, mode: str, sample: int,
                   budget: int, workers: int, ledger_dir: Path | None, categories: list[str] | None = None,
                   keytable: KeyTable | None = None, skip_keys: set[str] | None = None, progress: Any = None
                   ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """返り値: (health, stats)。
    health[cat] = {"ok": set, "bad": {key: issues}, "scope": set}  … 台帳で最新と確認できた分と、評価の範囲
    """
    if mode not in MODES:
        raise ValueError(f"mode は {'/'.join(MODES)} のいずれか: {mode!r}")
    health: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {"mode": mode, "budget": budget, "validated_now": 0, "categories": {}}
    if mode == "off":
        return health, stats
    names = categories if categories is not None else [
        c.name for c in RACE_CATEGORIES if c.level[env] != "skip" and c.name in schemas.SCHEMAS]
    left = budget
    plan: list[tuple[str, dict[str, float], Ledger, set[str], list[str]]] = []
    for cat in names:
        if cat not in schemas.SCHEMAS:
            continue
        keys = {k: ts for ys in present.get(cat, {}).values() for k, ts in ys.items()
                if k[4:6] in JRA_PLACES and k not in (skip_keys or ())}
        ledger = Ledger(cat, ledger_dir)
        ledger.prune(set(keys))
        if mode == "sample":
            scope = {k for _, k in sorted(((ts, k) for k, ts in keys.items()), reverse=True)[:sample]}
        else:
            scope = set(keys)
        todo = sorted((k for k in scope if ledger.state(k, keys[k]) is None), reverse=True)     # 新しい race_id から
        plan.append((cat, keys, ledger, scope, todo))
    grand = sum(len(t) for *_, t in plan)
    grand = min(grand, budget) if (mode == "full" and budget > 0) else grand                    # 進捗表示用の総数
    finished = 0
    for cat, keys, ledger, scope, todo in plan:
        limit = None if (mode == "sample" or budget <= 0) else min(len(todo), max(left, 0))     # None = 無制限
        base_done = finished
        cb = (lambda n, c=cat, b=base_done: progress("健全性の検証", b + n, grand, c)) if progress and grand else None
        done = refresh_ledger(storage, ledger, keys, todo, limit=limit, workers=workers, keytable=keytable, on_progress=cb)
        finished += done
        if progress and grand:
            progress("健全性の検証", finished, grand, cat)
        if mode == "full" and budget > 0:
            left -= done
        ledger.save()
        ok, bad, examples, na = set(), {}, {}, set()
        for k in scope:
            st = ledger.state(k, keys[k])
            if st and st[0] == "ok":
                ok.add(k)
            elif st and st[0] == "na":
                na.add(k)
            elif st:
                bad[k] = st[1]
                examples[k] = ledger.bad[k].get("examples", [])
        health[cat] = {"ok": ok, "bad": bad, "scope": scope, "examples": examples, "na": na}
        stats["validated_now"] += done
        stats["categories"][cat] = {"scope": len(scope), "healthy": len(ok), "invalid": len(bad), "not_available": len(na),
                                    "unvalidated": len(scope) - len(ok) - len(bad) - len(na), "validated_now": done}
    stats["budget_left"] = max(left, 0) if (mode == "full" and budget > 0) else None
    stats["complete"] = all(c["unvalidated"] == 0 for c in stats["categories"].values())
    return health, stats


# ── レポート ─────────────────────────────────────────────────────────────

def schema_report(env: str, storage: Any, present: dict[str, dict[str, dict[str, float]]], sample: int, *,
                  health: dict[str, dict[str, Any]] | None = None, stats: dict[str, Any] | None = None,
                  mode: str = "sample") -> dict[str, Any]:
    """スキーマ fingerprint・カテゴリ別の健全性・未定義カテゴリ。health 省略時は sample 件を検証して作る。"""
    if health is None:
        health, stats = validate_scope(env, storage, present, mode="sample" if sample > 0 else "off", sample=sample,
                                       budget=0, workers=4, ledger_dir=None)
        mode = "sample" if sample > 0 else "off"
    cats: dict[str, Any] = {}
    for cat, h in health.items():
        sampled = len(h["ok"]) + len(h["bad"])              # 取得不可スタブ(na)は健全性の母数に入れない
        if not sampled:
            continue
        issues: dict[str, int] = {}
        for codes in h["bad"].values():
            for c in codes:
                issues[c] = issues.get(c, 0) + 1
        ex_count: dict[str, int] = {}
        for lines in h.get("examples", {}).values():
            for line in lines[:1]:
                ex_count[line] = ex_count.get(line, 0) + 1
        cats[cat] = {"sampled": sampled, "passed": len(h["ok"]), "advisory": bool(schemas.SCHEMAS[cat].get("advisory")),
                     "issues": issues, "failed_keys": sorted(h["bad"], reverse=True)[:MAX_FAILED_KEYS],
                     "examples": [f"{line} ×{n}" for line, n in sorted(ex_count.items(), key=lambda x: -x[1])[:3]],
                     "scope": len(h["scope"]), "unvalidated": len(h["scope"]) - sampled - len(h.get("na", ())),
                     "not_available": len(h.get("na", ()))}
    return {"fingerprint": schemas.schema_fingerprint(), "version": schemas.SCHEMA_VERSION, "sample": sample, "mode": mode,
            "categories": cats, "undefined": undefined_categories(), "defined": len(schemas.SCHEMAS),
            "stats": {k: v for k, v in (stats or {}).items()}}


def schema_findings(env: str, rep: dict[str, Any]) -> list[dict]:
    out = []
    levels = {c.name: c.level[env] for c in RACE_CATEGORIES}
    for cat, r in rep["categories"].items():
        bad = r["sampled"] - r["passed"]
        if not bad:
            continue
        top = ", ".join(f"{m}×{n}" for m, n in sorted(r["issues"].items(), key=lambda x: -x[1])[:3])
        sev = "fail" if levels.get(cat) == "required" and not r["advisory"] else "warn"
        out.append({"severity": sev, "area": "スキーマ",
                    "message": f"{cat}: 検証した {r['sampled']} 件中 {bad} 件がスキーマ不適合（{top}）",
                    "hint": "例: " + ", ".join(r["failed_keys"]) + (" ／ 値: " + " ; ".join(r["examples"][:2]) if r.get("examples") else "")
                            + " ／ 不適合の race_id は race_ids/invalid_*.txt、詳細は invalid_detail.json。再取得は scrape_plan.json"})
    st = rep.get("stats") or {}
    if rep.get("mode") == "full" and st and not st.get("complete"):
        pend = sum(c["unvalidated"] for c in st["categories"].values())
        out.append({"severity": "info", "area": "スキーマ", "message": f"健全性の検証が未完了: 未検証 {pend} 件（今回 {st['validated_now']} 件を検証）",
                    "hint": "再実行で続きから検証（台帳に保存済み）。1 回の件数は DATA_HEALTH_VALIDATE_BUDGET"})
    for u in rep["undefined"]:
        if u["status"] == "missing":
            out.append({"severity": "warn", "area": "スキーマ", "message": f"{u['category']}: スキーマも未定義の理由も未登録",
                        "hint": "schema_defs.json に定義または no_schema_reason を追加（CI で検出されます）"})
        elif u["status"] == "pending" and env != "dev":
            out.append({"severity": "info", "area": "スキーマ", "message": f"{u['category']}: スキーマ未定義（実データ観測待ち）",
                        "hint": "stg/prod で schema_infer collect --source storage → apply"})
    return out
