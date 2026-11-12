"""再学習の上流（段 3 追走難度 → 段 4 ラップ予想 → 段 7 前段 S3）をホストで切り離して実行するドライバ。

    cd /home/jovyan/work/keiba-vpn && setsid nohup /opt/conda/bin/python3 /home/jovyan/work/retrain_state/retrain_upstream.py \
        > /home/jovyan/work/retrain_state/logs/upstream.out 2>&1 < /dev/null &

- 出力はすべて $RETRAIN_OUT（既定 /home/jovyan/work/retrain_out）。本番の入力（tp54_out・tp55_out3・final_stage）と repo の config は書き換えない。
- 段の開始・完了・失敗は retrain_ctl に記録する。済みの段は飛ばし、実行中のまま残った段（前回の中断）は reset してやり直す。
- 失敗した段で止まる（自動で再試行しない）。重い処理は /tmp/heavy_run.sh 経由。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("RETRAIN_STATE_DIR") or "/home/jovyan/work/retrain_state")
OUT = Path(os.environ.get("RETRAIN_OUT") or "/home/jovyan/work/retrain_out")
REPO = Path(os.environ.get("KEIBA_REPO") or "/home/jovyan/work/keiba-vpn")
PY = os.environ.get("PYTHON") or "/opt/conda/bin/python3"
HEAVY = "/tmp/heavy_run.sh"
CTL = Path(__file__).resolve().parent / "retrain_ctl.py"
OLD_TP54 = Path("/home/jovyan/work/tp54_out")
OLD_TP55 = Path("/home/jovyan/work/tp55_out3")
YEARS = [str(y) for y in range(2020, 2026)]

ENV_COMMON = {"PYTHONPATH": ".", "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def ctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(CTL), *args], capture_output=True, text=True, env={**os.environ, "RETRAIN_STATE_DIR": str(STATE)})


def patched(module: str, assigns: dict[str, str], argv: list[str], call: str) -> list[str]:
    """モジュールの固定パスを差し替えて main を呼ぶ（repo の data/ は root 所有で、本番の成果物を上書きしないため）。"""
    pre = "; ".join(f"m.{k}=Path({v!r})" for k, v in assigns.items())
    code = f"import sys; from pathlib import Path; import {module} as m; {pre}; sys.argv={['x', *argv]!r}; m.{call}()"
    return [PY, "-c", code]


def steps() -> dict[str, dict]:
    t53, t54, t55, v6 = OUT / "tp53", OUT / "tp54", OUT / "tp55", OUT / "v6"
    fs = OUT / "training_pipeline"
    e54 = {"TP54_OUT": str(t54), "TP54_CONFIG_DIR": str(t54 / "config_v2"), "TP54_THREADS": "3"}
    e55 = {"TP55_OUT": str(t55), "TP55_THREADS": "3"}
    ev6 = {"V6_OUT": str(v6), "V6_EASE2": str(t54 / "groups/EASE2.parquet"), "V6_HORSE_EASE": str(t54 / "horse_ease.parquet"),
           "V6_LAP": str(t55 / "groups/LAP.parquet"), "V6_THREADS": "4"}
    return {
        "s3": {
            "cmds": [
                ("tp53", patched("src.scripts.data.tp53_dataset", {"OUT": str(t53)}, ["--years", *YEARS], "main"), {}),
                ("tp54_dataset", patched("src.scripts.data.tp54_dataset", {"TP53": str(t53)}, [], "build"), e54),
                ("tp54_eval", [PY, "-m", "src.scripts.data.tp54_model", "eval"], e54),
                ("tp54_oof", [PY, "-m", "src.scripts.data.tp54_model", "oof"], e54),
                ("tp54_final", [PY, "-m", "src.scripts.data.tp54_model", "final"], e54),
            ],
            "outputs": [t53 / "horse.parquet", t54 / "horse_ease.parquet", t54 / "groups/EASE2.parquet",
                        t54 / "config_v2/model.txt", t54 / "config_v2/model_demand.txt", t54 / "config_v2/meta.json"],
            "parity": [(t54 / "groups/EASE2.parquet", OLD_TP54 / "groups/EASE2.parquet"), (t54 / "horse_ease.parquet", OLD_TP54 / "horse_ease.parquet")],
        },
        "s4": {
            "cmds": [
                ("tp55_build", [PY, "-m", "src.scripts.data.tp55_build", "--out", str(t55)], e55),
                ("tp55_lap_group", [PY, "-m", "src.scripts.data.tp55_win_eval", "make-group"], e55),
            ],
            "outputs": [t55 / "races_feat.parquet", t55 / "groups/LAP.parquet"],
            "parity": [(t55 / "groups/LAP.parquet", OLD_TP55 / "groups/LAP.parquet")],
        },
        "s7": {
            "cmds": [
                ("v6_build_e", [PY, "-m", "src.scripts.data.v6_stage", "build", "e"], ev6),
                *[(f"fit_stage_{y}", patched("src.scripts.data.fit_final_stage_v6", {"OLD": str(fs)}, ["--variant", "e", "--fit-end", y], "main"), ev6) for y in ("2024", "2025")],
            ],
            "outputs": [v6 / "s3x_e.parquet", fs / "final_stage/v6_e_2024/parity_vs_oof_2025.json", fs / "final_stage/v6_e_2025/parity_vs_oof_2025.json"],
            "parity": [(v6 / "s3x_e.parquet", Path("/home/jovyan/work/keiba-vpn/data/local/modeling/training_pipeline_v4/v6/s3x_e.parquet"))],
        },
    }


def parity(new: Path, old: Path) -> dict:
    """新旧の同名の数値列で、共通の (race_id, horse_id) の相関と行数差を見る（同じ定義・同じ入力なら 1 に近い）。"""
    import numpy as np
    import pandas as pd

    if not old.exists():
        return {"old": "なし"}
    a, b = pd.read_parquet(new), pd.read_parquet(old)
    key = [k for k in ("race_id", "horse_id") if k in a.columns and k in b.columns]
    res = {"rows_new": len(a), "rows_old": len(b)}
    if key:
        for d in (a, b):
            for k in key:
                d[k] = d[k].astype(str)
        m = a.drop_duplicates(key).merge(b.drop_duplicates(key), on=key, suffixes=("", "_old"))
        res["rows_common"] = len(m)
        cors = {}
        for c in a.columns:
            if c in key or c + "_old" not in m.columns or not np.issubdtype(m[c].dtype, np.number):
                continue
            x, y = m[c].to_numpy(float), m[c + "_old"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 10 and x[ok].std() > 0 and y[ok].std() > 0:
                cors[c] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)
        res["corr_with_old"] = cors
        if cors:
            res["corr_min"] = min(cors.values())
    return res


def run_step(sid: str, spec: dict) -> bool:
    LOGS = STATE / "logs"
    LOGS.mkdir(parents=True, exist_ok=True)
    r = ctl("start", sid, "--note", "retrain_upstream")
    if r.returncode != 0:
        log(f"{sid}: start 拒否: {r.stdout}{r.stderr}")
        return False
    t0 = time.time()
    with open(LOGS / f"{sid}.log", "a", encoding="utf-8") as lf:
        for label, cmd, env in spec["cmds"]:
            lf.write(f"\n===== [{time.strftime('%F %T')}] {label}\n")
            lf.flush()
            log(f"{sid}: {label} 開始")
            rc = subprocess.run([HEAVY, f"retrain_{sid}_{label}", "nice", "-n", "15", *cmd], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, **ENV_COMMON, **env}).returncode
            if rc != 0:
                ctl("fail", sid, "--note", f"{label} が rc={rc}。ログ: {LOGS / f'{sid}.log'}")
                log(f"{sid}: {label} 失敗 rc={rc}")
                return False
    missing = [str(p) for p in spec["outputs"] if not p.exists() or p.stat().st_mtime < t0 - 5]
    if missing:
        ctl("fail", sid, "--note", "出力が無い／古い: " + ", ".join(missing))
        log(f"{sid}: 出力不足 {missing}")
        return False
    par = {}
    for new, old in spec["parity"]:
        try:
            par[new.name] = parity(new, old)
        except Exception as e:  # noqa: BLE001 — 比較の失敗で本体を失敗にしない
            par[new.name] = {"error": repr(e)}
    (OUT / f"parity_{sid}.json").write_text(json.dumps(par, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: {kk: v.get(kk) for kk in ("rows_new", "rows_old", "rows_common", "corr_min")} for k, v in par.items()}
    ctl("done", sid, "--note", f"{round((time.time() - t0) / 60)}分。出力 {OUT}。旧版との比較 {json.dumps(brief, ensure_ascii=False)}（詳細 parity_{sid}.json）")
    log(f"{sid}: 完了")
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in steps():
        st = json.loads((STATE / "state.json").read_text(encoding="utf-8"))["steps"][sid]["status"] if (STATE / "state.json").exists() else "pending"
        if st in ("done", "skipped"):
            continue
        if st == "running":
            ctl("reset", sid, "--note", "ドライバ再開のため前回の中断を戻す")
    for sid, spec in steps().items():
        st = json.loads((STATE / "state.json").read_text(encoding="utf-8"))["steps"][sid]["status"]
        if st in ("done", "skipped"):
            log(f"{sid}: {st} のため飛ばす")
            continue
        if not run_step(sid, spec):
            return 1
    log("上流（s3・s4・s7）完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
