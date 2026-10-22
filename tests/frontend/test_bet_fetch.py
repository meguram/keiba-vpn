"""frontend/lib/betFetch.ts: 推奨馬券 API の失敗の分類（原因別の文言）。tsc と node があるときだけ実行する。"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TSC = ROOT / "frontend" / "node_modules" / ".bin" / "tsc"

pytestmark = pytest.mark.skipif(not TSC.exists() or shutil.which("node") is None, reason="tsc / node が無い環境")


def _run(tmp_path, expr: str):
    subprocess.run([str(TSC), str(ROOT / "frontend/lib/betFetch.ts"), "--outDir", str(tmp_path), "--module", "commonjs", "--target", "es2019", "--skipLibCheck"], check=True, capture_output=True)
    js = f"const m = require({json.dumps(str(tmp_path / 'betFetch.js'))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_failure_table(tmp_path):
    r = _run(tmp_path, """{
      offline: m.classifyFetchThrow(false),
      network: m.classifyFetchThrow(true),
      unknown_online: m.classifyFetchThrow(undefined),
      redirect_opaque: m.isSessionRedirect({type: 'opaqueredirect', status: 0}),
      redirect_302: m.isSessionRedirect({type: 'basic', status: 302}),
      ok_200: m.isSessionRedirect({type: 'basic', status: 200}),
      s500: m.classifyHttpFailure(500, null),
      s502_msg: m.classifyHttpFailure(502, 'bad'),
      s404: m.classifyHttpFailure(404, '予測結果がありません'),
      s400_default: m.classifyHttpFailure(400, undefined),
      render: m.classifyRenderError(),
      session: m.sessionExpired(),
    }""")
    assert r["offline"]["kind"] == "offline" and not r["offline"]["reload"]
    assert r["network"]["kind"] == "network" and r["network"]["reload"] and "再読み込み" in r["network"]["message"]
    assert r["unknown_online"]["kind"] == "network"
    assert r["redirect_opaque"] and r["redirect_302"] and not r["ok_200"]
    assert r["session"]["kind"] == "session" and r["session"]["reload"]
    assert r["s500"]["kind"] == "server" and "HTTP 500" in r["s500"]["message"]
    assert "bad" in r["s502_msg"]["message"] and "HTTP 502" in r["s502_msg"]["message"]
    assert r["s404"]["kind"] == "client" and r["s404"]["message"] == "予測結果がありません"
    assert "HTTP 400" in r["s400_default"]["message"]
    assert r["render"]["kind"] == "render" and "API には接続できています" in r["render"]["message"]
    # 『接続できませんでした』の文言になるのは fetch 例外（network）だけ
    kinds_with_phrase = [k for k, v in r.items() if isinstance(v, dict) and "接続できませんでした" in v.get("message", "")]
    assert sorted(kinds_with_phrase) == ["network", "unknown_online"]
