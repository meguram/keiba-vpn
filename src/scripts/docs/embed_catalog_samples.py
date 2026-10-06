"""docs/html/data/DATASET_CATALOG.html の「データセット一覧」に、各データのサンプルを折り畳みで埋め込む。

各行（A01〜G05）をクリックすると、サンプル（JSON は辞書型、特徴量などはテーブル）が開く。何度実行しても同じ結果（冪等）。

サンプルの出所:
  * 実データ … docs/requirements/data/scrape_process_samples/（過去に実スクレイプして保存した 1 レース・1 頭分）と、この PC にある
               ナレッジ・設定ファイル、data_features_reference.html に載っている先頭行
  * dev モック … make_dev_mock が生成する架空データ（スキーマ適合）。実データが手元に無いカテゴリ用
  * 変換ロジックの適用結果 … export_tables / build_horse_entity_store の変換を実サンプルに適用したもの
スキーマや取得形式を変えたら、このスクリプトを再実行してサンプルを更新する:
  python -m src.scripts.docs.embed_catalog_samples
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CAT = ROOT / "docs/html/data/DATASET_CATALOG.html"
SAMP = ROOT / "docs/requirements/data/scrape_process_samples"
E = html.escape
REAL = "実データ（2026-06-12 に実スクレイプして保存したサンプル）"
MOCK = "dev モック（架空データ。スキーマ適合）"

CSS = """
tr.ds{cursor:pointer}tr.ds:hover td{background:rgba(79,139,255,.08)}
tr.ds td.id::before{content:"▸ ";color:var(--accent-2)}tr.ds.open td.id::before{content:"▾ "}
tr.sample td{padding:0;background:#0d1730;border-bottom:1px solid var(--border)}
tr.sample details{padding:4px 14px}tr.sample summary{cursor:pointer;color:var(--muted);font-size:12px;padding:4px 0}
.smp-p{margin:8px 0 14px;padding:8px 12px;border:1px solid var(--border);border-radius:10px;background:#0b1428}
.smp-h{font-size:13px}.smp-src{font-size:11px;color:var(--muted);margin-left:8px}
.smp-n{font-size:12px;color:#c9d6ff;margin:2px 0}.smp-t{font-size:11px;color:var(--muted);margin-top:4px}
pre.smp{max-height:420px;overflow:auto;margin:4px 0 0;font-size:12px;background:#08101f;border:1px solid var(--border);border-radius:8px;padding:10px 12px}
table.smp{font-size:11.5px}table.smp th,table.smp td{padding:4px 8px;white-space:nowrap}
.smp-note{font-size:12.5px;color:#c9d6ff;margin:8px 0}.smp-hint{font-size:13px}
.smp-hint button{margin-left:8px;background:#1a2748;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:2px 10px;cursor:pointer}
"""
JS = """<script>
(function(){
  var rows=document.querySelectorAll('tr.ds');
  function det(tr){var n=tr.nextElementSibling;return n&&n.classList.contains('sample')?n.querySelector('details'):null;}
  function set(tr,open){var d=det(tr);if(d){d.open=open;tr.classList.toggle('open',open);}}
  rows.forEach(function(tr){
    tr.addEventListener('click',function(e){if(e.target.closest('a'))return;var d=det(tr);if(d)set(tr,!d.open);});
    var d=det(tr);if(d)d.addEventListener('toggle',function(){tr.classList.toggle('open',d.open);});
  });
  var all=document.getElementById('smpAll'),none=document.getElementById('smpNone');
  if(all)all.onclick=function(){rows.forEach(function(t){set(t,true);});};
  if(none)none.onclick=function(){rows.forEach(function(t){set(t,false);});};
})();
</script>
"""


def trim(o: Any, n: int = 2, depth: int = 0) -> Any:
    """リストは先頭 n 件に短縮（短縮した旨を末尾に入れる）。辞書のキーが多ければ先頭 12 個。_meta は時刻などだけ。"""
    if isinstance(o, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(o.items()):
            if k == "_meta":
                out[k] = {kk: vv for kk, vv in (v or {}).items() if kk in ("scraped_at_jst", "source", "race_list_source", "version")}
                continue
            if i >= 12 and depth > 0:
                out["…"] = f"他 {len(o) - 12} キー"
                break
            out[k] = trim(v, n, depth + 1)
        return out
    if isinstance(o, list):
        head = [trim(x, n, depth + 1) for x in o[:n]]
        return head + ([f"…（他 {len(o) - n} 件）"] if len(o) > n else [])
    return o


def P(label: str, data: Any, source: str, note: str = "") -> dict[str, Any]:
    return {"kind": "json", "label": label, "data": trim(data), "source": source, "note": note}


def T(label: str, cols: list, rows: list, source: str, note: str = "") -> dict[str, Any]:
    return {"kind": "table", "label": label, "cols": cols, "rows": rows, "source": source, "note": note}


def N(text: str) -> dict[str, Any]:
    return {"kind": "note", "text": text}


def build_samples() -> dict[str, list[dict[str, Any]]]:
    tmp = Path(tempfile.mkdtemp())
    prev = os.environ.get("KEIBA_PAGE_REFERENCE_DIR")
    os.environ["KEIBA_PAGE_REFERENCE_DIR"] = str(tmp / "pr")          # モック生成が実データの page_reference に触れないように
    try:
        return _build_samples(tmp)
    finally:
        if prev is None:
            os.environ.pop("KEIBA_PAGE_REFERENCE_DIR", None)
        else:
            os.environ["KEIBA_PAGE_REFERENCE_DIR"] = prev


def _build_samples(tmp: Path) -> dict[str, list[dict[str, Any]]]:
    import yaml

    from src.pipeline.build_horse_entity_store import _flatten_pedigree_json
    from src.scraper.dev_store import DevStore
    from src.scraper.export_tables import _data_to_flat_rows
    from src.scraper.storage import HybridStorage
    from src.scripts.data import make_dev_mock

    make_dev_mock.generate(date(2026, 10, 6), tmp / "mock")
    store, cmap = DevStore(tmp / "mock"), HybridStorage.CATEGORY_MAP

    def mock(cat: str) -> dict[str, Any]:
        d = store.read(cat, store.list_keys(cat, cmap[cat])[0], cmap[cat])
        if "_meta" in d:                                   # 実行のたびに変わる時刻は固定し、再生成しても同じ結果にする
            d["_meta"] = {**d["_meta"], "scraped_at_jst": "2026-10-06 00:00:00"}
        if "predicted_at" in d:
            d["predicted_at"] = "2026-10-06T00:00:00+00:00"
        return d

    def real(name: str) -> dict[str, Any]:
        return json.loads((SAMP / f"{name}.json").read_text(encoding="utf-8"))

    def jload(rel: str) -> Any:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))

    S: dict[str, list[dict[str, Any]]] = {}
    S["A01"] = [P("race_lists/20230625.json", real("nk_race_list"), REAL, "races は 36 件中 先頭 2 件に短縮")]
    S["A02"] = [P("race_day_schedule/20230625.json", real("nk_race_day_schedule"), REAL, "slots は先頭 2 件に短縮")]
    S["A03"] = [P("race_shutuba/2023/202309030811.json", real("nk_shutuba_entries"), REAL, "entries は 17 頭中 先頭 2 頭に短縮")]
    S["A04"] = [P("race_odds/2023/202309030811.json", real("nk_odds"), REAL)]
    S["A05"] = [P("race_pair_odds/…json", mock("race_pair_odds"), MOCK, "umaren / wide / umatan は各先頭 2 件に短縮")]
    S["A06"] = [P("race_paddock/2023/202309030811.json", real("nk_paddock"), REAL)]
    S["A07"] = [P("race_index/2023/202309030811.json", real("nk_speed_index"), REAL)]
    S["A08"] = [P("race_barometer/2023/202309030811.json", real("nk_barometer"), REAL)]
    S["A09"] = [P("race_trainer_comment/…json", mock("race_trainer_comment"), MOCK)]
    S["A10"] = [P("race_result_on_time/2023/202309030811.json", real("nk_result_on_time"), REAL)]
    S["A11"] = [P("race_result/2023/202309030811.json", real("nk_db_race_result"), REAL, "entries は先頭 2 頭、payoff・corner_passing も短縮")]
    S["A12"] = [P("race_result_lap/2023/202309030811.json", real("nk_db_per_horse_lap"), REAL)]
    S["A13"] = [P("horse_result/…json", mock("horse_result"), MOCK,
                  "race_history は先頭 2 件に短縮。実データは horse_profile ＋ horse_race_history（B01 に実サンプル）")]
    S["A14"] = [P("horse_pedigree_5gen/2019/2019105219.json", real("nk_horse_pedigree"), REAL, "ancestors は 62 件中 先頭 2 件に短縮")]
    S["A15"] = [P("horse_training/2019/2019105219.json", real("nk_horse_training"), REAL, "entries は先頭 2 件に短縮")]
    S["A16"] = [P("broodmare_mating/…json", mock("broodmare_mating"), MOCK)]
    S["A17"] = [P("jra_cushion/2026.json", mock("jra_cushion"), MOCK, "records は先頭 2 件に短縮")]
    S["A18"] = [P("smartrc_race/…json", mock("smartrc_race"), MOCK, "取得中止のカテゴリ。スキーマ上の最小形のみ")]

    S["B01"] = [P("race_shutuba_meta", real("nk_shutuba_race_meta"), REAL), P("race_result_meta", real("nk_db_race_info"), REAL),
                P("race_result_payoff", real("nk_db_payoff"), REAL), P("race_result_track", real("nk_db_track"), REAL),
                P("race_result_corner", real("nk_db_corner"), REAL), P("race_result_lap_times", real("nk_db_lap"), REAL),
                P("horse_profile", real("nk_horse_profile"), REAL),
                P("horse_race_history", real("nk_horse_history"), REAL, "race_history は先頭 2 件に短縮")]
    S["B02"] = [P("requirement_row_trace/race_…_nk_shutuba_entries.json", mock("requirement_row_trace"), MOCK)]
    hn_path = ROOT / "data/calculated_data/knowledge/horse_name_index.json"
    S["B03"] = [N("horse_name カテゴリ自体は save() 元が無く、馬名の検索は次の索引（data/calculated_data/knowledge/horse_name_index.json）で代替しています。")]
    if hn_path.exists():
        hn = jload("data/calculated_data/knowledge/horse_name_index.json")
        S["B03"].append(P("horse_name_index.json", hn, f"実データ（この PC の索引。{hn.get('total_horses', '?')} 頭）", "horses は先頭 2 件に短縮"))
    S["B04"] = [P("race_detail/…json", mock("race_detail"), MOCK, "entries は先頭 2 件に短縮"),
                P("race_performance/…json（スキーマ未定義・形式は暫定）", mock("race_performance"), MOCK, "実データ観測後に置き換える")]

    flat = _data_to_flat_rows(real("nk_db_race_result"), "race_result")
    cols = list(flat[0].keys())
    S["C01"] = [T("race_result_flat.parquet（先頭 3 行・先頭 14 列）", cols[:14], [[r.get(c) for c in cols[:14]] for r in flat[:3]], REAL,
                  f"全 {len(cols)} 列。export_tables の変換ロジック（_data_to_flat_rows）を実サンプルに適用した結果")]
    doc = (ROOT / "docs/html/data/data_features_reference.html").read_text(encoding="utf-8")

    def doc_table(section_id: str, max_rows: int = 3, max_cols: int | None = None) -> tuple[list, list]:
        i = doc.index(f'id="{section_id}"')
        m = re.search(r"<table.*?</table>", doc[i:i + 40000], re.S)
        rows = []
        for tr in re.findall(r"<tr.*?</tr>", m.group(0), re.S):
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
            if cells:
                rows.append(cells)
        head, body = rows[0], rows[1:1 + max_rows]
        return (head[:max_cols], [r[:max_cols] for r in body]) if max_cols else (head, body)

    h, b = doc_table("raw-flat-detail", 3, 10)
    DOCSRC = "実データ（data_features_reference.html に掲載の先頭行）"
    S["C02"] = [T("base_tbl/2020/shutuba.parquet（先頭 3 行）", h[:4], [r[:4] for r in b], DOCSRC, "4 キーのみ（race_id, horse_id, jockey_id, trainer_id）")]
    S["C03"] = [T("race_horse_tbl 系: raw flat 由来の列（先頭 3 行・先頭 10 列）", h, b, DOCSRC,
                  "race_tbl / race_horse_tbl / horse_tbl の全列は data_features_reference.html の「登録列一覧」参照")]
    h, b = doc_table("jt-stats-sample", 3, 12)
    S["C04"] = [T("jt_race_features.parquet（先頭 3 行・先頭 12 列）", h, b, DOCSRC, "空欄は履歴が無い（初出走など）ための欠損")]
    rr = real("nk_db_race_result")
    S["C05"] = [T("target/rank_tbl/2023/rank.parquet（先頭 3 行）", ["race_id", "horse_id", "rank"],
                  [[rr["race_id"], e["horse_id"], float(e["finish_position"])] for e in rr["entries"][:3]], REAL,
                  "build_rank_target と同じ定義（rank = finish_position を数値化）を実サンプルに適用")]
    ped = _flatten_pedigree_json(real("nk_horse_pedigree"), "2019105219")
    pcols = list(ped.columns)[:10]
    S["C06"] = [T("horse/ped_tbl/2019/2019105219.parquet（先頭 4 行・先頭 10 列）", pcols,
                  [[None if str(v) == "<NA>" else v for v in r] for r in ped[pcols].head(4).itertuples(index=False)], REAL,
                  f"全 {len(ped.columns)} 列（牡スロットのみのロング形式）。build_horse_entity_store の変換（_flatten_pedigree_json）を実サンプルに適用")]
    S["C07"] = [N("サンプルなし: layer_a_train.parquet は列数が多く、dev に未生成です（python3 -m src.pipeline.build_layer_a_dataset で学習PCに生成）。")]

    for k in ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"):
        S[k] = [N("サンプルなし: dev に未生成の成果物です（学習PCで生成）。生成コマンドは一覧の「取得元」列を参照。")]
    S["D01"] = [N("サンプルなし: 10gen は 5gen（A14）の祖先に 6〜10 世代を接木した JSON（via で接木元を追跡）。dev に未生成です。")]
    S["D09"] = [N("サンプルなし: data/calculated_data/{growth_curve,tracking_difficulty}/ は dev が空です（集計ジョブで学習PCに生成）。")]

    my = jload("data/calculated_data/knowledge/myostatin_genes.json")
    if isinstance(my.get("stallions"), dict):
        my["stallions"] = dict(list(my["stallions"].items())[:2])
    S["E01"] = [P("myostatin_genes.json", my, "実データ（手動作成ナレッジ。この PC）", "stallions は先頭 2 件に短縮")]
    S["E02"] = [P("_meta.json", jload("data/research/other_predictor_umauma/_meta.json"), "実データ（この PC）"),
                P("articles_manifest.json", jload("data/research/other_predictor_umauma/articles_manifest.json"), "実データ（この PC）", "先頭 2 件に短縮")]
    S["E03"] = [P("meta/person/jockey_00001.json", jload("data/calculated_data/meta/person/jockey_00001.json"),
                  "実データ（dev にあるサンプル。yearly_stats は空）")]
    cfg = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))
    S["E04"] = [P("config/settings.yaml（抜粋）", {k: cfg[k] for k in list(cfg)[:4]}, "設定ファイル（git 管理）", "先頭 4 セクションを抜粋。dict に変換して表示")]
    S["E05"] = [P("config/megu_predict_params.json", jload("config/megu_predict_params.json"), "設定ファイル（git 管理）")]
    env_keys: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            k, v = m.groups()
            env_keys[k] = "<秘密>" if re.search(r"SECRET|PASSWORD|PRIVATE_KEY|TOKEN|_pw$", k, re.I) else v
    S["E06"] = [P(".env（キー名の例。秘密の値は伏せる）", dict(list(env_keys.items())[:24]), "テンプレート .env.example から（値はダミー）",
                  "先頭 24 キー。.env 自体は git 管理外")]

    mods = [[str(p.relative_to(ROOT / "models")), f"{p.stat().st_size:,} B"] for p in sorted((ROOT / "models").rglob("*")) if p.is_file()]
    S["F01"] = [T("models/ のファイル", ["ファイル", "サイズ"], mods, "実データ（git 管理のモデル成果物）")]
    S["F02"] = [N("サンプルなし: アンサンブルの公開（publish_model の CLI 化 T-005）が未実装のため、latest.json の実物がありません。")]
    S["F03"] = [N("サンプルなし: MLflow の runs / artifacts は dev に無く、学習PC で生成されます（形式は MLflow の標準）。")]
    S["F04"] = [P("race_predictions/2026/…json", mock("race_predictions"), MOCK, "horses は先頭 2 頭に短縮。スキーマ未定義のため形式は暫定（実データ観測後に確定）")]
    S["F05"] = [P(f"{c}/…json", mock(c), MOCK, "スキーマ未定義のため形式は暫定")
                for c in ("tracking_difficulty", "final_odds_prediction", "finish_order_prediction")]

    from src.scraper.job_queue import ScrapeJobQueue

    try:
        job = ScrapeJobQueue._normalize_incoming_job(None, {"job_kind": "race", "target_id": "202605030211", "tasks": ["race_result"]})
        S["G01"] = [P("data/queue/scrape_queue.json の jobs[0]（投入時の正規化結果）", job, "コードの正規化結果（ScrapeJobQueue）",
                      '実行後は status / started_at / retry_count などが付く。ファイル全体は {"jobs": [...], "updated_at": ...}')]
    except Exception:  # noqa: BLE001
        S["G01"] = [N('サンプルなし: ジョブはキュー投入時に生成されます（data/queue/scrape_queue.json は {"jobs": [...], "updated_at": ...}）。')]
    S["G02"] = [N("サンプルなし: auto_scrape_status.json は cron 実行後に生成されます（dev にはありません）。")]
    S["G03"] = [N("サンプルなし: キャッシュ・ミラーは GCS の JSON と同一内容のコピーです（例は A 群を参照）。")]
    from src.db import models

    S["G04"] = [T("PostgreSQL のテーブル（SQLAlchemy モデルから）", ["テーブル", "列数", "先頭の列"],
                  [[t.name, len(t.columns), ", ".join(c.name for c in list(t.columns)[:7])] for t in models.Base.metadata.sorted_tables],
                  "コードのモデル定義（src/db/models）")]
    S["G05"] = [N("サンプルなし: Redis には整形済みの API レスポンス等が短時間キャッシュされます（キー名は src/api/cache/redis_cache.py）。")]
    return S


def render_panel(p: dict[str, Any]) -> str:
    if p["kind"] == "note":
        return f'<p class="smp-note">{E(p["text"])}</p>'
    head = f'<div class="smp-h"><b>{E(p["label"])}</b> <span class="smp-src">{E(p["source"])}</span></div>'
    note = f'<div class="smp-n">{E(p["note"])}</div>' if p.get("note") else ""
    if p["kind"] == "json":
        kind = "辞書型（dict）" if isinstance(p["data"], dict) else "リスト（list）"        # 実ファイルの最上位がリストのもの（articles_manifest.json 等）
        body = f'<div class="smp-t">{kind}</div><pre class="smp"><code>{E(json.dumps(p["data"], ensure_ascii=False, indent=2))}</code></pre>'
    else:
        th = "".join(f"<th>{E(str(c))}</th>" for c in p["cols"])
        tr = "".join("<tr>" + "".join(f"<td>{E('' if v is None else str(v))}</td>" for v in r) + "</tr>" for r in p["rows"])
        body = f'<div class="smp-t">テーブル</div><div class="wrap"><table class="smp"><tr>{th}</tr>{tr}</table></div>'
    return f'<div class="smp-p">{head}{note}{body}</div>'


def sample_row(panels: list[dict[str, Any]]) -> str:
    kinds = {"json": "辞書型 JSON", "table": "テーブル", "note": "サンプルなし"}
    first = next((p["kind"] for p in panels if p["kind"] != "note"), "note")
    n = sum(1 for p in panels if p["kind"] != "note")
    label = "サンプルなし" if first == "note" else f"サンプル（{kinds[first]}" + (f"・{n} 種" if n > 1 else "") + "）"
    return f'<tr class="sample"><td colspan="7"><details><summary>{label}</summary>' + "".join(render_panel(p) for p in panels) + "</details></td></tr>"


HINT = ('<p class="smp-hint">各データの行をクリックすると<b>サンプル</b>が折り畳みで開きます（JSON は辞書型、特徴量などは表）。'
        '<button type="button" id="smpAll">すべて開く</button><button type="button" id="smpNone">すべて閉じる</button><br>'
        '<small>サンプルの出所：<b>実データ</b>（2026-06-12 に実スクレイプして保存したもの。1 レース・1 頭分）／<b>dev モック</b>（架空。スキーマ適合）／'
        '設定ファイルなど。長いリストは先頭だけに短縮しています。「サンプルなし」は dev に無い・形式が未確定のものです。'
        '再生成: <code>python -m src.scripts.docs.embed_catalog_samples</code></small></p>')


def embed(path: Path = CAT) -> int:
    S = build_samples()
    s = path.read_text(encoding="utf-8")
    a, b = s.index('<h2 id="catalog">'), s.index('<h2 id="matrix">')
    sec = s[a:b]
    sec = re.sub(r'<tr class="sample">.*?</details></td></tr>', "", sec, flags=re.S)       # 入れ子の表の </tr> で止まらないよう行末まで
    sec = re.sub(r'<tr class="ds"( data-id="[^"]*")?>', "<tr>", sec)
    missing: list[str] = []

    def add(m: re.Match) -> str:
        rid = m.group(1)
        if rid not in S:
            missing.append(rid)
            return m.group(0)
        return m.group(0).replace("<tr>", f'<tr class="ds" data-id="{rid}">', 1) + sample_row(S[rid])

    sec = re.sub(r'<tr><td class="id">([A-G]\d\d[a-z]?)</td>.*?</tr>', add, sec, flags=re.S)
    if missing:
        raise SystemExit(f"サンプル定義が無い行: {missing}（build_samples に追加してください）")
    sec = re.sub(r'<p class="smp-hint">.*?</p>', "", sec, flags=re.S)
    head = '<h2 id="catalog">4. データセット一覧（用途・取得元・保存先）</h2>'
    sec = sec.replace(head, head + HINT, 1)
    s = s[:a] + sec + s[b:]
    s = re.sub(r"\n/\* smp-begin \*/.*?/\* smp-end \*/\n*(?=</style>)", "\n", s, flags=re.S)
    s = s.replace("</style>", "/* smp-begin */" + CSS + "/* smp-end */\n</style>", 1)
    s = re.sub(r"\n*<script>\n\(function\(\)\{\n  var rows=document.querySelectorAll\('tr.ds'\);.*?</script>\n", "", s, flags=re.S)
    s = s.replace("</body>", JS + "</body>", 1)
    path.write_text(s, encoding="utf-8")
    return sum(1 for _ in re.finditer(r'<tr class="ds"', s))


def main() -> int:
    n = embed()
    print(f"サンプルを埋め込みました: {n} 行 → {CAT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
