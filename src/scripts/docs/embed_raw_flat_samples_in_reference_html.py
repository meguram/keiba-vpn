#!/usr/bin/env python3
"""
`docs/html/data_features_reference.html` 内の埋め込みマーカーを更新する。

- `RAW_FLAT_SAMPLES_*` — raw flat 由来 columns の折り畳みサンプル
- `JT_STATS_REF_*` — `race_horse_tbl/<YYYY>/jt_race_features.parquet` サンプル（折り畳み）
- `KEYS_SAMPLE_*` — 出馬表ベースの ID・コース条件・例示列（結合キー列は HTML 上で強調）
- `POPULATION_EMBED_*` — registry custom 列の母集団カウント
- `COLUMN_STATS_EMBED_*` — custom 列ごとの行数・サイズ・欠損・min/max/mean 表
- `RP_WIDE_SAMPLE_*` — custom 列全結合の先頭行サンプル
- `REGISTRY_MERGED_SAMPLES_*` — custom / その他登録列のマージサンプル（折り畳み2段）
- `RACE_STORE_SAMPLE_*` — `race_performance/races/` の JSON 抜粋

  python scripts/docs/embed_raw_flat_samples_in_reference_html.py
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HTML_PATH = ROOT / "docs/html/data_features_reference.html"
FEATURES_ROOT = ROOT / "data/local/features"
REGISTRY_PATH = FEATURES_ROOT / "_registry.json"
RACE_STORE_ROOT = FEATURES_ROOT / "race_performance/races"
LEGACY_COLS = FEATURES_ROOT / "columns"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def resolve_jt_race_features_sample_path(root: Path = FEATURES_ROOT) -> Path | None:
    """年別 ``race_horse_tbl/<YYYY>/jt_race_features.parquet`` または旧単一ファイル。"""
    rhd = root / "race_horse_tbl"
    if not rhd.is_dir():
        return None
    flat = rhd / "jt_race_features.parquet"
    if flat.is_file():
        return flat
    for y in sorted(p.name for p in rhd.iterdir() if p.is_dir() and len(p.name) == 4 and p.name.isdigit()):
        p = rhd / y / "jt_race_features.parquet"
        if p.is_file():
            return p
    return None


from src.pipeline.features.raw_table_features import (  # noqa: E402
    INDEX_SPECS,
    INDEX_SPEED_RECENT_EXPANDED_SPECS,
    PADDOCK_SPECS,
    SHUTUBA_ID_PRESENCE_STORE_NAMES,
    SHUTUBA_PAST_SPECS,
    SHUTUBA_SPECS,
    TRAINER_COMMENT_SPECS,
)


def esc(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    if len(s) > 140:
        s = s[:137] + "…"
    return html.escape(s)


def is_join_key_column(name: str) -> bool:
    """訓練結合スパインで重要な列（HTML サンプル表の強調表示用）。"""
    if name in ("race_id", "horse_id", "horse_number", "jockey_id", "trainer_id"):
        return True
    if name == "shutuba_venue_code":
        return True
    if name in (
        "shutuba_surface",
        "shutuba_distance",
        "shutuba_course_type",
        "shutuba_direction",
    ):
        return True
    if name in ("shutuba_horse_id", "shutuba_jockey_id", "shutuba_trainer_id"):
        return True
    if name.endswith("_present") and "id" in name:
        return True
    if name in ("jt_row_jockey_id", "jt_row_trainer_id"):
        return True
    return False


def table_html(df: pd.DataFrame, max_cols: int = 10, *, mark_join_keys: bool = True) -> str:
    names = list(df.columns)
    if mark_join_keys:
        keys = [c for c in names if is_join_key_column(c)]
        rest = [c for c in names if c not in keys]
        names = keys + rest
    cols = names[:max_cols]
    parts = ['<table class="sample-table"><thead><tr>']
    for c in cols:
        if mark_join_keys and is_join_key_column(c):
            parts.append(f'<th class="key-col">{html.escape(c)}</th>')
        else:
            parts.append(f"<th>{html.escape(c)}</th>")
    parts.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        parts.append("<tr>")
        for c in cols:
            if mark_join_keys and is_join_key_column(c):
                parts.append(f'<td class="cell-long key-col">{esc(r[c])}</td>')
            else:
                parts.append(f'<td class="cell-long">{esc(r[c])}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def parquet_path_for_stem(reg: dict, stem: str) -> Path | None:
    meta = reg.get(stem)
    if isinstance(meta, dict) and meta.get("rel_path"):
        p = FEATURES_ROOT / meta["rel_path"]
        if p.is_file():
            return p
    leg = LEGACY_COLS / f"{stem}.parquet"
    if leg.is_file():
        return leg
    return None


def merge_sample(stems: list[str], reg: dict, n: int = 3) -> pd.DataFrame:
    stems_ok = [s for s in stems if parquet_path_for_stem(reg, s)]
    if not stems_ok:
        return pd.DataFrame()
    p0 = parquet_path_for_stem(reg, stems_ok[0])
    assert p0 is not None
    base = pd.read_parquet(p0).head(n)
    for s in stems_ok[1:]:
        p = parquet_path_for_stem(reg, s)
        if p is None:
            continue
        o = pd.read_parquet(p)
        meta = reg.get(s) or {}
        mk = meta.get("merge_keys") if isinstance(meta, dict) else None
        if not mk:
            mk = [c for c in ("race_id", "horse_id", "jockey_id", "trainer_id") if c in base.columns and c in o.columns]
        on = [k for k in mk if k in base.columns and k in o.columns]
        if not on:
            continue
        base = base.merge(o, on=on, how="left")
    return base


def merge_sample_safe(stems: list[str], n: int = 3) -> pd.DataFrame:
    reg = load_registry()
    stems = [s for s in stems if parquet_path_for_stem(reg, s)]
    if not stems:
        return pd.DataFrame()
    return merge_sample(stems, reg, n)


def load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_parquet_keys_in_order(reg: dict) -> list[str]:
    out: list[str] = []
    for k, v in reg.items():
        if not isinstance(v, dict):
            continue
        if parquet_path_for_stem(reg, k) is not None:
            out.append(k)
    return out


def custom_registry_stems(reg: dict) -> list[str]:
    return [k for k in registry_parquet_keys_in_order(reg) if reg[k].get("source") == "custom"]


def other_registry_stems(reg: dict) -> list[str]:
    keys = registry_parquet_keys_in_order(reg)
    cust = set(custom_registry_stems(reg))
    others = [k for k in keys if k not in cust]

    def sort_key(x: str) -> tuple[int, str]:
        if x.startswith("shutuba_"):
            return (0, x)
        if x.startswith("idx_"):
            return (1, x)
        return (2, x)

    return sorted(others, key=sort_key)


def fmt_float_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    try:
        if pd.isna(v):
            return ""
    except TypeError:
        pass
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return ""
        return f"{x:.4f}"
    except (TypeError, ValueError):
        return "—"


def pq_rows(stem: str) -> int:
    reg = load_registry()
    p = parquet_path_for_stem(reg, stem)
    return pq.read_metadata(p).num_rows if p is not None and p.is_file() else 0


def build_keys_sample_fragment() -> str:
    """出馬表ベースの ID・コース条件・例示値（訓練結合スパインのサンプル）。"""
    stems = [
        "shutuba",
        "shutuba_distance",
        "shutuba_surface",
        "shutuba_venue_code",
        "shutuba_race_horse",
        "shutuba_horse",
    ]
    reg = load_registry()
    if parquet_path_for_stem(reg, "idx_speed_avg") is not None:
        stems.append("idx_speed_avg")
    df = merge_sample_safe(stems, 3)
    if df.empty:
        return (
            '    <p class="small">（出馬表系 <span class="mono">shutuba_*</span> の Parquet が不足しています。'
            '<span class="mono">python -m src.pipeline.register_raw_table_features</span> で生成してください。）</p>'
        )
    tbl = table_html(df, max_cols=min(16, len(df.columns)))
    return "\n".join(
        [
            '    <details class="raw-collapsible" id="keys-sample-inner">',
            "      <summary>予測ベース行のキー例（<span class=\"mono\">race_id</span>・<span class=\"mono\">horse_id</span>・"
            "<span class=\"mono\">shutuba_*</span> の ID・競馬場コード・芝ダ・距離・例示列）</summary>",
            '      <div class="raw-collapsible-body">',
            '        <p class="small">水色見出しの列は結合キー（エンティティ ID・競馬場・コース条件・距離など）。'
            " <span class=\"mono\">*_present</span> は ID の有効フラグ（<span class=\"mono\">pipeline/id_value_policy.py</span>）。"
            " <span class=\"mono\">place_id</span> 設計上の競馬場キーは本データでは <span class=\"mono\">shutuba_venue_code</span> に対応。</p>",
            f'        <div class="table-scroll">{tbl}</div>',
            "      </div>",
            "    </details>",
        ]
    )


def build_population_embed_fragment() -> str:
    reg = load_registry()
    stems = custom_registry_stems(reg)
    if not stems:
        return (
            '      <ul class="compact" style="margin-bottom:0">\n'
            '        <li>（<span class="mono">source=custom</span> の Parquet がブロック配下に見つかりません）</li>\n'
            "      </ul>"
        )
    n_files = len(stems)
    p0 = parquet_path_for_stem(reg, stems[0])
    assert p0 is not None
    n_rows = pq.read_metadata(p0).num_rows
    df_r = pd.read_parquet(p0, columns=["race_id"])
    n_races = int(df_r["race_id"].nunique())
    try:
        dfn = pd.read_parquet(p0, columns=["horse_id"])
        hs = dfn["horse_id"].dropna().astype(str)
        lo_s, hi_s = (hs.min(), hs.max()) if len(hs) else ("—", "—")
    except Exception:
        lo_s, hi_s = "—", "—"
    return "\n".join(
        [
            '      <ul class="compact" style="margin-bottom:0">',
            f"        <li>ファイル数: <strong>{n_files}</strong>（<span class=\"mono\">_registry.json</span> の <span class=\"mono\">source=custom</span> かつ Parquet 存在）</li>",
            f"        <li>各行ファイル共通の行数: <strong>{n_rows:,}</strong></li>",
            f"        <li>ユニーク <span class=\"mono\">race_id</span> 数: <strong>{n_races:,}</strong></li>",
            f"        <li><span class=\"mono\">horse_id</span> の辞書順範囲: <strong>{html.escape(lo_s)}</strong> ～ <strong>{html.escape(hi_s)}</strong></li>",
            "      </ul>",
        ]
    )


def build_column_stats_embed_fragment() -> str:
    reg = load_registry()
    stems = custom_registry_stems(reg)
    rows_html: list[str] = []
    for stem in stems:
        p = parquet_path_for_stem(reg, stem)
        if p is None or not p.is_file():
            continue
        n = pq.read_metadata(p).num_rows
        size_kib = p.stat().st_size / 1024
        s = pd.read_parquet(p, columns=[stem])[stem]
        missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            vmin = fmt_float_cell(float(s.min()))
            vmax = fmt_float_cell(float(s.max()))
            vmean = fmt_float_cell(float(s.mean()))
        else:
            vmin = vmax = vmean = "—"
        rows_html.append(
            "<tr>"
            f'<td class="mono">{html.escape(stem)}</td>'
            f'<td class="num">{n:,}</td>'
            f'<td class="num">{size_kib:.1f} KiB</td>'
            f'<td class="num">{missing:,}</td>'
            f'<td class="num">{html.escape(vmin)}</td>'
            f'<td class="num">{html.escape(vmax)}</td>'
            f'<td class="num">{html.escape(vmean)}</td>'
            "</tr>"
        )
    if not rows_html:
        return '        <p class="small">（統計対象の custom 列がありません）</p>'
    inner = "\n          ".join(rows_html)
    return f"""        <div class="table-scroll">
      <table class="dense">
        <thead>
          <tr>
            <th>ファイル（値列）</th>
            <th>行数</th>
            <th>サイズ</th>
            <th>欠損</th>
            <th>min</th>
            <th>max</th>
            <th>mean</th>
          </tr>
        </thead>
        <tbody>
          {inner}
        </tbody>
      </table>
    </div>"""


def build_rp_wide_sample_fragment() -> str:
    reg = load_registry()
    stems = custom_registry_stems(reg)
    if not stems:
        return '    <p class="small">（custom 列のマージサンプルを生成できません）</p>'
    df = merge_sample_safe(stems, 3)
    if df.empty:
        return '    <p class="small">（マージ結果が空です）</p>'
    ncols = len(df.columns)
    tbl = table_html(df, max_cols=min(48, ncols))
    note = (
        f"registry の <span class=\"mono\">source=custom</span> {len(stems)} 列をレジストリの <span class=\"mono\">merge_keys</span>（通常 <span class=\"mono\">race_id, horse_id</span>）で結合。"
        f"表は先頭 {min(48, ncols)} 列まで表示（全 {ncols} 列）。"
        " 訓練母表では <span class=\"mono\">race_id + horse_id</span>（および出馬表の <span class=\"mono\">jockey_id</span> / <span class=\"mono\">trainer_id</span>）で"
        "外部特徴へ join する方針です。"
    )
    return "\n".join(
        [
            '    <details class="raw-collapsible" id="rp-wide-sample">',
            '      <summary>レースパフォーマンス登録列のマージサンプル（先頭3行・横スクロール）</summary>',
            '      <div class="raw-collapsible-body">',
            f'        <p class="small">{note}</p>',
            f'        <div class="table-scroll">{tbl}</div>',
            "      </div>",
            "    </details>",
        ]
    )


def build_registry_merged_samples_fragment() -> str:
    reg = load_registry()
    custom = custom_registry_stems(reg)
    other = other_registry_stems(reg)[:22]
    parts: list[str] = []

    if custom:
        df_c = merge_sample_safe(custom, 3)
        tbl_c = table_html(df_c, max_cols=min(40, len(df_c.columns))) if not df_c.empty else "<p>（空）</p>"
        parts.extend(
            [
                '    <details class="raw-collapsible" id="registry-samples-custom">',
                f'      <summary>登録列マージ（<span class="mono">source=custom</span>・{len(custom)} 列・先頭3行）</summary>',
                '      <div class="raw-collapsible-body">',
                '        <p class="small">水色＝<span class="mono">race_id</span>・<span class="mono">horse_id</span>。'
                "<span class=\"mono\">shutuba_horse_id</span> 等で馬・騎手・調教師の集計特徴へ二次 join する前提。</p>",
                f'        <div class="table-scroll">{tbl_c}</div>',
                "      </div>",
                "    </details>",
                "",
            ]
        )
    else:
        parts.append('    <p class="small">（custom 登録列の Parquet がありません）</p>\n')

    if other:
        df_o = merge_sample_safe(other, 2)
        ncol = len(df_o.columns) if not df_o.empty else 0
        tbl_o = table_html(df_o, max_cols=min(28, ncol)) if not df_o.empty else "<p>（空）</p>"
        parts.extend(
            [
                '    <details class="raw-collapsible" id="registry-samples-other">',
                f'      <summary>登録列マージ（出馬表・指数・パドック等・先頭 {len(other)} stem・先頭2行）</summary>',
                '      <div class="raw-collapsible-body">',
                (
                    '        <p class="small">先頭 stem は <span class="mono">shutuba_*</span> → <span class="mono">idx_*</span> 優先。'
                    "指数系は行が少ないレースでは left join で欠損になります。"
                    ' 水色列は結合キー（<span class="mono">race_id</span>・馬番・ID・競馬場・コース条件）。</p>'
                ),
                f'        <div class="table-scroll">{tbl_o}</div>',
                "      </div>",
                "    </details>",
            ]
        )
    else:
        parts.append('    <p class="small">（custom 以外の登録列 Parquet がありません）</p>')

    return "\n".join(parts)


def build_race_store_sample_fragment() -> str:
    if not RACE_STORE_ROOT.is_dir():
        return (
            '    <p class="small">（<span class="mono">data/features/race_performance/races/</span> がありません）</p>'
        )
    picked: Path | None = None
    for year_dir in sorted(RACE_STORE_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        js = sorted(year_dir.glob("*.json"))
        if js:
            picked = js[0]
            break
    if picked is None:
        return '    <p class="small">（JSON ファイルが見つかりません）</p>'
    rel = picked.relative_to(ROOT)
    try:
        obj = json.loads(picked.read_text(encoding="utf-8"))
        full_text = json.dumps(obj, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, OSError):
        return f'    <p class="small">（読み取り失敗: <span class="mono">{html.escape(str(rel))}</span>）</p>'
    max_len = 14_000
    full_len = len(full_text)
    truncated = full_len > max_len
    text = full_text[:max_len] + "\n…（以下省略）" if truncated else full_text
    pre_body = html.escape(text)
    omit = "・省略あり" if truncated else ""
    return "\n".join(
        [
            '    <details class="raw-collapsible" id="race-store-json-sample">',
            f'      <summary>レース JSON サンプル（<span class="mono">{html.escape(str(rel))}</span>）</summary>',
            '      <div class="raw-collapsible-body">',
            f"        <p class=\"small\">最大 {max_len:,} 文字まで表示（全長 {full_len:,} 文字{omit}）。</p>",
            f'        <pre class="mono-wrap">{pre_body}</pre>',
            "      </div>",
            "    </details>",
        ]
    )


def build_fragment() -> str:
    reg = load_registry()
    sh_stems = [
        "shutuba",
        "shutuba_distance",
        "shutuba_surface",
        "shutuba_venue_code",
        "shutuba_grade",
        "shutuba_race_horse",
        "shutuba_horse",
        "shutuba_race_jockey",
        "shutuba_race_trainer",
    ]
    df_sh = merge_sample(sh_stems, reg, 3)

    idx_stems = ["idx_speed_avg", "idx_speed_recent_1", "idx_speed_recent_2", "idx_speed_recent_3"]
    df_idx = merge_sample(idx_stems, reg, 3)

    pp = parquet_path_for_stem(reg, "past_past_races")
    pt = parquet_path_for_stem(reg, "past_training")
    if pp is not None and pt is not None:
        df_past = pd.read_parquet(pp).head(2)
        df_past = df_past.merge(pd.read_parquet(pt), on=["race_id", "horse_id"], how="left")
    else:
        df_past = pd.DataFrame()

    df_pad = merge_sample(["paddock_rank", "paddock_comment"], reg, 3)

    df_tc = merge_sample(
        ["tcomment_evaluation", "tcomment_trainer_name", "tcomment_questioner", "tcomment_comment"],
        reg,
        2,
    )

    shutuba_list = ", ".join(f'<span class="mono">{s.store_name}</span>' for s in SHUTUBA_SPECS)
    presence_list = ", ".join(f'<span class="mono">{n}</span>' for n in SHUTUBA_ID_PRESENCE_STORE_NAMES)
    index_list = ", ".join(
        f'<span class="mono">{s.store_name}</span>'
        for s in (INDEX_SPECS + INDEX_SPEED_RECENT_EXPANDED_SPECS)
    )

    n_sh, n_idx, n_past, n_pad, n_tc = map(
        pq_rows,
        ["shutuba_distance", "idx_speed_avg", "past_past_races", "paddock_rank", "tcomment_comment"],
    )

    lines = [
        '    <h3 id="raw-flat-detail">追加 Parquet（raw flat 由来）の詳細とサンプル</h3>',
        '    <p class="small">登録は <span class="mono">python -m src.pipeline.register_raw_table_features</span>。'
        "サンプルはローカル <span class=\"mono\">data/features/*_tbl/*.parquet</span>（レジストリ参照）の先頭行を結合したものです"
        "（環境・再登録で変わります）。折り畳みを開くと表が表示されます。</p>",
        "",
        '    <details class="raw-collapsible">',
        "      <summary>出馬表系 <span class=\"mono\">shutuba_*</span>（"
        f"{len(SHUTUBA_SPECS)} stem・レース条件サンプル行数 {n_sh:,}）</summary>",
        '      <div class="raw-collapsible-body">',
        "        <p><strong>ソース</strong> <span class=\"mono\">race_shutuba_flat</span>。"
        "<span class=\"mono\">base_tbl/shutuba</span> は <span class=\"mono\">race_id, horse_id, jockey_id, trainer_id</span> のみ。"
        "馬名・血統等は <span class=\"mono\">horse_tbl/shutuba_horse</span>、当該レースの馬行は <span class=\"mono\">race_horse_tbl/shutuba_race_horse</span>、"
        "レース条件列は <span class=\"mono\">race_tbl</span>（<span class=\"mono\">race_id</span>）。"
        "訓練結合では <span class=\"mono\">shutuba_horse_id</span> / <span class=\"mono\">shutuba_jockey_id</span> / <span class=\"mono\">shutuba_trainer_id</span> および"
        " <span class=\"mono\">shutuba_venue_code</span>＋芝ダ（<span class=\"mono\">shutuba_surface</span>）＋距離（<span class=\"mono\">shutuba_distance</span>）で外部特徴に繋ぐ。"
        "<strong>除外</strong>: 単勝オッズ・人気（LayerA 方針）。</p>",
        f'        <p class="small mono-wrap">stem 一覧: {shutuba_list}</p>',
        "        <p class=\"small\"><strong>文字列 ID の欠損</strong>（<span class=\"mono\">horse_id, jockey_id, trainer_id</span>）: "
        "空・<span class=\"mono\">&quot;0&quot;</span> は <strong>null</strong>（0 埋めしない）。"
        f'有効可否は <span class="mono">race_horse_tbl/shutuba_race_horse</span> 内の列 {presence_list}（<span class="mono">int8</span>、1=有効・0=欠損/プレースホルダ）で表す。</p>',
        '        <p class="small"><strong>サンプル</strong>（水色＝結合キー列。各 ID・競馬場コード・芝ダ・距離・馬名・斤量 等）</p>',
        f'        <div class="table-scroll">{table_html(df_sh)}</div>',
        "      </div>",
        "    </details>",
        "",
        '    <details class="raw-collapsible">',
        "      <summary>指数系 <span class=\"mono\">idx_*</span>（"
        f"{len(INDEX_SPECS) + len(INDEX_SPEED_RECENT_EXPANDED_SPECS)} stem・{n_idx:,} 行）</summary>",
        '      <div class="raw-collapsible-body">',
        "        <p><strong>ソース</strong> <span class=\"mono\">race_index_flat</span>。"
        "指数対象外の馬は行が無く、出馬表より行数が少ない点に注意。"
        " <span class=\"mono\">idx_speed_recent_{1,2,3}</span> は近3走セル（元リストのプレースホルダ 0 は null）。"
        " <span class=\"mono\">time_index_m</span> はストアに含めない。</p>",
        f'        <p class="small mono-wrap">stem 一覧: {index_list}</p>',
        '        <p class="small"><strong>サンプル</strong></p>',
        f'        <div class="table-scroll">{table_html(df_idx)}</div>',
        "      </div>",
        "    </details>",
        "",
        '    <details class="raw-collapsible">',
        f"      <summary>過去走・調教ブロック <span class=\"mono\">past_*</span>（{len(SHUTUBA_PAST_SPECS)} 列・{n_past:,} 行）</summary>",
        '      <div class="raw-collapsible-body">',
        "        <p><strong>ソース</strong> <span class=\"mono\">race_shutuba_past_flat</span>。"
        "生テキスト／JSON 相当。学習前にパース・集約する前提。</p>",
        '        <p class="small"><strong>サンプル</strong>（セルは 140 字で省略）</p>',
        f'        <div class="table-scroll">{table_html(df_past, max_cols=4)}</div>',
        "      </div>",
        "    </details>",
        "",
        '    <details class="raw-collapsible">',
        f"      <summary>パドック <span class=\"mono\">paddock_*</span>（{len(PADDOCK_SPECS)} 列・{n_pad:,} 行）</summary>",
        '      <div class="raw-collapsible-body">',
        "        <p><strong>ソース</strong> <span class=\"mono\">race_paddock_flat</span>。"
        "登録時に <code>drop_duplicates(race_id, horse_id, keep='last')</code>。"
        "2020 年ファイルは馬列なしのためスキップ。</p>",
        f'        <div class="table-scroll">{table_html(df_pad)}</div>',
        "      </div>",
        "    </details>",
        "",
        '    <details class="raw-collapsible">',
        f"      <summary>厩舎コメント <span class=\"mono\">tcomment_*</span>（{len(TRAINER_COMMENT_SPECS)} 列・{n_tc:,} 行）</summary>",
        '      <div class="raw-collapsible-body">',
        "        <p><strong>ソース</strong> <span class=\"mono\">race_trainer_comment_flat</span>。"
        "キー正規化済み（欠損キー行は落ちる）。</p>",
        '        <p class="small"><strong>サンプル</strong></p>',
        f'        <div class="table-scroll">{table_html(df_tc, max_cols=6)}</div>',
        "      </div>",
        "    </details>",
    ]
    return "\n".join(lines)


def build_jt_stats_fragment() -> str:
    """騎手・調教師 jt_race_features のサンプル表（行数メタ付き・折り畳み）。"""
    jt_path = resolve_jt_race_features_sample_path()
    if jt_path is None or not jt_path.is_file():
        return "\n".join(
            [
                '    <details class="raw-collapsible" id="jt-stats-sample-table">',
                "      <summary><code>jt_race_features.parquet</code> サンプル（未生成）</summary>",
                '      <div class="raw-collapsible-body">',
                "        <p class=\"small\">（未生成）<span class=\"mono\">data/features/race_horse_tbl/年/jt_race_features.parquet</span> を"
                " <span class=\"mono\">python -m src.pipeline.build_jockey_trainer_stats</span> で生成してください。</p>",
                "      </div>",
                "    </details>",
            ]
        )
    meta = pq.read_metadata(jt_path)
    n = meta.num_rows
    schema = pq.read_schema(jt_path)
    names = {f.name for f in schema}
    preferred = [
        "race_id",
        "horse_id",
        "jt_row_jockey_id",
        "jt_row_trainer_id",
        "jt_result_date",
        "jt_race_datetime",
        "jk_prior_all_starts",
        "jk_prior_all_win_rate",
        "jk_prior_all_avg_pass_first",
        "jk_roll10_avg_pass_first",
        "tr_prior_all_starts",
        "tr_prior_all_win_rate",
    ]
    cols = [c for c in preferred if c in names]
    if not cols:
        cols = list(schema.names)[:10]
    df = pd.read_parquet(jt_path, columns=cols).head(3)
    tbl = table_html(df, max_cols=len(df.columns))
    return "\n".join(
        [
            '    <details class="raw-collapsible" id="jt-stats-sample-table">',
            f'      <summary><code>jt_race_features.parquet</code> サンプル（{n:,} 行・代表列）</summary>',
            '      <div class="raw-collapsible-body">',
            f"        <p class=\"small\"><strong>行数</strong>（Parquet メタ）: {n:,}</p>",
            "        <p class=\"small\">母行への結合キーは <span class=\"mono\">race_id + horse_id</span>。"
            " <span class=\"mono\">jt_row_jockey_id</span> / <span class=\"mono\">jt_row_trainer_id</span> は出馬表の"
            " <span class=\"mono\">jockey_id</span> / <span class=\"mono\">trainer_id</span> と照合し、別途 <span class=\"mono\">jockey_id</span> キーの騎手特徴・"
            "<span class=\"mono\">trainer_id</span> キーの調教師特徴へ繋げる想定。</p>",
            f'        <div class="table-scroll">{tbl}</div>',
            "      </div>",
            "    </details>",
        ]
    )


def _sub_block(text: str, label: str, replacement: str) -> str:
    begin, end = label.split("|", 1)
    pat = re.compile(
        rf"(<!-- {re.escape(begin)} -->\n)(.*?)(\n    <!-- {re.escape(end)} -->)",
        re.DOTALL,
    )
    if not pat.search(text):
        print(f"note: markers {begin}/{end} not found; skip", file=sys.stderr)
        return text
    return pat.sub(r"\1" + replacement + r"\3", text, count=1)


def main() -> int:
    if not HTML_PATH.exists():
        print("missing", HTML_PATH, file=sys.stderr)
        return 1
    text = HTML_PATH.read_text(encoding="utf-8")
    pat = re.compile(
        r"(<!-- RAW_FLAT_SAMPLES_BEGIN -->\n)(.*?)(\n    <!-- RAW_FLAT_SAMPLES_END -->)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        print("markers RAW_FLAT_SAMPLES_BEGIN/END not found", file=sys.stderr)
        return 1
    new_text = pat.sub(r"\1" + build_fragment() + r"\3", text, count=1)

    blocks: list[tuple[str, str]] = [
        ("JT_STATS_REF_BEGIN|JT_STATS_REF_END", build_jt_stats_fragment()),
        ("KEYS_SAMPLE_BEGIN|KEYS_SAMPLE_END", build_keys_sample_fragment()),
        ("POPULATION_EMBED_BEGIN|POPULATION_EMBED_END", build_population_embed_fragment()),
        ("COLUMN_STATS_EMBED_BEGIN|COLUMN_STATS_EMBED_END", build_column_stats_embed_fragment()),
        ("RP_WIDE_SAMPLE_BEGIN|RP_WIDE_SAMPLE_END", build_rp_wide_sample_fragment()),
        ("REGISTRY_MERGED_SAMPLES_BEGIN|REGISTRY_MERGED_SAMPLES_END", build_registry_merged_samples_fragment()),
        ("RACE_STORE_SAMPLE_BEGIN|RACE_STORE_SAMPLE_END", build_race_store_sample_fragment()),
    ]
    for label, frag in blocks:
        new_text = _sub_block(new_text, label, frag)

    HTML_PATH.write_text(new_text, encoding="utf-8")
    print("updated", HTML_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
