"""めぐコラム『通説シリーズ』のうち、通説の検証（folklore_verify.py）で既に検定した型を、検証の結果からそのまま短い記事にする。

folklore.py は通説ごとにコラム側で条件を書いて集計するが、こちらは検証の章（2016〜2025 年・クラスタ頑健・発見→確認の 2 段の補正）の
結果（<FOLKLORE_OUT>/tests.parquet・meta.json）を読むだけなので、定義・見込み・格が検証と完全に同じになる。

1 記事 = 1 グループ（folklore_groups.py）。主の型（手で指定）の判定が記事の判定になり、同じグループの型は表に格のまま並べる。
  数字の囲み 3 つ / 結論の文 / 表（グループの型ごとの走数・実際・見込み・差・格）/ グラフ 1 枚（主の型を切り口で分けたもの）/ 関連コラムへのリンク
判定は folklore.tier_verdict: S・A・B は通説の向きなら傾向あり・逆なら逆の傾向、C は条件付き（逆なら逆の傾向）、格なしは傾向なし。
表の走数・実際・見込み・差（発見）は発見期間（全年あれば 2016〜2022 年）、差（確認）は確認期間（2023〜2025 年）。

使い方: python -m src.research.columns.folklore（通説の集計の記事に続けて、こちらの記事も書く。--only で絞れる）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.columns import folklore as F

VERIFY_DIR = Path(os.environ.get("FOLKLORE_OUT", "/home/jovyan/work/folklore_out"))
MIN_CELL = 300
TABLE_COLUMNS = ["型（定義）", "通説の向き", "走数", "実際", "見込み", "差（発見）", "差（確認）", "格"]

# 主の型を切り口で分けるときの優先順（走数 MIN_CELL 以上のセルが 3 つ以上ある最初のもの）と、セルの並び
DIM_PRIORITY = ["面×距離帯", "クラス", "季節", "人気帯", "頭数", "年齢", "性", "面"]
KEY_ORDER = {
    "面×距離帯": ["芝~1300", "芝1400", "芝1600", "芝1800-2200", "芝2300~", "ダ~1300", "ダ1400", "ダ1600", "ダ1800-2200", "ダ2300~"],
    "クラス": ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "重賞"],
    "季節": ["春", "夏", "秋", "冬"],
    "人気帯": ["1-3人気", "4-6人気", "7-9人気", "10人気~"],
    "頭数": ["~10頭", "11-14頭", "15頭~"],
    "年齢": ["2歳", "3歳", "4歳以上"],
    "性": ["牡・セ", "牝"],
    "面": ["芝", "ダ"],
}
DIM_LABEL = {"面×距離帯": "面×距離", "クラス": "クラス", "季節": "季節", "人気帯": "人気", "頭数": "頭数", "年齢": "年齢", "性": "性", "面": "芝・ダート"}
# 記事にしない型: 偽の型（Z）、オッズの時系列が要るもの（X01〜X03）、レース後の値を使うもの（J06）
EXCLUDED = ("Z", "X01", "X02", "X03", "J06")


@dataclass
class Group:
    slug: str
    title: str                       # 「〜は走る？」の問い。結論は自動で後ろに付ける
    label: str                       # 主の型の短い呼び名（数字の囲みの見出し）
    claim: str                       # 検証した通説
    categories: list[str]
    main: str                        # 主の型の hid（記事の判定はこの型の格）
    hids: list[str]                  # 記事に載せる型（主を含む。表の並びはこの順）
    question: str = ""               # 調べたこと（1 文）
    links: list[tuple[str, str]] = field(default_factory=list)   # 関連する既存コラム (slug, 一言)

    def all_hids(self) -> list[str]:
        return self.hids if self.main in self.hids else [self.main, *self.hids]


def folklore_slugs() -> set[str]:
    """通説シリーズの記事の slug（集計で作る記事 ＋ 検証から作る記事）。短くする処理やテストが、通説の記事を見分けるのに使う。"""
    from src.research.columns.folklore_groups import GROUPS
    from src.research.columns.folklore_specs import SPECS
    return {s.slug for s in SPECS} | {g.slug for g in GROUPS}


def load_results() -> tuple[pd.DataFrame, dict[str, dict]]:
    tests = pd.read_parquet(VERIFY_DIR / "tests.parquet")
    meta = {m["hid"]: m for m in json.loads((VERIFY_DIR / "meta.json").read_text(encoding="utf-8"))}
    return tests, meta


def _years(ys: list[int] | None) -> str:
    return f"{min(ys)}〜{max(ys)}年" if ys else "–"


def _row(tests: pd.DataFrame, hid: str) -> pd.Series:
    r = tests[(tests["hid"] == hid) & (tests["dim"] == "全体")]
    if r.empty:
        raise KeyError(f"{hid}: 検証の結果に『全体』の行が無い（検定できなかった型は記事にできない）")
    return r.iloc[0]


def _tier(r: pd.Series) -> str:
    return str(r["adj_tier"])


def _tier_word(t: str) -> str:
    return f"格 {t}" if t != "無" else "格なし"


def headline(direction: int, verdict: str, reversed_: bool) -> str:
    if reversed_:
        return "むしろ人気ほど走らない" if direction > 0 else "むしろ人気以上に走る"
    if verdict == "supported":
        return "人気以上に走る" if direction > 0 else "人気ほど走らない"
    if verdict == "partial":
        return "傾向はあるが弱い"
    return "人気どおりで上積みなし"


def tip(direction: int, verdict: str, reversed_: bool) -> str:
    if reversed_:
        return ("通説とは逆に、人気になりすぎています。人気で当てはまるなら、少し割り引くくらいがちょうどよさそうです。" if direction > 0
                else "通説とは逆に、人気より走っています。この理由で割り引くのは損になりやすいです。")
    if verdict == "partial":
        return "差は小さく、確かさも低い（格 C）です。印を動かすなら小さめに。"
    table = F.DEFAULT_TIPS if direction > 0 else F.DEFAULT_TIPS_NEG
    return table["supported" if verdict == "supported" else "not_supported"]


def _cells(tests: pd.DataFrame, hid: str) -> tuple[str, list[dict]] | None:
    """主の型を切り口で分けたセル（走数が十分な切り口の最初のもの）。"""
    for dim in DIM_PRIORITY:
        rows = tests[(tests["hid"] == hid) & (tests["dim"] == dim) & (tests["n1"] >= MIN_CELL)]
        if len(rows) < 3:
            continue
        order = {k: i for i, k in enumerate(KEY_ORDER[dim])}
        rows = rows.assign(_o=rows["key"].map(order)).dropna(subset=["_o"]).sort_values("_o")
        if len(rows) < 3:
            continue
        return dim, [{"label": str(r["key"]), "a": float(r["y3_1"]), "e": float(r["p_mkt_1"]), "n": int(r["n1"]), "p_exp": None} for _, r in rows.iterrows()]
    return None


def _grade_cell(r: pd.Series) -> str:
    """表の『格』: 格が付いたら、通説どおりか逆かも添える。"""
    t = _tier(r)
    return "格なし" if t == "無" else f"格 {t}・{'通説どおり' if bool(r['adj_agree']) else '通説と逆'}"


def _table_row(meta: dict, r: pd.Series) -> list[str]:
    conf = r["adj_c"]
    return [str(meta["how"]), "走る" if int(meta["dir"]) > 0 else "割引", f"{int(r['n1']):,}", F.pct(float(r["y3_1"])), F.pct(float(r["p_mkt_1"])),
            F.pt(float(r["adj"])), "–" if pd.isna(conf) else F.pt(float(conf)), _grade_cell(r)]


def build(g: Group, tests: pd.DataFrame, meta: dict[str, dict], published: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """記事・ガイド・集計結果（results.json 用）を作る。"""
    m = _row(tests, g.main)
    mm = meta[g.main]
    direction = int(mm["dir"])
    agree = bool(m["adj_agree"])
    verdict, rev = F.tier_verdict(_tier(m), agree)
    head = headline(direction, verdict, rev)
    main_a, main_e, main_d = float(m["y3_1"]), float(m["p_mkt_1"]), float(m["adj"])
    n_main = int(m["n1"])
    summary = (f"{g.label}の複勝率は {F.pct(main_a)}（オッズから見込まれる {F.pct(main_e)}）。"
               f"人気をそろえた差は {F.pt(main_d)}（{_tier_word(_tier(m))}）でした。")
    rows_meta = [(h, meta[h], _row(tests, h)) for h in g.all_hids()]
    others = [(h, mt, r) for h, mt, r in rows_meta if h != g.main]
    hit = [f"{mt['how']}（{'通説どおり' if bool(r['adj_agree']) else '通説と逆'}）" for h, mt, r in others if _tier(r) in ("S", "A", "B")]
    extra = f"同じ記事の中で格 S・A・B が付いた型: {'、'.join(hit)}。" if hit else ""
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": f"{g.label}の複勝率", "value": F.pct(main_a), "sub": f"{n_main:,} 走（発見期間）"},
            {"label": "オッズから見込まれる複勝率", "value": F.pct(main_e), "sub": "同じ人気の馬の平均"},
            {"label": "オッズをそろえた差", "value": F.pt(main_d), "sub": f"当てはまらない馬と比べて・{_tier_word(_tier(m))}・確認期間 {'–' if pd.isna(m['adj_c']) else F.pt(float(m['adj_c']))}"},
        ]},
        {"type": "text", "text": f"{summary}{tip(direction, verdict, rev)}{extra}"},
        {"type": "table", "columns": TABLE_COLUMNS, "rows": [_table_row(mt, r) for h, mt, r in rows_meta],
         "caption": f"発見期間 {_years(mm.get('disc'))}、確認期間 {_years(mm.get('conf'))}。差は『当てはまる馬 − 当てはまらない馬』を、人気をそろえて比べたもの。"},
    ]
    cells = _cells(tests, g.main)
    if cells:
        dim, cs = cells
        ch = F.chart(f"{g.label}を{DIM_LABEL[dim]}で分けると", DIM_LABEL[dim], cs,
                     f"当てはまる馬の実際の複勝率と、同じ人気の馬の見込み（発見期間 {_years(mm.get('disc'))}）。走数が {MIN_CELL} 未満の区分は省いています。")
        ch.pop("compare", None)
        body.append(ch)
    for slug, desc in g.links:
        body.append({"type": "link", "href": f"/column/{slug}", "label": "関連するコラム", "desc": desc})
    body.append({"type": "note", "text": f"主の型の定義: {mm['how']}。{_years(mm.get('years'))}の検定（{_years(mm.get('disc'))}で見つけ、{_years(mm.get('conf'))}で確かめる。全体の型でまとめて多重比較を補正）の格に判定を合わせています。"})
    doc = {
        "slug": g.slug,
        "title": f"{g.title}——{head}",
        "claim": g.claim,
        "verdict": verdict,
        "summary": summary,
        "categories": g.categories,
        "published": published,
        "updated": F.TODAY if published != F.TODAY else None,
        "dataset": {"period": _years(mm.get("years")), "conditions": "JRA全競馬場の平地（障害除く）",
                    "sample": f"{int(mm.get('n_elig') or 0):,} 走中 {int(mm.get('n_flag') or 0):,} 走が該当（主の型）"},
        "body": body,
    }
    q = g.question or f"『{g.claim}』は本当か。"
    guide = {
        "lead": [f"調べたこと: {q}", f"答え: {head}（同じ人気の馬と比べて {F.pt(main_d)}、{_tier_word(_tier(m))}）。", f"使い方: {tip(direction, verdict, rev)}"],
        "terms": ["複勝率", "オッズをそろえた差", "pt"],
        "glossary": [{"term": "格（S・A・B・C）", "desc": "偶然ではない確かさの目安です。S がもっとも確か、次いで A・B・C。『格なし』は、差があっても偶然と区別できないという意味です。発見期間で見つけた差が、確認期間でも同じ向きに出たかも含めて決めています。"}],
        "guides": [
            {"match": {"stats": True}, "title": "数字の読み方",
             "text": "『オッズから見込まれる複勝率』は、同じくらいの人気の馬が普段どれだけ 3 着以内に来るか。『オッズをそろえた差』は、当てはまる馬と当てはまらない馬を同じ人気で比べた差なので、『実際 − 見込み』と少し違い、符号が逆に見えることがあります。"},
            {"match": {"table": TABLE_COLUMNS}, "title": "表の読み方",
             "text": "同じテーマの型を並べています。『通説の向き』は、その通説が『走る』と言っているか『割引』と言っているか。『格』が付いた型だけが、偶然とは考えにくい差で、通説どおりか逆かも書いてあります。"},
        ],
    }
    result = {"slug": g.slug, "verdict": verdict, "reversed": rev, "tier": _tier(m), "diff": main_d, "n": n_main}
    return doc, guide, result


def build_all(only: set[str], dry_run: bool) -> tuple[dict[str, dict], list[str]]:
    """全グループの記事を書く。(結果, 作れなかったグループの説明) を返す。検証の結果が無い（別の PC など）ときは何もしない。
    型が検証の結果に無いグループは飛ばして続け、呼び出し側が結果を保存したあとで失敗にする。"""
    from src.research.columns.folklore_groups import GROUPS
    if not (VERIFY_DIR / "tests.parquet").exists():
        print(f"検証の結果が無いので、検証から作る記事は書きません: {VERIFY_DIR}")
        return {}, []
    tests, meta = load_results()
    out: dict[str, dict] = {}
    errors: list[str] = []
    for g in GROUPS:
        if only and g.slug not in only:
            continue
        try:
            doc, guide, res = build(g, tests, meta, F.published_date(g.slug))
        except KeyError as e:
            errors.append(f"{g.slug}: {e}")
            print(f"【作れません】{g.slug}: {e}")
            continue
        print(f"{g.slug:30s} {res['verdict']:14s} 主={g.main:5s} n={res['n']:>7,} 差={F.pt(res['diff'])} 格={res['tier']}")
        out[g.slug] = res
        if not dry_run:
            F.write_doc(g.slug, doc, guide)
    return out, errors
