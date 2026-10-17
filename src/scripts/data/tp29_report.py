"""tp-29 の結果を docs の 1 章（course_aptitude_groups.html）にする。

python -m src.scripts.data.tp29_report   # ホストで実行（data/local/modeling/training_pipeline/tp29/ の結果を読む）。出力は docs/html/modeling/chapters/ と /tmp/tp29_report.md|json
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline/tp29"
CHAPTER = ROOT / "docs/html/modeling/chapters/course_aptitude_groups.html"
STYLE_SRC = ROOT / "docs/html/modeling/chapters/course_performance_similarity.html"
VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
BAND = {"S": "〜1300m", "M": "1400〜1600m", "I": "1700〜2000m", "L": "2100〜2400m", "X": "2500m〜"}
ATTR_JP = {"sire": "父", "dam_sire": "母父", "jockey": "騎手", "trainer": "調教師", "style": "脚質", "gate": "馬番位置", "bracket": "枠", "age": "年齢", "sex": "性別", "jw": "斤量",
           "bw": "馬体重", "wtchg": "体重増減", "rest": "前走間隔", "clsstep": "クラス変更", "distgap": "距離変更", "nco": "同コース経験"}
TOK_JP = {"front": "逃げ・先行", "mid_front": "好位", "mid_back": "中団", "back": "後方", "in": "内", "mid": "中", "out": "外", "b12": "1-2枠", "b34": "3-4枠", "b56": "5-6枠", "b78": "7-8枠",
          "r_first": "初出走", "r_le14": "14日以内", "r_15_35": "15〜35日", "r_36_70": "36〜70日", "r_71_140": "71〜140日", "r_gt140": "141日〜", "cls_down": "降級", "cls_same": "同級",
          "cls_up": "昇級", "dg_shorter": "距離短縮", "dg_same": "同距離", "dg_longer": "距離延長", "nco_0": "初コース", "nco_1": "1走", "nco_2_3": "2〜3走", "nco_4p": "4走〜",
          "wc_m6": "−6kg以下", "wc_m2": "−2〜−6kg", "wc_0": "±2kg", "wc_p2": "+2〜+6kg", "wc_p6": "+6kg以上", "a2": "2歳", "a3": "3歳", "a4": "4歳", "a5": "5歳", "a6p": "6歳〜",
          "w_le52": "〜52kg", "w_53_54": "53〜54kg", "w_55_56": "55〜56kg", "w_57_58": "57〜58kg", "w_59p": "59kg〜", "bw_lt440": "〜439kg", "bw_440_480": "440〜479kg",
          "bw_480_520": "480〜519kg", "bw_gt520": "520kg〜"}


def course_label(k: str) -> str:
    p = k.split("|")
    if len(p) == 4:
        v = {"-": "", "?": "(内外不明)"}.get(p[3], p[3] + "回り")
        return f"{VENUE.get(p[0], p[0])}{p[1]}{p[2]}m{v}"
    if len(p) == 3:
        return f"{VENUE.get(p[0], p[0])}{p[1]}{BAND.get(p[2], p[2])}"
    return f"{p[0]}{BAND.get(p[1], p[1])}"


class Namer:
    def __init__(self):
        from src.scripts.data import tp29_eval as ev

        df = ev.load_frame()
        self.jockey = df.drop_duplicates("jockey_id").set_index("jockey_id")["jockey_name"].to_dict()
        self.trainer = df.drop_duplicates("trainer_id").set_index("trainer_id")["trainer_name"].to_dict()
        keys = pd.read_parquet(ev.PED / "pedigree_keys.parquet")
        self.by_sire = keys[keys["sire_id"] != ""].drop_duplicates("sire_id").set_index("sire_id")["horse_id"].to_dict()
        self.by_dam = keys[keys["dam_sire_id"] != ""].drop_duplicates("dam_sire_id").set_index("dam_sire_id")["horse_id"].to_dict()

    def _anc(self, horse_id: str, g: int, p: int) -> str:
        for d in (ROOT / "data/local/horse_pedigree_5gen", ROOT / "data/local/horse_pedigree_10gen"):
            f = d / horse_id[:4] / f"{horse_id}.json"
            if f.exists():
                for a in json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []:
                    if a.get("generation") == g and a.get("position") == p:
                        return str(a.get("name") or "")
        return ""

    def tok(self, t: str) -> str:
        a, v = t.split("=", 1)
        jp = ATTR_JP.get(a, a)
        if a == "jockey":
            return f"{jp}:{self.jockey.get(v, v)}"
        if a == "trainer":
            return f"{jp}:{self.trainer.get(v, v)}"
        if a == "sire":
            return f"{jp}:{self._anc(self.by_sire[v], 1, 0) or v}" if v in self.by_sire else f"{jp}:{v}"
        if a == "dam_sire":
            return f"{jp}:{self._anc(self.by_dam[v], 2, 2) or v}" if v in self.by_dam else f"{jp}:{v}"
        if a == "sex":
            return f"{jp}:{v}"
        return f"{jp}:{TOK_JP.get(v, v)}"


def _tbl(head: list[str], rows: list[list], cls: list[str] | None = None) -> str:
    h = "".join("<th>" + html.escape(x) + "</th>" for x in head)
    lines = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            attr = ' class="num"' if (cls and cls[i] == "n") else ""
            cells.append("<td" + attr + ">" + str(c) + "</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tr>" + h + "</tr>" + "".join(lines) + "</table>"


def build() -> None:
    counts = json.loads((OUT / "counts.json").read_text(encoding="utf-8"))
    imp = json.loads((OUT / "impact/report.json").read_text(encoding="utf-8")) if (OUT / "impact/report.json").exists() else {}
    lrr = json.loads((OUT / "lr_report.json").read_text(encoding="utf-8")) if (OUT / "lr_report.json").exists() else {}
    nm = Namer()

    # 1) 件数
    cnt_rows = []
    for mode, label in (("pop", "市場版（人気）"), ("model", "モデル期待版（v3 OOF の勝率順位）")):
        for r in counts[mode]:
            cnt_rows.append([label, r["year"], f"{r['rows']:,}", f"{r['labelable']:,}", f"{r['g1']:,}", f"{r['g2']:,}", f"{r['g1'] / max(r['labelable'], 1) * 100:.1f}%", f"{r['g2'] / max(r['labelable'], 1) * 100:.1f}%"])
    cnt_tbl = _tbl(["版", "年", "行数", "ラベル付与可", "G1 期待外れ", "G2 期待以上", "G1 率", "G2 率"], cnt_rows, ["", "n", "n", "n", "n", "n", "n", "n"])

    # 2) 有意な属性（生の差 vs コース固有の超過）
    sig_rows = []
    fin = {lv: pd.read_parquet(OUT / f"tables_final_pop_{lv}.parquet") for lv in ("k1", "k2", "k3")}
    for lv, lname in (("k1", "k1 場×面×距離×内外"), ("k2", "k2 場×面×距離帯"), ("k3", "k3 面×距離帯")):
        t = fin[lv].copy()
        t["attr"] = t["tok"].str.split("=").str[0]
        top = t[t["sig"]]["attr"].value_counts()
        topx = t[t["xsig"]]["attr"].value_counts()
        sig_rows.append([lname, f"{len(t):,}", f"{int(t['sig'].sum()):,}", " / ".join(f"{ATTR_JP.get(a, a)}{n}" for a, n in top.head(4).items()),
                         f"{int(t['xsig'].sum()):,}", " / ".join(f"{ATTR_JP.get(a, a)}{n}" for a, n in topx.head(4).items())])
    sig_tbl = _tbl(["コースの粒度", "検定した（コース×属性）数", "G2−G1 の差で有意（FDR 10%）", "内訳（上位）", "コース固有の超過で有意", "内訳"], sig_rows, ["", "n", "n", "", "n", ""])

    # 3) コース固有の超過で有意なもの（全件）
    xrows = []
    for lv in ("k1", "k2", "k3"):
        t = fin[lv]
        for _, r in t[t["xsig"]].sort_values("xp").head(40).iterrows():
            side = "期待以上(G2)に多い" if r["xlor"] > 0 else "期待外れ(G1)に多い"
            xrows.append([lv, html.escape(course_label(r["k"])), html.escape(nm.tok(r["tok"])), int(r["n2"]), int(r["n1"]), f"{r['xlor']:+.2f}", f"{r['xp']:.1e}", side])
    x_tbl = _tbl(["粒度", "コース", "属性", "G2 の件数", "G1 の件数", "超過の対数比", "p 値", "向き"], xrows, ["", "", "", "n", "n", "n", "n", ""])

    # 4) 参考: G2 と G1 の差（生）のコース別の上位。大きいコース k2 を件数順に
    ck = pd.read_csv(OUT / "counts_k2_pop.csv").sort_values("g2", ascending=False).head(12)
    t2 = fin["k2"]
    craw = []
    for k in ck["k2"]:
        s = t2[(t2["k"] == k) & t2["sig"]]
        pos = s.sort_values("lor", ascending=False).head(4)
        neg = s.sort_values("lor").head(4)
        craw.append([html.escape(course_label(k)), int(ck.loc[ck["k2"] == k, "g2"].iloc[0]), int(ck.loc[ck["k2"] == k, "g1"].iloc[0]),
                     "、".join(f"{html.escape(nm.tok(r['tok']))}({r['lor']:+.1f})" for _, r in pos[pos["lor"] > 0].iterrows()) or "-",
                     "、".join(f"{html.escape(nm.tok(r['tok']))}({r['lor']:+.1f})" for _, r in neg[neg["lor"] < 0].iterrows()) or "-"])
    raw_tbl = _tbl(["コース（k2）", "G2 件数", "G1 件数", "G2（期待以上）に多い属性（対数比）", "G1（期待外れ）に多い属性（対数比）"], craw, ["", "n", "n", "", ""])

    # 5) 勝率予測への増分
    cmp_rows = []
    for label, c in (imp.get("compare") or {}).items():
        cmp_rows.append([html.escape(label), f"{c['Δwin']:+.5f}", f"{c['win_hi']:+.5f}", f"{c['Δtop2']:+.5f}", f"{c['Δtop3']:+.5f}", "採用" if c.get("採用") else "不採用"])
    cmp_tbl = _tbl(["比較", "Δ勝率", "95%上限", "Δ連対", "Δ複勝", "採否"], cmp_rows, ["", "n", "n", "n", "n", ""]) if cmp_rows else "<p class='muted'>（評価の完了待ち）</p>"
    lv = imp.get("levels") or {}
    lvl_tbl = _tbl(["変種", "勝率 LogLoss", "連対", "複勝"], [[k, f"{v['logloss_win']:.5f}", f"{v['logloss_top2']:.5f}", f"{v['logloss_top3']:.5f}"] for k, v in lv.items()], ["", "n", "n", "n"]) if lv else ""

    # 6) 人気を制御した予測力
    lr_rows = []
    for tname, tlabel in (("G2_among_pop6plus", "人気6位以降の馬の中で G2（期待以上）を当てる"), ("G1_among_pop1to3", "人気1〜3位の馬の中で G1（期待外れ）を当てる")):
        for Y, r in (lrr.get(tname) or {}).items():
            if "skip" in r:
                continue
            for sname, sl in (("course_raw", "コース: G2−G1 の差"), ("course_excess", "コース: 固有の超過"), ("horse_hist", "馬自身の G1/G2 履歴"), ("all", "全部")):
                v = r[sname]
                lr_rows.append([tlabel, Y, r["auc_ctrl"], sl, f"{v['dAUC']:+.4f}", f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"])
    lr_tbl = _tbl(["問い", "検証年", "基準 AUC（人気順位・頭数・経験を制御）", "足したもの", "ΔAUC", "95% 区間"], lr_rows, ["", "n", "n", "", "n", "n"]) if lr_rows else ""

    cm = imp.get("compare") or {}

    def _g(label):
        c = cm.get(label)
        return f"{c['Δwin']:+.5f}（上限 {c['win_hi']:+.5f}）" if c else "-"

    concl = ("V0 に足した勝率 LogLoss の差は、市場版の全部 " + _g("市場版 全部（P_all − V0）") + "、G2−G1 の差のみ " + _g("市場版 G2−G1 の差のみ（P_raw − V0）")
             + "、コース固有の超過のみ " + _g("市場版 コース固有の超過のみ（P_x − V0）") + "、馬自身の履歴のみ " + _g("市場版 馬自身の履歴のみ（P_hz − V0）")
             + "。現行の前段 3 列を足した版に重ねると " + _g("市場版 全部（PS_all − V0S3）") + "。モデル期待版は " + _g("モデル期待版 全部（M_all − V0）")
             + "（ラベルのある 2023・2024 年だけでは " + _g("モデル期待版 全部（M_all − V0） [2023-24のみ]") + "）。いずれも採用基準（効果 −0.0002 以下、複勝の悪化なし）に届かない。")
    style = re.search(r"<style>.*?</style>", STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>期待とずれた馬のグループ差分によるコース適性（tp-29）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>期待とずれた馬のグループ差分によるコース適性（tp-29）</h1>
<p class="muted">ユーザー提案: 『期待外れ（人気馬の凡走）』と『期待以上（人気薄の好走）』の 2 つの馬のグループの統計的な差を、コースへの相性の数値にできないか。</p>
<div class="note"><b>グループの定義.</b>
 G1（期待外れ）= 期待順位 1〜3 位で、着順 6 着以降、かつ 3 着馬との着差（走破タイム差）が 0.3 秒以上。
 G2（期待以上）= 期待順位 6 位以降で 3 着以内、または 4 着以降でも 3 着馬との着差が 0.2 秒以内（3 着以内とみなす）。
 除外 = 着順 ≤ 0（取消・除外・中止・失格）、走破タイム ≤ 0 か欠損、期待順位が無い馬、3 着馬が定まらないレース。
 期待順位は<b>市場版 = 人気</b>（過去レースのラベルを作る材料としてだけ使い、当日の人気は特徴量に使わない）と、<b>モデル期待版 = v3 の OOF 勝率の順位</b>（市場を一切混ぜない対照。OOF のある 2022〜2024 年のみ）の 2 通り。<br>
 コースは 3 階層（k1 場×面×距離×内外、k2 場×面×距離帯、k3 面×距離帯）。属性は種牡馬・母父・騎手・調教師の ID、脚質、馬番位置・枠、年齢・性別、斤量・馬体重・増減、前走間隔、クラス・距離の変更、同コース経験。
 各コース×属性について G2 と G1 の出現率を縮約付き対数オッズ比で比べ、Fisher 検定 + BH-FDR（10%）で有意なものだけを使う。さらに『他のコース全体での G2−G1 の差』を引いた<b>コース固有の超過</b>（差の差、Wald 検定 + BH-FDR）も測る。
 年 Y の行は『年 Y より前のラベル』だけで作った統計で採点する（先読みなし）。馬自身の G1/G2 履歴は日付が厳密に前の走だけを数える。</div>
<h2>結論</h2>
<ul>
<li><b>『G2 と G1 の差』の大半は、コースの特徴ではなく『人気馬と人気薄馬の一般的な違い』だった。</b>有意な属性の多くは騎手・調教師・父で、どのコースでも G1 側に有力騎手・有力厩舎、G2 側に格下の騎手・厩舎が出る（人気の構成を映している）。この差だけで特徴量を作っても勝率予測は改善しない。</li>
<li><b>『コース固有の超過』（他のコースの差を引いたもの）で有意に残るのはごく少数</b>（k3 で 22、k2 で 7、k1 で 6）。内容はほぼ『脚質 × 面』（ダート短〜中距離で逃げ・先行が期待外れに多い、芝で逃げ・先行が期待以上に多い など）で、概形・距離の違いで説明がつく。</li>
<li><b>勝率予測への増分は基準未満で、特徴量としては不採用.</b>{concl}</li>
<li>人気を制御しても、適性スコアが『期待以上』『期待外れ』を当てる力は小さい（下表）。馬自身の G1/G2 履歴のほうが、コース由来のスコアよりやや効く。</li>
</ul>
<h2>グループの件数</h2>{cnt_tbl}
<h2>有意な属性の数（最終の統計: 2020〜2024 年）</h2>{sig_tbl}
<h2>コース固有の超過で有意だったもの（市場版）</h2><p class="muted">他のコース全体での G2−G1 の差を引いた差の差。『期待以上に多い』『期待外れに多い』は、そのコースで全体平均よりどちらに偏るか。</p>{x_tbl}
<h2>参考: コース別の G2 と G1 の差が大きい属性（件数の多い 12 コース、市場版）</h2><p class="muted">一般的な『人気馬と人気薄馬の違い』を含む。コース固有の要因は上の表で見る。</p>{raw_tbl}
<h2>勝率予測への増分（LightGBM 固定、valid 2022〜2024 の expanding、2 seed。test 2025 は未使用）</h2>
<p class="muted">基準 V0 = 修正後の学習表の tp-03 採用 175 列、V0S3 = V0 + 現行の前段 3 列。P = 市場版、M = モデル期待版（ラベルのある 2023・2024 年のみ効く）。raw = G2−G1 の差、x = コース固有の超過、hz = 馬自身の履歴。</p>{cmp_tbl}{lvl_tbl}
<h2>人気を制御したときの適性スコアの予測力</h2><p class="muted">検証年 Y は『Y より前の年で学習した LR』で評価。基準は人気順位・頭数・相対人気・馬の経験（過去の出走数）を入れた LR の AUC。</p>{lr_tbl}
<p class="muted">コード: <code>src/pipeline/features/tp29_course_aptitude.py</code>、<code>src/scripts/data/tp29_eval.py</code>、<code>tp29_report.py</code>。結果の生データ: <code>data/local/modeling/training_pipeline/tp29/</code>。</p>
</body></html>"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


if __name__ == "__main__":
    build()
