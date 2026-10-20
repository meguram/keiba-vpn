"""モデリング総覧(book.html)の用語集。データサイエンスの専門用語を、やさしい説明と図解つきで定義する。

- 本文中の用語（``pats``）は、build_docs_book が点線の下線にし、クリックするとその場に解説が開く。
- 巻末の「用語集」の章には、同じ解説が折りたたみ（details）で並ぶ。
- 説明は『ひとことで』『たとえ』『このプロジェクトでは』の順。プロジェクトの事実は各章の記述に合わせている。
- 新しい用語は ``TERMS`` に足す（id は英数字とハイフン）。図は ``DIAGRAMS`` の関数名を ``diagram`` に書く。
"""

from __future__ import annotations

import html
import math
import random
import re
from dataclasses import dataclass, field

CATS = [
    ("metric", "評価の指標（モデルの良さの測り方）"),
    ("validate", "検証の考え方（ズルをせずに確かめる）"),
    ("model", "モデルと学習の道具"),
    ("prob", "確率・順位・馬券の理屈"),
    ("feature", "特徴量の作り方"),
    ("project", "このプロジェクトの用語"),
]


@dataclass
class Term:
    id: str
    name: str
    cat: str
    pats: list[str]
    short: str
    body: str = ""            # 解説（HTML）。<p> 単位で。『ひとことで』は short が出る
    diagram: str = ""         # DIAGRAMS のキー
    see: list[str] = field(default_factory=list)   # 関連する章（slug）


# ── 図解（インライン SVG）─────────────────────────────────────────────────────

C_TEXT, C_MUT, C_ACC, C_OK, C_WARN, C_BAD, C_BOX = "#e6ecff", "#9aa7c7", "#7aa9ff", "#34d399", "#fbbf24", "#f87171", "#1a2a52"


def _svg(w: int, h: int, inner: str, label: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}" '
            f'font-family="Noto Sans JP, system-ui, sans-serif" font-size="12">{inner}</svg>')


def _t(x, y, s, fill=C_TEXT, size=12, anchor="start", weight=400):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def _cap(x, y, lines, fill=C_MUT, size=11, anchor="middle", gap=15):
    return "".join(_t(x, y + i * gap, s, fill, size, anchor) for i, s in enumerate(lines))


def _tv(x, y, s, fill=C_MUT, size=11):
    return f'<text transform="rotate(-90 {x} {y})" x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="middle">{s}</text>'


def _r(x, y, w, h, fill=C_BOX, stroke="none", rx=4, op=1.0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" fill-opacity="{op}"/>'


def _l(x1, y1, x2, y2, stroke=C_MUT, w=1.4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{w}"{d}/>'


def d_logloss() -> str:
    pts = []
    for i in range(1, 100):
        p = i / 100
        x = 50 + p * 300
        y = 150 - min(-math.log(p), 3.2) / 3.2 * 120
        pts.append(f"{x:.1f},{y:.1f}")
    inner = _l(50, 150, 350, 150) + _l(50, 30, 50, 150) + f'<polyline points="{" ".join(pts)}" fill="none" stroke="{C_ACC}" stroke-width="2.2"/>'
    for p, lab in ((0.9, "0.1"), (0.5, "0.69"), (0.1, "2.3")):
        x = 50 + p * 300
        y = 150 - (-math.log(p)) / 3.2 * 120
        inner += f'<circle cx="{x}" cy="{y:.1f}" r="4" fill="{C_WARN}"/>' + (_t(x - 4, y - 10, f"p={p} → {lab}", C_WARN, 11, "end") if p > 0.8 else _t(x + 8, y - 6, f"p={p} → {lab}", C_WARN, 11))
    inner += _t(200, 172, "正解につけた確率 p", C_MUT, 11, "middle") + _t(50, 22, "ペナルティ ↑", C_MUT, 11, "middle") + _t(360, 60, "自信満々で外すと", C_BAD, 11, "end") + _t(360, 74, "急に大きくなる", C_BAD, 11, "end")
    return _svg(380, 185, inner, "LogLoss: 正解に付けた確率が小さいほどペナルティが急に増える")


def d_calibration() -> str:
    inner = _l(50, 150, 250, 150) + _l(50, 30, 50, 150) + _l(50, 150, 250, 30, C_MUT, 1.2, "4 3")
    inner += f'<path d="M50,150 C110,130 150,60 250,34" fill="none" stroke="{C_BAD}" stroke-width="2.2"/>'
    inner += f'<path d="M50,150 C110,120 160,80 250,30" fill="none" stroke="{C_OK}" stroke-width="2.2" stroke-dasharray="1 0"/>'
    inner += _t(150, 172, "モデルが言う確率", C_MUT, 11, "middle") + _tv(22, 90, "実際の的中率")
    inner += _t(256, 40, "理想（点線）", C_MUT, 11) + _t(256, 64, "較正前（赤）", C_BAD, 11) + _t(256, 80, "較正後に近づく", C_OK, 11)
    return _svg(380, 185, inner, "較正: モデルの確率と実際の的中率を一致させる")


def d_oof() -> str:
    inner = ""
    names = ["A", "B", "C", "D"]
    for row in range(4):
        y = 20 + row * 34
        inner += _t(14, y + 16, f"回{row + 1}", C_MUT, 11)
        for col in range(4):
            x = 50 + col * 70
            if col == row:
                inner += _r(x, y, 64, 26, C_WARN, op=0.9) + _t(x + 32, y + 17, f"{names[col]} を予測", "#1b1b1b", 11, "middle", 700)
            else:
                inner += _r(x, y, 64, 26, C_ACC, op=0.35) + _t(x + 32, y + 17, f"{names[col]} で学習", C_TEXT, 10.5, "middle")
    inner += _t(50, 170, "各区間の予測は『その区間を見ずに学習したモデル』だけで作る → 4 区間ぶんの予測を並べたものが OOF", C_OK, 11.5)
    return _svg(360, 184, inner, "OOF: 学習に使わなかった区間を予測して並べる")


def d_expanding() -> str:
    inner = _t(14, 18, "時間 →", C_MUT, 11)
    rows = [("fold1", 3, 1), ("fold2", 4, 1), ("fold3", 5, 1)]
    for i, (n, tr, va) in enumerate(rows):
        y = 30 + i * 44
        inner += _t(8, y + 17, n, C_MUT, 11)
        w = 42
        for k in range(tr):
            inner += _r(52 + k * (w + 2), y, w, 26, C_ACC, op=0.55)
        x = 52 + tr * (w + 2)
        inner += _r(x, y, 10, 26, "#444", op=0.9) + _r(x + 12, y, w, 26, C_WARN, op=0.95) + _t(x + 12 + w / 2, y + 17, "valid", "#1b1b1b", 11, "middle", 700)
    inner += _r(52, 168, 10, 14, "#444", op=0.9) + _t(68, 179, "= purge / embargo（学習と評価の間の“あき”。このプロジェクトは 7 日 + 3 日）", C_MUT, 10.5)
    inner += _r(52, 190, 24, 14, C_ACC, op=0.55) + _t(82, 201, "= train（年ごとに学習期間が増える）　", C_MUT, 10.5)
    return _svg(440, 214, inner, "年単位で学習期間を延ばす時系列の検証")


def d_leak() -> str:
    inner = _l(20, 90, 400, 90, C_MUT, 2)
    inner += _t(200, 34, "予測する時点（レース直前）", C_WARN, 11.5, "middle", 700)
    inner += _r(30, 70, 160, 40, "#164e3f") + _t(110, 94, "使ってよい（過去）", C_TEXT, 11.5, "middle")
    inner += _r(210, 70, 180, 40, "#5b2530") + _t(300, 94, "使ってはいけない（未来）", C_TEXT, 11.5, "middle") + _l(200, 40, 200, 150, C_WARN, 2, "5 3")
    inner += _t(300, 126, "このレースの結果・確定オッズ・", C_BAD, 10.5, "middle") + _t(300, 140, "同日の後のレースの結果 など", C_BAD, 10.5, "middle")
    inner += _t(110, 126, "前走までの成績・前日の出馬表・", C_OK, 10.5, "middle") + _t(110, 140, "それより前に終わったレース など", C_OK, 10.5, "middle")
    return _svg(420, 156, inner, "リーク: 予測時点より未来の情報を使ってしまうこと")


def d_adversarial() -> str:
    rnd = random.Random(3)
    inner = ""
    for k, (cx, col, lab) in enumerate(((90, C_ACC, "train"), (250, C_WARN, "valid/test"))):
        inner += _r(cx - 70, 22, 140, 120, "none", C_MUT, 8)
        for _ in range(26):
            x = cx + rnd.uniform(-55, 55) + (0 if k == 0 else 38)
            y = 82 + rnd.uniform(-45, 45)
            inner += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.2" fill="{col}" fill-opacity="0.85"/>'
        inner += _t(cx, 16, lab, col, 11.5, "middle", 700)
    inner += _t(170, 162, "見分けられるほど（AUC が 1 に近いほど）学習と評価で“世界が違う”", C_MUT, 11, "middle")
    return _svg(340, 176, inner, "adversarial validation: 学習データと評価データを見分けられるか")


def d_ablation() -> str:
    bars = [("全部入り", 0.2221, C_OK), ("− A（上がり）", 0.2254, C_WARN), ("− D（休養/体重）", 0.2240, C_WARN), ("− F1", 0.2222, C_MUT)]
    inner = _t(14, 18, "LogLoss（小さいほど良い）※図は説明用の数値", C_MUT, 11)
    for i, (n, v, c) in enumerate(bars):
        y = 34 + i * 30
        w = (v - 0.2200) / 0.0070 * 220 + 40
        inner += _t(14, y + 15, n, C_TEXT, 11.5) + _r(130, y, w, 22, c, op=0.6) + _t(136 + w, y + 15, f"{v:.4f}", C_TEXT, 11)
    inner += _t(14, 168, "外して悪くなる → その部品は効いている / 変わらない → 要らない", C_MUT, 11)
    return _svg(420, 182, inner, "ablation: 部品を 1 つずつ外して効果を測る")


def d_auc() -> str:
    inner = _l(50, 150, 220, 150) + _l(50, 30, 50, 150) + _l(50, 150, 220, 30, C_MUT, 1.2, "4 3")
    inner += f'<path d="M50,150 C70,70 110,45 220,30" fill="none" stroke="{C_OK}" stroke-width="2.4"/>' + f'<path d="M50,150 L50,150 L220,30 L220,150 Z" fill="none"/>'
    inner += _t(135, 172, "外れを誤って当たりとした割合", C_MUT, 11, "middle") + _tv(22, 90, "当たりを拾えた率")
    inner += _t(232, 70, "曲線の下の面積 = AUC", C_OK, 11.5) + _t(232, 90, "0.5 = 当てずっぽう", C_MUT, 11) + _t(232, 108, "1.0 = 完璧", C_MUT, 11)
    return _svg(400, 185, inner, "AUC: 当たりを外れより上に並べられる確率")


def d_harville() -> str:
    inner = _t(14, 18, "例: 4 頭の勝率 40% / 30% / 20% / 10%", C_MUT, 11)
    inner += _t(14, 46, "1 着:", C_TEXT, 11.5, weight=700)
    for i, (p, c) in enumerate(((40, C_ACC), (30, C_OK), (20, C_WARN), (10, C_BAD))):
        inner += _r(58 + sum(x for x in (40, 30, 20, 10)[:i]) * 3, 32, p * 3, 22, c, op=0.7) + _t(58 + sum(x for x in (40, 30, 20, 10)[:i]) * 3 + p * 1.5, 48, f"{p}%", "#101010", 11, "middle", 700)
    inner += _t(14, 92, "1 着が A のとき、2 着は“残りの勝率の比”で決まる:", C_TEXT, 11.5, weight=700)
    for i, (p, c) in enumerate(((30, C_OK), (20, C_WARN), (10, C_BAD))):
        pp = p / 60 * 100
        inner += _r(58 + sum((30, 20, 10)[:i]) / 60 * 300, 100, p / 60 * 300, 22, c, op=0.7) + _t(58 + sum((30, 20, 10)[:i]) / 60 * 300 + p / 60 * 150, 116, f"{pp:.0f}%", "#101010", 11, "middle", 700)
    inner += _t(14, 150, "→ これを繰り返すと、馬連・ワイド・3連複の的中確率が計算できる", C_OK, 11.5)
    return _svg(380, 166, inner, "Harville: 勝率から 2 着・3 着の確率を順に計算する")


def d_softmax() -> str:
    scores = [2.0, 1.2, 0.4, -0.4]
    inner = ""
    for col, (T, lab) in enumerate(((0.5, "温度 低（尖る）"), (1.0, "温度 1（標準）"), (3.0, "温度 高（なだらか）"))):
        ex = [math.exp(s / T) for s in scores]
        z = sum(ex)
        x0 = 20 + col * 130
        inner += _t(x0 + 50, 18, lab, C_ACC, 11, "middle", 700)
        for i, e in enumerate(ex):
            h = e / z * 100
            inner += _r(x0 + i * 26, 120 - h, 20, h, C_ACC, op=0.75) + _t(x0 + i * 26 + 10, 134, f"{e / z * 100:.0f}", C_MUT, 10, "middle")
    inner += _cap(205, 158, ["スコア（点数）を合計 100% の確率に直すのが softmax。", "温度で“自信の強さ（尖り具合）”を調整する"], C_MUT, 11)
    return _svg(410, 190, inner, "softmax と温度")


def d_ewm() -> str:
    inner = _t(14, 18, "直近ほど重い（指数加重移動平均）", C_MUT, 11)
    for i in range(8):
        w = 120 * (0.65 ** i)
        inner += _r(40 + i * 40, 150 - w, 30, w, C_ACC, op=0.8) + _t(55 + i * 40, 166, f"{i + 1}走前", C_MUT, 10, "middle")
    inner += _t(200, 40, "重み", C_WARN, 11.5)
    return _svg(380, 178, inner, "EWM: 新しい走ほど重みが大きい")


def d_elo() -> str:
    inner = _r(30, 40, 110, 50, C_BOX, C_ACC) + _t(85, 62, "馬 A  1500", C_TEXT, 12, "middle") + _t(85, 80, "強い相手に勝つ", C_OK, 10.5, "middle")
    inner += _r(250, 40, 110, 50, C_BOX, C_WARN) + _t(305, 62, "馬 B  1600", C_TEXT, 12, "middle") + _t(305, 80, "格下に負ける", C_BAD, 10.5, "middle")
    inner += _l(140, 65, 250, 65, C_MUT, 1.6, "4 3") + _t(195, 58, "対戦", C_MUT, 10.5, "middle")
    inner += _t(85, 118, "↑ +20（想定より大きく上がる）", C_OK, 11, "middle") + _t(305, 118, "↓ −20（想定より大きく下がる）", C_BAD, 11, "middle")
    inner += _t(195, 150, "結果が“予想（強さの差）”と違うほど、レーティングが大きく動く", C_MUT, 11, "middle")
    return _svg(390, 166, inner, "Elo: 予想との差でレーティングを更新")


def d_shrink() -> str:
    inner = _l(30, 105, 390, 105, C_MUT, 1.2, "4 3") + _t(394, 109, "全体平均", C_MUT, 10)
    for i, (n, raw, shr) in enumerate(((2, 130, 94), (5, 118, 90), (30, 106, 90), (200, 92, 90))):
        x = 60 + i * 90
        inner += f'<circle cx="{x}" cy="{150 - raw * 0.5}" r="5" fill="{C_BAD}"/>' + f'<circle cx="{x}" cy="{150 - shr * 0.5}" r="5" fill="{C_OK}"/>' + _l(x, 150 - raw * 0.5, x, 150 - shr * 0.5, C_WARN, 1.2, "2 2")
        inner += _t(x, 170, f"{n} 件", C_MUT, 10.5, "middle")
    inner += _cap(30, 20, ["赤 = 生の平均 / 緑 = 縮約後。", "件数が少ないほど全体平均に引き寄せる"], C_TEXT, 11, "start")
    return _svg(450, 184, inner, "縮約: 少ない件数の平均を全体平均に近づける")


def d_stack() -> str:
    inner = ""
    for i, n in enumerate(("モデル1", "モデル2", "モデル3")):
        y = 24 + i * 36
        inner += _r(20, y, 90, 26, C_BOX, C_ACC) + _t(65, y + 17, n, C_TEXT, 11.5, "middle") + _l(110, y + 13, 170, 82, C_MUT, 1.4)
    inner += _r(170, 62, 110, 40, C_ACC, op=0.35) + _t(225, 80, "混ぜる", C_TEXT, 12, "middle", 700) + _t(225, 95, "平均 / 学習", C_MUT, 10.5, "middle")
    inner += _l(280, 82, 330, 82, C_MUT, 1.6) + _r(330, 66, 80, 32, C_OK, op=0.4) + _t(370, 87, "最終確率", C_TEXT, 11.5, "middle")
    inner += _cap(215, 150, ["平均して混ぜる = アンサンブル", "混ぜ方も学習する = スタッキング（学習に使わなかった予測 OOF で混ぜ方を学ぶ）"], C_MUT, 10.5)
    return _svg(430, 184, inner, "アンサンブルとスタッキング")


def d_tiers() -> str:
    xs = [("前日\n18:00", 50, "tier 0", C_ACC), ("当日朝", 140, "tier 1", C_OK), ("T-60", 235, "tier 2", C_WARN), ("T-15", 325, "オッズ", C_BAD), ("発走", 395, "", C_MUT)]
    inner = _l(20, 70, 410, 70, C_MUT, 2)
    for lab, x, tier, col in xs:
        inner += f'<circle cx="{x}" cy="70" r="6" fill="{col}"/>' + "".join(_t(x, 50 - 14 * k, ln, C_TEXT, 11, "middle") for k, ln in enumerate(reversed(lab.split("\n")))) + (_t(x, 98, tier, col, 11, "middle", 700) if tier else "")
    for x, lines in ((50, ["枠順", "斤量・馬柱"]), (140, ["馬場傾向", "先行レース"]), (235, ["馬体重", "（発表後）"]), (325, ["市場", "オッズ"])):
        inner += _cap(x, 126, lines, C_MUT, 10.5, gap=13)
    inner += _cap(215, 162, ["T-X = 発走の X 分前。分かる情報が時間とともに増えるので、", "特徴量を“いつ分かるか”で分ける"], C_MUT, 10.5)
    return _svg(430, 186, inner, "T-X と tier: 情報がいつ分かるか")


def d_layers() -> str:
    inner = _r(20, 20, 170, 70, C_BOX, C_ACC) + _t(105, 44, "LayerA", C_ACC, 14, "middle", 700) + _t(105, 62, "市場に頼らない", C_TEXT, 11, "middle") + _t(105, 78, "実力・状態の確率", C_TEXT, 11, "middle")
    inner += _r(240, 20, 170, 70, C_BOX, C_WARN) + _t(325, 44, "LayerB", C_WARN, 14, "middle", 700) + _t(325, 62, "市場オッズと合成", C_TEXT, 11, "middle") + _t(325, 78, "期待値 → 買い目", C_TEXT, 11, "middle")
    inner += _l(190, 55, 240, 55, C_MUT, 1.8) + _t(215, 48, "確率", C_MUT, 10, "middle")
    inner += _cap(215, 118, ["オッズ・人気は LayerA に入れない", "（入れると“市場の写し”になり、独自の見立てが消える）"], C_MUT, 10.5)
    return _svg(430, 150, inner, "LayerA と LayerB")


def d_target_enc() -> str:
    inner = _t(20, 20, "カテゴリ（文字） → 数字", C_MUT, 11)
    rows = [("東京", "勝率 9.8%"), ("中山", "勝率 8.6%"), ("新規の競馬場", "データ少 → 全体平均寄り")]
    for i, (a, b) in enumerate(rows):
        y = 34 + i * 34
        inner += _r(20, y, 110, 26, C_BOX, C_ACC) + _t(75, y + 17, a, C_TEXT, 11.5, "middle") + _l(130, y + 13, 190, y + 13, C_MUT, 1.6) + _r(190, y, 200, 26, C_ACC, op=0.3) + _t(290, y + 17, b, C_TEXT, 11, "middle")
    inner += _cap(20, 146, ["注意: 結果を使って作る数字なので、同じレースの結果が混ざる", "“リーク”に弱い → OOF や as-of で慎重に作る"], C_WARN, 10.5, "start")
    return _svg(420, 180, inner, "target encoding")


DIAGRAMS = {
    "logloss": d_logloss, "calibration": d_calibration, "oof": d_oof, "expanding": d_expanding, "leak": d_leak, "adversarial": d_adversarial,
    "ablation": d_ablation, "auc": d_auc, "harville": d_harville, "softmax": d_softmax, "ewm": d_ewm, "elo": d_elo, "shrink": d_shrink,
    "stack": d_stack, "tiers": d_tiers, "layers": d_layers, "target_enc": d_target_enc,
}


# ── 用語 ─────────────────────────────────────────────────────────────────────

T = Term
TERMS: list[Term] = [
    # 評価の指標
    T("logloss", "LogLoss（対数損失）", "metric", ["LogLoss", "logloss", "対数損失"],
      "確率予測の“外し方のきつさ”を測る点数。小さいほど良い。",
      "<p><b>たとえ:</b> 天気予報で「降水確率 90%」と言って降らなかった予報士は、「50%」と言った人よりも強く責められる。LogLoss はそれを数式にしたもの。正解に付けた確率が小さいほど罰が急に重くなるので、<b>自信満々で外すこと</b>を強く嫌う。</p>"
      "<p><b>このプロジェクトでは:</b> 勝つ確率（p_win）の主指標。例えば全部入りの特徴量で 0.222 程度（ベースライン 0.2326）。差が 0.0001 でも seed 間のばらつき程度なので、0.001 以上の差を意味のある改善とみる。</p>",
      "logloss", ["training_pipeline_decisions", "feature_strategy_tp03_tp04"]),
    T("brier", "Brier スコア", "metric", ["Brier", "ブライアスコア"], "予測確率と結果（当たり=1／外れ=0）の差の二乗の平均。小さいほど良い。",
      "<p>LogLoss より“外し方の極端さ”に寛容で、確率の当てはまり具合を素直に見られる。LogLoss と一緒に併記して使う。</p>", "", ["training_pipeline_decisions"]),
    T("ece", "ECE（期待較正誤差）", "metric", ["ECE"], "モデルが言う確率と、実際の的中率のずれの平均。0 に近いほど“確率として正直”。",
      "<p>予測確率を 0–10%、10–20% … のように区切り、各区間で「言った確率の平均」と「実際に当たった割合」の差を平均する。<b>確率を買い目の期待値に使う</b>ので、順位が当たることより、この“正直さ”が大事。</p>", "calibration", ["training_pipeline_decisions"]),
    T("auc", "AUC", "metric", ["AUC"], "当たりを外れより上に並べられる確率。0.5 = 当てずっぽう、1.0 = 完璧。",
      "<p>ランダムに「当たり 1 頭」と「外れ 1 頭」を選んだとき、モデルが当たりのほうに高い点を付けている確率。順位付けの力を見る指標で、確率の“正直さ”までは見ない。</p>"
      "<p><b>注意:</b> adversarial validation では意味が逆。AUC が高い = 学習と評価で分布が違う、という“悪い知らせ”として使う。</p>", "auc", ["train_valid_test_split_strategy"]),
    T("ndcg", "NDCG", "metric", ["NDCG"], "順位づけの良さ。上位に正解を置けているほど高い（0〜1）。", "<p>1 着を 2 着に置き間違えるほうが、15 着と 16 着を間違えるより重く減点される。ランキング学習（LambdaRank など）の評価でよく使う。</p>"),
    T("mae", "MAE（平均絶対誤差）", "metric", ["MAE"], "予測と実際の差（絶対値）の平均。単位がそのまま分かる。", "<p>例: 上がり 3F の予測の MAE が 0.87 秒なら、平均して 0.87 秒ほど外している。</p>"),
    T("spearman", "Spearman（順位相関）", "metric", ["Spearman"], "2 つの順位がどれだけ似ているか（−1〜1）。", "<p>数値そのものでなく“順番”の一致を見る。1 に近いほど、予測の順位と実際の順位が揃っている。</p>"),
    T("roi", "ROI・回収率", "metric", ["ROI", "回収率"], "賭けた金額に対する払い戻しの割合。100% を超えれば黒字。",
      "<p>例: 1 万円賭けて 9,000 円戻れば回収率 90%。<b>期待回収率</b>は“モデルの確率 × オッズ”の予想値で、画面では 105% 以上を目安に色分けしている。</p>", "", ["exotic_bet_probabilities"]),
    # 検証
    T("oof", "OOF（Out-Of-Fold）予測", "validate", ["OOF"], "学習に使わなかった部分だけを予測して並べたもの。ズルのない予測の作り方。",
      "<p><b>たとえ:</b> 模試の採点。勉強に使った問題で採点しても実力は分からない。<b>勉強に使っていない問題</b>の成績だけを集めれば、本番に近い実力が見える。</p>"
      "<p><b>このプロジェクトでは:</b> 前段の予測（位置取り・ペースなど）を後段のモデルに渡すとき、前段が“自分の学習に使ったレース”を予測した値を渡すと結果を覚えているぶん良く見えすぎる。そこで時系列で OOF を作り、後段にはそれだけを渡す。2020〜2021 年の行は前段の学習期間なので、OOF が付かず欠損になる。</p>",
      "oof", ["feature_strategy_tp03_tp04", "modeling_strategy_tp04_tp07"]),
    T("cv", "交差検証（CV）", "validate", ["交差検証", r"\bCV\b"], "データをいくつかに分け、学習と評価を入れ替えながら繰り返して、実力を確かめる方法。",
      "<p>1 回の分け方だと“たまたま”の影響を受けるので、複数回評価して平均する。競馬は時間の流れがあるので、ランダムに分けず<b>時系列の分割</b>にする。</p>", "expanding", ["train_valid_test_split_strategy"]),
    T("expanding", "時系列分割（expanding window）", "validate", ["expanding window", "yearly_expanding_folds", "時系列分割", "時系列 CV", "時系列CV"],
      "学習期間を年ごとに延ばし、その次の年で評価する検証。未来を学習に使わない。",
      "<p>このプロジェクトでは valid を 2022・2023・2024 年として、それぞれの前の年までで学習する。<b>2025 年（test）は最後まで見ない</b>。学習と評価の境目には purge（7 日）と embargo（3 日）の“あき”を置く。</p>", "expanding", ["train_valid_test_split_strategy", "training_pipeline_decisions"]),
    T("purge", "purge / embargo", "validate", ["purge", "embargo", "Purged"], "学習と評価の境目に置く“あき”。境目の情報漏れ（近いレース同士の似た情報）を防ぐ。",
      "<p>例えば同じ週のレースは、馬場や出走馬が重なっていて互いに情報を持っている。学習の終わりと評価の始まりをぴったりくっつけると、その情報が漏れるので、数日あけておく。</p>", "expanding", ["train_valid_test_split_strategy"]),
    T("groupkfold", "GroupKFold", "validate", ["GroupKFold"], "同じレース（グループ）の馬が学習側と評価側に分かれないように分割する方法。",
      "<p>同じレースの馬は互いに情報を持っている（強い馬がいれば他の馬は不利になる）ので、バラバラにすると評価が甘くなる。</p>"),
    T("fold", "fold・seed", "validate", ["fold", "seed", "Seed"], "fold = 検証の 1 回分の分け方。seed = 乱数の種（変えると結果が少し変わる）。",
      "<p>seed を 3 通り回して平均し、<b>結果のばらつき（約 0.0001）より大きい差</b>だけを意味のある差とみる。</p>"),
    T("valid", "valid / test", "validate", ["valid", "Valid", "test 2025"], "valid = 調整のために何度も見る評価用。test = 最後に一度だけ見る本番の答え合わせ。",
      "<p>test を調整に使うと、“テストの答えに合わせて作った”ことになり、本当の実力が分からなくなる。</p>", "", ["train_valid_test_split_strategy"]),
    T("leak", "リーク（情報漏れ）", "validate", ["リーク", "leak", "リークなし"], "予測する時点では知り得ない情報が、学習や特徴量に混ざること。",
      "<p><b>たとえ:</b> テスト前に答案を見てしまうこと。成績は良く見えるが、本番では全く役に立たない。</p>"
      "<p><b>このプロジェクトでは:</b> 特徴量は“レース前に分かる情報だけ”で作る（as-of）。オッズ・人気は母表に入れない。2026-10 に、当日の直前傾向の計算で<b>後のレースの結果が混ざる</b>不具合を見つけて直した。</p>", "leak", ["pre_race_feature_whitelist", "feature_strategy_tp03_tp04"]),
    T("asof", "as-of（その時点の）", "validate", ["as-of", "as of"], "“その時点で分かっていた情報だけ”で計算すること。",
      "<p>例: 2024-05-05 のレースの特徴量は、2024-05-04 までの結果だけで作る。同じ日の他のレースの結果も使わない（時刻が前のレースを除く）。</p>", "leak", ["feature_strategy_tp03_tp04"]),
    T("adv", "adversarial validation", "validate", ["adversarial validation", "adversarial"], "学習データと評価データを見分ける分類器を作り、“時代ずれ”の大きさを測る方法。",
      "<p>見分けられない（AUC が 0.5 に近い）なら同じ分布。簡単に見分けられるなら、評価期間は学習期間と性質が違っていて、成績が学習どおりには出ないかもしれない。</p>", "adversarial", ["training_pipeline_decisions", "feature_strategy_tp03_tp04"]),
    T("drift", "時代ずれ（ドリフト）", "validate", ["時代ずれ", "ドリフト", "drift"], "年が進むにつれて、データの性質が変わっていくこと。",
      "<p>例: 年々増える“経験の長い馬”の割合、クラス構成の変化。学習した時代と評価する時代で性質が違うと、精度が落ちる。</p>", "adversarial", ["training_pipeline_decisions"]),
    T("ablation", "ablation（部品外し実験）", "validate", ["ablation", "Ablation", "ablation"], "部品（特徴量のグループ）を 1 つずつ外して、どれが効いているかを調べる実験。",
      "<p>外して成績が悪くなればその部品は効いている。変わらなければ要らない（複雑さを減らせる）。tp-03 で採用する特徴量を決めるのに使った。</p>", "ablation", ["training_pipeline_decisions"]),
    T("baseline", "ベースライン", "validate", ["ベースライン", "baseline", "B0"], "比べる基準になる、いちばん単純なモデル。",
      "<p>新しい工夫が効いたかは“ベースラインからの差”で判断する。先にベースラインを固定してから改善を足すのが鉄則。</p>", "", ["reference_insights"]),
    T("overfit", "過学習", "validate", ["過学習", "オーバーフィット", "overfit"], "学習データを“覚えすぎて”、新しいデータでは外れるようになること。",
      "<p>練習問題の答えを丸暗記して、本番の問題が解けない状態。valid / test で確かめる。</p>"),
    T("eda", "EDA（探索的データ分析）", "validate", ["EDA"], "モデルを作る前に、データの形・偏り・欠損を眺めて確かめる作業。"),
    # モデル
    T("gbdt", "GBDT・LightGBM・CatBoost・XGBoost", "model", ["GBDT", "LightGBM", "CatBoost", "XGBoost"], "小さな決定木を何本も足し合わせて予測する、表形式データで定番のモデル。",
      "<p>1 本目の木の間違いを 2 本目が直し、3 本目がさらに直す…と積み重ねる。LightGBM・CatBoost・XGBoost は同じ考え方の別実装で、速さやカテゴリ変数の扱いが違う。このプロジェクトでは LightGBM を基準に使い、CatBoost などと比較している。</p>"),
    T("ltr", "ランキング学習（LambdaRank / YetiRank）", "model", ["LambdaRank", "YetiRank", "ランキング学習", r"\bLTR\b"], "“順位そのもの”を良くするように学習する方法。",
      "<p>1 頭ずつの当たり外れでなく、同じレースの馬同士の並びを直接学ぶ。NDCG などで評価する。</p>"),
    T("optuna", "Optuna・ハイパーパラメータ", "model", ["Optuna", "ハイパーパラメータ"], "モデルの設定値（木の深さ、学習率など）。Optuna は良い設定値を自動で探す道具。",
      "<p>ハイパーパラメータは“学習の前に人が決める設定”。探しすぎると valid に合わせすぎるので、二層の CV（内側で探し、外側で評価）にする。</p>", "", ["hyperparameter_tuning_workflow"]),
    T("ensemble", "アンサンブル・スタッキング", "model", ["アンサンブル", "スタッキング", "stacking"], "複数のモデルを混ぜて 1 つの予測にすること。混ぜ方まで学習するのがスタッキング。",
      "<p>得意不得意の違うモデルを混ぜると、単独より安定して良くなることが多い。スタッキングでは混ぜ方を学習するので、OOF（学習に使わなかった予測）で混ぜ方を決める。</p>", "stack", ["modeling_experiment_report"]),
    T("rl", "強化学習", "model", ["強化学習"], "試行錯誤で“どう行動すると得か”を学ぶ方法。ここでは買い方（どの馬券にいくら）の学習。",
      "<p>状態（レースの確率・オッズ）→ 行動（買い方）→ 報酬（回収）の繰り返しで方策を学ぶ。過去データのバックテスト環境で学ばせる設計（tp-08）。</p>", "", ["betting_policy_reinforcement_learning", "modeling_experiment_report"]),
    T("softmax", "softmax・温度", "model", ["softmax", "Softmax", "温度"], "点数を、合計 100% の確率に直す計算。温度で“尖り具合”を調整する。",
      "<p>温度が低いと 1 位の確率が大きくなり、高いとなだらかになる。レース内の全馬の勝率を合計 100% にそろえるために使う。</p>", "softmax", ["modeling_experiment_report"]),
    T("calibration", "較正（キャリブレーション）", "model", ["較正", "calibration", "Platt"], "モデルの確率を、実際の的中率に合うように直すこと。",
      "<p>「30% と言った馬が実際は 20% しか来ない」ような癖を、後から補正する。期待値の計算に使う確率なので重要。Platt scaling や温度の調整などの手法がある。</p>", "calibration", ["training_pipeline_decisions"]),
    T("loss", "損失関数（目的関数）", "model", ["損失関数", "目的関数"], "“どれだけ外れたか”を一つの数字にする式。モデルはこれが小さくなるよう学習する。",
      "<p>何を損失にするかで、モデルが“何を頑張るか”が変わる（勝ち馬の確率を当てる、順位を当てる、など）。tp-05a では複数の目的関数を比較した。</p>", "", ["tp05a_experiment_report"]),
    # 確率・馬券
    T("harville", "Harville・割引 Harville", "prob", ["Harville", "割引 Harville"], "各馬の勝率から、2 着・3 着の確率を順に計算する古典的な方法。",
      "<p>1 着が決まったら、残りの馬の勝率の比で 2 着が決まる、と考える。実際は人気薄の 2・3 着を過小に見積もりがちなので、割引（α）で緩める。連対率・複勝率に合うよう α をレースごとに当てはめる。</p>", "harville", ["exotic_bet_probabilities"]),
    T("pl", "Plackett–Luce", "prob", ["Plackett-Luce", "Plackett–Luce", "Plackett"], "順位がどう決まるかを、“強さに比例して順に選ばれる”と考えるモデル。", "<p>Harville と同じ形の確率モデルで、順位のデータから各馬の強さを推定するのにも使う。</p>", "harville", ["exotic_bet_probabilities"]),
    T("benter", "Benter 型（市場との合成）", "prob", ["Benter"], "自分のモデルの確率と、市場（オッズ）の確率を、2 段階で合成する有名な方法。",
      "<p>市場は多くの情報を織り込んでいる。独自モデルが“市場にない情報”を持つ分だけを足すイメージ。LayerB でこの考え方を使う。</p>", "layers", ["modeling_strategy_tp04_tp07"]),
    T("kelly", "Kelly 基準", "prob", ["Kelly", "ケリー"], "期待値とオッズから、資金のどれだけの割合を賭けるのが長期的に最適かを決める式。",
      "<p>期待値が高く、勝ちやすいほど多く賭け、そうでなければ控える。実際は予測の誤差を考えて 1/4 などの“分数ケリー”にする。</p>", "", ["betting_policy_reinforcement_learning"]),
    T("ev", "期待値", "prob", ["期待値"], "平均するといくら戻ってくるか。確率 × オッズ（の払い戻し）。",
      "<p>当たる確率が 10%、オッズが 15 倍なら期待値は 1.5（150%）。モデルの確率が正直（較正されている）ことが前提。</p>", "", ["exotic_bet_probabilities"]),
    # 特徴量
    T("feature", "特徴量", "feature", ["特徴量"], "モデルに与える入力の“手がかり”。例: 前走の上がり、休養日数、馬体重の増減。", "<p>生のデータそのものではなく、予測に役立つ形に加工した数字。このプロジェクトでは履歴から作る 127 列が中心。</p>"),
    T("targetenc", "target encoding", "feature", ["target encoding", "ターゲットエンコーディング"], "カテゴリ（競馬場、騎手など）を、“その結果の平均”に置き換えて数字にする方法。",
      "<p>便利だが、結果を使って作るぶんリークしやすい。OOF や as-of で慎重に作る。</p>", "target_enc", ["advanced_modeling_techniques"]),
    T("zscore", "標準化（z スコア）", "feature", ["標準化", "z スコア", "z-score"], "平均を 0、ばらつきを 1 にそろえること。違う条件の数字を同じ物差しで比べられる。", "<p>例: 上がり 3F を“そのレースの平均からどれだけ速いか”に直すと、レースが違っても比べられる。</p>"),
    T("shrink", "縮約・経験ベイズ", "feature", ["縮約", "経験ベイズ", "ベイズ"], "件数が少ない平均を、全体の平均に引き寄せて安定させる方法。",
      "<p>たとえば 2 戦 2 勝の馬の“勝率 100%”はあてにならない。全体の勝率に寄せて、件数が増えるほど本来の値を信じる。</p>", "shrink", ["feature_strategy_tp03_tp04"]),
    T("fixed", "固定効果", "feature", ["固定効果"], "条件ごとの“固定的な差”（例: 斤量 1kg あたり何秒遅くなるか）を推定して、補正に使うこと。"),
    T("ewm", "EWM（指数加重移動平均）", "feature", ["EWM", "ewm"], "新しいものほど重くして平均すること。直近の調子を重視できる。", "<p>特徴量名の ewm2・ewm6 は“半減期”の違い。小さいほど直近だけを強く見る。</p>", "ewm", ["feature_strategy_tp03_tp04"]),
    T("elo", "Elo・Glicko・TrueSkill", "feature", ["Elo", "Glicko", "TrueSkill"], "対戦結果から“強さの数値”を更新していくレーティング。",
      "<p>強い相手に勝てば大きく上がり、格下に負ければ大きく下がる。相手関係（誰に勝ったか）を反映できるのが利点。</p>", "elo", ["feature_strategy_tp03_tp04"]),
    T("rpci", "RPCI・PCI（ペース指数）", "feature", ["RPCI", "PCI"], "前半と後半の速さの配分を示す指数。前傾（前半が速い）か後傾かが分かる。",
      "<p>このプロジェクトでは“ラスト 3F を除く 1F 平均 ÷ ラスト 3F の 1F 平均 × 50”で自前に定義した（JRA の公式式とは一致しない可能性がある）。</p>", "", ["feature_strategy_tp03_tp04"]),
    T("timefigure", "TimeFigure", "feature", ["TimeFigure"], "走破タイムを、距離・馬場などをならして“能力値”に直したもの。", "<p>同じ 1:34.5 でも、重い馬場と軽い馬場では価値が違う。それをならして異なるレース間で比べられるようにする。</p>", "", ["race_performance_rating_system"]),
    T("quantile", "分位点・外れ値", "feature", ["分位点", "外れ値"], "分位点 = 小さい順に並べたときの何割目の値。外れ値 = ほかから大きく離れた値。", "<p>外れ値の影響を抑えるために、分位点で上下を切る（クリップ）ことがある。</p>"),
    T("corr", "相関・多重共線性", "feature", ["多重共線性", "相関"], "相関 = 2 つの数字が一緒に動く度合い。似た特徴量が複数あると、効果を正しく分けられない（多重共線性）。", "<p>相関が約 0.9 以上のものは代表を 1 つ残す、という基準を ablation で使っている。</p>"),
    T("missing", "欠損（NA）", "feature", ["欠損", r"\bNA\b"], "値が入っていないこと。“理由”によって扱いが変わる。",
      "<p>初出走で過去走がない（構造上ありえない）、データを取得できていない、加工で落ちた、などが混ざる。欠損の原因を分けて確認する監査ノートブック（tp-01b）がある。</p>", "", ["training_pipeline_decisions"]),
    T("latent", "潜在変数・埋め込み・クラスタ", "feature", ["潜在", "埋め込み", "クラスタ"], "直接は観測できない“隠れた性質”を数値（ベクトル）で表したもの。似たもの同士をまとめるのがクラスタ。", "<p>例: 血統の似ている馬を同じクラスタにして、血統の“タイプ”として使う。</p>"),
    # このプロジェクト
    T("layers", "LayerA / LayerB", "project", ["LayerA", "LayerB"], "LayerA = 市場（オッズ）に頼らない確率モデル。LayerB = 市場と合成して期待値・買い目にする層。",
      "<p>オッズ・人気は LayerA に入れない。入れると“市場の写し”になり、独自の見立て（市場とのズレ）が消えるため。</p>", "layers", ["modeling_strategy_tp04_tp07", "pre_race_feature_whitelist"]),
    T("stage", "Stage0〜7（多段階パイプライン）", "project", [r"Stage\d", "Stage"], "最終確率を作るまでの段階（母表 → 潜在表現 → 位置取り → ペース → 馬場 → 血統適性 → 最終確率 → 市場統合）。", "<p>段階ごとに原因を切り分けられるのが利点。</p>", "", ["staged_probability_pipeline"]),
    T("tier", "T-X と tier", "project", [r"T-\d+", "T-X", "tier"], "T-X = 発走の X 分前。tier = 特徴量が“いつ分かるか”の区分（0=前日、1=当日朝、2=T-60 の馬体重の発表後）。",
      "<p>情報は時間とともに増える。tier で分けておけば、朝にまとめて作っておき、T-60 では tier 2 だけを計算し直す、といった増分計算ができる。</p>", "tiers", ["feature_strategy_tp03_tp04", "feature_serving_low_memory"]),
    T("inference", "推論モード", "project", ["推論モード"], "当日の出走馬を“結果がまだ無い行”として履歴に足し、学習と同じ計算で特徴量を作る仕組み。", "<p>学習と推論で計算が食い違う（スキュー）のを防ぐため、同じコードを使う。毎日の蓄積（07:40）で食い違いを記録する。</p>", "", ["feature_serving_low_memory"]),
    T("flat", "flat 表・Parquet", "project", ["flat", "Parquet"], "flat 表 = JSON を“1 行 1 馬”に平らにした表。Parquet = 列ごとに圧縮して速く読める表形式のファイル。", "<p>race_result_flat などが履歴特徴量の入力になる。毎日 07:00 に当年分が増分更新される。</p>", "", ["data_flow_schedule"]),
    T("gcs", "GCS・MLflow", "project", ["GCS", "MLflow"], "GCS = Google のクラウドストレージ（取得データの正本）。MLflow = 実験の記録（パラメータ・指標）を残す道具。", "<p>取得した JSON は GCS に保存。学習の実験は MLflow のサーバに記録され、ホストの 23230 番ポートで見られる。</p>", "", ["data_flow_schedule", "modeling_experiment_report"]),
    T("tp", "tp-00〜tp-08", "project", [r"tp-0\d[a-z]?"], "学習パイプラインの notebook の番号。00 環境確認 → 01 データ分割 → 02 特徴量 → 03 ablation → 04 前段モデル → 05 最終モデル → 06 較正・評価 → 07 登録 → 08 馬券の強化学習。", "<p>確定した事項は“確定事項ログ”に番号ごとに記録される。</p>", "", ["training_pipeline_decisions"]),
    T("fitend", "FIT_END（全体フィットの上限日）", "project", ["FIT_END"], "全期間にまたがる値（縮約の強さ、分位点など）を決めるときに使ってよいデータの上限日（2021-12-21）。", "<p>これ以降のデータは“値を決める”ためには使わない。評価期間の情報が、特徴量の定義に混ざらないようにするため。</p>", "leak", ["feature_strategy_tp03_tp04"]),
    # 期待値・馬券の検証
    T("evcalibration", "期待値の較正", "prob", ["期待値の較正", "較正後の期待値", "期待回収率（較正後）", "EV の較正"],
      "モデルが出した期待値を、過去に実際に買ったときの回収率に合わせて補正すること。表示した期待値と実現が一致するようにする。",
      "<p><b>たとえ:</b> 体重計が常に 2kg 重く出るなら、その分を引いて表示する。モデルの期待値は市場に勝てていないぶん実現より高く出る（馬連で平均 1.07 に対し実現 0.58）ので、過去の実績との対応表（単調回帰）で補正する。</p>"
      "<p><b>このプロジェクトでは:</b> 年 Y の補正は Y より前の年だけで当てはめる。画面には「期待回収率（較正後）」と、過去の年別の実現回収率の範囲を併記する（tp-49）。較正後は期待値のほとんどが 0.9〜1.0 に収まり、「モデルに優位がある」とは言えない水準になる。</p>", "calibration", ["ev_calibration_adopt", "ev_calibration_audit"]),
    T("isotonic", "単調回帰（isotonic regression）", "model", ["単調回帰", "isotonic", "Isotonic"],
      "「予測が高いほど結果も高い」という向きだけを守って、階段状に当てはめる回帰。形の仮定が少ない。",
      "<p>期待値の較正で、「予測 EV の帯ごとの実現回収率」を、順序を崩さずに滑らかにならして対応表にするのに使う。直線を仮定する線形の縮約より柔軟で、データが少ない帯では不安定になるので、保守案へのフォールバックを持たせる。</p>", "", ["ev_calibration_adopt"]),
    T("winnerscurse", "選抜効果（winner's curse）", "validate", ["winner's curse", "winners curse", "選抜効果", "勝者の呪い"],
      "良く見えたものを選ぶと、選んだ時点の値は偶然の上振れを含み、次は平均に戻りやすい現象。",
      "<p>期待値が高い買い目を選ぶと、選んだ時点の期待値はモデルの誤差のうち「たまたま高く出た分」を含む。だから較正は全馬ではなく、<b>実際に選ばれる買い目</b>で当てはめる。特徴量の探索でも、第 1 段で通った群が第 2 段で縮むのは同じ理由で、2 段階の確認はこれを抑えるためにある。</p>", "", ["ev_calibration_adopt", "feature_investigation_catalog"]),
    T("realitycheck", "Reality Check（多重比較の補正）", "validate", ["Reality Check", "reality check", "リアリティチェック"],
      "多数の戦略を試して最良のものを選ぶと、偶然でも良く見える。その選抜を補正して、最良の戦略が本当に基準を超えているかを検定する方法（White 型）。",
      "<p>戦略を 3,000 本試せば、偶然だけで回収率の高いものが出る。reality check は、全戦略の成績のばらつきをレース単位の取り直し（ブートストラップ）で再現し、「最良の戦略の成績が、偶然の最良として十分ありうるか」を p 値で返す。p が大きければ（例: 0.86〜1.00）、見かけの優位は偶然の範囲。</p>"
      "<p><b>このプロジェクトでは:</b> 馬券の調査（tp-36・43・44・45・48）で、試した戦略の総数を記録したうえで必ず適用する。区間下限が 100% を超えたものは無く、p はいずれも小さくなかった。</p>", "", ["betting_edge_search", "trifecta_true_odds"]),
    T("walkforward", "walk-forward（前向き検証）", "validate", ["walk-forward", "walk forward", "ウォークフォワード"],
      "評価する年より前のデータだけで学習・較正し、その年を評価することを、年を進めながら繰り返す検証。",
      "<p>未来の情報が過去の判断に混ざらないので、実際に運用したときの成績に近い。たとえば 2024 年の評価には 2023 年までで当てはめた較正や方策だけを使い、2026 年のような未使用の年は、方策を事前に固定したうえで 1 回だけ評価する。</p>", "expanding", ["betting_edge_search", "ev_calibration_adopt"]),
    T("offlinerl", "オフライン強化学習・反実仮想", "model", ["オフライン強化学習", "オフライン RL", "反実仮想"],
      "実際に試さなくても、過去の記録から「別の行動を取っていたらどうなったか」を評価して方策を学ぶ強化学習。",
      "<p>馬券では、過去の全レース・全買い目の結果（払戻）が既知なので、「その買い目を買っていたら」の報酬を直接計算できる（full-information）。普通の強化学習のように実際に買って探索する必要がなく、保守的 Q 学習（CQL）などで「データにない行動を過大評価しない」ように学習する。</p>", "", ["betting_policy_reinforcement_learning", "betting_edge_search"]),
    T("takeout", "控除率・払戻率", "prob", ["控除率", "払戻率"],
      "賭け金のうち主催者が取る割合（控除率）と、払い戻される割合（払戻率）。JRA は単勝・複勝が 20%（払戻率 80%）、馬連・ワイドが 22.5%、馬単・3 連複が 25%、3 連単が 27.5%。",
      "<p>全員が市場どおりに均等に買うと、長期の回収率は払戻率に収束する。だから期待値が 1 を超えるには、市場の価格より当たりやすい買い目を見つけて、この差を取り返す必要がある。オッズの Σ(1/オッズ) は払戻率の逆数（単勝で 1.25）になるので、オッズの品質の検査にも使う。</p>", "", ["ev_calibration_audit", "trifecta_true_odds"]),

]

BY_ID = {t.id: t for t in TERMS}


# ── 本文への注釈付け ──────────────────────────────────────────────────────────

_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
_SKIP_TAGS = {"code", "pre", "a", "h1", "h2", "h3", "h4", "h5", "h6", "svg", "script", "style", "button", "summary", "textarea", "title", "option"}
_TOKEN = re.compile(r"(<!--.*?-->|<[^>]*>)", re.S)
_TAGNAME = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)")


def _alt(pat: str) -> str:
    pre = r"(?<![A-Za-z0-9_])" if _ASCII_ALNUM.match(pat[:1]) else ""
    post = r"(?![A-Za-z0-9_])" if pat[-1:].isascii() and pat[-1:] not in ")" else ""
    return f"{pre}(?:{pat}){post}"


def _compile() -> tuple[re.Pattern, list[Term]]:
    parts = []
    for i, term in enumerate(TERMS):
        alts = "|".join(_alt(a) for a in sorted(term.pats, key=len, reverse=True))
        parts.append(f"(?P<g{i}>{alts})")
    return re.compile("|".join(parts)), TERMS


_RX, _ORDER = _compile()


def annotate(body: str, used: dict[str, set[str]], slug: str) -> str:
    """本文の中の用語のうち、章ごとに最初の 1 回だけを押せるボタンにする。コード・リンク・見出し・図・折りたたみの見出しは触らない。
    used[用語id] にこの章の slug を足す。"""
    seen: set[str] = set()
    out: list[str] = []
    skip_name, skip_depth = None, 0
    for tok in _TOKEN.split(body):
        if not tok:
            continue
        if tok.startswith("<"):
            m = _TAGNAME.match(tok)
            if m and not tok.startswith("<!--"):
                closing, name = m.group(1) == "/", m.group(2).lower()
                selfclose = tok.endswith("/>")
                if skip_name:
                    if name == skip_name and not selfclose:
                        skip_depth += -1 if closing else 1
                        if skip_depth == 0:
                            skip_name = None
                elif not closing and not selfclose and (name in _SKIP_TAGS or re.search(r'class="[^"]*\b(?:mermaid|gl-card|gl-fig|no-gl)\b', tok)):
                    skip_name, skip_depth = name, 1
            out.append(tok)
            continue
        if skip_name:
            out.append(tok)
            continue

        def sub(m: re.Match) -> str:
            term = _ORDER[int(m.lastgroup[1:])]
            if term.id in seen:
                return m.group(0)
            seen.add(term.id)
            used.setdefault(term.id, set()).add(slug)
            return f'<button type="button" class="gl-term" data-gl="{term.id}" title="クリックで用語の説明を開く">{m.group(0)}</button>'

        out.append(_RX.sub(sub, tok))
    return "".join(out)


# ── 用語集の章 ──────────────────────────────────────────────────────────────

GLOSSARY_SLUG = "glossary"
GLOSSARY_TITLE = "用語集（データサイエンスの専門用語）"
GLOSSARY_BLURB = ("本文に出てくる専門用語の解説。本文の点線の下線がついた語をクリックするとその場に説明が開く。ここでは同じ説明を分野別にまとめ、"
                  "各用語が使われている章へのリンクも付ける。")


def render_chapter(used: dict[str, set[str]], chapter_label: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """用語集の章の本文(HTML)と、節(id, 見出し)の一覧を返す。chapter_label は slug → 『第N章 タイトル』。"""
    secs, parts = [], ['<p>用語は分野別に並べている。見出しをクリックすると説明（たとえ・このプロジェクトでの使われ方・図）が開く。'
                       '「使われている章」から、実際にその語が出てくる章へ飛べる。</p>'
                       '<p><button type="button" class="gl-all" data-open="1">すべて開く</button> <button type="button" class="gl-all" data-open="0">すべて閉じる</button> '
                       '<input type="search" class="gl-q" placeholder="用語をしぼり込む" aria-label="用語をしぼり込む"></p>']
    for cid, ctitle in CATS:
        items = [t for t in TERMS if t.cat == cid]
        if not items:
            continue
        hid = f"{GLOSSARY_SLUG}--cat-{cid}"
        secs.append((hid, ctitle))
        parts.append(f'<h2 id="{hid}">{html.escape(ctitle)}</h2>')
        for t in items:
            fig = f'<div class="gl-fig">{DIAGRAMS[t.diagram]()}</div>' if t.diagram else ""
            used_in = "".join(f'<a href="#ch-{s}">{html.escape(chapter_label[s])}</a>' for s in chapter_label if s in used.get(t.id, ()))
            see = "".join(f'<a href="#ch-{s}">{html.escape(chapter_label[s])}</a>' for s in t.see if s in chapter_label)
            foot = ""
            if see:
                foot += f'<p class="gl-links"><b>詳しくは:</b> {see}</p>'
            if used_in:
                foot += f'<p class="gl-links"><b>使われている章:</b> {used_in}</p>'
            parts.append(
                f'<details class="gl-item" id="gl-{t.id}" data-name="{html.escape(t.name + " " + " ".join(t.pats), quote=True)}">'
                f'<summary><b>{html.escape(t.name)}</b><span class="gl-short">{html.escape(t.short)}</span></summary>'
                f'<div class="gl-body"><p class="gl-one">{html.escape(t.short)}</p>{fig}{t.body}{foot}</div></details>'
            )
    return "".join(parts), secs
