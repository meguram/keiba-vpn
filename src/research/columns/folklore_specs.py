"""めぐコラム『通説シリーズ』の通説の定義（1 記事 = 1 Spec）。集計と記事の形は folklore.py。

通説の出どころ: docs/html/modeling/chapters/folklore_hypotheses.html（A〜K の ID をコメントに残す）。
条件はすべて予想の時点で分かるもの（前走まで・出馬表・追い切り）。今走の結果やオッズは条件に使わない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.columns.folklore import Axis, Spec

ALL = "JRA全競馬場の平地（障害除く）"


def has_prev(d: pd.DataFrame) -> pd.Series:
    return (d["nstart"] > 0) & d["interval"].notna()


def cls_band(d: pd.DataFrame) -> pd.Series:
    c = d["cls"]
    return pd.Series(np.select([c == 0, c == 1, c.isin([2, 3]), c >= 4], ["新馬・未勝利", "1勝クラス", "2勝・3勝", "オープン以上"], None), index=d.index)


CLS = ["新馬・未勝利", "1勝クラス", "2勝・3勝", "オープン以上"]


def surf_dist(d: pd.DataFrame) -> pd.Series:
    band = pd.cut(d["distance"], [0, 1400, 1800, 2200, 9999], labels=["短距離", "マイル", "中距離", "長距離"]).astype(str)
    return d["surface"].str.replace("ダート", "ダ") + band


SD = ["芝短距離", "芝マイル", "芝中距離", "芝長距離", "ダ短距離", "ダマイル", "ダ中距離"]


def pos_band(col: str):
    def f(d: pd.DataFrame) -> pd.Series:
        q = d[col]
        return pd.Series(np.select([q <= 0.25, q <= 0.6, q > 0.6], ["前寄り", "中団", "後方"], None), index=d.index)
    return f


POS = ["前寄り", "中団", "後方"]


def surface(d: pd.DataFrame) -> pd.Series:
    return d["surface"]


def pop_band(d: pd.DataFrame) -> pd.Series:
    p = d["popularity"]
    return pd.Series(np.select([p <= 3, p <= 6, p > 6], ["1〜3番人気", "4〜6番人気", "7番人気以下"], None), index=d.index)


POPS = ["1〜3番人気", "4〜6番人気", "7番人気以下"]
REGION = {"東京": "中央", "中山": "中央", "京都": "中央", "阪神": "中央", "札幌": "ローカル", "函館": "ローカル", "福島": "ローカル",
          "新潟": "ローカル", "中京": "ローカル", "小倉": "ローカル"}
HARD = ["一杯", "強め", "G強", "G一", "直強", "直一"]


def overseas_region(d: pd.DataFrame) -> pd.Series:
    v = d["prev_overseas_venue"].fillna("")
    hk = v.isin(["シャティン", "ハッピーバレー"])
    me = v.isin(["メイダン", "キングアブドゥルアジーズ", "カタール", "アルライヤン"])
    us = v.str.contains("サンタアニタ|チャーチル|デルマー|キーンランド|サラトガ|アメリカ|ベルモント|ガルフストリーム|ランドウィック|フレミントン|ムーニーバレー|ローズヒル|コーフィールド")
    eu = v.str.contains("ロンシャン|シャンティイ|ドーヴィル|フランス|アスコット|ヨーク|ニューマーケット|エプソム|グッドウッド|カラ|レパーズタウン|バーデン|ケルン|ミュンヘン|ニューベリー|ドンカスター|サンクルー|イギリス|アイルランド|ドイツ")
    return pd.Series(np.select([hk, me, eu, us], ["香港", "中東", "欧州", "北米・豪州"], None), index=d.index)


SPECS: list[Spec] = [
    # ── ローテーション: 距離 ──
    Spec(  # A01
        slug="distance-extension", title="距離延長は走る？", claim="距離延長の馬は好走する",
        categories=["travel/distance"],
        universe=lambda d: has_prev(d) & (d["surface"] == d["p_surface"]),
        groups=[("距離延長\n（全体）", lambda d: d["dist_diff"] >= 100),
                ("+100〜200m", lambda d: d["dist_diff"].between(100, 200)),
                ("+300〜400m", lambda d: d["dist_diff"].between(300, 400)),
                ("+500m〜", lambda d: d["dist_diff"] >= 500)],
        axis=Axis("クラス", cls_band, CLS, "距離延長の馬をクラスで分けると"),
        ref_label="同距離・短縮の馬",
        question="前走より距離が延びる馬は、人気以上に走るのか。クラスで違うか。",
        conditions=ALL + "。前走と同じ面（芝→芝、ダ→ダ）の出走",
        definition="距離延長 = 前走より 100m 以上長い。比べる相手は、前走と同じ距離か短くなった馬です。"),
    Spec(  # A02
        slug="distance-shortening", title="距離短縮は走る？", claim="距離短縮の馬は好走する（前走で先行して止まった馬は特に）",
        categories=["travel/distance"],
        universe=lambda d: has_prev(d) & (d["surface"] == d["p_surface"]),
        groups=[("距離短縮\n（全体）", lambda d: d["dist_diff"] <= -100),
                ("−100〜200m", lambda d: d["dist_diff"].between(-200, -100)),
                ("−300〜400m", lambda d: d["dist_diff"].between(-400, -300)),
                ("−500m〜", lambda d: d["dist_diff"] <= -500)],
        axis=Axis("前走の4角の位置", pos_band("p_q4"), POS, "距離短縮の馬を、前走の4角の位置で分けると"),
        ref_label="同距離・延長の馬",
        question="前走より距離が短くなる馬は、人気以上に走るのか。前走で先行した馬は特にそうか。",
        conditions=ALL + "。前走と同じ面の出走",
        definition="距離短縮 = 前走より 100m 以上短い。4角の位置は前走の最後のコーナーの順位を頭数で割った値（前 1/4 = 前寄り）。"),
    # ── ローテーション: コース替わり ──
    Spec(  # A08
        slug="direction-switch", title="右回り⇄左回りの替わりは割引？", claim="回りが替わると成績が落ちる",
        categories=["travel/course-change"], direction=-1,
        universe=lambda d: has_prev(d) & d["p_direction"].isin(["右", "左"]) & d["direction"].isin(["右", "左"]),
        groups=[("回りが替わる", lambda d: d["direction"] != d["p_direction"]),
                ("右→左", lambda d: (d["p_direction"] == "右") & (d["direction"] == "左")),
                ("左→右", lambda d: (d["p_direction"] == "左") & (d["direction"] == "右"))],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="同じ回りの馬",
        question="前走と回り（右・左）が替わる馬は、人気ほど走らないのか。",
        definition="新潟の直線 1000m は回りの判定から外しています。"),
    Spec(  # A32
        slug="local-to-central", variants=[], title="ローカル⇄中央の替わりで成績は変わる？", claim="ローカルで好走した馬は中央場で通用しない",
        categories=["travel/course-change"], direction=-1,
        universe=lambda d: has_prev(d) & d["p_venue"].isin(list(REGION)) & (d["p_finish_position"].between(1, 3)),
        groups=[("ローカルで好走→中央", lambda d: (d["p_venue"].map(REGION) == "ローカル") & (d["venue"].map(REGION) == "中央")),
                ("中央で好走→ローカル", lambda d: (d["p_venue"].map(REGION) == "中央") & (d["venue"].map(REGION) == "ローカル"))],
        axis=Axis("クラス", cls_band, CLS),
        ref_label="同じ区分の場が続く馬",
        question="前走ローカル（札幌・函館・福島・新潟・中京・小倉）で 3 着以内だった馬は、中央 4 場（東京・中山・京都・阪神）で人気ほど走らないのか。",
        conditions=ALL + "。前走 3 着以内の馬"),
    # ── ローテーション: 間隔・叩き ──
    Spec(  # A12
        slug="second-start-after-layoff", title="叩き2戦目は上積みがある？", claim="休み明けを叩いた 2 戦目は上積みがある",
        categories=["travel/interval"],
        universe=lambda d: d["since_layoff"] >= 1,
        groups=[("叩き2戦目", lambda d: d["since_layoff"] == 2),
                ("休み明け初戦", lambda d: d["since_layoff"] == 1),
                ("叩き3戦目", lambda d: d["since_layoff"] == 3),
                ("4戦目以降", lambda d: d["since_layoff"] >= 4)],
        axis=Axis("年齢", lambda d: pd.Series(np.select([d["age"] <= 3, d["age"] <= 5, d["age"] > 5], ["2〜3歳", "4〜5歳", "6歳以上"], None), index=d.index),
                  ["2〜3歳", "4〜5歳", "6歳以上"]),
        ref_label="休み明け初戦・3戦目以降",
        question="休み明け（中 10 週以上）を 1 回使ったあとの 2 戦目は、人気以上に走るのか。",
        definition="休み明け = 前走から 70 日以上あいた出走。そこから数えて 2 走目が叩き 2 戦目です。"),
    Spec(  # A14
        slug="quick-turnaround", title="連闘・中1週の馬は走る？", claim="連闘・中 1 週で使ってくる馬は勝負気配",
        categories=["travel/interval"],
        universe=has_prev,
        groups=[("連闘・中1週", lambda d: d["interval"] <= 14),
                ("連闘（中0週）", lambda d: d["interval"] <= 7),
                ("中1週", lambda d: d["interval"].between(8, 14)),
                ("中2〜3週", lambda d: d["interval"].between(15, 28))],
        axis=Axis("前走の着順", lambda d: pd.Series(np.select([d["p_finish_position"].between(1, 3), d["p_finish_position"].between(4, 9), d["p_finish_position"] >= 10], ["前走1〜3着", "前走4〜9着", "前走10着〜"], None), index=d.index),
                  ["前走1〜3着", "前走4〜9着", "前走10着〜"]),
        ref_label="中2週以上の馬",
        question="前走から 2 週間以内に使ってくる馬は、人気以上に走るのか。"),
    Spec(  # A13
        slug="layoff-first-start", title="休み明け初戦は割引？", claim="休み明けの初戦は走らない",
        categories=["travel/interval"], direction=-1,
        universe=has_prev,
        groups=[("休み明け初戦\n（中10週〜）", lambda d: d["interval"] >= 70),
                ("中10週〜半年", lambda d: d["interval"].between(70, 180)),
                ("半年〜1年", lambda d: d["interval"].between(181, 365)),
                ("1年以上", lambda d: d["interval"] > 365)],
        axis=Axis("調教師の前年の勝利数", lambda d: pd.Series(np.select([d["trank"] <= 20, d["trank"] <= 100, d["trank"] > 100], ["上位20人", "21〜100位", "101位〜"], None), index=d.index),
                  ["上位20人", "21〜100位", "101位〜"], "休み明け初戦を、調教師の格で分けると"),
        ref_label="間隔が 10 週未満の馬",
        question="前走から 10 週以上あいた馬は人気ほど走らないのか。休み明けに強い厩舎はあるか。"),
    # ── ローテーション: 海外帰り ──
    Spec(  # A34〜A36
        slug="overseas-return", title="海外帰りの初戦は割引？", claim="海外遠征から帰った初戦は走らない",
        categories=["travel/overseas"], direction=-1,
        universe=has_prev,
        groups=[("海外帰りの初戦", lambda d: d["prev_overseas"].eq(True)),
                ("香港帰り", lambda d: overseas_region(d) == "香港"),
                ("中東帰り", lambda d: overseas_region(d) == "中東"),
                ("欧州帰り", lambda d: overseas_region(d) == "欧州"),
                ("北米・豪州帰り", lambda d: overseas_region(d) == "北米・豪州")],
        ref_label="国内の前走から使う馬",
        question="海外のレースを走って帰ってきた馬の、国内の初戦は人気ほど走らないのか。遠征先で違うか。",
        conditions=ALL + "。出馬表の過去 5 走で、直前の 1 走が海外のレースだった馬",
        definition="海外帰り = 出馬表に載る直前の 1 走が海外（メイダン・シャティン・ロンシャンなど）。重賞級の馬が多いので件数は少なめです。"),
    # ── ローテーション: クラス ──
    Spec(  # A15
        slug="class-up-first", title="昇級初戦は通用しない？", claim="昇級初戦は通用しない（勝ち方次第）",
        categories=["travel/class"], direction=-1,
        universe=lambda d: has_prev(d) & d["cls"].between(1, 4) & d["p_cls"].notna(),
        groups=[("昇級初戦", lambda d: (d["p_finish_position"] == 1) & (d["cls_diff"] == 1)),
                ("前走1番人気で勝って昇級", lambda d: (d["p_finish_position"] == 1) & (d["cls_diff"] == 1) & (d["p_popularity"] == 1)),
                ("前走4番人気以下で勝って昇級", lambda d: (d["p_finish_position"] == 1) & (d["cls_diff"] == 1) & (d["p_popularity"] >= 4))],
        axis=Axis("昇級したクラス", cls_band, ["1勝クラス", "2勝・3勝", "オープン以上"]),
        ref_label="同じクラスを走る馬",
        question="前走で勝ってクラスが上がった馬は、人気ほど走らないのか。",
        definition="昇級初戦 = 前走 1 着で、今回が 1 つ上のクラス。"),
    # ── 前走の内容 ──
    Spec(  # A17
        slug="close-loss-last-time", title="前走は僅差の負け。次は走る？", claim="前走で 0.5 秒以内に負けた馬は次走好走する",
        categories=["prev/result"],
        universe=lambda d: has_prev(d) & (d["p_finish_position"] >= 2) & d["p_margin_sec"].notna(),
        groups=[("前走0.5秒以内の負け", lambda d: d["p_margin_sec"] <= 0.5),
                ("0.0〜0.2秒", lambda d: d["p_margin_sec"] <= 0.2),
                ("0.3〜0.5秒", lambda d: d["p_margin_sec"].between(0.3, 0.5)),
                ("0.6〜1.0秒", lambda d: d["p_margin_sec"].between(0.6, 1.0))],
        axis=Axis("前走の人気", lambda d: pd.Series(np.select([d["p_popularity"] <= 3, d["p_popularity"] <= 6, d["p_popularity"] > 6], POPS, None), index=d.index), POPS,
                  "前走0.5秒以内の負けを、前走の人気で分けると"),
        ref_label="前走で 0.6 秒以上負けた馬",
        question="前走で勝ち馬から 0.5 秒以内に負けた馬は、人気以上に走るのか。",
        conditions=ALL + "。前走 2 着以下の馬"),
    Spec(  # A18
        slug="fastest-closer-loser", title="前走上がり最速で負けた馬は次に来る？", claim="前走で上がり最速なのに負けた馬は次走来る",
        categories=["prev/result"],
        universe=lambda d: has_prev(d) & (d["p_finish_position"] >= 4),
        groups=[("前走上がり最速で4着以下", lambda d: d["p_l3f_rank"] == 1),
                ("上がり2〜3位", lambda d: d["p_l3f_rank"].between(2, 3))],
        axis=Axis("前走の4角の位置", pos_band("p_q4"), POS),
        ref_label="前走上がり 4 位以下で 4 着以下の馬",
        question="前走で上がり 3F が最も速かったのに 4 着以下だった馬は、次に人気以上に走るのか。",
        conditions=ALL + "。前走 4 着以下の馬"),
    Spec(  # A23
        slug="big-loss-rebound", title="前走大敗からの巻き返しは『替わり』次第？", claim="前走大敗の馬は、条件が替われば巻き返す",
        categories=["prev/result"],
        universe=lambda d: has_prev(d) & (d["p_finish_position"] >= 10),
        groups=[("替わりあり（1つ以上）", lambda d: changes(d) >= 1),
                ("替わり1つ", lambda d: changes(d) == 1),
                ("替わり2つ以上", lambda d: changes(d) >= 2)],
        axis=Axis("前走の人気", lambda d: pd.Series(np.select([d["p_popularity"] <= 3, d["p_popularity"] <= 6, d["p_popularity"] > 6], POPS, None), index=d.index), POPS),
        ref_label="替わりのない馬",
        question="前走 10 着以下に負けた馬は、距離・芝ダート・騎手・馬場のどれかが替われば、人気以上に巻き返すのか。",
        conditions=ALL + "。前走 10 着以下の馬",
        definition="替わり = 距離（±200m 以上）・芝ダート・騎手・馬場（良⇄稍重以上）のうち、前走から変わったものの数。"),
    Spec(  # A22
        slug="bad-going-excuse", title="道悪で負けた馬の良馬場替わりは買い？", claim="前走の道悪で凡走した馬は良馬場で巻き返す",
        categories=["prev/result"],
        universe=lambda d: has_prev(d) & (d["p_finish_position"] >= 4) & (d["track_condition"] == "良"),
        groups=[("前走重・不良で4着以下→今回良", lambda d: d["p_track_condition"].isin(["重", "不良"])),
                ("前走稍重で4着以下→今回良", lambda d: d["p_track_condition"] == "稍重")],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="前走も良で 4 着以下の馬",
        question="前走の重・不良馬場で負けた馬は、良馬場に替わって人気以上に走るのか。",
        conditions=ALL + "。今回が良馬場で、前走 4 着以下の馬"),
    Spec(  # A20
        slug="high-pace-survivor", title="前走ハイペースを先行して粘った馬は強い？", claim="前走ハイペースを先行して好走した馬は次も走る",
        categories=["prev/pace"],
        universe=lambda d: has_prev(d) & d["p_pace_rel"].notna() & (d["p_finish_position"].between(1, 5)),
        groups=[("ハイペースを先行して5着以内", lambda d: (d["p_pace_rel"] <= -1.0) & (d["p_q4"] <= 0.3)),
                ("スローを先行して5着以内", lambda d: (d["p_pace_rel"] >= 1.0) & (d["p_q4"] <= 0.3)),
                ("ハイペースを後方から5着以内", lambda d: (d["p_pace_rel"] <= -1.0) & (d["p_q4"] >= 0.6)),
                ("スローを後方から5着以内", lambda d: (d["p_pace_rel"] >= 1.0) & (d["p_q4"] >= 0.6))],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="前走 5 着以内のほかの馬",
        question="前走が前に厳しいハイペースだったのに、先行して 5 着以内に粘った馬は、人気以上に走るのか（展開に逆らった好走は本物か）。",
        conditions=ALL + "。前走 5 着以内の馬",
        definition="ハイペース = 前半 3F と後半 3F の差が、そのコースの平均より 1 秒以上前寄り（前半が速い）。スローはその逆。先行 = 前走の 4 角で前 3 割。"),
    Spec(  # A21
        slug="slow-pace-closer", title="スローで差して届かずの馬は過大評価？", claim="前走スローで差して届かなかった馬は、次走で人気ほど走らない",
        categories=["prev/pace"], direction=-1,
        universe=lambda d: has_prev(d) & d["p_pace_rel"].notna() & (d["p_finish_position"] >= 2),
        groups=[("スローを後方から上がり3位以内で負け", lambda d: (d["p_pace_rel"] >= 1.0) & (d["p_q4"] >= 0.6) & (d["p_l3f_rank"] <= 3))],
        axis=Axis("前走の着順", lambda d: pd.Series(np.select([d["p_finish_position"].between(2, 3), d["p_finish_position"].between(4, 5), d["p_finish_position"] >= 6], ["前走2〜3着", "前走4〜5着", "前走6着〜"], None), index=d.index),
                  ["前走2〜3着", "前走4〜5着", "前走6着〜"]),
        ref_label="前走で負けたほかの馬",
        question="前走がスローで、後ろから速い上がりを使って届かなかった馬は、次走で人気ほど走らないのか。",
        conditions=ALL + "。前走 2 着以下の馬"),
    Spec(  # A19
        slug="deep-closer-board", title="前走4角後方から掲示板の馬は次に勝ち負け？", claim="前走 4 角 10 番手以下から掲示板に来た馬は次走で勝ち負けになる",
        categories=["prev/pace"],
        universe=lambda d: has_prev(d) & (d["p_finish_position"].between(1, 5)),
        groups=[("前走4角10番手以下から5着以内", lambda d: (d["p_q4"] * (d["p_starters"] - 1) + 1) >= 10)],
        axis=Axis("頭数", lambda d: pd.Series(np.select([d["starters"] <= 12, d["starters"] <= 15, d["starters"] > 15], ["12頭以下", "13〜15頭", "16頭以上"], None), index=d.index),
                  ["12頭以下", "13〜15頭", "16頭以上"]),
        ref_label="前走 5 着以内のほかの馬",
        question="前走で 4 角 10 番手以下の後方から 5 着以内に来た馬は、人気以上に走るのか。",
        conditions=ALL + "。前走 5 着以内の馬"),
    Spec(  # G06
        slug="lone-front-runner", title="逃げ馬が1頭だけのレースは残る？", claim="同型のいない逃げ馬は残る",
        categories=["style/front-runner"],
        universe=lambda d: has_prev(d) & (d["p_q1"] == 0),
        groups=[("同型なし（前走逃げが自分だけ）", lambda d: d.groupby("race_id")["p_q1"].transform(lambda s: (s == 0).sum()) == 1),
                ("同型1頭", lambda d: d.groupby("race_id")["p_q1"].transform(lambda s: (s == 0).sum()) == 2),
                ("同型2頭以上", lambda d: d.groupby("race_id")["p_q1"].transform(lambda s: (s == 0).sum()) >= 3)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="同型がいる馬",
        question="前走で最初のコーナーを先頭で回った馬が、今回そのような馬が自分だけのとき、人気以上に残るのか。",
        conditions=ALL + "。前走で最初のコーナーを先頭で通過した馬",
        definition="同型 = 同じレースで、前走の最初のコーナーを先頭で通過した馬。"),
    # ── 実績 ──
    Spec(  # A26
        slug="course-specialist", title="コース巧者は人気以上に走る？", claim="同じコースで好走経験がある馬は人気以上に走る",
        categories=["ability/record"],
        universe=lambda d: has_prev(d) & (d["course_runs_before"] >= 1),
        groups=[("同コースで3着内あり", lambda d: d["course_top3_before"] >= 1),
                ("3着内1回", lambda d: d["course_top3_before"] == 1),
                ("3着内2回以上", lambda d: d["course_top3_before"] >= 2)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="同コースを走ったが 3 着内のない馬",
        question="同じ競馬場・面・距離で 3 着以内に来たことのある馬は、人気以上に走るのか。",
        conditions=ALL + "。同じコースを走ったことのある馬"),
    Spec(  # J03
        slug="fastest-closing-count", title="上がり最速の多い馬は崩れない？", claim="上がり最速を何度も出している馬は崩れない",
        categories=["ability/record"],
        universe=lambda d: d["nstart"] >= 5,
        groups=[("上がり最速3回以上", lambda d: d["l3f_best_count"] >= 3),
                ("1〜2回", lambda d: d["l3f_best_count"].between(1, 2))],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="上がり最速のない馬",
        question="これまでのレースで上がり 3F 最速を何度も出している馬は、人気以上に走るのか。",
        conditions=ALL + "。JRA 平地で 5 戦以上している馬"),
    Spec(  # A27
        slug="g1-step", title="前走G1組は格下の重賞で強い？", claim="前走 G1 を走った馬は G2・G3 で人気以上に走る",
        categories=["race/class"],
        universe=lambda d: has_prev(d) & d["grade"].isin(["G2", "G3"]),
        groups=[("前走G1組", lambda d: d["p_grade"] == "G1"),
                ("前走G1で6着以下", lambda d: (d["p_grade"] == "G1") & (d["p_finish_position"] >= 6))],
        axis=Axis("今回のグレード", lambda d: d["grade"], ["G2", "G3"]),
        ref_label="前走が G1 以外の馬",
        question="前走で G1 を走った馬は、G2・G3 で人気以上に走るのか（大敗でも見直せるか）。",
        conditions="JRA の G2・G3（平地）"),
    # ── 調教 ──
    Spec(  # B01〜B03
        slug="final-workout-jockey", title="最終追い切りに本番の騎手が乗ると勝負気配？", claim="最終追い切りに本番の騎手が乗っている馬は勝負気配",
        categories=["training/rider"],
        universe=lambda d: d["w_has"].eq(True),
        groups=[("最終追い切りに本番の騎手", lambda d: d["w_last_by_jockey"].eq(True)),
                ("1週前に本番の騎手", lambda d: (d["w_wk_by_jockey"].eq(True)) & (d["w_last_by_jockey"].ne(True))),
                ("直前1か月に一度でも", lambda d: d["w_any_by_jockey"].eq(True))],
        axis=Axis("騎手", lambda d: pd.Series(np.select([d["nstart"] == 0, d["jockey_id"] == d["p_jockey_id"], d["jockey_id"] != d["p_jockey_id"]], ["初出走", "継続騎乗", "乗り替わり"], None), index=d.index),
                  ["初出走", "継続騎乗", "乗り替わり"], "最終追い切りに本番の騎手が乗った馬を、騎手の状況で分けると"),
        ref_label="本番の騎手が最終追い切りに乗っていない馬",
        question="最終追い切りに、レースで乗る騎手自身がまたがった馬は、人気以上に走るのか。乗り替わりのときは特にそうか。",
        conditions=ALL + "。追い切りの記録がある出走",
        definition="追い切りの乗り手は姓（『鮫島駿』のような略記）で記録されているので、本番の騎手名と突き合わせて同じ人と判定しています。同じ姓の騎手がいる場合はまれに取り違えます。"),
    Spec(  # B05
        slug="slope-acceleration", title="坂路で終いが一番速い加速ラップは好調のサイン？", claim="坂路で最後の 1F が最も速い（加速ラップ）馬は好調",
        categories=["training/workout"],
        universe=lambda d: (d["w_kind"] == "slope") & d["w_accel"].notna(),
        groups=[("坂路で加速ラップ", lambda d: d["w_accel"] < 0),
                ("加速して終い12.4秒以下", lambda d: (d["w_accel"] < 0) & (d["w_l1"] <= 12.4)),
                ("終いが落ちた（減速）", lambda d: d["w_accel"] > 0.3)],
        axis=Axis("芝・ダートと距離", surf_dist, SD),
        ref_label="坂路で加速しなかった馬",
        question="最終追い切りが坂路で、最後の 1F がその前より速かった（加速ラップ）馬は、人気以上に走るのか。",
        conditions=ALL + "。最終追い切りが坂路の出走",
        definition="加速ラップ = 最後の 1F が、その前の 2F の 1F 平均より速い。"),
    Spec(  # B07・B08
        slug="slope-vs-wood", variants=[3], title="短距離は坂路組、長距離はウッド組？", claim="短距離は坂路で仕上げた馬、長距離はウッドで仕上げた馬が走る",
        categories=["training/workout"],
        universe=lambda d: d["w_kind"].isin(["slope", "wood"]),
        groups=[("短距離×坂路", lambda d: (d["distance"] <= 1400) & (d["w_kind"] == "slope")),
                ("短距離×ウッド", lambda d: (d["distance"] <= 1400) & (d["w_kind"] == "wood")),
                ("2000m以上×坂路", lambda d: (d["distance"] >= 2000) & (d["w_kind"] == "slope")),
                ("2000m以上×ウッド", lambda d: (d["distance"] >= 2000) & (d["w_kind"] == "wood"))],
        ref_label="ほかの組み合わせ",
        question="最終追い切りのコース（坂路・ウッド）と距離の組み合わせで、走りやすさは変わるのか。",
        conditions=ALL + "。最終追い切りが坂路かウッドの出走"),
    Spec(  # B12
        slug="hard-workout-after-layoff", title="休み明けで一杯に追われた馬は仕上がっている？", claim="休み明けで最終追い切りを一杯・強めに追われた馬は仕上がっている",
        categories=["training/workout"],
        universe=lambda d: (d["since_layoff"] == 1) & d["w_leg_color"].notna(),
        groups=[("休み明けで最終を強め以上", lambda d: d["w_leg_color"].isin(HARD)),
                ("一杯", lambda d: d["w_leg_color"].isin(["一杯", "G一", "直一"]))],
        axis=Axis("休養の長さ", lambda d: pd.Series(np.select([d["interval"] <= 180, d["interval"] <= 365, d["interval"] > 365], ["半年以内", "半年〜1年", "1年以上"], None), index=d.index),
                  ["半年以内", "半年〜1年", "1年以上"]),
        ref_label="休み明けで最終が馬なりの馬",
        question="休み明けの初戦で、最終追い切りを強めや一杯に追われた馬は、人気以上に走るのか。",
        conditions=ALL + "。休み明け（中 10 週以上）の初戦"),
    Spec(  # B13
        slug="workout-volume-layoff", title="休み明けは乗り込み量が多いほど走る？", claim="休み明けで追い切りの本数が多い馬は走る",
        categories=["training/workout"],
        universe=lambda d: (d["since_layoff"] == 1) & d["w_n30"].notna(),
        groups=[("直前1か月に7本以上", lambda d: d["w_n30"] >= 7),
                ("4〜6本", lambda d: d["w_n30"].between(4, 6)),
                ("3本以下", lambda d: d["w_n30"] <= 3)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="6 本以下の馬",
        question="休み明けの初戦で、直前 1 か月の追い切りの本数が多い馬は、人気以上に走るのか。",
        conditions=ALL + "。休み明け（中 10 週以上）の初戦"),
    Spec(  # B20
        slug="workout-rank", title="追い切り評価Aの馬は走る？", claim="追い切りの評価が A の馬は好走する",
        categories=["training/workout"],
        universe=lambda d: d["w_rank"].isin(["A", "B", "C", "D"]),
        groups=[("評価A", lambda d: d["w_rank"] == "A"), ("評価B", lambda d: d["w_rank"] == "B"),
                ("評価C", lambda d: d["w_rank"] == "C"), ("評価D", lambda d: d["w_rank"] == "D")],
        axis=Axis("人気", pop_band, POPS, "評価Aの馬を人気で分けると"),
        ref_label="評価 B〜D の馬",
        question="最終追い切りの評価（A〜D）が A の馬は、人気以上に走るのか。",
        conditions=ALL + "。最終追い切りに評価が付いた出走"),
    # ── 騎手 ──
    Spec(  # C01
        slug="trainer-jockey-combo", title="調教師と騎手の『いつものコンビ』は走る？", claim="調教師と騎手の組み合わせ（主戦コンビ）は好走率が高い",
        categories=["jockey/combo"],
        universe=lambda d: d["trainer_runs"] >= 100,
        groups=[("コンビ30回以上", lambda d: d["pair_runs"] >= 30),
                ("10〜29回", lambda d: d["pair_runs"].between(10, 29)),
                ("1〜9回", lambda d: d["pair_runs"].between(1, 9)),
                ("初コンビ", lambda d: d["pair_runs"] == 0)],
        axis=Axis("騎手の前年の勝利数", lambda d: pd.Series(np.select([d["jrank"] <= 10, d["jrank"] <= 40, d["jrank"] > 40], ["上位10人", "11〜40位", "41位〜"], None), index=d.index),
                  ["上位10人", "11〜40位", "41位〜"], "コンビ30回以上を、騎手の格で分けると"),
        ref_label="コンビ 29 回以下の馬",
        question="同じ調教師と騎手の組み合わせでたくさん乗っている『いつものコンビ』は、人気以上に走るのか。",
        conditions=ALL + "。2016 年以降の出走が 100 回以上ある調教師の馬",
        definition="コンビの回数 = 2016 年以降、その調教師の馬にその騎手が乗った回数（今回を含まない）。"),
    Spec(  # C02
        slug="rare-top-jockey", title="普段乗せない上位騎手を乗せたら勝負？", claim="その厩舎が普段乗せない上位騎手を乗せたら勝負気配",
        categories=["jockey/combo"],
        universe=lambda d: (d["trainer_runs"] >= 200) & (d["jrank"] <= 10),
        groups=[("上位騎手との初〜2回目のコンビ", lambda d: d["pair_runs"] <= 2),
                ("3〜9回", lambda d: d["pair_runs"].between(3, 9)),
                ("10回以上", lambda d: d["pair_runs"] >= 10)],
        axis=Axis("人気", pop_band, POPS),
        ref_label="何度も乗せている上位騎手",
        question="その厩舎がほとんど乗せたことのない上位騎手（前年の勝利数で上位 10 人）を乗せたとき、人気以上に走るのか。",
        conditions=ALL + "。前年の勝利数が上位 10 人の騎手が乗った、出走 200 回以上の調教師の馬"),
    Spec(  # C03
        slug="jockey-chose-other", title="主戦が他の馬を選んだら割引？", claim="前走の騎手が同じレースの別の馬に乗るなら、その馬は割引",
        categories=["jockey/context"], direction=-1,
        universe=lambda d: has_prev(d) & (d["jockey_id"] != d["p_jockey_id"]),
        groups=[("前走の騎手が同じレースの別の馬に騎乗", lambda d: d["prev_jockey_elsewhere"].eq(True))],
        axis=Axis("人気", pop_band, POPS),
        ref_label="ほかの乗り替わりの馬",
        question="前走で乗った騎手が、今回は同じレースの別の馬に乗っている（騎手がほかの馬を選んだ）とき、残された馬は人気ほど走らないのか。",
        conditions=ALL + "。前走から騎手が替わった馬"),
    Spec(  # C05
        slug="jockey-continuation", title="テン乗りより継続騎乗？", claim="テン乗りより継続騎乗のほうが走る",
        categories=["jockey/context"],
        universe=has_prev,
        groups=[("継続騎乗（2戦以上続けて）", lambda d: d["jockey_streak"] >= 1),
                ("2戦目", lambda d: d["jockey_streak"] == 1),
                ("3〜4戦目", lambda d: d["jockey_streak"].between(2, 3)),
                ("5戦目以上", lambda d: d["jockey_streak"] >= 4)],
        axis=Axis("騎手の前年の勝利数", lambda d: pd.Series(np.select([d["jrank"] <= 10, d["jrank"] <= 40, d["jrank"] > 40], ["上位10人", "11〜40位", "41位〜"], None), index=d.index),
                  ["上位10人", "11〜40位", "41位〜"]),
        ref_label="乗り替わり（テン乗り）の馬",
        question="前走と同じ騎手が続けて乗る馬は、乗り替わりの馬より人気以上に走るのか。"),
    Spec(  # C11
        slug="jockey-hot-day", title="その日に勝っている騎手は後半も来る？", claim="その日すでに勝っている騎手は、後のレースでも来る",
        categories=["jockey/context"],
        universe=lambda d: d["jday_rides"] >= 2,
        groups=[("その日すでに1勝以上", lambda d: d["jday_wins_before"] >= 1),
                ("1勝", lambda d: d["jday_wins_before"] == 1),
                ("2勝以上", lambda d: d["jday_wins_before"] >= 2)],
        axis=Axis("人気", pop_band, POPS),
        ref_label="その日まだ勝っていない騎手",
        question="同じ日の前のレースですでに勝っている騎手は、そのあとのレースで人気以上に走らせるのか。",
        conditions=ALL + "。その日に 2 鞍以上乗った騎手"),
    Spec(  # C10
        slug="jockey-single-ride", title="1日1鞍だけの騎乗は勝負鞍？", claim="その日 1 鞍だけ乗りに来た騎手の馬は勝負鞍",
        categories=["jockey/context"],
        universe=lambda d: d["jday_rides"].notna(),
        groups=[("その日1鞍だけ", lambda d: d["jday_rides"] == 1),
                ("2〜3鞍", lambda d: d["jday_rides"].between(2, 3))],
        axis=Axis("騎手の前年の勝利数", lambda d: pd.Series(np.select([d["jrank"] <= 10, d["jrank"] <= 40, d["jrank"] > 40], ["上位10人", "11〜40位", "41位〜"], None), index=d.index),
                  ["上位10人", "11〜40位", "41位〜"]),
        ref_label="4 鞍以上乗る騎手",
        question="その日に 1 鞍だけ乗る騎手（遠征してきた騎手など）の馬は、人気以上に走るのか。"),
    # ── 厩舎・陣営 ──
    Spec(  # D01
        slug="northern-farm-layoff", title="ノーザンファーム生産の休み明けは仕上がっている？", claim="ノーザンファーム生産馬は休み明けでも走る（外厩仕上げ）",
        categories=["stable/owner-breeder"],
        universe=lambda d: (d["since_layoff"] == 1) & d["producer"].notna(),
        groups=[("ノーザンファーム生産", lambda d: d["producer"].str.contains("ノーザンファーム", na=False)),
                ("社台ファーム・社台コーポレーション", lambda d: d["producer"].str.contains("社台", na=False))],
        axis=Axis("休養の長さ", lambda d: pd.Series(np.select([d["interval"] <= 180, d["interval"] <= 365, d["interval"] > 365], ["半年以内", "半年〜1年", "1年以上"], None), index=d.index),
                  ["半年以内", "半年〜1年", "1年以上"]),
        ref_label="ほかの生産者",
        question="休み明けの初戦で、ノーザンファーム生産の馬は人気以上に走るのか（外厩で仕上がっていると言われる）。",
        conditions=ALL + "。休み明け（中 10 週以上）の初戦"),
    Spec(  # D04
        slug="stable-two-runners", variants=[], title="同じ厩舎の2頭出しは人気薄の方？", claim="同じ厩舎から 2 頭以上出ると、人気薄の方が来る",
        categories=["stable/trainer"],
        universe=lambda d: d["stable_n"] >= 2,
        groups=[("人気薄の方", lambda d: d["stable_pop_rank"] >= 2),
                ("人気の方", lambda d: d["stable_pop_rank"] == 1)],
        axis=Axis("人気", pop_band, POPS),
        ref_label="人気の方",
        question="同じ調教師の馬が 2 頭以上出たとき、人気のない方が人気以上に走るのか。",
        conditions=ALL + "。同じ調教師の馬が 2 頭以上出たレース"),
    Spec(  # D05
        slug="trainer-change", title="転厩初戦は変わり身がある？", claim="転厩した初戦は変わり身がある",
        categories=["stable/trainer"],
        universe=lambda d: has_prev(d) & d["p_trainer_id"].notna(),
        groups=[("転厩初戦", lambda d: d["trainer_id"] != d["p_trainer_id"])],
        axis=Axis("移った先の調教師の前年の勝利数", lambda d: pd.Series(np.select([d["trank"] <= 20, d["trank"] <= 100, d["trank"] > 100], ["上位20人", "21〜100位", "101位〜"], None), index=d.index),
                  ["上位20人", "21〜100位", "101位〜"]),
        ref_label="同じ調教師のまま",
        question="前走と調教師が替わった（転厩した）初戦の馬は、人気以上に走るのか。"),
    # ── 馬場・枠 ──
    Spec(  # F03
        slug="same-day-bias", variants=[1, 2, 3], title="その日の馬場の偏りは後のレースも続く？", claim="その日の前のレースで内（外）が来ているなら、後のレースも内（外）",
        categories=["track/bias"],
        universe=lambda d: d["day_prior_races"] >= 3,
        groups=[("内が来ている日の内枠", lambda d: (d["day_gate"] <= 0.38) & (d["gate_rel"] < 1 / 3)),
                ("外が来ている日の外枠", lambda d: (d["day_gate"] >= 0.62) & (d["gate_rel"] >= 2 / 3)),
                ("前が残っている日の前走先行馬", lambda d: (d["day_q4"] <= 0.30) & (d["p_q1"] <= 0.25)),
                ("差しが届いている日の前走後方の馬", lambda d: (d["day_q4"] >= 0.55) & (d["p_q1"] >= 0.6))],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"], "内が来ている日の内枠を、芝・ダートで分けると"),
        ref_label="ほかの馬",
        question="同じ日・同じ競馬場・同じ面の前のレース（3 レース以上）で、内枠や先行馬ばかり来ているとき、後のレースでも同じ傾向の馬が人気以上に走るのか。",
        definition="その日の偏り = そのレースより前に終わった同じ面のレースの 3 着以内の馬の、枠の位置（内 0〜外 1）と 4 角の位置の平均。内が来ている = 平均 0.38 以下。"),
    Spec(  # F01
        slug="opening-week-inside", variants=[3], title="開幕週は内枠、最終週は外枠？", claim="開幕週は内枠が有利、開催の終わりは外枠が有利",
        categories=["track/gate"],
        universe=lambda d: d["surface"] == "芝",
        groups=[("開幕週（1〜2日目）の内枠", lambda d: (d["kai_day"] <= 2) & (d["gate_rel"] < 1 / 3)),
                ("開幕週の外枠", lambda d: (d["kai_day"] <= 2) & (d["gate_rel"] >= 2 / 3)),
                ("最終週（7日目〜）の内枠", lambda d: (d["kai_day"] >= 7) & (d["gate_rel"] < 1 / 3)),
                ("最終週の外枠", lambda d: (d["kai_day"] >= 7) & (d["gate_rel"] >= 2 / 3))],
        ref_label="ほかの馬",
        question="芝の開催の初めは内枠が、終わりは外枠が人気以上に走るのか。",
        conditions="JRA全競馬場の芝（障害除く）",
        definition="開催日は、その開催（例: 5 回東京）の何日目か。内枠 = 馬番が内側 1/3、外枠 = 外側 1/3。"),
    Spec(  # G08
        slug="full-gate-outside", title="フルゲートの大外は不利？", claim="フルゲートの大外枠は不利",
        categories=["track/gate"], direction=-1,
        universe=lambda d: d["starters"] >= 16,
        groups=[("大外（外から2頭）", lambda d: d["horse_number"] >= d["field_size"] - 1)],
        axis=Axis("芝・ダートと距離", surf_dist, SD),
        ref_label="ほかの馬番",
        question="16 頭以上のレースで、外から 2 頭（大外）の馬は人気ほど走らないのか。",
        conditions=ALL + "。16 頭以上が走ったレース"),
    Spec(  # G13
        slug="even-gate", title="偶数番（後入れ）は有利？", claim="偶数の馬番（ゲートの後入れ）は出遅れにくく有利",
        categories=["track/gate"],
        universe=lambda d: d["starters"] >= 8,
        groups=[("偶数の馬番", lambda d: d["horse_number"] % 2 == 0)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="奇数の馬番",
        question="ゲートに後から入る偶数の馬番は、人気以上に走るのか。",
        conditions=ALL + "。8 頭以上が走ったレース"),
    Spec(  # F13
        slug="mudder", title="道悪巧者は重・不良で買い？", claim="過去に重・不良で好走した馬は、道悪で人気以上に走る",
        categories=["track/going"],
        universe=lambda d: d["track_condition"].isin(["重", "不良"]) & (d["bad_runs_before"] >= 1),
        groups=[("重・不良で3着内の経験あり", lambda d: d["bad_top3_before"] >= 1),
                ("2回以上", lambda d: d["bad_top3_before"] >= 2)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="重・不良を走って 3 着内のない馬",
        question="過去に重・不良馬場で 3 着以内に来たことのある馬は、今回の重・不良で人気以上に走るのか。",
        conditions=ALL + "。今回が重・不良で、過去に重・不良を走ったことのある馬"),
    Spec(  # H03
        slug="big-horse-heavy-going", variants=[], title="大型馬は道悪・ダートで有利？", claim="大型馬（520kg 以上）は道悪で有利",
        categories=["track/going", "horse/body"],
        universe=lambda d: d["weight"].between(300, 700) & d["track_condition"].isin(["重", "不良"]),
        groups=[("520kg以上", lambda d: d["weight"] >= 520),
                ("440kg未満", lambda d: d["weight"] < 440)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="440〜518kg の馬",
        question="重・不良馬場で、馬体重 520kg 以上の大型馬は人気以上に走るのか。",
        conditions=ALL + "。重・不良馬場のレース"),
    # ── 馬体・個体 ──
    Spec(  # H02
        slug="weight-drop-big", title="馬体重−10kg以上は割引？", claim="馬体重の大幅減（−10kg 以上）は割引",
        categories=["horse/body"], direction=-1,
        universe=lambda d: has_prev(d) & d["weight_change"].notna(),
        groups=[("−10kg以上の減少", lambda d: d["weight_change"] <= -10),
                ("−10〜−14kg", lambda d: d["weight_change"].between(-14, -10)),
                ("−16kg以上", lambda d: d["weight_change"] <= -16)],
        axis=Axis("性別", lambda d: d["sex"].map({"牡": "牡", "牝": "牝", "セ": "セン"}), ["牡", "牝", "セン"]),
        ref_label="−8kg〜+8kg の馬",
        question="前走から馬体重が 10kg 以上減った馬は、人気ほど走らないのか。牝馬は特にそうか。"),
    Spec(  # H04
        slug="heavy-impost", variants=[], title="斤量が重くなると割引？", claim="前走より斤量が 2kg 以上重くなる馬は割引",
        categories=["horse/body"], direction=-1,
        universe=lambda d: has_prev(d) & d["weight_rel"].notna(),
        groups=[("斤量+2kg以上", lambda d: d["weight_rel"] >= 2),
                ("斤量−2kg以下", lambda d: d["weight_rel"] <= -2)],
        axis=Axis("馬体重", lambda d: pd.Series(np.select([d["weight"] < 440, d["weight"] < 500, d["weight"] >= 500], ["440kg未満", "440〜498kg", "500kg以上"], None), index=d.index),
                  ["440kg未満", "440〜498kg", "500kg以上"], "斤量+2kg以上を、馬体重で分けると"),
        ref_label="斤量の変化が ±1.5kg 以内の馬",
        question="前走より斤量が 2kg 以上重くなる馬は、人気ほど走らないのか。小柄な馬は特にそうか。"),
    Spec(  # H08
        slug="old-horse", title="高齢馬（7歳以上）は来ない？", claim="7 歳以上の高齢馬は来ない（ダートは別）",
        categories=["horse/age"], direction=-1,
        universe=lambda d: d["age"] >= 4,
        groups=[("7歳以上", lambda d: d["age"] >= 7),
                ("7歳以上の芝", lambda d: (d["age"] >= 7) & (d["surface"] == "芝")),
                ("7歳以上のダート", lambda d: (d["age"] >= 7) & (d["surface"] == "ダート"))],
        axis=Axis("クラス", cls_band, ["1勝クラス", "2勝・3勝", "オープン以上"]),
        ref_label="4〜6 歳の馬",
        question="7 歳以上の馬は、人気ほど走らないのか。芝とダートで違うか。",
        conditions=ALL + "。4 歳以上の出走"),
    Spec(  # A30
        slug="three-year-old-summer", title="夏の3歳馬は古馬相手に買い？", claim="夏（6〜9 月）の 3 歳以上戦では 3 歳馬が有利",
        categories=["horse/age"],
        universe=lambda d: d["date"].dt.month.between(6, 9) & (d.groupby("race_id")["age"].transform("min") == 3) & (d.groupby("race_id")["age"].transform("max") >= 4),
        groups=[("3歳馬", lambda d: d["age"] == 3)],
        axis=Axis("クラス", cls_band, ["1勝クラス", "2勝・3勝", "オープン以上"]),
        ref_label="4 歳以上の馬",
        question="6〜9 月の 3 歳以上のレースで、3 歳馬は古馬より人気以上に走るのか。",
        conditions=ALL + "。6〜9 月の、3 歳馬と古馬が一緒に走ったレース"),
    Spec(  # H07
        slug="older-mare", title="5歳以上の牝馬は下降線？", claim="牝馬は 5 歳以上になると成績が落ちる",
        categories=["horse/sex"], direction=-1,
        universe=lambda d: (d["sex"] == "牝") & (d["age"] >= 4),
        groups=[("5歳以上の牝馬", lambda d: d["age"] >= 5),
                ("5歳", lambda d: d["age"] == 5),
                ("6歳以上", lambda d: d["age"] >= 6)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="4 歳の牝馬",
        question="牝馬は 5 歳以上になると、人気ほど走らなくなるのか。",
        conditions=ALL + "。4 歳以上の牝馬"),
    # ── レース ──
    Spec(  # I05
        slug="last-race-favorite", variants=[], title="最終レースは荒れる？", claim="最終レースは荒れる（人気馬が飛ぶ）",
        categories=["upset/conditions"], direction=-1,
        universe=lambda d: d["popularity"] <= 3,
        groups=[("最終レース（12R）の1〜3番人気", lambda d: d["race_no"] == 12),
                ("1R（朝一）の1〜3番人気", lambda d: d["race_no"] == 1),
                ("メイン（11R）の1〜3番人気", lambda d: d["race_no"] == 11)],
        axis=Axis("芝・ダート", surface, ["芝", "ダート"]),
        ref_label="ほかのレースの 1〜3 番人気",
        question="最終レースの人気馬は、ほかのレースの人気馬より人気ほど走らないのか。",
        conditions=ALL + "。1〜3 番人気の馬"),
]


def changes(d: pd.DataFrame) -> pd.Series:
    """前走からの替わりの数: 距離（±200m 以上）・芝ダート・騎手・馬場（良⇄稍重以上）。"""
    dist = d["dist_diff"].abs() >= 200
    surf = d["surface"] != d["p_surface"]
    jk = d["jockey_id"] != d["p_jockey_id"]
    going = (d["track_condition"] == "良") != (d["p_track_condition"] == "良")
    return dist.astype(int) + surf.astype(int) + jk.astype(int) + going.astype(int)
