"""通説シリーズの記事のグループ定義（検証の結果から作る記事。作り方は folklore_verified.py）。

1 グループ = 1 記事。主の型（記事の判定になる）は手で決め、結果を見てから選び直さない。
同じテーマの既存コラム（folklore_specs.py や以前からの記事）は links で案内するだけで、既存の記事の中身は触らない。
記事にしない型（検定できない型・偽の型・オッズの時系列が要る型・レース後の値を使う型）は folklore_verified.EXCLUDED と meta.json の na。
"""

from __future__ import annotations

from src.research.columns.folklore_verified import Group

G = Group

GROUPS: list[Group] = [
    # ── ローテーション・前走 ──
    G(slug="course-repeat", title="同じコースの続戦・コースの形の替わりは走る？", label="前走と同じコースの続戦",
      claim="前走と同じコースの続戦や、回り・坂・直線の長さの替わりには、走りへの影響がある",
      categories=["travel/course-change"], main="A03", hids=["A03", "A06", "A07", "A08", "A09", "A10", "A32", "A46", "A47"],
      question="前走と同じコース（場・面・距離）を続けて走る馬や、回り・坂・直線の長さが替わる馬は、人気以上（以下）に走るのか。",
      links=[("direction-switch", "右回り⇄左回りの替わり"), ("local-to-central", "ローカル⇄中央の替わり")]),
    G(slug="class-up-details", title="昇級初戦を細かく見ると？", label="連勝中の昇級初戦",
      claim="連勝中の勢いのある昇級初戦は通用する",
      categories=["travel/class"], main="A42", hids=["A42", "A16", "A28", "K09", "K17", "L02"],
      question="連勝中の昇級初戦や、格上挑戦・時計の速い勝ち上がりは、人気以上に走るのか。",
      links=[("class-up-first", "昇級初戦は通用しない？")]),
    G(slug="prev-race-excuses", title="前走の不利・再戦・安定感は次走に生きる？", label="直近5走で3着内3回以上・勝ち無し",
      claim="勝ち切れなくても毎回3着以内に来る馬は堅い。前走の不利や再戦の相手は次走に生きる",
      categories=["prev/result"], main="A43", hids=["A43", "A24", "A29", "A45", "K01", "K13", "L09"],
      question="前走の不利、再戦の相手との力関係、勝ち切れないが堅い馬は、次走で人気以上に走るのか。",
      links=[("close-loss-last-time", "前走は僅差の負け"), ("fastest-closer-loser", "前走上がり最速で負けた馬")]),
    G(slug="layoff-details", title="休み明けの細かい通説は本当か？", label="休み明けを好走して中4週以内で使う2戦目",
      claim="休み明けの初戦を好走して間隔を詰めて使う2戦目は、反動で割引",
      categories=["travel/interval"], main="A41", hids=["A41", "A13", "A40", "H19", "L08", "K07", "K11"],
      question="休み明けを好走した直後の詰めた2戦目や、厩舎の休み明けの強さ・休養の長さは、人気以上（以下）の走りにつながるのか。",
      links=[("layoff-first-start", "休み明け初戦は割引？"), ("second-start-after-layoff", "叩き2戦目は上積みがある？"), ("quick-turnaround", "連闘・中1週の馬は走る？")]),
    G(slug="lap-type", title="得意なラップの型を再現できるレースは走る？", label="今回の想定の型で好走経験がある馬",
      claim="馬の得意なラップ（瞬発・消耗・持続）を再現できるレースで買う。展開に恵まれた好走は次走で割引",
      categories=["prev/pace"], main="R01", hids=["R01", "R02", "R03", "R04", "R05", "R06", "R07"],
      question="そのコースで出やすいラップの型に合う馬や、前走が展開の恩恵・不利だった馬は、人気以上に走るのか。",
      links=[("pace-scenario", "展開シナリオ"), ("pace-formation", "逃げ馬の数とペース")]),
    G(slug="terrain-fit", title="コースの地形の得意・不得意は当てになる？", label="地形の得意が2つ以上重なる馬",
      claim="地形（直線の長さ・坂・回り・小回り）の得意なファクターが重なる競馬場で買い、苦手が重なれば消す",
      categories=["track/terrain"], main="T05", hids=["T05", "T01", "T02", "T03", "T04", "T06"],
      question="過去の走りから見た地形の得意・不得意は、人気以上（以下）の走りにつながるのか。",
      links=[("direction-switch", "右回り⇄左回りの替わり")]),
    # ── 調教 ──
    G(slug="workout-speed", title="坂路の終いの時計が速い馬は走る？", label="坂路の終い1Fが12.4秒以下",
      claim="坂路の終いの時計が速い馬は好調で走る",
      categories=["training/workout"], main="B29", hids=["B29", "B06", "B08", "B22", "B26", "B35"],
      question="最終追い切りの坂路で終い1Fが速い馬（新馬・冬・重い馬場を含む）は、人気以上に走るのか。",
      links=[("slope-acceleration", "坂路の加速ラップ")]),
    G(slug="workout-pattern", title="調教の強さ・間隔・パターンの通説は本当か？", label="前走好走時と同じ調教パターン",
      claim="前走で好走したときと同じ調教パターンは再現する。いつもと違う調教は勝負か不安",
      categories=["training/workout"], main="B15", hids=["B15", "B07", "B12", "B16", "B17", "B18", "B19", "B31", "B34", "B37"],
      question="調教の強さ・間隔・曜日・厩舎のいつものパターンとの違いは、人気以上（以下）の走りにつながるのか。",
      links=[("hard-workout-after-layoff", "休み明けで一杯に追われた馬"), ("workout-volume-layoff", "乗り込み量"), ("slope-vs-wood", "坂路とウッド")]),
    G(slug="workout-notes", title="調教の短評・乗り手・場所の通説は本当か？", label="短評に否定的な語がある馬",
      claim="調教の短評が否定的な馬は割引。乗り手や調教場所にも意味がある",
      categories=["training/rider"], main="B21", hids=["B21", "B04", "B23", "B24", "B25", "B27", "I07"],
      question="調教の短評・追い切りの乗り手・調教の場所は、人気以上（以下）の走りにつながるのか。",
      links=[("final-workout-jockey", "最終追い切りに本番の騎手"), ("workout-rank", "追い切り評価A")]),
]
