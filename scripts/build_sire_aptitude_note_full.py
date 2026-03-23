#!/usr/bin/env python3
"""
note「Pedigree investigation」に登場する種牡馬・牝系を可能な限り網羅し、
sire_aptitude_note.json を再生成する。

既存 JSON の stallions エントリは手調整済みとして優先（上書きしない）。
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "knowledge" / "sire_aptitude_note.json"

AXIS_IDS = [
    "zaitech_stamina",
    "us_speed_sustain",
    "eu_burst",
    "eu_power_stamina",
    "aus_flat_outside",
    "de_power_inside",
    "early_maturity",
    "late_maturity",
    "wet_turf",
    "fast_turf",
    "kyoto_outer",
    "kyoto_inner_speed",
    "tokyo_mile_vm",
    "tokyo_2400_turf",
    "dirt_power",
    "straight_flat_speed",
    "stamina_mid",
]


def zvec(extra: dict | None = None) -> dict[str, float]:
    o = {k: 0.0 for k in AXIS_IDS}
    if extra:
        for k, v in extra.items():
            if k in o:
                o[k] = float(v)
    return o


# 系統テンプレ（記事の大枠に対応する弱い先験 + 未記載名用）
T = {
    "neutral": zvec(),
    "deep_sunday": zvec(
        {
            "eu_burst": 0.45,
            "stamina_mid": 0.42,
            "straight_flat_speed": 0.48,
            "fast_turf": 0.38,
            "eu_power_stamina": 0.32,
        }
    ),
    "kingkame_line": zvec({"eu_burst": 0.62, "fast_turf": 0.28, "late_maturity": 0.18}),
    "roberto_screen": zvec(
        {"eu_burst": 0.48, "dirt_power": 0.28, "tokyo_mile_vm": 0.32, "us_speed_sustain": 0.32}
    ),
    "mrp_line": zvec({"us_speed_sustain": 0.58, "fast_turf": 0.52, "dirt_power": 0.35}),
    "storm_line": zvec(
        {"eu_burst": 0.52, "fast_turf": 0.42, "wet_turf": 0.28, "tokyo_mile_vm": 0.3}
    ),
    "deputy_line": zvec({"dirt_power": 0.55, "us_speed_sustain": 0.48, "eu_power_stamina": 0.28}),
    "sadler_galileo": zvec(
        {"eu_power_stamina": 0.52, "eu_burst": 0.38, "wet_turf": 0.32, "stamina_mid": 0.4}
    ),
    "danhill_harbinger": zvec(
        {"eu_burst": 0.46, "fast_turf": 0.36, "kyoto_outer": 0.28, "tokyo_mile_vm": 0.32}
    ),
    "danzig_line": zvec(
        {"us_speed_sustain": 0.46, "kyoto_outer": 0.42, "eu_power_stamina": 0.36, "fast_turf": 0.32}
    ),
    "aus_line": zvec(
        {"aus_flat_outside": 0.52, "stamina_mid": 0.42, "late_maturity": 0.32, "eu_power_stamina": 0.3}
    ),
    "german_line": zvec(
        {"de_power_inside": 0.54, "eu_burst": 0.36, "fast_turf": 0.34, "tokyo_2400_turf": 0.28}
    ),
    "eu_stayer": zvec(
        {"stamina_mid": 0.58, "eu_power_stamina": 0.54, "zaitech_stamina": 0.22, "wet_turf": 0.32}
    ),
    "eu_miler": zvec({"eu_burst": 0.54, "fast_turf": 0.46, "tokyo_mile_vm": 0.42}),
    "jpn_zaitech": zvec({"zaitech_stamina": 0.52, "stamina_mid": 0.52, "eu_power_stamina": 0.32}),
    "jpn_speed_mile": zvec({"fast_turf": 0.48, "us_speed_sustain": 0.44, "eu_burst": 0.34}),
    "early_type": zvec({"early_maturity": 0.55, "eu_burst": 0.36, "fast_turf": 0.32}),
    "late_type": zvec({"late_maturity": 0.52, "stamina_mid": 0.35}),
    "wet_eu": zvec({"wet_turf": 0.52, "eu_burst": 0.46, "eu_power_stamina": 0.38}),
    "ribot_millreef": zvec({"wet_turf": 0.55, "eu_burst": 0.5, "eu_power_stamina": 0.46}),
    "us_classic": zvec({"us_speed_sustain": 0.5, "dirt_power": 0.42, "stamina_mid": 0.38}),
    "nhk_turf": zvec({"eu_burst": 0.4, "straight_flat_speed": 0.35, "stamina_mid": 0.38}),
    "minor_followup": zvec({"eu_power_stamina": 0.22, "stamina_mid": 0.22}),
}

# (名前, テンプレキー, 要約) — 記事「種牡馬」見出しベース（日本馬）
JP_STALLIONS: list[tuple[str, str, str]] = [
    ("第四フロリースカツプ", "jpn_speed_mile", "スピード寄り在来牝系の記述。"),
    ("アグネスタキオン", "early_type", "ロイヤルスキー色で早熟・2歳からの記述。"),
    ("アジアエクスプレス", "storm_line", "ストームキャット直系・前半からの持続勝負。道悪で打点上がりやすい。"),
    ("アドマイヤムーン", "wet_eu", "道悪適性を高める因子。ファインニードル産駒にも同様の記述。"),
    ("アメリカンペイトリオット", "mrp_line", "ミスプロ系統の系譜（記事では産駒例のみ）。"),
    ("アラジ", "early_type", "Noverre と3/4同血・Le Fabuleux で早熟強化の記述。"),
    ("アンバーシャダイ", "jpn_zaitech", "高スタミナ・晩成要素。メジロライアン例。"),
    ("イスラボニータ", "minor_followup", "記事では追記のみ。"),
    ("ウォーエンブレム", "minor_followup", "記事では追記のみ。"),
    ("エタン", "eu_stayer", "高い持続力・スタミナ伝達の記述。"),
    ("カラヴァッジオ", "storm_line", "軽い牝系スピード持続×ストームキャット系の記述。"),
    ("カリフォルニアクローム", "us_classic", "米型スピード持続・Hペース持続の記述。"),
    ("キングヘイロー", "danhill_harbinger", "Lyphard 系だが Sir Gaylord 4×4 で Sir Ivor 色。緩いコーナーでパフォUPの記述。"),
    ("ゴールドアクター", "minor_followup", "追記のみ。"),
    ("サクラバクシンオー", "jpn_zaitech", "母サクラハゴロモでスタミナ・短距離全速力型の記述。"),
    ("サトノクラウン", "sadler_galileo", "中距離・欧州気質・使って良くなるの記述。"),
    ("ザファクター", "danhill_harbinger", "Danzig 直系の記述。"),
    ("サンデーサイレンス", "deep_sunday", "瞬発と母方増強・道中スピード補完の記述。"),
    ("シーホーク", "nhk_turf", "父 Herbager・Ballade 牝系で平坦パフォUPの記述。"),
    ("ジャッジアンジェルーチ", "mrp_line", "Bold Ruler の柔らかさでスピード強化の記述。"),
    ("ジャングルポケット", "jpn_zaitech", "高いスタミナ伝達の記述。"),
    ("シャンハイボビー", "us_classic", "パワー持続・前傾ラップ持続に強い産駒の記述。"),
    ("ジョーカプチーノ", "sadler_galileo", "欧州牝系×持続でTS限界も後半長い脚の記述。"),
    ("シンボリルドルフ", "minor_followup", "追記のみ。"),
    ("スクリーンヒーロー", "jpn_zaitech", "母母ダイナアクトレスで野芝・父グラスワンダー中山の記述。"),
    ("スニッツェル", "aus_line", "使って良くなる豪州血統の記述。"),
    ("スピードシンボリ", "minor_followup", "追記のみ。"),
    ("セントクレスピン", "eu_stayer", "父 Aureole・気性難スタミナの記述。"),
    ("タリスマニック", "sadler_galileo", "Sadler's Wells 直系・日本ではダート・道悪芝で瞬発の記述。"),
    ("タワーオブロンドン", "eu_miler", "欧州スピード・体重で距離帯分岐の記述。"),
    ("ダンカーク", "us_classic", "米型スピード持続・前半からの流れの記述。"),
    ("ダンシングブレーヴ", "minor_followup", "追記のみ。"),
    ("チチカステナンゴ", "wet_eu", "Grey Sovereign 直系・マイル前後道悪瞬発の記述。"),
    ("ディスクリートキャット", "storm_line", "ストームキャット×スタミナ牝系でHペーススタミナの記述。"),
    ("ティッカネン", "minor_followup", "追記のみ。"),
    ("ディープブリランテ", "deep_sunday", "ディープ直系の中長距離イメージ。"),
    ("デインヒル", "danhill_harbinger", "豪・欧短距離系統の中心。記事内デインヒル節参照。"),
    ("テスコボーイ", "minor_followup", "追記のみ。"),
    ("トウカイテイオー", "minor_followup", "追記のみ。"),
    ("トニービン", "eu_stayer", "Grey Sovereign・大舞台の横綱競馬で流れ込む型の記述。"),
    ("トライマイベスト", "minor_followup", "追記のみ。"),
    ("ドリームジャーニー", "deep_sunday", "オルフェーヴルより欧州瞬発・タフ中緩みの記述。"),
    ("ナイスダンサー", "nhk_turf", "トウカイナチュラル系の記述。"),
    ("ノーアテンション", "eu_stayer", "長距離スタミナ・坂苦手の記述。"),
    ("ノーザンテースト", "jpn_zaitech", "Hyperion クロスでスタミナ・瞬発・古馬成長の記述。"),
    ("ノヴェリスト", "german_line", "Monsun×母系でバランス型の記述。"),
    ("パーソロン", "eu_miler", "瞬間加速増強因子の記述。"),
    ("ビッグアーサー", "sadler_galileo", "Kingmambo×Sadler's Wells で短距離欧州脚の記述。"),
    ("ファインニードル", "wet_eu", "アドマイヤムーン等により道悪高打点の記述。"),
    ("ファルブラヴ", "sadler_galileo", "エリシオと3/4・Fairy King×Slewpy の記述。"),
    ("フォーティナイナー", "ribot_millreef", "母方 Ribot でパワースタミナ・道悪の記述。"),
    ("フジキセキ", "early_type", "母方 In Reality・Le Fabuleux で早熟の記述。"),
    ("ブライアンズタイム", "eu_stayer", "根性スタミナ・急坂・上りのかかる馬場の記述。"),
    ("ブリックスアンドモルタル", "storm_line", "Giant's Causeway 系・小回り弱めの記述。"),
    ("ヘニーヒューズ", "deputy_line", "良ダート vs 凍結剤でパワー不足の記述。"),
    ("マインドユアビスケッツ", "us_classic", "中間緩まない持続ラップ適性の記述。"),
    ("マルゼンスキー", "minor_followup", "追記のみ。"),
    ("マンハッタンカフェ", "german_line", "ドイツ S ライン・春タフ・中京ダートの記述。"),
    ("メジロマックイーン", "jpn_zaitech", "超ステイヤー・牝系にスタミナの記述。"),
    ("メンデス", "eu_miler", "Grey Sovereign でフランス寄りスピードの記述。"),
    ("モンジュー", "eu_miler", "Top Ville・Grey Sovereign でスピード増強の記述。"),
    ("ラインゴールド", "eu_stayer", "重い長距離スタミナ・凱旋門タイプの記述。"),
    ("ラストタイクーン", "eu_miler", "1600〜2000m 高パフォーマンスの記述。"),
    ("ラブリーデイ", "wet_eu", "欧州スタミナ×ダンスインザダークで道悪瞬発の記述。"),
    ("リアルインパクト", "deep_sunday", "ディープ直系・距離で瞬発寄りに振れるの記述。"),
    ("リアルシャダイ", "jpn_zaitech", "ゆっくり追走・長距離で末脚の記述。"),
    ("リオンディーズ", "wet_eu", "足元・緩むとパフォUPの記述。"),
    ("リマンド", "jpn_zaitech", "スタミナ増強・京都坂・タフ馬場の記述。"),
    ("ロージズインメイ", "eu_stayer", "Speak John で粘り持久力の記述。"),
    ("ワークフォース", "german_line", "Lombard→Allegretta 系・道悪凱旋門の記述。"),
    ("ヴィクトワールピサ", "deep_sunday", "Machiavellian・Busted で下り坂適性の記述。"),
]

# 記事「種牡馬」内の海外・基礎種牡馬（見出し名をキーに）
INT_STALLIONS: list[tuple[str, str, str]] = [
    ("Acatenango", "german_line", "平坦巧者・急坂苦手・後世に伝えるの記述。"),
    ("Affirmed", "early_type", "米ダート中距離・早熟因子の記述。"),
    ("Ahonoora", "eu_miler", "欧州軽スピード・日本マイル以下保管の記述。"),
    ("Alibhai", "eu_stayer", "長距離スタミナ・Dixieland Band クロスで長距離強化の記述。"),
    ("Alzao", "danzig_line", "Lyphard×Sir Ivor でパワースピード持続の記述。"),
    ("Alleged", "ribot_millreef", "Ribot 系欧州パワーの記述。"),
    ("Anabaa", "eu_stayer", "長距離適性・持続ラップスタミナの記述。"),
    ("Aureole", "jpn_zaitech", "気性難・母方で長距離スタミナの記述。"),
    ("Bletchingly", "aus_line", "豪芝短距離・パワー持続短距離の記述。"),
    ("Blushing Groom", "kyoto_outer", "長く脚を使う・京都外回りの記述。"),
    ("Bold Ruler", "mrp_line", "米快速系（記事では追記・補強因子）。"),
    ("Brigadier Gerard", "eu_miler", "欧州1600〜2000 スピードの記述。"),
    ("Buckpasser", "us_classic", "記事では追記参照。"),
    ("Bull Dog", "danzig_line", "Nijinsky 母父系でパワー質スピード持続の記述。"),
    ("Busted", "late_type", "晩成ステイヤー・下り坂・京都長距離の記述。"),
    ("Cadeaux Genereux", "eu_stayer", "スタミナ質スピード持続・エタン牝系でスタミナの記述。"),
    ("Caerleon", "sadler_galileo", "追記のみ。"),
    ("Cape Cross", "danzig_line", "Danzig×Ahonoora バランスの記述。"),
    ("Caro", "fast_turf", "レコード型高速馬場で打点が上がるの記述。"),
    ("Cozzene", "minor_followup", "追記のみ。"),
    ("Crimson Satan", "eu_stayer", "筋肉量・馬格（スカーレット一族）の記述。"),
    ("Danehill Dancer", "danhill_harbinger", "デインヒル色・ラスト1F減速展開の記述。"),
    ("Danzig", "danzig_line", "軽スピード＋パワー質スピード持続の記述。"),
    ("Dark Angel", "eu_miler", "欧州短距離・Night Shift でパワー持続の記述。"),
    ("Darshaan", "ribot_millreef", "Mill Reef–Never Bend 欧州瞬発の記述。"),
    ("Delta Judge", "eu_stayer", "Alibhai 経由でスタミナ底上げの記述。"),
    ("Dixieland Band", "jpn_zaitech", "スタミナ・急坂・中山の記述。"),
    ("Dominion", "early_type", "2歳リーディング色で早期仕上がりの記述。"),
    ("Dr. Fager", "mrp_line", "強スピード因子・米快速の記述。"),
    ("Dubawi", "eu_miler", "ドバイ芝・平坦パワー直線の記述。"),
    ("Dynaformer", "ribot_millreef", "Roberto×His Majesty でスタミナの記述。"),
    ("Exceed And Excel", "aus_line", "豪短距離・溜めて一脚・外差しキレの記述。"),
    ("Fairy King", "sadler_galileo", "Sadler's Wells 全兄弟・持続・平坦向きの記述。"),
    ("Fappiano", "minor_followup", "追記のみ。"),
    ("Francis S.", "stamina_mid", "距離融通の記述。"),
    ("Frankel", "eu_stayer", "Mossborough で距離融通・母父は欧州短距離質の記述。"),
    ("Galileo", "sadler_galileo", "重い欧州パワー・母父では持続＋道悪瞬発の記述。"),
    ("Giant's Causeway", "aus_line", "Rahy で平坦・コーナリング不器用の記述。"),
    ("Gone West", "mrp_line", "ミスプロ×Secretariat・軽スピード持続・加齢で硬さ・持続の記述。"),
    ("Graustark", "eu_stayer", "高スタミナ・His Majesty 全兄弟の記述。"),
    ("Green Dancer", "eu_stayer", "長距離スタミナ・急坂で落とすタイプの記述。"),
    ("Green Desert", "mrp_line", "日本高速スピード＋持続・Courtly Dee で長距離もの記述。"),
    ("Grey Sovereign", "eu_miler", "母方でスピード増強・春の補助の記述。"),
    ("Hail to Reason", "eu_burst", "欧州馬場対応の瞬発の記述。"),
    ("Halo", "minor_followup", "追記のみ。"),
    ("Herbager", "nhk_turf", "Ballade 牝系・平坦UPの記述。"),
    ("High Top", "eu_power_stamina", "Dante 直系じりパワーの記述。"),
    ("His Majesty", "eu_stayer", "Graustark 全兄弟スタミナの記述。"),
    ("Hyperion", "late_type", "晩成・スタミナ・クロスで瞬発増の記述。"),
    ("Iffraaj", "eu_miler", "欧短・芝短パワースピード持続・道悪前傾の記述。"),
    ("In Reality", "mrp_line", "記事では追記・米快速補強。"),
    ("Invincible Spirit", "mrp_line", "Green Desert×スタミナ牝系・短距離パワーの記述。"),
    ("Justify", "storm_line", "ストームキャット直系・米欧中間のスピード持続の記述。"),
    ("Kingmambo", "eu_miler", "キンカメにキレを伝える・Monevassia 対比の記述。"),
    ("Konigsstuhl", "german_line", "ドイツ K ライン・スピード質瞬発の記述。"),
    ("Kris", "minor_followup", "追記のみ。"),
    ("Kris S", "minor_followup", "記事表記 Kris. S を統一。"),
    ("Le Fabuleux", "early_type", "早熟・2歳完成・反動に注意の記述。"),
    ("Le Havre", "wet_eu", "フランス×ドイツで道悪の記述。"),
    ("Lemon Drop Kid", "minor_followup", "追記のみ。"),
    ("Lombard", "german_line", "パワー系ドイツ・Urban Sea への記述。"),
    ("Lomitas", "german_line", "Surumu でスピード質ドイツ・欧州瞬発の記述。"),
    ("Lord at War", "minor_followup", "追記のみ。"),
    ("Lost Code", "deep_sunday", "サンデー×平坦スピード・京都外・香港の記述。"),
    ("Lyphard", "wet_eu", "春の Lyphard・道悪・洋芝の記述。"),
    ("Machiavellian", "wet_eu", "欧マイル×ミスプロでタフ瞬発の記述。"),
    ("Mahmoud", "minor_followup", "追記のみ。"),
    ("Mill Reef", "ribot_millreef", "欧州スピード・クロスで長距離もの記述。"),
    ("Miswaki", "mrp_line", "ミスパサの軽スピードの記述。"),
    ("Monsun", "german_line", "ドイツ・直系/母方で瞬発・内差しの記述。"),
    ("More Than Ready", "mrp_line", "Halo×Woodman×Nasrullah×Turn-to 快速の記述。"),
    ("Mossborough", "eu_stayer", "パワースタミナ・距離融通の記述。"),
    ("Mr. Prospector", "mrp_line", "超高速芝・近年は欧州パワーも必要の記述。"),
    ("Nasrullah", "minor_followup", "追記のみ。"),
    ("Night Shift", "danzig_line", "Northern Dancer×Chop Chop・Danzig 的パワー持続の記述。"),
    ("Nijinsky", "kyoto_outer", "京都外回り・パワー質スピード持続の記述。"),
    ("Niniski", "minor_followup", "追記のみ。"),
    ("Northern Dancer", "eu_burst", "再加速・長く脚を使うの記述。"),
    ("Noverre", "early_type", "Le Fabuleux で早期完成・Le Havre 父の記述。"),
    ("Nureyev", "sadler_galileo", "Sadler's Wells / Fairy King と3/4同血の記述。"),
    ("Petingo", "early_type", "2歳最優秀牡で早熟血統の記述。"),
    ("Pine Bluff", "early_type", "柔らかさ・Almahmoud クロスで早熟の記述。"),
    ("Princely Gift", "eu_miler", "操縦性・反応速度の記述。"),
    ("Princequillo", "minor_followup", "追記のみ。"),
    ("Rahy", "nhk_turf", "Ballade 要素で平坦パフォの記述。"),
    ("Raise a Native", "mrp_line", "軽スピード補完の記述。"),
    ("Ribot", "ribot_millreef", "スタミナ欧州瞬発・牝系で道悪瞬発の記述。"),
    ("Roberto", "eu_power_stamina", "パワー持続・間接伝達の記述。"),
    ("Rockefella", "eu_stayer", "長距離スタミナ・菊花で食い込むの記述。"),
    ("Round Table", "eu_miler", "欧速芝・日本では短めでパワー持続の記述。"),
    ("Sadler's Wells", "sadler_galileo", "欧州瞬発・日本では軽すぎて寄与率低め・直系はパワー強めの記述。"),
    ("Sanctus", "minor_followup", "追記のみ。"),
    ("Sassafras", "eu_stayer", "スタミナ・Theatrical 牝は柔らか差しの記述。"),
    ("Sea The Stars", "sadler_galileo", "追記のみ（Urban Sea 系）。"),
    ("Seattle Slew", "mrp_line", "Fairy King と相性・軽スピード持続・成長でマイルシフトの記述。"),
    ("Secretariat", "mrp_line", "米快速・牝系で軽スピード持続の記述。"),
    ("Seeking the Gold", "mrp_line", "ミスプロ×Buckpasser の記述。"),
    ("Shamardal", "wet_eu", "道悪凱旋門で好走産駒・パワー欧州瞬発の記述。"),
    ("Shirley Heights", "eu_burst", "欧州瞬発・牝系はタフ馬場志向の記述。"),
    ("Sicambre", "jpn_zaitech", "日本長距離・末脚残しの記述。"),
    ("Silver Hawk", "eu_stayer", "Roberto 系で持続＞瞬発の記述。"),
    ("Sir Gaylord", "aus_line", "豪型・後半長い脚・パワー持続の記述。"),
    ("Sir Ivor", "aus_line", "Sir Gaylord 直仔・同様の持続イメージ。"),
    ("Sir Tristram", "minor_followup", "追記のみ。"),
    ("Siyouni", "minor_followup", "追記のみ。"),
    ("Star Kingdom", "aus_line", "豪・2400 前後スタミナ補完の記述。"),
    ("Storm Cat", "storm_line", "スパートスピード・道悪・欧州と組み合わせの記述。"),
    ("Strawberry Road", "aus_line", "豪 Nijinsky 系・叩いて良化・中距離持続の記述。"),
    ("Surumu", "german_line", "ドイツ牝系・直系スタミナ母方瞬発の記述。"),
    ("Tiger Hill", "eu_stayer", "Rockefella・Aureole でスタミナの記述。"),
    ("Tiznow", "minor_followup", "追記のみ。"),
    ("Thatching", "eu_miler", "欧短・日本では1800 前後限界の記述。"),
    ("Theatrical", "eu_stayer", "Sassafras スタミナ・牝は柔らか差しの記述。"),
    ("The Axe", "mrp_line", "米軽スピードの記述。"),
    ("The Tetrarch", "us_speed_sustain", "芦毛・スピード持続を長く伝えるの記述。"),
    ("Tom Fool", "mrp_line", "米快速軽スピードの記述。"),
    ("Top Ville", "eu_miler", "欧州の中でもスピード寄りの記述。"),
    ("Tourbillon", "eu_stayer", "長距離スタミナ・パーソロン系の記述。"),
    ("Tudor Minstrel", "jpn_zaitech", "長距離質スタミナの記述。"),
    ("Turn-to", "minor_followup", "追記のみ。"),
    ("Unbridled", "minor_followup", "追記のみ。"),
    ("Unbridled's Song", "minor_followup", "追記のみ。"),
    ("Urban Sea", "german_line", "Lombard→Allegretta パワー・凱旋門の記述。"),
    ("Vaguely Noble", "us_speed_sustain", "逃げ粘着質スピード・スマートファルコン例の記述。"),
    ("Vice Regent", "deputy_line", "Deputy Minister 経路の記述。"),
    ("Wild Risk", "early_type", "牝系で早熟度が上がるの記述。"),
    ("Woodman", "mrp_line", "ミスパサ軽スピードの記述。"),
    ("Zabeel", "aus_line", "中長・外差し・直線スタミナ質の記述。"),
]

# 記事「牝系」見出し — 母系ナレッジ（同じ17軸で表現）
BROODMARE_LINES: list[tuple[str, str, str]] = [
    ("ウインドインハーヘア", "late_type", "溜めればパフォUP・母父ディープで出やすいの記述。"),
    ("クラフテイワイフ", "fast_turf", "高速馬場スピード・重い種牡馬でもレコード対応の記述。"),
    ("サクラハゴロモ", "jpn_zaitech", "アンバーシャダイ全妹・スタミナ・サクラバクシンオー母の記述。"),
    ("シーザリオ", "early_type", "Le Fabuleux で早熟・エピファネイアを裏付けるの記述。"),
    ("スカーレットブーケ", "eu_power_stamina", "Crimson Satan で筋肉・馬格・前から粘りの記述。"),
    ("ソラリア", "aus_line", "チリ産・平坦適性の記述。"),
    ("ダンシングキイ", "kyoto_inner_speed", "サンデー×でダンスインザ系・京都舞台の記述。"),
    ("ハッピートレイルズ", "eu_power_stamina", "パワー・Aureole でスタミナ・早熟の記述。"),
    ("モシーン", "aus_line", "豪牝系・使って良くなるの記述。"),
    ("ラスティックベル", "danzig_line", "Francis S. で距離・Raise a Native クロスで軽スピードの記述。"),
    ("ワンオブアクライン", "eu_burst", "中京相性・中盤緩みで加速の記述。"),
    ("ヴアインゴールド", "kyoto_outer", "平坦・京都外・1200〜1600 で活躍の記述。"),
    ("Allegretta", "german_line", "Lombard ライン・パワー・凱旋門馬券内の記述。"),
    ("Ballade", "nhk_turf", "平坦スピード・夏新潟等の記述。"),
    ("Brigid", "early_type", "早熟・春安定・快速牝系の記述。"),
    ("Highclere", "straight_flat_speed", "上り坂弱・下り坂強・Hypericum でスタミナの記述。"),
    ("My Bupers", "mrp_line", "軽スピード・パワー馬場で落ちる産駒の記述。"),
    ("Monevassia", "danzig_line", "キングマンボ姉・Nureyev でパワー持続・キレ負けの記述。"),
    ("Pacific Princess", "eu_stayer", "Damascus・Mossborough・菊花に強い牝系の記述。"),
]


def _stallion_entry(template_key: str, summary: str) -> dict:
    t = T.get(template_key, T["neutral"])
    return {"axes": deepcopy(t), "summary_ja": summary}


def main() -> None:
    with open(JSON_PATH, encoding="utf-8") as f:
        bundle = json.load(f)

    existing = bundle.get("stallions", {})
    out_stallions: dict[str, dict] = {}

    for name, tmpl, summ in JP_STALLIONS + INT_STALLIONS:
        out_stallions[name] = _stallion_entry(tmpl, summ)

    # 手調整済みを最優先で上書き
    for k, v in existing.items():
        out_stallions[k] = deepcopy(v)

    broodmare: dict[str, dict] = {}
    for name, tmpl, summ in BROODMARE_LINES:
        broodmare[name] = _stallion_entry(tmpl, summ)

    bundle["stallions"] = dict(sorted(out_stallions.items(), key=lambda x: x[0]))
    bundle["broodmare_lines"] = dict(sorted(broodmare.items(), key=lambda x: x[0]))
    bundle["meta"]["version"] = 2
    bundle["meta"]["counts"] = {
        "stallions": len(bundle["stallions"]),
        "broodmare_lines": len(bundle["broodmare_lines"]),
    }

    # エイリアス拡充（英名・表記ゆれ）
    aliases = bundle.get("aliases", {})
    extra_aliases = {
        "Deep Sunday": "サンデーサイレンス",
        "Sunday Silence": "サンデーサイレンス",
        "Heart's Cry": "ハーツクライ",
        "Kris S.": "Kris S",
        "Kris. S": "Kris S",
        "Stay Gold": "ステイゴールド",
        "Storm cat": "Storm Cat",
        "ストームキャット": "Storm Cat",
        "Sadler’s Wells": "Sadler's Wells",
        "Sadlers Wells": "Sadler's Wells",
    }
    for a, c in extra_aliases.items():
        if a not in aliases:
            aliases[a] = c
    bundle["aliases"] = dict(sorted(aliases.items(), key=lambda x: x[0].lower()))

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        "Wrote",
        JSON_PATH,
        "stallions=",
        len(bundle["stallions"]),
        "broodmare_lines=",
        len(bundle["broodmare_lines"]),
    )


if __name__ == "__main__":
    main()
