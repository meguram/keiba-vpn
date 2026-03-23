# データソース一覧

netkeiba.com および SmartRC (smartrc.jp) から取得する全データソースの仕様書。
各ページの URL パターン、認証要否、パーサー/クライアント、ストレージカテゴリ名、取得フィールドを定義する。

---

## 概要

### netkeiba.com

| # | データソース | URL ドメイン | 認証 | パーサー | ストレージカテゴリ | キー |
|---|---|---|---|---|---|---|
| 1 | レース結果 | db.netkeiba.com | 不要 | `RaceResultParser` | `race_result` | race_id |
| 2 | 出馬表 | race.netkeiba.com | 不要 | `RaceCardParser` | `race_shutuba` | race_id |
| 3 | タイム指数 | race.netkeiba.com | **要** | `SpeedIndexParser` | `race_index` | race_id |
| 4 | 馬柱・調教 | race.netkeiba.com | 不要 | `ShutubaPastParser` | `race_shutuba_past` | race_id |
| 5 | オッズ (単勝/複勝) | race.netkeiba.com | 不要 | `OddsParser` | `race_odds` | race_id |
| 6 | パドック評価 | race.netkeiba.com | **要** | `PaddockParser` | `race_paddock` | race_id |
| 7 | 偏差値バロメーター | race.netkeiba.com | **要** | `BarometerParser` | `race_barometer` | race_id |
| 8 | 追い切り (調教詳細) | race.netkeiba.com | 不要 | `OikiriParser` | `race_oikiri` | race_id |
| 9 | 馬情報 (プロフィール+戦績) | db.netkeiba.com | 不要 | `HorseParser` | `horse_result` | horse_id |
| 10 | 日付別レース一覧 | db.netkeiba.com | 不要 | `RaceListParser` | `race_lists` | date |

> **認証「要」**: netkeiba プレミアム会員のログインが必要。認証情報は `.env` の `netkeiba_id` / `netkeiba_pw` で管理。

### SmartRC (smartrc.jp)

| # | データソース | API エンドポイント | 認証 | クライアント | ストレージカテゴリ | キー |
|---|---|---|---|---|---|---|
| 11 | 出走馬 (独自指標含む) | `runners/view` | **要** | `SmartRCClient` | `smartrc_runners` | race_id |
| 12 | 全出走馬 (拡張) | `allrunners/view` | **要** | `SmartRCClient` | `smartrc_runners` | race_id |
| 13 | 双馬メモ付き出走馬 | `bbrunners/view` | **要** | `SmartRCClient` | `smartrc_runners` | race_id |
| 14 | トレンドデータ | `trendrunners/view` | **要** | `SmartRCClient` | `smartrc_trend` | race_id |
| 15 | 傾向集計 | `trendaggregates/view` | **要** | `SmartRCClient` | `smartrc_trend` | race_id |
| 16 | 亀ポイント | `kamepoints/view` | **要** | `SmartRCClient` | `smartrc_runners` | race_id |
| 17 | 馬情報 (SmartRC) | `horses/view` | **要** | `SmartRCClient` | `smartrc_horse` | horse_id |
| 18 | 馬過去成績 | `runnerresults/view` | **要** | `SmartRCClient` | `smartrc_horse` | horse_id |
| 19 | 近親馬 | `relatives/view` | **要** | `SmartRCClient` | `smartrc_horse` | horse_id |
| 20 | 開催日一覧 | `days/view` | 不要 | `SmartRCClient` | - | - |

> **SmartRC 認証**: netkeiba とは別アカウント。`.env` の `SMARTRC_LOGIN` / `SMARTRC_PASSWORD` で管理。
> SmartRC は JSON API (ExtJS ベース) であり HTML パースは不要。
> API ベース: `https://www.smartrc.jp/v3/smartrc.php/{endpoint}`

### SmartRC 独自データフィールド (netkeiba にないもの)

| フィールド | 説明 | カテゴリ |
|---|---|---|
| `tb_memo` | 双馬メモ — 前走の枠・馬場・脚質・ローテ不利の記録 | レース分析 |
| `tb` | トラックバイアス — 馬場・コース取り・展開の有利不利 | レース分析 |
| `evaluation` | 前走評価 (A〜E) | レース分析 |
| `est_popularity` | 推定人気 | 予想 |
| `popularity_rank` | 人気ランク (A〜E) | 予想 |
| `ten_p` | テンP — 序盤3F順位パターン (15/30/50) | ペース |
| `ten_t` | テンT — 補正済み序盤3F最速タイム | ペース |
| `agari_p` | 上がりP — 末脚3F順位パターン | ペース |
| `agari_t` | 上がりT — 補正済み上がり3F最速タイム | ペース |
| `ten_1f` | テン1F — 補正済みテン1Fタイム | ペース |
| `blood_system` | 血統系統カラーリング (小系統の色分け) | 血統 |
| `country_type` | 国系統 (日/米/欧) | 血統 |
| `country_eval` | 国系統評価 (プレミアム限定) | 血統 |
| `rotation` | ロ — ローテーション (延/同/短) | 適性 |
| `cross_surface` | 異種 — 芝⇔ダート経験 | 適性 |
| `cr` | CR — 独自指標 | 適性 |
| `share` | シェア — 種牡馬産駒賞金比率 (1〜8) | 血統 |
| `kame_point` | 亀谷推奨ポイント | 予想 |

---

## ストレージ構成

### GCS (source of truth)

```
gs://{BUCKET}/chuou/data/
  ├── race_result/{race_id}.json       # レース結果
  ├── race_shutuba/{race_id}.json      # 出馬表
  ├── race_index/{race_id}.json        # タイム指数
  ├── race_shutuba_past/{race_id}.json # 馬柱・調教
  ├── race_odds/{race_id}.json         # オッズ
  ├── race_paddock/{race_id}.json      # パドック評価
  ├── race_barometer/{race_id}.json    # 偏差値バロメーター
  ├── race_oikiri/{race_id}.json       # 追い切り
  ├── horse_result/{horse_id}.json     # 馬情報
  ├── smartrc_runners/{race_id}.json   # SmartRC レースデータ
  ├── smartrc_horse/{horse_id}.json    # SmartRC 馬データ
  └── smartrc_trend/{race_id}.json     # SmartRC トレンド
```

### GCS (HTML アーカイブ)

```
gs://{BUCKET}/chuou/html/
  ├── race_result/{race_id}.html.gz
  ├── race_shutuba/{race_id}.html.gz
  ├── ...
  └── horse_result/{horse_id}.html.gz
```

> SmartRC は JSON API のため HTML アーカイブなし。

### ローカル

```
data/local/
  └── race_lists/{date}.json   # 日付別レース一覧 (ローカルのみ)
data/meta/
  ├── structure/               # ページ構造フィンガープリント
  └── logs/                    # アクセスログ
```

---

## 認証情報 (.env)

```dotenv
# netkeiba.com
netkeiba_id=<email>
netkeiba_pw=<password>

# SmartRC (smartrc.jp) — 別アカウント
SMARTRC_LOGIN=<email>
SMARTRC_PASSWORD=<password>

# GCS
GCS_BUCKET_NAME=<bucket>
GOOGLE_APPLICATION_CREDENTIALS=<path>
```
