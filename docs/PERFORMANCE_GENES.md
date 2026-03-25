# 競走馬パフォーマンス遺伝子ナレッジベース

競走馬の運動能力に関連する遺伝子マーカーのナレッジベースと推論システム。

## 概要

本システムは以下の遺伝子マーカーをカバーします：

1. **MSTN (Myostatin)** - ミオスタチン遺伝子：筋量制御、距離適性の主要因子
2. **DMRT3** - 歩様遺伝子：ギャロップ vs 速歩・側対歩
3. **PDK4** - エネルギー代謝遺伝子：グルコース vs 脂肪酸酸化
4. **COX4I2** - ミトコンドリア機能：酸素利用効率、持久力
5. **BIEC2-808543** - 距離適性SNPマーカー（MSTN座近傍）
6. その他候補遺伝子（研究段階）

## データファイル

### `data/knowledge/myostatin_genes.json`
- ミオスタチン遺伝子（MSTN）専用のナレッジベース
- 130頭以上の競走馬・種牡馬の遺伝子型データ
- 産駒逆推論による継続的更新

### `data/knowledge/performance_genes.json`
- MSTN以外のパフォーマンス関連遺伝子
- 遺伝子機能、多型、競走能力との関連
- 遺伝子間相互作用の記録

## Pythonモジュール

### `research/myostatin.py`
ミオスタチン遺伝子専用の解析モジュール。

```python
from research.myostatin import MyostatinLookup

mstn = MyostatinLookup()

# 馬の遺伝子型を検索（種牡馬・競走馬どちらも対象）
info = mstn.get_sire_info("ディープインパクト")
# → {'genotype': 'TT', 'confidence': 'confirmed', ...}

# 産駒の遺伝子型確率を推定
probs = mstn.predict_offspring("ディープインパクト", "キングカメハメハ")
# → {'CC': 0.0, 'CT': 0.5, 'TT': 0.5}

# 距離適合度を算出
affinity = mstn.distance_affinity("ディープインパクト", "キングカメハメハ", 2400)
# → 0.8547（高適合）
```

### `research/performance_genes.py`
MSTN以外のパフォーマンス遺伝子解析モジュール。

```python
from research.performance_genes import PerformanceGenes

pg = PerformanceGenes()

# DMRT3（歩様）の適合度
gait = pg.dmrt3_gait_type("CC")  # → "gallop"
suitability = pg.dmrt3_thoroughbred_suitability("CC")  # → 1.0

# PDK4（代謝）の距離適性
profile = pg.pdk4_distance_preference("AA")
# → {'optimal_distance': 2200, 'stamina_score': 0.9, ...}

# 複数遺伝子の統合プロファイル
combined = pg.get_combined_distance_profile(
    mstn="CT",
    pdk4="GA",
    biec2="CT",
    distance_m=1800
)
# → {
#     'optimal_distance': 1750,
#     'distance_score': 0.95,
#     'speed_index': 0.5,
#     'stamina_index': 0.5,
#     ...
# }

# 極端表現型（超スプリント/超ステイヤー）の判定
extreme = pg.check_extreme_phenotype(mstn="TT", pdk4="AA")
# → {'is_extreme': True, 'phenotype': 'ultra_stayer', ...}
```

## CLI使用例

### ミオスタチン遺伝子

```bash
# 馬の遺伝子型を検索（種牡馬・競走馬どちらも対象）
python3 -m research.myostatin lookup ディープインパクト

# 産駒の遺伝子型確率を推定
python3 -m research.myostatin predict ディープインパクト キングカメハメハ --distance 2000

# 登録馬一覧（種牡馬・競走馬）
python3 -m research.myostatin list

# 統計情報
python3 -m research.myostatin stats
```

### その他のパフォーマンス遺伝子

```bash
# 登録遺伝子一覧
python3 -m research.performance_genes list

# 遺伝子情報表示
python3 -m research.performance_genes gene DMRT3

# DMRT3遺伝子型の評価
python3 -m research.performance_genes dmrt3 CC

# PDK4遺伝子型の距離適性
python3 -m research.performance_genes pdk4 AA

# 統合的距離適性プロファイル
python3 -m research.performance_genes combined \
  --mstn CT --pdk4 GA --biec2 CT --distance 2000

# 極端表現型チェック
python3 -m research.performance_genes extreme --mstn TT --pdk4 AA
```

## パイプラインへの統合

### `pipeline/feature_builder.py`

特徴量生成時に自動的に遺伝子関連特徴量を追加：

```python
from research.myostatin import get_lookup as get_mstn
from research.performance_genes import get_lookup as get_perf_genes

mstn = get_mstn()
pg = get_perf_genes()

# ミオスタチン特徴量（既存）
mstn_features = mstn.offspring_features(
    sire_name="ディープインパクト",
    dam_sire_name="キングカメハメハ",
    distance_m=2000
)
# → {
#     'sire_mstn_c': 0.0, 'sire_mstn_t': 1.0,
#     'mstn_cc_prob': 0.0, 'mstn_ct_prob': 0.5, 'mstn_tt_prob': 0.5,
#     'mstn_distance_affinity': 0.85,
#     'mstn_speed_index': 0.0,
#     ...
# }

# 統合的距離プロファイル（新規）
# ※種牡馬のPDK4/BIEC2遺伝子型データが蓄積されたら利用可能
combined_profile = pg.get_combined_distance_profile(
    mstn="TT",  # MSTNはmstn.predict_offspringから取得
    pdk4="GA",  # 将来的に種牡馬DBから取得
    biec2="CT",
    distance_m=2000
)
# → {
#     'optimal_distance': 2100,
#     'distance_score': 0.92,
#     'speed_index': 0.3,
#     'stamina_index': 0.7,
#     ...
# }
```

生成される特徴量：

- `mstn_cc_prob`, `mstn_ct_prob`, `mstn_tt_prob`: 産駒遺伝子型確率
- `mstn_distance_affinity`: MSTN単独での距離適合度
- `mstn_speed_index`: スプリント寄り度（0=ステイヤー, 1=スプリンター）
- `gene_optimal_distance`: 統合的最適距離推定値
- `gene_distance_score`: 当該レース距離での適合度
- `gene_speed_index`: 統合スピード指数
- `gene_stamina_index`: 統合スタミナ指数

## 遺伝子型の推定戦略

### 直接データ（信頼度：高）
1. 遺伝子検査結果の公表（一口馬主クラブ、@Plucky_Liege等）
2. 学術論文での個体遺伝子型報告
3. 商業的検査サービス（Equinome, LRC等）のデータ

### 逆推論（信頼度：中）
1. **産駒の遺伝子型から親を推定**
   - 例：サトノカルナバル（CC確定）→ 父キタサンブラックはCアレル保有確定
   - 例：ゴールドシップ産駒にCC型 → ゴールドシップはCT推定（TT排除）

2. **メンデル遺伝の制約**
   - TT産駒 → 両親ともTアレル保有
   - CC産駒 → 両親ともCアレル保有

### 血統推論（信頼度：低〜中）
1. 父の遺伝子型から子の型を推定
2. 同系統馬の傾向を援用
3. 競走成績・産駒傾向から間接推定

## データ更新プロセス

### 1. 新規情報の収集
- X/Twitter（@Plucky_Liege等）での産駒遺伝子型報告
- 一口馬主クラブの検査結果公表
- 学術論文・業界誌の新知見

### 2. 逆推論の実行
```python
from research.myostatin import MyostatinLookup

mstn = MyostatinLookup()

# 産駒の遺伝子型から親の制約を推論
result = mstn.reverse_infer(
    offspring_genotype="CC",
    sire_name="キタサンブラック"
)
# → {
#     'sire_constraints': {
#         'has_c': True,
#         'possible_genotypes': ['CC', 'CT'],
#         'excludes': ['TT']
#     },
#     ...
# }
```

### 3. ナレッジベースの更新
- `myostatin_genes.json` を直接編集
- 種牡馬だけでなく、主要な競走馬（GI勝ち馬等）も記録
- `confidence` レベルを適切に設定
- `source` フィールドに情報源を記録
- `update_log` に変更履歴を追加

### 4. 検証
```bash
# 整合性チェック（遺伝法則に矛盾がないか）
python3 -m research.myostatin stats

# 産駒推定のテスト
python3 -m research.myostatin predict <父名> <母父名>
```

## 遺伝子間相互作用

### MSTN × PDK4
- **TT（MSTN）× AA（PDK4）** → 超ステイヤー（2400m以上）
  - スリム体型 + 脂肪酸酸化優位 = 極長距離特化
- **CC（MSTN）× GG（PDK4）** → 超スプリンター（1000-1200m）
  - 筋肉質体型 + グルコース代謝優位 = 極短距離特化

### MSTN × BIEC2-808543
- 染色体上で近接（連鎖不平衡）
- 同時遺伝する傾向が強い
- 両マーカーを組み合わせると距離適性予測精度が向上

### DMRT3 × MSTN
- 独立に遺伝（異なる染色体）
- DMRT3は歩様、MSTNは筋量・距離適性を独立に制御
- サラブレッドではDMRT3=CCがほぼ固定されているため、実質的相互作用なし

## 限界と注意点

### 1. 多遺伝子形質
競走能力は数百〜数千の遺伝子が関与する複雑形質。単一遺伝子での説明は限定的。

### 2. 環境要因の大きさ
調教、栄養、騎乗技術、馬場状態等の環境要因が遺伝要因と同等かそれ以上に重要。

### 3. 遺伝子型の浸透率
同じ遺伝子型でも個体差が大きい。遺伝子型は「傾向」であり「確定」ではない。

### 4. データの偏り
- エリート馬（GI勝ち馬、種牡馬）のデータは豊富だが、一般馬は不足
- 日本馬中心で、欧米馬のデータは限定的
- 遺伝子検査は一口馬主クラブの馬が中心（一般馬のデータは少ない）

### 5. 未知の遺伝子
現在のマーカーは氷山の一角。多くの重要遺伝子が未発見。

## 今後の展開

### 短期（〜6ヶ月）
- [ ] 主要馬のPDK4, BIEC2遺伝子型データ収集
- [ ] パイプラインへの統合的距離プロファイル実装
- [ ] WebUI（`/performance-genes` ページ）の追加

### 中期（〜1年）
- [ ] 5世代血統からの遺伝子型確率推定
- [ ] 産駒実績データとの相関分析（遺伝子型vs勝率・着度数）
- [ ] 調教データとの統合（遺伝子型別の最適調教プラン）

### 長期（1年以上）
- [ ] ゲノムワイド育種価（GBLUP）の導入
- [ ] エピジェネティクス要因の研究
- [ ] マルチオミクス統合（遺伝子発現、代謝物等）

## 参考文献

### 学術論文
1. Hill EW et al. (2010) "A sequence polymorphism in MSTN predicts sprinting ability and racing stamina in thoroughbred horses" PLoS ONE
2. Andersson LS et al. (2012) "Mutations in DMRT3 affect locomotion in horses and spinal circuit function in mice" Nature 488:642-646
3. McGivney BA et al. (2010) "MSTN genotypes in Thoroughbred horses influence skeletal muscle gene expression and racetrack performance" Functional Genomics
4. Gu J et al. (2009) "A genome scan for positive selection in thoroughbred horses" Genomics
5. Hill EW et al. (2019) "Moderate and high intensity sprint exercise induce differential responses in COX4I2 and PDK4 gene expression in Thoroughbred horse skeletal muscle" Equine Veterinary Journal

### ウェブリソース
- [LRC スピード遺伝子検査](https://sg-test.lrc.or.jp/sg.php)
- [Equinome - 遺伝子検査サービス](https://equinome.com/)
- [馬ふり - ミオスタチン遺伝子解説](https://uma-furi.com/myostatin/)
- [@Plucky_Liege (X/Twitter)](https://x.com/Plucky_Liege) - 産駒遺伝子型報告
- [中日スポーツ - 獣医師記者コラム](https://chunichi.co.jp/)

### データベース
- NCBI Gene Database
- Ensembl Genome Browser (Horse)
- UCSC Genome Browser (EquCab3.0)

## ライセンスと免責

### データの利用
- 本ナレッジベースは公開情報・学術文献から構築
- 種牡馬遺伝子型の推定は確定情報ではない
- 商業利用の際は独自の検証を推奨

### 免責事項
- 遺伝子型データの正確性を保証しない
- 本情報に基づく馬券購入・馬体評価の結果に責任を負わない
- 育種・調教への適用は専門家に相談すること
