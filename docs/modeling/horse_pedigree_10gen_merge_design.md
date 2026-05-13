# 5 世代血統表 ×「5 世代目の種牡馬」の 5 世代マージによる深さ 10 相当の血統設計

## 1. 背景

- netkeiba の `/horse/ped/{id}/` から得られる **固定レイアウトの 5 世代表**（最大 62 祖先、`generation` 1..5、`position` 世代内 0 始まり）だけでは、表の外側（6 代目以降）の祖先は見えない。
- 一方、**5 世代目に現れる種牡馬（牡スロット）** それぞれについて、同じく **5 世代血統 JSON**（既存 `horse_pedigree_5gen`）を取得できるなら、その種牡馬を「小さな根」とした 5 世代分を **主表の該当ノードの下に接木**することで、主観測馬から **最大 5+5=10 世代** 先までの祖先情報を機械可読に拡張できる（「ページが 10 世代」ではなく **データ上の深さ** が 10 になる）。

## 2. 用語

| 用語 | 意味 |
|------|------|
| 主馬 S | 対象の `subject_horse_id` |
| 主表 | S の 5 世代パース結果 `ancestors`（`research.pedigree_similarity.parse_blood_table_5gen` 相当） |
| アンカー H | 主表上で **generation = 5** かつ **牡スロット**（`is_male_pedigree_slot(5, p)==True`）かつ **`horse_id` が有効**な祖先馬 |
| 枝表 | H を根とみたときの 5 世代データ（`horse_pedigree_5gen[H]` の `ancestors`。H 自身は含めない。generation 1 = H の父母） |

## 3. 拡張対象ノード（アンカー選定ルール）

主表の各行のうち、次をすべて満たすものをアンカーとする。

1. `generation == 5`
2. `is_male_pedigree_slot(5, position)` が True（同一世代内で偶数 `position` が牡。`research.pedigree_similarity.is_male_pedigree_slot` と一致）
3. `horse_id` が null / 空 / プレースホルダでない（`sanitize_netkeiba_string_id` 後に有効）

**牝馬はアンカーにしない**（ユーザー要件「5 世代目の**種牡馬**」）。将来牝系深掘りする場合は別フラグで拡張可能。

## 4. 深さとマージの写像

- 主表における H の **主馬からの深さ** は `depth(H) = 5`（世代ラベルと一致）。
- 枝表の祖先は H から見て `local_generation` 1..5（H の父母が 1）。
- **主馬 S から見た深さ**（グローバル世代ラベル）:

```text
global_depth = 5 + local_generation
```

- よって `local_generation` の最大 5 に対し、`global_depth` の最大は **10**。これが「5+5」の意味する上限。

**注意**: `global_depth` は netkeiba HTML の「世代番号」とは別の **論理深さ** である。Parquet では `primary_generation`（主表のみの 1..5）と `merged_global_depth`（1..10）を併記するとデバッグしやすい。

## 5. 一意キー（推奨）: FM パス

5 世代表は完全二分木の切断図として扱える。主馬から「父側=F / 母側=M」で辿った列を **長さ = global_depth の文字列**（例 `FFMFMM...`）とすると、

- 主表の各行（1..5 世代）は **一意の FM パス**（長さ 1..5）に変換できる（`position` の左から右の並びと `is_paternal_side` の分割を再帰的に適用）。
- アンカー H の枝表の各祖先は、**「主表で H に至るパス」`path(S→H)`** と **「H の枝表での相対パス」`path(H→x)`**（H の父母を F/M の第 1 歩とする）を連結した `path(S→x) = path(S→H) + path(H→x)` で表せる（長さ最大 10）。

**主キー案**（ロング表）:

- `subject_horse_id`
- `path_fm` — 上記 FM 文字列（長さ 1..10）
- `merged_global_depth` = `len(path_fm)`
- `source` — `primary` | `merged_gen5_sire`
- `anchor_horse_id` — `merged` 行のみ H を格納。`primary` は null

インブリードで同一 `horse_id` が複数 `path_fm` に乗る場合は **行を分けて保持**（血統解析ではパスが意味を持つ）。

## 6. アルゴリズム（バッチ）

```
入力: subject_horse_id S, primary = parse(S)
出力: rows[]（ロング表）

1. primary の各行を path_fm に射影し rows に追加（source=primary, anchor=null）。
2. アンカー集合 A = { H | primary 行で gen=5 ∧ 牡スロット ∧ horse_id 有効 }
3. H ∈ A を順に（または並列キューで）:
   a. ped_H = load horse_pedigree_5gen[H]（無ければキュー投入してスキップ可）
   b. path_SH = path_fm(S→H)  # 長さ 5
   c. ped_H.ancestors の各行（local_gen, local_pos, horse_id, name, ...）について:
        path_Hx = path_fm_from_anchor_table(local_gen, local_pos)  # 長さ local_gen
        path_Sx = path_SH + path_Hx
        merged_depth = len(path_Sx)  # <= 10 を保証
        rows に追加（source=merged_gen5_sire, anchor=H, merged_global_depth=merged_depth）
4. path_Sx 重複（同一 path 二重投入）があれば後勝ち or 先勝ちをポリシーで固定（推奨: 先勝ち＝主表優先）
5. Parquet へ書き出し（既存 ped_tbl 列と整合するなら path_fm / merged_global_depth を追加列で載せる）
```

**第 2 ラウンド**（6 世代目の牡の 5 世代をさらにマージ）は、本設計の「一回の 5+5」より先には進めない（深さ 15 になる）。要件が増えたら別設計。

## 7. データ供給・運用

1. **前提**: アンカー H の `horse_pedigree_5gen` がローカル／GCS に存在すること（`scraper.horse_pedigree_5gen_bulk` / キュー `horse_pedigree_5gen`）。
2. **欠損**: `ped_H` が無い H は `merge_status=missing_anchor_ped` としてマニフェストに集計。主表の 5 世代は常に出力。
3. **鮮度**: H の JSON が古い場合、H の子孫側だけ再フェッチして `ped_tbl/{shard}/{S}.parquet` を上書き（馬単位ストアの単位と一致）。
4. **計算量**: アンカー数は最大 **5 世代目の牡スロット数**（完全表なら 16）。馬ごとに最大 16 本の枝表マージで抑えられる。

## 8. 実装フェーズ（推奨）

| Phase | 内容 |
|-------|------|
| 1 | `(generation, position) → path_fm`（主表 1..5）とその逆の単体テスト（`research.pedigree_similarity` の父系/牡定義と整合） |
| 2 | アンカー抽出 + `path_SH` + 枝表 `path_Hx` 連結 + 重複ポリシー |
| 3 | `pipeline/build_horse_entity_store` の ped 生成経路にオプション `--merge-gen5-sires` で結合 |
| 4 | キュー連携（アンカー未取得時の自動投入） |

## 9. 関連コード

- 5 世代パース: `research.pedigree_similarity.parse_blood_table_5gen`
- 牡スロット: `is_male_pedigree_slot`, `MALE_PEDIGREE_SLOTS_5GEN`
- ローカル JSON: `data/local/horse_pedigree_5gen/{4桁}/{horse_id}.json`, `research.pedigree_local_store`
- 馬単位出力: `pipeline/build_horse_entity_store.py`, `pipeline/horse_entity_layout.py`
- 骨子モジュール（写像・定数の置き場）: `pipeline/horse_pedigree_expand.py`
