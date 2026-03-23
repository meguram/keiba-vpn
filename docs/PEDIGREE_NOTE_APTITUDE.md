# 血統適性ナレッジ（note まとめ → 特徴量 → モデル）

参照: [Pedigree investigation｜めぐめぐ](https://note.com/lopi_1998/n/n7ebbccf54c2b)（各国血統・コース・種牡馬コメントの整理）

## 戦略

1. **知識の構造化**  
   記事の主張を「解釈可能な軸」に分解する（例: 欧州瞬発、米型スピード持続、道悪、京都外回り、東京 VM 型など）。各軸は **-1〜1 の主観スコア**（著者の観点）として `data/knowledge/sire_aptitude_note.json` に格納する。

2. **種牡馬・牝系の適性表**  
   `stallions` が **種牡馬×軸**、`broodmare_lines` が **牝系ライン×軸**（同じ 17 軸）。  
   - 一覧 API: `GET /api/pedigree/note-aptitude/table`（`rows` と `broodmare_rows`）  
   - 記事の全リストから JSON を再生成: `python3 scripts/build_sire_aptitude_note_full.py`（既存 `stallions` の手調整分はマージで保持）  
   - CSV 化: `stallion_table_rows` / `broodmare_table_rows` をそれぞれ書き出す。

3. **各馬への接続（ブレンド）**  
   **父 1.0 + 母父 0.55 + 牝系ライン 0.35**（牝系名が解決したときのみ第 3 項が効く）。母父のみの場合は従来どおり父+母父。  
   パイプラインでは `entry` / `profile` に `broodmare_line`（JSON の牝系キーまたは `aliases` 経由で解決できる表記）があれば自動で乗る。  
   将来的には 5 代血統から祖先名ヒットでベクトルを足す、あるいは `research/bloodline_vector.py` の系統埋め込みと concat する。

4. **レース条件との整合（距離帯）**  
   `research/sire_aptitude_note.DIST_AXIS_WEIGHTS` でスプリント／マイル／中距離／長距離ごとに軸の重みを変え、`note_apt_dist_fit` として **1 次元の距離整合スコア**に圧縮。これはあくまでヒューリスティックで、**本番の重み付けは学習データに任せる**のが望ましい。

5. **モデリングへの載せ方**  
   - `pipeline/feature_builder._pedigree_features` が **`note_apt_*` 列を自動付与**（ミオスタチン特徴と同列）。  
   - 学習時: 既存の GBDT / 線形モデルにそのまま入力し、重要度で軸の寄与を検証。  
   - 校正版: 残差分析で「記事スコアと実績のズレ」が大きい種牡馬だけ JSON を更新する、または係数でキャリブレーション。

## 実行・確認

```bash
# デモ（父・母父・距離）
python3 -m research.sire_aptitude_note

# JSON の軸キー整合チェック（開発時）
python3 -c "import json; from pathlib import Path; ..."  # CI に載せる場合は scripts 化可
```

API 例（サーバ起動時）:

- `GET /api/pedigree/note-aptitude?sire=ディープインパクト&dam_sire=キングカメハメハ&broodmare_line=Monevassia&distance_m=2000&venue=京都`
- `GET /api/pedigree/note-aptitude/table`
- **レース全頭の 3 次元マップ（パワー・欧州瞬発・TS 素地）**  
  - **ブラウザ**: `/note-aptitude-race`（`?race_id=...` で初期表示可）  
  - **API**: `GET /api/pedigree/race-note-3d?race_id=202505010511&mode=shallow` または `mode=5gen`（5世代は `horse_pedigree_5gen` が必要。父／母経路の係数は `docs/NOTE_APTITUDE_5GEN.md`）  
  出走は `race_shutuba` 優先、無ければ `race_result`＋各馬 `horse_result` で血統補完。CLI: `python3 -m research.note_aptitude_race_map <race_id> [--5gen]`

## 限界

- スコアは **教師データではなくエキスパート寄与**であり、再現性は JSON の版管理に依存する。  
- 英名・表記ゆれは `aliases` で吸収。未登録の種牡馬・牝系はゼロベクトル。  
- 「予測」とは **特徴量供給 + 説明用ヒューリスティック** を指し、単体では勝率確率を保証しない。
