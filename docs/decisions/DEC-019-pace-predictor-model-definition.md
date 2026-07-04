# DEC-019: pace_predictor モデル定義の確定: T-10/T-11 対応

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-07 |
| **矛盾ID** | S-7-A |

---

### コンテキスト

MASTER.md §6-4 と AREA-07 §3-2 は `pace_predictor` を T-10（ペースカテゴリ分類: HIGH/MIDDLE/SLOW）と T-11（1F毎ラップ回帰）を担当するモデルとして定義しているが、`src/pipeline/mlflow/catalog.py` は `pace_predictor` を「1角・3角タイム予測」として登録しており定義が乖離している。`lap_predictor.py` と `lap_lstm.py` が T-11 を担当しているが MLflow カタログに未登録。

---

### 決定事項

`pace_predictor` の定義を「T-10: ペースカテゴリ分類（HIGH/MIDDLE/SLOW）」に限定する。T-11 のラップ予測は `lap_predictor`（LightGBM per-furlong）と `lap_lstm`（LSTM、Phase 4）を別モデルとして MLflow カタログに登録する。

```python
# catalog.py の更新後イメージ
ModelSpec(key="pace_predictor", title="ペース予測（T-10）",
          description="T-10: ペースカテゴリ分類（HIGH / MIDDLE / SLOW）"),
ModelSpec(key="lap_predictor", title="ラップ予測 LightGBM（T-11）",
          description="T-11: 1F 毎ラップ秒数予測（LightGBM regression）"),
ModelSpec(key="lap_lstm", title="ラップ予測 LSTM（T-11 Phase 4）",
          description="T-11: 1F 毎ラップ秒数予測（LSTM 時系列回帰、Phase 4 以降）"),
```

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| pace_predictor=T-10、lap系=T-11（採用） | 仕様書準拠・責務分離 | catalog.py の変更が必要 |
| pace_predictor に T-10/T-11 両方持たせる | モデル統合 | 責務が曖昧・1角・3角タイムの定義が不明 |

---

### 影響範囲

- `src/pipeline/mlflow/catalog.py` の `pace_predictor.description` 更新
- `lap_predictor`・`lap_lstm` のエントリを `catalog.py` に追加
- `docs/decisions/AREA-07-modeling.md` §3-2 のモデル対応表を確認・更新

---

### 備考

「1角・3角タイム」予測は archive の実装に存在するが、AREA 仕様書の T-10/T-11 定義には含まれない。archive の機能は別途 backlog として管理し、AREA 仕様書定義を優先する。
