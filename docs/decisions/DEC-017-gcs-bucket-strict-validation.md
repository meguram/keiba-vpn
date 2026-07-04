# DEC-017: GCS_BUCKET 環境変数: 未設定時は起動失敗（fail-fast）

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-06 |
| **矛盾ID** | S-5-A |

---

### コンテキスト

AREA-06 §3-3 は `GCS_BUCKET = os.environ["GCS_BUCKET"]`（KeyError で即失敗）を仕様としているが、実装の `src/config/data_paths.py` は `os.environ.get("GCS_BUCKET", "")` でフォールバックし、空文字による不正なパス `gs:///chuou/data/...` が生成される。エラーが遅延して発生し、デバッグが困難になる。

---

### 決定事項

`src/config/data_paths.py` の `_gcs_bucket()` を RuntimeError を raise する形式に変更し、未設定時に起動直後に失敗させる。ローカルテスト用には `KEIBA_GCS_ENABLED=false` 環境変数で GCS パスの解決をスキップできるようにする。

```python
def _gcs_bucket() -> str:
    bucket = os.environ.get("GCS_BUCKET")
    if not bucket:
        raise RuntimeError(
            "GCS_BUCKET environment variable is not set. "
            "Set it in .env or export GCS_BUCKET=<your-bucket-name>"
        )
    return bucket
```

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| fail-fast（採用） | 早期検知・仕様書準拠・不正パス防止 | ローカル開発で .env 設定必須 |
| デフォルト空文字のまま | ローカル起動は楽 | 不正パスが遅延発見・デバッグ困難 |

---

### 影響範囲

- `src/config/data_paths.py` の `_gcs_bucket()` 変更
- `.env.example` に `GCS_BUCKET=<your-bucket-name>` の記載を追加・補足

---

### 備考

GCS を使わないローカルテストは `KEIBA_GCS_ENABLED=false` などの環境変数で制御する設計が望ましい。現時点では .env に `GCS_BUCKET` を設定することをオンボーディング手順に含める。
