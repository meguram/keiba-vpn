# DEC-020: 認証方式: JWT Bearer トークン実装（Cookie との併用）

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-03 |
| **矛盾ID** | S-8-A |
| **関連 DEC** | DEC-016 |

---

### コンテキスト

AREA-03 §4 は API クライアント向けに `Authorization: Bearer <token>` ヘッダーによる認証を仕様として定義しているが、実装は Cookie 専用で Bearer 検証ロジックが存在しない。外部クライアント・CI/CD からの API 呼び出しに対応できない。

---

### 決定事項

フェーズ分割で対応する:

1. **即時（DEC-016 users テーブル実装後）**: ログイン API（`POST /api/v1/auth/login`）がセッション Cookie と同時に JWT トークンを発行する。
2. **認証ミドルウェア更新**: `Authorization: Bearer <jwt>` を Cookie の代替として受け付ける。トークン有効期限は 24 時間（環境変数 `JWT_EXPIRY_HOURS` で設定可能）。
3. **CI/CD 向け**: 環境変数 `KEIBA_API_TOKEN` での認証を許容する（テスト用途）。

```python
# auth.py の Bearer 検証イメージ
def get_current_user(request):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return verify_jwt(token)
    return get_user_from_session(request)
```

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| JWT Bearer（採用） | 仕様書準拠・API クライアント対応 | 実装工数中程度 |
| Cookie のみ継続 | 変更なし | 外部クライアント対応不可 |

---

### 影響範囲

- `src/api/auth.py` に JWT 生成・検証ロジック追加
- `src/api/flask_app.py` に `POST /api/v1/auth/login` エンドポイント追加
- `.env.example` に `JWT_SECRET`・`JWT_EXPIRY_HOURS` 追加

---

### 備考

JWT ライブラリは `PyJWT` を使用。`requirements.txt` への追加が必要か確認する。
