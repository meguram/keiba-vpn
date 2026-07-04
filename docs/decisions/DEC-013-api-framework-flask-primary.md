# DEC-013: APIフレームワーク統一: Flask を正として FastAPI を段階廃止

| 項目 | 内容 |
|------|------|
| **日付** | 2026-07-04 |
| **ステータス** | accepted |
| **担当** | Orchestrator |
| **関連 AREA** | AREA-03 |
| **矛盾ID** | S-1-A |

---

### コンテキスト

仕様書（AREA-03・MASTER.md）はバックエンドフレームワークとして Flask を明記しているが、実装では src/api/app.py が FastAPI で稼働し、src/api/flask_app.py が Flask の別実装として並存している。CI テストは flask_app.py（Flask）を対象としており、本番稼働コードと乖離している。

---

### 決定事項

Flask（src/api/flask_app.py）を唯一の本番 API として確定する。FastAPI（src/api/app.py）は archive/src/api/app.py に移動し、廃止マークを付ける。フロントエンドの KEIBA_API_URL デフォルトは port 5000（Flask）のまま維持する。src/api/auth.py の PUBLIC_API_PREFIXES は /api/v1/ ベースに統一する。

---

### 選択肢と比較

| 選択肢 | メリット | デメリット |
|--------|---------|-----------|
| Flask（採用） | 仕様書との整合・CIがすでに対象 | 非同期IO非対応 |
| FastAPI | 非同期・Pydantic型安全 | 仕様書と乖離・二重メンテ |

---

### 影響範囲

- src/api/app.py を廃止（archive/ へ移動）
- src/api/auth.py の PUBLIC_API_PREFIXES を /api/v1/ ベースに修正
- wsgi.py・main.py のエントリポイントを flask_app.py に統一
- CI の対象は現在のまま（tests/e2e/test_api_v1.py の create_app = Flask の flask_app）

---

### 備考

FastAPI（app.py）が持つ血統・分析エンドポイントのうち、flask_app.py に未移植のものは DEC-014 フォローで移植またはプロキシ対応とする。
