# AREA-12 — 馬券シミュレーション（Betting Simulation）
**Status**: DRAFT | **Last Updated**: 2026-07-10

---

## 0. 目的・背景

### 0-1. 解決したい問題

馬券最適化（`/betting`、AREA-02 §4-14）は「1レースでのKelly最適ポートフォリオ」を出力するが、
**同じ戦略で買い続けた場合に軍資金がどう推移するか**は可視化されていない。

- 期待値がプラスでも短期では破産しうるのか？
- Kelly倍率を下げると安全になるが成長も遅くなるトレードオフはどの程度か？
- 何レース連続で運用すれば優位性が統計的に表れ始めるか？

これらの問いに答えるツールとして、**モンテカルロシミュレーション**により軍資金の確率的推移を可視化する開発者向けページを追加する。

### 0-2. 位置づけ

| 項目 | 内容 |
|---|---|
| URL | `/betting-simulation` |
| 認証 | 管理者専用（`/betting` と同等の ADMIN チェック） |
| ホーム配置 | `ADMIN_CATEGORIES` — `num="02"` 「💰 馬券の最適化」セクションに追加 |
| 計算場所 | **フロントエンドのみ**（Monte Carlo は JS で完結。API 不要） |
| 関連AREA | AREA-02（フロントエンド）, AREA-03（バックエンド/betting API） |

---

## 1. 機能要件

### 1-1. 必須機能（MVP）

| ID | 機能 |
|---|---|
| F-1 | モンテカルロシミュレーション（試行数・レース数・勝率・オッズ・Kelly倍率を入力し実行） |
| F-2 | 軍資金推移グラフ（パーセンタイルバンド付き折れ線） |
| F-3 | 最終軍資金分布ヒストグラム（破産ゾーン強調） |
| F-4 | KPI カード（5指標: 中央値最終軍資金 / 利益確率 / 破産確率 / 中央値最大ドローダウン / 期待成長率） |
| F-5 | `/betting` の最適化結果からパラメータを引き継ぐ（localStorage 経由） |
| F-6 | `USE_MOCK` 時はモックパラメータで即時動作 |

### 1-2. 拡張機能（Phase 2）

| ID | 機能 |
|---|---|
| F-7 | 複数戦略の並列比較（Full Kelly / Half Kelly / Quarter Kelly を同時描画） |
| F-8 | ドローダウングラフ（全試行の最大ドローダウン分布） |
| F-9 | API連携モード: `/api/v1/betting/optimize` 呼び出し→自動パラメータ取得 |
| F-10 | 結果の PNG エクスポート（canvas → download） |

---

## 2. シミュレーションアルゴリズム

### 2-1. 入力パラメータ

| 変数 | 記号 | UI ラベル | デフォルト | 範囲 |
|---|---|---|---|---|
| 初期軍資金 | `B₀` | 初期軍資金（円） | 100,000 | 1,000〜10,000,000 |
| 勝率 | `p` | 期待勝率（%） | 20.0 | 1〜99 |
| 単勝オッズ | `b+1` | 単勝オッズ（倍） | 5.5 | 1.1〜100 |
| Kelly 倍率 | `α` | Kelly 倍率 | 0.25 | 0.05〜1.0 |
| シミュレーションレース数 | `N` | レース数 | 100 | 10〜1000 |
| 試行回数 | `M` | 試行回数 | 1,000 | 100〜5,000 |
| 破産閾値 | `B_ruin` | 破産とみなす軍資金（円） | B₀ × 0.10 | B₀ × 0.01〜B₀ × 0.50 |

### 2-2. Kelly フラクション計算

```
b     = オッズ倍率 - 1     （例: 5.5倍 → b = 4.5）
q     = 1 - p              （負け確率）
f*    = (p × (b+1) - 1) / b = (p × オッズ倍率 - 1) / b
f_bet = α × f*             （実際に賭けるフラクション: Kelly倍率×フルKelly）
```

> edge = `p × (b+1) - 1` が正の場合のみ賭ける（期待値プラスの確認）。  
> edge ≤ 0 の場合は警告表示し、シミュレーションは実行するが結果を赤で警告する。

### 2-3. 1レースの処理

```typescript
function simulateOneRace(bankroll: number, p: number, b: number, f_bet: number): number {
  const stake = Math.min(f_bet * bankroll, bankroll);  // 軍資金を超えない
  const win   = Math.random() < p;
  return win
    ? bankroll + stake * b   // 勝ち: 掛け金 × b が純利益
    : bankroll - stake;      // 負け: 掛け金を失う
}
```

### 2-4. 1試行のシミュレーション

```typescript
function simulateTrial(B0, p, b, f_bet, N, B_ruin): TrialResult {
  const trajectory: number[] = [B0];  // race 0 = 初期値
  let bankroll = B0;
  let peakBankroll = B0;
  let maxDrawdown = 0;
  let ruined = false;

  for (let i = 0; i < N; i++) {
    bankroll = simulateOneRace(bankroll, p, b, f_bet);
    bankroll = Math.max(bankroll, 0);   // 借金なし

    // ドローダウン更新
    if (bankroll > peakBankroll) peakBankroll = bankroll;
    const dd = (peakBankroll - bankroll) / peakBankroll;
    if (dd > maxDrawdown) maxDrawdown = dd;

    trajectory.push(bankroll);

    // 破産判定（閾値以下で以降は固定）
    if (bankroll <= B_ruin) {
      ruined = true;
      while (trajectory.length <= N) trajectory.push(bankroll);
      break;
    }
  }

  return { trajectory, finalBankroll: bankroll, maxDrawdown, ruined };
}
```

### 2-5. M 試行の集計

```typescript
function runSimulation(params): SimulationResult {
  const { B0, p, b, alpha, N, M, B_ruin } = params;
  const b_net  = b - 1;        // オッズを net odds に変換
  const f_star = (p * b - 1) / b_net;
  const f_bet  = alpha * f_star;

  const trials = Array.from({ length: M }, () =>
    simulateTrial(B0, p, b_net, f_bet, N, B_ruin)
  );

  // レースiでのパーセンタイル
  const trajectoryPercentiles = computePercentiles(trials, [10, 25, 50, 75, 90]);

  return {
    params,
    f_star,
    f_bet,
    edge: p * b - 1,                                    // 期待値超過分
    medianFinal: percentile(trials.map(t => t.finalBankroll), 50),
    meanFinal:   mean(trials.map(t => t.finalBankroll)),
    ruinRate:    trials.filter(t => t.ruined).length / M,
    profitRate:  trials.filter(t => t.finalBankroll > B0).length / M,
    medianMaxDD: percentile(trials.map(t => t.maxDrawdown), 50),
    p90MaxDD:    percentile(trials.map(t => t.maxDrawdown), 90),
    trajectoryPercentiles,  // shape: { p10, p25, p50, p75, p90 }[]  長さ N+1
    finalDistribution: trials.map(t => t.finalBankroll),
  };
}
```

> **注意**: `M=1000` で `N=100` の場合、ループ回数は 100,000 回。最新ブラウザで 50〜200ms 程度。  
> `M=5000` では 500ms 超になる場合があるため、Web Worker への移行を検討する（Phase 2）。

---

## 3. UI 仕様

### 3-1. ページ全体レイアウト

```
┌─────────────────────────────────────────────────────────────────┐
│ ← ホーム  /  馬券シミュレーション              🎰 馬券シミュレーション │  ← ページヘッダー
├────────────────────┬────────────────────────────────────────────┤
│  設定パネル（左）  │  メインコンテンツ（右）                        │
│                    │                                            │
│  📥 入力パラメータ  │  KPI カード × 5                             │
│  ────────────────  │  ─────────────────────────────────────     │
│  初期軍資金        │  軍資金推移グラフ（大）                       │
│  期待勝率          │  ─────────────────────────────────────     │
│  単勝オッズ        │  最終軍資金分布ヒストグラム（中）              │
│  Kelly 倍率       │                                            │
│  レース数          │                                            │
│  試行回数          │                                            │
│  破産閾値          │                                            │
│                    │                                            │
│  ▶ シミュレーション実行│                                          │
│                    │                                            │
│  📊 最適化から引継ぎ│                                            │
└────────────────────┴────────────────────────────────────────────┘
```

モバイルでは設定パネルがトップに折りたたみ表示（`<details>` または Accordion）。

---

### 3-2. 設定パネル

| UI 要素 | 種別 | 補足 |
|---|---|---|
| 初期軍資金 | `number input` | 1,000〜10,000,000。カンマ区切り表示 |
| 期待勝率 | `range + number` | 1〜99%。スライダー付き |
| 単勝オッズ | `range + number` | 1.1〜100倍。小数第1位 |
| Kelly 倍率 | `select` | 1.0（Full）/ 0.5（Half）/ 0.25（Quarter）/ カスタム |
| レース数 | `select` | 50 / 100 / 200 / 500 |
| 試行回数 | `select` | 500 / 1,000 / 2,000 |
| 破産閾値（%） | `range + number` | 初期軍資金の 1%〜50%。デフォルト 10% |
| ▶ 実行ボタン | `button` | 押下でシミュレーション開始 |
| 📊 最適化から引継ぎ | `button` | localStorage の `betting_last_optimize` からパラメータを復元 |

**引継ぎボタンの動作**:
1. `localStorage.getItem('betting_last_optimize')` から `win_prob` / `win_odds` / `kelly_fraction` を読み込む
2. 対応フィールドに値を自動入力
3. 「最適化結果 (race_id: XXXX) から引き継ぎました」トースト表示

---

### 3-3. KPI カード（5枚）

| # | ラベル | 値の例 | 色 |
|---|---|---|---|
| K-1 | 中央値 最終軍資金 | ¥182,400 | 値に応じてB₀超→緑/B₀以下→橙 |
| K-2 | 利益達成率 | 73.2% | ≥60%→緑 / ≥40%→橙 / 未満→赤 |
| K-3 | 破産確率 | 4.8% | ≤5%→緑 / ≤15%→橙 / 超過→赤 |
| K-4 | 中央値 最大ドローダウン | 31.4% | ≤30%→緑 / ≤50%→橙 / 超過→赤 |
| K-5 | 期待成長率 | +82.4% | 符号でプラス→緑/マイナス→赤 |

---

### 3-4. 軍資金推移グラフ（F-2）

**ライブラリ**: Chart.js v4（既存利用と統一）

| 要素 | 仕様 |
|---|---|
| X 軸 | レース番号 0〜N |
| Y 軸 | 軍資金（円）。対数スケール切り替えボタン付き |
| p50 線 | 太い実線、青系 |
| p25〜p75 バンド | 薄い塗り（青, opacity 0.25）|
| p10〜p90 バンド | さらに薄い塗り（青, opacity 0.12）|
| 初期軍資金横線 | 破線グレー、ラベル「初期: ¥B₀」 |
| 破産閾値横線 | 破線赤、ラベル「破産閾値: ¥B_ruin」 |
| ツールチップ | レース番号・p10/p50/p90の3値を表示 |

---

### 3-5. 最終軍資金分布ヒストグラム（F-3）

**ライブラリ**: Chart.js v4 (Bar)

| 要素 | 仕様 |
|---|---|
| X 軸 | 最終軍資金（円）。ビン幅は自動（スタージェスの公式） |
| Y 軸 | 試行数（頻度） |
| ビン色 | 破産閾値以下 → 赤 / B₀以下 → 橙 / B₀以上 → 緑 |
| 縦線 | p50 = 青破線、初期軍資金 = グレー破線 |

---

### 3-6. エッジ・推奨表示

設定パネル下部に算出結果サマリーを常時表示:

```
期待値（edge）: +9.3%  →  ✅ 賭け推奨（正の期待値）
フルKelly f*:  18.2%   →  今回の実賭け (Quarter): 4.6%
```

edge ≤ 0 の場合:
```
期待値（edge）: -2.1%  →  ⚠ 期待値マイナス（賭けは非推奨）
```

---

## 4. データフロー

```
[設定パネル入力]
      │
      ▼
[パラメータバリデーション]
  edge = p × odds - 1
  f*   = (p × odds - 1) / (odds - 1)
      │
      ▼
[Monte Carlo ループ（JS同期）]
  M 試行 × N レース
      │
      ▼
[集計（percentiles, distribution）]
      │
      ├──▶ KPI カード更新
      ├──▶ 軍資金推移グラフ更新
      └──▶ 最終軍資金ヒストグラム更新
```

**API呼び出し**: なし（MVP）。  
F-9（Phase 2）で `/api/v1/betting/optimize` を呼び出してパラメータを自動取得するオプションを追加予定。

---

## 5. localStorage 連携（`/betting` → `/betting-simulation`）

`/betting` 側で最適化実行時に以下をシリアライズして保存:

```typescript
// /betting/page.tsx 側に追加
localStorage.setItem('betting_last_optimize', JSON.stringify({
  race_id:        result.race_id,
  win_prob:       result.bets[0]?.win_prob ?? null,   // 最も高い kelly_f の馬
  win_odds:       result.bets[0]?.win_odds ?? null,
  kelly_fraction: result.kelly_fraction,
  bankroll:       result.bankroll,
  saved_at:       new Date().toISOString(),
}));
```

`/betting-simulation` 側で「最適化から引継ぎ」ボタン押下時に復元:

```typescript
const raw = localStorage.getItem('betting_last_optimize');
if (raw) {
  const data = JSON.parse(raw);
  setWinProb(data.win_prob * 100);   // % 表示
  setWinOdds(data.win_odds);
  setKellyMult(data.kelly_fraction);
  setBankroll(data.bankroll);
}
```

---

## 6. モックデータ

`lib/mock.ts` に追加する定数:

```typescript
export const MOCK_SIMULATION_PARAMS = {
  bankroll:       100_000,
  win_prob:       0.200,    // 20%
  win_odds:       5.5,      // 5.5倍
  kelly_fraction: 0.25,     // Quarter Kelly
  n_races:        100,
  n_trials:       1_000,
  ruin_threshold: 0.10,     // 初期軍資金の10%
};
```

`USE_MOCK=true` の場合: 上記パラメータをデフォルト値として使用し、認証チェックをスキップ。

---

## 7. 認証・権限

`/betting` と同じパターンを踏襲:

```typescript
useEffect(() => {
  if (USE_MOCK) { setIsAdmin(true); return; }
  fetch('/api/v1/auth/status', { credentials: 'include' })
    .then(r => r.ok ? r.json() : { logged_in: false, is_admin: false })
    .then(d => setIsAdmin(!!d.logged_in && !!d.is_admin))
    .catch(() => setIsAdmin(false));
}, []);
```

未認証・非 Admin の場合は「🔒 管理者専用ページ」ロック画面を表示（`/betting` と同一 UI）。

---

## 8. homeData.ts 追加仕様

`ADMIN_CATEGORIES` の `num="02"` セクション (`cards` 配列) に追加:

```typescript
{
  href:   "/betting-simulation",
  accent: "var(--home-green)",
  icon:   "🎰",
  title:  "馬券シミュレーション",
  desc:   "Kelly基準の買い戦略を買い続けた場合の軍資金推移をモンテカルロシミュレーションで可視化。破産確率・最大ドローダウン・期待成長率を評価。",
  tags:   ["モンテカルロ", "破産確率", "ドローダウン", "期待値"],
}
```

---

## 9. 確定済み仕様

| 項目 | 決定 |
|---|---|
| 計算実行場所 | フロントエンド（JS同期）。API 不要 |
| 乱数 | `Math.random()` (Xorshift 相当)。再現性不要のため seed 固定なし |
| 借金 | なし（bankroll の下限 = 0） |
| 最低試行数 | M ≥ 100（プリセットに 100 は含めない、最小 500） |
| ライブラリ | Chart.js v4（既存踏襲）。D3 は不使用 |
| スタイル | CSS Variables + Tailwind（既存踏襲） |
| Admin 認証 | `/api/v1/auth/status` チェック（`USE_MOCK` でバイパス） |

---

## 10. 未決定事項

| # | 項目 | 検討ポイント |
|---|---|---|
| S-1 | Web Worker 移行の閾値 | M=5000 で体感 lag が生じるようなら async 化。まず同期で実装し体感評価 |
| S-2 | 複数戦略同時比較（F-7）の UI | 折れ線3本 + 凡例か、セレクトタブ切り替えか |
| S-3 | バックテスト実績との照合 | `/api/v1/betting/backtest` エンドポイントが整備されたら実測値のオーバーレイを追加 |

---

## 11. 実装タスク分解

| # | タスク | 担当 | 優先度 |
|---|---|---|---|
| T-1 | `AREA-12-betting-simulation.md` 仕様書作成 | — | ✅ 完了 |
| T-2 | `lib/mock.ts` に `MOCK_SIMULATION_PARAMS` 追加 | frontend-engineer | 高 |
| T-3 | `app/betting-simulation/page.tsx` 実装（設定パネル + KPI + グラフ） | frontend-engineer | 高 |
| T-4 | `components/home/homeData.ts` にカード追加 | frontend-engineer | 高 |
| T-5 | `/betting/page.tsx` に `localStorage` 保存を追加 | frontend-engineer | 中 |
| T-6 | Phase 2: ドローダウングラフ（F-8）追加 | frontend-engineer | 低 |
| T-7 | Phase 2: API連携モード（F-9）追加 | fullstack-integrator | 低 |
