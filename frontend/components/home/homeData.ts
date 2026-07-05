export type NavCardItem = {
  href: string;
  accent: string;
  icon: string;
  title: string;
  desc: string;
  tags: string[];
  onClick?: "focusJump";
};

export type CategorySection = {
  num: string;
  numStyle?: string;
  title: string;
  desc: string;
  catColor: string;
  badge?: string;
  singleRow?: boolean;
  cards: NavCardItem[];
};

export const PUBLIC_CATEGORIES: CategorySection[] = [
  {
    num: "04",
    title: "🤖 AI 予測",
    desc: "追走難度・レース別予測ランキング",
    catColor: "var(--home-orange)",
    cards: [
      {
        href: "/tracking-difficulty",
        accent: "var(--home-orange)",
        icon: "📊",
        title: "追走難度分析",
        desc: "ゲート位置・隣枠の馬の傾向・脚質構成から各馬のポジション取得容易度を推定。ML モデルによるスコアリング。",
        tags: ["ゲート", "脚質", "隣枠", "ML"],
      },
      {
        href: "#",
        accent: "var(--home-orange)",
        icon: "🏁",
        title: "レース予測結果",
        desc: "個別レースの出馬表・結果・オッズ・馬情報・「AI 予測」タブで推奨印 (◎○▲△☆) を確認。レース ID で直接移動可能。",
        tags: ["出馬表", "オッズ", "AI 推奨印"],
        onClick: "focusJump",
      },
    ],
  },
  {
    num: "03",
    title: "🧬 血統",
    desc: "血統研究・コース適性・構造マップ・遺伝子マーカー",
    catColor: "var(--home-purple)",
    cards: [
      {
        href: "/bloodline",
        accent: "var(--home-purple)",
        icon: "🧬",
        title: "血統 × 距離・コース 研究",
        desc: "種牡馬・母父の血統が距離適性・コース特性 (坂/直線/枠順/馬場/芝種) にどう作用するかを事前計算済みアーティファクトでまとめて閲覧。",
        tags: ["距離適性", "コース", "最適条件"],
      },
      {
        href: "/bloodline-vector",
        accent: "var(--home-purple)",
        icon: "🗺️",
        title: "血統マップ",
        desc: "種牡馬を血統ベクトルで埋め込み、2D マップ上で関係性を可視化。",
        tags: ["ベクトル", "UMAP"],
      },
      {
        href: "/pedigree-map",
        accent: "var(--home-purple)",
        icon: "🌳",
        title: "血統構造（サイアー系図）",
        desc: "種牡馬を父系で階層化して、サイアーライン全体を俯瞰。",
        tags: ["サイアー系図", "階層"],
      },
      {
        href: "/bloodline-cluster",
        accent: "var(--home-purple)",
        icon: "🪐",
        title: "メタクラスタ判定",
        desc: "馬名から血統メタクラスタを判定し、強み・弱みを推定。",
        tags: ["クラスタ", "強み/弱み"],
      },
      {
        href: "/pedigree-race-stats",
        accent: "var(--home-purple)",
        icon: "📈",
        title: "血統構成分析",
        desc: "レース別に出走馬の血統構成を集計し、共通する血統傾向を抽出。",
        tags: ["構成", "クロス"],
      },
      {
        href: "/myostatin",
        accent: "var(--home-pink)",
        icon: "🧪",
        title: "MSTN 遺伝子",
        desc: "マイオスタチン遺伝子型（C:C / C:T / T:T）の予測と距離適性。",
        tags: ["遺伝子", "距離型"],
      },
    ],
  },
  {
    num: "05",
    title: "📊 データ分析",
    desc: "成長曲線・適性・馬場速度などのドメイン分析",
    catColor: "var(--home-cyan)",
    cards: [
      {
        href: "/growth-curve",
        accent: "var(--home-green)",
        icon: "📈",
        title: "成長曲線",
        desc: "馬ごとの年齢別パフォーマンスを成長曲線として可視化。早熟・晩成判定。",
        tags: ["年齢", "パフォーマンス"],
      },
      {
        href: "/note-aptitude-race",
        accent: "var(--home-pink)",
        icon: "🎯",
        title: "適性 3D（NOTE 軸）",
        desc: "瞬発・持続・スタミナの 3D 空間で馬の適性ベクトルを可視化、レースとの相性を判定。",
        tags: ["3D", "適性ベクトル"],
      },
      {
        href: "/track-speed",
        accent: "var(--home-cyan)",
        icon: "⚡",
        title: "馬場速度",
        desc: "開催・トラック別の馬場速度指標を集計、当日の馬場傾向の参考に。",
        tags: ["馬場", "クッション"],
      },
    ],
  },
];

export const DEV_CATEGORIES: CategorySection[] = [
  {
    num: "02",
    title: "💰 馬券の最適化",
    desc: "期待値・的中率・ROI の最適化",
    catColor: "var(--home-green)",
    badge: "DEV",
    singleRow: true,
    cards: [
      {
        href: "/betting",
        accent: "var(--home-green)",
        icon: "💴",
        title: "馬券最適化",
        desc: "AI 予測ランキング × オッズから、期待値最大の券種・買い目をシミュレートし、ROI を可視化。バックテストとの整合も確認。",
        tags: ["期待値", "ROI", "バックテスト", "複合最適化"],
      },
    ],
  },
  {
    num: "01",
    title: "🛠 開発者モード",
    desc: "データ取得・チェック・運用ツール",
    catColor: "#f59e0b",
    badge: "DEV",
    cards: [
      {
        href: "/monitor",
        accent: "#f59e0b",
        icon: "📡",
        title: "モニター",
        desc: "スクレイピング状況・取得済みデータの俯瞰ボード。日付別レース一覧から個別レース詳細に遷移。",
        tags: ["monitor", "日付別"],
      },
      {
        href: "/data-viewer",
        accent: "#f59e0b",
        icon: "🗄️",
        title: "データビューア",
        desc: "GCS / ローカル上の生 JSON・特徴量 parquet を構造化して閲覧。馬・レース・騎手 etc.",
        tags: ["data-viewer", "parquet"],
      },
      {
        href: "/queue-status",
        accent: "#f59e0b",
        icon: "⏳",
        title: "スクレイピングキュー",
        desc: "ジョブキュー・ワーカー稼働・進捗・失敗一覧を一覧表示。手動キックや再投入も可能。",
        tags: ["queue", "workers"],
      },
      {
        href: "/scrape-upcoming",
        accent: "#f59e0b",
        icon: "⏰",
        title: "未来レース取得",
        desc: "出馬表・カレンダーなど将来日のレース情報を先行取得。",
        tags: ["upcoming", "出馬表"],
      },
      {
        href: "/server-logs",
        accent: "#f59e0b",
        icon: "📜",
        title: "サーバーログ",
        desc: "logs/*.log の末尾を Web から確認、エラー原因をすぐに追跡。",
        tags: ["logs", "tail"],
      },
    ],
  },
];

export const API_ENDPOINTS = [
  { method: "GET" as const, href: "/api/scrape-status?date=20260315", path: "/api/scrape-status", desc: "取得状況", external: true },
  { method: "GET" as const, href: "/api/scrape-dates?raw_keys=1", path: "/api/scrape-dates?raw_keys=1", desc: "日付キー一覧", external: true },
  { method: "GET" as const, href: "/api/scrape-jobs", path: "/api/scrape-jobs", desc: "ジョブ状況", external: true },
  { method: "POST" as const, href: "#", path: "/api/odds/train", desc: "オッズ予測モデル学習", action: "startOddsTrain" as const },
  { method: "GET" as const, href: "#", path: "/api/odds/train/status", desc: "オッズ学習状態（更新）", action: "pollOddsTrainOnce" as const },
  { method: "POST" as const, href: "#", path: "/api/simulation/run", desc: "バランス最適化", action: "runSimulation" as const },
  { method: "GET" as const, href: "/api/simulation/params", path: "/api/simulation/params", desc: "最適化結果", external: true },
  { method: "GET" as const, href: "/api/structure-status", path: "/api/structure-status", desc: "構造チェック", external: true },
  { method: "GET" as const, href: "/api/structure-fingerprints", path: "/api/structure-fingerprints", desc: "FP 情報", external: true },
];
