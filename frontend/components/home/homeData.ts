export type NavCardItem = {
  href: string;
  accent: string;
  icon: string;
  title: string;
  desc: string;
  tags: string[];
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
    desc: "今週の予測・追走難度・過去の予測結果",
    catColor: "var(--home-orange)",
    cards: [
      {
        href: "/weekly-predictions",
        accent: "var(--home-orange)",
        icon: "🏆",
        title: "今週のAI予測",
        desc: "今週のレース別 AI 推奨印 (◎○▲△☆) を一覧表示。出馬表・オッズ・個別レース詳細へ遷移可能。",
        tags: ["出馬表", "オッズ", "AI 推奨印"],
      },
      {
        href: "/tracking-difficulty",
        accent: "var(--home-orange)",
        icon: "📊",
        title: "追走難度分析",
        desc: "ゲート位置・隣枠の馬の傾向・脚質構成から各馬のポジション取得容易度を推定。ML モデルによるスコアリング。",
        tags: ["ゲート", "脚質", "隣枠", "ML"],
      },
      {
        href: "/races",
        accent: "var(--home-orange)",
        icon: "📋",
        title: "過去の予測結果",
        desc: "過去レースの出馬表・レース結果・AI 予測との照合を確認。レース詳細ページから各馬の推奨印と実際の着順を比較。",
        tags: ["レース一覧", "着順", "予測照合"],
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

export const ADMIN_CATEGORIES: CategorySection[] = [
  {
    num: "02",
    title: "💰 馬券の最適化",
    desc: "期待値・的中率・ROI の最適化",
    catColor: "var(--home-green)",
    badge: "ADMIN",
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
];
