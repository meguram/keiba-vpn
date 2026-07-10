export type NavCardItem = {
  href: string;
  accent: string;
  icon: string;
  title: string;
  desc: string;
  tags: string[];
};

export type TabbedNavTab = {
  href: string;
  accent: string;
  icon: string;
  title: string;
  desc: string;
  tags: string[];
};

export type TabbedCardItem = {
  tabbed: true;
  title: string;
  tabs: TabbedNavTab[];
};

export function isTabbedCard(
  card: NavCardItem | TabbedCardItem
): card is TabbedCardItem {
  return (card as TabbedCardItem).tabbed === true;
}

export type CategorySection = {
  num: string;
  numStyle?: string;
  title: string;
  desc: string;
  catColor: string;
  badge?: string;
  singleRow?: boolean;
  cards: (NavCardItem | TabbedCardItem)[];
};

export const PUBLIC_CATEGORIES: CategorySection[] = [
  {
    num: "04",
    title: "🤖 AI 予測",
    desc: "今週の予測・追走難度・過去の予測結果",
    catColor: "var(--home-orange)",
    cards: [
      {
        tabbed: true as const,
        title: "今週の予測",
        tabs: [
          {
            href: "/weekly-predictions",
            accent: "var(--home-orange)",
            icon: "🏆",
            title: "今週のAI予測",
            desc: "今週のレース別 AI 推奨印 (◎○▲△☆) を一覧表示。出馬表・オッズ・個別レース詳細へ遷移可能。",
            tags: ["出馬表", "オッズ", "AI 推奨印"],
          },
          {
            href: "/megu-index",
            accent: "var(--home-cyan)",
            icon: "📊",
            title: "今週のめぐ指数",
            desc: "開催日ごとの全レースについて、めぐ指数の高い馬順にランキング表示。各馬のパフォーマンス指数を一覧で比較できる。",
            tags: ["めぐ指数", "ランキング", "パフォーマンス"],
          },
          {
            href: "/pedigree-race-stats",
            accent: "var(--home-purple)",
            icon: "🧬",
            title: "今週の血統傾向分析",
            desc: "予測されたレース質と出走馬の血統を照合し、各馬の血統適性スコアを点数化。対象レースでどれだけ血統的に優位かを可視化。",
            tags: ["血統相性", "レース質", "スコア"],
          },
        ],
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
        href: "/stallion-notes",
        accent: "var(--home-purple)",
        icon: "🐴",
        title: "種牡馬メモ",
        desc: "種牡馬・牝系ごとの血統ドメイン知識ベース。特徴・コース適性・配合傾向を整理。",
        tags: ["種牡馬", "牝系", "知識ベース"],
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
        href: "/data-analysis",
        accent: "var(--home-cyan)",
        icon: "🔬",
        title: "詳細データ分析",
        desc: "着順・タイム・AI予測・ROI など複数指標を自由に組み合わせて集計・可視化。分布・散布・ランキング・時系列の4モード対応。PostgreSQL からリアルタイム集計。",
        tags: ["分布分析", "散布図", "ランキング", "時系列"],
      },
      {
        href: "/growth-curve",
        accent: "var(--home-green)",
        icon: "📈",
        title: "成長曲線",
        desc: "馬ごとの年齢別パフォーマンスを成長曲線として可視化。早熟・晩成判定。",
        tags: ["年齢", "パフォーマンス"],
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
    num: "03",
    title: "🧬 血統（開発者向け）",
    desc: "開発者向けの高度な血統分析ツール",
    catColor: "var(--home-purple)",
    badge: "ADMIN",
    singleRow: true,
    cards: [
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
      {
        href: "/betting-simulation",
        accent: "var(--home-green)",
        icon: "🎰",
        title: "馬券シミュレーション",
        desc: "Kelly 基準の買い戦略を買い続けた場合の軍資金推移をモンテカルロシミュレーションで可視化。破産確率・最大ドローダウン・期待成長率を評価。",
        tags: ["モンテカルロ", "破産確率", "ドローダウン", "期待値"],
      },
    ],
  },
];
