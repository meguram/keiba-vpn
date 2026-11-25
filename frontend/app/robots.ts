import type { MetadataRoute } from "next";

// このサイトは一般公開向けではない。全クローラーを拒否する（robots.txt は守る相手にしか効かない。実効的な防御は API 側の認証・レート制限）。
// 主要な AI 学習・検索クローラーは名前でも明示する。
const AI_AND_SCRAPER_BOTS = [
  "GPTBot",
  "ChatGPT-User",
  "OAI-SearchBot",
  "ClaudeBot",
  "Claude-Web",
  "anthropic-ai",
  "Google-Extended",
  "GoogleOther",
  "CCBot",
  "PerplexityBot",
  "Perplexity-User",
  "Bytespider",
  "Amazonbot",
  "Applebot-Extended",
  "cohere-ai",
  "Diffbot",
  "FacebookBot",
  "meta-externalagent",
  "ImagesiftBot",
  "Omgilibot",
  "YouBot",
  "Timpibot",
  "AhrefsBot",
  "SemrushBot",
  "MJ12bot",
  "DotBot",
  "PetalBot",
  "Scrapy",
];

// 広告（AdSense）の巡回だけは許す。審査・広告の中身の判定・所有の確認に要る（support.google.com/adsense/answer/10532）。
// 検索の索引は今までどおり拒否（下の * と X-Robots-Tag の noindex）。
const ADSENSE_BOTS = ["Mediapartners-Google", "Google-Display-Ads-Bot"];

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      // /api/ は閉じたまま（Nav の罠のリンクをたどって巡回の IP が止められないように。API の要求も増やさない）
      ...ADSENSE_BOTS.map((userAgent) => ({ userAgent, allow: "/", disallow: "/api/" })),
      { userAgent: "*", disallow: "/" },
      ...AI_AND_SCRAPER_BOTS.map((userAgent) => ({ userAgent, disallow: "/" })),
    ],
  };
}
