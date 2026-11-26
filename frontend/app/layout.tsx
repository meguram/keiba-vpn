import { EnvBadge } from "@/components/EnvBadge";
import { TableSorter } from "@/components/TableSorter";
import { MaintenanceShell } from "@/components/MaintenanceShell";
import { NavProgress } from "@/components/NavProgress";
import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { SiteFooter } from "@/components/SiteFooter";
import { AdSenseLoader } from "@/components/ads/AdSenseLoader";
import { ADSENSE_CLIENT } from "@/lib/adsense";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0d1117",
};

export const metadata: Metadata = {
  title: "めぐ競馬",
  description: "競馬予測・データ分析",
  // iOS が race_id・オッズ等の数字列を電話番号リンクにしない
  formatDetection: { telephone: false, address: false, email: false },
  // ロゴは public/megu-icon.png（Next が配る）。以前の /data/image/icon.jpg は Cloud Run に無く、本番・stg とも 404 で、冷えた起動のときは 14 秒待っていた（2026-11-26）
  icons: {
    icon: [{ url: "/megu-icon.png", type: "image/png" }],
    shortcut: "/megu-icon.png",
    apple: "/megu-icon.png",
  },
  // AdSense のサイトの所有の確認（meta タグ）。全区分で同じ HTML にするため、スクリプトとは別に置く（第三者のコードは読まない）
  other: { "google-adsense-account": ADSENSE_CLIENT },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <EnvBadge />
        <TableSorter />
        <AdSenseLoader />
        <MaintenanceShell
          chrome={
            <>
              <Nav />
              <NavProgress />
            </>
          }
        >
          {children}
          <SiteFooter />
        </MaintenanceShell>
      </body>
    </html>
  );
}
