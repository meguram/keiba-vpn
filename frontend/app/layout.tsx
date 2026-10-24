import { EnvBadge } from "@/components/EnvBadge";
import { TableSorter } from "@/components/TableSorter";
import { MaintenanceShell } from "@/components/MaintenanceShell";
import { NavProgress } from "@/components/NavProgress";
import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

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
  // ロゴはホストの data/local/image/icon.jpg（FastAPI の /data/image/ を next.config.js の rewrites で配信）
  icons: {
    icon: [{ url: "/data/image/icon.jpg", type: "image/jpeg" }],
    shortcut: "/data/image/icon.jpg",
    apple: "/data/image/icon.jpg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <EnvBadge />
        <TableSorter />
        <MaintenanceShell
          chrome={
            <>
              <Nav />
              <NavProgress />
            </>
          }
        >
          {children}
        </MaintenanceShell>
      </body>
    </html>
  );
}
