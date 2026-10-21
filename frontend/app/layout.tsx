import { MaintenanceShell } from "@/components/MaintenanceShell";
import { NavProgress } from "@/components/NavProgress";
import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "keiba-vpn",
  description: "競馬予測・データ分析",
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
