import type { Metadata } from "next";
import { MaintenancePage } from "@/components/MaintenanceShell";

// メンテナンス中は middleware が全ページをここへ書き換える（URL は変えず、503 + noindex で返す）。
export const metadata: Metadata = {
  title: "メンテナンス中 | めぐ競馬",
  robots: { index: false, follow: false },
};

export default function Page() {
  return <MaintenancePage />;
}
