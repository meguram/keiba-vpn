import type { Metadata } from "next";
import { Suspense } from "react";
import { HomeDashboard } from "@/components/home/HomeDashboard";
import "@/styles/home.css";

export const metadata: Metadata = {
  title: "ML-AutoPilot Keiba",
  description: "Multi-Agent Horse Racing AI Prediction System",
};

export default function DashboardPage() {
  return (
    <Suspense fallback={null}>
      <HomeDashboard />
    </Suspense>
  );
}
