import type { Metadata } from "next";
import { HomeDashboard } from "@/components/home/HomeDashboard";
import "@/styles/home.css";

export const metadata: Metadata = {
  title: "ML-AutoPilot Keiba",
  description: "Multi-Agent Horse Racing AI Prediction System",
};

export default function DashboardPage() {
  return <HomeDashboard />;
}
