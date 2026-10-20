"use client";

import { useEffect, useMemo, useState } from "react";
import { getJson, prefetchJson } from "@/lib/httpCache";
import { PositionForecast } from "@/components/PositionForecast";
import { GATE_TONE, PACE_TONE } from "@/lib/positionForecast";
import type { ForecastHorse, GateBias, PaceForecast, PositionForecast as PosFc } from "@/lib/positionForecast";
import { bracketColor, bracketOf } from "@/lib/bracket";

/* 出馬表の下に置く、付随予測の折りたたみ。既定は閉じる（開閉は覚える）。
   - 展開予想: 予測 API が返す pace_forecast / gate_bias / 各馬の position_forecast（取得済みなので追加の通信なし）
   - 想定ラップ・位置取りの流れ: /api/race/{id}/tracking-difficulty（重いので開いたときに初めて取得） */

export type ExtraEntry = {
  horse_number?: number;
  horse_name?: string;
  bracket_number?: number | null;
  position_forecast?: PosFc | null;
};

type TdEntry = { horse_number?: number; tracking_difficulty?: { ease_pct?: number; label?: string } };
type TdFlow = {
  horse_number?: number;
  horse_name?: string;
  style?: string;
  flow_pattern?: string;
  stamina_concern?: boolean;
  stamina?: { stamina_label?: string };
  expected_last_3f?: { seconds?: number; delta_sec?: number; label?: string };
};
type TdData = {
  error?: string;
  pace_prediction?: {
    pace_type?: string;
    pace_index?: number;
    predicted_pattern?: string;
    lap_times?: { first_1f?: number; first_3f?: number };
    last_3f_baseline_sec?: number;
    front_runner_count?: number;
    stalker_count?: number;
    pace_factors?: { baseline_source?: string; empirical_n?: number };
  };
  position_flow?: TdFlow[];
  entries?: TdEntry[];
};

type SectionKey = "pos" | "lap" | "flow";
const STORE_KEY = "wp-extra-open";

function loadOpen(): Record<SectionKey, boolean> {
  const base = { pos: false, lap: false, flow: false };
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    return raw ? { ...base, ...JSON.parse(raw) } : base;
  } catch {
    return base;
  }
}

function Section({
  id, title, summary, open, onToggle, onHover, children,
}: {
  id: string; title: string; summary?: React.ReactNode; open: boolean; onToggle: () => void; onHover?: () => void; children: React.ReactNode;
}) {
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 8, background: "var(--surface)", overflow: "hidden" }}>
      <button
        type="button"
        onClick={onToggle}
        onMouseEnter={onHover}
        onTouchStart={onHover}
        aria-expanded={open}
        aria-controls={id}
        style={{ width: "100%", minHeight: 44, padding: "8px 12px", display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", textAlign: "left", background: "var(--surface2)", color: "var(--text)", border: "none", cursor: "pointer", font: "inherit" }}
      >
        <span aria-hidden style={{ display: "inline-block", width: 12, color: "var(--text-dim)", transform: open ? "rotate(90deg)" : "none", transition: "transform .12s" }}>▶</span>
        <strong style={{ fontSize: 14 }}>{title}</strong>
        {summary && <span style={{ fontSize: 12, color: "var(--text-dim)", minWidth: 0 }}>{summary}</span>}
      </button>
      {open && <div id={id} style={{ padding: "10px 12px 12px", borderTop: "1px solid var(--border)" }}>{children}</div>}
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p style={{ margin: "0 0 8px", fontSize: 12, color: "var(--text-dim)", lineHeight: 1.6 }}>{children}</p>;
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{ flex: "1 1 120px", minWidth: 0, padding: "8px 10px", borderRadius: 8, background: "var(--surface2)", border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800, color: "#f0f6fc", fontVariantNumeric: "tabular-nums" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{sub}</div>}
    </div>
  );
}

const sec = (v?: number) => (v != null ? v.toFixed(1) : "—");

export function PredExtras({
  raceId, entries, pace, gate,
}: {
  raceId: string;
  entries: ExtraEntry[];
  pace?: PaceForecast | null;
  gate?: GateBias | null;
}) {
  const [open, setOpen] = useState<Record<SectionKey, boolean>>({ pos: false, lap: false, flow: false });
  const [td, setTd] = useState<TdData | null>(null);
  const [tdState, setTdState] = useState<"idle" | "loading" | "error">("idle");
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => { setOpen(loadOpen()); }, []);

  const toggle = (k: SectionKey) => setOpen((cur) => {
    const next = { ...cur, [k]: !cur[k] };
    try { window.localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch { /* 記憶できなくても動く */ }
    return next;
  });

  const tdUrl = `/api/race/${raceId}/tracking-difficulty`;
  const needTd = open.lap || open.flow;
  useEffect(() => {
    if (!needTd) return;
    let alive = true;
    setTdState("loading");
    getJson<TdData>(tdUrl, { ttlMs: 60_000, force: reloadKey > 0 })
      .then((d) => { if (alive) { setTd(d); setTdState("idle"); } })
      .catch((e: { status?: number }) => {
        if (!alive) return;
        // 404 = このレースの事前計算が無い（取得の失敗ではなく「出せない」）
        if (e?.status === 404) { setTd({ error: "このレースの事前計算データがまだありません" }); setTdState("idle"); }
        else setTdState("error");
      });
    return () => { alive = false; };
  }, [needTd, tdUrl, reloadKey]);

  const horses: ForecastHorse[] = useMemo(
    () => [...entries]
      .filter((p) => p.horse_number != null)
      .sort((a, b) => (a.horse_number ?? 0) - (b.horse_number ?? 0))
      .map((p) => ({ no: p.horse_number as number, name: p.horse_name ?? "", bracket: bracketOf(p.horse_number as number, entries.length, p.bracket_number), f: p.position_forecast ?? null })),
    [entries],
  );
  const hasPos = horses.some((h) => h.f);
  const gateTone = gate ? GATE_TONE[gate.level] : null;
  const paceTone = pace ? PACE_TONE[pace.level] : null;

  const pp = td && !td.error ? td.pace_prediction : undefined;
  const tdBroken = !!td?.error;
  const flowRows = useMemo(() => {
    const ease = new Map<number, TdEntry["tracking_difficulty"]>();
    for (const e of td?.entries ?? []) if (e.horse_number != null) ease.set(e.horse_number, e.tracking_difficulty);
    const byNo = new Map(horses.map((h) => [h.no, h]));
    return [...(td?.position_flow ?? [])]
      .filter((f) => f.horse_number != null)
      .sort((a, b) => (a.horse_number ?? 0) - (b.horse_number ?? 0))
      .map((f) => ({ f, h: byNo.get(f.horse_number as number), ease: ease.get(f.horse_number as number) }));
  }, [td, horses]);

  const tdLoading = tdState === "loading" && !td;
  const tdError = tdState === "error" && !td;
  const retry = (
    <button type="button" onClick={() => setReloadKey((n) => n + 1)} style={{ marginLeft: 8, minHeight: 44, padding: "0 14px", borderRadius: 8, border: "1px solid var(--accent)", background: "rgba(59,130,246,0.15)", color: "var(--accent)", cursor: "pointer", fontSize: 13 }}>
      再読み込み
    </button>
  );
  const tdStatus = tdLoading
    ? <div role="status" style={{ fontSize: 13, color: "var(--text-dim)", padding: "6px 0" }}>取得中…</div>
    : tdError
      ? <div role="alert" style={{ fontSize: 13, color: "var(--err)" }}>⚠️ 取得できませんでした。{retry}</div>
      : null;

  if (!horses.length) return null;

  return (
    <div aria-label="付随する予測" style={{ display: "grid", gap: 8, marginTop: 12 }}>
      <Section
        id={`wp-extra-pos-${raceId}`}
        title="展開予想（位置取り・ペース）"
        open={open.pos}
        onToggle={() => toggle("pos")}
        summary={[
          paceTone && pace ? `ペース ${pace.label}` : "",
          gateTone ? `枠順 ${gateTone.label}` : "",
        ].filter(Boolean).join(" ・ ") || (hasPos ? "" : "出せません")}
      >
        {hasPos || pace ? (
          <PositionForecast horses={horses} pace={pace} gate={gate} />
        ) : (
          <Note>この条件では出せません（出走馬の過去走の通過順位がありません）。</Note>
        )}
      </Section>

      <Section
        id={`wp-extra-lap-${raceId}`}
        title="想定ラップ"
        open={open.lap}
        onToggle={() => toggle("lap")}
        onHover={() => prefetchJson(tdUrl, { ttlMs: 60_000 })}
        summary={pp?.lap_times ? `前半3F ${sec(pp.lap_times.first_3f)}秒 ・ 上がり基準 ${sec(pp.last_3f_baseline_sec)}秒` : "前半と上がりの想定タイム"}
      >
        {tdStatus}
        {tdBroken && <Note>この条件では出せません（{String(td?.error)}）。</Note>}
        {td && !tdBroken && !pp?.lap_times && <Note>この条件では出せません（過去の同条件レースのラップが足りません）。</Note>}
        {pp?.lap_times && (
          <>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <Stat label="前半 1F" value={`${sec(pp.lap_times.first_1f)}秒`} />
              <Stat label="前半 3F" value={`${sec(pp.lap_times.first_3f)}秒`} />
              <Stat label="上がり 3F の基準" value={`${sec(pp.last_3f_baseline_sec)}秒`} sub="同条件の平均" />
              <Stat label="ペース" value={pp.pace_type ?? "—"} sub={pp.pace_index != null ? `指数 ${pp.pace_index}` : undefined} />
            </div>
            {pp.predicted_pattern && <Note>{pp.predicted_pattern}</Note>}
            <Note>
              同じ条件の過去レースの平均ラップに、出走馬の脚質の構成を加味した統計ベースの目安です（モデルの予測ではありません）。
              {pp.pace_factors?.empirical_n != null && ` 参照した過去レース ${pp.pace_factors.empirical_n} 件。`}
              {" "}上の「展開予想」のペースとは算出方法が違うため、表現がずれることがあります。
            </Note>
          </>
        )}
      </Section>

      <Section
        id={`wp-extra-flow-${raceId}`}
        title="位置取りの流れ・追走難度"
        open={open.flow}
        onToggle={() => toggle("flow")}
        onHover={() => prefetchJson(tdUrl, { ttlMs: 60_000 })}
        summary="馬ごとの脚質・想定上がり・追走のしやすさ"
      >
        {tdStatus}
        {tdBroken && <Note>この条件では出せません（{String(td?.error)}）。</Note>}
        {td && !tdBroken && flowRows.length === 0 && <Note>この条件では出せません（出走馬の過去走が足りません）。</Note>}
        {flowRows.length > 0 && (
          <>
            <Note>追走難度: 数字が大きいほど、その馬にとって前に付いていきやすい（％）。想定上がりは同条件の基準との差（−が速い）。過去走からの統計ベースの目安です。</Note>
            <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 6 }}>
              {flowRows.map(({ f, h, ease }) => {
                const b = bracketColor(h?.bracket);
                const d = f.expected_last_3f?.delta_sec;
                return (
                  <li key={f.horse_number} style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", padding: "6px 8px", borderRadius: 8, background: "var(--surface2)", border: "1px solid var(--border)", fontSize: 12 }}>
                    <span style={{ minWidth: 26, height: 26, borderRadius: 999, background: b.bg, color: b.color, border: `1px solid ${b.border}`, boxSizing: "border-box", display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13 }}>{f.horse_number}</span>
                    <strong style={{ flex: "1 1 110px", minWidth: 0, fontSize: 13, color: "#f0f6fc", overflowWrap: "anywhere" }}>{h?.name || f.horse_name || ""}</strong>
                    <span style={{ color: "var(--text-dim)" }}>脚質 <b style={{ color: "var(--text)" }}>{f.style ?? "—"}</b></span>
                    {f.flow_pattern && <span style={{ color: "var(--text-dim)" }}>流れ <b style={{ color: "var(--text)" }}>{f.flow_pattern}</b></span>}
                    {f.stamina?.stamina_label && <span style={{ color: f.stamina_concern ? "#fbbf24" : "var(--text-dim)" }}>消耗 <b>{f.stamina.stamina_label}</b></span>}
                    {f.expected_last_3f?.seconds != null && (
                      <span style={{ color: "var(--text-dim)" }}>想定上がり <b style={{ color: "var(--text)" }}>{f.expected_last_3f.seconds.toFixed(1)}</b>{d != null && `（${d > 0 ? "+" : ""}${d.toFixed(1)}）`}</span>
                    )}
                    {ease?.ease_pct != null && <span style={{ color: "var(--text-dim)" }}>追走 <b style={{ color: "var(--text)" }}>{ease.ease_pct.toFixed(0)}%</b>{ease.label ? `（${ease.label}）` : ""}</span>}
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </Section>
    </div>
  );
}
