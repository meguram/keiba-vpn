"use client";

import { useEffect, useMemo, useState } from "react";
import { getJson, prefetchJson } from "@/lib/httpCache";
import { PositionForecast } from "@/components/PositionForecast";
import { GATE_TONE, PACE_TONE, buildTdMap } from "@/lib/positionForecast";
import type { ForecastHorse, GateBias, PaceForecast, PositionForecast as PosFc } from "@/lib/positionForecast";
import { bracketOf } from "@/lib/bracket";

/* 出馬表の下に置く、付随予測の折りたたみ。既定は閉じる（開閉は覚える）。
   - 展開予想: 予測 API が返す pace_forecast / gate_bias / 各馬の position_forecast（取得済み）に、
     追走難度・消耗・想定上がり（/api/race/{id}/tracking-difficulty。重いので開いたときに初めて取得）を重ねる
   - 想定ラップ: 同じ tracking-difficulty を共有（開閉どちらで取得しても 1 回） */

export type ExtraEntry = {
  horse_number?: number;
  horse_name?: string;
  bracket_number?: number | null;
  position_forecast?: PosFc | null;
  l3f_coherent?: L3fCoherent | null;
};

/** 整合層（KEIBA_PACE_COHERENCE=1 のときだけ API が返す）。位置取り側の先行勢から推定したペースと、位置×ペースで決めた想定上がり */
export type PaceUnified = {
  level: PaceForecast["level"];
  label: string;
  z?: number;
  first_3f_sec?: number | null;
  last_3f_baseline_sec?: number | null;
};
export type L3fCoherent = { dev_sec: number; seconds?: number | null; rank?: number };

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

type SectionKey = "pos" | "lap";
const STORE_KEY = "wp-extra-open";

function loadOpen(): Record<SectionKey, boolean> {
  const base = { pos: false, lap: false };
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    const s = raw ? JSON.parse(raw) : {};
    return { pos: !!(s.pos ?? s.flow), lap: !!s.lap };
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
  raceId, entries, pace: paceApi, paceUnified, gate,
}: {
  raceId: string;
  entries: ExtraEntry[];
  pace?: PaceForecast | null;
  paceUnified?: PaceUnified | null;
  gate?: GateBias | null;
}) {
  const unified = !!(paceUnified && paceUnified.level !== "unknown");
  const pace = useMemo(
    () => (unified && paceUnified && paceApi ? { ...paceApi, level: paceUnified.level, label: paceUnified.label } : paceApi),
    [unified, paceUnified, paceApi],
  );
  const [open, setOpen] = useState<Record<SectionKey, boolean>>({ pos: false, lap: false });
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
  const needTd = open.lap || open.pos;
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
  const tdMap = useMemo(() => {
    const m = buildTdMap(td?.position_flow, td?.entries);
    if (!unified || m.size === 0) return m;
    for (const e of entries) {
      const c = e.l3f_coherent;
      if (e.horse_number == null || !c || c.seconds == null) continue;
      m.set(e.horse_number, { ...(m.get(e.horse_number) ?? {}), last3f_sec: c.seconds, last3f_delta: c.dev_sec });
    }
    return m;
  }, [td, unified, entries]);
  const baseline = (unified ? paceUnified?.last_3f_baseline_sec : null) ?? pp?.last_3f_baseline_sec;
  const first3f = (unified ? paceUnified?.first_3f_sec : null) ?? pp?.lap_times?.first_3f;

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
        title="展開予想"
        open={open.pos}
        onToggle={() => toggle("pos")}
        onHover={() => prefetchJson(tdUrl, { ttlMs: 60_000 })}
        summary={[
          paceTone && pace ? `ペース ${pace.label}` : "",
          gateTone ? `枠順 ${gateTone.label}` : "",
          pp?.lap_times || (unified && baseline != null) ? `上がり基準 ${sec(baseline)}秒` : "",
        ].filter(Boolean).join(" ・ ") || (hasPos ? "位置取り・追走難度" : "出せません")}
      >
        {hasPos || pace ? (
          <PositionForecast
            embedded
            horses={horses}
            pace={pace}
            gate={gate}
            td={tdBroken ? null : tdMap}
            tdNote={
              <>
                {tdStatus}
                {tdBroken && <Note>追走難度・消耗・想定上がりは出せません（{String(td?.error)}）。位置取りの予想だけ表示しています。</Note>}
                {td && !tdBroken && tdMap.size === 0 && <Note>追走難度は出せません（出走馬の過去走が足りません）。</Note>}
                {tdMap.size > 0 && (
                  <Note>
                    追走難度（大きいほど前に付いていきやすい）・消耗は、位置取りの予想とは別に、過去走の統計で算出しています。脚質の表現がずれることがあります。
                    {unified
                      ? " 想定上がり（基準との差、−が速い）は、位置取り（後ろの馬ほど速い）とペースから決めた値で、追走難度とは別です。"
                      : " 想定上がり（基準との差、−が速い）も、位置取りとは別の過去走の統計です。"}
                    「展開」は、想定ペース（ハイなら差し・追込、スローなら逃げ・先行が向く）に追走のしやすさ（楽なら +、困難なら −）を足し合わせた目安で、AI予測ではありません。
                  </Note>
                )}
              </>
            }
          />
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
        summary={pp?.lap_times ? `前半3F ${sec(first3f)}秒 ・ 上がり基準 ${sec(baseline)}秒` : "前半と上がりの想定タイム"}
      >
        {tdStatus}
        {tdBroken && <Note>この条件では出せません（{String(td?.error)}）。</Note>}
        {td && !tdBroken && !pp?.lap_times && <Note>この条件では出せません（過去の同条件レースのラップが足りません）。</Note>}
        {pp?.lap_times && (
          <>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
              <Stat label="前半 1F" value={`${sec(pp.lap_times.first_1f)}秒`} />
              <Stat label="前半 3F" value={`${sec(first3f)}秒`} />
              <Stat label="上がり 3F の基準" value={`${sec(baseline)}秒`} sub="同条件の平均" />
              <Stat label="ペース" value={unified ? (paceUnified?.label ?? "—") : (pp.pace_type ?? "—")} sub={!unified && pp.pace_index != null ? `指数 ${pp.pace_index}` : undefined} />
            </div>
            {!unified && pp.predicted_pattern && <Note>{pp.predicted_pattern}</Note>}
            <Note>
              同じ条件の過去レースの平均ラップに、出走馬の脚質の構成を加味した統計ベースの目安です（モデルの予測ではありません）。
              {pp.pace_factors?.empirical_n != null && ` 参照した過去レース ${pp.pace_factors.empirical_n} 件。`}
              {unified
                ? " ペースは上の「展開予想」と同じ判定にそろえています（位置取りの先行勢から推定）。"
                : " 上の「展開予想」のペースとは算出方法が違うため、表現がずれることがあります。"}
            </Note>
          </>
        )}
      </Section>

    </div>
  );
}
