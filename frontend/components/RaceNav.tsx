"use client";

import { fetchRaceList } from "@/lib/httpCache";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { PendingLink } from "@/components/PendingLink";
import { USE_MOCK, MOCK_WEEKLY_RACES } from "@/lib/mock";
import { LEVEL_STYLE, TIER_LEGEND, TARGET_TIER_HINT, ValueLevel, chipHint, chipText, hasTier, isTargetValues, useRaceValuesState, mixedProbCalibration } from "@/lib/raceValue";
import { useNarrow } from "@/components/RaceStrip";

type RaceRow = { race_id: string; round: number; venue: string; race_name: string };

// 開催日ごとの一覧を共有（レース間を移動するたびに取り直さない）
const listCache = new Map<string, RaceRow[]>();

async function loadList(date: string): Promise<RaceRow[]> {
  const hit = listCache.get(date);
  if (hit) return hit;
  let rows: RaceRow[] = [];
  if (USE_MOCK) {
    rows = MOCK_WEEKLY_RACES.map((r) => ({ race_id: r.race_id, round: Number(r.round), venue: String(r.venue), race_name: String(r.race_name) }));
  } else {
    const d = (await fetchRaceList(date).catch(() => ({ races: [] }))) as { races?: Record<string, unknown>[] } | Record<string, unknown>[];
    rows = ((Array.isArray(d) ? d : d.races) ?? []).map((r: Record<string, unknown>) => ({
      race_id: String(r.race_id),
      round: Number(r.round ?? Number(String(r.race_id).slice(-2))),
      venue: String(r.venue ?? ""),
      race_name: String(r.race_name ?? ""),
    }));
  }
  listCache.set(date, rows);
  return rows;
}

const BTN: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 6, fontSize: 12,
  border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", textDecoration: "none", whiteSpace: "nowrap",
};

/** レース詳細の上部ナビ: 同じ開催場の前後のレース・R 一覧・他の開催場（同じ R）へ移動する。date は YYYYMMDD。 */
export function RaceNav({ raceId, date }: { raceId: string; date: string }) {
  const [rows, setRows] = useState<RaceRow[]>([]);
  const { values, loading: valuesLoading, failed: valuesFailed } = useRaceValuesState(date);
  const valuesState = { loading: valuesLoading, failed: valuesFailed };
  const narrow = useNarrow();

  useEffect(() => {
    if (!narrow) return;
    document.querySelector<HTMLElement>('nav[aria-label="レース移動"] [aria-current="page"][data-level]')?.scrollIntoView({ inline: "center", block: "nearest" });
  }, [narrow, raceId, rows]);

  useEffect(() => {
    let alive = true;
    if (!date) return;
    loadList(date).then((r) => alive && setRows(r)).catch(() => alive && setRows([]));
    return () => { alive = false; };
  }, [date]);

  const cur = rows.find((r) => r.race_id === raceId);
  // JRA の競馬場コード順（race_id の 5〜6 桁目）
  const venues = useMemo(() => {
    const code = new Map<string, string>();
    for (const r of rows) if (r.venue && !code.has(r.venue)) code.set(r.venue, r.race_id.slice(4, 6));
    return [...code.keys()].sort((a, b) => (code.get(a) ?? "").localeCompare(code.get(b) ?? ""));
  }, [rows]);
  const sameVenue = useMemo(
    () => rows.filter((r) => r.venue === cur?.venue).sort((a, b) => a.round - b.round),
    [rows, cur?.venue],
  );
  if (!cur || sameVenue.length === 0) return null;

  const idx = sameVenue.findIndex((r) => r.race_id === raceId);
  const prev = sameVenue[idx - 1];
  const next = sameVenue[idx + 1];
  const atVenue = (v: string) => {
    const list = rows.filter((r) => r.venue === v);
    return list.find((r) => r.round === cur.round) ?? [...list].sort((a, b) => a.round - b.round)[0];
  };
  const listHref = `/weekly-predictions?date=${date}&venue=${encodeURIComponent(cur.venue)}`;

  return (
    <nav aria-label="レース移動" style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: narrow ? "4px 6px" : "12px 16px", marginBottom: narrow ? 8 : 16, display: "grid", gridTemplateColumns: "minmax(0, 1fr)", gap: narrow ? 3 : 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: narrow ? "nowrap" : "wrap", overflowX: narrow ? "auto" : undefined }}>
        {venues.length > 1 && venues.map((v) => {
          const target = atVenue(v);
          const on = v === cur.venue;
          return (
            <PendingLink
              key={v}
              href={`/race/${target?.race_id ?? raceId}`}
              aria-current={on ? "page" : undefined}
              title={on ? "表示中の開催場" : `${v} ${target?.round ?? ""}R へ`}
              style={{
                ...BTN, fontSize: 13, fontWeight: on ? 700 : 500, flex: "none", minHeight: narrow ? 38 : undefined,
                border: `1px solid ${on ? "var(--accent)" : "var(--border)"}`,
                background: on ? "rgba(59,130,246,0.15)" : "var(--surface)",
                color: on ? "var(--accent)" : "var(--text-dim)",
              }}
            >
              {v}
              {(() => {
                const hot = rows.filter((r) => r.venue === v && values.get(r.race_id)?.level === "hot").length;
                return hot > 0 ? <span style={{ marginLeft: 6, fontSize: 11, fontWeight: 700, color: "#fb923c" }} title="特に価値の高いレース">★{hot}</span> : null;
              })()}
            </PendingLink>
          );
        })}
        <Link href={listHref} style={{ ...BTN, marginLeft: "auto", flex: "none", minHeight: narrow ? 38 : undefined, color: "var(--accent)" }}>
          {narrow ? "一覧へ" : `${cur.venue}の一覧へ（${date.slice(4, 6)}/${date.slice(6, 8)}）`}
        </Link>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: narrow ? "nowrap" : "wrap" }}>
        {!narrow && (prev ? (
          <PendingLink href={`/race/${prev.race_id}`} style={BTN} title={prev.race_name}>← {prev.round}R</PendingLink>
        ) : <span style={{ ...BTN, opacity: 0.35 }}>← 前のR</span>)}

        <div style={narrow
          ? { display: "flex", gap: 6, flexWrap: "nowrap", flex: 1, overflowX: "auto", padding: "2px 2px", scrollSnapType: "x proximity", WebkitOverflowScrolling: "touch" }
          : { display: "flex", gap: 4, flexWrap: "wrap", flex: 1, justifyContent: "center" }}>
          {sameVenue.map((r) => {
            const on = r.race_id === raceId;
            const v = values.get(r.race_id);
            const lv: ValueLevel = v?.level ?? "none";
            const st = LEVEL_STYLE[lv];
            return (
              <PendingLink
                key={r.race_id}
                href={`/race/${r.race_id}`}
                aria-current={on ? "page" : undefined}
                title={`${r.round}R ${r.race_name}\n${chipHint(v, valuesState)}`}
                data-level={lv}
                style={{
                  minWidth: narrow ? 46 : 46, minHeight: narrow ? 40 : undefined, flex: "none", scrollSnapAlign: "center", display: "inline-flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 1,
                  padding: narrow ? "3px 5px" : "5px 6px", borderRadius: 8, textDecoration: "none", lineHeight: 1.15,
                  border: `1px solid ${st.border}`, background: st.bg, color: st.color,
                  boxShadow: [on ? "0 0 0 2px var(--bg), 0 0 0 4px #e6edf3" : "", st.glow ?? ""].filter(Boolean).join(", ") || undefined,
                  opacity: lv === "pass" || lv === "none" ? (on ? 1 : 0.9) : 1,
                  transform: lv === "hot" ? "translateY(-1px)" : undefined,
                }}
              >
                <span style={{ fontSize: 14, fontWeight: on || lv === "hot" ? 800 : 600 }}>{lv === "hot" ? `★${r.round}` : r.round}</span>
                <span style={{ fontSize: 10, fontWeight: 600, opacity: 0.95 }}>
                  {chipText(v, valuesState)}
                </span>
              </PendingLink>
            );
          })}
        </div>

        {!narrow && (next ? (
          <PendingLink href={`/race/${next.race_id}`} style={BTN} title={next.race_name}>{next.round}R →</PendingLink>
        ) : <span style={{ ...BTN, opacity: 0.35 }}>次のR →</span>)}
      </div>

      {values.size > 0 && !narrow && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center", fontSize: 11, color: "var(--text-dim)" }}>
          {hasTier(values) ? (
            <>
              <span>買い目の推奨度:</span>
              {TIER_LEGEND.map((t) => (
                <span key={t.label} title={isTargetValues(values) ? TARGET_TIER_HINT[t.label] : LEVEL_STYLE[t.level].hint} style={{ padding: "1px 8px", borderRadius: 6, border: `1px solid ${LEVEL_STYLE[t.level].border}`, background: LEVEL_STYLE[t.level].bg, color: LEVEL_STYLE[t.level].color, opacity: t.level === "pass" ? 0.8 : 1 }}>
                  {t.level === "hot" ? "★ " : ""}{t.label}
                </span>
              ))}
              <span>{isTargetValues(values) ? "（選んだ買い目のどれかが当たる確率の大きさ。回収率の高さではありません）" : "（過去の実績に合わせて補正した目安。年による変動が大きい）"}</span>
            </>
          ) : (
            <>
              <span>推奨の期待回収率（較正後）:</span>
              {(["hot", "good", "ok", "pass"] as ValueLevel[]).map((k) => (
            <span key={k} title={LEVEL_STYLE[k].hint} style={{ padding: "1px 8px", borderRadius: 6, border: `1px solid ${LEVEL_STYLE[k].border}`, background: LEVEL_STYLE[k].bg, color: LEVEL_STYLE[k].color, opacity: k === "pass" ? 0.8 : 1 }}>
              {LEVEL_STYLE[k].label}
            </span>
          ))}
              <span>（過去の実績に合わせて補正した目安。年による変動が大きい）</span>
            </>
          )}
          {mixedProbCalibration(values) && <span title="オッズが出たレースは確率を市場補正して計算し、段の閾値と当日の順位を、補正しないレース（オッズ未確定・予測オッズ）とは別に付けています">※ 市場補正したレースとしないレースは基準が別</span>}
        </div>
      )}
    </nav>
  );
}
