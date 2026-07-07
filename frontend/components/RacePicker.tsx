"use client";

import { useEffect, useState, useCallback } from "react";
import { USE_MOCK, getMockRaceDates, MOCK_WEEKLY_RACES } from "@/lib/mock";

export type RaceItem = {
  race_id: string;
  race_name?: string;
  venue?: string;
  round?: number | string;
  name?: string;
  [key: string]: unknown;
};

type Props = {
  onRaceSelect: (raceId: string, label: string) => void;
  defaultRaceId?: string;
  analyzeLabel?: string;
  onAnalyze?: (raceId: string) => void;
  analyzing?: boolean;
  statusMsg?: string;
  extraControls?: React.ReactNode;
};

const SEL = {
  bg: "var(--surface2)",
  border: "1px solid var(--border)",
  color: "var(--text)",
  padding: "6px 10px",
  borderRadius: 5,
  fontSize: 12,
} as React.CSSProperties;

const BTN = {
  background: "var(--accent)",
  color: "#fff",
  border: "none",
  padding: "7px 16px",
  borderRadius: 5,
  fontSize: 12,
  cursor: "pointer",
  fontWeight: 600,
} as React.CSSProperties;

export function RacePicker({
  onRaceSelect,
  onAnalyze,
  analyzeLabel = "分析実行",
  analyzing = false,
  statusMsg,
  extraControls,
}: Props) {
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [racesForDate, setRacesForDate] = useState<RaceItem[]>([]);
  const [venues, setVenues] = useState<string[]>([]);
  const [selectedVenue, setSelectedVenue] = useState("");
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingRaces, setLoadingRaces] = useState(false);

  useEffect(() => {
    if (USE_MOCK) {
      const mockDates = getMockRaceDates();
      setDates(mockDates);
      setSelectedDate(mockDates[0]);
      setLoadingDates(false);
      return;
    }
    (async () => {
      try {
        const res = await fetch(`/api/scrape-dates?picker_past_days=14`, { cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        const list: string[] = data.dates ?? data ?? [];
        setDates(list);
        if (list.length) setSelectedDate(list[0]);
      } catch { /* ignore */ } finally {
        setLoadingDates(false);
      }
    })();
  }, []);

  const loadRaces = useCallback(async (date: string) => {
    if (!date) return;
    setLoadingRaces(true);
    setRacesForDate([]);
    setVenues([]);
    setSelectedVenue("");
    setSelectedRaceId("");
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 150));
      const list: RaceItem[] = MOCK_WEEKLY_RACES.map((r) => ({ ...r, date, name: r.race_name }));
      setRacesForDate(list);
      const venueSet = [...new Set(list.map((r) => r.venue ?? "").filter(Boolean))];
      setVenues(venueSet);
      if (venueSet.length) {
        setSelectedVenue(venueSet[0]);
        const first = list.find((r) => r.venue === venueSet[0]);
        if (first) { setSelectedRaceId(first.race_id); onRaceSelect(first.race_id, raceLabel(first)); }
      }
      setLoadingRaces(false);
      return;
    }
    try {
      const res = await fetch(`/api/race-list/${date}`);
      if (!res.ok) return;
      const data = await res.json();
      const list: RaceItem[] = data.races ?? data ?? [];
      setRacesForDate(list);
      const venueSet = [...new Set(list.map((r) => r.venue ?? "").filter(Boolean))];
      setVenues(venueSet);
      if (venueSet.length) {
        setSelectedVenue(venueSet[0]);
        const first = list.find((r) => r.venue === venueSet[0]);
        if (first) {
          setSelectedRaceId(first.race_id);
          onRaceSelect(first.race_id, raceLabel(first));
        }
      }
    } catch { /* ignore */ } finally {
      setLoadingRaces(false);
    }
  }, [onRaceSelect]);

  useEffect(() => {
    if (selectedDate) loadRaces(selectedDate);
  }, [selectedDate, loadRaces]);

  function raceLabel(r: RaceItem) {
    const rnd = r.round ?? "";
    const nm = r.race_name ?? r.name ?? r.race_id;
    return rnd ? `${rnd}R ${nm}` : String(nm);
  }

  const filteredRaces = racesForDate.filter(
    (r) => !selectedVenue || r.venue === selectedVenue
  );

  function onVenueChange(v: string) {
    setSelectedVenue(v);
    const first = racesForDate.find((r) => r.venue === v);
    if (first) {
      setSelectedRaceId(first.race_id);
      onRaceSelect(first.race_id, raceLabel(first));
    }
  }

  function onRaceChange(id: string) {
    setSelectedRaceId(id);
    const r = racesForDate.find((x) => x.race_id === id);
    if (r) onRaceSelect(id, raceLabel(r));
  }

  return (
    <div
      style={{
        background: "var(--surface)",
        borderBottom: "1px solid var(--border)",
        padding: "12px 24px",
        display: "flex",
        alignItems: "center",
        gap: 12,
        flexWrap: "wrap",
      }}
    >
      <label style={{ fontSize: 12, color: "var(--text-dim)" }}>開催日:</label>
      <select
        style={SEL}
        value={selectedDate}
        onChange={(e) => setSelectedDate(e.target.value)}
        disabled={loadingDates}
      >
        {loadingDates ? (
          <option>読み込み中…</option>
        ) : dates.length === 0 ? (
          <option>データなし</option>
        ) : (
          dates.map((d) => <option key={d} value={d}>{d}</option>)
        )}
      </select>

      <label style={{ fontSize: 12, color: "var(--text-dim)" }}>会場:</label>
      <select
        style={SEL}
        value={selectedVenue}
        onChange={(e) => onVenueChange(e.target.value)}
        disabled={loadingRaces || venues.length === 0}
      >
        {venues.length === 0 ? (
          <option>-</option>
        ) : (
          venues.map((v) => <option key={v} value={v}>{v}</option>)
        )}
      </select>

      <label style={{ fontSize: 12, color: "var(--text-dim)" }}>レース:</label>
      <select
        style={SEL}
        value={selectedRaceId}
        onChange={(e) => onRaceChange(e.target.value)}
        disabled={loadingRaces || filteredRaces.length === 0}
      >
        {filteredRaces.length === 0 ? (
          <option>-</option>
        ) : (
          filteredRaces.map((r) => (
            <option key={r.race_id} value={r.race_id}>
              {raceLabel(r)}
            </option>
          ))
        )}
      </select>

      {onAnalyze && (
        <button
          style={{ ...BTN, opacity: (!selectedRaceId || analyzing) ? 0.5 : 1, cursor: (!selectedRaceId || analyzing) ? "not-allowed" : "pointer" }}
          disabled={!selectedRaceId || analyzing}
          onClick={() => selectedRaceId && onAnalyze(selectedRaceId)}
        >
          {analyzing ? "分析中…" : analyzeLabel}
        </button>
      )}

      {extraControls}

      {statusMsg && (
        <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{statusMsg}</span>
      )}
    </div>
  );
}
