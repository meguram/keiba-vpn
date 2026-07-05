"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_ENDPOINTS,
  CategorySection,
  DEV_CATEGORIES,
  NavCardItem,
  PUBLIC_CATEGORIES,
} from "@/components/home/homeData";

type GcsStatus = "green" | "orange" | "red";
type OddsTrainState = {
  visible: boolean;
  phase: string;
  pct: number;
  detail: string;
};

function NavCard({
  card,
  onFocusJump,
}: {
  card: NavCardItem;
  onFocusJump: () => void;
}) {
  const inner = (
    <>
      <span className="card-icon">{card.icon}</span>
      <div className="card-title">
        {card.title} <span className="arrow">→</span>
      </div>
      <div className="card-desc">{card.desc}</div>
      <div className="card-tags">
        {card.tags.map((tag) => (
          <span key={tag} className="tag">
            {tag}
          </span>
        ))}
      </div>
    </>
  );

  if (card.onClick === "focusJump") {
    return (
      <a
        href="#"
        className="nav-card"
        style={{ ["--card-accent" as string]: card.accent }}
        onClick={(e) => {
          e.preventDefault();
          onFocusJump();
        }}
      >
        {inner}
      </a>
    );
  }

  return (
    <Link href={card.href} className="nav-card" style={{ ["--card-accent" as string]: card.accent }}>
      {inner}
    </Link>
  );
}

function CategoryBlock({
  section,
  onFocusJump,
}: {
  section: CategorySection;
  onFocusJump: () => void;
}) {
  return (
    <div className="category" style={{ ["--cat-color" as string]: section.catColor }}>
      <div className="category-header">
        <span className="category-num" style={section.numStyle ? { background: section.numStyle } : undefined}>
          {section.num}
        </span>
        <span className="category-title">{section.title}</span>
        {section.badge && <span className="category-badge">{section.badge}</span>}
        <span className="category-desc">{section.desc}</span>
      </div>
      <div className={`nav-grid${section.singleRow ? " single-card-row" : ""}`}>
        {section.cards.map((card) => (
          <NavCard key={card.title} card={card} onFocusJump={onFocusJump} />
        ))}
      </div>
    </div>
  );
}

function weekdayLabel(dateStr: string) {
  const y = dateStr.slice(0, 4);
  const m = dateStr.slice(4, 6);
  const day = dateStr.slice(6, 8);
  const dt = new Date(+y, +m - 1, +day);
  const weekday = ["日", "月", "火", "水", "木", "金", "土"][dt.getDay()];
  const weekColor =
    dt.getDay() === 0 ? "var(--home-red)" : dt.getDay() === 6 ? "var(--home-accent)" : "var(--home-text-dim)";
  return { y, m, day, weekday, weekColor };
}

export function HomeDashboard() {
  const [isDev, setIsDev] = useState(false);
  const [gcsStatus, setGcsStatus] = useState<GcsStatus>("green");
  const [gcsLabel, setGcsLabel] = useState("GCS");
  const [raceInfo, setRaceInfo] = useState<string | null>(null);
  const [recentDates, setRecentDates] = useState<string[] | null>(null);
  const [jumpInput, setJumpInput] = useState("");
  const [oddsTrain, setOddsTrain] = useState<OddsTrainState>({
    visible: false,
    phase: "—",
    pct: 0,
    detail: "待機中",
  });
  const jumpRef = useRef<HTMLInputElement>(null);
  const oddsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const focusJump = useCallback(() => {
    if (jumpRef.current) {
      jumpRef.current.focus();
      jumpRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const v = window.prompt("レース ID（12桁）または日付（8桁）を入力");
    if (!v) return;
    const t = String(v).trim().replace(/[^0-9]/g, "");
    if (t.length === 8) window.location.href = `/monitor?date=${t}`;
    else if (t.length >= 10) window.location.href = `/race/${t}`;
  }, []);

  const jump = useCallback(() => {
    const v = jumpInput.trim();
    if (!v) return;
    if (v.length === 8 && /^\d{8}$/.test(v)) {
      window.location.href = `/monitor?date=${v}`;
    } else if (v.length >= 10 && /^\d+$/.test(v)) {
      window.location.href = `/race/${v}`;
    } else {
      window.location.href = `/monitor?date=${v.replace(/[^0-9]/g, "")}`;
    }
  }, [jumpInput]);

  const renderOddsTrainProgress = useCallback((d: Record<string, unknown>) => {
    const p = (d.progress as Record<string, unknown>) || {};
    const running = Boolean(d.running || p.running);
    if (!running && !p.phase) {
      setOddsTrain((prev) => ({ ...prev, visible: false }));
      return;
    }
    const pct = p.pct != null ? Number(p.pct) : 0;
    const parts: string[] = [];
    if (p.message) parts.push(String(p.message));
    if (p.total) parts.push(`${p.current || 0} / ${p.total}`);
    if (p.elapsed_sec != null) parts.push(`${p.elapsed_sec}s`);
    const extra = p.extra as Record<string, unknown> | undefined;
    if (extra?.n_rows) parts.push(`rows ${Number(extra.n_rows).toLocaleString()}`);
    if (d.error || p.error) parts.push(`⚠ ${d.error || p.error}`);
    if (!running && d.result) {
      const r = d.result as Record<string, unknown>;
      parts.push(`完了: train=${r.n_train_rows} eval=${r.n_eval_rows} ${r.training_time_sec}s`);
    }
    setOddsTrain({
      visible: true,
      phase: `${p.phase || (running ? "running" : "idle")}${pct ? ` ${pct}%` : ""}`,
      pct: Math.min(100, pct),
      detail: parts.join(" · ") || (running ? "学習中…" : "完了"),
    });
  }, []);

  const pollOddsTrainOnce = useCallback(async () => {
    try {
      const r = await fetch("/api/odds/train/status", { cache: "no-store" });
      const d = await r.json();
      renderOddsTrainProgress(d);
      return d;
    } catch {
      return null;
    }
  }, [renderOddsTrainProgress]);

  const startOddsTrain = useCallback(async () => {
    if (!confirm("想定オッズモデルの学習を開始しますか？（データセット構築に長時間かかります）")) return;
    try {
      const r = await fetch("/api/odds/train", { method: "POST" });
      const d = await r.json();
      if (d.status === "already_running") {
        alert("すでに学習が実行中です。下の進捗バーで状態を確認できます。");
      } else if (d.status !== "started") {
        alert(JSON.stringify(d));
        return;
      }
      setOddsTrain({ visible: true, phase: "init", pct: 0, detail: "開始…" });
      if (oddsPollRef.current) clearInterval(oddsPollRef.current);
      oddsPollRef.current = setInterval(pollOddsTrainOnce, 2000);
      pollOddsTrainOnce();
    } catch (e) {
      alert(`エラー: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [pollOddsTrainOnce]);

  const pollSimulation = useCallback(async () => {
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 5000));
      try {
        const r = await fetch("/api/simulation/status");
        const d = await r.json();
        if (!d.running) {
          if (d.result) {
            alert(
              `最適化完了！\nROI: ${d.result.roi}\n的中率: ${(d.result.hit_rate * 100).toFixed(1)}%\nα: ${d.result.prob_weight}\n対象: ${d.result.n_races}レース`,
            );
          } else if (d.error) {
            alert(`シミュレーション失敗: ${d.error}`);
          }
          return;
        }
      } catch {
        /* retry */
      }
    }
  }, []);

  const runSimulation = useCallback(async () => {
    if (!confirm("バックテストシミュレーションを開始しますか？（数分かかる場合があります）")) return;
    try {
      const r = await fetch("/api/simulation/run", { method: "POST" });
      const d = await r.json();
      if (d.status === "started") {
        alert("シミュレーション開始しました。完了後は /api/simulation/params で結果を確認できます。");
        pollSimulation();
      } else {
        alert(JSON.stringify(d));
      }
    } catch (e) {
      alert(`エラー: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [pollSimulation]);

  useEffect(() => {
    fetch("/api/v1/auth/status", { credentials: "include" })
      .then((r) => (r.ok ? r.json() : { logged_in: false, is_developer: false }))
      .then((d) => setIsDev(!!d.logged_in && !!d.is_developer))
      .catch(() => setIsDev(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/scrape-dates", { cache: "no-store" });
        const data = await res.json();
        if (cancelled) return;
        const dates: string[] = data.dates || [];
        if (data.gcs_enabled !== undefined) {
          if (data.gcs_enabled) {
            setGcsStatus("green");
            setGcsLabel("GCS 接続中");
          } else {
            setGcsStatus("orange");
            setGcsLabel("GCS 未接続");
          }
        }
        if (dates.length > 0) {
          setRaceInfo(`${dates.length}日分のデータ`);
          if (isDev) setRecentDates(dates.slice(0, 8));
        } else if (isDev) {
          setRecentDates([]);
        }
      } catch {
        if (!cancelled) {
          setGcsStatus("red");
          setGcsLabel("GCS ?");
          if (isDev) setRecentDates([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isDev]);

  useEffect(() => {
    if (!isDev) return;
    pollOddsTrainOnce();
    return () => {
      if (oddsPollRef.current) clearInterval(oddsPollRef.current);
    };
  }, [isDev, pollOddsTrainOnce]);

  const handleApiAction = (action: string) => {
    if (action === "startOddsTrain") startOddsTrain();
    else if (action === "pollOddsTrainOnce") pollOddsTrainOnce();
    else if (action === "runSimulation") runSimulation();
  };

  return (
    <div className="home-root">
      <div className="hero">
        <h1>
          <span className="icon-wrap">
            <span className="icon-glow" />
            <span className="icon-ring" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/data/image/icon.jpg" alt="Meguro" className="icon" />
          </span>{" "}
          ML-<span className="accent">AutoPilot</span> Keiba
        </h1>
        <p className="subtitle">MULTI-AGENT HORSE RACING AI PREDICTION SYSTEM</p>
        <div className="status-row">
          <span className="status-chip">
            <span className={`dot ${gcsStatus}`} /> <span>{gcsLabel}</span>
          </span>
          <span className="status-chip">
            <span className="dot green" /> <span>Server OK</span>
          </span>
          {raceInfo && (
            <span className="status-chip">
              <span className="dot green" /> <span>{raceInfo}</span>
            </span>
          )}
        </div>
      </div>

      <div className="main">
        {isDev && (
          <div className="quick-jump">
            <h3>🔍 レースID / 日付で直接移動</h3>
            <div className="jump-form">
              <input
                ref={jumpRef}
                type="text"
                id="jumpInput"
                placeholder="例: 202606020609 or 20260315"
                value={jumpInput}
                onChange={(e) => setJumpInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") jump();
                }}
              />
              <button type="button" onClick={jump}>
                移動
              </button>
              <span className="jump-hint">レースID (12桁) → レース詳細 / 日付 (8桁) → モニターボード</span>
            </div>
          </div>
        )}

        {PUBLIC_CATEGORIES.map((section) => (
          <CategoryBlock key={section.num} section={section} onFocusJump={focusJump} />
        ))}

        {isDev && (
          <>
            {DEV_CATEGORIES.map((section) => (
              <CategoryBlock key={section.num} section={section} onFocusJump={focusJump} />
            ))}

            <div className="category">
              <div className="category-header">
                <span className="category-num" style={{ background: "var(--home-accent)" }}>
                  📅
                </span>
                <span className="category-title">最近のレース日</span>
                <span className="category-desc">モニターボードでスクレイピング進捗を確認</span>
              </div>
              <div className="recent-dates-grid">
                {recentDates === null && (
                  <div style={{ color: "var(--home-text-dim)", fontSize: 13, padding: 16 }}>読み込み中...</div>
                )}
                {recentDates !== null && recentDates.length === 0 && (
                  <div style={{ color: "var(--home-text-dim)", fontSize: 13, padding: 16 }}>
                    レース日データがまだありません
                  </div>
                )}
                {recentDates?.map((d) => {
                  const { y, m, day, weekday, weekColor } = weekdayLabel(d);
                  return (
                    <Link key={d} className="recent-date-card" href={`/monitor?date=${d}`}>
                      <div className="rd-date">
                        📅 {y}/{m}/{day} <span style={{ color: weekColor }}>({weekday})</span>
                      </div>
                      <div className="rd-hint">モニターで開く</div>
                    </Link>
                  );
                })}
              </div>
            </div>

            <div className="category">
              <div className="category-header">
                <span className="category-num" style={{ background: "var(--home-text-dim)" }}>
                  ⚙️
                </span>
                <span className="category-title">API エンドポイント</span>
                <span className="category-desc">代表的なバックエンド API への直接アクセス</span>
              </div>
              <div className="api-section">
                <div className="api-grid">
                  {API_ENDPOINTS.map((item) =>
                    item.external ? (
                      <a
                        key={item.path}
                        className="api-item"
                        href={item.href}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <span className={`api-method ${item.method.toLowerCase()}`}>{item.method}</span>
                        <span className="api-path">{item.path}</span>
                        <span className="api-desc">{item.desc}</span>
                      </a>
                    ) : (
                      <button
                        key={item.path}
                        type="button"
                        className="api-item"
                        onClick={() => item.action && handleApiAction(item.action)}
                      >
                        <span className={`api-method ${item.method.toLowerCase()}`}>{item.method}</span>
                        <span className="api-path">{item.path}</span>
                        <span className="api-desc">{item.desc}</span>
                      </button>
                    ),
                  )}
                </div>
                <div className={`odds-train-panel${oddsTrain.visible ? " visible" : ""}`}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <strong style={{ fontSize: 14 }}>想定オッズモデル学習</strong>
                    <span style={{ fontSize: 12, color: "var(--home-text-dim)" }}>{oddsTrain.phase}</span>
                  </div>
                  <div className="odds-train-bar-wrap">
                    <div className="odds-train-bar" style={{ width: `${oddsTrain.pct}%` }} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--home-text-dim)", lineHeight: 1.5 }}>{oddsTrain.detail}</div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <footer>
        ML-AutoPilot Keiba &mdash; Multi-Agent AI Prediction System &mdash; データソース:{" "}
        <a href="https://db.netkeiba.com/" target="_blank" rel="noreferrer">
          netkeiba.com
        </a>
      </footer>
    </div>
  );
}
