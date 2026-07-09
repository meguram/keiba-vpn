"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ENTRIES_RAW from "./entries.json";
import { useAuthStatus } from "@/lib/hooks/useAuthStatus";

/* ── 型 ── */
type Entry = {
  id: string;
  name: string;
  cat: string;
  badgeColor: string;
  badgeLabel: string;
  nkUrl: string;
  content: string;
  isCustom?: boolean;
};

const BASE_ENTRIES: Entry[] = ENTRIES_RAW as Entry[];

/* ── ユーザーに表示するカテゴリ ── */
const USER_VISIBLE_CATS = new Set(["種牡馬", "牝系"]);

/* ── カテゴリ表示順 ── */
const CAT_ORDER = ["各国の血統特徴", "特定の海外レース", "大枠分類", "系統組み合わせ", "種牡馬", "牝系"];

/* ── テキスト整形コンポーネント ── */
function ContentLines({ text }: { text: string }) {
  if (!text?.trim()) return <p style={{ color: "var(--text-dim)", fontStyle: "italic" }}>（記載なし）</p>;
  return (
    <>
      {text.split("\n").map((line, i) => {
        const t = line.trim();
        if (!t) return <div key={i} style={{ height: 6 }} />;
        if (/^【.+】/.test(t))
          return <p key={i} style={{ fontWeight: 700, color: "#fff", marginTop: 10, marginBottom: 3 }}>{t}</p>;
        if (/^[❶❷❸❹❺❻❼❽❾]/.test(t)) return <p key={i}>{t}</p>;
        if (t.startsWith("★")) return <p key={i} style={{ color: "var(--err)", fontWeight: 600 }}>{t}</p>;
        if (t.startsWith("・") || /^- /.test(t))
          return <p key={i} style={{ paddingLeft: 14 }}>{t}</p>;
        if (/^->/.test(t))
          return <p key={i} style={{ paddingLeft: 14, color: "var(--text-dim)" }}>{t}</p>;
        if (/^\d+\.\s/.test(t)) return <p key={i} style={{ paddingLeft: 14 }}>{t}</p>;
        return <p key={i}>{t}</p>;
      })}
    </>
  );
}

/* ── localStorage ── */
const LS_PREFIX = "ped_edit_";
const LS_CUSTOM = "ped_custom_entries";
const LS_DEV = "stallion_dev_mode";

function loadEdits(): Record<string, string> {
  try {
    const out: Record<string, string> = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)!;
      if (k.startsWith(LS_PREFIX)) out[k.slice(LS_PREFIX.length)] = localStorage.getItem(k)!;
    }
    return out;
  } catch { return {}; }
}

function loadCustomEntries(): Entry[] {
  try {
    const raw = localStorage.getItem(LS_CUSTOM);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

/* ── サーバー API ── */
async function fetchServerOverrides(): Promise<Record<string, string>> {
  try {
    const res = await fetch("/api/v1/stallion-notes/overrides");
    if (res.ok) return await res.json();
  } catch {}
  return {};
}

async function saveServerOverride(id: string, content: string): Promise<boolean> {
  try {
    const res = await fetch("/api/v1/stallion-notes/overrides", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ id, content }),
    });
    return res.ok;
  } catch { return false; }
}

async function fetchServerCustom(): Promise<Entry[]> {
  try {
    const res = await fetch("/api/v1/stallion-notes/custom");
    if (res.ok) return await res.json();
  } catch {}
  return [];
}

async function saveServerCustom(entry: Omit<Entry, "isCustom">): Promise<boolean> {
  try {
    const res = await fetch("/api/v1/stallion-notes/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(entry),
    });
    return res.ok;
  } catch { return false; }
}

async function deleteServerCustom(id: string): Promise<boolean> {
  try {
    const res = await fetch(`/api/v1/stallion-notes/custom/${encodeURIComponent(id)}`, {
      method: "DELETE",
      credentials: "include",
    });
    return res.ok;
  } catch { return false; }
}

/* ── 追加モーダル（開発者のみ） ── */
function AddModal({
  defaultCat, cats, onAdd, onClose,
}: {
  defaultCat: string;
  cats: string[];
  onAdd: (e: Omit<Entry, "isCustom">) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [cat, setCat] = useState(defaultCat);
  const [content, setContent] = useState("");
  const [nkUrl, setNkUrl] = useState("");
  const [err, setErr] = useState("");

  function submit() {
    if (!name.trim()) { setErr("名前を入力してください"); return; }
    onAdd({ id: name.trim(), name: name.trim(), cat, badgeColor: "#2e7d32", badgeLabel: cat, nkUrl, content });
  }

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "8px 12px",
    background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text)", fontSize: 13, outline: "none",
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.6)", zIndex: 200, display: "flex", alignItems: "center", justifyContent: "center" }}
      onClick={onClose}>
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "28px 28px 22px", width: 480, maxWidth: "95vw", boxShadow: "0 8px 32px rgba(0,0,0,.4)" }}
        onClick={e => e.stopPropagation()}>
        <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20, color: "#fff" }}>エントリを追加</h3>
        {[
          { label: "名前", node: <input style={inputStyle} value={name} onChange={e => setName(e.target.value)} placeholder="種牡馬名 / 牝系名 など" /> },
          { label: "カテゴリ", node: <select style={inputStyle} value={cat} onChange={e => setCat(e.target.value)}>{cats.map(c => <option key={c}>{c}</option>)}</select> },
          { label: "netkeiba URL（任意）", node: <input style={inputStyle} value={nkUrl} onChange={e => setNkUrl(e.target.value)} placeholder="https://db.netkeiba.com/horse/ped/..." /> },
          { label: "メモ内容", node: <textarea style={{ ...inputStyle, minHeight: 140, resize: "vertical" }} value={content} onChange={e => setContent(e.target.value)} /> },
        ].map(({ label, node }) => (
          <div key={label} style={{ marginBottom: 14 }}>
            <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", marginBottom: 5 }}>{label}</label>
            {node}
          </div>
        ))}
        {err && <p style={{ color: "var(--err)", fontSize: 12, marginBottom: 8 }}>{err}</p>}
        <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
          <button onClick={submit} style={{ padding: "7px 18px", background: "var(--accent)", color: "#fff", border: "none", borderRadius: 5, fontSize: 13, fontWeight: 600, cursor: "pointer" }}>追加</button>
          <button onClick={onClose} style={{ padding: "7px 14px", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", borderRadius: 5, fontSize: 13, cursor: "pointer" }}>キャンセル</button>
        </div>
      </div>
    </div>
  );
}

/* ── メインコンポーネント ── */
export default function StallionNotesPage() {
  const { isAdmin } = useAuthStatus();
  const [entries, setEntries] = useState<Entry[]>(BASE_ENTRIES);
  const [localEdits, setLocalEdits] = useState<Record<string, string>>({});
  const [isDev, setIsDev] = useState(false);
  const [expandedCats, setExpandedCats] = useState<Set<string>>(new Set(["種牡馬"]));
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [addModalCat, setAddModalCat] = useState("種牡馬");

  /* 初期化 */
  useEffect(() => {
    const localEditsMap = loadEdits();
    setLocalEdits(localEditsMap);
    const custom = loadCustomEntries();
    if (custom.length > 0) setEntries([...BASE_ENTRIES, ...custom.map(e => ({ ...e, isCustom: true }))]);

    /* 開発者モード: URLに?dev=1 or sessionStorageにフラグ */
    const params = new URLSearchParams(window.location.search);
    if (params.get("dev") === "1") {
      try { sessionStorage.setItem(LS_DEV, "1"); } catch {}
    }
    try {
      setIsDev(sessionStorage.getItem(LS_DEV) === "1");
    } catch {}

    /* サーバー保存済み上書きをマージ（サーバー側が優先） */
    fetchServerOverrides().then(serverOverrides => {
      if (Object.keys(serverOverrides).length > 0) {
        setLocalEdits(prev => ({ ...prev, ...serverOverrides }));
        try {
          for (const [id, content] of Object.entries(serverOverrides)) {
            localStorage.setItem(LS_PREFIX + id, content);
          }
        } catch {}
      }
    });

    /* サーバーからカスタムエントリをロード（localStorage より優先） */
    fetchServerCustom().then(serverCustom => {
      if (serverCustom.length > 0) {
        setEntries(prev => {
          const serverIds = new Set(serverCustom.map((e: Entry) => e.id));
          const localOnly = prev.filter(e => e.isCustom && !serverIds.has(e.id));
          return [...BASE_ENTRIES, ...serverCustom.map((e: Entry) => ({ ...e, isCustom: true as const })), ...localOnly];
        });
        try { localStorage.setItem(LS_CUSTOM, JSON.stringify(serverCustom)); } catch {}
      }
    });
  }, []);

  /* isAdmin になった瞬間も編集モードを有効化 */
  const canEdit = isDev || isAdmin;

  /* カテゴリごとにグループ化 */
  const catMap = entries.reduce<Record<string, Entry[]>>((acc, e) => {
    if (!acc[e.cat]) acc[e.cat] = [];
    acc[e.cat].push(e);
    return acc;
  }, {});
  const allCats = [...CAT_ORDER.filter(c => catMap[c]), ...Object.keys(catMap).filter(c => !CAT_ORDER.includes(c))];

  /* 表示するカテゴリ */
  const visibleCats = canEdit ? allCats : allCats.filter(c => USER_VISIBLE_CATS.has(c));

  /* コンテンツ取得 */
  function getContent(id: string): string {
    return localEdits[id] ?? entries.find(e => e.id === id)?.content ?? "";
  }

  /* エントリ選択 */
  const selectEntry = useCallback((id: string) => {
    setSelectedId(id);
    setEditingId(null);
    setSearchQuery("");
    setShowSearch(false);
  }, []);

  /* カテゴリ開閉 */
  function toggleCat(cat: string) {
    setExpandedCats(prev => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });
  }

  const [savingId, setSavingId] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  /* 編集保存 */
  async function saveEdit() {
    if (!editingId) return;
    const newEdits = { ...localEdits, [editingId]: editText };
    setLocalEdits(newEdits);
    try { localStorage.setItem(LS_PREFIX + editingId, editText); } catch {}
    setEditingId(null);

    // サーバーへ保存
    setSavingId(editingId);
    setSaveError(null);
    const ok = await saveServerOverride(editingId, editText);
    setSavingId(null);
    if (!ok) setSaveError("サーバー保存に失敗しました（ローカルには保存済み）");
  }

  /* 編集リセット */
  function resetEdit(id: string) {
    const { [id]: _, ...rest } = localEdits;
    setLocalEdits(rest);
    try { localStorage.removeItem(LS_PREFIX + id); } catch {}
    if (editingId === id) setEditingId(null);
  }

  /* カスタムエントリ追加 */
  const [addingEntry, setAddingEntry] = useState(false);

  async function addEntry(e: Omit<Entry, "isCustom">) {
    const newEntry: Entry = { ...e, isCustom: true };
    const newEntries = [...entries, newEntry];
    setEntries(newEntries);
    try { localStorage.setItem(LS_CUSTOM, JSON.stringify(newEntries.filter(x => x.isCustom))); } catch {}
    setExpandedCats(prev => new Set([...prev, e.cat]));
    selectEntry(e.id);
    setShowAddModal(false);
    // サーバーへ保存
    setAddingEntry(true);
    await saveServerCustom(e);
    setAddingEntry(false);
  }

  /* カスタムエントリ削除 */
  async function deleteEntry(id: string) {
    const newEntries = entries.filter(e => e.id !== id);
    setEntries(newEntries);
    try { localStorage.setItem(LS_CUSTOM, JSON.stringify(newEntries.filter(x => x.isCustom))); } catch {}
    if (selectedId === id) setSelectedId(null);
    await deleteServerCustom(id);
  }

  /* 検索（表示カテゴリ内のみ） */
  const visibleCatSet = new Set(visibleCats);
  const searchResults = searchQuery.trim()
    ? entries
        .filter(e => visibleCatSet.has(e.cat))
        .filter(e => e.name.toLowerCase().includes(searchQuery.toLowerCase()))
        .slice(0, 20)
    : [];

  const selectedEntry = selectedId ? entries.find(e => e.id === selectedId) ?? null : null;

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)", color: "var(--text)", overflow: "hidden" }}>

      {/* ページヘッダー */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "12px 20px", display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
        <div>
          <h1 style={{ fontSize: 16, fontWeight: 700, color: "#fff" }}>🐴 種牡馬メモ</h1>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>血統ドメイン知識ベース — サイドバーから馬名を選択</span>
        </div>

        {canEdit && (
          <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 4, background: "rgba(245,158,11,0.15)", color: "var(--warn)", border: "1px solid rgba(245,158,11,0.3)", fontWeight: 700 }}>
            DEV MODE
          </span>
        )}

        {/* 検索 */}
        <div style={{ marginLeft: "auto", position: "relative" }}>
          <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-dim)", fontSize: 13, pointerEvents: "none" }}>🔍</span>
          <input
            type="text"
            placeholder="馬名を検索..."
            value={searchQuery}
            onChange={e => { setSearchQuery(e.target.value); setShowSearch(true); }}
            onFocus={() => setShowSearch(true)}
            onBlur={() => setTimeout(() => setShowSearch(false), 200)}
            style={{ width: 200, padding: "7px 12px 7px 32px", background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)", borderRadius: 20, color: "var(--text)", fontSize: 13, outline: "none" }}
          />
          {showSearch && searchResults.length > 0 && (
            <div style={{ position: "absolute", top: "calc(100% + 6px)", right: 0, width: 300, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 4px 16px rgba(0,0,0,.4)", zIndex: 100, maxHeight: 360, overflowY: "auto" }}>
              <div style={{ padding: "8px 14px", fontSize: 11, color: "var(--text-dim)", borderBottom: "1px solid var(--border)" }}>{searchResults.length}件</div>
              {searchResults.map(e => (
                <button key={e.id} onMouseDown={() => selectEntry(e.id)}
                  style={{ display: "block", width: "100%", padding: "8px 14px", background: "none", border: "none", color: "var(--text)", textAlign: "left", cursor: "pointer", fontSize: 13 }}>
                  {e.name}
                  <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}>{e.cat}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ボディ */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* サイドバー */}
        <nav style={{ width: 240, background: "var(--surface)", borderRight: "1px solid var(--border)", overflowY: "auto", flexShrink: 0, paddingBottom: 16 }}>
          {/* 種牡馬/牝系 追加ボタン */}
          {canEdit && (
            <div style={{ padding: "10px 12px", borderBottom: "1px solid rgba(36,48,73,0.5)", display: "flex", gap: 6 }}>
              <button
                onClick={() => { setAddModalCat("種牡馬"); setShowAddModal(true); }}
                style={{ flex: 1, padding: "5px 0", background: "rgba(46,125,50,0.12)", border: "1px solid rgba(46,125,50,0.35)", borderRadius: 5, color: "#4caf50", fontSize: 11.5, fontWeight: 600, cursor: "pointer" }}
              >＋ 種牡馬</button>
              <button
                onClick={() => { setAddModalCat("牝系"); setShowAddModal(true); }}
                style={{ flex: 1, padding: "5px 0", background: "rgba(156,39,176,0.1)", border: "1px solid rgba(156,39,176,0.35)", borderRadius: 5, color: "#ce93d8", fontSize: 11.5, fontWeight: 600, cursor: "pointer" }}
              >＋ 牝系</button>
            </div>
          )}
          {visibleCats.map(cat => {
                  const items = (catMap[cat] ?? []).filter(e => canEdit || !e.isCustom || USER_VISIBLE_CATS.has(e.cat));
            const expanded = expandedCats.has(cat);
            return (
              <div key={cat} style={{ borderBottom: "1px solid rgba(36,48,73,0.5)" }}>
                <div style={{ display: "flex", alignItems: "stretch" }}>
                  <button onClick={() => toggleCat(cat)}
                    style={{ flex: 1, display: "flex", alignItems: "center", gap: 6, padding: "9px 14px", background: "none", border: "none", cursor: "pointer", color: "var(--text)", textAlign: "left", fontSize: 12.5, fontWeight: 600 }}>
                    <span style={{ flex: 1 }}>{cat}</span>
                    <span style={{ fontSize: 10, color: "var(--text-dim)", background: "rgba(107,125,149,0.15)", borderRadius: 10, padding: "1px 7px" }}>{items.length}</span>
                    <span style={{ fontSize: 10, color: "var(--text-dim)", display: "inline-block", transform: expanded ? "rotate(180deg)" : "none", transition: "transform .2s" }}>▾</span>
                  </button>
                  {canEdit && (
                    <button onClick={() => { setAddModalCat(cat); setShowAddModal(true); }}
                      style={{ width: 28, background: "none", border: "none", borderLeft: "1px solid rgba(36,48,73,0.5)", cursor: "pointer", color: "var(--text-dim)", fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center" }}
                      title={`${cat}に追加`}>+</button>
                  )}
                </div>
                {expanded && (
                  <ul style={{ listStyle: "none", display: "block" }}>
                    {items.map(e => (
                      <li key={e.id}>
                        <button onClick={() => selectEntry(e.id)}
                          style={{
                            display: "flex", alignItems: "center", gap: 4, width: "100%",
                            padding: "6px 14px 6px 24px",
                            background: selectedId === e.id ? "rgba(59,130,246,0.12)" : "none",
                            border: "none",
                            color: selectedId === e.id ? "var(--accent)" : "var(--text)",
                            textAlign: "left", cursor: "pointer", fontSize: 13,
                            fontWeight: selectedId === e.id ? 600 : 400,
                            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                          }}>
                          <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>{e.name || "（無題）"}</span>
                          {canEdit && localEdits[e.id] !== undefined && (
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", flexShrink: 0 }} title="ローカル編集済み" />
                          )}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </nav>

        {/* コンテンツエリア */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px", maxWidth: 860 }}>
          {!selectedEntry && (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text-dim)" }}>
              <div style={{ fontSize: 40, marginBottom: 16 }}>🐴</div>
              <p style={{ fontSize: 14 }}>左サイドバーのカテゴリを開き、馬名をクリックするとメモが表示されます</p>
            </div>
          )}

          {selectedEntry && (
            <>
              {/* パンくず */}
              <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 14, display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
                <span>🏠</span><span style={{ color: "var(--text-dim)" }}>›</span>
                <span>{selectedEntry.cat}</span><span style={{ color: "var(--text-dim)" }}>›</span>
                <span style={{ color: "var(--text)", fontWeight: 600 }}>{selectedEntry.name}</span>
                {canEdit && localEdits[selectedEntry.id] !== undefined && (
                  <span style={{ fontSize: 10, color: "#22c55e", background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 10, padding: "1px 8px", marginLeft: 4 }}>✏ 編集済み</span>
                )}
                {canEdit && selectedEntry.isCustom && (
                  <span style={{ fontSize: 10, color: "var(--warn)", background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 10, padding: "1px 7px", marginLeft: 4 }}>カスタム</span>
                )}
              </div>

              {/* カード */}
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "22px 26px", boxShadow: "0 1px 4px rgba(0,0,0,.2)" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 12 }}>
                  <span style={{ fontSize: 22, fontWeight: 700, color: "#fff", lineHeight: 1.3 }}>{selectedEntry.name || "（無題）"}</span>
                  {canEdit && editingId !== selectedEntry.id && (
                    <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                      <button
                        onClick={() => { setEditingId(selectedEntry.id); setEditText(getContent(selectedEntry.id)); }}
                        style={{ padding: "5px 13px", background: "rgba(107,125,149,0.1)", border: "1px solid var(--border)", borderRadius: 5, fontSize: 12, cursor: "pointer", color: "var(--text-dim)" }}>
                        ✏️ 編集
                      </button>
                      {selectedEntry.isCustom && (
                        <button
                          onClick={() => { if (confirm(`「${selectedEntry.name}」を削除しますか？`)) deleteEntry(selectedEntry.id); }}
                          style={{ padding: "5px 13px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 5, fontSize: 12, cursor: "pointer", color: "var(--err)" }}>
                          削除
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* バッジ */}
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
                  <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 4, fontSize: 11.5, fontWeight: 600, color: "#fff", background: selectedEntry.badgeColor || "#2e7d32" }}>
                    {selectedEntry.badgeLabel}
                  </span>
                  {selectedEntry.nkUrl && (
                    <a href={selectedEntry.nkUrl} target="_blank" rel="noopener noreferrer"
                      style={{ display: "inline-block", padding: "4px 12px", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 5, fontSize: 12, color: "var(--accent)", textDecoration: "none" }}>
                      netkeiba 血統ページ
                    </a>
                  )}
                </div>

                <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "0 0 14px" }} />

                {/* コンテンツ */}
                {canEdit && editingId === selectedEntry.id ? (
                  <div>
                    <textarea
                      value={editText}
                      onChange={e => setEditText(e.target.value)}
                      style={{ width: "100%", minHeight: 280, padding: "12px 14px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontSize: 13, lineHeight: 1.75, resize: "vertical", outline: "none" }}
                    />
                    <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
                      <button onClick={saveEdit} disabled={savingId === selectedEntry.id}
                        style={{ padding: "7px 18px", background: "var(--accent)", color: "#fff", border: "none", borderRadius: 5, fontSize: 13, fontWeight: 600, cursor: savingId ? "default" : "pointer", opacity: savingId ? 0.7 : 1 }}>
                        {savingId === selectedEntry.id ? "保存中..." : "保存"}
                      </button>
                      <button onClick={() => setEditingId(null)} style={{ padding: "7px 14px", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", borderRadius: 5, fontSize: 13, cursor: "pointer" }}>キャンセル</button>
                      {localEdits[selectedEntry.id] !== undefined && (
                        <button onClick={() => resetEdit(selectedEntry.id)}
                          style={{ marginLeft: "auto", padding: "7px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", color: "var(--err)", borderRadius: 5, fontSize: 13, cursor: "pointer" }}>
                          編集をリセット
                        </button>
                      )}
                    </div>
                    {saveError && <p style={{ fontSize: 12, color: "var(--err)", marginTop: 6 }}>{saveError}</p>}
                  </div>
                ) : (
                  <div style={{ fontSize: 13.5, lineHeight: 1.85, color: "#c8cdd6" }}>
                    <ContentLines text={getContent(selectedEntry.id)} />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* 追加モーダル */}
      {canEdit && showAddModal && (
        <AddModal
          defaultCat={addModalCat}
          cats={["種牡馬", "牝系", ...allCats.filter(c => c !== "種牡馬" && c !== "牝系")]}
          onAdd={addEntry}
          onClose={() => setShowAddModal(false)}
        />
      )}
    </div>
  );
}
