"use client";

import { useEffect } from "react";
import { compareCells, isNumericColumn, parseCell } from "@/lib/tableSort";

/* アプリ内のすべての表を、見出しを押して列ごとに並べ替えられるようにする（昇順→降順→元の順）。
   表ごとの実装は要らない。除外したい表（独自にソートを持つ・並び順に意味がある）は <table data-no-sort> にする。
   並べ替えられない表: 本体の行に colSpan / rowSpan がある（詳細行を挟む表など）。見出しが複数段の表は、いちばん下の段が対象。
   値は表示文字から読む（%・倍・円・時間・範囲・印に対応）。セルに data-sort="…" を付ければ、その値を優先する。 */

const leafHeader = (t: HTMLTableElement): HTMLTableRowElement | null => {
  const rows = t.tHead ? Array.from(t.tHead.rows) : [];
  return rows.length ? rows[rows.length - 1] : null;
};

const bodyRows = (t: HTMLTableElement): HTMLTableRowElement[] => (t.tBodies[0] ? Array.from(t.tBodies[0].rows) : []);

const sortable = (t: HTMLTableElement): boolean => {
  if (t.hasAttribute("data-no-sort")) return false;
  const rows = bodyRows(t);
  if (rows.length < 2) return false;
  return rows.every((r) => Array.from(r.cells).every((c) => c.colSpan <= 1 && c.rowSpan <= 1));
};

const colIndex = (th: HTMLTableCellElement): number => {
  let i = 0;
  for (const c of Array.from((th.parentElement as HTMLTableRowElement).cells)) {
    if (c === th) return i;
    i += c.colSpan;
  }
  return i;
};

function apply(t: HTMLTableElement) {
  const col = t.dataset.sortCol, dir = t.dataset.sortDir;
  const body = t.tBodies[0];
  if (!body) return;
  const rows = bodyRows(t);
  rows.forEach((r, i) => { if (r.dataset.origI == null) r.dataset.origI = String(1000 + i); });      // 後から React が足した行は元の順の末尾側
  let ordered: HTMLTableRowElement[];
  if (col == null || !dir) {
    ordered = [...rows].sort((a, b) => Number(a.dataset.origI) - Number(b.dataset.origI));
  } else {
    const ci = Number(col), d = dir === "asc" ? 1 : -1;
    const cells = rows.map((r) => {
      const c = r.cells[ci];
      return parseCell(c?.textContent ?? "", c?.getAttribute("data-sort"));
    });
    const numeric = isNumericColumn(cells);
    const idx = new Map(rows.map((r, i) => [r, i]));
    ordered = [...rows].sort((a, b) => compareCells(cells[idx.get(a)!], cells[idx.get(b)!], numeric, d as 1 | -1) || Number(a.dataset.origI) - Number(b.dataset.origI));
  }
  if (ordered.some((r, i) => r !== rows[i])) ordered.forEach((r) => body.appendChild(r));
}

function mark(t: HTMLTableElement) {
  const head = leafHeader(t);
  if (!head) return;
  const ok = sortable(t);
  Array.from(head.cells).forEach((th, i) => {
    if (ok) {
      th.setAttribute("data-sortable", "1");
      const active = t.dataset.sortCol === String(colIndex(th)) && t.dataset.sortDir;
      if (active) th.setAttribute("data-sort-dir", t.dataset.sortDir!); else th.removeAttribute("data-sort-dir");
      th.setAttribute("aria-sort", active ? (t.dataset.sortDir === "asc" ? "ascending" : "descending") : "none");
      void i;
    } else if (th.hasAttribute("data-sortable")) {
      th.removeAttribute("data-sortable");
      th.removeAttribute("data-sort-dir");
    }
  });
}

export function TableSorter() {
  useEffect(() => {
    const onClick = (ev: MouseEvent) => {
      const th = (ev.target as HTMLElement | null)?.closest?.("th[data-sortable]") as HTMLTableCellElement | null;
      if (!th) return;
      if ((ev.target as HTMLElement).closest("a,button,input,select,label")) return;      // 見出しの中のリンク・ボタンは邪魔しない
      const t = th.closest("table") as HTMLTableElement | null;
      if (!t || !sortable(t) || th.parentElement !== leafHeader(t)) return;
      const ci = String(colIndex(th));
      if (t.dataset.sortCol !== ci || !t.dataset.sortDir) { t.dataset.sortCol = ci; t.dataset.sortDir = "asc"; }
      else if (t.dataset.sortDir === "asc") t.dataset.sortDir = "desc";
      else { delete t.dataset.sortCol; delete t.dataset.sortDir; }
      apply(t);
      mark(t);
    };
    document.addEventListener("click", onClick);

    let timer: ReturnType<typeof setTimeout> | null = null;
    const scan = () => {
      timer = null;
      document.querySelectorAll("table").forEach((el) => {
        const t = el as HTMLTableElement;
        mark(t);
        if (t.dataset.sortCol != null && t.dataset.sortDir) apply(t);      // データが更新されて行が入れ替わっても、選んだ並びを保つ
      });
    };
    const mo = new MutationObserver(() => { if (timer == null) timer = setTimeout(scan, 150); });
    mo.observe(document.body, { childList: true, subtree: true });
    scan();
    return () => { document.removeEventListener("click", onClick); mo.disconnect(); if (timer) clearTimeout(timer); };
  }, []);
  return null;
}
