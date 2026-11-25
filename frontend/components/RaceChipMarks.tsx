"use client";

import { GRADE_STYLE, gradeOf, isGraded, shortName, startTimeOf } from "@/lib/raceGrade";

type ChipRace = { race_name?: string | null; start_time?: string | null; grade?: string | null; list_grade_icon?: string | null };

/** 重賞のチップの上辺の色の帯（box-shadow の内側に描くので、枠の色＝参考度の色・大きさは変えない）。重賞でなければ undefined */
export function chipAccent(r: ChipRace): string | undefined {
  const g = gradeOf(r);
  return g && isGraded(g) ? `inset 0 3px 0 ${GRADE_STYLE[g].border}` : undefined;
}

/** チップの説明（ホバー）の頭: 「11R 15:40 ［G1］マイルCS」 */
export function chipTitle(round: number | string, r: ChipRace): string {
  const g = gradeOf(r);
  const tm = startTimeOf(r);
  return `${round}R ${tm ? `${tm} ` : ""}${g ? `［${g}］` : ""}${r.race_name ?? ""}`;
}

/** 番号の上の格の札（G1・G2・G3・L・OP）。色だけに頼らず文字も出す */
export function GradeBadge({ r }: { r: ChipRace }) {
  const g = gradeOf(r);
  if (!g) return null;
  const s = GRADE_STYLE[g];
  return (
    <span style={{ fontSize: 9.5, fontWeight: 800, lineHeight: 1.2, padding: "0 5px", borderRadius: 4, background: s.bg, color: s.fg, border: `1px solid ${s.border}`, letterSpacing: 0.2 }}>
      {g}
    </span>
  );
}

/** 番号の下の、重賞の略称と発走時刻（データが無ければ出さない） */
export function ChipFoot({ r, narrow }: { r: ChipRace; narrow: boolean }) {
  const g = gradeOf(r);
  const tm = startTimeOf(r);
  const nm = g && isGraded(g) && !narrow ? shortName(r.race_name, 7) : "";   // スマホでは略称を出さない（チップの高さ・幅を抑える。ホバー・長押しの説明に名前がある）
  return (
    <>
      {nm && <span style={{ fontSize: 10, fontWeight: 700, maxWidth: 84, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{nm}</span>}
      {tm && <span style={{ fontSize: 10, opacity: 0.8, fontVariantNumeric: "tabular-nums" }}>{tm}</span>}
    </>
  );
}
