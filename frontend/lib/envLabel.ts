// 画面端の環境バッジに出す文字。prod / 空は非表示（空文字を返す）。それ以外は大文字・最大 8 文字。
export function envBadgeLabel(raw: string | null | undefined): string {
  const v = (raw ?? "").trim().toUpperCase();
  if (!v || v === "PROD" || v === "PRODUCTION") return "";
  return v.slice(0, 8);
}
