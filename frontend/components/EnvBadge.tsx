import { envBadgeLabel } from "@/lib/envLabel";

// stg（検証環境）のときだけ、画面の左上に小さな帯を出す。prod は何も出さない。ビルド時に固定（next.config.js の NEXT_PUBLIC_ENV_LABEL）。
export function EnvBadge() {
  const label = envBadgeLabel(process.env.NEXT_PUBLIC_ENV_LABEL);
  if (!label) return null;
  return (
    <div
      role="note"
      aria-label={`${label}環境（検証用）`}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 2147483000,
        pointerEvents: "none",
        padding: "1px 8px",
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: "0.08em",
        color: "#1a1300",
        background: "#fbbf24",
        borderBottomRightRadius: 6,
        opacity: 0.92,
      }}
    >
      {label}
    </div>
  );
}
