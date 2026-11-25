import Link from "next/link";
import { featureEnabled } from "@/lib/features";

/** サンプル買い目・回収率の近くに添える 1 行（フッターと同じ趣旨） */
export const BET_NOTE = "サンプル買い目と回収率は、過去の集計と AI の予想による参考情報です。将来の結果を保証しません。";

/** 全ページの下の注意書き（docs/operations/wording_glossary.md の「注意書き」）。年齢と、予想が参考情報であることの 2 点だけを出す。 */
export function SiteFooter() {
  return (
    <footer
      aria-label="ご注意"
      style={{ marginTop: 32, padding: "14px 16px 24px", borderTop: "1px solid var(--border)", fontSize: 11.5, lineHeight: 1.7, color: "var(--text-dim)" }}
    >
      <p style={{ margin: 0 }}>馬券の購入は 20 歳以上の方に限られます（20 歳未満の方は購入できません）。</p>
      <p style={{ margin: 0 }}>当サイトの予想・サンプル買い目は参考情報で、的中や結果を保証するものではありません。</p>
      <p style={{ margin: "6px 0 0" }}>
        <Link href="/legal/terms" style={{ color: "inherit" }}>利用規約</Link>・
        <Link href="/legal/privacy" style={{ color: "inherit" }}>プライバシーポリシー</Link>
        {featureEnabled("billing") && (  // 特商法の表記は billing（申込み）と一緒に公開する（lib/features.ts）
          <>・<Link href="/legal/tokushoho" style={{ color: "inherit" }}>特定商取引法に基づく表記</Link></>
        )}
      </p>
    </footer>
  );
}
