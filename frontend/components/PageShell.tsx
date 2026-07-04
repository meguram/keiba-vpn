import { AnalysisNote } from "@/components/Nav";

type Props = {
  title: string;
  description?: string;
  children?: React.ReactNode;
};

export function PageShell({ title, description, children }: Props) {
  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-bold">{title}</h1>
        {description && <p style={{ color: "var(--text-dim)" }}>{description}</p>}
      </header>
      {children}
      <AnalysisNote />
    </div>
  );
}
