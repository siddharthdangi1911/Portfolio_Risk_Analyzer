import type { ReactNode } from "react";
import { Icon } from "./Icons";

export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      {children && <p>{children}</p>}
    </header>
  );
}

export function Card({ title, sub, action, children, className = "" }: { title?: ReactNode; sub?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, sub, badge, tone = "neutral" }: { label: string; value: ReactNode; sub?: ReactNode; badge?: string; tone?: "good" | "neutral" | "accent" }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {badge && <div className={`badge ${tone}`}>{badge}</div>}
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function Callout({ children }: { children: ReactNode }) {
  return (
    <div className="callout">
      <div className="callout-icon">{Icon.bulb}</div>
      <p>{children}</p>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}>
          <i className="swatch" style={{ background: i.dashed ? `repeating-linear-gradient(90deg, ${i.color} 0 4px, transparent 4px 7px)` : i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
