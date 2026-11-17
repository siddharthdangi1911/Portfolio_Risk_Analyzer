"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Icon } from "./Icons";

export const PAGES = [
  { href: "/", label: "Overview", icon: Icon.overview },
  { href: "/data/", label: "Data", icon: Icon.data },
  { href: "/models/", label: "Model selection", icon: Icon.models },
  { href: "/results/", label: "Test results", icon: Icon.results },
  { href: "/explorer/", label: "Forecast explorer", icon: Icon.explorer },
  { href: "/errors/", label: "Errors & limits", icon: Icon.errors },
  { href: "/portfolio/", label: "Portfolio lab", icon: Icon.portfolio },
  { href: "/method/", label: "Method", icon: Icon.method },
];

const norm = (p: string) => (p.endsWith("/") ? p : p + "/");

function useActive() {
  const path = norm(usePathname() || "/");
  return (href: string) => norm(href) === path;
}

function NavLinks() {
  const isActive = useActive();
  return (
    <>
      {PAGES.map((p) => (
        <Link key={p.href} href={p.href} className={`nav-link${isActive(p.href) ? " active" : ""}`} aria-current={isActive(p.href) ? "page" : undefined}>
          {p.icon}
          {p.label}
        </Link>
      ))}
    </>
  );
}

function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);
  useEffect(() => {
    const t = document.documentElement.dataset.theme;
    setDark(t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);
  const flip = () => {
    const next = !dark;
    document.documentElement.dataset.theme = next ? "dark" : "light";
    try { localStorage.setItem("pra-theme", next ? "dark" : "light"); } catch { /* storage unavailable */ }
    setDark(next);
  };
  return (
    <button type="button" className="theme-toggle" onClick={flip} aria-label="Toggle colour theme">
      {dark ? Icon.sun : Icon.moon}
      {dark ? "Light mode" : "Dark mode"}
    </button>
  );
}

export function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Sections">
      <div className="logo">
        <div className="logo-mark">{Icon.logo}</div>
        <div>
          <div className="logo-title">Portfolio Risk Analyzer</div>
          <div className="logo-sub">ES 335 · Volatility forecasting</div>
        </div>
      </div>
      <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <NavLinks />
      </nav>
      <div className="sidebar-foot">
        <a className="foot-link" href="https://github.com/siddharthdangi1911/Portfolio_Risk_Analyzer">
          {Icon.github} Source on GitHub
        </a>
      </div>
    </aside>
  );
}

export function TopNav() {
  return (
    <nav className="topnav" aria-label="Sections">
      <NavLinks />
    </nav>
  );
}
