import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Sidebar, TopNav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Volatility Forecast Dashboard · Portfolio Risk Analyzer", template: "%s · Portfolio Risk Analyzer" },
  description:
    "ES 335 project: can machine learning forecast 20-day volatility of SPY, QQQ, TLT and GLD better than simple historical baselines, and what does it mean for portfolio risk?",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1 };

// Applies the saved theme before first paint (avoids a light/dark flash).
const themeScript = `try{var t=localStorage.getItem("pra-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <div className="shell">
          <Sidebar />
          <div>
            <TopNav />
            <main className="main">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
