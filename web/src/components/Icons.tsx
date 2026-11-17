// Small stroke icons (24px grid, drawn at 18px).
import type { ReactNode } from "react";

const Svg = ({ children, size = 18 }: { children: ReactNode; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

export const Icon = {
  overview: <Svg><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></Svg>,
  data: <Svg><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></Svg>,
  models: <Svg><path d="M4 20V10" /><path d="M10 20V4" /><path d="M16 20v-7" /><path d="M22 20H2" /></Svg>,
  results: <Svg><path d="M9 12l2 2 4-4" /><circle cx="12" cy="12" r="9" /></Svg>,
  explorer: <Svg><path d="M3 17l5-6 4 4 4-7 5 9" /><path d="M3 21h18" /></Svg>,
  errors: <Svg><path d="M12 3l9.5 16.5H2.5L12 3z" /><path d="M12 10v4" /><path d="M12 17.5v.01" /></Svg>,
  portfolio: <Svg><path d="M12 3v9l7.8 4.5" /><circle cx="12" cy="12" r="9" /></Svg>,
  method: <Svg><path d="M8 6l-6 6 6 6" /><path d="M16 6l6 6-6 6" /></Svg>,
  sun: <Svg size={16}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Svg>,
  moon: <Svg size={16}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></Svg>,
  github: <Svg size={15}><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3.1-.3 6.4-1.5 6.4-7A5.4 5.4 0 0 0 20 4.8 5 5 0 0 0 19.9 1S18.7.7 16 2.5a13.4 13.4 0 0 0-7 0C6.3.7 5.1 1 5.1 1A5 5 0 0 0 5 4.8a5.4 5.4 0 0 0-1.5 3.7c0 5.4 3.3 6.6 6.4 7a3.4 3.4 0 0 0-.9 2.6V22" /></Svg>,
  bulb: <Svg><path d="M9 18h6" /><path d="M10 22h4" /><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z" /></Svg>,
  logo: <Svg><path d="M3 17l5-6 4 4 4-7 5 9" /></Svg>,
};
