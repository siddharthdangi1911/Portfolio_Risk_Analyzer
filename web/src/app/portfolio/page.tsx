import type { Metadata } from "next";
import { PortfolioLab } from "@/components/PortfolioLab";
import { PageHead } from "@/components/ui";
import { ASSETS, FAM, site } from "@/lib/data";

export const metadata: Metadata = { title: "Portfolio lab" };

export default function PortfolioPage() {
  return (
    <>
      <PageHead eyebrow="Interactive" title="Portfolio lab">
        Mix stocks, bonds and gold to see realised risk over the test period, and how the per-asset forecasts become a portfolio risk forecast.
      </PageHead>
      <PortfolioLab P={site.portfolio} assets={ASSETS} fam={FAM} />
    </>
  );
}
