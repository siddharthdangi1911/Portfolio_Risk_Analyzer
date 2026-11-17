import type { Metadata } from "next";
import { Explorer } from "@/components/Explorer";
import { PageHead } from "@/components/ui";
import { FAM, site } from "@/lib/data";

export const metadata: Metadata = { title: "Forecast explorer" };

export default function ExplorerPage() {
  return (
    <>
      <PageHead eyebrow="Interactive" title="Forecast explorer">
        Pick a period, an ETF and a date window. The accuracy table recalculates for exactly that window, so you can see where the model helps and where it fails.
      </PageHead>
      <Explorer test={site.timeline} val={site.val_timeline} fam={FAM} />
    </>
  );
}
