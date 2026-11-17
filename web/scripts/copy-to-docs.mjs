// Copies the static export (web/out) to ../docs, which GitHub Pages serves.
import { cpSync, existsSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(web, "out");
const docs = join(web, "..", "docs");
if (!existsSync(out)) throw new Error("web/out not found - run `next build` first");
rmSync(docs, { recursive: true, force: true });
cpSync(out, docs, { recursive: true });
// GitHub Pages runs Jekyll by default, which drops folders starting with "_" (like _next).
writeFileSync(join(docs, ".nojekyll"), "");
console.log("Static site copied to docs/");
