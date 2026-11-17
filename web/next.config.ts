import type { NextConfig } from "next";

// Static export for GitHub Pages: `npm run export:docs` writes the site to ../docs.
// The repo is served under /Portfolio_Risk_Analyzer, so all links and assets use that base path.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "/Portfolio_Risk_Analyzer";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
