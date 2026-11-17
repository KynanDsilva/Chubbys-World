import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/** @type {(phase: string) => import('next').NextConfig} */
const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://127.0.0.1:8000";

const nextConfig = (phase) => ({
  // Keep `next dev` and `next build` artifacts separate so builds never corrupt a live dev server.
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${PYTHON_API_URL}/api/:path*`,
      },
    ];
  },
});

export default nextConfig;
