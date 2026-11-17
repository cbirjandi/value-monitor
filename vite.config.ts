import { execSync } from "node:child_process"
import { defineConfig } from "vite"

/** The commit this build came from, recorded in exported runs so results can be traced to the exact code. */
function buildId() {
  if (process.env.GITHUB_SHA) return process.env.GITHUB_SHA.slice(0, 12)
  try {
    return execSync("git rev-parse --short=12 HEAD").toString().trim()
  } catch {
    return "dev"
  }
}

// Relative asset paths, so the build works at any GitHub Pages path (https://<you>.github.io/<repo>/).
export default defineConfig({
  base: "./",
  define: { __BUILD__: JSON.stringify(buildId()) },
  // The in-browser model library is large; it loads only when that mode is chosen.
  build: { chunkSizeWarningLimit: 7000 },
})
