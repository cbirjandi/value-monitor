import { defineConfig } from "vite"

// Relative asset paths, so the build works at any GitHub Pages path (https://<you>.github.io/<repo>/).
export default defineConfig({
  base: "./",
  // The in-browser model library is large; it loads only when that mode is chosen.
  build: { chunkSizeWarningLimit: 7000 },
})
