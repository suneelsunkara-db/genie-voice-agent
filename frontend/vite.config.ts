import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Locale bundles under src/locales are offline-generated build artifacts.
    // Ignore them from the file watcher so regenerating them (e.g. while the
    // dev server is up) doesn't invalidate the i18n glob and remount the app —
    // which would tear down an in-progress voice call. They're still bundled at
    // build time and lazy-loaded at runtime; a manual refresh picks up new ones.
    watch: { ignored: ["**/src/locales/**"] },
  },
});
