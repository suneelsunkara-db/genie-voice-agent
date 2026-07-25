// Emits the English UI-copy message catalog to src/locales/en.json — the source
// the offline translator (scripts/i18n/translate_locales.py) localizes into every
// other supported language. Run from the frontend/ dir via esbuild + node:
//
//   node_modules/.bin/esbuild scripts/extract_catalog.ts --bundle \
//     --platform=node --format=esm --outfile=/tmp/extract_catalog.mjs
//   node /tmp/extract_catalog.mjs
//
// (esbuild resolves the app's extensionless imports and strips types; the copy
// modules are guarded so importing them outside Vite is safe.)
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { EN_CATALOG } from "../src/i18n";

const outDir = resolve(process.cwd(), "src/locales");
mkdirSync(outDir, { recursive: true });
const outFile = resolve(outDir, "en.json");
writeFileSync(outFile, `${JSON.stringify(EN_CATALOG, null, 2)}\n`, "utf8");
console.log(`Wrote ${Object.keys(EN_CATALOG).length} keys -> ${outFile}`);
