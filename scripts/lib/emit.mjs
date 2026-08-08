/**
 * Write-or-verify for generated files.
 *
 * Every generator in `scripts/` has the same two modes: write the files, or (in
 * CI) assert the committed copies already match and fail loudly if they don't.
 * That check is the whole point — a generated artifact nobody verifies is just a
 * copy that silently rots. This is that logic, once.
 */

import fs from "node:fs";
import path from "node:path";

/**
 * @param {string} rootDir       Directory the relative paths are resolved against.
 * @param {Record<string,string>} files  Relative path → exact file content.
 * @param {{check?: boolean, label?: string, fixHint?: string}} opts
 * @returns {number} count of files that were written (or that drifted, in check mode).
 */
export function emitFiles(rootDir, files, opts = {}) {
  const { check = false, label = rootDir, fixHint = "regenerate and commit the result" } = opts;
  let changed = 0;

  for (const [rel, content] of Object.entries(files)) {
    const dest = path.join(rootDir, rel);
    const current = fs.existsSync(dest) ? fs.readFileSync(dest, "utf8") : null;
    if (current === content) continue;
    changed += 1;
    if (check) {
      console.error(`✗ out of date: ${rel}`);
      continue;
    }
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, content);
    console.log(`✓ wrote ${rel}`);
  }

  if (check) {
    if (changed > 0) {
      console.error(`\n${changed} file(s) in ${label} are stale — ${fixHint}.`);
      process.exit(1);
    }
    console.log(`✓ ${label} is up to date`);
  }
  return changed;
}

/** `import()` needs a file:// URL on Windows, where a bare path reads as a protocol. */
export function pathToFileUrl(p) {
  return new URL(`file://${p.replace(/\\/g, "/").replace(/^([A-Za-z]:)/, "/$1")}`).href;
}
