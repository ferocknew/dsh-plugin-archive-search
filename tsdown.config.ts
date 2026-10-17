import { defineConfig } from 'tsdown'

/**
 * Build config for DeepSeek Harness 0.2.1-alpha.3.
 *
 * Two entries, both ESM: the Host plugin (`lib/index.js`, loaded by the dsh
 * Host process) and the Web Client module (`lib/client.js`, loaded by the
 * Harness Web UI). Harness and React packages stay unresolved at build time
 * because the running Harness supplies them.
 */
export default defineConfig([
  {
    name: 'archive-search-host',
    entry: { index: 'src/index.ts' },
    outDir: 'lib',
    format: 'esm',
    platform: 'node',
    target: 'node22',
    deps: { neverBundle: [/^@deepseek-ai\//] },
    outExtensions: () => ({ js: '.js' }),
    clean: true,
    dts: false,
    sourcemap: false,
  },
  {
    name: 'archive-search-client',
    entry: { client: 'src/client.ts' },
    outDir: 'lib',
    format: 'cjs',
    platform: 'browser',
    target: 'es2022',
    deps: { neverBundle: [/^@deepseek-ai\//, /^react($|\/)/, /^react-dom($|\/)/] },
    outExtensions: () => ({ js: '.js' }),
    clean: false,
    dts: false,
    sourcemap: false,
    banner: `window.__ModuleLoader__.load({\n  id: 'dsh-plugin-archive-search',\n  factory(require) {\n    const exports = {};\n    const module = { exports };\n`,
    footer: `\n    return module.exports;\n  },\n});\n`,
  },
])
