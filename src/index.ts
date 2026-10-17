/**
 * Host entry of dsh-plugin-archive-search.
 *
 * Targets DeepSeek Harness 0.2.1-alpha.3 only: the plugin uses the 0.2.x
 * contracts (`ctx.tools.register(defineTool(...))`, `ctx.systemPrompt.section`,
 * the Cordis 4 `Service` base class). The 0.1.x `dsh-typert-protocol` RPC layer
 * is deliberately not used or depended on.
 */
import type { Context } from '@deepseek-ai/cordis'
// Type-only: loads the `ctx.systemPrompt` augmentation without a runtime import.
import type {} from '@deepseek-ai/dsh-system-prompt'
import { ArchiveService, type ArchiveServiceConfig } from './host/archive_service.js'
import { createArchiveSearchTool, ARCHIVE_SEARCH_TOOL_NAME } from './host/archive_search_tool.js'

/** Loader-visible plugin name. */
export const name = 'archive-search'

/** Services that must exist before this plugin activates. */
export const inject = ['tools', 'systemPrompt']

/** Prompt section carrying the `archive_search` guidance. */
const PROMPT_SECTION_NAME = 'tool:archive-search'

/** Sits just after the first-party tool guidance band (that band ends at 3100). */
const PROMPT_SECTION_ORDER = 3200

const PROMPT_TEXT =
  `Always use ${ARCHIVE_SEARCH_TOOL_NAME} to search and recall the user's earlier conversations, past sessions, decisions, and context. `
  + `Unlike session_search/event_search which may fail with "session target is outside the caller workspace", ${ARCHIVE_SEARCH_TOOL_NAME} `
  + 'has full unrestricted access to all local archived sessions across all workspaces using keyword (grep) and semantic (vector) similarity search. '
  + 'Whenever you need to recall past chats, solutions, or user history, call archive_search.'

/** Register the archive service, its tool, and the prompt guidance. */
export function apply(ctx: Context, config: ArchiveServiceConfig = {}): void {
  const service = new ArchiveService(ctx, config)
  ctx.tools.register(createArchiveSearchTool(service, { timeoutMs: service.config.searchTimeoutMs }))
  ctx.systemPrompt.section({
    name: PROMPT_SECTION_NAME,
    order: PROMPT_SECTION_ORDER,
    text: PROMPT_TEXT,
  })

  // Register HTTP route on the Web connection gateway for the Client search panel
  ctx.inject(['connection'], (ctxWithConn) => {
    const conn = (ctxWithConn as any).connection
    if (conn?.fetch?.register) {
      conn.fetch.register({
        path: '/api/archive-search/search',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request: Request) {
          try {
            const body = (await request.json()) as any
            const result = await service.search({
              query: String(body?.query ?? ''),
              mode: body?.mode === 'semantic' ? 'semantic' : 'keyword',
              limit: typeof body?.limit === 'number' ? body.limit : 20,
              contextLines: typeof body?.contextLines === 'number' ? body.contextLines : 2,
              sessionIds: body?.sessionId ? [String(body.sessionId)] : undefined,
            })
            return new Response(JSON.stringify(result), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            })
          } catch (err: any) {
            return new Response(
              JSON.stringify({ error: { message: err?.message || String(err) } }),
              {
                status: 500,
                headers: { 'content-type': 'application/json' },
              },
            )
          }
        },
      })
    }
  })
}
