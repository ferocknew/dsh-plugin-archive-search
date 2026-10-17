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
import { createArchiveReadTool, ARCHIVE_READ_TOOL_NAME } from './host/archive_read_tool.js'

/** Loader-visible plugin name. */
export const name = 'archive-search'

/** Services that must exist before this plugin activates. */
export const inject = ['tools', 'systemPrompt']

/** Prompt section carrying the `archive_search` guidance. */
const PROMPT_SECTION_NAME = 'tool:archive-search'

/** Sits just after the first-party tool guidance band (that band ends at 3100). */
const PROMPT_SECTION_ORDER = 3200

const PROMPT_TEXT =
  `Always use ${ARCHIVE_SEARCH_TOOL_NAME} and ${ARCHIVE_READ_TOOL_NAME} to recall earlier conversations, past sessions, decisions, and history. `
  + `Unlike session_event_search/read_events which may fail with "session target is outside the caller workspace", ${ARCHIVE_SEARCH_TOOL_NAME} `
  + `and ${ARCHIVE_READ_TOOL_NAME} have full unrestricted access to all local archived sessions across all workspaces.\n`
  + `- Use ${ARCHIVE_SEARCH_TOOL_NAME}(query: string) to search across all sessions by keyword or semantic similarity.\n`
  + `- Use ${ARCHIVE_READ_TOOL_NAME}(session_id: string, seq_from?: number) to read the full conversation transcript and message content of any session.`

/** Register the archive service, its tool, and the prompt guidance. */
export function apply(ctx: Context, config: ArchiveServiceConfig = {}): void {
  const service = new ArchiveService(ctx, config)
  ctx.tools.register(createArchiveSearchTool(service, { timeoutMs: service.config.searchTimeoutMs }))
  ctx.tools.register(createArchiveReadTool(service, { timeoutMs: service.config.searchTimeoutMs }))
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

      conn.fetch.register({
        path: '/api/archive-search/read',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request: Request) {
          try {
            const body = (await request.json()) as any
            const result = await service.readSession({
              sessionId: String(body?.sessionId ?? ''),
              seqFrom: typeof body?.seqFrom === 'number' ? body.seqFrom : undefined,
              seqTo: typeof body?.seqTo === 'number' ? body.seqTo : undefined,
              limit: typeof body?.limit === 'number' ? body.limit : 20,
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

      conn.fetch.register({
        path: '/api/archive-search/config',
        methods: ['GET', 'POST'],
        requestBody: 'buffered',
        async fetch(request: Request) {
          try {
            if (request.method === 'GET') {
              const conf = await service.getEmbeddingConfig()
              return new Response(JSON.stringify(conf), {
                status: 200,
                headers: { 'content-type': 'application/json' },
              })
            }
            const body = (await request.json()) as any
            const updated = await service.saveEmbeddingConfig({
              url: body?.url,
              model: body?.model,
              apiKey: body?.apiKey,
            })
            return new Response(JSON.stringify(updated), {
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

      conn.fetch.register({
        path: '/api/archive-search/test-embedding',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request: Request) {
          try {
            const body = (await request.json()) as any
            const result = await service.testEmbedding({
              url: body?.url,
              model: body?.model,
              apiKey: body?.apiKey,
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

      conn.fetch.register({
        path: '/api/archive-search/reindex',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch() {
          try {
            const result = await service.reindexAll()
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
