/**
 * The model-facing `archive_search` tool: a typed declaration over
 * {@link ArchiveService} plus the Markdown rendering of its hits.
 *
 * Pinned to the DeepSeek Harness 0.2.1-alpha.3 tool contract
 * (`@deepseek-ai/dsh-tools` `defineTool`).
 */
import { defineTool, type ParameterSchemaSpec, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { MatchedLine, SearchResponse, SearchResultItem } from '../core/types.js'
import {
  ArchiveSearchError,
  DEFAULT_ARCHIVE_LIMIT,
  DEFAULT_SEARCH_TIMEOUT_MS,
  type ArchiveService,
} from './archive_service.js'

/** Model-facing tool name. */
export const ARCHIVE_SEARCH_TOOL_NAME = 'archive_search'

/** Ceiling for one rendered result, so a wide match set cannot flood the context. */
const MAX_RESULT_CHARS = 12_000

const TOOL_DESCRIPTION =
  "Search the local archive of all user's earlier conversations and sessions across all workspaces (unrestricted by workspace boundaries). "
  + 'Supports literal keywords, regex, or semantic similarity ranking. Use this tool whenever you need to recall past context, prior solutions, or history.'

const PARAMETERS = {
  query: {
    type: 'string',
    required: true,
    description: 'What to recall: keywords, a regular expression, or a natural-language description.',
  },
  mode: {
    type: 'string',
    description:
      'Search mode: "keyword" (default, exact/regex matching) or "semantic" (vector similarity ranking).',
  },
  limit: {
    type: 'integer',
    description: `Maximum number of matches to return. Defaults to ${DEFAULT_ARCHIVE_LIMIT}.`,
  },
  session_id: {
    type: 'string',
    description: 'Restrict the search to one archived conversation id.',
  },
} satisfies ParameterSchemaSpec

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

/** Options the plugin entry resolves from its own config. */
export interface ArchiveSearchToolOptions {
  timeoutMs?: number
}

/** Build the `archive_search` definition bound to one archive service. */
export function createArchiveSearchTool(
  service: ArchiveService,
  options: ArchiveSearchToolOptions = {},
): ToolDefinition {
  return defineTool({
    name: ARCHIVE_SEARCH_TOOL_NAME,
    description: TOOL_DESCRIPTION,
    parameters: PARAMETERS,
    output: TEXT_OUTPUT,
    timeoutMs: options.timeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      try {
        const response = await service.search(
          {
            query: args.query,
            ...(args.mode === undefined ? {} : { mode: args.mode }),
            ...(args.limit === undefined ? {} : { limit: args.limit }),
            ...(args.session_id === undefined ? {} : { sessionId: args.session_id }),
          },
          exec.signal,
        )
        return formatArchiveSearchResult(response)
      } catch (error) {
        if (error instanceof ArchiveSearchError) {
          throw new Error(`${error.code}: ${error.message}`)
        }
        throw error
      }
    },
  })
}

/** Render one search response as a compact Markdown digest the model reads directly. */
export function formatArchiveSearchResult(response: SearchResponse): string {
  const { query, mode, results, total } = response
  const matches = total === 1 ? '1 match' : `${total} matches`
  const header = `## archive_search "${query}" (${mode}) — ${results.length} of ${matches}`
  const scope = response.scannedMessages === 0
    ? `${response.scannedSessions} conversation(s)`
    : `${response.scannedMessages} archived message(s) in ${response.scannedSessions} conversation(s)`
  const coverage = `Searched ${scope} in ${response.tookMs} ms.`

  if (results.length === 0) {
    return [
      header,
      '',
      coverage,
      '',
      'No match. Retry with broader or fewer keywords, without session_id, or with mode "semantic".',
    ].join('\n')
  }

  const sections: string[] = [header, '', coverage, '']
  let rendered = sections.join('\n').length
  let included = 0
  for (const [position, item] of results.entries()) {
    const section = formatResultItem(item, position + 1)
    if (included > 0 && rendered + section.length > MAX_RESULT_CHARS) break
    sections.push(section)
    rendered += section.length
    included += 1
  }
  if (included < results.length) {
    sections.push(`(${results.length - included} further match(es) omitted for length.)`)
  } else if (total > results.length) {
    sections.push(`(Capped at limit ${response.returned}; narrow the query to see the remaining ${total - results.length}.)`)
  }
  return sections.join('\n')
}

function formatResultItem(item: SearchResultItem, position: number): string {
  const title = item.title === undefined || item.title === '' ? '' : ` "${item.title}"`
  const location = [
    `${formatTime(item.time)}`,
    `${item.role}${item.kind === 'message' ? '' : `/${item.kind}`}`,
    `session ${item.sessionId}${title}`,
  ]
  const lines = [item.snippet]
  if (item.matches.length > 0) {
    lines.length = 0
    for (const [index, match] of item.matches.entries()) {
      if (index > 0) lines.push('  …')
      lines.push(...formatMatch(match))
    }
  }
  return [
    `### ${position}. score ${formatScore(item.score)} · ${location.join(' · ')}`,
    '```text',
    ...lines,
    '```',
    `Tip: Read full conversation by calling archive_read(session_id: "${item.sessionId}", seq_from: ${Math.max(0, item.seq - 2)}, limit: 15)`,
    '',
  ].join('\n')
}

function formatMatch(match: MatchedLine): string[] {
  return [
    ...match.before,
    `> ${match.text}`,
    ...match.after,
  ]
}

function formatScore(score: number): string {
  return Number.isFinite(score) ? score.toFixed(3) : 'n/a'
}

function formatTime(time: number): string {
  if (!Number.isFinite(time) || time <= 0) return 'unknown time'
  return new Date(time).toISOString()
}
