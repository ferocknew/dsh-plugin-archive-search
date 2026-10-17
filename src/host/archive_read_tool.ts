/**
 * The model-facing `archive_read` tool: reads full messages and transcripts
 * from any archived session across all workspaces, bypassing workspace barriers.
 */
import { defineTool, type ParameterSchemaSpec, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import type { MessageItem, SessionRecord } from '../core/types.js'
import {
  DEFAULT_SEARCH_TIMEOUT_MS,
  type ArchiveService,
} from './archive_service.js'

export const ARCHIVE_READ_TOOL_NAME = 'archive_read'

const TOOL_DESCRIPTION =
  "Read the full conversation transcript and message content of an archived session. "
  + "Use this to inspect complete past conversations, decisions, code snippets, or user inquiries across all workspaces without isolation restrictions."

const PARAMETERS = {
  session_id: {
    type: 'string',
    required: true,
    description: 'The session id or id prefix (e.g. "session-e3682507") to read.',
  },
  seq_from: {
    type: 'integer',
    description: 'Inclusive starting message sequence number (seq). Defaults to 0.',
  },
  seq_to: {
    type: 'integer',
    description: 'Inclusive ending message sequence number (seq).',
  },
  limit: {
    type: 'integer',
    description: 'Maximum number of messages to return. Defaults to 20.',
  },
} satisfies ParameterSchemaSpec

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

export interface ArchiveReadToolOptions {
  timeoutMs?: number
}

export function createArchiveReadTool(
  service: ArchiveService,
  options: ArchiveReadToolOptions = {},
): ToolDefinition {
  return defineTool({
    name: ARCHIVE_READ_TOOL_NAME,
    description: TOOL_DESCRIPTION,
    parameters: PARAMETERS,
    output: TEXT_OUTPUT,
    timeoutMs: options.timeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const response = await service.readSession(
        {
          sessionId: String(args.session_id),
          seqFrom: typeof args.seq_from === 'number' ? args.seq_from : undefined,
          seqTo: typeof args.seq_to === 'number' ? args.seq_to : undefined,
          limit: typeof args.limit === 'number' ? args.limit : 20,
        },
        exec.signal,
      )
      return formatArchiveReadResult(response)
    },
  })
}

export interface ArchiveReadResult {
  session: SessionRecord
  messages: MessageItem[]
  totalInSession: number
}

export function formatArchiveReadResult(result: ArchiveReadResult): string {
  const { session, messages, totalInSession } = result
  const lines: string[] = []
  lines.push(`## Session: ${session.title || session.id}`)
  lines.push(`- **Session ID**: \`${session.id}\``)
  if (session.cwd) lines.push(`- **Workspace CWD**: \`${session.cwd}\``)
  lines.push(`- **Created**: ${new Date(session.createdAt).toISOString()}`)
  lines.push(`- **Messages Returned**: ${messages.length} / ${totalInSession} total`)
  lines.push('')

  if (messages.length === 0) {
    lines.push('_No messages found in the requested sequence range._')
    return lines.join('\n')
  }

  for (const msg of messages) {
    const timeStr = new Date(msg.time).toLocaleTimeString()
    lines.push(`### [seq ${msg.seq}] ${msg.role.toUpperCase()} (${timeStr})`)
    if (msg.kind && msg.kind !== 'message') {
      lines.push(`_Kind: ${msg.kind}${msg.toolName ? ` | Tool: ${msg.toolName}` : ''}_`)
    }
    lines.push('')
    lines.push(msg.text.trim())
    lines.push('')
    lines.push('---')
  }

  return lines.join('\n')
}
