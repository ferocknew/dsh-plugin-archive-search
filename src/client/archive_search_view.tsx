/**
 * 归档搜索面板视图。
 *
 * 只依赖 React 与 DSH 主题 CSS 变量（`--dsw-alias-*`），深浅色由主题变量自动适配；
 * 数据来自注入的 `ArchiveSearchTransport`，视图自身不发起网络请求、不读本地文件。
 */
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactElement, type ReactNode } from 'react'
import type {
  HighlightRange,
  MatchedLine,
  MessageRole,
  SearchMode,
  SearchResponse,
  SearchResultItem,
} from '../core/types.js'

/** 文案读取器；由 DSH locale 服务绑定命名空间后提供。 */
export type Translate = (key: string) => string

/** 一次归档检索请求：与 Host 端 `archive_search` 入参同构，另加 UI 可调的上下文行数。 */
export interface ArchiveSearchRequest {
  query: string
  mode: SearchMode
  limit: number
  contextLines: number
  sessionId?: string
}

/** 视图与 Host 之间的唯一数据边界，由插件入口注入实现。 */
export interface ArchiveSearchTransport {
  search(request: ArchiveSearchRequest, signal?: AbortSignal): Promise<SearchResponse>
}

export interface ArchiveSearchViewProps {
  transport: ArchiveSearchTransport
  /** 由 slot 注入的文案读取器；缺省时回落到内置文案。 */
  t?: Translate
  /** 初始会话过滤，通常由宿主面板传入当前会话。 */
  initialSessionId?: string
}

/** 文案表（扁平 key → 文本，`{name}` 为插值占位符）。 */
export type ArchiveSearchCopy = Record<string, string>

/** 本插件拥有的文案；`client.ts` 用同一份注册到 locale 服务。 */
export const ARCHIVE_SEARCH_COPY: { zh: ArchiveSearchCopy; en: ArchiveSearchCopy } = {
  zh: {
    nav: '归档搜索',
    title: '归档搜索',
    subtitle: '在本地会话归档中检索历史对话：关键词匹配或语义相似度。',
    placeholder: '输入关键词或正则，按 Enter 检索',
    search: '检索',
    searching: '检索中…',
    reset: '清空',
    mode: '检索模式',
    'mode.keyword': '关键词',
    'mode.semantic': '语义',
    'mode.keyword.hint': '按文本或正则匹配，命中处高亮显示。',
    'mode.semantic.hint': '按向量余弦相似度排序，适合模糊描述。',
    options: '检索选项',
    'option.limit': '结果条数',
    'option.context': '上下文行数',
    'option.session': '会话范围',
    'option.session.placeholder': '留空检索全部会话',
    'option.session.clear': '清除会话过滤',
    'option.session.filtered': '已过滤会话 {session}',
    summary: '命中 {total} 条 · 返回 {returned} 条 · 耗时 {took} ms',
    'summary.scanned': '扫描 {sessions} 个会话 / {messages} 条消息',
    'state.idle': '输入关键词开始检索。',
    'state.loading': '正在检索归档…',
    'state.empty': '没有匹配的记录，换个关键词或切换到语义模式试试。',
    'state.error': '检索失败：{message}',
    retry: '重试',
    expand: '展开上下文',
    collapse: '收起上下文',
    copy: '复制',
    'copy.done': '已复制',
    'copy.failed': '复制失败',
    score: '得分',
    'role.user': '用户',
    'role.assistant': '助手',
    'role.tool': '工具',
    'role.system': '系统',
    'kind.tool_call': '工具调用',
    'kind.tool_result': '工具结果',
    'kind.reasoning': '思考',
    'kind.summary': '摘要',
    'kind.title': '标题',
    'kind.message': '消息',
    'session.unknown': '未知会话',
    'context.empty': '无上下文行',
  },
  en: {
    nav: 'Archive search',
    title: 'Archive search',
    subtitle: 'Search earlier conversations in the local archive by keyword or semantic similarity.',
    placeholder: 'Type a keyword or regex, press Enter to search',
    search: 'Search',
    searching: 'Searching…',
    reset: 'Clear',
    mode: 'Mode',
    'mode.keyword': 'Keyword',
    'mode.semantic': 'Semantic',
    'mode.keyword.hint': 'Matches text or a regular expression and highlights each hit.',
    'mode.semantic.hint': 'Ranks by vector cosine similarity; better for fuzzy descriptions.',
    options: 'Options',
    'option.limit': 'Results',
    'option.context': 'Context lines',
    'option.session': 'Session scope',
    'option.session.placeholder': 'Leave blank to search every session',
    'option.session.clear': 'Clear session filter',
    'option.session.filtered': 'Filtered to session {session}',
    summary: '{total} hits · {returned} shown · {took} ms',
    'summary.scanned': 'scanned {sessions} sessions / {messages} messages',
    'state.idle': 'Type a keyword to start searching.',
    'state.loading': 'Searching the archive…',
    'state.empty': 'No matching records. Try another keyword or the semantic mode.',
    'state.error': 'Search failed: {message}',
    retry: 'Retry',
    expand: 'Show context',
    collapse: 'Hide context',
    copy: 'Copy',
    'copy.done': 'Copied',
    'copy.failed': 'Copy failed',
    score: 'Score',
    'role.user': 'User',
    'role.assistant': 'Assistant',
    'role.tool': 'Tool',
    'role.system': 'System',
    'kind.tool_call': 'Tool call',
    'kind.tool_result': 'Tool result',
    'kind.reasoning': 'Reasoning',
    'kind.summary': 'Summary',
    'kind.title': 'Title',
    'kind.message': 'Message',
    'session.unknown': 'Unknown session',
    'context.empty': 'No context lines',
  },
}

/** 结果条数下拉项，上限与 Host 端 `maxLimit` 默认值一致。 */
const LIMIT_CHOICES = [10, 20, 30, 50] as const

/** 上下文行数下拉项。 */
const CONTEXT_CHOICES = [0, 1, 2, 3, 5] as const

const DEFAULT_LIMIT = 20
const DEFAULT_CONTEXT_LINES = 2

type SearchStatus = 'idle' | 'loading' | 'ready' | 'error'

interface SearchForm {
  mode: SearchMode
  limit: number
  contextLines: number
  sessionId: string
}

/** 命中行的角色徽标配色，取自主题的语义色变量。 */
const ROLE_TONE: Record<MessageRole, { bg: string; fg: string }> = {
  user: { bg: 'var(--dsw-alias-brand-primary)', fg: 'var(--dsw-alias-label-primary-foreground)' },
  assistant: { bg: 'var(--dsw-alias-state-success-tertiary)', fg: 'var(--dsw-alias-state-success-primary)' },
  tool: { bg: 'var(--dsw-alias-state-warn-tertiary)', fg: 'var(--dsw-alias-state-warn-label)' },
  system: { bg: 'var(--dsw-alias-bg-layer-3)', fg: 'var(--dsw-alias-label-secondary)' },
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '20px 24px 28px',
    color: 'var(--dsw-alias-label-primary)',
    fontSize: '13px',
    lineHeight: 1.55,
  },
  header: { display: 'flex', flexDirection: 'column', gap: '2px' },
  title: { fontSize: '16px', fontWeight: 600, margin: 0 },
  subtitle: { fontSize: '12px', color: 'var(--dsw-alias-label-secondary)', margin: 0 },
  form: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
  segment: {
    display: 'inline-flex',
    gap: '2px',
    padding: '2px',
    borderRadius: '8px',
    border: '1px solid var(--dsw-alias-border-l2)',
    background: 'var(--dsw-alias-bg-layer-2)',
  },
  segmentButton: {
    height: '28px',
    padding: '0 12px',
    border: 'none',
    borderRadius: '6px',
    background: 'transparent',
    color: 'var(--dsw-alias-label-secondary)',
    fontSize: '12px',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  segmentButtonActive: {
    background: 'var(--dsw-alias-bg-layer-1)',
    color: 'var(--dsw-alias-label-primary)',
    boxShadow: '0 1px 2px var(--dsw-alias-bg-mask-1)',
  },
  input: {
    flex: '1 1 320px',
    minWidth: '220px',
    height: '32px',
    padding: '0 10px',
    borderRadius: '8px',
    border: '1px solid var(--dsw-alias-border-l2)',
    background: 'var(--dsw-alias-bg-layer-1)',
    color: 'inherit',
    fontFamily: 'inherit',
    fontSize: '13px',
    outline: 'none',
  },
  primaryButton: {
    height: '32px',
    padding: '0 18px',
    border: 'none',
    borderRadius: '8px',
    background: 'var(--dsw-alias-button-primary-fill)',
    color: 'var(--dsw-alias-label-primary-foreground)',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  ghostButton: {
    height: '26px',
    padding: '0 10px',
    borderRadius: '6px',
    border: '1px solid var(--dsw-alias-border-l2)',
    background: 'transparent',
    color: 'var(--dsw-alias-label-secondary)',
    fontSize: '12px',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  options: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    flexWrap: 'wrap',
    fontSize: '12px',
    color: 'var(--dsw-alias-label-secondary)',
  },
  field: { display: 'inline-flex', alignItems: 'center', gap: '6px' },
  select: {
    height: '26px',
    padding: '0 6px',
    borderRadius: '6px',
    border: '1px solid var(--dsw-alias-border-l2)',
    background: 'var(--dsw-alias-bg-layer-1)',
    color: 'var(--dsw-alias-label-primary)',
    fontFamily: 'inherit',
    fontSize: '12px',
  },
  sessionInput: {
    height: '26px',
    width: '260px',
    padding: '0 8px',
    borderRadius: '6px',
    border: '1px solid var(--dsw-alias-border-l2)',
    background: 'var(--dsw-alias-bg-layer-1)',
    color: 'var(--dsw-alias-label-primary)',
    fontFamily: 'inherit',
    fontSize: '12px',
  },
  summary: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap',
    fontSize: '12px',
    color: 'var(--dsw-alias-label-tertiary)',
  },
  notice: {
    padding: '10px 12px',
    borderRadius: '8px',
    border: '1px solid var(--dsw-alias-border-l1)',
    background: 'var(--dsw-alias-bg-layer-1)',
    color: 'var(--dsw-alias-label-secondary)',
    fontSize: '12px',
  },
  noticeError: {
    padding: '10px 12px',
    borderRadius: '8px',
    border: '1px solid var(--dsw-alias-state-error-primary)',
    color: 'var(--dsw-alias-state-error-primary)',
    fontSize: '12px',
  },
  list: { display: 'flex', flexDirection: 'column', gap: '10px' },
  card: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '10px 12px',
    borderRadius: '10px',
    border: '1px solid var(--dsw-alias-border-l1)',
    background: 'var(--dsw-alias-bg-layer-1)',
  },
  cardHead: { display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 },
  roleBadge: {
    flex: '0 0 auto',
    padding: '1px 6px',
    borderRadius: '4px',
    fontSize: '11px',
    lineHeight: '16px',
  },
  kindBadge: {
    flex: '0 0 auto',
    padding: '1px 6px',
    borderRadius: '4px',
    fontSize: '11px',
    lineHeight: '16px',
    border: '1px solid var(--dsw-alias-border-l2)',
    color: 'var(--dsw-alias-label-tertiary)',
  },
  sessionLabel: {
    flex: '0 1 auto',
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: '12px',
    color: 'var(--dsw-alias-label-secondary)',
  },
  sessionId: {
    flex: '0 1 auto',
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    fontSize: '11px',
    color: 'var(--dsw-alias-label-dimmed)',
  },
  spacer: { flex: '1 1 auto' },
  meta: {
    flex: '0 0 auto',
    fontSize: '11px',
    color: 'var(--dsw-alias-label-tertiary)',
    whiteSpace: 'nowrap',
  },
  snippet: { margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '13px' },
  mark: {
    background: 'var(--dsw-alias-state-warn-tertiary)',
    color: 'var(--dsw-alias-label-primary)',
    borderRadius: '2px',
    padding: '0 1px',
  },
  actions: { display: 'flex', alignItems: 'center', gap: '8px' },
  context: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '8px 10px',
    borderRadius: '8px',
    background: 'var(--dsw-alias-bg-layer-2)',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    fontSize: '12px',
    overflowX: 'auto',
  },
  contextLine: { display: 'flex', gap: '8px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  contextLineDim: { color: 'var(--dsw-alias-label-dimmed)' },
  contextLineNo: {
    flex: '0 0 auto',
    minWidth: '2.5em',
    textAlign: 'right',
    color: 'var(--dsw-alias-label-caption)',
    userSelect: 'none',
  },
} satisfies Record<string, CSSProperties>

/** 取文案：优先用宿主注入的读取器，其次内置文案，最后回落 key 本身。 */
function make_reader(t: Translate | undefined): (key: string, params?: Record<string, string | number>) => string {
  return (key, params) => {
    const raw = t?.(key) ?? ARCHIVE_SEARCH_COPY.zh[key] ?? key
    if (params === undefined) return raw
    return raw.replace(/\{(\w+)\}/g, (match, name: string) => {
      const value = params[name]
      return value === undefined ? match : String(value)
    })
  }
}

/** 按命中区间渲染高亮；区间越界或乱序时按游标裁剪，保证不重复不丢失文本。 */
function render_highlight(text: string, ranges: readonly HighlightRange[]): ReactNode {
  if (ranges.length === 0) return text
  const nodes: ReactNode[] = []
  let cursor = 0
  ranges.forEach((range, index) => {
    const start = Math.max(cursor, Math.min(range.start, text.length))
    const end = Math.max(start, Math.min(range.end, text.length))
    if (start > cursor) nodes.push(text.slice(cursor, start))
    if (end > start) {
      nodes.push(
        <mark key={`hl-${index}`} style={styles.mark}>
          {text.slice(start, end)}
        </mark>,
      )
    }
    cursor = end
  })
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function format_time(time: number): string {
  if (!Number.isFinite(time) || time <= 0) return '—'
  return new Date(time).toLocaleString()
}

function format_score(item: SearchResultItem): string {
  if (!Number.isFinite(item.score)) return '—'
  return item.source === 'vector' ? item.score.toFixed(4) : item.score.toFixed(2)
}

/** 复制到剪贴板的纯文本：头部元信息 + 摘要 + 已展开的上下文行。 */
function build_copy_text(
  item: SearchResultItem,
  context_label: string,
  expanded: boolean,
): string {
  const lines = [
    `[${format_time(item.time)}] ${item.role}/${item.kind} session=${item.sessionId} seq=${item.seq}`,
    item.title === undefined ? '' : `title: ${item.title}`,
    '',
    item.snippet,
  ]
  if (expanded && item.matches.length > 0) {
    lines.push('', `--- ${context_label} ---`)
    for (const match of item.matches) {
      for (const line of match.before) lines.push(`  ${line}`)
      lines.push(`> ${match.line}: ${match.text}`)
      for (const line of match.after) lines.push(`  ${line}`)
    }
  }
  return lines.filter((line, index) => !(line === '' && index > 0 && lines[index - 1] === '')).join('\n')
}

/** 写入剪贴板；`navigator.clipboard` 不可用时回落到隐藏 textarea + execCommand。 */
async function write_clipboard(text: string): Promise<void> {
  const clipboard = globalThis.navigator?.clipboard
  if (clipboard !== undefined) {
    await clipboard.writeText(text)
    return
  }
  const area = globalThis.document?.createElement('textarea')
  if (area === undefined) throw new Error('clipboard is unavailable')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  globalThis.document.body.appendChild(area)
  area.select()
  const copied = globalThis.document.execCommand?.('copy') ?? false
  area.remove()
  if (!copied) throw new Error('clipboard is unavailable')
}

function context_line_text(line: string): string {
  return line.length === 0 ? ' ' : line
}

function render_context(matches: readonly MatchedLine[], empty_label: string): ReactNode {
  if (matches.length === 0) return <div style={styles.contextLineDim}>{empty_label}</div>
  return matches.map((match, match_index) => (
    <div key={`match-${match_index}`} style={styles.contextLine}>
      <span style={styles.contextLineNo}>{match.line}</span>
      <span style={{ minWidth: 0 }}>
        {match.before.map((line, index) => (
          <div key={`b${index}`} style={styles.contextLineDim}>
            {context_line_text(line)}
          </div>
        ))}
        <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {render_highlight(match.text, match.ranges)}
        </div>
        {match.after.map((line, index) => (
          <div key={`a${index}`} style={styles.contextLineDim}>
            {context_line_text(line)}
          </div>
        ))}
      </span>
    </div>
  ))
}

/** 归档搜索结果面板。 */
export function ArchiveSearchView(props: ArchiveSearchViewProps): ReactElement {
  const { transport, initialSessionId } = props
  const read = useMemo(() => make_reader(props.t), [props.t])

  const [form, set_form] = useState<SearchForm>({
    mode: 'keyword',
    limit: DEFAULT_LIMIT,
    contextLines: DEFAULT_CONTEXT_LINES,
    sessionId: initialSessionId ?? '',
  })
  const [query, set_query] = useState('')
  const [submitted, set_submitted] = useState('')
  const [status, set_status] = useState<SearchStatus>('idle')
  const [response, set_response] = useState<SearchResponse | null>(null)
  const [error, set_error] = useState('')
  const [expanded, set_expanded] = useState<ReadonlySet<string>>(() => new Set<string>())
  const [copied, set_copied] = useState('')
  const [known_sessions, set_known_sessions] = useState<readonly string[]>([])

  const abort_ref = useRef<AbortController | null>(null)
  const copied_timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => () => {
    abort_ref.current?.abort()
    if (copied_timer.current !== null) clearTimeout(copied_timer.current)
  }, [])

  const execute_search = useCallback(
    async (text: string, overrides: Partial<SearchForm> = {}) => {
      const trimmed = text.trim()
      if (trimmed.length === 0) {
        set_status('idle')
        set_error('')
        set_response(null)
        set_submitted('')
        return
      }
      const effective = { ...form, ...overrides }
      abort_ref.current?.abort()
      const controller = new AbortController()
      abort_ref.current = controller
      set_status('loading')
      set_error('')
      set_submitted(trimmed)
      try {
        const result = await transport.search(
          {
            query: trimmed,
            mode: effective.mode,
            limit: effective.limit,
            contextLines: effective.contextLines,
            ...(effective.sessionId.trim().length === 0 ? {} : { sessionId: effective.sessionId.trim() }),
          },
          controller.signal,
        )
        if (controller.signal.aborted) return
        set_response(result)
        set_status('ready')
        set_known_sessions((previous) => {
          const merged = new Set(previous)
          for (const item of result.results) merged.add(item.sessionId)
          return [...merged]
        })
      } catch (cause) {
        if (controller.signal.aborted) return
        set_error(cause instanceof Error ? cause.message : String(cause))
        set_status('error')
      }
    },
    [form, transport],
  )

  const on_submit = useCallback(
    (event: { preventDefault: () => void }) => {
      event.preventDefault()
      void execute_search(query)
    },
    [execute_search, query],
  )

  const switch_mode = useCallback(
    (mode: SearchMode) => {
      set_form((previous) => ({ ...previous, mode }))
      // 模式决定排序与命中口径，切换后立即用同一关键词重跑，避免展示与模式不一致。
      if (submitted.length > 0) void execute_search(submitted, { mode })
    },
    [execute_search, submitted],
  )

  const clear_query = useCallback(() => {
    abort_ref.current?.abort()
    set_query('')
    set_submitted('')
    set_response(null)
    set_status('idle')
    set_error('')
  }, [])

  const toggle_expanded = useCallback((message_id: string) => {
    set_expanded((previous) => {
      const next = new Set(previous)
      if (next.has(message_id)) next.delete(message_id)
      else next.add(message_id)
      return next
    })
  }, [])

  const copy_item = useCallback(
    async (item: SearchResultItem) => {
      const is_expanded = expanded.has(item.messageId)
      try {
        await write_clipboard(build_copy_text(item, read('expand'), is_expanded))
        set_copied(item.messageId)
        if (copied_timer.current !== null) clearTimeout(copied_timer.current)
        copied_timer.current = setTimeout(() => set_copied(''), 1600)
      } catch {
        set_copied(`!${item.messageId}`)
        if (copied_timer.current !== null) clearTimeout(copied_timer.current)
        copied_timer.current = setTimeout(() => set_copied(''), 1600)
      }
    },
    [expanded, read],
  )

  const results = response?.results ?? []
  const summary = useMemo(() => {
    if (response === null) return ''
    return `${read('summary', {
      total: response.total,
      returned: response.returned,
      took: Math.round(response.tookMs),
    })} · ${read('summary.scanned', {
      sessions: response.scannedSessions,
      messages: response.scannedMessages,
    })}`
  }, [read, response])

  const session_hint = form.sessionId.trim().length === 0
    ? read('option.session.placeholder')
    : read('option.session.filtered', { session: form.sessionId.trim() })

  return (
    <section style={styles.root}>
      <header style={styles.header}>
        <h2 style={styles.title}>{read('title')}</h2>
        <p style={styles.subtitle}>{read('subtitle')}</p>
      </header>

      <form style={styles.form} onSubmit={on_submit}>
        <div style={styles.segment} role="group" aria-label={read('mode')}>
          <button
            type="button"
            style={form.mode === 'keyword' ? { ...styles.segmentButton, ...styles.segmentButtonActive } : styles.segmentButton}
            title={read('mode.keyword.hint')}
            aria-pressed={form.mode === 'keyword'}
            onClick={() => switch_mode('keyword')}
          >
            {read('mode.keyword')}
          </button>
          <button
            type="button"
            style={form.mode === 'semantic' ? { ...styles.segmentButton, ...styles.segmentButtonActive } : styles.segmentButton}
            title={read('mode.semantic.hint')}
            aria-pressed={form.mode === 'semantic'}
            onClick={() => switch_mode('semantic')}
          >
            {read('mode.semantic')}
          </button>
        </div>
        <input
          style={styles.input}
          type="search"
          value={query}
          placeholder={read('placeholder')}
          aria-label={read('title')}
          onChange={(event) => set_query(event.target.value)}
        />
        <button type="submit" style={styles.primaryButton} disabled={status === 'loading'}>
          {status === 'loading' ? read('searching') : read('search')}
        </button>
        <button type="button" style={styles.ghostButton} onClick={clear_query}>
          {read('reset')}
        </button>
      </form>

      <div style={styles.options}>
        <span style={styles.field}>
          <label htmlFor="archive-search-limit">{read('option.limit')}</label>
          <select
            id="archive-search-limit"
            style={styles.select}
            value={form.limit}
            onChange={(event) => set_form((previous) => ({ ...previous, limit: Number(event.target.value) }))}
          >
            {LIMIT_CHOICES.map((choice) => (
              <option key={choice} value={choice}>
                {choice}
              </option>
            ))}
          </select>
        </span>
        <span style={styles.field}>
          <label htmlFor="archive-search-context">{read('option.context')}</label>
          <select
            id="archive-search-context"
            style={styles.select}
            value={form.contextLines}
            onChange={(event) => set_form((previous) => ({ ...previous, contextLines: Number(event.target.value) }))}
          >
            {CONTEXT_CHOICES.map((choice) => (
              <option key={choice} value={choice}>
                {choice}
              </option>
            ))}
          </select>
        </span>
        <span style={styles.field}>
          <label htmlFor="archive-search-session">{read('option.session')}</label>
          <input
            id="archive-search-session"
            style={styles.sessionInput}
            list="archive-search-session-options"
            value={form.sessionId}
            placeholder={session_hint}
            onChange={(event) => set_form((previous) => ({ ...previous, sessionId: event.target.value }))}
          />
          <datalist id="archive-search-session-options">
            {known_sessions.map((session_id) => (
              <option key={session_id} value={session_id} />
            ))}
          </datalist>
          {form.sessionId.trim().length === 0 ? null : (
            <button
              type="button"
              style={styles.ghostButton}
              onClick={() => set_form((previous) => ({ ...previous, sessionId: '' }))}
            >
              {read('option.session.clear')}
            </button>
          )}
        </span>
      </div>

      {summary.length === 0 ? null : <div style={styles.summary}>{summary}</div>}

      {status === 'error' ? (
        <div style={styles.noticeError}>
          <div>{read('state.error', { message: error })}</div>
          <button type="button" style={{ ...styles.ghostButton, marginTop: '8px' }} onClick={() => void execute_search(submitted)}>
            {read('retry')}
          </button>
        </div>
      ) : null}

      {status === 'loading' && results.length === 0 ? <div style={styles.notice}>{read('state.loading')}</div> : null}

      {status === 'idle' ? <div style={styles.notice}>{read('state.idle')}</div> : null}

      {status === 'ready' && results.length === 0 ? <div style={styles.notice}>{read('state.empty')}</div> : null}

      <div style={styles.list}>
        {results.map((item) => {
          const is_expanded = expanded.has(item.messageId)
          const tone = ROLE_TONE[item.role]
          const copy_state = copied === item.messageId
            ? read('copy.done')
            : copied === `!${item.messageId}`
              ? read('copy.failed')
              : read('copy')
          return (
            <article key={item.messageId} style={styles.card}>
              <div style={styles.cardHead}>
                <span style={{ ...styles.roleBadge, background: tone.bg, color: tone.fg }}>
                  {read(`role.${item.role}`)}
                </span>
                {item.kind === 'message' ? null : <span style={styles.kindBadge}>{read(`kind.${item.kind}`)}</span>}
                <span style={styles.sessionLabel} title={item.title ?? item.sessionId}>
                  {item.title ?? read('session.unknown')}
                </span>
                <span style={styles.sessionId} title={item.filePath}>
                  {item.sessionId}
                </span>
                <span style={styles.spacer} />
                <span style={styles.meta} title={read('score')}>
                  {format_time(item.time)} · {format_score(item)}
                </span>
              </div>

              <p style={styles.snippet}>{render_highlight(item.snippet, item.ranges)}</p>

              {item.matches.length === 0 ? null : (
                <div style={styles.actions}>
                  <button type="button" style={styles.ghostButton} onClick={() => toggle_expanded(item.messageId)}>
                    {is_expanded ? read('collapse') : `${read('expand')} (${item.matches.length})`}
                  </button>
                </div>
              )}

              {is_expanded ? <div style={styles.context}>{render_context(item.matches, read('context.empty'))}</div> : null}

              <div style={styles.actions}>
                <button type="button" style={styles.ghostButton} onClick={() => void copy_item(item)}>
                  {copy_state}
                </button>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
