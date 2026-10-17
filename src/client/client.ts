/**
 * dsh-plugin-archive-search 的 Client 入口：把归档搜索面板注册进 DSH Web Client
 * 的设置页（`settings.section` seat），只适配 DeepSeek Harness 0.2.1-alpha.3。
 *
 * 运行期契约（由 `@deepseek-ai/dsh-client-modules` 装载）：
 * - `package.json` 的 `dsh.client` 声明 `platform: 'web'`，并把本包的非基线模块依赖
 *   列进 `inject`；面板只用基线模块 `react` / `react/jsx-runtime`，故 `inject`
 *   只需保证 locale / settings 两个客户端插件先装载。
 * - 构建产物必须是 `window.__ModuleLoader__.load({ id, factory })` 信封，
 *   `factory(require)` 的返回值即本模块的导出（`apply` 与 `inject`）。
 *
 * Host 通信契约（Host 端需以 exact fetch 路由暴露 `archiveSearch.search`）：
 * - `POST /api/archive-search/search`，请求体 `{ query, mode, limit, contextLines, sessionId? }`；
 * - 命中返回 200 + `SearchResponse` JSON；失败返回非 2xx，可选 `{ error: { message } }`。
 */
import type { ComponentType } from 'react'
import type {
  HighlightRange,
  MatchedLine,
  MessageKind,
  MessageRole,
  SearchMode,
  SearchResponse,
  SearchResultItem,
} from '../core/types.js'
import {
  ARCHIVE_SEARCH_COPY,
  ArchiveSearchView,
  type ArchiveSearchCopy,
  type ArchiveSearchTransport,
  type Translate,
} from './archive_search_view.js'

/** 本插件拥有的 locale 命名空间。 */
export const NS = 'settings.archiveSearch'

/** 面板在设置页侧栏中的位置。 */
const SECTION_ORDER = 40

/** Host 端提供的归档检索路由。 */
export const ARCHIVE_SEARCH_ROUTE = '/api/archive-search/search'

/** cordis 的释放器。 */
type Disposer = () => void

/** slot 注册声明。 */
export interface ClientSlotRegistration {
  name: string
  id: string
  order?: number
  label?: () => string
  locale?: string
  inject?: () => Record<string, unknown>
}

/** `ctx.slots`：按名字声明式占用一个 seat。 */
export interface ClientSlots {
  inject(name: string, register: () => Disposer): Disposer
  register<P>(slot: ClientSlotRegistration, component: ComponentType<P>): Disposer
}

/** `ctx.locale`：字典注册与按命名空间绑定。 */
export interface ClientLocale {
  bind(namespace: string): Translate
  register(namespace: string, dictionaries: { zh: ArchiveSearchCopy; en: ArchiveSearchCopy }): Disposer
}

/**
 * Client 半边的 cordis 上下文。只用结构类型声明本插件实际消费的服务，
 * 因为 `slots` / `locale` 由运行期的客户端插件提供，没有可安装的类型包。
 */
export interface ClientContext {
  slots: ClientSlots
  locale: ClientLocale
  effect(callback: () => Disposer | void, label?: string): Disposer
}

/** 本浏览器插件需要的客户端服务。 */
export const inject = ['slots', 'locale']

/**
 * 创建走 Host fetch 路由的传输实现。
 * @param base_url - 路由前缀，默认同源；Electron 等非 HTTP 载体可在装配处覆盖。
 */
export function create_http_transport(base_url = ''): ArchiveSearchTransport {
  const route = `${base_url}${ARCHIVE_SEARCH_ROUTE}`
  return {
    async search(request, signal) {
      const fetch_impl = globalThis.fetch
      if (typeof fetch_impl !== 'function') {
        throw new Error('archive-search: this client has no fetch implementation')
      }
      const response = await fetch_impl(route, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(request),
        ...(signal === undefined ? {} : { signal }),
      })
      const payload = await read_json(response)
      if (!response.ok) throw new Error(error_message(response.status, payload))
      return to_search_response(payload)
    },
  }
}

/** 把面板注册进设置页；Host 端能力缺席时面板自行显示失败原因。 */
export function apply(ctx: ClientContext): void {
  const t = ctx.locale.bind(NS)
  ctx.effect(() => ctx.locale.register(NS, ARCHIVE_SEARCH_COPY), 'archive-search: dictionaries')
  const transport = create_http_transport()
  ctx.effect(
    () =>
      ctx.slots.inject('settings.section', () =>
        ctx.slots.register(
          {
            name: 'settings.section',
            id: 'archive-search',
            order: SECTION_ORDER,
            label: () => t('nav'),
            locale: NS,
            inject: () => ({ transport }),
          },
          ArchiveSearchView,
        ),
      ),
    'archive-search: settings page',
  )
}

async function read_json(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function error_message(status: number, payload: unknown): string {
  const record = as_record(payload)
  const nested = record === null ? null : as_record(record['error'])
  const message =
    (nested === null ? null : as_string(nested['message'])) ?? (record === null ? null : as_string(record['message']))
  return message ?? `HTTP ${status}`
}

const ROLES: readonly MessageRole[] = ['system', 'user', 'assistant', 'tool']
const KINDS: readonly MessageKind[] = ['message', 'reasoning', 'tool_call', 'tool_result', 'summary', 'title']

/** 校验收口：把 Host 返回的 JSON 归一化成视图可直接消费的 `SearchResponse`。 */
function to_search_response(raw: unknown): SearchResponse {
  const record = as_record(raw)
  if (record === null) throw new Error('archive-search: the Host returned a non-object response')
  const raw_results = record['results']
  if (!Array.isArray(raw_results)) throw new Error('archive-search: the Host response carries no results array')
  const results: SearchResultItem[] = []
  for (const entry of raw_results) {
    const item = to_result_item(entry)
    if (item !== null) results.push(item)
  }
  return {
    query: as_string(record['query']) ?? '',
    mode: record['mode'] === 'semantic' ? 'semantic' : 'keyword',
    results,
    total: as_number(record['total']) ?? results.length,
    returned: as_number(record['returned']) ?? results.length,
    offset: as_number(record['offset']) ?? 0,
    tookMs: as_number(record['tookMs']) ?? 0,
    scannedSessions: as_number(record['scannedSessions']) ?? 0,
    scannedMessages: as_number(record['scannedMessages']) ?? 0,
  }
}

/** 缺 sessionId / messageId 的条目无法定位与复制，直接丢弃而不是补假值。 */
function to_result_item(raw: unknown): SearchResultItem | null {
  const record = as_record(raw)
  if (record === null) return null
  const sessionId = as_string(record['sessionId'])
  const messageId = as_string(record['messageId'])
  if (sessionId === null || messageId === null) return null
  const title = as_string(record['title'])
  const cwd = as_string(record['cwd'])
  const toolName = as_string(record['toolName'])
  return {
    sessionId,
    messageId,
    filePath: as_string(record['filePath']) ?? '',
    seq: as_number(record['seq']) ?? 0,
    role: one_of(record['role'], ROLES) ?? 'user',
    kind: one_of(record['kind'], KINDS) ?? 'message',
    time: as_number(record['time']) ?? 0,
    score: as_number(record['score']) ?? 0,
    snippet: as_string(record['snippet']) ?? '',
    ranges: to_ranges(record['ranges']),
    matches: to_matches(record['matches']),
    source: record['source'] === 'vector' ? 'vector' : 'grep',
    ...(title === null ? {} : { title }),
    ...(cwd === null ? {} : { cwd }),
    ...(toolName === null ? {} : { toolName }),
  }
}

/** 高亮区间按起点排序并裁掉空区间，视图即可按游标顺序消费。 */
function to_ranges(value: unknown): HighlightRange[] {
  if (!Array.isArray(value)) return []
  const ranges: HighlightRange[] = []
  for (const entry of value) {
    const record = as_record(entry)
    if (record === null) continue
    const start = as_number(record['start'])
    const end = as_number(record['end'])
    if (start === null || end === null) continue
    const range = { start: Math.max(0, Math.trunc(start)), end: Math.max(0, Math.trunc(end)) }
    if (range.end > range.start) ranges.push(range)
  }
  return ranges.sort((left, right) => left.start - right.start)
}

function to_matches(value: unknown): MatchedLine[] {
  if (!Array.isArray(value)) return []
  const matches: MatchedLine[] = []
  for (const entry of value) {
    const record = as_record(entry)
    if (record === null) continue
    matches.push({
      line: as_number(record['line']) ?? 0,
      text: as_string(record['text']) ?? '',
      before: to_lines(record['before']),
      after: to_lines(record['after']),
      ranges: to_ranges(record['ranges']),
    })
  }
  return matches
}

function to_lines(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.filter((line): line is string => typeof line === 'string')
}

function as_record(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function as_string(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function as_number(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function one_of<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  if (typeof value !== 'string') return null
  return (allowed as readonly string[]).includes(value) ? (value as T) : null
}
