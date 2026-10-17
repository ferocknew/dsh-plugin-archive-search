/**
 * Archive retrieval service: one process-wide `ctx.archiveSearch` over the
 * user's local conversation archive, shared by the `archive_search` tool and by
 * profile RPC callers.
 *
 * Pinned to the DeepSeek Harness 0.2.1-alpha.3 service model
 * (`@deepseek-ai/cordis` 4.x). The index is a derived cache over the session
 * log, so every refresh may rebuild it from disk.
 */
import { Service, type Context } from '@deepseek-ai/cordis'
import { ArchiveIndexer } from '../core/archive_indexer.js'
import type { IndexStats, SearchMode, SearchResponse } from '../core/types.js'

/** Service name this instance registers under. */
export const ARCHIVE_SERVICE_NAME = 'archiveSearch'

/** Maximum matches returned when the caller omits `limit`. */
export const DEFAULT_ARCHIVE_LIMIT = 5

/** Default freshness window before a search triggers an incremental rescan. */
export const DEFAULT_REFRESH_INTERVAL_MS = 60_000

/** Default ceiling applied to a caller-supplied `limit`. */
export const DEFAULT_MAX_LIMIT = 50

/** Default cooperative deadline for one `archive_search` call. */
export const DEFAULT_SEARCH_TIMEOUT_MS = 30_000

/** Characters kept from one matched message in a search response. */
export const SNIPPET_CHARS = 400

/** Context lines kept around each matched line in a search response. */
export const SNIPPET_CONTEXT_LINES = 2

/** Matched lines kept per message in a search response. */
export const MATCHES_PER_MESSAGE = 3

/** One archive search request, from the tool or from an RPC caller. */
export interface ArchiveSearchRequest {
  query: string
  /** Match strategy: `keyword` or `semantic`. Any other value is rejected here. */
  mode?: string
  limit?: number
  /** Restrict the search to one archived conversation. */
  sessionId?: string
}

/** Why an archive search could not produce a response. */
export type ArchiveSearchErrorCode =
  | 'EMPTY_QUERY'
  | 'INVALID_REQUEST'
  | 'INDEX_UNAVAILABLE'
  | 'SEARCH_FAILED'
  | 'ABORTED'

/** Caller-facing archive failure carrying a stable code. */
export class ArchiveSearchError extends Error {
  constructor(code: ArchiveSearchErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'ArchiveSearchError'
    this.code = code
  }

  readonly code: ArchiveSearchErrorCode
}

/** Config as written in the profile's `cordis.patch.yml`. */
export interface ArchiveServiceConfig {
  /** Archive root; defaults to the harness session root. */
  sessionsRoot?: string
  /** Index mirror directory; defaults to the harness archive-search directory. */
  indexDir?: string
  refreshIntervalMs?: number
  maxLimit?: number
  searchTimeoutMs?: number
}

/** Config with every default applied. */
export interface ResolvedArchiveServiceConfig {
  sessionsRoot?: string
  indexDir?: string
  refreshIntervalMs: number
  maxLimit: number
  searchTimeoutMs: number
}

/** Apply defaults and reject values that would break the index or a search. */
export function resolveArchiveServiceConfig(
  config: ArchiveServiceConfig = {},
): ResolvedArchiveServiceConfig {
  const refreshIntervalMs = config.refreshIntervalMs ?? DEFAULT_REFRESH_INTERVAL_MS
  const maxLimit = config.maxLimit ?? DEFAULT_MAX_LIMIT
  const searchTimeoutMs = config.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS
  assertPositiveInteger(refreshIntervalMs, 'refreshIntervalMs')
  assertPositiveInteger(maxLimit, 'maxLimit')
  assertPositiveInteger(searchTimeoutMs, 'searchTimeoutMs')
  return {
    ...(config.sessionsRoot === undefined ? {} : { sessionsRoot: config.sessionsRoot }),
    ...(config.indexDir === undefined ? {} : { indexDir: config.indexDir }),
    refreshIntervalMs,
    maxLimit,
    searchTimeoutMs,
  }
}

/**
 * Owns the index lifecycle around {@link ArchiveIndexer}: a lazy first index, a
 * freshness window, single-flight refreshes, and the request validation its
 * callers (model tool, RPC) are not trusted to perform.
 */
export class ArchiveService extends Service {
  static readonly provide = ARCHIVE_SERVICE_NAME

  readonly config: ResolvedArchiveServiceConfig

  private readonly indexer: ArchiveIndexer

  private refreshedAt = 0

  private pendingRefresh: Promise<void> | undefined

  constructor(ctx: Context, config: ArchiveServiceConfig = {}) {
    super(ctx, ARCHIVE_SERVICE_NAME)
    this.config = resolveArchiveServiceConfig(config)
    this.indexer = new ArchiveIndexer({
      ...(this.config.sessionsRoot === undefined ? {} : { sessionsRoot: this.config.sessionsRoot }),
      ...(this.config.indexDir === undefined ? {} : { indexDir: this.config.indexDir }),
    })
    ctx.effect(() => () => { this.indexer.close() }, 'archiveSearch.close')
  }

  /** Size and freshness of the index, without triggering a scan. */
  stats(): IndexStats {
    return this.indexer.stats()
  }

  /** Milliseconds since the last refresh, or Infinity before the first one. */
  age(): number {
    return this.refreshedAt === 0 ? Number.POSITIVE_INFINITY : Date.now() - this.refreshedAt
  }

  /** Force the next search to rescan the archive. */
  invalidate(): void {
    this.refreshedAt = 0
  }

  /** Rescan now, ignoring the freshness window. */
  async refresh(signal?: AbortSignal): Promise<IndexStats> {
    await this.ensureIndexed(signal, true)
    return this.indexer.stats()
  }

  /** Search the archive, refreshing the index first when it is stale. */
  async search(request: ArchiveSearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
    const query = typeof request?.query === 'string' ? request.query.trim() : ''
    if (query.length === 0) {
      throw new ArchiveSearchError('EMPTY_QUERY', 'archive_search requires a non-empty query')
    }
    const mode = resolveMode(request.mode)
    const limit = resolveLimit(request.limit, this.config.maxLimit)
    const sessionId = resolveSessionId(request.sessionId)

    await this.ensureIndexed(signal, false)
    assertNotAborted(signal)

    try {
      return await this.indexer.search({
        query,
        mode,
        limit,
        contextLines: SNIPPET_CONTEXT_LINES,
        snippetLength: SNIPPET_CHARS,
        maxMatchesPerMessage: MATCHES_PER_MESSAGE,
        ...(sessionId === undefined ? {} : { sessionIds: [sessionId] }),
      })
    } catch (error) {
      if (error instanceof ArchiveSearchError) throw error
      throw new ArchiveSearchError(
        'SEARCH_FAILED',
        `${mode} search failed: ${describeError(error)}`,
        { cause: error },
      )
    }
  }

  /** Read full conversation transcript of one session, bypassing workspace boundaries. */
  async readSession(
    request: { sessionId: string; seqFrom?: number; seqTo?: number; limit?: number },
    signal?: AbortSignal,
  ): Promise<{ session: any; messages: any[]; totalInSession: number }> {
    const rawId = typeof request?.sessionId === 'string' ? request.sessionId.trim() : ''
    if (!rawId) {
      throw new ArchiveSearchError('INVALID_REQUEST', 'session_id is required')
    }

    await this.ensureIndexed(signal, false)
    assertNotAborted(signal)

    const cleanRawId = rawId.toLowerCase()
    const strippedId = cleanRawId.replace(/^session-/, '')
    const prefixedId = cleanRawId.startsWith('session-') ? cleanRawId : `session-${cleanRawId}`

    let session = this.indexer.getSession(rawId)
      || this.indexer.getSession(cleanRawId)
      || this.indexer.getSession(strippedId)
      || this.indexer.getSession(prefixedId)

    if (!session) {
      // 容错前缀匹配与包含匹配
      const all = this.indexer.listSessions()
      const found = all.find((s) => {
        const sLower = s.id.toLowerCase()
        const sStripped = sLower.replace(/^session-/, '')
        return sLower === cleanRawId
          || sStripped === strippedId
          || sLower.startsWith(cleanRawId)
          || sStripped.startsWith(strippedId)
          || sLower.includes(strippedId)
      })
      if (found) session = found
    }

    if (!session) {
      throw new ArchiveSearchError('INVALID_REQUEST', `Session not found: ${rawId}`)
    }

    const seqFrom = request.seqFrom ?? 0
    const seqTo = request.seqTo ?? Number.MAX_SAFE_INTEGER
    const limit = Math.min(Math.max(1, request.limit ?? 20), 100)

    const filtered = session.messages.filter((m) => m.seq >= seqFrom && m.seq <= seqTo)
    const messages = filtered.slice(0, limit)

    return {
      session,
      messages,
      totalInSession: session.messages.length,
    }
  }

  /** Index once, then reuse it until the freshness window expires. */
  private async ensureIndexed(signal: AbortSignal | undefined, force: boolean): Promise<void> {
    const fresh = this.refreshedAt !== 0
      && Date.now() - this.refreshedAt < this.config.refreshIntervalMs
    if (fresh && !force) return
    const pending = this.pendingRefresh
    if (pending !== undefined) return pending
    const refresh = this.runRefresh(signal)
    this.pendingRefresh = refresh
    try {
      await refresh
    } finally {
      if (this.pendingRefresh === refresh) this.pendingRefresh = undefined
    }
  }

  private async runRefresh(signal: AbortSignal | undefined): Promise<void> {
    assertNotAborted(signal)
    try {
      await this.indexer.ensureIndexed()
    } catch (error) {
      throw new ArchiveSearchError(
        'INDEX_UNAVAILABLE',
        `cannot index the conversation archive at ${this.indexer.sessionsRoot}: ${describeError(error)}`,
        { cause: error },
      )
    }
    this.refreshedAt = Date.now()
  }
}

function resolveMode(mode: unknown): SearchMode {
  return mode === 'semantic' ? 'semantic' : 'keyword'
}

function resolveLimit(limit: unknown, maxLimit: number): number {
  if (limit === undefined || limit === null) return Math.min(DEFAULT_ARCHIVE_LIMIT, maxLimit)
  if (typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1) {
    throw new ArchiveSearchError('INVALID_REQUEST', 'limit must be a positive integer')
  }
  return Math.min(limit, maxLimit)
}

function resolveSessionId(sessionId: unknown): string | undefined {
  if (sessionId === undefined || sessionId === null) return undefined
  if (typeof sessionId !== 'string' || sessionId.trim().length === 0) {
    throw new ArchiveSearchError('INVALID_REQUEST', 'session_id must be a non-empty string')
  }
  return sessionId.trim()
}

function assertNotAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new ArchiveSearchError('ABORTED', 'archive search was cancelled')
  }
}

function assertPositiveInteger(value: number, field: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError(`archive-search: ${field} must be a positive integer`)
  }
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
