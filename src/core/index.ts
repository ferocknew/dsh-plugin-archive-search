/**
 * 检索内核统一出口。
 *
 * 1) 内核原生 API：`ArchiveIndexer` / `GrepEngine` / `VectorIndex` 及完整类型（见各模块）；
 * 2) 宿主契约兼容层：`scan_archive_sessions` / `grep_search` / `vector_search` + `ArchiveChunk`，
 *    供 DSH 侧 `archive_service.ts` 以扁平消息块的方式调用。
 */
export * from './types';
export * from './grep_search';
export * from './vector_search';
export * from './archive_indexer';

import { GrepEngine } from './grep_search';
import { ArchiveIndexer, ArchiveIndexerOptions, resolveSessionsRoot } from './archive_indexer';
import {
  DEFAULT_LIMIT,
  DEFAULT_MIN_SCORE,
  LogFn,
  MessageKind,
  MessageRole,
  SearchResultItem,
  SessionRecord,
} from './types';
import { VectorDocument, VectorIndex, VectorHit } from './vector_search';

/* ------------------------------------------------------------------ */
/* 宿主契约：扁平消息块                                                  */
/* ------------------------------------------------------------------ */

/** 一条可检索的归档消息（会话元信息已展开，便于宿主直接渲染）。 */
export interface ArchiveChunk {
  id: string;
  session_id: string;
  role: string;
  text: string;
  /** ISO 时间字符串；未知时为空串。 */
  timestamp: string;
  /** 检索得分；索引阶段为 0。 */
  score: number;
  title?: string;
  kind?: string;
  seq?: number;
  tool_name?: string;
  file_path?: string;
  cwd?: string;
}

export type ArchiveHit = ArchiveChunk;

export interface ScanArchiveSessionsOptions {
  /** 会话根目录，默认 ~/.dsh/sessions。 */
  root?: string;
  /** 索引镜像文件路径（宿主 cache_path），默认 ~/.dsh/archive-search/archive_meta.json。 */
  cache_path?: string;
  signal?: AbortSignal;
  logger?: LogFn;
}

export interface ScanArchiveSessionsResult {
  chunks: ArchiveChunk[];
  sessions: number;
  files: number;
  /** 本次扫描完成时间（ISO）。 */
  scanned_at: string;
}

export interface ArchiveQueryOptions {
  query: string;
  limit?: number;
  /** 限定单个会话 id。 */
  session_id?: string;
  /** semantic 模式最低相似度，默认 0.05。 */
  min_score?: number;
}

function to_message_role(role: string | undefined): MessageRole {
  return role === 'system' || role === 'user' || role === 'assistant' || role === 'tool' ? role : 'user';
}

function to_message_kind(kind: string | undefined): MessageKind {
  switch (kind) {
    case 'message':
    case 'reasoning':
    case 'tool_call':
    case 'tool_result':
    case 'summary':
    case 'title':
      return kind;
    default:
      return 'message';
  }
}

function chunk_time(chunk: ArchiveChunk): number {
  if (!chunk.timestamp) return 0;
  const parsed = Date.parse(chunk.timestamp);
  return Number.isFinite(parsed) ? parsed : 0;
}

/** 展开索引结果：每条消息一个扁平块（含会话标题与 cwd）。 */
export function sessions_to_chunks(sessions: SessionRecord[]): ArchiveChunk[] {
  const chunks: ArchiveChunk[] = [];
  for (const session of sessions) {
    for (const message of session.messages) {
      chunks.push({
        id: message.id,
        session_id: session.id,
        role: message.role,
        kind: message.kind,
        text: message.text,
        timestamp: message.time > 0 ? new Date(message.time).toISOString() : '',
        score: 0,
        title: session.title,
        seq: message.seq,
        tool_name: message.toolName,
        file_path: session.filePath,
        cwd: session.cwd,
      });
    }
  }
  return chunks;
}

/** 扫描（必要时增量重建）归档索引，返回扁平消息块。 */
export async function scan_archive_sessions(
  options: ScanArchiveSessionsOptions = {},
): Promise<ScanArchiveSessionsResult> {
  const indexerOptions: ArchiveIndexerOptions = {
    sessionsRoot: options.root ?? resolveSessionsRoot(),
    signal: options.signal,
    logger: options.logger,
  };
  if (options.cache_path) indexerOptions.cachePath = options.cache_path;
  const indexer = new ArchiveIndexer(indexerOptions);
  const stats = await indexer.ensureIndexed();
  const chunks = sessions_to_chunks(indexer.listSessions());
  indexer.close();
  return {
    chunks,
    sessions: stats.sessions,
    files: stats.filesScanned,
    scanned_at: new Date().toISOString(),
  };
}

/* ------------------------------------------------------------------ */
/* 兼容层：基于扁平块的检索                                              */
/* ------------------------------------------------------------------ */

function chunks_to_sessions(chunks: ArchiveChunk[]): SessionRecord[] {
  const bySession = new Map<string, SessionRecord>();
  for (const chunk of chunks) {
    let session = bySession.get(chunk.session_id);
    if (!session) {
      session = {
        id: chunk.session_id,
        filePath: chunk.file_path ?? '',
        cwd: chunk.cwd,
        title: chunk.title,
        createdAt: 0,
        updatedAt: 0,
        fileSize: 0,
        fileMtimeMs: 0,
        messages: [],
      };
      bySession.set(chunk.session_id, session);
    }
    session.messages.push({
      id: chunk.id,
      seq: chunk.seq ?? session.messages.length,
      role: to_message_role(chunk.role),
      kind: to_message_kind(chunk.kind),
      text: chunk.text,
      time: chunk_time(chunk),
      toolName: chunk.tool_name,
    });
  }
  return Array.from(bySession.values());
}

function chunk_to_hit(chunk: ArchiveChunk | undefined, item: SearchResultItem): ArchiveHit {
  const time = item.time > 0 ? new Date(item.time).toISOString() : (chunk?.timestamp ?? '');
  return {
    id: item.messageId,
    session_id: item.sessionId,
    role: item.role,
    kind: item.kind,
    text: chunk?.text ?? item.snippet,
    timestamp: time,
    score: item.score,
    title: chunk?.title ?? item.title,
    seq: item.seq,
    tool_name: item.toolName ?? chunk?.tool_name,
    file_path: item.filePath || chunk?.file_path,
    cwd: item.cwd ?? chunk?.cwd,
  };
}

/** 关键词/正则检索（同步）。 */
export function grep_search(chunks: ArchiveChunk[], options: ArchiveQueryOptions): ArchiveHit[] {
  const query = typeof options?.query === 'string' ? options.query : '';
  if (!query.trim()) return [];
  const limit = options.limit ?? DEFAULT_LIMIT;
  const engine = new GrepEngine();
  const response = engine.search(chunks_to_sessions(chunks), {
    query,
    mode: 'keyword',
    limit,
    ...(options.session_id ? { sessionIds: [options.session_id] } : {}),
  });
  const byId = new Map(chunks.map((chunk) => [chunk.id, chunk]));
  return response.results.map((item) => chunk_to_hit(byId.get(item.messageId), item));
}

const vector_index_cache = new WeakMap<ArchiveChunk[], { index: VectorIndex; byId: Map<string, ArchiveChunk> }>();

/**
 * 语义检索（异步）。同一 chunks 数组只建一次向量索引（宿主每次扫描产生新数组，
 * 因此等价于“每次重扫重建一次”）。
 */
export async function vector_search(
  chunks: ArchiveChunk[],
  options: ArchiveQueryOptions,
): Promise<ArchiveHit[]> {
  const query = typeof options?.query === 'string' ? options.query : '';
  if (!query.trim() || chunks.length === 0) return [];
  const limit = options.limit ?? DEFAULT_LIMIT;
  const minScore = options.min_score ?? DEFAULT_MIN_SCORE;

  let cached = vector_index_cache.get(chunks);
  if (!cached) {
    const byId = new Map(chunks.map((chunk) => [chunk.id, chunk]));
    const index = new VectorIndex();
    index.addDocuments(
      chunks.map(
        (chunk): VectorDocument => ({
          id: chunk.id,
          sessionId: chunk.session_id,
          seq: chunk.seq ?? 0,
          text: chunk.text,
          role: to_message_role(chunk.role),
          kind: to_message_kind(chunk.kind),
          time: chunk_time(chunk),
          filePath: chunk.file_path,
          toolName: chunk.tool_name,
        }),
      ),
    );
    cached = { index, byId };
    vector_index_cache.set(chunks, cached);
  }

  const filter = options.session_id
    ? (doc: VectorDocument): boolean => doc.sessionId === options.session_id
    : undefined;
  const hits: VectorHit[] = cached.index.search(query, { limit, minScore, filter });
  return hits.map((hit) =>
    chunk_to_hit(cached!.byId.get(hit.doc.id), {
      sessionId: hit.doc.sessionId,
      filePath: hit.doc.filePath ?? '',
      title: cached!.byId.get(hit.doc.id)?.title,
      cwd: cached!.byId.get(hit.doc.id)?.cwd,
      messageId: hit.doc.id,
      seq: hit.doc.seq,
      role: hit.doc.role ?? 'user',
      kind: hit.doc.kind ?? 'message',
      time: hit.doc.time ?? 0,
      score: Number(hit.score.toFixed(6)),
      snippet: '',
      ranges: [],
      matches: [],
      source: 'vector',
      toolName: hit.doc.toolName,
    }),
  );
}
