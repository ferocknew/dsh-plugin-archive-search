/**
 * 归档检索内核公共类型定义。
 * 约束：本目录禁止引入 sqlite / 原生 C++ 依赖，仅使用 Node 内置模块。
 */

/** 检索模式：keyword = 文本/正则匹配，semantic = 向量余弦检索。 */
export type SearchMode = 'keyword' | 'semantic';

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** 单条消息片段的来源类别，用于过滤与展示。 */
export type MessageKind =
  | 'message'
  | 'reasoning'
  | 'tool_call'
  | 'tool_result'
  | 'summary'
  | 'title';

export interface MessageItem {
  /** 全局唯一 id：`${sessionId}:${seq}:${ordinal}`，同时作为向量索引文档 id。 */
  id: string;
  /** 会话内的记录序号（DSH session v4 的 seq）。 */
  seq: number;
  role: MessageRole;
  kind: MessageKind;
  text: string;
  /** epoch 毫秒。 */
  time: number;
  turn?: number;
  step?: number;
  toolName?: string;
  callId?: string;
}

export interface SessionRecord {
  id: string;
  filePath: string;
  cwd?: string;
  title?: string;
  createdAt: number;
  updatedAt: number;
  origin?: string;
  parentSession?: string;
  agentPreset?: string;
  delegationDepth?: number;
  fileSize: number;
  fileMtimeMs: number;
  messages: MessageItem[];
}

export interface HighlightRange {
  start: number;
  end: number;
}

/** 命中行及其上下文窗口（before/after 为该行前后的原文行，已按 maxLineLength 截断）。 */
export interface MatchedLine {
  /** 1 基行号。 */
  line: number;
  text: string;
  before: string[];
  after: string[];
  /** 相对 text 的高亮区间，按 start 升序且互不重叠。 */
  ranges: HighlightRange[];
}

export interface SearchOptions {
  query: string;
  mode?: SearchMode;
  /** 返回条数上限，默认 20。 */
  limit?: number;
  /** 跳过条数，默认 0。 */
  offset?: number;
  /** 命中行前后各取多少行上下文，默认 2。 */
  contextLines?: number;
  caseSensitive?: boolean;
  /** keyword 模式下把 query 整体当作正则。 */
  regex?: boolean;
  /** keyword 模式多词时是否要求全部命中（AND），默认 true；false 为 OR。 */
  matchAll?: boolean;
  roles?: MessageRole[];
  kinds?: MessageKind[];
  sessionIds?: string[];
  /** 仅检索 cwd 以此前缀开头的会话。 */
  cwdPrefix?: string;
  /** 消息时间下界（含）。 */
  since?: number;
  /** 消息时间上界（含）。 */
  until?: number;
  /** semantic 模式最低余弦相似度，默认 0.05。 */
  minScore?: number;
  /** 摘要片段最大字符数，默认 240。 */
  snippetLength?: number;
  /** 单条消息最多返回的命中行数，默认 5。 */
  maxMatchesPerMessage?: number;
}

export interface SearchResultItem {
  sessionId: string;
  filePath: string;
  title?: string;
  cwd?: string;
  messageId: string;
  seq: number;
  role: MessageRole;
  kind: MessageKind;
  time: number;
  /** keyword 模式为词命中得分，semantic 模式为余弦相似度。 */
  score: number;
  snippet: string;
  /** 相对 snippet 的高亮区间。 */
  ranges: HighlightRange[];
  matches: MatchedLine[];
  source: 'grep' | 'vector';
  toolName?: string;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  results: SearchResultItem[];
  /** 过滤后命中总数（未受 limit/offset 影响）。 */
  total: number;
  returned: number;
  offset: number;
  tookMs: number;
  scannedSessions: number;
  scannedMessages: number;
}

export interface IndexStats {
  sessions: number;
  messages: number;
  /** 向量索引中的文档数。 */
  indexedDocuments: number;
  vectorDimensions: number;
  vectorFeatures: number;
  sessionsRoot: string;
  indexDir: string;
  filesScanned: number;
  filesParsed: number;
  filesSkipped: number;
  lastIndexedAt: number;
  errors: string[];
}

/** 外部/远程 Embedding 适配接口；实现后注入 VectorIndex 即可替换内置 TF-IDF 向量。 */
export interface EmbeddingProvider {
  id: string;
  dimensions: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export type LogFn = (message: string) => void;

export const DEFAULT_LIMIT = 20;
export const DEFAULT_CONTEXT_LINES = 2;
export const DEFAULT_SNIPPET_LENGTH = 240;
export const DEFAULT_MAX_MATCHES = 5;
export const DEFAULT_MIN_SCORE = 0.05;
/** 上下文行/命中行的单行最大长度，超出截断并加省略号。 */
export const MAX_LINE_LENGTH = 400;

export function normalizeSearchOptions(options: SearchOptions): Required<
  Pick<
    SearchOptions,
    | 'mode'
    | 'limit'
    | 'offset'
    | 'contextLines'
    | 'caseSensitive'
    | 'regex'
    | 'matchAll'
    | 'minScore'
    | 'snippetLength'
    | 'maxMatchesPerMessage'
  >
> &
  SearchOptions {
  return {
    ...options,
    mode: options.mode ?? 'keyword',
    limit: options.limit ?? DEFAULT_LIMIT,
    offset: options.offset ?? 0,
    contextLines: options.contextLines ?? DEFAULT_CONTEXT_LINES,
    caseSensitive: options.caseSensitive ?? false,
    regex: options.regex ?? false,
    matchAll: options.matchAll ?? true,
    minScore: options.minScore ?? DEFAULT_MIN_SCORE,
    snippetLength: options.snippetLength ?? DEFAULT_SNIPPET_LENGTH,
    maxMatchesPerMessage: options.maxMatchesPerMessage ?? DEFAULT_MAX_MATCHES,
  };
}
