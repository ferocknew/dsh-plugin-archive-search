/**
 * 归档索引器：解析 ~/.dsh/sessions 下的会话历史（jsonl / json / jsonl.zstd），
 * 协调 Grep 引擎与向量索引，提供全量索引、增量更新与统一检索入口。
 */
import { createReadStream } from 'node:fs';
import { promises as fs, type Dirent } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import zlib from 'node:zlib';
import { GrepEngine, buildSnippet, compileGrepPlan, grepText, truncateLine } from './grep_search';
import {
  EmbeddingProvider,
  HighlightRange,
  IndexStats,
  LogFn,
  MatchedLine,
  MessageItem,
  MessageKind,
  MessageRole,
  SearchOptions,
  SearchResponse,
  SearchResultItem,
  SessionRecord,
  normalizeSearchOptions,
} from './types';
import { VectorDocument, VectorIndex } from './vector_search';

export const INDEX_FORMAT_VERSION = 1;
export const DEFAULT_INDEX_DIR_NAME = 'archive-search';
export const META_FILE_NAME = 'archive_meta.json';
/** 向量镜像：gzip 压缩的 JSON（VectorIndex 读取时按魔数自动识别）。 */
export const VECTOR_FILE_NAME = 'archive_vectors.json.gz';
/** 会话文件名的候选优先级（同一目录内取第一个命中的扩展名）。 */
const SESSION_FILE_EXTENSIONS = ['.jsonl.zstd', '.jsonl.zst', '.jsonl', '.json'];

/** 默认会话目录：$DSH_HOME/sessions 或 ~/.dsh/sessions。 */
export function resolveSessionsRoot(): string {
  const dshHome = process.env.DSH_HOME?.trim();
  return dshHome ? path.join(dshHome, 'sessions') : path.join(os.homedir(), '.dsh', 'sessions');
}

/** 默认索引目录：$DSH_HOME/archive-search 或 ~/.dsh/archive-search。 */
export function resolveIndexDir(): string {
  const dshHome = process.env.DSH_HOME?.trim();
  return dshHome ? path.join(dshHome, DEFAULT_INDEX_DIR_NAME) : path.join(os.homedir(), '.dsh', DEFAULT_INDEX_DIR_NAME);
}

/* ------------------------------------------------------------------ */
/* zstd 多帧解压                                                        */
/* ------------------------------------------------------------------ */

const ZSTD_MAGIC = Buffer.from([0x28, 0xb5, 0x2f, 0xfd]);

export function zstdAvailable(): boolean {
  return typeof (zlib as unknown as Record<string, unknown>).zstdDecompressSync === 'function';
}

/**
 * 解压 zstd 文件。DSH 会话文件是「每个记录一个 zstd 帧」的拼接格式，
 * 普通流式解压只认第一帧，因此这里按帧头切分后逐帧解压。
 * 帧头字节可能出现在压缩数据内部，解压失败时向后合并候选边界重试。
 */
export function decompressZstdFrames(buf: Buffer): string {
  const decompress = (zlib as unknown as Record<string, (b: Buffer) => Buffer>).zstdDecompressSync;
  if (typeof decompress !== 'function') {
    throw new Error('当前 Node 运行时不支持 zstd 解压（需要 Node >= 22.15），请升级 Node 或改用 jsonl 会话文件');
  }
  const offsets: number[] = [];
  let found = buf.indexOf(ZSTD_MAGIC);
  if (found === -1) throw new Error('不是有效的 zstd 数据：未找到帧头');
  while (found !== -1) {
    offsets.push(found);
    found = buf.indexOf(ZSTD_MAGIC, found + ZSTD_MAGIC.length);
  }
  offsets.push(buf.length);
  const parts: string[] = [];
  let i = 0;
  while (i < offsets.length - 1) {
    let end = i + 1;
    let decoded: string | null = null;
    while (end < offsets.length && decoded === null) {
      const slice = buf.subarray(offsets[i], offsets[end]);
      try {
        decoded = decompress(slice).toString('utf8');
      } catch {
        // 末尾可能带少量填充字节
        for (let trim = 1; trim <= 8 && decoded === null; trim += 1) {
          try {
            decoded = decompress(buf.subarray(offsets[i], offsets[end] - trim)).toString('utf8');
          } catch {
            decoded = null;
          }
        }
      }
      if (decoded === null) end += 1;
    }
    if (decoded === null) throw new Error(`zstd 帧解码失败（偏移 ${offsets[i]}）`);
    parts.push(decoded);
    i = end;
  }
  return parts.join('');
}

/* ------------------------------------------------------------------ */
/* 内容块提取                                                           */
/* ------------------------------------------------------------------ */

interface ExtractedBlock {
  kind: MessageKind;
  text: string;
  toolName?: string;
  callId?: string;
}

function stringifyArguments(value: unknown): string {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/** 容错提取消息正文：兼容 DSH 内容块、OpenAI 风格对象与纯字符串。 */
export function extractBlocks(value: unknown, defaultKind: MessageKind = 'message'): ExtractedBlock[] {
  const out: ExtractedBlock[] = [];
  const walk = (node: unknown): void => {
    if (node === null || node === undefined) return;
    if (typeof node === 'string') {
      if (node.trim()) out.push({ kind: defaultKind, text: node });
      return;
    }
    if (typeof node === 'number' || typeof node === 'boolean') return;
    if (Array.isArray(node)) {
      for (const item of node) walk(item);
      return;
    }
    const obj = node as Record<string, unknown>;
    const type = typeof obj.type === 'string' ? obj.type : undefined;
    if (type === 'text' && typeof obj.text === 'string') {
      out.push({ kind: typeof obj.text === 'string' && defaultKind === 'tool_result' ? 'tool_result' : 'message', text: obj.text });
      return;
    }
    if ((type === 'reasoning' || type === 'thinking') && typeof obj.text === 'string') {
      out.push({ kind: 'reasoning', text: obj.text });
      return;
    }
    if (type === 'tool-call' || type === 'tool_call' || type === 'tool_use') {
      const toolName = typeof obj.name === 'string' ? obj.name : typeof obj.toolName === 'string' ? obj.toolName : undefined;
      const callId = typeof obj.callId === 'string' ? obj.callId : typeof obj.id === 'string' ? obj.id : undefined;
      const args = stringifyArguments(obj.arguments ?? obj.input ?? obj.args);
      out.push({ kind: 'tool_call', text: args, toolName, callId });
      return;
    }
    if (typeof obj.text === 'string') {
      out.push({ kind: defaultKind, text: obj.text });
      return;
    }
    if (obj.content !== undefined) {
      walk(obj.content);
      return;
    }
    if (obj.message !== undefined) walk(obj.message);
  };
  walk(value);
  return out;
}

/* ------------------------------------------------------------------ */
/* 会话文件读取与解析                                                    */
/* ------------------------------------------------------------------ */

export interface SessionFileInfo {
  filePath: string;
  fileSize: number;
  fileMtimeMs: number;
  sessionId: string;
}

function detectExtension(fileName: string): string | null {
  const lower = fileName.toLowerCase();
  for (const ext of SESSION_FILE_EXTENSIONS) {
    if (lower.endsWith(ext)) return ext;
  }
  return null;
}

async function readJsonlRecords(filePath: string): Promise<{ records: unknown[]; errors: string[] }> {
  const records: unknown[] = [];
  const errors: string[] = [];
  const rl = readline.createInterface({
    input: createReadStream(filePath, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  let lineNo = 0;
  for await (const line of rl) {
    lineNo += 1;
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      records.push(JSON.parse(trimmed));
    } catch {
      if (errors.length < 5) errors.push(`${filePath}:${lineNo} JSON 解析失败`);
    }
  }
  return { records, errors };
}

/** 读取单个会话文件，返回原始记录数组（不含归一化）。 */
export async function readSessionRecords(filePath: string): Promise<{ records: unknown[]; errors: string[] }> {
  const ext = detectExtension(path.basename(filePath)) ?? path.extname(filePath).toLowerCase();
  const errors: string[] = [];
  if (ext === '.jsonl.zstd' || ext === '.jsonl.zst' || ext === '.zst' || ext === '.zstd') {
    const buf = await fs.readFile(filePath);
    return { records: parseJsonlText(decompressZstdFrames(buf), filePath, errors), errors };
  }
  if (ext === '.json') {
    const text = await fs.readFile(filePath, 'utf8');
    try {
      const parsed: unknown = JSON.parse(text);
      if (Array.isArray(parsed)) return { records: parsed, errors };
      const obj = parsed as Record<string, unknown>;
      if (Array.isArray(obj.messages)) return { records: [parsed, ...obj.messages], errors };
      return { records: [parsed], errors };
    } catch {
      // 退化为按 JSONL 解析（有些 .json 实际是每行一条）
      return { records: parseJsonlText(text, filePath, errors), errors };
    }
  }
  const { records, errors: jsonlErrors } = await readJsonlRecords(filePath);
  return { records, errors: jsonlErrors };
}

function parseJsonlText(text: string, filePath: string, errors: string[]): unknown[] {
  const records: unknown[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i].trim();
    if (!trimmed) continue;
    try {
      records.push(JSON.parse(trimmed));
    } catch {
      if (errors.length < 5) errors.push(`${filePath}:${i + 1} JSON 解析失败`);
    }
  }
  return records;
}

function toRole(value: unknown, fallback: MessageRole): MessageRole {
  return value === 'system' || value === 'user' || value === 'assistant' || value === 'tool' ? value : fallback;
}

/**
 * 归一化记录数组 → SessionRecord。
 * 识别 DSH session v4 记录类型，同时兼容 {role, content} 形式的通用消息。
 */
export function recordsToSession(
  records: unknown[],
  filePath: string,
  fileSize: number,
  fileMtimeMs: number,
  fallbackId: string,
): SessionRecord | null {
  const session: SessionRecord = {
    id: fallbackId,
    filePath,
    createdAt: 0,
    updatedAt: 0,
    fileSize,
    fileMtimeMs,
    messages: [],
  };
  const callIdToName = new Map<string, string>();
  let ordinal = 0;
  let lastTime = 0;

  const push = (
    seq: number,
    time: number,
    role: MessageRole,
    kind: MessageKind,
    text: string,
    extra: { turn?: number; step?: number; toolName?: string; callId?: string } = {},
  ): void => {
    if (!text || !text.trim()) return;
    ordinal += 1;
    session.messages.push({
      id: `${session.id}:${seq}:${ordinal}`,
      seq,
      role,
      kind,
      text,
      time,
      turn: extra.turn,
      step: extra.step,
      toolName: extra.toolName,
      callId: extra.callId,
    });
  };

  for (let index = 0; index < records.length; index += 1) {
    const raw = records[index];
    if (!raw || typeof raw !== 'object') continue;
    const record = raw as Record<string, unknown>;
    const type = typeof record.type === 'string' ? record.type : '';
    const seq = typeof record.seq === 'number' ? record.seq : index;
    const time = typeof record.time === 'number' ? record.time : lastTime;
    lastTime = time;
    if (time > session.updatedAt) session.updatedAt = time;
    const data = (record.data ?? {}) as Record<string, unknown>;

    switch (type) {
      case 'session': {
        if (typeof record.id === 'string') session.id = record.id;
        if (typeof record.cwd === 'string') session.cwd = record.cwd;
        if (typeof record.createdAt === 'number') session.createdAt = record.createdAt;
        if (typeof record.origin === 'string') session.origin = record.origin;
        if (typeof record.parentSession === 'string') session.parentSession = record.parentSession;
        if (typeof record.agentPreset === 'string') session.agentPreset = record.agentPreset;
        if (typeof record.delegationDepth === 'number') session.delegationDepth = record.delegationDepth;
        break;
      }
      case 'session/title': {
        if (typeof data.title === 'string') {
          session.title = data.title;
          push(seq, time, 'system', 'title', data.title);
        }
        break;
      }
      case 'user/message': {
        const message = (data.message ?? data) as Record<string, unknown>;
        const blocks = extractBlocks(message.content ?? message.text, 'message');
        for (const block of blocks) {
          push(seq, time, 'user', 'message', block.text, {
            turn: typeof data.turn === 'number' ? data.turn : undefined,
            step: typeof data.step === 'number' ? data.step : undefined,
          });
        }
        break;
      }
      case 'assistant/message': {
        const message = (data.message ?? data) as Record<string, unknown>;
        const blocks = extractBlocks(message.content, 'message');
        for (const block of blocks) {
          const callId = block.callId;
          if (block.kind === 'tool_call' && callId && block.toolName) callIdToName.set(callId, block.toolName);
          push(seq, time, 'assistant', block.kind, block.text, {
            turn: typeof data.turn === 'number' ? data.turn : undefined,
            step: typeof data.step === 'number' ? data.step : undefined,
            toolName: block.toolName,
            callId,
          });
        }
        break;
      }
      case 'system/message': {
        const message = (data.message ?? data) as Record<string, unknown>;
        for (const block of extractBlocks(message.content ?? message.text, 'message')) {
          push(seq, time, 'system', block.kind, block.text);
        }
        break;
      }
      case 'tool/call': {
        const toolName = typeof data.name === 'string' ? data.name : undefined;
        const callId = typeof data.callId === 'string' ? data.callId : undefined;
        if (callId && toolName) callIdToName.set(callId, toolName);
        const args = stringifyArguments(data.arguments ?? data.input);
        const text = args ? `${toolName ?? 'tool'}\n${args}` : `${toolName ?? 'tool'}`;
        push(seq, time, 'assistant', 'tool_call', text, {
          turn: typeof data.turn === 'number' ? data.turn : undefined,
          step: typeof data.step === 'number' ? data.step : undefined,
          toolName,
          callId,
        });
        break;
      }
      case 'tool/result': {
        const message = (data.message ?? data) as Record<string, unknown>;
        const callId =
          typeof data.callId === 'string'
            ? data.callId
            : typeof (message as Record<string, unknown>).toolCallId === 'string'
              ? ((message as Record<string, unknown>).toolCallId as string)
              : undefined;
        const toolName = callId ? callIdToName.get(callId) : undefined;
        const blocks = extractBlocks(message.content ?? message.text ?? data.content, 'tool_result');
        for (const block of blocks) {
          push(seq, time, 'tool', 'tool_result', block.text, {
            turn: typeof data.turn === 'number' ? data.turn : undefined,
            step: typeof data.step === 'number' ? data.step : undefined,
            toolName,
            callId,
          });
        }
        break;
      }
      case 'compaction/summary': {
        for (const block of extractBlocks(data.summary, 'summary')) {
          push(seq, time, 'system', 'summary', block.text);
        }
        break;
      }
      case 'agent/inbox/spliced': {
        const inserted = data.inserted;
        if (Array.isArray(inserted)) {
          for (const item of inserted) {
            const message = (item ?? {}) as Record<string, unknown>;
            for (const block of extractBlocks(message.content ?? message.text, 'message')) {
              push(seq, time, 'user', 'message', block.text);
            }
          }
        }
        break;
      }
      default: {
        // 通用兜底：{role, content} 形式或带 message/content 的未知记录
        if (typeof record.role === 'string') {
          for (const block of extractBlocks(record.content ?? record.text, 'message')) {
            push(seq, time, toRole(record.role, 'user'), block.kind, block.text);
          }
          break;
        }
        const candidate = (data.message ?? data) as Record<string, unknown>;
        if (candidate && (candidate.content !== undefined || typeof candidate.text === 'string')) {
          const blocks = extractBlocks(candidate.content ?? candidate.text, 'message');
          if (blocks.length > 0 && data.message !== undefined) {
            const role = toRole(candidate.role, 'system');
            for (const block of blocks) push(seq, time, role, block.kind, block.text);
          }
        }
        break;
      }
    }
  }

  if (session.messages.length === 0) return null;
  if (session.createdAt === 0) {
    const first = session.messages[0];
    session.createdAt = first.time || fileMtimeMs;
  }
  if (session.updatedAt === 0) session.updatedAt = session.createdAt;
  return session;
}

/** 递归扫描 sessions 根目录，返回候选会话文件（同目录内按扩展名优先级去重）。 */
export async function scanSessionFiles(sessionsRoot: string, maxDepth = 3): Promise<SessionFileInfo[]> {
  const results: SessionFileInfo[] = [];
  const walk = async (dir: string, depth: number): Promise<void> => {
    let entries: Dirent[];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    const byBaseName = new Map<string, { filePath: string; priority: number }>();
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (depth < maxDepth) await walk(full, depth + 1);
        continue;
      }
      const ext = detectExtension(entry.name);
      if (!ext) continue;
      // 同一目录下同名文件只保留优先级最高的扩展名（如 session.v4.jsonl.zstd 覆盖 session.v4.jsonl）
      const baseName = entry.name.slice(0, entry.name.length - ext.length);
      const priority = SESSION_FILE_EXTENSIONS.indexOf(ext);
      const current = byBaseName.get(baseName);
      if (!current || priority < current.priority) byBaseName.set(baseName, { filePath: full, priority });
    }
    for (const [baseName, best] of byBaseName) {
      try {
        const stat = await fs.stat(best.filePath);
        // session.v4.jsonl.zstd 的 baseName 为 session.v4，需剥掉版本号后回退到目录名（DSH 用会话 id 作目录名）
        const stripped = baseName.replace(/\.v\d+$/i, '');
        const sessionId = stripped && stripped !== 'session' ? stripped : path.basename(dir);
        results.push({
          filePath: best.filePath,
          fileSize: stat.size,
          fileMtimeMs: stat.mtimeMs,
          sessionId,
        });
      } catch {
        // 文件在扫描过程中消失，忽略
      }
    }
  };
  await walk(sessionsRoot, 0);
  return results;
}

/* ------------------------------------------------------------------ */
/* 索引协调器                                                           */
/* ------------------------------------------------------------------ */

/** 协作式取消：signal 已中止时抛出 AbortError。 */
export function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  const error = new Error('archive index scan aborted');
  error.name = 'AbortError';
  throw error;
}

export interface ArchiveIndexerOptions {
  /** 会话根目录，默认 ~/.dsh/sessions（或 $DSH_HOME/sessions）。 */
  sessionsRoot?: string;
  /** 索引镜像目录，默认 ~/.dsh/archive-search（或 $DSH_HOME/archive-search）。 */
  indexDir?: string;
  /** 直接指定 meta 镜像文件路径（宿主侧 cache_path）；给出时其所在目录作为索引目录。 */
  cachePath?: string;
  /** 哈希投影维度，默认 4096。 */
  dimensions?: number;
  /** 外部 Embedding 适配器；注入后语义检索走外部向量。 */
  provider?: EmbeddingProvider | null;
  /** 单条消息参与向量化的最大字符数，默认 8000。 */
  maxDocumentChars?: number;
  /** 会话文件扫描深度，默认 3。 */
  maxDepth?: number;
  /** 协作式取消：每个会话文件前检查一次。 */
  signal?: AbortSignal;
  logger?: LogFn;
}

interface PersistedMeta {
  version: number;
  generatedAt: number;
  sessionsRoot: string;
  dimensions: number;
  providerId: string | null;
  files: Array<{ path: string; size: number; mtimeMs: number; sessionId: string }>;
}

/**
 * 索引协调器：
 * - `fullIndex()` 全量解析并重建向量索引；
 * - `incrementalUpdate()` 仅处理新增/变更/删除的文件；
 * - `ensureIndexed()` 读取持久化镜像后做增量补齐（插件启动入口）；
 * - `search()` 按 mode 路由到 Grep 或向量引擎。
 */
export class ArchiveIndexer {
  readonly sessionsRoot: string;
  readonly indexDir: string;
  readonly metaPath: string;
  readonly vectorPath: string;
  readonly vectorIndex: VectorIndex;
  private readonly grep = new GrepEngine();
  private readonly logger?: LogFn;
  private readonly maxDepth: number;
  private readonly signal?: AbortSignal;
  private sessions = new Map<string, SessionRecord>();
  /** messageId → 消息，用于给不含正文的向量镜像补齐文本。 */
  private messagesById = new Map<string, MessageItem>();
  private fileStats = new Map<string, SessionFileInfo>();
  private errors: string[] = [];
  private lastIndexedAt = 0;
  private filesParsed = 0;
  private filesSkipped = 0;
  private stateLoaded = false;
  private refreshed = false;
  private vectorDirty = false;

  constructor(options: ArchiveIndexerOptions = {}) {
    this.sessionsRoot = options.sessionsRoot ?? resolveSessionsRoot();
    this.indexDir =
      options.indexDir ?? (options.cachePath ? path.dirname(options.cachePath) : resolveIndexDir());
    this.metaPath = options.cachePath ?? path.join(this.indexDir, META_FILE_NAME);
    this.vectorPath = path.join(this.indexDir, VECTOR_FILE_NAME);
    this.logger = options.logger;
    this.maxDepth = options.maxDepth ?? 3;
    this.signal = options.signal;
    this.vectorIndex = new VectorIndex({
      dimensions: options.dimensions,
      provider: options.provider ?? null,
      maxDocumentChars: options.maxDocumentChars,
    });
  }

  get indexed(): boolean {
    return this.refreshed;
  }

  listSessions(): SessionRecord[] {
    return Array.from(this.sessions.values());
  }

  getSession(sessionId: string): SessionRecord | undefined {
    return this.sessions.get(sessionId);
  }

  /** 全量索引：清空内存与向量索引后重新解析所有会话文件。 */
  async fullIndex(): Promise<IndexStats> {
    this.sessions.clear();
    this.messagesById.clear();
    this.fileStats.clear();
    this.vectorIndex.clear();
    this.vectorDirty = true;
    this.stateLoaded = true;
    await this.refresh();
    await this.save();
    return this.stats();
  }

  /** 增量更新：只解析新增/变更文件，移除已删除会话，其余复用内存与镜像。 */
  async incrementalUpdate(): Promise<IndexStats> {
    await this.refresh();
    if (!this.stateLoaded) {
      this.stateLoaded = true;
      await this.load();
    }
    await this.save();
    return this.stats();
  }

  /** 启动入口：先读镜像，再做增量补齐。 */
  async ensureIndexed(): Promise<IndexStats> {
    if (!this.stateLoaded) {
      await this.load();
      this.stateLoaded = true;
    }
    await this.refresh();
    await this.save();
    return this.stats();
  }

  private async refresh(): Promise<void> {
    const startedAt = Date.now();
    this.errors = [];
    this.filesParsed = 0;
    this.filesSkipped = 0;
    throwIfAborted(this.signal);
    const files = await scanSessionFiles(this.sessionsRoot, this.maxDepth);
    const seen = new Set<string>();
    let vectorDirty = false;

    for (const file of files) {
      throwIfAborted(this.signal);
      seen.add(file.filePath);
      const prev = this.fileStats.get(file.filePath);
      const unchanged = Boolean(
        prev && prev.fileSize === file.fileSize && prev.fileMtimeMs === file.fileMtimeMs,
      );
      const reused = unchanged ? this.sessions.get(prev!.sessionId) : undefined;

      let session = reused;
      if (session) {
        this.filesSkipped += 1;
      } else {
        session = await this.parseFile(file) ?? undefined;
        this.filesParsed += 1;
        if (!session) {
          if (prev) {
            this.sessions.delete(prev.sessionId);
            if (this.vectorIndex.removeSession(prev.sessionId) > 0) vectorDirty = true;
            this.fileStats.delete(file.filePath);
          }
          continue;
        }
        if (prev && prev.sessionId !== session.id) {
          this.sessions.delete(prev.sessionId);
          if (this.vectorIndex.removeSession(prev.sessionId) > 0) vectorDirty = true;
        }
        this.sessions.set(session.id, session);
      }

      this.fileStats.set(file.filePath, {
        filePath: file.filePath,
        fileSize: file.fileSize,
        fileMtimeMs: file.fileMtimeMs,
        sessionId: session.id,
      });

      if (!this.vectorIndex.hasSession(session.id)) {
        if (this.vectorIndex.removeSession(session.id) > 0) vectorDirty = true;
        const docs = this.buildDocuments(session);
        if (docs.length > 0) {
          this.vectorIndex.addDocuments(docs);
          vectorDirty = true;
        }
      }
    }

    for (const [filePath, info] of Array.from(this.fileStats.entries())) {
      if (seen.has(filePath)) continue;
      this.fileStats.delete(filePath);
      this.sessions.delete(info.sessionId);
      if (this.vectorIndex.removeSession(info.sessionId) > 0) vectorDirty = true;
      this.logger?.(`[archive-search] 移除已删除会话 ${info.sessionId}`);
    }

    this.lastIndexedAt = Date.now();
    this.refreshed = true;
    this.vectorDirty = this.vectorDirty || vectorDirty;
    this.messagesById.clear();
    for (const session of this.sessions.values()) {
      for (const message of session.messages) this.messagesById.set(message.id, message);
    }
    this.logger?.(
      `[archive-search] 索引完成：${this.sessions.size} 会话 / ${this.vectorIndex.size} 文档，` +
        `解析 ${this.filesParsed} 文件，复用 ${this.filesSkipped} 文件，耗时 ${Date.now() - startedAt}ms`,
    );
  }

  private async parseFile(file: SessionFileInfo): Promise<SessionRecord | null> {
    try {
      const { records, errors } = await readSessionRecords(file.filePath);
      for (const error of errors) if (this.errors.length < 20) this.errors.push(error);
      return recordsToSession(records, file.filePath, file.fileSize, file.fileMtimeMs, file.sessionId);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (this.errors.length < 20) this.errors.push(`${file.filePath}: ${message}`);
      this.logger?.(`[archive-search] 解析失败 ${file.filePath}: ${message}`);
      return null;
    }
  }

  private buildDocuments(session: SessionRecord): VectorDocument[] {
    const docs: VectorDocument[] = [];
    for (const message of session.messages) {
      if (!message.text || !message.text.trim()) continue;
      docs.push({
        id: message.id,
        sessionId: session.id,
        seq: message.seq,
        text: message.text,
        role: message.role,
        kind: message.kind,
        time: message.time,
        filePath: session.filePath,
        toolName: message.toolName,
      });
    }
    return docs;
  }

  /**
   * 写入 meta 与向量镜像。
   * 向量镜像不含正文（正文由会话原文补齐），且仅在向量变化时落盘。
   */
  async save(): Promise<void> {
    const meta: PersistedMeta = {
      version: INDEX_FORMAT_VERSION,
      generatedAt: this.lastIndexedAt,
      sessionsRoot: this.sessionsRoot,
      dimensions: this.vectorIndex.dimensions,
      providerId: this.vectorIndex.provider?.id ?? null,
      files: Array.from(this.fileStats.values()).map((info) => ({
        path: info.filePath,
        size: info.fileSize,
        mtimeMs: info.fileMtimeMs,
        sessionId: info.sessionId,
      })),
    };
    await fs.mkdir(this.indexDir, { recursive: true });
    const tmp = `${this.metaPath}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(meta), 'utf8');
    await fs.rename(tmp, this.metaPath);
    if (this.vectorDirty) {
      await this.vectorIndex.saveToFile(this.vectorPath, { includeText: false });
      this.vectorDirty = false;
    }
  }

  /** 读取 meta 与向量镜像；缺失或版本不符时保持空索引并返回 false。 */
  async load(): Promise<boolean> {
    let meta: PersistedMeta | null = null;
    try {
      meta = JSON.parse(await fs.readFile(this.metaPath, 'utf8')) as PersistedMeta;
    } catch {
      return false;
    }
    if (!meta || meta.version !== INDEX_FORMAT_VERSION) return false;
    if (meta.sessionsRoot !== this.sessionsRoot) return false;
    this.fileStats.clear();
    for (const file of meta.files ?? []) {
      this.fileStats.set(file.path, {
        filePath: file.path,
        fileSize: file.size,
        fileMtimeMs: file.mtimeMs,
        sessionId: file.sessionId,
      });
    }
    this.lastIndexedAt = meta.generatedAt ?? 0;
    const loaded = await this.vectorIndex.loadFromFile(this.vectorPath);
    if (loaded && meta.providerId !== (this.vectorIndex.provider?.id ?? null)) {
      // provider 变更后镜像向量不可复用，重建 TF-IDF 索引
      this.vectorIndex.clear();
      return false;
    }
    return loaded;
  }

  /** 检索入口；未索引时自动执行 ensureIndexed。 */
  async search(options: SearchOptions): Promise<SearchResponse> {
    if (!this.refreshed) await this.ensureIndexed();
    const opts = normalizeSearchOptions(options);
    if (opts.mode === 'semantic') return this.semanticSearch(opts);
    return this.grep.search(this.listSessions(), opts);
  }

  private async semanticSearch(opts: ReturnType<typeof normalizeSearchOptions>): Promise<SearchResponse> {
    const startedAt = Date.now();
    const limit = opts.offset + opts.limit;
    const filter = this.buildVectorFilter(opts);
    const queryOptions = { limit, minScore: opts.minScore, filter };
    const { hits, total } = this.vectorIndex.provider
      ? await this.vectorIndex.searchAsyncWithTotal(opts.query, queryOptions)
      : this.vectorIndex.searchWithTotal(opts.query, queryOptions);
    const page = hits.slice(opts.offset, opts.offset + opts.limit);
    const results = page.map((hit) => this.toSearchResult(hit, opts));
    return {
      query: opts.query,
      mode: 'semantic',
      results,
      total,
      returned: results.length,
      offset: opts.offset,
      tookMs: Date.now() - startedAt,
      scannedSessions: this.sessions.size,
      scannedMessages: this.vectorIndex.size,
    };
  }

  private buildVectorFilter(
    opts: ReturnType<typeof normalizeSearchOptions>,
  ): (doc: VectorDocument) => boolean {
    return (doc) => {
      if (opts.roles?.length && (!doc.role || !opts.roles.includes(doc.role))) return false;
      if (opts.kinds?.length && (!doc.kind || !opts.kinds.includes(doc.kind))) return false;
      if (opts.sessionIds?.length && !opts.sessionIds.includes(doc.sessionId)) return false;
      if (opts.cwdPrefix && !(this.sessions.get(doc.sessionId)?.cwd ?? '').startsWith(opts.cwdPrefix)) {
        return false;
      }
      const time = doc.time ?? 0;
      if (opts.since !== undefined && time < opts.since) return false;
      if (opts.until !== undefined && time > opts.until) return false;
      return true;
    };
  }

  /** 向量命中 → 统一结果项：正文能被关键词定位时复用 Grep 摘要与高亮，否则取首行。 */
  private toSearchResult(
    hit: { score: number; doc: VectorDocument },
    opts: ReturnType<typeof normalizeSearchOptions>,
  ): SearchResultItem {
    const doc = hit.doc;
    const session = this.sessions.get(doc.sessionId);
    // 镜像不保存正文，正文以内存中的会话原文为准
    const text = this.messagesById.get(doc.id)?.text ?? doc.text;
    const plan = compileGrepPlan(opts.query, { caseSensitive: false, regex: false });
    const grep =
      plan.terms.length > 0
        ? grepText(text, plan, {
            contextLines: opts.contextLines,
            maxMatchesPerMessage: opts.maxMatchesPerMessage,
          })
        : { matches: [] as MatchedLine[], hitCount: 0, matchedTerms: 0 };

    let snippet: string;
    let ranges: HighlightRange[] = [];
    let matches: MatchedLine[] = grep.matches;
    if (grep.matches.length > 0) {
      const built = buildSnippet(grep.matches[0].text, grep.matches[0].ranges, opts.snippetLength);
      snippet = built.text;
      ranges = built.ranges;
    } else {
      const firstLine = text.split(/\r?\n/).find((line) => line.trim().length > 0) ?? '';
      const built = buildSnippet(truncateLine(firstLine), [], opts.snippetLength);
      snippet = built.text;
      matches = [{ line: 1, text: built.text, before: [], after: [], ranges: [] }];
    }

    return {
      sessionId: doc.sessionId,
      filePath: doc.filePath ?? session?.filePath ?? '',
      title: session?.title,
      cwd: session?.cwd,
      messageId: doc.id,
      seq: doc.seq,
      role: doc.role ?? 'user',
      kind: doc.kind ?? 'message',
      time: doc.time ?? 0,
      score: Number(hit.score.toFixed(6)),
      snippet,
      ranges,
      matches,
      source: 'vector',
      toolName: doc.toolName,
    };
  }

  stats(): IndexStats {
    let messages = 0;
    for (const session of this.sessions.values()) messages += session.messages.length;
    const vectorStats = this.vectorIndex.stats();
    return {
      sessions: this.sessions.size,
      messages,
      indexedDocuments: vectorStats.documents,
      vectorDimensions: vectorStats.dimensions,
      vectorFeatures: vectorStats.features,
      sessionsRoot: this.sessionsRoot,
      indexDir: this.indexDir,
      filesScanned: this.fileStats.size,
      filesParsed: this.filesParsed,
      filesSkipped: this.filesSkipped,
      lastIndexedAt: this.lastIndexedAt,
      errors: this.errors.slice(0, 20),
    };
  }

  /** 预留：释放 provider 侧资源。 */
  close(): void {
    // 纯内存实现无需释放；保留给外部 Embedding provider 的未来清理逻辑
  }
}

/** 一次性检索便捷入口：建索引 → 检索 → 返回结果。 */
export async function searchArchive(
  options: SearchOptions,
  indexerOptions: ArchiveIndexerOptions = {},
): Promise<SearchResponse> {
  const indexer = new ArchiveIndexer(indexerOptions);
  await indexer.ensureIndexed();
  const response = await indexer.search(options);
  indexer.close();
  return response;
}
