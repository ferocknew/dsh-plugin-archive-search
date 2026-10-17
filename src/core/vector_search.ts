/**
 * 纯 TypeScript 向量检索引擎：零 SQLite、零原生 C++ 依赖。
 *
 * 组成：
 * 1. 中英文双字符 N-gram + 词元 特征提取（CJK 取相邻双字，ASCII 取整词 + 字符二元组）；
 * 2. FNV-1a 哈希投影（带符号，降低冲突偏置）到固定维度；
 * 3. TF-IDF 加权 + L2 归一化，输出 Float32Array 稠密向量；
 * 4. 余弦相似度 + 最小堆 Top-K；
 * 5. 内存索引 + JSON 镜像持久化；
 * 6. 预留 EmbeddingProvider 接口，注入后切换为外部嵌入向量。
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { EmbeddingProvider, MessageKind, MessageRole } from './types';

export const DEFAULT_DIMENSIONS = 4096;
export const DEFAULT_MAX_FEATURES = 4000;
export const DEFAULT_MAX_DOCUMENT_CHARS = 8000;
export const VECTOR_INDEX_FORMAT_VERSION = 1;

/* ------------------------------------------------------------------ */
/* 哈希与特征提取                                                       */
/* ------------------------------------------------------------------ */

/** FNV-1a 32 位哈希；seed 可变，用于同特征取不同投影。 */
export function hash32(input: string, seed = 0x811c9dc5): number {
  let h = seed >>> 0;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function isCjkCodePoint(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) || // CJK 统一表意
    (cp >= 0x3400 && cp <= 0x4dbf) || // 扩展 A
    (cp >= 0xf900 && cp <= 0xfaff) || // 兼容表意
    (cp >= 0x3040 && cp <= 0x30ff) || // 日文假名
    (cp >= 0xac00 && cp <= 0xd7af) || // 韩文
    (cp >= 0x20000 && cp <= 0x2ffff) // 扩展 B+
  );
}

function isWordCodePoint(cp: number): boolean {
  return (
    (cp >= 0x30 && cp <= 0x39) || // 0-9
    (cp >= 0x41 && cp <= 0x5a) || // A-Z
    (cp >= 0x61 && cp <= 0x7a) || // a-z
    cp === 0x5f // _
  );
}

/**
 * 提取原始特征串：
 * - CJK 连续串 → 双字 N-gram（`c:`），单字串退化为 `c:字`，长度 ≤ 8 时额外加整串短语特征（`C:`）；
 * - ASCII 词元 → 整词（`w:`），长度 ≥ 3 时再加字符双字组（`b:`）。
 */
export function tokenizeFeatures(text: string, maxFeatures = DEFAULT_MAX_FEATURES): string[] {
  const features: string[] = [];
  let run = '';
  let runKind: 'cjk' | 'word' | null = null;

  const flush = (): void => {
    if (!run) return;
    if (runKind === 'cjk') {
      const chars = Array.from(run);
      if (chars.length === 1) {
        features.push(`c:${chars[0]}`);
      } else {
        for (let i = 0; i + 1 < chars.length; i += 1) {
          features.push(`c:${chars[i]}${chars[i + 1]}`);
        }
      }
      if (chars.length <= 8 && chars.length > 1) features.push(`C:${run}`);
    } else if (runKind === 'word') {
      const word = run.toLowerCase();
      features.push(`w:${word}`);
      if (word.length >= 3) {
        for (let i = 0; i + 1 < word.length; i += 1) {
          features.push(`b:${word.slice(i, i + 2)}`);
        }
      }
    }
    run = '';
    runKind = null;
  };

  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    const kind: 'cjk' | 'word' | null = isCjkCodePoint(cp)
      ? 'cjk'
      : isWordCodePoint(cp)
        ? 'word'
        : null;
    if (kind === null) {
      flush();
      continue;
    }
    if (kind !== runKind) flush();
    run += ch;
    runKind = kind;
    if (features.length >= maxFeatures) {
      run = '';
      runKind = null;
      break;
    }
  }
  flush();
  return features.length > maxFeatures ? features.slice(0, maxFeatures) : features;
}

/** 把特征串投影为稀疏向量：索引 = hash % dims，符号由第二哈希决定。 */
export function projectFeature(feature: string, dimensions: number): { index: number; sign: number } {
  const index = hash32(feature) % dimensions;
  const sign = (hash32(feature, 0x9e3779b1) & 1) === 0 ? 1 : -1;
  return { index, sign };
}

/**
 * 文本 → 稀疏词频表（index → 带符号计数）。
 * 哈希冲突会把不同特征累加到同一维，符号可部分抵消冲突带来的相似度虚高。
 */
export function buildTermFrequency(
  text: string,
  dimensions: number,
  maxFeatures = DEFAULT_MAX_FEATURES,
): Map<number, number> {
  const tf = new Map<number, number>();
  for (const feature of tokenizeFeatures(text, maxFeatures)) {
    const { index, sign } = projectFeature(feature, dimensions);
    const next = (tf.get(index) ?? 0) + sign;
    if (next === 0) tf.delete(index);
    else tf.set(index, next);
  }
  return tf;
}

/* ------------------------------------------------------------------ */
/* 向量数学                                                             */
/* ------------------------------------------------------------------ */

export function dotProduct(a: Float32Array, b: Float32Array): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i += 1) sum += a[i] * b[i];
  return sum;
}

export function l2Norm(v: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i += 1) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

/** 原地或副本 L2 归一化（零向量原样返回）。 */
export function normalizeVector(v: Float32Array, inPlace = false): Float32Array {
  const out = inPlace ? v : Float32Array.from(v);
  const norm = l2Norm(out);
  if (norm > 0) {
    for (let i = 0; i < out.length; i += 1) out[i] /= norm;
  }
  return out;
}

/** 通用余弦相似度；输入无需预先归一化。 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const dot = dotProduct(a, b);
  if (dot === 0) return 0;
  const denom = l2Norm(a) * l2Norm(b);
  return denom === 0 ? 0 : dot / denom;
}

/* ------------------------------------------------------------------ */
/* Top-K 最小堆                                                         */
/* ------------------------------------------------------------------ */

/** 固定容量的 Top-K 最小堆：堆顶为当前 K 个元素中的最小得分。 */
export class TopKHeap<T> {
  private heap: T[] = [];

  constructor(
    readonly capacity: number,
    private readonly scoreOf: (value: T) => number,
  ) {}

  get size(): number {
    return this.heap.length;
  }

  push(value: T): void {
    if (this.capacity <= 0) return;
    if (this.heap.length < this.capacity) {
      this.heap.push(value);
      this.siftUp(this.heap.length - 1);
      return;
    }
    if (this.scoreOf(value) > this.scoreOf(this.heap[0])) {
      this.heap[0] = value;
      this.siftDown(0);
    }
  }

  /** 降序返回（得分相同按插入顺序稳定）。 */
  toSorted(): T[] {
    return this.heap
      .map((value, index) => ({ value, index }))
      .sort((a, b) => this.scoreOf(b.value) - this.scoreOf(a.value) || a.index - b.index)
      .map((entry) => entry.value);
  }

  private siftUp(index: number): void {
    let i = index;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.scoreOf(this.heap[parent]) <= this.scoreOf(this.heap[i])) break;
      [this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
      i = parent;
    }
  }

  private siftDown(index: number): void {
    let i = index;
    const n = this.heap.length;
    for (;;) {
      const left = i * 2 + 1;
      const right = left + 1;
      let smallest = i;
      if (left < n && this.scoreOf(this.heap[left]) < this.scoreOf(this.heap[smallest])) smallest = left;
      if (right < n && this.scoreOf(this.heap[right]) < this.scoreOf(this.heap[smallest])) smallest = right;
      if (smallest === i) break;
      [this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
      i = smallest;
    }
  }
}

/** 便捷函数：对 {score, value} 列表取 Top-K，降序。 */
export function topKByScore<T>(items: Array<{ score: number; value: T }>, k: number): Array<{ score: number; value: T }> {
  const heap = new TopKHeap<{ score: number; value: T }>(k, (entry) => entry.score);
  for (const item of items) heap.push(item);
  return heap.toSorted();
}

/* ------------------------------------------------------------------ */
/* TF-IDF 向量化器                                                      */
/* ------------------------------------------------------------------ */

/** TF-IDF：df 按“哈希后维度”统计（与稀疏向量维度一致），文档增删会推进 version。 */
export class TfidfVectorizer {
  private df = new Map<number, number>();
  private docCount = 0;
  private versionValue = 0;

  constructor(readonly dimensions: number = DEFAULT_DIMENSIONS) {}

  get documentCount(): number {
    return this.docCount;
  }

  get featureCount(): number {
    return this.df.size;
  }

  /** 文档或 IDF 统计变化时递增，向量缓存据此失效。 */
  get version(): number {
    return this.versionValue;
  }

  addDocument(tf: Map<number, number>): void {
    for (const index of tf.keys()) this.df.set(index, (this.df.get(index) ?? 0) + 1);
    this.docCount += 1;
    this.versionValue += 1;
  }

  removeDocument(tf: Map<number, number>): void {
    for (const index of tf.keys()) {
      const next = (this.df.get(index) ?? 0) - 1;
      if (next <= 0) this.df.delete(index);
      else this.df.set(index, next);
    }
    this.docCount = Math.max(0, this.docCount - 1);
    this.versionValue += 1;
  }

  idf(index: number): number {
    const df = this.df.get(index) ?? 0;
    return Math.log((this.docCount + 1) / (df + 1)) + 1;
  }

  /** 稀疏词频 → TF-IDF 加权并 L2 归一化的稠密向量。 */
  transform(tf: Map<number, number>): Float32Array {
    const vec = new Float32Array(this.dimensions);
    for (const [index, value] of tf) {
      if (index < 0 || index >= this.dimensions || value === 0) continue;
      vec[index] += value * this.idf(index);
    }
    return normalizeVector(vec, true);
  }

  exportDf(): Array<[number, number]> {
    return Array.from(this.df.entries());
  }

  importDf(entries: Array<[number, number]>, docCount: number): void {
    this.df = new Map(entries);
    this.docCount = docCount;
    this.versionValue += 1;
  }

  clear(): void {
    this.df.clear();
    this.docCount = 0;
    this.versionValue += 1;
  }
}

/* ------------------------------------------------------------------ */
/* 向量索引                                                             */
/* ------------------------------------------------------------------ */

export interface VectorDocument {
  id: string;
  sessionId: string;
  seq: number;
  text: string;
  role?: MessageRole;
  kind?: MessageKind;
  time?: number;
  filePath?: string;
  toolName?: string;
}

export interface VectorHit {
  id: string;
  score: number;
  doc: VectorDocument;
}

export interface VectorQueryOptions {
  limit?: number;
  minScore?: number;
  /** 参与排序前的过滤（如按角色/会话过滤），避免 Top-K 被过滤项挤占。 */
  filter?: (doc: VectorDocument) => boolean;
}

export interface VectorSearchResult {
  hits: VectorHit[];
  /** 通过过滤且得分 > 0 的文档总数（不受 limit 影响）。 */
  total: number;
}

export interface VectorIndexOptions {
  dimensions?: number;
  provider?: EmbeddingProvider | null;
  maxFeaturesPerDocument?: number;
  maxDocumentChars?: number;
}

export interface SaveVectorIndexOptions {
  /**
   * 是否把文档正文写进镜像，默认 true。
   * 置 false 时镜像只保留词频与元信息（体积大幅缩小）；检索仍可用，
   * 但正文需由上层（如 ArchiveIndexer 用会话原文）补齐后再渲染摘要。
   */
  includeText?: boolean;
}

interface InternalDocument {
  doc: VectorDocument;
  tf: Map<number, number>;
  /** TF-IDF/native 向量的生成版本，与 vectorizer.version 不一致时需重算。 */
  vecVersion: number;
  vec?: Float32Array;
  /** 外部 provider 生成的向量。 */
  providerVec?: Float32Array;
}

interface PersistedVectorIndex {
  version: number;
  dimensions: number;
  providerId: string | null;
  documentCount: number;
  df: Array<[number, number]>;
  documents: Array<{
    id: string;
    sessionId: string;
    seq: number;
    role?: MessageRole;
    kind?: MessageKind;
    time?: number;
    filePath?: string;
    toolName?: string;
    text: string;
    tf: Array<[number, number]>;
    vec?: string;
  }>;
}

function encodeVector(vec: Float32Array): string {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).toString('base64');
}

function decodeVector(encoded: string, dimensions: number): Float32Array {
  const buf = Buffer.from(encoded, 'base64');
  const copy = new Uint8Array(buf.byteLength);
  copy.set(buf);
  const vec = new Float32Array(copy.buffer, 0, Math.floor(buf.byteLength / 4));
  if (vec.length === dimensions) return vec;
  const resized = new Float32Array(dimensions);
  resized.set(vec.subarray(0, Math.min(vec.length, dimensions)));
  return resized;
}

/**
 * 内存向量索引：
 * - 默认走内置 TF-IDF（同步 search）；
 * - 注入 EmbeddingProvider 后走外部嵌入（用 await searchAsync）。
 */
export class VectorIndex {
  readonly vectorizer: TfidfVectorizer;
  readonly dimensions: number;
  readonly maxFeaturesPerDocument: number;
  readonly maxDocumentChars: number;
  provider: EmbeddingProvider | null;

  private documents = new Map<string, InternalDocument>();
  private sessionDocs = new Map<string, Set<string>>();
  private pendingEmbedding = new Set<string>();

  constructor(options: VectorIndexOptions = {}) {
    this.dimensions = options.dimensions ?? options.provider?.dimensions ?? DEFAULT_DIMENSIONS;
    this.vectorizer = new TfidfVectorizer(this.dimensions);
    this.provider = options.provider ?? null;
    this.maxFeaturesPerDocument = options.maxFeaturesPerDocument ?? DEFAULT_MAX_FEATURES;
    this.maxDocumentChars = options.maxDocumentChars ?? DEFAULT_MAX_DOCUMENT_CHARS;
  }

  get size(): number {
    return this.documents.size;
  }

  addDocument(doc: VectorDocument): void {
    const existing = this.documents.get(doc.id);
    if (existing) this.removeDocument(doc.id);
    const text = doc.text.length > this.maxDocumentChars ? doc.text.slice(0, this.maxDocumentChars) : doc.text;
    const stored: VectorDocument = { ...doc, text };
    const tf = buildTermFrequency(text, this.dimensions, this.maxFeaturesPerDocument);
    const entry: InternalDocument = { doc: stored, tf, vecVersion: -1 };
    this.documents.set(stored.id, entry);
    this.vectorizer.addDocument(tf);
    let bucket = this.sessionDocs.get(stored.sessionId);
    if (!bucket) {
      bucket = new Set();
      this.sessionDocs.set(stored.sessionId, bucket);
    }
    bucket.add(stored.id);
    if (this.provider) this.pendingEmbedding.add(stored.id);
  }

  addDocuments(docs: VectorDocument[]): void {
    for (const doc of docs) this.addDocument(doc);
  }

  removeDocument(id: string): boolean {
    const entry = this.documents.get(id);
    if (!entry) return false;
    this.vectorizer.removeDocument(entry.tf);
    this.documents.delete(id);
    this.pendingEmbedding.delete(id);
    const bucket = this.sessionDocs.get(entry.doc.sessionId);
    if (bucket) {
      bucket.delete(id);
      if (bucket.size === 0) this.sessionDocs.delete(entry.doc.sessionId);
    }
    return true;
  }

  removeSession(sessionId: string): number {
    const bucket = this.sessionDocs.get(sessionId);
    if (!bucket) return 0;
    let removed = 0;
    for (const id of Array.from(bucket)) {
      if (this.removeDocument(id)) removed += 1;
    }
    return removed;
  }

  hasDocument(id: string): boolean {
    return this.documents.has(id);
  }

  hasSession(sessionId: string): boolean {
    return this.sessionDocs.has(sessionId);
  }

  getDocument(id: string): VectorDocument | undefined {
    return this.documents.get(id)?.doc;
  }

  clear(): void {
    this.documents.clear();
    this.sessionDocs.clear();
    this.pendingEmbedding.clear();
    this.vectorizer.clear();
  }

  stats(): { documents: number; dimensions: number; features: number; providerId: string | null; pendingEmbeddings: number } {
    return {
      documents: this.documents.size,
      dimensions: this.dimensions,
      features: this.vectorizer.featureCount,
      providerId: this.provider?.id ?? null,
      pendingEmbeddings: this.pendingEmbedding.size,
    };
  }

  /** 计算（可能缓存的）文档向量。 */
  private vectorOf(entry: InternalDocument): Float32Array {
    if (entry.vec && entry.vecVersion === this.vectorizer.version) return entry.vec;
    entry.vec = this.vectorizer.transform(entry.tf);
    entry.vecVersion = this.vectorizer.version;
    return entry.vec;
  }

  /** 同步检索（TF-IDF 路径）。注入 provider 时请改用 searchAsync。 */
  search(query: string, options: VectorQueryOptions = {}): VectorHit[] {
    return this.searchWithTotal(query, options).hits;
  }

  searchWithTotal(query: string, options: VectorQueryOptions = {}): VectorSearchResult {
    if (this.provider) {
      throw new Error('VectorIndex 已注入 EmbeddingProvider，请使用 searchAsync() 进行检索');
    }
    const limit = Math.max(0, options.limit ?? 10);
    const minScore = options.minScore ?? 0;
    if (limit === 0 || this.documents.size === 0) return { hits: [], total: 0 };
    const queryVec = this.vectorizer.transform(
      buildTermFrequency(query, this.dimensions, this.maxFeaturesPerDocument),
    );
    const heap = new TopKHeap<VectorHit>(limit, (hit) => hit.score);
    let total = 0;
    for (const entry of this.documents.values()) {
      if (options.filter && !options.filter(entry.doc)) continue;
      const score = dotProduct(queryVec, this.vectorOf(entry));
      if (score <= 0 || score < minScore) continue;
      total += 1;
      heap.push({ id: entry.doc.id, score, doc: entry.doc });
    }
    return { hits: heap.toSorted(), total };
  }

  /** 异步检索：provider 模式下先补齐向量，再走余弦相似度。 */
  async searchAsync(query: string, options: VectorQueryOptions = {}): Promise<VectorHit[]> {
    return (await this.searchAsyncWithTotal(query, options)).hits;
  }

  async searchAsyncWithTotal(query: string, options: VectorQueryOptions = {}): Promise<VectorSearchResult> {
    if (!this.provider) return this.searchWithTotal(query, options);
    await this.embedPending();
    const limit = Math.max(0, options.limit ?? 10);
    const minScore = options.minScore ?? 0;
    if (limit === 0 || this.documents.size === 0) return { hits: [], total: 0 };
    const [rawQueryVec] = await this.provider.embed([query]);
    const normalizedQuery = normalizeVector(Float32Array.from(rawQueryVec));
    const heap = new TopKHeap<VectorHit>(limit, (hit) => hit.score);
    let total = 0;
    for (const entry of this.documents.values()) {
      if (!entry.providerVec) continue;
      if (options.filter && !options.filter(entry.doc)) continue;
      const score = cosineSimilarity(normalizedQuery, entry.providerVec);
      if (score <= 0 || score < minScore) continue;
      total += 1;
      heap.push({ id: entry.doc.id, score, doc: entry.doc });
    }
    return { hits: heap.toSorted(), total };
  }

  /** 为缺失或失效的外部向量批量补齐。 */
  async embedPending(): Promise<number> {
    if (!this.provider || this.pendingEmbedding.size === 0) return 0;
    const ids = Array.from(this.pendingEmbedding);
    const entries = ids.map((id) => this.documents.get(id)).filter((e): e is InternalDocument => Boolean(e));
    const batchSize = 64;
    let done = 0;
    for (let start = 0; start < entries.length; start += batchSize) {
      const batch = entries.slice(start, start + batchSize);
      const vectors = await this.provider.embed(batch.map((e) => e.doc.text));
      batch.forEach((entry, i) => {
        const vec = vectors[i];
        if (!vec) return;
        entry.providerVec = normalizeVector(Float32Array.from(vec));
        this.pendingEmbedding.delete(entry.doc.id);
        done += 1;
      });
    }
    return done;
  }

  /** 切换/清除 Embedding provider；切换后已有文档需要重新 embedPending。 */
  setProvider(provider: EmbeddingProvider | null): void {
    if (provider && provider.dimensions !== this.dimensions) {
      throw new Error(
        `EmbeddingProvider 维度不匹配：索引 ${this.dimensions} 维，provider ${provider.dimensions} 维`,
      );
    }
    this.provider = provider;
    this.pendingEmbedding.clear();
    if (provider) {
      for (const id of this.documents.keys()) this.pendingEmbedding.add(id);
    }
  }

  /**
   * 写入镜像：默认 gzip 压缩的 JSON（读取时按魔数自动识别，明文 JSON 同样可读）。
   * 先写临时文件再 rename，避免留下半截文件。
   */
  async saveToFile(filePath: string, options: SaveVectorIndexOptions = {}): Promise<void> {
    const includeText = options.includeText ?? true;
    const payload: PersistedVectorIndex = {
      version: VECTOR_INDEX_FORMAT_VERSION,
      dimensions: this.dimensions,
      providerId: this.provider?.id ?? null,
      documentCount: this.documents.size,
      df: this.vectorizer.exportDf(),
      documents: Array.from(this.documents.values()).map((entry) => {
        const item: PersistedVectorIndex['documents'][number] = {
          id: entry.doc.id,
          sessionId: entry.doc.sessionId,
          seq: entry.doc.seq,
          role: entry.doc.role,
          kind: entry.doc.kind,
          time: entry.doc.time,
          filePath: entry.doc.filePath,
          toolName: entry.doc.toolName,
          text: includeText ? entry.doc.text : '',
          tf: Array.from(entry.tf.entries()),
        };
        if (entry.providerVec) item.vec = encodeVector(entry.providerVec);
        return item;
      }),
    };
    const json = Buffer.from(JSON.stringify(payload), 'utf8');
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    const tmpPath = `${filePath}.tmp`;
    await fs.writeFile(tmpPath, zlib.gzipSync(json));
    await fs.rename(tmpPath, filePath);
  }

  /** 从镜像恢复；维度不一致或文件不存在时返回 false 并保持原状。 */
  async loadFromFile(filePath: string): Promise<boolean> {
    let raw: Buffer;
    try {
      raw = await fs.readFile(filePath);
    } catch {
      return false;
    }
    let payload: PersistedVectorIndex;
    try {
      const text = raw[0] === 0x1f && raw[1] === 0x8b ? zlib.gunzipSync(raw).toString('utf8') : raw.toString('utf8');
      payload = JSON.parse(text) as PersistedVectorIndex;
    } catch {
      return false;
    }
    if (!payload || payload.version !== VECTOR_INDEX_FORMAT_VERSION) return false;
    if (payload.dimensions !== this.dimensions) return false;
    this.clear();
    this.vectorizer.importDf(payload.df ?? [], payload.documentCount ?? 0);
    for (const item of payload.documents ?? []) {
      const doc: VectorDocument = {
        id: item.id,
        sessionId: item.sessionId,
        seq: item.seq,
        text: item.text ?? '',
        role: item.role,
        kind: item.kind,
        time: item.time,
        filePath: item.filePath,
        toolName: item.toolName,
      };
      const entry: InternalDocument = {
        doc,
        tf: new Map(item.tf ?? []),
        vecVersion: -1,
      };
      if (item.vec && this.provider) entry.providerVec = decodeVector(item.vec, this.dimensions);
      this.documents.set(doc.id, entry);
      let bucket = this.sessionDocs.get(doc.sessionId);
      if (!bucket) {
        bucket = new Set();
        this.sessionDocs.set(doc.sessionId, bucket);
      }
      bucket.add(doc.id);
      if (this.provider && !entry.providerVec) this.pendingEmbedding.add(doc.id);
    }
    // df 统计已从镜像恢复，但向量缓存版本需与 vectorizer.version 对齐
    for (const entry of this.documents.values()) entry.vecVersion = -1;
    return true;
  }
}
