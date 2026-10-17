/**
 * 检索内核自测脚本：`npx tsx src/core/test_search.ts`
 *
 * 覆盖：特征提取与向量数学、TF-IDF 语义排序、Embedding 适配器、Grep 关键词/正则/上下文、
 * 会话文件解析（jsonl / json / 多帧 jsonl.zstd）、全量与增量索引、JSON 镜像持久化、宿主契约兼容层。
 * 退出码 0 = 全部通过；1 = 存在失败。
 */
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';

import {
  ArchiveChunk,
  ArchiveIndexer,
  GrepEngine,
  TopKHeap,
  TfidfVectorizer,
  VectorIndex,
  buildTermFrequency,
  buildSnippet,
  compileGrepPlan,
  cosineSimilarity,
  grep_search,
  grepText,
  hash32,
  l2Norm,
  normalizeVector,
  projectFeature,
  readSessionRecords,
  recordsToSession,
  scanSessionFiles,
  scan_archive_sessions,
  tokenizeFeatures,
  topKByScore,
  vector_search,
  zstdAvailable,
} from './index';
import type { EmbeddingProvider } from './types';

/* ------------------------------------------------------------------ */
/* 测试脚手架                                                           */
/* ------------------------------------------------------------------ */

let passed = 0;
const failures: string[] = [];
let current = '';

function check(condition: boolean, message: string): void {
  if (condition) return;
  throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, message = ''): void {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  check(ok, `${message} 期望 ${JSON.stringify(expected)}，实际 ${JSON.stringify(actual)}`);
}

function assertClose(actual: number, expected: number, tolerance: number, message = ''): void {
  check(
    Math.abs(actual - expected) <= tolerance,
    `${message} 期望 ${expected}±${tolerance}，实际 ${actual}`,
  );
}

function assertIncludes(haystack: string, needle: string, message = ''): void {
  check(haystack.includes(needle), `${message} 文本中未找到 ${JSON.stringify(needle)}`);
}

async function test(name: string, fn: () => void | Promise<void>): Promise<void> {
  current = name;
  try {
    await fn();
    passed += 1;
    console.log(`  PASS  ${name}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    failures.push(`${name}: ${message}`);
    console.log(`  FAIL  ${name}\n        ${message}`);
  }
}

async function section(name: string, fn: () => Promise<void>): Promise<void> {
  console.log(`\n== ${name} ==`);
  await fn();
}

/* ------------------------------------------------------------------ */
/* 固定数据与夹具                                                       */
/* ------------------------------------------------------------------ */

const T1 = 1_700_000_000_000; // 2023-11-14T22:13:20Z
const T2 = T1 + 60_000;
const T3 = T1 + 120_000;

function session_record(id: string, cwd: string): unknown {
  return { type: 'session', version: 4, id, createdAt: T1, cwd, origin: 'desktop' };
}

function user_record(seq: number, time: number, text: string): unknown {
  return { type: 'user/message', seq, time, data: { turn: 1, step: 1, content: [{ type: 'text', text }] } };
}

function assistant_record(seq: number, time: number, text: string, reasoning?: string): unknown {
  const content: unknown[] = [];
  if (reasoning) content.push({ type: 'reasoning', text: reasoning });
  content.push({ type: 'text', text });
  return { type: 'assistant/message', seq, time, data: { turn: 1, step: 1, message: { role: 'assistant', content } } };
}

function tool_call_record(seq: number, time: number, callId: string, name: string, args: unknown): unknown {
  return {
    type: 'assistant/message',
    seq,
    time,
    data: {
      turn: 1,
      step: 1,
      message: { role: 'assistant', content: [{ type: 'tool-call', callId, name, arguments: args }] },
    },
  };
}

function tool_result_record(seq: number, time: number, callId: string, text: string): unknown {
  return {
    type: 'tool/result',
    seq,
    time,
    data: {
      turn: 1,
      step: 1,
      message: { role: 'tool', toolCallId: callId, content: [{ type: 'text', text }] },
    },
  };
}

function title_record(seq: number, time: number, title: string): unknown {
  return { type: 'session/title', seq, time, data: { title, messageSeqs: [1] } };
}

/** 每个 JSON 记录一个 zstd 帧，模拟 DSH 的 session.v4.jsonl.zstd 布局。 */
function encode_zstd_frames(records: unknown[]): Buffer {
  const compress = (zlib as unknown as Record<string, (b: Buffer) => Buffer>).zstdCompressSync;
  return Buffer.concat(records.map((record) => compress(Buffer.from(`${JSON.stringify(record)}\n`, 'utf8'))));
}

function fake_provider(dimensions = 64): EmbeddingProvider {
  return {
    id: `fake-${dimensions}`,
    dimensions,
    async embed(texts: string[]): Promise<Float32Array[]> {
      return texts.map((text) => {
        const vec = new Float32Array(dimensions);
        for (const feature of tokenizeFeatures(text)) {
          const { index, sign } = projectFeature(feature, dimensions);
          vec[index] += sign;
        }
        return normalizeVector(vec, true);
      });
    },
  };
}

interface Fixture {
  root: string;
  indexDir: string;
  sessionsRoot: string;
}

/** 三份会话文件：多帧 zstd、纯 jsonl、通用 json。 */
async function create_fixture(): Promise<Fixture> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'archive-search-test-'));
  const sessionsRoot = path.join(root, 'sessions');
  const indexDir = path.join(root, 'index');

  const sessionA = [
    session_record('sess-alpha', 'C:\\work\\alpha'),
    title_record(1, T1, '向量检索内核设计'),
    user_record(2, T1, '帮我实现一个纯 TypeScript 的向量检索引擎，不要 SQLite。'),
    assistant_record(3, T1, '可以：使用中文字符双字组 N-gram 加 TF-IDF 哈希投影，再用余弦相似度排序。'),
    tool_call_record(4, T2, 'call-1', 'run_code', { description: '跑测试', code: 'npm run test_search' }),
    tool_result_record(5, T2, 'call-1', 'PASS 全部 42 个断言通过'),
  ];
  const sessionB = [
    session_record('sess-beta', 'C:\\work\\beta'),
    title_record(1, T1, 'Postgres 迁移计划'),
    user_record(2, T1, 'remember to vacuum the postgres database after migration'),
    assistant_record(3, T2, 'Alembic 迁移完成后执行 VACUUM ANALYZE，并检查连接池配置。'),
    user_record(4, T3, '另外记得给 archive-search 插件补充单元测试。'),
  ];
  const sessionC = {
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: [{ type: 'text', text: '解释一下倒排索引和向量召回的区别。' }] },
      { role: 'assistant', content: [{ type: 'text', text: '倒排索引按词项精确匹配，向量召回按语义相似度排序。' }] },
    ],
  };

  if (zstdAvailable()) {
    await fs.mkdir(path.join(sessionsRoot, 'proj-a', 'sess-alpha'), { recursive: true });
    await fs.writeFile(path.join(sessionsRoot, 'proj-a', 'sess-alpha', 'session.v4.jsonl.zstd'), encode_zstd_frames(sessionA));
  } else {
    await fs.mkdir(path.join(sessionsRoot, 'proj-a', 'sess-alpha'), { recursive: true });
    await fs.writeFile(
      path.join(sessionsRoot, 'proj-a', 'sess-alpha', 'session.v4.jsonl'),
      `${sessionA.map((r) => JSON.stringify(r)).join('\n')}\n`,
      'utf8',
    );
  }
  await fs.mkdir(path.join(sessionsRoot, 'proj-a', 'sess-beta'), { recursive: true });
  await fs.writeFile(
    path.join(sessionsRoot, 'proj-a', 'sess-beta', 'session.v4.jsonl'),
    `${sessionB.map((r) => JSON.stringify(r)).join('\n')}\n`,
    'utf8',
  );
  await fs.mkdir(path.join(sessionsRoot, 'proj-b', 'sess-gamma'), { recursive: true });
  await fs.writeFile(path.join(sessionsRoot, 'proj-b', 'sess-gamma', 'session.json'), JSON.stringify(sessionC), 'utf8');

  return { root, indexDir, sessionsRoot };
}

/* ------------------------------------------------------------------ */
/* 1. 特征提取与向量数学                                                */
/* ------------------------------------------------------------------ */

async function test_features(): Promise<void> {
  await test('中文双字 N-gram 与英文词元特征', () => {
    const features = tokenizeFeatures('向量检索 engine v2');
    for (const expected of ['c:向量', 'c:量检', 'c:检索', 'C:向量检索', 'w:engine', 'w:v2', 'b:en']) {
      check(features.includes(expected), `缺少特征 ${expected}，实际 ${features.join(',')}`);
    }
  });

  await test('哈希投影落在维度内且可复现', () => {
    const a = projectFeature('c:向量', 4096);
    const b = projectFeature('c:向量', 4096);
    assertEqual(a, b, '同特征投影应稳定');
    check(a.index >= 0 && a.index < 4096, '索引越界');
    check(a.sign === 1 || a.sign === -1, '符号应为 ±1');
    check(hash32('abc') === hash32('abc'), 'hash32 应稳定');
    check(hash32('abc') !== hash32('abd'), 'hash32 应区分不同输入');
  });

  await test('词频向量非空且可归一化', () => {
    const tf = buildTermFrequency('向量检索向量', 4096);
    check(tf.size > 0, '词频表为空');
    const vectorizer = new TfidfVectorizer(4096);
    vectorizer.addDocument(tf);
    const vec = vectorizer.transform(tf);
    assertClose(l2Norm(vec), 1, 1e-6, 'TF-IDF 向量应 L2 归一化');
    assertEqual(vec.length, 4096, '向量维度');
  });

  await test('余弦相似度：同向、正交、对称与缩放不变', () => {
    const a = Float32Array.from([1, 2, 3]);
    const b = Float32Array.from([2, 4, 6]);
    const orthogonal = Float32Array.from([3, 0, -1]);
    assertClose(cosineSimilarity(a, a), 1, 1e-6, '同向量');
    assertClose(cosineSimilarity(a, b), 1, 1e-6, '缩放不变');
    assertClose(cosineSimilarity(a, orthogonal), 0, 1e-6, '正交为 0');
    assertClose(cosineSimilarity(a, orthogonal), cosineSimilarity(orthogonal, a), 1e-9, '对称');
    assertClose(cosineSimilarity(new Float32Array(3), a), 0, 1e-9, '零向量');
  });

  await test('归一化向量模长为 1', () => {
    const vec = normalizeVector(Float32Array.from([3, 4]));
    assertClose(vec[0], 0.6, 1e-6, 'x');
    assertClose(vec[1], 0.8, 1e-6, 'y');
    assertClose(l2Norm(vec), 1, 1e-6, '模长');
  });

  await test('Top-K 最小堆与全排序一致', () => {
    const scores = [0.2, 0.9, 0.5, 0.99, 0.1, 0.7, 0.35];
    const heap = new TopKHeap<{ score: number; value: number }>(3, (entry) => entry.score);
    scores.forEach((score, value) => heap.push({ score, value }));
    const top = heap.toSorted().map((entry) => entry.score);
    const expected = scores.slice().sort((a, b) => b - a).slice(0, 3);
    assertEqual(top, expected, 'Top-3');
    assertEqual(heap.size, 3, '堆容量');

    const small = topKByScore(
      scores.map((score) => ({ score, value: score })),
      2,
    );
    assertEqual(small.map((entry) => entry.score), [0.99, 0.9], 'topKByScore');
    const zero = new TopKHeap<number>(0, (value) => value);
    zero.push(1);
    assertEqual(zero.size, 0, 'K=0 时不保留元素');
  });
}

/* ------------------------------------------------------------------ */
/* 2. 向量索引与 TF-IDF 语义检索                                         */
/* ------------------------------------------------------------------ */

function sample_docs(): Array<{ id: string; sessionId: string; seq: number; text: string }> {
  return [
    { id: 'd1', sessionId: 's1', seq: 1, text: '向量检索引擎使用余弦相似度对候选文档排序' },
    { id: 'd2', sessionId: 's1', seq: 2, text: '今天天气不错，适合出门散步和拍照' },
    { id: 'd3', sessionId: 's2', seq: 1, text: 'Postgres 数据库迁移后需要执行 VACUUM ANALYZE' },
    { id: 'd4', sessionId: 's2', seq: 2, text: 'the semantic search ranks candidate messages by cosine similarity' },
    { id: 'd5', sessionId: 's3', seq: 1, text: '倒排索引按词项精确匹配，适合关键词检索' },
  ];
}

async function test_vector_index(): Promise<void> {
  await test('中文语义排序：查询命中最相关文档', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const hits = index.search('余弦相似度 排序', { limit: 3 });
    check(hits.length > 0, '应有命中');
    check(hits[0].id === 'd1', `首位应为 d1，实际 ${hits[0].id}`);
    for (let i = 1; i < hits.length; i += 1) {
      check(hits[i - 1].score >= hits[i].score, '得分应降序');
    }
  });

  await test('近义查询仍可召回（非连续字面匹配）', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    // “语义相似度召回排序” 在任何文档中都不作为连续子串出现，靠双字组重叠召回
    const hits = index.search('语义相似度召回排序', { limit: 5 });
    check(hits.length > 0, '应召回共享双字组的文档');
    check(hits[0].id === 'd1', `首位应为 d1，实际 ${hits.map((hit) => hit.id).join(',')}`);
  });

  await test('英文查询排序正确', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const hits = index.search('cosine similarity ranking', { limit: 2 });
    check(hits[0].id === 'd4', `首位应为 d4，实际 ${hits[0].id}`);
  });

  await test('无关查询被 minScore 过滤', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const hits = index.search('zzzzqqqq 完全无关内容', { limit: 5, minScore: 0.05 });
    assertEqual(hits.length, 0, '无关查询应返回空');
  });

  await test('limit 与 total 语义', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const result = index.searchWithTotal('检索', { limit: 2, minScore: 0 });
    check(result.hits.length <= 2, 'limit 未生效');
    check(result.total >= result.hits.length, 'total 应不小于返回条数');
  });

  await test('删除文档与会话后不再被召回', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    check(index.removeDocument('d1'), '应删除 d1');
    check(!index.hasDocument('d1'), 'd1 应不存在');
    assertEqual(index.removeSession('s2'), 2, 's2 应删除 2 条');
    assertEqual(index.size, 2, '剩余文档数');
    const hits = index.search('cosine similarity search', { limit: 5, minScore: 0 });
    check(!hits.some((hit) => hit.id === 'd4'), '已删除文档不应出现');
  });

  await test('重复 addDocument 覆盖而不重复计数', () => {
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const before = index.size;
    index.addDocument({ id: 'd1', sessionId: 's1', seq: 1, text: '向量检索引擎改写后的内容' });
    assertEqual(index.size, before, '数量不应变化');
    assertEqual(index.getDocument('d1')?.text, '向量检索引擎改写后的内容', '内容应更新');
  });

  await test('JSON 镜像持久化后可原样检索', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vector-index-'));
    const file = path.join(dir, 'vectors.json');
    const index = new VectorIndex();
    index.addDocuments(sample_docs());
    const expected = index.search('余弦相似度', { limit: 3, minScore: 0 });
    await index.saveToFile(file);

    const restored = new VectorIndex();
    check(await restored.loadFromFile(file), '加载应成功');
    assertEqual(restored.size, index.size, '文档数应一致');
    const actual = restored.search('余弦相似度', { limit: 3, minScore: 0 });
    assertEqual(actual.map((hit) => hit.id), expected.map((hit) => hit.id), '命中顺序应一致');
    actual.forEach((hit, i) => assertClose(hit.score, expected[i].score, 1e-6, '得分应一致'));
    await fs.rm(dir, { recursive: true, force: true });
  });

  await test('维度不匹配的镜像被拒绝', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vector-index-'));
    const file = path.join(dir, 'vectors.json');
    const index = new VectorIndex({ dimensions: 256 });
    index.addDocument(sample_docs()[0]);
    await index.saveToFile(file);
    const other = new VectorIndex({ dimensions: 512 });
    assertEqual(await other.loadFromFile(file), false, '不同维度应拒绝加载');
    assertEqual(await other.loadFromFile(path.join(dir, 'missing.json')), false, '缺失文件应返回 false');
    await fs.rm(dir, { recursive: true, force: true });
  });

  await test('外部 Embedding provider：异步检索与向量镜像', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vector-provider-'));
    const file = path.join(dir, 'vectors.json');
    const provider = fake_provider(64);
    const index = new VectorIndex({ provider });
    index.addDocuments(sample_docs());
    assertEqual(index.stats().pendingEmbeddings, 5, '注入 provider 后应全部待嵌入');
    const embedded = await index.embedPending();
    assertEqual(embedded, 5, '应补齐 5 条向量');
    assertEqual(index.stats().pendingEmbeddings, 0, '待嵌入应清空');

    let syncThrew = false;
    try {
      index.search('任意查询');
    } catch {
      syncThrew = true;
    }
    check(syncThrew, 'provider 模式下同步 search 应抛错');

    const hits = await index.searchAsync('cosine similarity ranking', { limit: 2 });
    check(hits[0].id === 'd4', `provider 检索首位应为 d4，实际 ${hits[0].id}`);

    await index.saveToFile(file);
    const restored = new VectorIndex({ provider: fake_provider(64) });
    check(await restored.loadFromFile(file), 'provider 镜像应加载成功');
    assertEqual(restored.stats().pendingEmbeddings, 0, '镜像中的向量应可直接复用');
    const restoredHits = await restored.searchAsync('cosine similarity ranking', { limit: 2 });
    assertEqual(restoredHits.map((hit) => hit.id), hits.map((hit) => hit.id), '恢复后命中应一致');
    await fs.rm(dir, { recursive: true, force: true });
  });
}

/* ------------------------------------------------------------------ */
/* 3. Grep 引擎                                                         */
/* ------------------------------------------------------------------ */

function grep_sessions() {
  return [
    {
      id: 's1',
      filePath: 'C:\\a\\session.v4.jsonl',
      cwd: 'C:\\a',
      title: '向量检索内核设计',
      createdAt: T1,
      updatedAt: T3,
      fileSize: 10,
      fileMtimeMs: 10,
      messages: [
        {
          id: 's1:1:1',
          seq: 1,
          role: 'system' as const,
          kind: 'title' as const,
          text: '向量检索内核设计',
          time: T1,
        },
        {
          id: 's1:2:2',
          seq: 2,
          role: 'user' as const,
          kind: 'message' as const,
          text: '第一行\n帮我实现 Vector Search 引擎\n第三行\n第四行',
          time: T2,
        },
        {
          id: 's1:3:3',
          seq: 3,
          role: 'assistant' as const,
          kind: 'message' as const,
          text: '可以用 TF-IDF 加余弦相似度。',
          time: T3,
        },
      ],
    },
    {
      id: 's2',
      filePath: 'C:\\b\\session.v4.jsonl',
      cwd: 'C:\\b',
      title: 'Postgres 迁移',
      createdAt: T1,
      updatedAt: T2,
      fileSize: 10,
      fileMtimeMs: 10,
      messages: [
        {
          id: 's2:1:1',
          seq: 1,
          role: 'user' as const,
          kind: 'message' as const,
          text: 'VACUUM ANALYZE 与连接池检查',
          time: T1,
        },
      ],
    },
  ];
}

async function test_grep(): Promise<void> {
  const engine = new GrepEngine();

  await test('大小写不敏感与敏感两种模式', () => {
    const insensitive = engine.search(grep_sessions(), { query: 'vector search', limit: 5 });
    check(insensitive.total === 1, `应命中 1 条，实际 ${insensitive.total}`);
    assertEqual(insensitive.results[0].sessionId, 's1', '命中会话');
    const sensitive = engine.search(grep_sessions(), { query: 'vector search', caseSensitive: true, limit: 5 });
    assertEqual(sensitive.total, 0, '大小写敏感时不应命中');
    const exact = engine.search(grep_sessions(), { query: 'Vector Search', caseSensitive: true, limit: 5 });
    assertEqual(exact.total, 1, '大小写一致时应命中');
  });

  await test('高亮区间与摘要切片一致', () => {
    const response = engine.search(grep_sessions(), { query: 'vector search', limit: 5 });
    const item = response.results[0];
    check(item.ranges.length > 0, '应有高亮区间');
    const highlighted = item.ranges
      .map((range) => item.snippet.slice(range.start, range.end).toLowerCase())
      .sort();
    assertEqual(highlighted, ['search', 'vector'], '每个高亮区间应恰好覆盖一个词');
    const span = item.snippet
      .slice(item.ranges[0].start, item.ranges[item.ranges.length - 1].end)
      .toLowerCase();
    assertIncludes(span, 'vector search', '高亮区间应覆盖命中短语');
    const line = item.matches[0];
    assertEqual(line.line, 2, '命中行号应为第 2 行');
    assertEqual(line.before, ['第一行'], '上文窗口');
    assertEqual(line.after, ['第三行', '第四行'], '下文窗口');
  });

  await test('上下文窗口可配置且不超过边界', () => {
    const one = engine.search(grep_sessions(), { query: 'Vector', contextLines: 1, limit: 5 });
    assertEqual(one.results[0].matches[0].before, ['第一行'], 'contextLines=1 上文');
    assertEqual(one.results[0].matches[0].after, ['第三行'], 'contextLines=1 下文');
    const none = engine.search(grep_sessions(), { query: 'Vector', contextLines: 0, limit: 5 });
    assertEqual(none.results[0].matches[0].before, [], 'contextLines=0 无上文');
    assertEqual(none.results[0].matches[0].after, [], 'contextLines=0 无下文');
    const wide = engine.search(grep_sessions(), { query: '第四行', contextLines: 5, limit: 5 });
    assertEqual(wide.results[0].matches[0].after, [], '末尾行之后不应有内容');
    assertEqual(wide.results[0].matches[0].before.length, 3, '末尾行之前的全部内容');
  });

  await test('多词 AND / OR 语义', () => {
    const and = engine.search(grep_sessions(), { query: '余弦 相似度 不存在词', limit: 5 });
    assertEqual(and.total, 0, 'AND 模式应要求全部命中');
    const or = engine.search(grep_sessions(), { query: '余弦 不存在词', matchAll: false, limit: 5 });
    assertEqual(or.total, 1, 'OR 模式应命中一条');
  });

  await test('引号短语作为整体匹配', () => {
    const hit = engine.search(grep_sessions(), { query: '"VACUUM ANALYZE"', limit: 5 });
    assertEqual(hit.total, 1, '短语应命中');
    assertEqual(hit.results[0].sessionId, 's2', '短语命中会话');
    const miss = engine.search(grep_sessions(), { query: '"ANALYZE VACUUM"', limit: 5 });
    assertEqual(miss.total, 0, '顺序不同的短语不应命中');
  });

  await test('正则模式与多行高亮', () => {
    const response = engine.search(grep_sessions(), { query: 'T[F]-?IDF|余弦', regex: true, limit: 5 });
    check(response.total >= 1, '正则应命中');
    const snippetText = response.results.map((item) => item.snippet).join(' ');
    assertIncludes(snippetText, 'TF-IDF', '正则命中内容');
    let threw = false;
    try {
      engine.search(grep_sessions(), { query: '([', regex: true });
    } catch {
      threw = true;
    }
    check(threw, '非法正则应抛错');
  });

  await test('角色 / 会话 / 时间过滤', () => {
    const userOnly = engine.search(grep_sessions(), { query: 'VACUUM', roles: ['user'], limit: 5 });
    assertEqual(userOnly.total, 1, 'user 角色命中');
    const assistantOnly = engine.search(grep_sessions(), { query: 'VACUUM', roles: ['assistant'], limit: 5 });
    assertEqual(assistantOnly.total, 0, 'assistant 角色不应命中');
    const scoped = engine.search(grep_sessions(), { query: 'TF-IDF', sessionIds: ['s2'], limit: 5 });
    assertEqual(scoped.total, 0, '会话过滤应生效');
    const recent = engine.search(grep_sessions(), { query: 'VACUUM', since: T2 + 1, limit: 5 });
    assertEqual(recent.total, 0, '时间下界过滤应生效');
    const titles = engine.search(grep_sessions(), { query: '向量检索内核设计', kinds: ['title'], limit: 5 });
    assertEqual(titles.total, 1, 'kind 过滤应命中标题');
  });

  await test('limit / offset / total 分页', () => {
    const all = engine.search(grep_sessions(), { query: '检索 相似度 ANALYZE', matchAll: false, limit: 10 });
    check(all.total >= 2, '应有多条命中');
    const first = engine.search(grep_sessions(), { query: '检索 相似度 ANALYZE', matchAll: false, limit: 1 });
    assertEqual(first.results.length, 1, 'limit=1');
    assertEqual(first.total, all.total, 'total 不受 limit 影响');
    const second = engine.search(grep_sessions(), { query: '检索 相似度 ANALYZE', matchAll: false, limit: 1, offset: 1 });
    check(second.results[0].messageId !== first.results[0].messageId, 'offset 应翻页');
    assertEqual(second.offset, 1, 'offset 回显');
  });

  await test('空查询与纯空白查询返回空结果', () => {
    assertEqual(engine.search(grep_sessions(), { query: '   ' }).total, 0, '空白查询');
    assertEqual(engine.search(grep_sessions(), { query: '' }).total, 0, '空查询');
  });

  await test('摘要截断保留高亮并加省略号', () => {
    const long = `${'前置内容'.repeat(60)}关键词命中${'后置内容'.repeat(60)}`;
    const plan = compileGrepPlan('关键词命中');
    const grep = grepText(long, plan, { contextLines: 0, maxMatchesPerMessage: 1 });
    const snippet = buildSnippet(grep.matches[0].text, grep.matches[0].ranges, 40);
    check(snippet.text.length <= 42, `摘要过长：${snippet.text.length}`);
    assertIncludes(snippet.text, '关键词命中', '摘要应含命中词');
    snippet.ranges.forEach((range) => {
      assertIncludes(snippet.text.slice(range.start, range.end), '关键词命中', '高亮区间仍指向命中词');
    });
  });
}

/* ------------------------------------------------------------------ */
/* 4. 归档索引器（解析 / 全量 / 增量 / 持久化）                          */
/* ------------------------------------------------------------------ */

async function test_indexer(): Promise<void> {
  const fixture = await create_fixture();
  const hasZstd = zstdAvailable();

  await test('扫描会话文件（含同名扩展名去重）', async () => {
    const files = await scanSessionFiles(fixture.sessionsRoot);
    assertEqual(files.length, 3, `应扫描到 3 个文件，实际 ${files.length}`);
    check(files.some((file) => file.sessionId === 'sess-alpha'), '应识别 sess-alpha');
  });

  await test('全量索引解析三种格式与记录类型', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    const stats = await indexer.fullIndex();
    assertEqual(stats.sessions, 3, '会话数');
    assertEqual(stats.errors.length, 0, `解析不应有错误：${stats.errors.join(';')}`);
    const alpha = indexer.getSession('sess-alpha');
    check(Boolean(alpha), '应存在 sess-alpha');
    check(hasZstd ? Boolean(alpha) : true, 'zstd 可用时应解析 zstd 会话');
    assertEqual(alpha?.title, '向量检索内核设计', '标题记录');
    assertEqual(alpha?.cwd, 'C:\\work\\alpha', 'cwd');
    const kinds = alpha?.messages.map((message) => message.kind) ?? [];
    for (const kind of ['title', 'message', 'tool_call', 'tool_result']) {
      check(kinds.includes(kind as never), `缺少消息类型 ${kind}：${kinds.join(',')}`);
    }
    const toolCall = alpha?.messages.find((message) => message.kind === 'tool_call');
    assertEqual(toolCall?.toolName, 'run_code', '工具名');
    const result = alpha?.messages.find((message) => message.kind === 'tool_result');
    assertEqual(result?.toolName, 'run_code', '工具结果应回填工具名');
    assertEqual(result?.role, 'tool', '工具结果角色');
    const gamma = indexer.getSession('sess-gamma');
    assertEqual(gamma?.messages.length, 3, '通用 json 会话消息数');
    indexer.close();
  });

  await test('关键词检索可跨格式命中并带上下文', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    await indexer.ensureIndexed();
    const response = await indexer.search({ query: '真空 迁移', matchAll: false, limit: 5 });
    check(response.total >= 1, '应命中 Postgres 会话');
    assertEqual(response.results[0].sessionId, 'sess-beta', '命中会话');
    const tool = await indexer.search({ query: 'npm run test_search', limit: 5 });
    assertEqual(tool.total, 1, '应命中工具调用参数');
    assertEqual(tool.results[0].kind, 'tool_call', '命中类型');
    const empty = await indexer.search({ query: '绝不可能出现的词组xyz', limit: 5 });
    assertEqual(empty.total, 0, '无命中');
    indexer.close();
  });

  await test('语义检索（TF-IDF）跨会话排序', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    await indexer.ensureIndexed();
    const response = await indexer.search({ query: '倒排索引与向量召回的区别', mode: 'semantic', limit: 3 });
    check(response.results.length > 0, '语义检索应有结果');
    assertEqual(response.results[0].sessionId, 'sess-gamma', '最相关会话');
    assertEqual(response.mode, 'semantic', '模式回显');
    check(response.results[0].score > 0, '得分应为正');
    const scoped = await indexer.search({
      query: '向量检索引擎',
      mode: 'semantic',
      limit: 3,
      sessionIds: ['sess-beta'],
    });
    scoped.results.forEach((item) => assertEqual(item.sessionId, 'sess-beta', '会话过滤'));
    indexer.close();
  });

  await test('增量更新：无变更时复用，变更时只重解析该文件', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    await indexer.fullIndex();
    const first = await indexer.incrementalUpdate();
    assertEqual(first.filesParsed, 0, '无变更不应重新解析');
    assertEqual(first.filesSkipped, 3, '应复用 3 个文件');

    const betaPath = path.join(fixture.sessionsRoot, 'proj-a', 'sess-beta', 'session.v4.jsonl');
    const extra = user_record(5, T3 + 60_000, '增量新增记录：记得给 zstd 多帧读取补充测试。');
    await fs.appendFile(betaPath, `${JSON.stringify(extra)}\n`, 'utf8');
    const second = await indexer.incrementalUpdate();
    assertEqual(second.filesParsed, 1, '应只重解析变更文件');
    assertEqual(second.filesSkipped, 2, '其余文件应复用');
    const hit = await indexer.search({ query: 'zstd 多帧读取', limit: 5 });
    assertEqual(hit.total, 1, '新增内容应可检索');
    assertEqual(hit.results[0].sessionId, 'sess-beta', '新增内容所在会话');

    await fs.rm(path.join(fixture.sessionsRoot, 'proj-b'), { recursive: true, force: true });
    const third = await indexer.incrementalUpdate();
    assertEqual(third.sessions, 2, '删除文件后会话数应减少');
    const removed = await indexer.search({ query: '倒排索引', mode: 'semantic', limit: 5 });
    check(
      !removed.results.some((item) => item.sessionId === 'sess-gamma'),
      '已删除会话不应再被召回',
    );
    indexer.close();
  });

  await test('JSON 镜像持久化：重启后免重建即可检索', async () => {
    const writer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    await writer.fullIndex();
    const vectorPath = writer.vectorPath;
    const docs = writer.stats().indexedDocuments;
    check(docs > 0, '应写入向量文档');
    check((await fs.stat(vectorPath)).size > 0, '向量镜像文件应存在');

    const reader = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    check(await reader.load(), '镜像应加载成功');
    assertEqual(reader.stats().indexedDocuments, docs, '恢复的文档数应一致');
    const offline = await reader.search({ query: '余弦相似度', mode: 'semantic', limit: 3 });
    check(offline.results.length > 0, '仅靠镜像即可语义检索');
    const stats = await reader.ensureIndexed();
    assertEqual(stats.filesParsed, 0, '复用镜像后不应重新解析（会话仍在镜像中）');
    reader.close();
    writer.close();
  });

  await test('search 首次调用自动建索引', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: fixture.sessionsRoot, indexDir: fixture.indexDir });
    assertEqual(indexer.indexed, false, '初始未索引');
    const response = await indexer.search({ query: '余弦', limit: 5 });
    assertEqual(indexer.indexed, true, '检索后应已索引');
    check(response.total >= 1, '应命中');
    indexer.close();
  });

  await test('meta 与根目录不匹配时拒绝复用镜像', async () => {
    const other = new ArchiveIndexer({
      sessionsRoot: path.join(fixture.root, 'other-sessions'),
      indexDir: fixture.indexDir,
    });
    assertEqual(await other.load(), false, '根目录不同的镜像不应加载');
    const stats = await other.incrementalUpdate();
    assertEqual(stats.sessions, 0, '空目录应得到 0 会话');
    other.close();
  });

  await test('多帧 zstd 解析（含帧内伪帧头）', async () => {
    if (!hasZstd) {
      console.log('        SKIP  Node 运行时不支持 zstd');
      return;
    }
    const compress = (zlib as unknown as Record<string, (b: Buffer) => Buffer>).zstdCompressSync;
    const header = Buffer.from(`${JSON.stringify(session_record('sess-zstd', 'C:\\work\\zstd'))}\n`, 'utf8');
    // 记录正文里嵌入原始 zstd 帧头字节，制造伪帧边界
    const tricky = Buffer.concat([
      Buffer.from('{"type":"user/message","seq":1,"time":1,"data":{"content":"伪帧头 ', 'utf8'),
      Buffer.from([0x28, 0xb5, 0x2f, 0xfd]),
      Buffer.from(' 之后仍需完整解析"}}\n', 'utf8'),
    ]);
    const tail = Buffer.from(`${JSON.stringify(assistant_record(2, T2, `长内容：${'数据'.repeat(500)}`))}\n`, 'utf8');
    const buffer = Buffer.concat([compress(header), compress(tricky), compress(tail)]);

    const records = await read_records_from_buffer(buffer);
    assertEqual(records.length, 3, '应解出 3 条记录');
    const parsed = recordsToSession(records, 'C:\\tmp\\session.v4.jsonl.zstd', buffer.length, T1, 'sess-zstd');
    check(Boolean(parsed), '应解析出会话');
    assertEqual(parsed?.messages.length, 2, '消息数');
    assertIncludes(parsed?.messages[1].text ?? '', '数据数据', '长内容完整');
  });

  await fs.rm(fixture.root, { recursive: true, force: true });
}

/** 把内存中的多帧 zstd 缓冲落盘后走真实读取路径。 */
async function read_records_from_buffer(buffer: Buffer): Promise<unknown[]> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'zstd-frame-'));
  const file = path.join(dir, 'session.v4.jsonl.zstd');
  await fs.writeFile(file, buffer);
  const { records, errors } = await readSessionRecords(file);
  assertEqual(errors.length, 0, `zstd 读取不应报错：${errors.join(';')}`);
  await fs.rm(dir, { recursive: true, force: true });
  return records;
}

/* ------------------------------------------------------------------ */
/* 5. 宿主契约兼容层                                                     */
/* ------------------------------------------------------------------ */

async function test_host_contract(): Promise<void> {
  const fixture = await create_fixture();
  const cachePath = path.join(fixture.root, 'cache', 'index.json');

  await test('scan_archive_sessions 返回扁平消息块', async () => {
    const scanned = await scan_archive_sessions({ root: fixture.sessionsRoot, cache_path: cachePath });
    assertEqual(scanned.sessions, 3, '会话数');
    assertEqual(scanned.files, 3, '文件数');
    check(scanned.chunks.length > 5, `消息块数量异常：${scanned.chunks.length}`);
    check(Number.isFinite(Date.parse(scanned.scanned_at)), 'scanned_at 应为可解析时间');
    const chunk = scanned.chunks.find((item) => item.session_id === 'sess-alpha');
    check(Boolean(chunk), '应包含 sess-alpha 的消息块');
    assertEqual(chunk?.title, '向量检索内核设计', '块级标题');
    assertEqual(chunk?.cwd, 'C:\\work\\alpha', '块级 cwd');
    check(Number.isFinite(Date.parse(chunk?.timestamp ?? '')), 'timestamp 应为 ISO 时间');
    assertEqual(chunk?.score, 0, '索引阶段得分应为 0');
  });

  await test('grep_search 兼容层：得分、全文与 limit', async () => {
    const scanned = await scan_archive_sessions({ root: fixture.sessionsRoot, cache_path: cachePath });
    const hits = grep_search(scanned.chunks, { query: 'alembic 迁移', limit: 5 });
    check(hits.length >= 1, '应命中');
    assertEqual(hits[0].session_id, 'sess-beta', '命中会话');
    assertIncludes(hits[0].text.toLowerCase(), 'alembic', '返回全文供上层截断');
    check(hits[0].score > 0, 'grep 得分应为正');
    check(Number.isFinite(Date.parse(hits[0].timestamp)), '命中时间可解析');
    assertEqual(grep_search(scanned.chunks, { query: 'alembic', limit: 1 }).length, 1, 'limit 生效');
    assertEqual(grep_search(scanned.chunks, { query: '   ', limit: 5 }).length, 0, '空白查询');
    const scoped = grep_search(scanned.chunks, { query: '向量', limit: 5, session_id: 'sess-beta' });
    scoped.forEach((hit) => assertEqual(hit.session_id, 'sess-beta', 'session_id 过滤'));
  });

  await test('vector_search 兼容层：异步排序与索引缓存', async () => {
    const scanned = await scan_archive_sessions({ root: fixture.sessionsRoot, cache_path: cachePath });
    const hits = await vector_search(scanned.chunks, { query: '倒排索引和向量召回的区别', limit: 3 });
    check(hits.length > 0, '语义检索应有结果');
    assertEqual(hits[0].session_id, 'sess-gamma', '最相关会话');
    const again = await vector_search(scanned.chunks, { query: '倒排索引和向量召回的区别', limit: 3 });
    assertEqual(again.map((hit) => hit.id), hits.map((hit) => hit.id), '同一 chunks 结果应稳定（走缓存索引）');
    const scoped = await vector_search(scanned.chunks, {
      query: '向量检索',
      limit: 3,
      session_id: 'sess-beta',
    });
    scoped.forEach((hit) => assertEqual(hit.session_id, 'sess-beta', 'session_id 过滤'));
    assertEqual((await vector_search([], { query: '任意' })).length, 0, '空块数组');
  });

  await test('ArchiveChunk 类型可用于自定义检索', () => {
    const chunks: ArchiveChunk[] = [
      {
        id: 'c1',
        session_id: 's-x',
        role: 'user',
        text: '自定义块检索向量',
        timestamp: new Date(T1).toISOString(),
        score: 0,
      },
    ];
    const hits = grep_search(chunks, { query: '向量', limit: 5 });
    assertEqual(hits.length, 1, '自定义块应可检索');
    assertEqual(hits[0].id, 'c1', '块 id 保留');
  });

  await fs.rm(fixture.root, { recursive: true, force: true });
}

/* ------------------------------------------------------------------ */
/* 6. 真实归档集成（只读，可跳过）                                        */
/* ------------------------------------------------------------------ */

async function test_real_archive(): Promise<void> {
  const realRoot = path.join(os.homedir(), '.dsh', 'sessions');
  let entries: string[] = [];
  try {
    entries = await fs.readdir(realRoot);
  } catch {
    console.log('  SKIP  ~/.dsh/sessions 不存在，跳过真实归档集成测试');
    return;
  }
  if (entries.length === 0) {
    console.log('  SKIP  ~/.dsh/sessions 为空，跳过真实归档集成测试');
    return;
  }
  const indexDir = await fs.mkdtemp(path.join(os.tmpdir(), 'archive-real-'));

  await test('真实归档：全量索引无错误', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: realRoot, indexDir });
    const stats = await indexer.fullIndex();
    check(stats.sessions > 0, '应解析出会话');
    check(stats.messages > 0, '应解析出消息');
    assertEqual(stats.errors.length, 0, `解析错误：${stats.errors.slice(0, 3).join(';')}`);
    check(stats.indexedDocuments > 0, '应建立向量文档');
    console.log(
      `        真实归档：${stats.sessions} 会话 / ${stats.messages} 消息 / ${stats.filesScanned} 文件 / ${stats.vectorFeatures} 特征`,
    );
    indexer.close();
  });

  await test('真实归档：关键词与语义检索可用', async () => {
    const indexer = new ArchiveIndexer({ sessionsRoot: realRoot, indexDir });
    await indexer.ensureIndexed();
    const keyword = await indexer.search({ query: 'plugin', limit: 5 });
    check(keyword.total > 0, '关键词检索应有命中');
    const semantic = await indexer.search({ query: '插件配置怎么写', mode: 'semantic', limit: 5 });
    check(semantic.results.length > 0, '语义检索应有命中');
    console.log(
      `        语义样例：${semantic.results
        .slice(0, 3)
        .map((item) => `${item.sessionId.slice(0, 8)}#${item.seq}=${item.score}`)
        .join(' / ')}`,
    );
    indexer.close();
  });

  await fs.rm(indexDir, { recursive: true, force: true });
}

/* ------------------------------------------------------------------ */
/* 入口                                                                */
/* ------------------------------------------------------------------ */

async function main(): Promise<void> {
  const startedAt = Date.now();
  console.log(`archive-search 检索引擎自测（Node ${process.version}，zstd=${zstdAvailable() ? '可用' : '不可用'}）`);
  await section('1. 特征提取与向量数学', test_features);
  await section('2. 向量索引与语义检索', test_vector_index);
  await section('3. Grep 引擎', test_grep);
  await section('4. 归档索引器', test_indexer);
  await section('5. 宿主契约兼容层', test_host_contract);
  await section('6. 真实归档集成', test_real_archive);

  const total = passed + failures.length;
  console.log(`\n结果：${passed}/${total} 通过，耗时 ${Date.now() - startedAt}ms`);
  if (failures.length > 0) {
    console.log('失败用例：');
    for (const failure of failures) console.log(`  - ${failure}`);
    process.exitCode = 1;
  }
}

void main().catch((error) => {
  console.error(`测试脚本异常（${current}）：`, error);
  process.exitCode = 1;
});
