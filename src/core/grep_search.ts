/**
 * Grep 引擎：纯文本关键词 / 正则检索，带上下文窗口与高亮区间。
 * 仅使用 indexOf 与 RegExp，无第三方依赖。
 */
import {
  HighlightRange,
  MatchedLine,
  MAX_LINE_LENGTH,
  MessageItem,
  SearchOptions,
  SearchResponse,
  SearchResultItem,
  SessionRecord,
  normalizeSearchOptions,
} from './types';

export interface GrepPlan {
  raw: string;
  /** 关键词模式下解析出的词项（支持引号短语）；正则模式为空数组。 */
  terms: string[];
  caseSensitive: boolean;
  regex: boolean;
  matchAll: boolean;
  /** 正则模式使用；每次 test/exec 前必须重置 lastIndex。 */
  regexSource?: RegExp;
}

/** 解析查询词项：空白分词，支持 "双引号短语" 与 '单引号短语'。 */
export function parseQueryTerms(query: string): string[] {
  const terms: string[] = [];
  let current = '';
  let quote: string | null = null;
  for (const ch of query) {
    if (quote) {
      if (ch === quote) {
        if (current) terms.push(current);
        current = '';
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      if (current.trim()) terms.push(current.trim());
      current = '';
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      if (current.trim()) terms.push(current.trim());
      current = '';
      continue;
    }
    current += ch;
  }
  if (current.trim()) terms.push(current.trim());
  return terms;
}

/** 由普通查询字符串编译出匹配计划，正则语法错误在此抛出。 */
export interface GrepPlanOptions {
  caseSensitive?: boolean;
  regex?: boolean;
  matchAll?: boolean;
}

export function compileGrepPlan(query: string, options: GrepPlanOptions = {}): GrepPlan {
  const caseSensitive = options?.caseSensitive ?? false;
  const regex = options?.regex ?? false;
  const matchAll = options?.matchAll ?? true;
  const plan: GrepPlan = {
    raw: query,
    terms: [],
    caseSensitive,
    regex,
    matchAll,
  };
  if (regex) {
    const flags = caseSensitive ? 'g' : 'gi';
    // 语法错误在此抛出，交由上层提示用户
    plan.regexSource = new RegExp(query, flags);
    return plan;
  }
  const terms = parseQueryTerms(query);
  plan.terms = caseSensitive ? terms : terms.map((t) => t.toLowerCase());
  return plan;
}

/** 查找 term 在 line 中的全部（不重叠）出现位置；from 为起始下标，便于分块续扫。 */
export function findTermRanges(
  line: string,
  term: string,
  caseSensitive: boolean,
  from = 0,
): HighlightRange[] {
  const ranges: HighlightRange[] = [];
  if (!term) return ranges;
  if (caseSensitive) {
    let idx = line.indexOf(term, from);
    while (idx !== -1) {
      ranges.push({ start: idx, end: idx + term.length });
      idx = line.indexOf(term, idx + term.length);
    }
    return ranges;
  }
  const hay = line.toLowerCase();
  // toLowerCase 可能改变长度（如 İ），此时退化为大小写敏感扫描，避免错位高亮
  if (hay.length !== line.length) return findTermRanges(line, term, true, from);
  const needle = term.toLowerCase();
  let idx = hay.indexOf(needle, from);
  while (idx !== -1) {
    ranges.push({ start: idx, end: idx + needle.length });
    idx = hay.indexOf(needle, idx + needle.length);
  }
  return ranges;
}

function regexRanges(line: string, regex: RegExp, limit: number): HighlightRange[] {
  const ranges: HighlightRange[] = [];
  regex.lastIndex = 0;
  let guard = 0;
  let m = regex.exec(line);
  while (m && guard < limit) {
    guard += 1;
    if (m[0].length === 0) {
      regex.lastIndex += 1;
    } else {
      ranges.push({ start: m.index, end: m.index + m[0].length });
    }
    m = regex.exec(line);
  }
  regex.lastIndex = 0;
  return ranges;
}

/** 合并重叠/相邻区间，保证升序且互不重叠。 */
export function mergeRanges(ranges: HighlightRange[]): HighlightRange[] {
  if (ranges.length <= 1) return ranges.slice();
  const sorted = ranges.slice().sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: HighlightRange[] = [sorted[0]];
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = merged[merged.length - 1];
    const cur = sorted[i];
    if (cur.start <= prev.end) {
      if (cur.end > prev.end) prev.end = cur.end;
    } else {
      merged.push({ ...cur });
    }
  }
  return merged;
}

/** 单行扫描：返回该行全部词项/正则的高亮区间（已合并）。 */
export function scanLine(line: string, plan: GrepPlan, limit = 200): HighlightRange[] {
  if (plan.regex && plan.regexSource) {
    return regexRanges(line, plan.regexSource, limit);
  }
  const ranges: HighlightRange[] = [];
  for (const term of plan.terms) {
    ranges.push(...findTermRanges(line, term, plan.caseSensitive));
    if (ranges.length > limit * 2) break;
  }
  return mergeRanges(ranges);
}

function quickHitCheck(text: string, plan: GrepPlan): boolean {
  if (plan.regex) {
    if (!plan.regexSource) return false;
    plan.regexSource.lastIndex = 0;
    const hit = plan.regexSource.test(text);
    plan.regexSource.lastIndex = 0;
    return hit;
  }
  if (plan.terms.length === 0) return false;
  if (plan.caseSensitive) {
    return plan.matchAll
      ? plan.terms.every((t) => text.includes(t))
      : plan.terms.some((t) => text.includes(t));
  }
  const hay = text.toLowerCase();
  // 长度变化说明存在特殊大小写字符，保守放行交给逐行精确扫描
  if (hay.length !== text.length) return true;
  return plan.matchAll
    ? plan.terms.every((t) => hay.includes(t))
    : plan.terms.some((t) => hay.includes(t));
}

export function truncateLine(line: string, maxLength = MAX_LINE_LENGTH): string {
  return line.length <= maxLength ? line : `${line.slice(0, maxLength)}…`;
}

/**
 * 以首个命中区间为中心截取摘要，返回相对摘要文本的高亮区间。
 * 行首缩进会被裁掉并同步平移区间。
 */
export function buildSnippet(
  line: string,
  ranges: HighlightRange[],
  maxLength: number,
): { text: string; ranges: HighlightRange[] } {
  let text = line;
  let shift = 0;
  const leading = text.length - text.trimStart().length;
  if (leading > 0 && text.trim().length > 0) {
    text = text.slice(leading);
    shift = -leading;
  }
  let outRanges = ranges.map((r) => ({ start: r.start + shift, end: r.end + shift }));
  if (text.length <= maxLength) {
    return { text, ranges: outRanges.filter((r) => r.end > 0 && r.start < text.length) };
  }
  const first = outRanges.find((r) => r.end > 0) ?? { start: 0, end: 0 };
  const rangeLen = Math.min(first.end - first.start, maxLength);
  let windowStart = Math.max(0, first.start - Math.floor((maxLength - rangeLen) / 2));
  windowStart = Math.min(windowStart, text.length - maxLength + 1);
  windowStart = Math.max(0, windowStart);
  const windowEnd = Math.min(text.length, windowStart + maxLength);
  let snippet = text.slice(windowStart, windowEnd);
  let prefix = 0;
  if (windowStart > 0) {
    snippet = `…${snippet}`;
    prefix = 1;
  }
  if (windowEnd < text.length) snippet = `${snippet}…`;
  outRanges = outRanges
    .map((r) => ({ start: r.start - windowStart + prefix, end: r.end - windowStart + prefix }))
    .filter((r) => r.end > 0 && r.start < snippet.length)
    .map((r) => ({ start: Math.max(0, r.start), end: Math.min(snippet.length, r.end) }));
  return { text: snippet, ranges: outRanges };
}

export interface GrepTextResult {
  matches: MatchedLine[];
  hitCount: number;
  /** 命中的词项数量（正则模式为 1）。 */
  matchedTerms: number;
}

/** 在单条文本中执行逐行扫描，输出命中行与上下文窗口。 */
export function grepText(
  text: string,
  plan: GrepPlan,
  options?: Pick<SearchOptions, 'contextLines' | 'maxMatchesPerMessage' | 'snippetLength'>,
): GrepTextResult {
  const contextLines = options?.contextLines ?? 2;
  const maxMatches = options?.maxMatchesPerMessage ?? 5;
  const empty: GrepTextResult = { matches: [], hitCount: 0, matchedTerms: 0 };
  if (!text || !quickHitCheck(text, plan)) return empty;

  const lines = text.split(/\r?\n/);
  const matches: MatchedLine[] = [];
  const matchedTerms = new Set<string>();
  let hitCount = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const ranges = scanLine(lines[i], plan);
    if (ranges.length === 0) continue;
    hitCount += ranges.length;
    if (!plan.regex) {
      for (const term of plan.terms) {
        if (findTermRanges(lines[i], term, plan.caseSensitive).length > 0) matchedTerms.add(term);
      }
    }
    if (matches.length < maxMatches) {
      const before: string[] = [];
      for (let b = Math.max(0, i - contextLines); b < i; b += 1) {
        before.push(truncateLine(lines[b]));
      }
      const after: string[] = [];
      for (let a = i + 1; a <= Math.min(lines.length - 1, i + contextLines); a += 1) {
        after.push(truncateLine(lines[a]));
      }
      matches.push({
        line: i + 1,
        text: truncateLine(lines[i]),
        before,
        after,
        ranges,
      });
    }
  }

  if (matches.length === 0) return empty;
  return {
    matches,
    hitCount,
    matchedTerms: plan.regex ? 1 : matchedTerms.size,
  };
}

interface ScoredResult {
  item: SearchResultItem;
  order: number;
}

function passesFilters(
  message: MessageItem,
  session: SessionRecord,
  options: ReturnType<typeof normalizeSearchOptions>,
): boolean {
  if (options.roles && options.roles.length > 0 && !options.roles.includes(message.role)) {
    return false;
  }
  if (options.kinds && options.kinds.length > 0 && !options.kinds.includes(message.kind)) {
    return false;
  }
  if (options.sessionIds && options.sessionIds.length > 0 && !options.sessionIds.includes(session.id)) {
    return false;
  }
  if (options.cwdPrefix && !(session.cwd ?? '').startsWith(options.cwdPrefix)) return false;
  if (options.since !== undefined && message.time < options.since) return false;
  if (options.until !== undefined && message.time > options.until) return false;
  return true;
}

/**
 * 关键词检索：跨会话逐消息扫描。
 * 得分 = 命中词项数 + min(命中次数, 100) * 0.01，标题命中额外 +1，用于排序而非阈值过滤。
 */
export class GrepEngine {
  search(sessions: SessionRecord[], options: SearchOptions): SearchResponse {
    const startedAt = Date.now();
    const opts = normalizeSearchOptions(options);
    const plan = compileGrepPlan(opts.query, opts);
    const results: ScoredResult[] = [];
    let scannedMessages = 0;
    let order = 0;

    if (plan.regex ? !plan.regexSource : plan.terms.length === 0) {
      return {
        query: opts.query,
        mode: 'keyword',
        results: [],
        total: 0,
        returned: 0,
        offset: opts.offset,
        tookMs: Date.now() - startedAt,
        scannedSessions: 0,
        scannedMessages: 0,
      };
    }

    for (const session of sessions) {
      if (opts.sessionIds?.length && !opts.sessionIds.includes(session.id)) continue;
      if (opts.cwdPrefix && !(session.cwd ?? '').startsWith(opts.cwdPrefix)) continue;
      for (const message of session.messages) {
        if (!passesFilters(message, session, opts)) continue;
        scannedMessages += 1;
        const grep = grepText(message.text, plan, opts);
        if (grep.matches.length === 0) continue;
        const first = grep.matches[0];
        const snippet = buildSnippet(first.text, first.ranges, opts.snippetLength);
        const score =
          grep.matchedTerms + Math.min(grep.hitCount, 100) * 0.01 + (message.kind === 'title' ? 1 : 0);
        results.push({
          order: (order += 1),
          item: {
            sessionId: session.id,
            filePath: session.filePath,
            title: session.title,
            cwd: session.cwd,
            messageId: message.id,
            seq: message.seq,
            role: message.role,
            kind: message.kind,
            time: message.time,
            score: Number(score.toFixed(4)),
            snippet: snippet.text,
            ranges: snippet.ranges,
            matches: grep.matches,
            source: 'grep',
            toolName: message.toolName,
          },
        });
      }
    }

    results.sort(
      (a, b) =>
        b.item.score - a.item.score ||
        b.item.time - a.item.time ||
        a.order - b.order,
    );
    const total = results.length;
    const page = results.slice(opts.offset, opts.offset + opts.limit);

    return {
      query: opts.query,
      mode: 'keyword',
      results: page.map((r) => r.item),
      total,
      returned: page.length,
      offset: opts.offset,
      tookMs: Date.now() - startedAt,
      scannedSessions: sessions.length,
      scannedMessages,
    };
  }
}

export function keywordSearch(sessions: SessionRecord[], options: SearchOptions): SearchResponse {
  return new GrepEngine().search(sessions, options);
}
