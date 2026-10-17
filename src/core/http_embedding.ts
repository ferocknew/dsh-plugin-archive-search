/**
 * HTTP Embedding Provider: 连接 OpenAI / Ollama / SiliconFlow / 兼容网关的标准 /v1/embeddings 接口。
 * 纯原生 fetch 实现，零 C++、零 SQLite、零本地 ONNX 依赖。
 */
import type { EmbeddingProvider } from './types.js'

export interface HttpEmbeddingConfig {
  /** 接口完整 URL，如 "http://192.168.50.203:11434/v1/embeddings" */
  url?: string
  /** 模型名称，如 "qwen3-embedding:0.6b", "bge-m3" */
  model?: string
  /** 鉴权 Bearer Key（本地 Ollama 不需要） */
  apiKey?: string
  /** 显式声明的维度；若不传则在首次探测时自适应获取 */
  dimensions?: number
  /** 单批次最大文本条数，默认 16 */
  batchSize?: number
  /** 请求超时时间，默认 15000ms */
  timeoutMs?: number
}

export const DEFAULT_EMBEDDING_URL = 'http://192.168.50.203:11434/v1/embeddings'
export const DEFAULT_EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
export const DEFAULT_DIMENSIONS = 1024

export class HttpEmbeddingProvider implements EmbeddingProvider {
  readonly id: string
  readonly url: string
  readonly model: string
  readonly apiKey?: string
  readonly batchSize: number
  readonly timeoutMs: number
  dimensions: number

  constructor(config: HttpEmbeddingConfig = {}) {
    this.url = config.url?.trim() || process.env.DSH_ARCHIVE_EMBEDDING_URL?.trim() || DEFAULT_EMBEDDING_URL
    this.model = config.model?.trim() || process.env.DSH_ARCHIVE_EMBEDDING_MODEL?.trim() || DEFAULT_EMBEDDING_MODEL
    this.apiKey = config.apiKey?.trim() || process.env.DSH_ARCHIVE_EMBEDDING_KEY?.trim() || undefined
    this.dimensions = config.dimensions ?? DEFAULT_DIMENSIONS
    this.batchSize = config.batchSize ?? 16
    this.timeoutMs = config.timeoutMs ?? 15000
    this.id = `http:${this.model}@${new URL(this.url).host}`
  }

  /**
   * 探测服务连通性并自动校准向量维度。
   * 成功返回 true，不可用返回 false（静默降级）。
   */
  async probe(): Promise<boolean> {
    try {
      const vecs = await this.fetchEmbeddings(['ping'])
      if (vecs.length > 0 && vecs[0].length > 0) {
        this.dimensions = vecs[0].length
        return true
      }
      return false
    } catch {
      return false
    }
  }

  /** 批量生成文本的 Float32Array 嵌入向量 */
  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return []
    const results: Float32Array[] = []

    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize)
      const vecs = await this.fetchEmbeddings(batch)
      results.push(...vecs)
    }

    return results
  }

  private async fetchEmbeddings(inputs: string[]): Promise<Float32Array[]> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)

    try {
      const headers: Record<string, string> = {
        'content-type': 'application/json',
      }
      if (this.apiKey) {
        headers['authorization'] = `Bearer ${this.apiKey}`
      }

      const res = await fetch(this.url, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: this.model,
          input: inputs,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(`Embedding API HTTP ${res.status}: ${text.slice(0, 200)}`)
      }

      const data = (await res.json()) as any
      if (!data || !Array.isArray(data.data)) {
        throw new Error('Embedding API returned invalid envelope: missing data array')
      }

      // 按 index 排序确保对应顺序一致
      const sorted = [...data.data].sort((a: any, b: any) => (a.index ?? 0) - (b.index ?? 0))
      return sorted.map((item: any) => Float32Array.from(item.embedding))
    } finally {
      clearTimeout(timer)
    }
  }
}
