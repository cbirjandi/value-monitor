import type { WebWorkerMLCEngine } from "@mlc-ai/web-llm"
import { systemPromptOf, type Engine, type Piece } from "./engine.ts"

export const LOCAL_MODELS = [
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 0.5B", note: "about 400 MB, fastest" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 1.5B", note: "about 1 GB, writes better" },
]

export const hasWebGPU = () => "gpu" in navigator

export type Progress = { fraction: number; text: string; done: boolean; error?: string }

/** One loaded model is kept for the whole visit, even if the settings are reopened. */
let loaded: { id: string; engine: Promise<WebWorkerMLCEngine> } | null = null
/** The model writes one reply at a time; later requests wait their turn. */
let tail: Promise<void> = Promise.resolve()

export function loadModel(id: string, onProgress: (p: Progress) => void): Promise<WebWorkerMLCEngine> {
  if (loaded?.id === id) return loaded.engine
  const engine = (async () => {
    if (!hasWebGPU()) throw new Error("This browser can't run models locally (no WebGPU). Use a current Chrome or Edge.")
    const { CreateWebWorkerMLCEngine } = await import("@mlc-ai/web-llm")
    const worker = new Worker(new URL("./llm-worker.ts", import.meta.url), { type: "module" })
    return CreateWebWorkerMLCEngine(worker, id, { initProgressCallback: (r) => onProgress({ fraction: r.progress, text: r.text, done: r.progress >= 1 }) })
  })()
  loaded = { id, engine }
  engine.then(
    () => onProgress({ fraction: 1, text: "Ready", done: true }),
    (e) => {
      loaded = null
      onProgress({ fraction: 0, text: "", done: true, error: e instanceof Error ? e.message : String(e) })
    },
  )
  return engine
}

/** A small model running in this browser writes the replies, with every token's probability. Scoring, if any, comes from `scorer`. */
export function localEngine(id: string, scorer: Engine | null, onProgress: (p: Progress) => void): Engine {
  const name = LOCAL_MODELS.find((m) => m.id === id)?.name ?? id
  const ready = loadModel(id, onProgress)
  return {
    label: `${name} · in browser`,
    kind: "local",
    canBranch: true,
    score: scorer?.score,
    drivers: scorer?.drivers,
    async *write(req, signal, prefix = "") {
      const llm = await ready
      const release = await turn()
      const stop = () => llm.interruptGenerate()
      signal.addEventListener("abort", stop)
      try {
        if (signal.aborted) return
        const prompt = `<|im_start|>system\n${systemPromptOf(req)}<|im_end|>\n<|im_start|>user\n${req.prompt}<|im_end|>\n<|im_start|>assistant\n${prefix}`
        const stream = await llm.completions.create({
          prompt,
          stream: true,
          logprobs: true,
          top_logprobs: 5,
          temperature: 1,
          top_p: 0.9,
          max_tokens: 200,
          stop: ["<|im_end|>", "<|endoftext|>"],
        })
        for await (const chunk of stream) {
          if (signal.aborted) break
          const choice = chunk.choices[0]
          if (!choice?.text) continue
          const tokens = choice.logprobs?.content ?? []
          if (tokens.length && tokens.map((t) => t.token).join("") === choice.text) for (const t of tokens) yield pieceOf(t)
          else yield tokens.length ? { ...pieceOf(tokens[tokens.length - 1]), text: choice.text } : { text: choice.text }
        }
      } finally {
        signal.removeEventListener("abort", stop)
        release()
      }
    },
  }
}

type TokenLogprob = { token: string; logprob: number; top_logprobs?: { token: string; logprob: number }[] }

function pieceOf(t: TokenLogprob): Piece {
  return { text: t.token, p: Math.exp(t.logprob), alts: (t.top_logprobs ?? []).map((a) => ({ text: a.token, p: Math.exp(a.logprob) })) }
}

function turn(): Promise<() => void> {
  let release!: () => void
  const mine = new Promise<void>((r) => (release = r))
  const before = tail
  tail = before.then(() => mine)
  return before.then(() => release)
}
