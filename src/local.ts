import type { WebWorkerMLCEngine } from "@mlc-ai/web-llm"
import { systemPromptOf, type Alt, type Engine, type Piece, type WriteRequest } from "./engine.ts"

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

/** Small models follow an example far better than an instruction, so every prompt carries one short exchange in the wanted style. */
const EXAMPLE_ASK = "My coworker keeps taking credit for my ideas in meetings. What should I do?"
const EXAMPLE_REPLY =
  "Start by sending a short recap email after you share an idea, so there is a record with your name on it. If it keeps happening, raise it with your coworker directly and calmly. Bring in your manager only if that doesn't work."

/** The model's own chat format, with the example exchange, ending where the reply (and any forced beginning) goes. */
function promptOf(req: WriteRequest, prefix: string) {
  return [
    `<|im_start|>system\n${systemPromptOf(req)}<|im_end|>`,
    `<|im_start|>user\n${EXAMPLE_ASK}<|im_end|>`,
    `<|im_start|>assistant\n${EXAMPLE_REPLY}<|im_end|>`,
    `<|im_start|>user\n${req.prompt}<|im_end|>`,
    `<|im_start|>assistant\n${prefix}`,
  ].join("\n")
}

/** Sampling is mildly focused, so differences between replies come mostly from the prompt rather than chance. */
const SAMPLING = { temperature: 0.7, top_p: 0.9, repetition_penalty: 1.1 }

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
      let finished = false
      try {
        if (signal.aborted) return
        const stream = await llm.completions.create({
          prompt: promptOf(req, prefix),
          stream: true,
          logprobs: true,
          top_logprobs: 5,
          ...SAMPLING,
          max_tokens: 140,
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
        finished = true
      } finally {
        // The reader may stop early (the reply is long enough); stop the model too, so the next request isn't kept waiting.
        if (!finished) stop()
        signal.removeEventListener("abort", stop)
        release()
      }
    },
    async probe(req, prefix, signal) {
      const llm = await ready
      const release = await turn()
      try {
        if (signal.aborted) return []
        const out = await llm.completions.create({ prompt: promptOf(req, prefix), stream: false, logprobs: true, top_logprobs: 5, ...SAMPLING, max_tokens: 1 })
        const first = out.choices[0]?.logprobs?.content?.[0]
        return (first?.top_logprobs ?? []).map((a) => ({ text: a.token, p: Math.exp(a.logprob) })).sort((a, b) => b.p - a.p)
      } finally {
        release()
      }
    },
  }
}

export type { Alt }

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
