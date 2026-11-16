import type { WebWorkerMLCEngine } from "@mlc-ai/web-llm"
import { systemPromptOf, type Alt, type Engine, type Piece, type WriteRequest } from "./engine.ts"

type Family = "qwen" | "llama" | "gemma" | "phi" | "mistral"

/** Models WebLLM can run in a browser, smallest first. Sizes are roughly the download and the graphics memory needed. */
export const LOCAL_MODELS: { id: string; name: string; note: string; family: Family }[] = [
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 0.5B", note: "0.4 GB · fastest, weakest", family: "qwen" },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", name: "Llama 3.2 1B", note: "0.9 GB · Meta", family: "llama" },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 1.5B", note: "1 GB", family: "qwen" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", name: "Gemma 2 2B", note: "1.9 GB · Google", family: "gemma" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", name: "Llama 3.2 3B", note: "2.3 GB · Meta, good balance", family: "llama" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", name: "Qwen2.5 3B", note: "2.5 GB", family: "qwen" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", name: "Phi 3.5 mini", note: "3.7 GB · Microsoft", family: "phi" },
  { id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC", name: "Mistral 7B", note: "4.6 GB · Mistral AI", family: "mistral" },
  { id: "Qwen2.5-7B-Instruct-q4f16_1-MLC", name: "Qwen2.5 7B", note: "5.1 GB", family: "qwen" },
  { id: "Llama-3.1-8B-Instruct-q4f16_1-MLC", name: "Llama 3.1 8B", note: "5 GB · best writing, needs 16 GB of memory", family: "llama" },
]

export const hasWebGPU = () => "gpu" in navigator

export type Progress = { fraction: number; text: string; done: boolean; error?: string }

/** One loaded model is kept for the whole visit, even if the settings are reopened. */
let loaded: { id: string; engine: Promise<WebWorkerMLCEngine> } | null = null
/** The model writes one reply at a time; later requests wait their turn. */
let tail: Promise<void> = Promise.resolve()

export function loadModel(id: string, onProgress: (p: Progress) => void): Promise<WebWorkerMLCEngine> {
  if (loaded?.id === id) return loaded.engine
  // Switching models frees the old one's graphics memory first.
  loaded?.engine.then((old) => old.unload()).catch(() => {})
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

/**
 * Each family's own chat format, with the example exchange, ending where the reply (and any forced beginning) goes.
 * WebLLM adds the model's start-of-text token itself, so it is not written here.
 */
const FORMATS: Record<Family, { prompt: (system: string, ask: string, prefix: string) => string; stop: string[] }> = {
  qwen: {
    prompt: (system, ask, prefix) =>
      `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${EXAMPLE_ASK}<|im_end|>\n<|im_start|>assistant\n${EXAMPLE_REPLY}<|im_end|>\n<|im_start|>user\n${ask}<|im_end|>\n<|im_start|>assistant\n${prefix}`,
    stop: ["<|im_end|>", "<|endoftext|>"],
  },
  llama: {
    prompt: (system, ask, prefix) =>
      `<|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${EXAMPLE_ASK}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n${EXAMPLE_REPLY}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${ask}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n${prefix}`,
    stop: ["<|eot_id|>", "<|end_of_text|>"],
  },
  gemma: {
    // Gemma has no system role; instructions go in the first user turn.
    prompt: (system, ask, prefix) =>
      `<start_of_turn>user\n${system}\n\n${EXAMPLE_ASK}<end_of_turn>\n<start_of_turn>model\n${EXAMPLE_REPLY}<end_of_turn>\n<start_of_turn>user\n${ask}<end_of_turn>\n<start_of_turn>model\n${prefix}`,
    stop: ["<end_of_turn>", "<eos>"],
  },
  phi: {
    prompt: (system, ask, prefix) =>
      `<|system|>\n${system}<|end|>\n<|user|>\n${EXAMPLE_ASK}<|end|>\n<|assistant|>\n${EXAMPLE_REPLY}<|end|>\n<|user|>\n${ask}<|end|>\n<|assistant|>\n${prefix}`,
    stop: ["<|end|>", "<|endoftext|>"],
  },
  mistral: {
    // Mistral has no system role either; instructions lead the first instruction block.
    prompt: (system, ask, prefix) => `[INST] ${system}\n\n${EXAMPLE_ASK}[/INST] ${EXAMPLE_REPLY}</s>[INST] ${ask}[/INST] ${prefix}`,
    stop: ["</s>", "[INST]"],
  },
}

function formatOf(id: string) {
  return FORMATS[LOCAL_MODELS.find((m) => m.id === id)?.family ?? "qwen"]
}

/** Sampling is mildly focused, so differences between replies come mostly from the prompt rather than chance. */
const SAMPLING = { temperature: 0.7, top_p: 0.9, repetition_penalty: 1.1 }

/** A small model running in this browser writes the replies, with every token's probability. Scoring, if any, comes from `scorer`. */
export function localEngine(id: string, scorer: Engine | null, onProgress: (p: Progress) => void): Engine {
  const name = LOCAL_MODELS.find((m) => m.id === id)?.name ?? id
  const format = formatOf(id)
  const promptOf = (req: WriteRequest, prefix: string) => format.prompt(systemPromptOf(req), req.prompt, prefix)
  const ready = loadModel(id, onProgress)
  return {
    label: `${name} · in browser`,
    kind: "local",
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
          max_tokens: 160,
          stop: format.stop,
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
