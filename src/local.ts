import type { WebWorkerMLCEngine } from "@mlc-ai/web-llm"
import { DEFAULT_TEMPERATURE, systemPromptOf, type Alt, type Engine, type Piece, type WriteRequest } from "./engine.ts"
import { valueOf, type ValueId } from "./values.ts"

type Family = "qwen" | "chatml" | "llama" | "gemma" | "phi" | "mistral" | "zephyr"

/** Models WebLLM can run in a browser, smallest first. Sizes are roughly the download and the graphics memory needed. */
/** `small` models get an example exchange in every prompt; larger ones follow instructions without it, and shorter prompts run faster. */
export const LOCAL_MODELS: { id: string; name: string; note: string; family: Family; steerable?: boolean; small?: boolean }[] = [
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 0.5B", note: "0.4 GB · fastest, weakest", family: "qwen", small: true },
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", name: "Llama 3.2 1B", note: "0.9 GB · Meta", family: "llama", small: true },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen2.5 1.5B", note: "1 GB", family: "qwen", small: true },
  { id: "gemma-2-2b-it-q4f16_1-MLC", name: "Gemma 2 2B", note: "1.9 GB · Google", family: "gemma" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", name: "Llama 3.2 3B", note: "2.3 GB · Meta, good balance", family: "llama" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", name: "Qwen2.5 3B", note: "2.5 GB", family: "qwen" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", name: "Phi 3.5 mini", note: "3.7 GB · Microsoft", family: "phi" },
  { id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC", name: "Mistral 7B", note: "4.6 GB · Mistral AI", family: "mistral" },
  { id: "Qwen2.5-7B-Instruct-q4f16_1-MLC", name: "Qwen2.5 7B", note: "5.1 GB", family: "qwen" },
  { id: "Llama-3.1-8B-Instruct-q4f16_1-MLC", name: "Llama 3.1 8B", note: "5 GB · best writing, needs 16 GB of memory", family: "llama" },
  // Nous Research's Hermes models are tuned to follow the system prompt rather than refuse on moral grounds,
  // so the values chosen here steer them much further than they steer the safety-tuned models above.
  { id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC", name: "TinyLlama 1.1B", note: "0.7 GB · fast, lightly tuned, weak writing", family: "zephyr", steerable: true, small: true },
  { id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC", name: "Hermes 3 3B", note: "2.3 GB · steerable, few refusals", family: "chatml", steerable: true },
  { id: "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC", name: "Hermes 2 Pro 7B", note: "4 GB · steerable, few refusals", family: "chatml", steerable: true },
  { id: "Hermes-3-Llama-3.1-8B-q4f16_1-MLC", name: "Hermes 3 8B", note: "4.9 GB · steerable, few refusals, needs 16 GB", family: "chatml", steerable: true },
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

type Turn = { role: "user" | "assistant"; text: string }

/**
 * Each family's own chat format: the conversation so far, then an open assistant turn that the reply (and any forced beginning) goes into.
 * WebLLM adds the model's start-of-text token itself, so it is not written here.
 */
const FORMATS: Record<Family, { render: (system: string, turns: Turn[], open: string) => string; stop: string[] }> = {
  qwen: {
    render: (system, turns, open) =>
      `<|im_start|>system\n${system}<|im_end|>\n` + turns.map((t) => `<|im_start|>${t.role}\n${t.text}<|im_end|>\n`).join("") + `<|im_start|>assistant\n${open}`,
    stop: ["<|im_end|>", "<|endoftext|>"],
  },
  // Hermes uses the same ChatML layout as Qwen, on Llama and Mistral tokenizers.
  chatml: {
    render: (system, turns, open) =>
      `<|im_start|>system\n${system}<|im_end|>\n` + turns.map((t) => `<|im_start|>${t.role}\n${t.text}<|im_end|>\n`).join("") + `<|im_start|>assistant\n${open}`,
    stop: ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"],
  },
  llama: {
    render: (system, turns, open) =>
      `<|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id|>` +
      turns.map((t) => `<|start_header_id|>${t.role}<|end_header_id|>\n\n${t.text}<|eot_id|>`).join("") +
      `<|start_header_id|>assistant<|end_header_id|>\n\n${open}`,
    stop: ["<|eot_id|>", "<|end_of_text|>"],
  },
  gemma: {
    // Gemma has no system role; instructions lead the first user turn.
    render: (system, turns, open) =>
      turns.map((t, i) => `<start_of_turn>${t.role === "user" ? "user" : "model"}\n${i === 0 ? `${system}\n\n` : ""}${t.text}<end_of_turn>\n`).join("") + `<start_of_turn>model\n${open}`,
    stop: ["<end_of_turn>", "<eos>"],
  },
  phi: {
    render: (system, turns, open) => `<|system|>\n${system}<|end|>\n` + turns.map((t) => `<|${t.role}|>\n${t.text}<|end|>\n`).join("") + `<|assistant|>\n${open}`,
    stop: ["<|end|>", "<|endoftext|>"],
  },
  // TinyLlama chat uses the Zephyr layout.
  zephyr: {
    render: (system, turns, open) => `<|system|>\n${system}</s>\n` + turns.map((t) => `<|${t.role}|>\n${t.text}</s>\n`).join("") + `<|assistant|>\n${open}`,
    stop: ["</s>", "<|user|>"],
  },
  mistral: {
    // Mistral has no system role either; instructions lead the first instruction block.
    render: (system, turns, open) =>
      turns.map((t, i) => (t.role === "user" ? `[INST] ${i === 0 ? `${system}\n\n` : ""}${t.text}[/INST]` : ` ${t.text}</s>`)).join("") + ` ${open}`,
    stop: ["</s>", "[INST]"],
  },
}

function formatOf(id: string) {
  return FORMATS[LOCAL_MODELS.find((m) => m.id === id)?.family ?? "qwen"]
}

const KEEP_GOING = "Please continue your answer with two or three more sentences that add new detail, reasons or next steps. Don't repeat what you already said."

/**
 * The conversation for a request. A normal request leaves the reply's beginning (`prefix`) in the open turn for the model to finish.
 * Asking for more puts the whole reply in a finished turn and asks, in a new one, to keep going: small models often just stop otherwise.
 */
function conversationOf(req: WriteRequest, prefix: string, withExample: boolean): { turns: Turn[]; open: string } {
  const turns: Turn[] = [
    ...(withExample ? ([{ role: "user", text: EXAMPLE_ASK }, { role: "assistant", text: EXAMPLE_REPLY }] as Turn[]) : []),
    { role: "user", text: req.prompt },
  ]
  if (req.more && prefix.trim()) return { turns: [...turns, { role: "assistant", text: prefix.trim() }, { role: "user", text: KEEP_GOING }], open: "" }
  return { turns, open: prefix }
}

/**
 * Plain sampling with no repetition or frequency penalties, so the probability shown for a word while writing is the same
 * distribution the no-values check reads afterward; the two can be compared directly.
 */
const samplingOf = (req: WriteRequest) => ({ temperature: req.temperature ?? DEFAULT_TEMPERATURE, top_p: 0.9, repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0 })

/** Longest an amplified reply runs per request, in words. */
const AMPLIFIED_MAX_TOKENS = 90

/** A small model running in this browser writes the replies, with every token's probability. Scoring, if any, comes from `scorer`. */
/**
 * `judge` decides who writes the comparison: another engine (Gemini), this same model ("self"), or nobody.
 */
export function localEngine(id: string, judge: Engine | "self" | null, onProgress: (p: Progress) => void): Engine {
  const model = LOCAL_MODELS.find((m) => m.id === id)
  const name = model?.name ?? id
  const format = formatOf(id)
  const promptOf = (req: WriteRequest, prefix: string) => {
    const { turns, open } = conversationOf(req, prefix, model?.small ?? true)
    return format.render(systemPromptOf({ ...req, more: false }), turns, open)
  }
  const ready = loadModel(id, onProgress)
  /** The model's top five next tokens for a prompt, most likely first, at the request's temperature. */
  async function nextWords(llm: Llm, prompt: string, req: WriteRequest): Promise<Alt[]> {
    const out = await llm.completions.create({ prompt, stream: false, logprobs: true, top_logprobs: 5, ...samplingOf(req), max_tokens: 1, stop: format.stop })
    const first = out.choices[0]?.logprobs?.content?.[0]
    return (first?.top_logprobs ?? []).map((a) => ({ text: a.token, p: Math.exp(a.logprob) })).sort((a, b) => b.p - a.p)
  }

  /**
   * Value amplification, after classifier-free guidance: at each step, read the next-word probabilities with the values (pv)
   * and without them (p0), score every candidate as log pv + γ·(log pv − log p0), and sample from those scores.
   * γ = 0 is ordinary writing with values; larger γ exaggerates whatever the values change. Only the top five candidates
   * of each are known, so a word missing from one list is given that list's smallest probability.
   * Each word costs two model calls, so this writes more slowly.
   */
  async function* amplified(llm: Llm, req: WriteRequest, prefix: string, signal: AbortSignal): AsyncIterable<Piece> {
    const gamma = req.amplify ?? 0
    const plainReq: WriteRequest = { ...req, values: req.against ?? [], blend: undefined, amplify: 0 }
    let text = prefix
    let needsSpace = !!req.more && !!prefix && !/\s$/.test(prefix)
    for (let step = 0; step < AMPLIFIED_MAX_TOKENS && !signal.aborted; step++) {
      const withValues = await nextWords(llm, promptOf(req, text), req)
      const without = await nextWords(llm, promptOf(plainReq, text), req)
      if (!withValues.length) return
      const floorV = withValues.at(-1)!.p
      const floor0 = without.at(-1)?.p ?? floorV
      const candidates = [...new Set([...withValues, ...without].map((a) => a.text))]
      const scored = candidates.map((t) => {
        const pv = withValues.find((a) => a.text === t)?.p ?? floorV
        const p0 = without.find((a) => a.text === t)?.p ?? floor0
        return { text: t, score: Math.log(pv) + gamma * (Math.log(pv) - Math.log(p0)) }
      })
      const max = Math.max(...scored.map((c) => c.score))
      const weights = scored.map((c) => Math.exp(c.score - max))
      const total = weights.reduce((a, b) => a + b, 0)
      const dist: Alt[] = scored.map((c, i) => ({ text: c.text, p: weights[i] / total })).sort((a, b) => b.p - a.p)
      let r = Math.random()
      const pick = dist.find((a) => (r -= a.p) <= 0) ?? dist[0]
      if (!pick.text || isStop(pick.text)) return
      let out = pick.text
      if (needsSpace && out.trim()) {
        needsSpace = false
        if (!/^\s/.test(out)) out = ` ${out}`
      }
      text += pick.text
      yield { text: out, p: pick.p, alts: dist.slice(0, 6), plain: without, amplified: gamma }
    }
  }

  async function selfCompare(ask: string, a: { text: string; values: ValueId[] }, b: { text: string; values: ValueId[] }, signal: AbortSignal) {
    const llm = await ready
    const release = await turn()
    try {
      if (signal.aborted) return []
      const question = [
        `A person asked an AI assistant:\n"${ask}"`,
        `Reply A (${side(a.values)}):\n"${a.text}"`,
        `Reply B (${side(b.values)}):\n"${b.text}"`,
        "In two or three short sentences, say how Reply B differs in substance from Reply A: the advice it gives, risks it raises or leaves out, and whom it serves. Start every sentence with \"Reply B\". Plain prose, no lists.",
      ].join("\n\n")
      const out = await llm.completions.create({
        prompt: format.render("You are a careful, neutral analyst.", [{ role: "user", text: question }], ""),
        stream: false,
        temperature: 0.3,
        max_tokens: 140,
        stop: format.stop,
      })
      const text = (out.choices[0]?.text ?? "").replace(/\*\*|^[-*•\d.)\s]+/gm, "").replace(/\s+/g, " ").trim()
      return (text.match(/[^.!?]+[.!?]+/g) ?? [text]).map((t) => t.trim()).filter((t) => t.length > 12).slice(0, 3)
    } finally {
      release()
    }
  }

  const side = (ids: ValueId[]) => (ids.length ? `told to put ${ids.map((id) => valueOf(id).name.toLowerCase()).join(", ")} first` : "no special values")
  const isStop = (t: string) => /<\|[^|]*\|>|<\/s>|<end_of_turn>|<eos>|\[INST\]/.test(t)

  return {
    label: `${name} · in browser`,
    kind: "local",
    judgeLabel: judge === "self" ? `${name}, in this browser` : judge ? judge.label : undefined,
    drivers: judge === "self" ? undefined : judge?.drivers,
    compare: judge === "self" ? selfCompare : judge?.compare,
    async *write(req, signal, prefix = "") {
      const llm = await ready
      const release = await turn()
      const stop = () => llm.interruptGenerate()
      signal.addEventListener("abort", stop)
      let finished = false
      try {
        if (signal.aborted) return
        if ((req.amplify ?? 0) > 0 && !req.blend && (req.values.length || req.against?.length)) {
          yield* amplified(llm, req, prefix, signal)
          finished = true
          return
        }
        const stream = await llm.completions.create({
          prompt: promptOf(req, prefix),
          stream: true,
          logprobs: true,
          top_logprobs: 5,
          ...samplingOf(req),
          max_tokens: 160,
          stop: format.stop,
        })
        // A new turn starts without a space, so the first piece of "more" gets one to join the reply.
        let needsSpace = !!req.more && !!prefix && !/\s$/.test(prefix)
        const joined = (piece: Piece): Piece => {
          if (!needsSpace || !piece.text.trim()) return piece
          needsSpace = false
          return /^\s/.test(piece.text) ? piece : { ...piece, text: ` ${piece.text}` }
        }
        for await (const chunk of stream) {
          if (signal.aborted) break
          const choice = chunk.choices[0]
          if (!choice?.text) continue
          const tokens = choice.logprobs?.content ?? []
          if (tokens.length && tokens.map((t) => t.token).join("") === choice.text) for (const t of tokens) yield joined(pieceOf(t))
          else yield joined(tokens.length ? { ...pieceOf(tokens[tokens.length - 1]), text: choice.text } : { text: choice.text })
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
        return await nextWords(llm, promptOf(req, prefix), req)
      } finally {
        release()
      }
    },
  }
}

export type { Alt }

type TokenLogprob = { token: string; logprob: number; top_logprobs?: { token: string; logprob: number }[] }
type Llm = WebWorkerMLCEngine


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
