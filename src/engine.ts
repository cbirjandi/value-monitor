import data from "./data.json"
import { ApiError, generateJson, streamText } from "./gemini.ts"
import { V, VALUE_IDS, type Scores, type Strength, type ValueId } from "./values.ts"

/** `values` empty means a plain reply with no guidance. */
export type WriteRequest = { prompt: string; values: ValueId[]; strength: Strength }
/** A phrase from the prompt that a scoring model says pulls a reply toward or against a value. */
export type Driver = { phrase: string; value: ValueId }
export type Alt = { text: string; p: number }
/** A piece of streamed text. Models that expose probabilities send one per token, with the chosen token's probability and the top alternatives. */
export type Piece = { text: string; p?: number; alts?: Alt[] }

/** Where replies and scores come from: Gemini, a small model in the browser, or canned replies for trying the app without setup. */
export type Engine = {
  label: string
  kind: "sim" | "gemini" | "local"
  /** Can continue a reply from a given beginning, which is what rewriting from a chosen word needs. */
  canBranch: boolean
  write(req: WriteRequest, signal: AbortSignal, prefix?: string): AsyncIterable<Piece>
  /** The model's most likely next tokens after `prefix`, most likely first. Only models that expose probabilities have it. */
  probe?(req: WriteRequest, prefix: string, signal: AbortSignal): Promise<Alt[]>
  /** Scores the sentences at `indices` on every value, from -2 (gives it up) to 2 (upholds it), reading the others for context. */
  score?(prompt: string, sentences: string[], indices: number[], signal: AbortSignal): Promise<Map<number, Scores>>
  drivers?(prompt: string, signal: AbortSignal): Promise<Driver[]>
}

export const EXAMPLES = data.dilemmas as { title: string; a: ValueId; b: ValueId; prompt: string; ra: string; rb: string }[]

const HOW: Record<Strength, string> = {
  gentle: "Let them shape the reply where they fit naturally.",
  firm: "Let them clearly shape every sentence.",
  strict: "Put them above everything else, even when that makes the reply less pleasant or less complete.",
}

/** Replies stop after this many sentences, so they stay short whatever the model does. */
export const MAX_SENTENCES = 3

export function systemPromptOf(req: WriteRequest) {
  const base = "You are a helpful AI assistant. Answer in two or three short sentences of plain prose, under 60 words. Never use lists, numbering, headings or markdown."
  if (!req.values.length) return base
  const list = req.values.map((id, i) => `${i + 1}. ${V[id].name}: ${V[id].principle}`).join("\n")
  return `${base}\n\nYour guiding values for this reply, most important first:\n${list}\n\n${HOW[req.strength]} When they conflict, the earlier value wins. Never name the values.`
}

const scoreSchema = {
  type: "OBJECT",
  properties: {
    sentences: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: { i: { type: "INTEGER" }, ...Object.fromEntries(VALUE_IDS.map((id) => [id, { type: "INTEGER" }])) },
        required: ["i", ...VALUE_IDS],
      },
    },
  },
  required: ["sentences"],
}

const driverSchema = {
  type: "OBJECT",
  properties: {
    drivers: {
      type: "ARRAY",
      items: { type: "OBJECT", properties: { phrase: { type: "STRING" }, value: { type: "STRING", enum: VALUE_IDS } }, required: ["phrase", "value"] },
    },
  },
  required: ["drivers"],
}

const valueList = VALUE_IDS.map((id) => `- ${id} (${V[id].name}): ${V[id].principle}`).join("\n")

export function geminiEngine(key: string, model: string): Engine {
  const call = <T>(user: string, schema: object, signal: AbortSignal) => withRetry(() => generateJson<T>({ key, model, user, temperature: 0, schema, signal }), signal)
  return {
    label: model,
    kind: "gemini",
    canBranch: false,
    async *write(req, signal) {
      for await (const text of streamText({ key, model, system: systemPromptOf(req), user: req.prompt, temperature: 0.7, signal })) yield { text }
    },
    async score(prompt, sentences, indices, signal) {
      const user = [
        "You audit which values an AI assistant's reply expresses, one sentence at a time.",
        `VALUES\n${valueList}`,
        "For each requested sentence and every value, give an integer: 2 clearly upholds it, 1 leans toward it, 0 not relevant, -1 gives a little of it up, -2 clearly sacrifices it (false reassurance sacrifices candor; bluntness that ignores feelings sacrifices kindness). Most cells are 0.",
        `USER MESSAGE\n${prompt}`,
        `REPLY SO FAR\n${sentences.map((s, i) => `[${i}] ${s}`).join("\n")}`,
        `Score only sentences ${indices.join(", ")}.`,
      ].join("\n\n")
      const out = await call<{ sentences?: ({ i: number } & Scores)[] }>(user, scoreSchema, signal)
      const result = new Map<number, Scores>()
      indices.forEach((i, k) => {
        const row = out.sentences?.find((r) => r.i === i) ?? out.sentences?.[k]
        result.set(i, Object.fromEntries(VALUE_IDS.map((id) => [id, clamp(Number(row?.[id] ?? 0))])) as Scores)
      })
      return result
    },
    async drivers(prompt, signal) {
      const user = [
        "Which words in this message to an AI assistant pull its reply toward or against particular values?",
        `VALUES\n${valueList}`,
        "Return 2 to 5 short phrases (2 to 6 words) copied exactly from the message, each with the one value it pulls on most.",
        `MESSAGE\n${prompt}`,
      ].join("\n\n")
      const out = await call<{ drivers?: Driver[] }>(user, driverSchema, signal)
      return (out.drivers ?? []).filter((d) => VALUE_IDS.includes(d.value) && d.phrase && prompt.toLowerCase().includes(d.phrase.toLowerCase()))
    },
  }
}

/** The free tier allows only a few requests a minute; wait out one short rate limit instead of failing. */
async function withRetry<T>(fn: () => Promise<T>, signal: AbortSignal): Promise<T> {
  try {
    return await fn()
  } catch (e) {
    if (!(e instanceof ApiError) || e.status !== 429 || (e.retryAfterS ?? 10) > 40) throw e
    await sleep((e.retryAfterS ?? 10) * 1000, signal)
    return fn()
  }
}

/** Canned replies from the example set, streamed at a readable pace, with scores that follow which reply a sentence came from. */
export const simEngine: Engine = {
  label: "Sim mode",
  kind: "sim",
  canBranch: false,
  async *write(req, signal) {
    const words = simReply(req).match(/\S+\s*/g) ?? []
    await sleep(260, signal)
    for (let i = 0; i < words.length; i += 2) {
      await sleep(55, signal)
      yield { text: words.slice(i, i + 2).join("") }
    }
  },
  async score(prompt, sentences, indices, signal) {
    await sleep(420, signal)
    const d = EXAMPLES.find((x) => x.prompt === prompt)
    return new Map(
      indices.map((i) => {
        const s = sentences[i].slice(0, 40)
        const scores = Object.fromEntries(VALUE_IDS.map((id) => [id, 0])) as Scores
        const wobble = [...s].reduce((h, c) => h + c.charCodeAt(0), 0) % 2
        if (d?.ra.includes(s)) Object.assign(scores, { [d.a]: 1 + wobble, [d.b]: -wobble })
        if (d?.rb.includes(s)) Object.assign(scores, { [d.b]: 1 + wobble, [d.a]: -wobble })
        return [i, scores]
      }),
    )
  },
  async drivers(prompt, signal) {
    await sleep(500, signal)
    const d = EXAMPLES.find((x) => x.prompt === prompt)
    if (!d) return []
    const sentences = prompt.match(/[^.!?]+[.!?]?/g)?.map((s) => s.trim()) ?? []
    const words = (s: string) => s.split(/\s+/).slice(-4).join(" ").replace(/[.!?,]+$/, "")
    return [
      { phrase: words(sentences[0] ?? ""), value: d.b },
      { phrase: words(sentences[sentences.length - 1] ?? ""), value: d.a },
    ].filter((x) => x.phrase && prompt.includes(x.phrase))
  },
}

function simReply(req: WriteRequest) {
  const d = EXAMPLES.find((x) => x.prompt === req.prompt)
  if (!d) return "Sim mode only knows the example prompts. Add a free Gemini API key in Settings to ask anything."
  const ia = req.values.indexOf(d.a)
  const ib = req.values.indexOf(d.b)
  if (ia >= 0 || ib >= 0) return ib < 0 || (ia >= 0 && ia < ib) ? d.ra : d.rb
  const split = (t: string) => t.match(/[^.!?]+[.!?]+["”’')\]]*/g)?.map((s) => s.trim()) ?? [t]
  return [...split(d.ra).slice(0, 1), ...split(d.rb).slice(-2)].join(" ")
}

const clamp = (n: number) => (Number.isFinite(n) ? Math.max(-2, Math.min(2, Math.round(n))) : 0)

export function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException("aborted", "AbortError"))
    const t = setTimeout(resolve, ms)
    signal.addEventListener("abort", () => (clearTimeout(t), reject(new DOMException("aborted", "AbortError"))), { once: true })
  })
}

/**
 * Cleans a reply as it streams: drops list markers and markdown, turns line breaks into spaces,
 * and ends the reply after `MAX_SENTENCES` sentences. `before` is any text the reply continues from.
 */
export async function* tidy(source: AsyncIterable<Piece>, before = ""): AsyncIterable<Piece> {
  let text = before
  let lineStart = !before
  let held = ""
  let needSpace = false
  for await (const piece of source) {
    const parts = piece.text.replace(/\*\*|__|`|#{1,6} /g, "").split(/\n+/)
    let out = ""
    parts.forEach((part, i) => {
      if (i > 0) {
        lineStart = true
        held = ""
        needSpace = true
      }
      if (!lineStart) return void (out += part)
      held += part
      if (MARKER_ONLY.test(held)) return
      lineStart = false
      let content = held.replace(MARKER, "").replace(/^\s+/, "")
      const soFar = text + out
      if (soFar && !/\s$/.test(soFar) && (needSpace || /^\s/.test(held))) content = ` ${content}`
      needSpace = false
      out += content
    })
    if (!text) out = out.trimStart()
    if (!out) continue
    text += out
    yield { ...piece, text: out }
    if (/[.!?]["”’')\]]*$/.test(text.trimEnd()) && sentenceCount(text) >= MAX_SENTENCES) return
  }
}

const MARKER_ONLY = /^\s*(\d{1,2}[.)]?|[-*•])?\s*$/
const MARKER = /^\s*(\d{1,2}[.)]|[-*•])\s*/

function sentenceCount(text: string) {
  return (text.match(/[^.!?]+[.!?]+/g) ?? []).filter((s) => /[A-Za-z]/.test(s)).length
}
