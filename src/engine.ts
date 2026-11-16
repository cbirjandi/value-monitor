import data from "./data.json"
import { ApiError, generateJson, streamText } from "./gemini.ts"
import { allValues, valueOf, type Scores, type Strength, type ValueId } from "./values.ts"

/** `values` empty means a plain reply with no guidance. `more` asks the model to keep going past a reply it may consider finished. */
export type WriteRequest = { prompt: string; values: ValueId[]; strength: Strength; more?: boolean }
/** A phrase from the prompt that a scoring model says pulls a reply toward or against a value. */
export type Driver = { phrase: string; value: ValueId }
export type Alt = { text: string; p: number }
/** A piece of streamed text. Models that expose probabilities send one per token, with the chosen token's probability and the top alternatives. */
export type Piece = { text: string; p?: number; alts?: Alt[] }

/** Where replies and scores come from: Gemini, a small model in the browser, or canned replies for trying the app without setup. */
export type Engine = {
  label: string
  kind: "sim" | "gemini" | "local"
  /** Continues a reply from `prefix` when given, which is how the future is rewritten from the cursor. */
  write(req: WriteRequest, signal: AbortSignal, prefix?: string): AsyncIterable<Piece>
  /** The model's most likely next tokens after `prefix`, most likely first. Only models that expose probabilities have it. */
  probe?(req: WriteRequest, prefix: string, signal: AbortSignal): Promise<Alt[]>
  /** Scores the sentences at `indices` on every value, reading the others for context. */
  score?(prompt: string, sentences: string[], indices: number[], signal: AbortSignal): Promise<Map<number, Scores>>
  drivers?(prompt: string, signal: AbortSignal): Promise<Driver[]>
}

export const EXAMPLES = data.dilemmas as { title: string; a: ValueId; b: ValueId; prompt: string; ra: string; rb: string }[]

const HOW: Record<Strength, string> = {
  gentle: "Let them shape the reply where they fit naturally.",
  firm: "Let them clearly shape every sentence.",
  strict: "Put them above everything else, even when that makes the reply less pleasant or less complete.",
}

const MORE = "The reply is not finished yet. Continue it with the next two or three sentences, adding new detail, reasons or next steps. Do not repeat, summarize or wrap up."

export function systemPromptOf(req: WriteRequest) {
  const base = "You are a helpful AI assistant. Answer in plain prose, in short sentences. Never use lists, numbering, headings or markdown."
  const more = req.more ? `\n\n${MORE}` : ""
  if (!req.values.length) return base + more
  const list = req.values.map((id, i) => `${i + 1}. ${valueOf(id).name}: ${valueOf(id).principle}`).join("\n")
  return `${base}\n\nYour guiding values for this reply, most important first:\n${list}\n\n${HOW[req.strength]} When they conflict, the earlier value wins. Never name the values.${more}`
}

/** Values are sent to the scorer under short keys, since custom ids can be long. */
function valueKeys() {
  const values = allValues()
  return { values, keys: values.map((_, i) => `v${i}`), list: values.map((v, i) => `- v${i} (${v.name}): ${v.principle}`).join("\n") }
}

export function geminiEngine(key: string, model: string): Engine {
  const call = <T>(user: string, schema: object, signal: AbortSignal) => withRetry(() => generateJson<T>({ key, model, user, temperature: 0, schema, signal }), signal)
  return {
    label: model,
    kind: "gemini",
    async *write(req, signal, prefix = "") {
      const user = prefix
        ? `${req.prompt}\n\n---\nYour reply so far is below. Continue it from its last word${req.more ? " with the next two or three sentences" : ""}. Output only the new words that come next; never repeat any of it.\n\n${prefix}`
        : req.prompt
      let head: string | undefined = prefix ? "" : undefined
      for await (const text of streamText({ key, model, system: systemPromptOf(req), user, temperature: 0.7, signal })) {
        if (head === undefined) {
          yield { text }
          continue
        }
        head += text
        if (head.length < 40) continue
        yield { text: joinAfter(prefix, withoutEcho(prefix, head)) }
        head = undefined
      }
      if (head) yield { text: joinAfter(prefix, withoutEcho(prefix, head)) }
    },
    async score(prompt, sentences, indices, signal) {
      const { values, keys, list } = valueKeys()
      const schema = {
        type: "OBJECT",
        properties: {
          sentences: {
            type: "ARRAY",
            items: { type: "OBJECT", properties: { i: { type: "INTEGER" }, ...Object.fromEntries(keys.map((k) => [k, { type: "INTEGER" }])) }, required: ["i", ...keys] },
          },
        },
        required: ["sentences"],
      }
      const user = [
        "You audit which values an AI assistant's reply expresses, one sentence at a time.",
        `VALUES\n${list}`,
        "For each requested sentence and every value, give an integer: 2 clearly upholds it, 1 leans toward it, 0 not relevant, -1 gives a little of it up, -2 clearly sacrifices it (false reassurance sacrifices candor; bluntness that ignores feelings sacrifices compassion). Most cells are 0.",
        `USER MESSAGE\n${prompt}`,
        `REPLY\n${sentences.map((s, i) => `[${i}] ${s}`).join("\n")}`,
        `Score only sentences ${indices.join(", ")}.`,
      ].join("\n\n")
      const out = await call<{ sentences?: ({ i: number } & Record<string, number>)[] }>(user, schema, signal)
      const result = new Map<number, Scores>()
      indices.forEach((i, n) => {
        const row = out.sentences?.find((r) => r.i === i) ?? out.sentences?.[n]
        result.set(i, Object.fromEntries(values.map((v, k) => [v.id, clamp(Number(row?.[keys[k]] ?? 0))])))
      })
      return result
    },
    async drivers(prompt, signal) {
      const { values, keys, list } = valueKeys()
      const schema = {
        type: "OBJECT",
        properties: { drivers: { type: "ARRAY", items: { type: "OBJECT", properties: { phrase: { type: "STRING" }, value: { type: "STRING", enum: keys } }, required: ["phrase", "value"] } } },
        required: ["drivers"],
      }
      const user = [
        "Which words in this message to an AI assistant pull its reply toward or against particular values?",
        `VALUES\n${list}`,
        "Return 2 to 5 short phrases (2 to 6 words) copied exactly from the message, each with the key of the one value it pulls on most.",
        `MESSAGE\n${prompt}`,
      ].join("\n\n")
      const out = await call<{ drivers?: { phrase: string; value: string }[] }>(user, schema, signal)
      return (out.drivers ?? [])
        .map((d) => ({ phrase: d.phrase, value: values[keys.indexOf(d.value)]?.id }))
        .filter((d): d is Driver => !!d.value && !!d.phrase && prompt.toLowerCase().includes(d.phrase.toLowerCase()))
    },
  }
}

/** Removes a repeat of the reply's ending that a model may echo before continuing. */
function withoutEcho(prefix: string, head: string) {
  const tail = prefix.trimEnd()
  const start = head.trimStart()
  for (let n = Math.min(tail.length, start.length, 120); n >= 6; n--) if (start.startsWith(tail.slice(-n))) return start.slice(n)
  return head
}

/** Makes sure a continuation joins the prefix with exactly one space where words meet. */
function joinAfter(prefix: string, text: string) {
  const t = text.replace(/^\s+/, "")
  if (!t) return ""
  return /\s$/.test(prefix) || /^[.,;:!?)”’'"]/.test(t) ? t : ` ${t}`
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
  async *write(req, signal, prefix = "") {
    const full = simReply(req)
    const rest = prefix ? simContinuation(full, prefix) : full
    if (!rest.trim().replace(/^[.\s]+$/, "")) return
    const words = rest.match(/\s*\S+/g) ?? []
    await sleep(260, signal)
    for (let i = 0; i < words.length; i += 2) {
      await sleep(70, signal)
      const pair = (words[i] ?? "") + (words[i + 1] ?? "")
      yield { text: i === 0 && !prefix ? pair.trimStart() : pair }
    }
  },
  async score(prompt, sentences, indices, signal) {
    await sleep(420, signal)
    const d = EXAMPLES.find((x) => x.prompt === prompt)
    return new Map(
      indices.map((i) => {
        const s = sentences[i].slice(0, 40)
        const scores: Scores = Object.fromEntries(allValues().map((v) => [v.id, 0]))
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
  if (!d) return "Sim mode only knows the example prompts. Open Settings to use Gemini or the in-browser model and ask anything."
  const ia = req.values.indexOf(d.a)
  const ib = req.values.indexOf(d.b)
  if (ia >= 0 || ib >= 0) return ib < 0 || (ia >= 0 && ia < ib) ? d.ra : d.rb
  return [...sentencesOf(d.ra).slice(0, 1), ...sentencesOf(d.rb).slice(-2)].join(" ")
}

/** Sim mode can't write new text, so after a cursor it continues with the sentences of the chosen reply that are not yet there. */
function simContinuation(full: string, prefix: string) {
  if (full.startsWith(prefix)) return full.slice(prefix.length)
  const done = (prefix.match(/[^.!?]+[.!?]+/g) ?? []).length
  const complete = /[.!?]["”’')\]]*\s*$/.test(prefix)
  const rest = sentencesOf(full).slice(complete ? done : done + 1).join(" ")
  return complete ? ` ${rest}` : `. ${rest}`
}

const sentencesOf = (t: string) => t.match(/[^.!?]+[.!?]+["”’')\]]*/g)?.map((s) => s.trim()) ?? [t]
const clamp = (n: number) => (Number.isFinite(n) ? Math.max(-2, Math.min(2, Math.round(n))) : 0)

export function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException("aborted", "AbortError"))
    const t = setTimeout(resolve, ms)
    signal.addEventListener("abort", () => (clearTimeout(t), reject(new DOMException("aborted", "AbortError"))), { once: true })
  })
}

/** Each request adds at most this many sentences; dragging the cursor to the end asks for more. */
const MAX_NEW_SENTENCES = 4

/** Cleans a reply as it streams: drops list markers and markdown and turns line breaks into spaces. `before` is any text the reply continues from. */
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
    if (/[.!?]["”’')\]]*$/.test(text.trimEnd()) && sentenceCount(text) - sentenceCount(before) >= MAX_NEW_SENTENCES) return
  }
}

const MARKER_ONLY = /^\s*(\d{1,2}[.)]?|[-*•])?\s*$/
const MARKER = /^\s*(\d{1,2}[.)]|[-*•])\s*/

function sentenceCount(text: string) {
  return (text.match(/[^.!?]+[.!?]+/g) ?? []).filter((s) => /[A-Za-z]/.test(s)).length
}
