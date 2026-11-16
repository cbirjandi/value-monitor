import data from "./data.json"
import { ApiError, generateJson, streamText } from "./gemini.ts"
import { allValues, valueOf, type Strength, type ValueId } from "./values.ts"

/**
 * `values` empty means a plain reply with no guidance. `more` asks the model to keep going past a reply it may consider finished.
 * `temperature` sets how freely the model samples. `amplify` (in-browser models only) pushes each word further in the direction
 * the values move the model's probabilities; 0 is off.
 */
export type WriteRequest = {
  prompt: string
  values: ValueId[]
  strength: Strength
  more?: boolean
  temperature?: number
  amplify?: number
  /** What amplification pushes away from: the other side's values (empty means no values). */
  against?: ValueId[]
  /** Steering: blend two value sets, `w` from 0 (all of `a`) to 1 (all of `b`). Overrides `values`. */
  blend?: { a: ValueId[]; b: ValueId[]; w: number }
  /** Check each stretch against the other side's values (`against`) while writing, so influence shows as the reply grows. */
  live?: boolean
}

export const DEFAULT_TEMPERATURE = 0.7
/** A phrase from the prompt that a scoring model says pulls a reply toward or against a value. */
export type Driver = { phrase: string; value: ValueId }
export type Alt = { text: string; p: number }
/** A piece of streamed text. Models that expose probabilities send one per token, with the chosen token's probability and the top alternatives. */
/** `id` is the model's token id, when known, so a reply can be replayed exactly under another prompt. */
export type Piece = {
  text: string
  p?: number
  alts?: Alt[]
  plain?: Alt[]
  amplified?: number
  id?: number
  /** The other side's view of this token (its own probability and top five), arriving a moment after the text. */
  later?: Promise<{ p: number; top: Alt[] } | undefined>
}

/** Where replies and scores come from: Gemini, a small model in the browser, or canned replies for trying the app without setup. */
export type Engine = {
  label: string
  kind: "sim" | "gemini" | "local"
  /** Continues a reply from `prefix` when given, which is how the future is rewritten from the cursor. */
  write(req: WriteRequest, signal: AbortSignal, prefix?: string): AsyncIterable<Piece>
  /**
   * Replays the token ids of a reply under `req` in one pass and returns, at each position, the probability of the replayed
   * token and the top five alternatives. Much faster than calling `probe` once per word.
   */
  replay?(req: WriteRequest, ids: number[], signal: AbortSignal, prefix?: string): Promise<{ p: number; top: Alt[] }[]>
  /** The model's most likely next tokens after `prefix`, most likely first. Only models that expose probabilities have it. */
  probe?(req: WriteRequest, prefix: string, signal: AbortSignal): Promise<Alt[]>
  drivers?(prompt: string, signal: AbortSignal): Promise<Driver[]>
  /** Who writes the comparison, when it isn't the engine itself. */
  judgeLabel?: string
  /** A few short points on how reply B differs in substance from reply A. */
  compare?(prompt: string, a: { text: string; values: ValueId[] }, b: { text: string; values: ValueId[] }, signal: AbortSignal): Promise<string[]>
}

export const EXAMPLES = data.dilemmas as { title: string; a: ValueId; b: ValueId; prompt: string; ra: string; rb: string }[]

const HOW: Record<Strength, string> = {
  gentle: "Let them shape the reply where they fit naturally.",
  firm: "Let them clearly shape every sentence.",
  strict: "Put them above everything else, even when that makes the reply less pleasant or less complete.",
}

const MORE = "The reply is not finished yet. Continue it with the next two or three sentences, adding new detail, reasons or next steps. Do not repeat, summarize or wrap up."

function heading(ids: ValueId[]) {
  return ids.some((id) => valueOf(id).persona)
    ? "Your guiding values and personality for this reply, most important first. Play the personality fully; it sets your attitude and conclusions:"
    : "Your guiding values for this reply, most important first:"
}

export function systemPromptOf(req: WriteRequest) {
  const base = "You are a helpful AI assistant. Answer in plain prose, in short sentences. Never use lists, numbering, headings or markdown."
  const more = req.more ? `\n\n${MORE}` : ""
  const list = (ids: ValueId[]) => ids.map((id, i) => `${i + 1}. ${valueOf(id).name}: ${valueOf(id).principle}`).join("\n")
  const b = req.blend
  if (b && b.a.length && b.b.length && b.w > 0.08 && b.w < 0.92) {
    const pb = Math.round(b.w * 10) * 10
    const pa = 100 - pb
    const lead = pb > 50 ? "B" : pa > 50 ? "A" : "neither; balance them"
    return `${base}\n\nTwo sets of values guide this reply. Weigh them about ${pa}% to ${pb}%.\nSet A (${pa}%):\n${list(b.a)}\nSet B (${pb}%):\n${list(b.b)}\n\n${HOW[req.strength]} Where the sets conflict, favor ${lead}. Never name the values.${more}`
  }
  const values = b ? (b.w >= 0.5 ? b.b : b.a) : req.values
  if (b && b.w > 0.08 && b.w < 0.92 && values.length) {
    // Only one side has values: the blend becomes how strongly that side's values apply.
    const share = Math.round((b.b.length ? b.w : 1 - b.w) * 100)
    return `${base}\n\n${heading(values)}\n${list(values)}\n\nApply them at about ${share}% strength: let them shape the reply ${share > 60 ? "clearly" : "noticeably but not completely"}. Never name the values.${more}`
  }
  if (!values.length) return base + more
  return `${base}\n\n${heading(values)}\n${list(values)}\n\n${HOW[req.strength]} When they conflict, the earlier value wins. Never name the values.${more}`
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
      for await (const text of streamText({ key, model, system: systemPromptOf(req), user, temperature: req.temperature ?? DEFAULT_TEMPERATURE, signal })) {
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
    async compare(prompt, a, b, signal) {
      const schema = { type: "OBJECT", properties: { points: { type: "ARRAY", items: { type: "STRING" } } }, required: ["points"] }
      const user = [
        `Two AI replies answer the same message. ${describeSide("A", a.values)} ${describeSide("B", b.values)}`,
        "List the two or three most important differences in substance: the advice given, facts or risks mentioned or left out, commitments, and who the reply serves. Ignore wording and style unless it changes the meaning.",
        "Each point is one plain sentence under 22 words that names what Reply B does differently. No preamble.",
        `MESSAGE\n${prompt}`,
        `REPLY A\n${a.text}`,
        `REPLY B\n${b.text}`,
      ].join("\n\n")
      const out = await call<{ points?: string[] }>(user, schema, signal)
      return (out.points ?? []).map((p) => p.trim()).filter(Boolean).slice(0, 3)
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

function describeSide(name: string, values: ValueId[]) {
  return values.length ? `Reply ${name} was told to hold these values: ${values.map((id) => valueOf(id).name).join(", ")}.` : `Reply ${name} had no particular values.`
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
  async compare(prompt, a, b, signal) {
    await sleep(700, signal)
    if (!EXAMPLES.some((x) => x.prompt === prompt)) return []
    const names = (ids: ValueId[]) => (ids.length ? ids.map((id) => valueOf(id).name.toLowerCase()).join(" and ") : "no particular values")
    return [`Reply A follows ${names(a.values)}; reply B follows ${names(b.values)}, and their advice differs accordingly.`, "Sim mode shows a canned comparison; use Gemini or an in-browser model for a real one."]
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
  if (req.blend) req = { ...req, values: req.blend.w >= 0.5 ? req.blend.b : req.blend.a }
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
