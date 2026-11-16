import data from "./data.json"
import { ApiError, decide, generateJson, streamText, type ScoreAnswer, type ScoreQuestion } from "./gemini.ts"
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
  /** A few short points on how the guided reply differs in substance from the plain one. */
  compare?(prompt: string, plain: string, guided: string, values: ValueId[], signal: AbortSignal): Promise<string[]>
}

/** Rubric levels for scoring a sentence on a value, -2 to 2. */
const LEVELS = ["Clearly sacrifices or works against this value", "Gives a little of it up", "Not involved", "Leans toward it", "Clearly upholds it"]

/** The expected level from Gemini Decision 1's probabilities, so scores are calibrated and continuous rather than rounded. */
function expectedOf(a: ScoreAnswer | undefined): number {
  if (!a) return 0
  const entries = Object.entries(a.probabilities ?? {})
  const numeric = entries.length > 0 && entries.every(([k]) => /^\d+$/.test(k))
  const oneBased = numeric && !entries.some(([k]) => k === "0") && entries.some(([k]) => k === String(LEVELS.length))
  let sum = 0
  let total = 0
  for (const [k, p] of entries) {
    let level = numeric ? Number(k) - (oneBased ? 1 : 0) : LEVELS.indexOf(a.legend?.[k] ?? k)
    if (level < 0 || level >= LEVELS.length) continue
    sum += p * (level - 2)
    total += p
  }
  if (total > 0) return sum / total
  return Number.isFinite(a.score) ? Math.max(-2, Math.min(2, a.score - 2)) : 0
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

/** Set once Gemini Decision 1 is unavailable for this key, so later scoring goes straight to the text model. */
let decisionsUnavailable = false

export function geminiEngine(key: string, model: string): Engine {
  const call = <T>(user: string, schema: object, signal: AbortSignal) => withRetry(() => generateJson<T>({ key, model, user, temperature: 0, schema, signal }), signal)

  /** Scores with Gemini Decision 1: one rubric question per sentence and value, answered as calibrated probabilities in one pass. */
  async function scoreWithDecisions(prompt: string, sentences: string[], indices: number[], signal: AbortSignal) {
    const values = allValues()
    const input = `USER MESSAGE\n${prompt}\n\nAI ASSISTANT'S REPLY, ONE SENTENCE PER LINE\n${sentences.map((s, i) => `[${i}] ${s}`).join("\n")}`
    const questions: Record<string, ScoreQuestion> = {}
    for (const i of indices)
      values.forEach((v, k) => {
        questions[`s${i}_v${k}`] = {
          type: "score",
          instructions: `How does reply sentence [${i}] treat the value "${v.name}" (${v.principle})? Judge it in the context of the user's message and the rest of the reply.`,
          criteria: LEVELS,
        }
      })
    const answers = await withRetry(() => decide(key, input, questions, signal), signal)
    return new Map(indices.map((i) => [i, Object.fromEntries(values.map((v, k) => [v.id, expectedOf(answers[`s${i}_v${k}`])]))] as [number, Scores]))
  }
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
      if (!decisionsUnavailable) {
        try {
          return await scoreWithDecisions(prompt, sentences, indices, signal)
        } catch (e) {
          if (signal.aborted || (e instanceof ApiError && e.status === 429)) throw e
          decisionsUnavailable = true
        }
      }
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
    async compare(prompt, plain, guided, values, signal) {
      const schema = { type: "OBJECT", properties: { points: { type: "ARRAY", items: { type: "STRING" } } }, required: ["points"] }
      const user = [
        `Two AI replies answer the same message. Reply B was told to hold these values: ${values.map((id) => valueOf(id).name).join(", ")}. Reply A had no values.`,
        "List the two or three most important differences in substance: the advice given, facts or risks mentioned or left out, commitments, and who the reply serves. Ignore wording and style unless it changes the meaning.",
        "Each point is one plain sentence under 22 words that names what Reply B does differently. No preamble.",
        `MESSAGE\n${prompt}`,
        `REPLY A (NO VALUES)\n${plain}`,
        `REPLY B (WITH VALUES)\n${guided}`,
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
  async compare(prompt, _plain, _guided, values, signal) {
    await sleep(700, signal)
    const d = EXAMPLES.find((x) => x.prompt === prompt)
    if (!d) return []
    const lead = values.includes(d.a) && (!values.includes(d.b) || values.indexOf(d.a) < values.indexOf(d.b)) ? d.a : d.b
    return [
      `The guided reply leans on ${valueOf(lead).name.toLowerCase()}: it ${lead === d.a ? "states the hard facts more directly" : "puts the person's situation first"}.`,
      "Sim mode shows a canned comparison; add a Gemini key in Settings for a real one.",
    ]
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
