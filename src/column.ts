import type { Alt, Engine, Piece, WriteRequest } from "./engine.ts"
import { colorOf, nameOf, type Scores, type ValueId } from "./values.ts"

/** Where a sentence sits in the column's text. */
export type Sentence = { text: string; start: number; end: number }
/**
 * Where a streamed piece sits in the column's text; `forced` marks a word the user picked to rewrite from.
 * `p` and `alts` are the model's probabilities for this position with the column's values; `plain` holds them with no values.
 */
export type Span = { start: number; end: number; p?: number; alts?: Alt[]; forced?: boolean; plain?: Alt[]; amplified?: number }

/**
 * How the values bear on one word, from the two distributions at that position (same model, same reply so far):
 * - `changed`: with values the model's first choice is a different word than with none, and it wrote that first choice;
 * - `boosted`: the first choice is the same, but values made the written word at least twice as likely;
 * - `chance`: the model wrote a word other than its own first choice, which is sampling, not values;
 * - `same`: values made no notable difference.
 */
export type Influence = {
  kind: "changed" | "boosted" | "chance" | "same"
  word: string
  withValues: Alt[]
  withoutValues: Alt[]
  pWith: number
  pWithout?: number
  rank: number
  /** log2 of how many times likelier the values made the written word: 1 is twice as likely, -1 half as likely. */
  lift: number
}

const BOOST = 2

export function influenceOf(piece: Span, text: string): Influence | undefined {
  if (piece.p === undefined || !piece.plain?.length) return
  const word = text.slice(piece.start, piece.end)
  const withValues = sortAlts([...(piece.alts ?? []), { text: word, p: piece.p }])
  const withoutValues = sortAlts(piece.plain)
  const rank = withValues.findIndex((a) => same(a.text, word)) + 1
  const pWithout = withoutValues.find((a) => same(a.text, word))?.p
  const topWith = withValues[0]
  const topWithout = withoutValues[0]
  const floorP = withoutValues.at(-1)?.p ?? 0.01
  const lift = Math.log2(piece.p / (pWithout ?? Math.min(floorP, piece.p)))
  const base = { word, withValues, withoutValues, pWith: piece.p, pWithout, rank, lift }
  if (rank > 1) return { kind: "chance", ...base }
  if (!same(topWith.text, topWithout.text)) return { kind: "changed", ...base }
  if (piece.p >= 0.1 && lift >= Math.log2(BOOST)) return { kind: "boosted", ...base }
  return { kind: "same", ...base }
}

/** Merges duplicate tokens and sorts, most likely first. */
function sortAlts(list: Alt[]) {
  const out: Alt[] = []
  for (const a of list) if (!out.some((b) => b.text === a.text)) out.push(a)
  return out.sort((a, b) => b.p - a.p)
}

const norm = (t: string) => t.trim().toLowerCase()
const same = (a: string, b: string) => a === b || norm(a) === norm(b)

/** A sentence counts as upholding or giving up a value from these scores on the -2 to 2 scale. */
export const UPHOLDS = 0.6
export const GIVES_UP = -0.6

export type Column = {
  kind: "plain" | "guided"
  title: string
  request: WriteRequest
  text: string
  pieces: Span[]
  sentences: Sentence[]
  scores: (Scores | undefined)[]
  streaming: boolean
  scoring: boolean
  /** Checking each word against the no-values model. */
  probing: boolean
  /** Whether the last run added any text. */
  grew?: boolean
  /** Pieces from this index on were written in a separate "keep going" turn and are re-read with values before comparing. */
  reprobeFrom?: number
  error?: string
}

export function newColumn(kind: Column["kind"], title: string, request: WriteRequest): Column {
  return { kind, title, request, text: "", pieces: [], sentences: [], scores: [], streaming: true, scoring: false, probing: false }
}

export const countWords = (text: string) => (text.match(/\S+/g) ?? []).length

/** Where the text is cut after `words` words, so everything before it is revealed. */
export function cutAfter(text: string, words: number) {
  if (words <= 0) return 0
  let n = 0
  for (const m of text.matchAll(/\S+/g)) if (++n === words) return m.index! + m[0].length
  return text.length
}

/** Cuts a reply back to `cut` characters in place, keeping the scores of sentences that end before the cut. */
export function truncateColumn(col: Column, cut: number) {
  const end = col.text.slice(0, cut).trimEnd().length
  col.text = col.text.slice(0, end)
  col.pieces = col.pieces.filter((p) => p.end <= end)
  const kept = col.sentences.filter((s) => s.end <= end).length
  col.scores = col.scores.slice(0, kept)
  col.sentences = splitSentences(col.text)
  col.error = undefined
  if ((col.reprobeFrom ?? Infinity) >= col.pieces.length) col.reprobeFrom = undefined
}

export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = []
  for (const m of text.matchAll(/[^.!?\n]+(?:[.!?]+["”’')\]]*|$)/g)) {
    const lead = m[0].length - m[0].trimStart().length
    const trimmed = m[0].trim()
    if (trimmed) out.push({ text: trimmed, start: m.index! + lead, end: m.index! + lead + trimmed.length })
  }
  return out
}

/**
 * Streams `source` into the column (after any text already in it) and scores sentences as soon as they are complete, a few at a time,
 * so the rail fills in while the reply is still being written. The last sentence may still grow, so it waits until the next one starts.
 */
export async function runColumn(col: Column, engine: Engine, source: AsyncIterable<Piece>, signal: AbortSignal, onChange: () => void) {
  const requested = new Set<number>()
  let inflight: Promise<void> | null = null
  const startLength = col.text.length
  col.streaming = true
  col.error = undefined
  col.sentences = splitSentences(col.text)

  const ready = () =>
    col.sentences.map((_, i) => i).filter((i) => !col.scores[i] && !requested.has(i) && (!col.streaming || i < col.sentences.length - 1))

  const kick = () => {
    if (inflight || col.error || !engine.score) return
    const batch = ready()
    if (batch.length < (col.streaming ? 2 : 1)) return
    batch.forEach((i) => requested.add(i))
    col.scoring = true
    onChange()
    inflight = engine
      .score(col.request.prompt, col.sentences.map((s) => s.text), batch, signal)
      .then((scores) => scores.forEach((s, i) => (col.scores[i] = s)))
      .catch((e) => void (signal.aborted || (col.error = messageOf(e))))
      .finally(() => {
        inflight = null
        col.scoring = false
        if (!signal.aborted) (onChange(), kick())
      })
  }

  try {
    for await (const piece of source) {
      const start = col.text.length
      col.text += piece.text
      col.pieces.push({ start, end: col.text.length, p: piece.p, alts: piece.alts, plain: piece.plain, amplified: piece.amplified })
      col.sentences = splitSentences(col.text)
      onChange()
      kick()
    }
    col.streaming = false
    col.sentences = splitSentences(col.text)
    if (!col.sentences.length) throw new Error("The model sent back an empty reply. Run it again.")
    col.grew = col.text.length > startLength
    onChange()
    kick()
    while (inflight || (engine.score && ready().length && !col.error)) {
      if (inflight) await inflight
      else kick()
    }
  } catch (e) {
    if (!signal.aborted) col.error = messageOf(e)
  } finally {
    if (!signal.aborted) {
      col.streaming = false
      col.scoring = false
      onChange()
    }
  }
}

/**
 * Asks the model, given the reply so far but no values, how likely each next word was, left to right, so words the values
 * changed light up as the check sweeps along. Pieces written in a separate "keep going" turn are also re-read with values,
 * so both distributions come from the same kind of request.
 */
export async function probeColumn(col: Column, engine: Engine, signal: AbortSignal, onChange: () => void) {
  if (!engine.probe || col.kind !== "guided") return
  const plain = { ...col.request, values: [], more: false }
  const guided = { ...col.request, more: false }
  col.probing = true
  onChange()
  try {
    for (const [k, piece] of col.pieces.entries()) {
      if (signal.aborted) return
      // Punctuation and spaces carry no value signal; skipping them saves a model call each.
      if (piece.p === undefined || piece.plain || !/[A-Za-z0-9]/.test(col.text.slice(piece.start, piece.end))) continue
      const before = col.text.slice(0, piece.start)
      if (k >= (col.reprobeFrom ?? Infinity)) {
        const withValues = await engine.probe(guided, before, signal)
        const word = col.text.slice(piece.start, piece.end)
        const mine = withValues.find((a) => same(a.text, word))
        if (withValues.length) {
          piece.alts = withValues
          piece.p = mine?.p ?? Math.min(piece.p, withValues.at(-1)!.p)
        }
      }
      const alts = await engine.probe(plain, before, signal)
      if (alts.length) piece.plain = alts
      onChange()
    }
  } catch (e) {
    if (!signal.aborted) col.error = messageOf(e)
  } finally {
    if (!signal.aborted) {
      col.probing = false
      onChange()
    }
  }
}

/** Renders a column into `root`, appending new text instead of redrawing it, so streamed words fade in. Pieces with probabilities become clickable tokens. */
export function createColumnView(root: HTMLElement) {
  root.innerHTML = `
    <header class="col-head"><h2></h2><span class="col-state"></span></header>
    <p class="col-error" hidden></p>
    <div class="rows"><button type="button" class="knob" aria-label="Reply cursor. Drag, or use the arrow keys, to move through the reply."></button></div>
    <footer class="col-sum"></footer>`
  const title = root.querySelector("h2")!
  const state = root.querySelector<HTMLElement>(".col-state")!
  const error = root.querySelector<HTMLElement>(".col-error")!
  const rows = root.querySelector<HTMLElement>(".rows")!
  const sum = root.querySelector<HTMLElement>(".col-sum")!
  const knob = root.querySelector<HTMLElement>(".knob")!
  let rowEls: { el: HTMLElement; start: number; done: number; piece: number }[] = []

  /** How many words of the reply a pointer at (x, y) asks to reveal: through the nearest word, or up to it if the pointer is on its left half. */
  function wordsAt(x: number, y: number) {
    const spans = [...rows.querySelectorAll<HTMLElement>(".text > span")]
    const last = spans.at(-1)
    const lastRect = last ? [...last.getClientRects()].at(-1) : undefined
    // Below the last line, or right of its end, means "all of it".
    if (last && lastRect && (y > lastRect.bottom || (y >= lastRect.top && x > lastRect.right))) return Number(last.dataset.w)
    let best = { d: Infinity, words: 0 }
    spans.forEach((span) => {
      for (const r of span.getClientRects()) {
        const dx = x < r.left ? r.left - x : x > r.right ? x - r.right : 0
        const dy = y < r.top ? r.top - y : y > r.bottom ? y - r.bottom : 0
        const d = dx * dx + dy * dy * 40
        if (d < best.d) best = { d, words: x > r.left + r.width / 2 ? Number(span.dataset.w) : Number(span.dataset.w0) }
      }
    })
    return best.words
  }

  function update(col: Column, chosen: ValueId[], scored: boolean, baseline: Column | undefined, reveal: number | null) {
    title.textContent = col.title
    root.classList.toggle("live", col.streaming || col.scoring || col.probing)
    root.classList.toggle("unscored", !scored)
    const flips = col.pieces.filter((p) => {
      const inf = influenceOf(p, col.text)
      return !!inf && (inf.kind === "changed" || (inf.kind !== "chance" && inf.lift >= 1))
    }).length
    state.textContent = col.streaming
      ? "writing"
      : col.scoring
        ? "scoring"
        : col.probing
          ? `comparing with no values · ${flips} shifted`
          : col.error
            ? ""
            : `${col.sentences.length} sentences${col.kind === "guided" && col.pieces.some((p) => p.plain) ? ` · ${flips} word${flips === 1 ? "" : "s"} shifted by your values` : ""}`
    error.hidden = !col.error
    error.textContent = col.error ?? ""

    const stale = col.sentences.length < rowEls.length || rowEls.some((r, i) => col.sentences[i]?.start !== r.start || col.sentences[i].end < r.done)
    if (stale) {
      rows.replaceChildren(knob)
      rowEls = []
    }
    if (col.sentences.length) rows.querySelector(".waiting")?.remove()
    col.sentences.forEach((s, i) => {
      if (!rowEls[i]) {
        const el = document.createElement("div")
        el.className = "row"
        el.innerHTML = `<div class="rail"></div><p class="text"></p><div class="pins"></div>`
        rows.append(el)
        rowEls[i] = { el, start: s.start, done: s.start, piece: i ? rowEls[i - 1].piece : 0 }
      }
      appendText(rowEls[i], col, s)
      paintRow(rowEls[i].el, col.scores[i], chosen, col.scoring && !col.scores[i])
    })
    if (!col.sentences.length && col.streaming && !rows.querySelector(".waiting")) rows.insertAdjacentHTML("afterbegin", `<p class="waiting">Waiting for the model</p>`)
    rows.querySelectorAll<HTMLElement>(".tok:not([data-checked])").forEach((el) => {
      const piece = col.pieces[Number(el.dataset.k)]
      const inf = piece && influenceOf(piece, col.text)
      if (!inf) return
      el.dataset.checked = "1"
      el.classList.toggle("flip", inf.kind === "changed")
      el.classList.toggle("boost", inf.kind === "boosted")
      el.style.setProperty("--lift", Math.max(0, Math.min(4, inf.lift)).toFixed(2))
    })

    const cut = reveal === null ? col.text.length : cutAfter(col.text, reveal)
    rows.querySelectorAll<HTMLElement>(".text > span").forEach((span) => span.classList.toggle("ghost", Number(span.dataset.s) >= cut))
    rowEls.forEach((r, i) => r.el.classList.toggle("future", (col.sentences[i]?.start ?? 0) >= cut))
    placeKnob(cut, col, reveal === null)
    paintSummary(sum, col, chosen, scored, baseline, reveal)
  }

  /** The knob sits after the last revealed word; while following a reply being written, it rides along at the end. */
  function placeKnob(cut: number, col: Column, following: boolean) {
    const shownSpans = [...rows.querySelectorAll<HTMLElement>(".text > span:not(.ghost)")]
    const last = shownSpans.at(-1)
    const box = rows.getBoundingClientRect()
    const rect = last ? [...last.getClientRects()].at(-1) : rows.querySelector(".text")?.getBoundingClientRect()
    knob.hidden = !rect
    if (!rect) return
    knob.style.left = `${(last ? rect.right : rect.left) - box.left}px`
    knob.style.top = `${rect.top - box.top + rect.height / 2}px`
    knob.classList.toggle("following", following && col.streaming)
    knob.classList.toggle("held", !following)
    knob.setAttribute("aria-valuenow", String(cut))
  }

  return { update, wordsAt, knob }
}

/** Adds the part of sentence `s` not yet on screen, one span per piece, clipped to the sentence. */
function appendText(row: { el: HTMLElement; done: number; piece: number }, col: Column, s: Sentence) {
  const text = row.el.querySelector(".text")!
  while (row.piece < col.pieces.length && col.pieces[row.piece].end <= row.done) row.piece++
  for (let k = row.piece; k < col.pieces.length && row.done < s.end; k++) {
    const piece = col.pieces[k]
    if (piece.start >= s.end) break
    const from = Math.max(piece.start, row.done)
    const to = Math.min(piece.end, s.end)
    if (to <= from) continue
    let segment = col.text.slice(from, to)
    if (piece.p !== undefined) {
      // Leading spaces stay outside the shaded token, so shading hugs the word.
      const lead = segment.length - segment.trimStart().length
      if (lead && lead < segment.length) text.append(segment.slice(0, lead))
      if (lead && lead < segment.length) segment = segment.slice(lead)
    }
    const span = document.createElement("span")
    span.textContent = segment
    span.className = "fresh"
    const segStart = to - segment.length
    span.dataset.s = String(segStart)
    span.dataset.w0 = String(countWords(col.text.slice(0, segStart)))
    span.dataset.w = String(countWords(col.text.slice(0, to)))
    if (piece.p !== undefined) {
      span.classList.add("tok")
      span.dataset.k = String(k)
      span.style.setProperty("--u", (1 - piece.p).toFixed(3))
      if (piece.forced) span.classList.add("forced")
    }
    text.append(span)
    row.done = to
  }
}

function paintRow(row: HTMLElement, scores: Scores | undefined, chosen: ValueId[], pending: boolean) {
  const rail = row.querySelector<HTMLElement>(".rail")!
  const pins = row.querySelector<HTMLElement>(".pins")!
  const key = chosen.join(",")
  if (rail.dataset.key !== key) {
    rail.dataset.key = key
    rail.innerHTML = chosen.map((id) => `<span class="m" data-id="${id}" style="--c:${colorOf(chosen, id)}"><i></i></span>`).join("")
  }
  row.classList.toggle("pending", pending)
  rail.querySelectorAll<HTMLElement>(".m").forEach((m) => {
    const id = m.dataset.id as ValueId
    const v = scores?.[id] ?? 0
    m.style.setProperty("--w", `${(Math.abs(v) / 2) * 50}%`)
    m.style.setProperty("--l", v >= 0 ? "50%" : `${50 - (Math.abs(v) / 2) * 50}%`)
    m.classList.toggle("neg", v < 0)
    m.title = scores ? `${nameOf(id)}: ${label(v)}` : nameOf(id)
  })
  const given = scores ? chosen.filter((id) => (scores[id] ?? 0) <= GIVES_UP) : []
  row.classList.toggle("gives", given.length > 0)
  const html = given.map((id) => `<span class="pin" style="--c:${colorOf(chosen, id)}">gives up ${nameOf(id).toLowerCase()}</span>`).join("")
  if (pins.innerHTML !== html) pins.innerHTML = html
}

/** Sums up the revealed part of the reply, so the bars move as the cursor does. The baseline is cut at the same word. */
function paintSummary(el: HTMLElement, col: Column, chosen: ValueId[], scored: boolean, baseline: Column | undefined, reveal: number | null) {
  const revealed = (c: Column) => {
    const cut = reveal === null ? c.text.length : cutAfter(c.text, reveal)
    return c.scores.filter((s, i): s is Scores => !!s && (c.sentences[i]?.start ?? 0) < cut)
  }
  const rows = revealed(col)
  el.classList.toggle("empty", !rows.length)
  if (!scored) {
    el.innerHTML = `<p class="hint">Add a Gemini key in Settings to also score each sentence's values.</p>`
    return
  }
  if (!chosen.length) {
    el.innerHTML = `<p class="hint">Choose values above to see how this reply serves them.</p>`
    return
  }
  const mean = (list: Scores[], id: ValueId) => (list.length ? list.reduce((t, r) => t + (r[id] ?? 0), 0) / list.length : 0)
  const base = baseline ? revealed(baseline) : []
  el.innerHTML = chosen
    .map((id) => {
      const m = mean(rows, id)
      const up = rows.filter((s) => (s[id] ?? 0) >= UPHOLDS).length
      const down = rows.filter((s) => (s[id] ?? 0) <= GIVES_UP).length
      const diff = base.length && rows.length && !col.streaming ? m - mean(base, id) : null
      const delta = diff === null || Math.abs(diff) < 0.05 ? "" : `<span class="delta ${diff > 0 ? "up" : "down"}">${diff > 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)}</span>`
      const w = (Math.min(Math.abs(m), 2) / 2) * 50
      return `<div class="sum-row" style="--c:${colorOf(chosen, id)}">
        <span class="sum-name">${nameOf(id)}</span>
        <span class="sum-bar"><i style="left:${m >= 0 ? 50 : 50 - w}%;width:${w}%"></i></span>
        <span class="sum-note">${rows.length ? `upheld ${up}${down ? ` · gave up ${down}` : ""}` : ""}${delta}</span>
      </div>`
    })
    .join("")
}

function label(v: number) {
  return ["clearly gives it up", "gives a little up", "not involved", "leans toward it", "clearly upholds it"][Math.max(0, Math.min(4, Math.round(v) + 2))] + ` (${v > 0 ? "+" : ""}${v.toFixed(1)})`
}

function messageOf(e: unknown) {
  return e instanceof Error ? e.message : String(e)
}
