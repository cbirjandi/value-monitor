import type { Alt, Engine, Piece, WriteRequest } from "./engine.ts"

/**
 * Where a streamed piece sits in the column's text; `forced` marks a word the user picked to rewrite from.
 * `p` and `alts` are the model's probabilities here with this reply's values; `other` holds them with the other side's values.
 */
export type Span = { start: number; end: number; p?: number; alts?: Alt[]; forced?: boolean; other?: Alt[]; amplified?: number; id?: number }
/** A stretch of a steered reply written with one blend, from `start` to the next segment; `w` runs from 0 (all A) to 1 (all B). */
export type Segment = { start: number; w: number }

export type Column = {
  side: "a" | "b" | "steer"
  title: string
  request: WriteRequest
  text: string
  pieces: Span[]
  segments: Segment[]
  streaming: boolean
  /** Checking each word against the other side's values. */
  probing: boolean
  error?: string
  /** Whether the last run added any text. */
  grew?: boolean
  /** Pieces from this index on were written in a separate "keep going" turn and are re-read before comparing. */
  reprobeFrom?: number
}

export function newColumn(side: Column["side"], title: string, request: WriteRequest): Column {
  return { side, title, request, text: "", pieces: [], segments: [], streaming: true, probing: false }
}

export const countWords = (text: string) => (text.match(/\S+/g) ?? []).length

/** Where the text is cut after `words` words, so everything before it is revealed. */
export function cutAfter(text: string, words: number) {
  if (words <= 0) return 0
  let n = 0
  for (const m of text.matchAll(/\S+/g)) if (++n === words) return m.index! + m[0].length
  return text.length
}

/** Cuts a reply back to `cut` characters in place. */
export function truncateColumn(col: Column, cut: number) {
  const end = col.text.slice(0, cut).trimEnd().length
  col.text = col.text.slice(0, end)
  col.pieces = col.pieces.filter((p) => p.end <= end)
  col.segments = col.segments.filter((s) => s.start < end || s.start === 0)
  col.error = undefined
  if ((col.reprobeFrom ?? Infinity) >= col.pieces.length) col.reprobeFrom = undefined
}

/**
 * How the values bear on one word, from the two distributions at that position (same model, same reply so far):
 * - `changed`: with this side's values the model's first choice is a different word than with the other side's, and it wrote it;
 * - `boosted`: the first choice is the same, but this side's values made the written word at least twice as likely;
 * - `chance`: the model wrote a word other than its own first choice, which is sampling, not values;
 * - `same`: the values made no notable difference.
 */
export type Influence = {
  kind: "changed" | "boosted" | "chance" | "same"
  word: string
  mine: Alt[]
  other: Alt[]
  pMine: number
  pOther?: number
  rank: number
  /** log2 of how many times likelier this side's values made the written word than the other side's. */
  lift: number
}

const BOOST = 2

export function influenceOf(piece: Span | undefined, text: string): Influence | undefined {
  if (!piece || piece.p === undefined || !piece.other?.length) return
  const word = text.slice(piece.start, piece.end)
  const mine = sortAlts([...(piece.alts ?? []), { text: word, p: piece.p }])
  const other = sortAlts(piece.other)
  const rank = mine.findIndex((a) => same(a.text, word)) + 1
  const pOther = other.find((a) => same(a.text, word))?.p
  const floor = other.at(-1)?.p ?? 0.01
  const lift = Math.log2(piece.p / (pOther ?? Math.min(floor, piece.p)))
  const base = { word, mine, other, pMine: piece.p, pOther, rank, lift }
  if (rank > 1) return { kind: "chance", ...base }
  if (!same(mine[0].text, other[0].text)) return { kind: "changed", ...base }
  if (piece.p >= 0.1 && lift >= Math.log2(BOOST)) return { kind: "boosted", ...base }
  return { kind: "same", ...base }
}

function sortAlts(list: Alt[]) {
  const out: Alt[] = []
  for (const a of list) if (!out.some((b) => b.text === a.text)) out.push(a)
  return out.sort((a, b) => b.p - a.p)
}

const norm = (t: string) => t.trim().toLowerCase()
const same = (a: string, b: string) => a === b || norm(a) === norm(b)

/** Streams `source` into the column, after any text already in it. `w` records the steering blend the new text is written with. */
/** The other side's top five at a token, plus the token itself (its probability is exact) when it isn't among them. */
function withWord(d: { p: number; top: Alt[] }, word: string): Alt[] {
  return d.top.some((a) => same(a.text, word)) ? d.top : [...d.top, { text: word, p: d.p }]
}

export async function runColumn(col: Column, source: AsyncIterable<Piece>, signal: AbortSignal, onChange: () => void, w?: number) {
  const startLength = col.text.length
  const pending: Promise<void>[] = []
  col.streaming = true
  col.error = undefined
  if (w !== undefined && (col.segments.at(-1)?.w !== w || !col.segments.length)) col.segments.push({ start: startLength, w })
  try {
    for await (const piece of source) {
      const start = col.text.length
      col.text += piece.text
      const span: Span = { start, end: col.text.length, p: piece.p, alts: piece.alts, other: piece.plain, amplified: piece.amplified, id: piece.id }
      col.pieces.push(span)
      if (piece.later)
        pending.push(
          piece.later.then((d) => {
            if (!d || signal.aborted || !col.pieces.includes(span)) return
            span.other = withWord(d, col.text.slice(span.start, span.end))
            onChange()
          }),
        )
      onChange()
    }
    // The last live checks land just after the text; wait for them, so the reply counts as done only once they have.
    await Promise.all(pending)
    col.grew = col.text.length > startLength
    if (!col.text.trim()) throw new Error("The model sent back an empty reply. Run it again.")
  } catch (e) {
    if (!signal.aborted) col.error = e instanceof Error ? e.message : String(e)
  } finally {
    if (!signal.aborted) {
      col.streaming = false
      onChange()
    }
  }
}

/**
 * Asks the model, given the reply so far but the other side's values, how likely each next word was, left to right,
 * so words this side's values changed light up as the check sweeps along. Pieces written in a separate "keep going"
 * turn are also re-read with this side's values, so both distributions come from the same kind of request.
 */
export async function probeColumn(col: Column, engine: Engine, against: WriteRequest, signal: AbortSignal, onChange: () => void) {
  if (!engine.probe || col.side === "steer") return
  const mine = { ...col.request, more: false, amplify: 0 }
  const theirs = { ...against, more: false, amplify: 0 }
  col.probing = true
  onChange()
  try {
    // Fast path: replay the reply's own tokens under the other side's values in one pass, as far as their ids are known.
    let n = 0
    while (n < col.pieces.length && col.pieces[n].id !== undefined) n++
    if (engine.replay && n > 0) {
      const reread = (col.reprobeFrom ?? Infinity) < n
      // Words already checked live are skipped, unless this side's own view must be re-read too.
      let from = 0
      if (!reread) while (from < n && col.pieces[from].other) from++
      const ids = col.pieces.slice(from, n).map((p) => p.id!)
      const prefix = from < col.pieces.length ? col.text.slice(0, col.pieces[from].start) : col.text
      const safe = (r: typeof mine) => (ids.length ? engine.replay!(r, ids, signal, prefix).catch(() => []) : Promise.resolve([]))
      const others = await safe(theirs)
      const own = reread ? await safe(mine) : []
      if (signal.aborted) return
      others.forEach((d, j) => {
        const k = from + j
        const piece = col.pieces[k]
        piece.other = withWord(d, col.text.slice(piece.start, piece.end))
        if (own[j] && k >= (col.reprobeFrom ?? Infinity)) {
          piece.p = own[j].p
          piece.alts = own[j].top
        }
      })
      onChange()
    }
    for (const [k, piece] of col.pieces.entries()) {
      if (signal.aborted) return
      // Punctuation and spaces carry no value signal; skipping them saves a model call each.
      if (piece.p === undefined || piece.other || !/[A-Za-z0-9]/.test(col.text.slice(piece.start, piece.end))) continue
      const before = col.text.slice(0, piece.start)
      if (k >= (col.reprobeFrom ?? Infinity)) {
        const fresh = await engine.probe(mine, before, signal)
        const word = col.text.slice(piece.start, piece.end)
        if (fresh.length) {
          piece.alts = fresh
          piece.p = fresh.find((a) => same(a.text, word))?.p ?? Math.min(piece.p, fresh.at(-1)!.p)
        }
      }
      const alts = await engine.probe(theirs, before, signal)
      if (alts.length) piece.other = alts
      onChange()
    }
  } catch (e) {
    if (!signal.aborted) col.error = e instanceof Error ? e.message : String(e)
  } finally {
    if (!signal.aborted) {
      col.probing = false
      onChange()
    }
  }
}

/** Renders a column into `root`, appending new text instead of redrawing it, so streamed words fade in. */
export function createColumnView(root: HTMLElement, withKnob: boolean) {
  root.innerHTML = `
    <header class="col-head"><h2></h2><span class="col-state"></span></header>
    <p class="col-error" hidden></p>
    <div class="body"><p class="text"></p>${withKnob ? `<button type="button" class="knob" aria-label="Reply cursor. Drag, or use the arrow keys, to move through the reply."></button>` : ""}</div>
    <footer class="col-foot" hidden></footer>`
  const title = root.querySelector("h2")!
  const state = root.querySelector<HTMLElement>(".col-state")!
  const error = root.querySelector<HTMLElement>(".col-error")!
  const body = root.querySelector<HTMLElement>(".body")!
  const text = root.querySelector<HTMLElement>(".text")!
  const foot = root.querySelector<HTMLElement>(".col-foot")!
  const knob = root.querySelector<HTMLElement>(".knob")
  let shown = 0
  let shownText = ""

  /** How many words a pointer at (x, y) asks to reveal: through the nearest word, or up to it if on its left half. */
  function wordsAt(x: number, y: number) {
    const spans = [...text.querySelectorAll<HTMLElement>("span[data-w]")]
    const last = spans.at(-1)
    const lastRect = last ? [...last.getClientRects()].at(-1) : undefined
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

  function update(col: Column, reveal: number | null, footHtml = "") {
    title.textContent = col.title
    root.classList.toggle("live", col.streaming || col.probing)
    root.classList.toggle("steer", col.side === "steer")
    const changed = col.pieces.filter((p) => {
      const inf = influenceOf(p, col.text)
      return !!inf && (inf.kind === "changed" || (inf.kind !== "chance" && inf.lift >= 1))
    }).length
    state.textContent = col.streaming
      ? `writing${col.pieces.some((p) => p.other) ? ` · ${changed} shifted so far` : ""}`
      : col.probing
        ? `comparing with the other side · ${changed} shifted`
        : col.error
          ? ""
          : `${countWords(col.text)} words${col.pieces.some((p) => p.other) ? ` · ${changed} shifted by these values` : ""}`
    error.hidden = !col.error
    error.textContent = col.error ?? ""

    if (!col.text.startsWith(shownText) || col.pieces.length < shown) {
      text.replaceChildren()
      shown = 0
      shownText = ""
    }
    if (col.pieces.length) text.querySelector(".waiting")?.remove()
    for (let k = shown; k < col.pieces.length; k++) appendPiece(text, col, k)
    shown = col.pieces.length
    shownText = col.text
    if (!col.text && col.streaming && !text.querySelector(".waiting")) text.innerHTML = `<span class="waiting">Waiting for the model</span>`

    text.querySelectorAll<HTMLElement>(".tok:not([data-checked])").forEach((el) => {
      const inf = influenceOf(col.pieces[Number(el.dataset.k)], col.text)
      if (!inf) return
      el.dataset.checked = "1"
      el.classList.toggle("flip", inf.kind === "changed")
      el.classList.toggle("boost", inf.kind === "boosted")
      el.style.setProperty("--lift", Math.max(0, Math.min(4, inf.lift)).toFixed(2))
    })

    const cut = reveal === null ? col.text.length : cutAfter(col.text, reveal)
    text.querySelectorAll<HTMLElement>("span[data-s]").forEach((span) => span.classList.toggle("ghost", Number(span.dataset.s) >= cut))
    if (knob) placeKnob(col, reveal === null)
    if (foot.innerHTML !== footHtml) foot.innerHTML = footHtml
    foot.hidden = !footHtml
  }

  /** The knob sits after the last revealed word; while following a reply being written, it rides along at the end. */
  function placeKnob(col: Column, following: boolean) {
    if (!knob) return
    const last = [...text.querySelectorAll<HTMLElement>("span[data-s]:not(.ghost)")].at(-1)
    const box = body.getBoundingClientRect()
    const rect = last ? [...last.getClientRects()].at(-1) : text.getBoundingClientRect()
    knob.hidden = !col.text && !col.streaming
    if (!rect) return
    knob.style.left = `${(last ? rect.right : rect.left) - box.left}px`
    knob.style.top = `${rect.top - box.top + (last ? rect.height / 2 : 12)}px`
    knob.classList.toggle("following", following && col.streaming)
    knob.classList.toggle("held", !following)
  }

  return { update, wordsAt, knob }
}

/**
 * Adds one piece as a span. Leading spaces stay outside it, so word highlights hug the word; in a steered reply they stay in,
 * so the blend underline runs unbroken along each stretch.
 */
function appendPiece(text: HTMLElement, col: Column, k: number) {
  const piece = col.pieces[k]
  let segment = col.text.slice(piece.start, piece.end)
  const lead = col.side === "steer" ? 0 : segment.length - segment.trimStart().length
  if (lead && lead < segment.length) {
    text.append(segment.slice(0, lead))
    segment = segment.slice(lead)
  }
  if (!segment.trim()) {
    if (segment) text.append(segment)
    return
  }
  const start = piece.end - segment.length
  const span = document.createElement("span")
  span.textContent = segment
  span.className = "fresh"
  span.dataset.s = String(start)
  span.dataset.w0 = String(countWords(col.text.slice(0, start)))
  span.dataset.w = String(countWords(col.text.slice(0, piece.end)))
  const seg = [...col.segments].reverse().find((s) => s.start <= piece.start)
  if (seg) {
    span.style.setProperty("--mix", seg.w.toFixed(2))
    const prev = col.pieces[k - 1]
    if (col.segments.length > 1 && seg.start > 0 && (!prev || prev.start < seg.start)) {
      span.classList.add("seg-start")
      span.title = `Steering changed here: ${Math.round((1 - seg.w) * 100)}% A · ${Math.round(seg.w * 100)}% B`
    }
  }
  if (piece.p !== undefined) {
    span.classList.add("tok")
    span.dataset.k = String(k)
    span.style.setProperty("--u", (1 - piece.p).toFixed(3))
    if (piece.forced) span.classList.add("forced")
  }
  text.append(span)
}
