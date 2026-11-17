import type { Alt, Engine, Piece, Run, TokenView, WriteRequest } from "./engine.ts"

/**
 * Where a streamed piece sits in the column's text; `forced` marks a word the user picked to rewrite from.
 * `p` and `alts` are the model's probabilities here with this reply's values; `other` holds them with the other side's values.
 */
export type Span = {
  start: number
  end: number
  p?: number
  alts?: Alt[]
  forced?: boolean
  other?: Alt[]
  amplified?: number
  /** The token this piece is, once the worker has measured it; with `run` and `k`, enough to re-check it in one pass. */
  tokenId?: number
  run?: Run
  k?: number
  /** Can't be scored: cleanup changed its text, or it isn't exactly one measured token. */
  unscorable?: boolean
}
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
 * How the values bear on one token, from the two distributions at that position (same model, same reply so far). This is a
 * conditional comparison: given this exact text so far, how likely was this token under each side's prompt. It does not
 * show that the values caused the reply as a whole to take this path.
 * - `changed`: with this side's values the model's first choice is a different token than with the other side's, and it wrote it;
 * - `boosted`: the first choice is the same, but this side's values made the written token at least twice as likely;
 * - `chance`: the model wrote a token other than its own first choice, which is sampling, not values;
 * - `same`: the values made no notable difference.
 */
export type Influence = {
  kind: "changed" | "boosted" | "chance" | "same"
  word: string
  mine: Alt[]
  other: Alt[]
  pMine: number
  /** Exact, unless `bound`: then the token was outside the other side's top five and its probability is below `pOther`. */
  pOther?: number
  bound: boolean
  /** Amplified writing: the probabilities are blended estimates, so only ranks are meaningful. */
  approximate: boolean
  rank: number
  /** log2 of how many times likelier this side's values made the written token than the other side's (a lower bound if `bound`). */
  lift: number
}

const BOOST = 2

export function influenceOf(piece: Span | undefined, text: string): Influence | undefined {
  if (!piece || piece.unscorable || piece.p === undefined || !piece.other?.length) return
  const word = text.slice(piece.start, piece.end)
  const written: Alt = { text: word, p: piece.p, id: piece.tokenId }
  const mine = sortAlts([...(piece.alts ?? []), written])
  const other = sortAlts(piece.other)
  const rank = mine.findIndex((a) => same(a, written)) + 1
  const found = other.find((a) => same(a, written))?.p
  const bound = found === undefined
  const pOther = found ?? other.at(-1)?.p
  const approximate = piece.amplified !== undefined
  const lift = approximate || !pOther ? 0 : Math.log2(piece.p / pOther)
  const base = { word, mine, other, pMine: piece.p, pOther, bound, approximate, rank, lift }
  if (rank > 1) return { kind: "chance", ...base }
  if (!same(mine[0], other[0])) return { kind: "changed", ...base }
  if (!approximate && piece.p >= 0.1 && lift >= Math.log2(BOOST)) return { kind: "boosted", ...base }
  return { kind: "same", ...base }
}

/** Whether a token counts as shifted by the values, for the column's count. */
export const isShifted = (inf: Influence | undefined) => !!inf && (inf.kind === "changed" || inf.kind === "boosted")

function sortAlts(list: Alt[]) {
  const out: Alt[] = []
  for (const a of list) if (!out.some((b) => same(a, b))) out.push(a)
  return out.sort((a, b) => b.p - a.p)
}

/** Tokens are the same if their ids match; without ids, only if their text is identical (" The" and "the" are different tokens). */
const same = (a: Alt, b: Alt) => (a.id !== undefined && b.id !== undefined ? a.id === b.id : a.text === b.text)

/** The other side's top five at a token, plus the written token itself, whose replayed probability is exact. */
function otherOf(other: TokenView, mine: { id?: number; text: string }): Alt[] {
  const written: Alt = { id: mine.id, text: mine.text, p: other.p }
  return other.top.some((a) => same(a, written)) ? other.top : [...other.top, written]
}

/** Streams `source` into the column, after any text already in it. `w` records the steering blend the new text is written with. */
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
      const span: Span = { start, end: col.text.length, p: piece.p, alts: piece.alts, other: piece.plain, amplified: piece.amplified, unscorable: piece.edited }
      col.pieces.push(span)
      if (piece.later && !piece.edited)
        pending.push(
          piece.later.then((m) => {
            if (signal.aborted || !col.pieces.includes(span)) return
            if (!m) span.unscorable = true
            else {
              // The worker's numbers replace the streamed estimates: both sides now come from the same computation.
              span.p = m.mine.p
              span.alts = m.mine.top
              span.tokenId = m.mine.id
              span.run = m.run
              span.k = m.k
              if (m.other) span.other = otherOf(m.other, m.mine)
            }
            onChange()
          }),
        )
      onChange()
    }
    // The last measurements land just after the text; wait for them (but not forever), so "done" means measured.
    await Promise.race([Promise.all(pending), new Promise((r) => setTimeout(r, 30000))])
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
 * Checks every measured token not yet compared against the other side's values. Tokens are grouped by the request that
 * wrote them and each group is replayed in one pass. A "keep writing" turn was written under a different prompt frame, so
 * its tokens are also re-read under this side's values in the ordinary frame, and both sides' numbers come from the same
 * kind of request. Only pieces never tied to a token (other engines) fall back to one model call per word.
 */
export async function probeColumn(col: Column, engine: Engine, against: WriteRequest, signal: AbortSignal, onChange: () => void) {
  if (!engine.probe || col.side === "steer") return
  const mine = { ...col.request, more: false, amplify: 0, live: false }
  const theirs = { ...against, more: false, amplify: 0, live: false }
  const needs = (p: Span) => p.p !== undefined && !p.unscorable && !p.other && p.amplified === undefined
  if (!col.pieces.some(needs)) return
  col.probing = true
  onChange()
  try {
    const runs = new Map<Run, Span[]>()
    for (const p of col.pieces) if (needs(p) && p.run && p.k !== undefined) runs.set(p.run, [...(runs.get(p.run) ?? []), p])
    if (engine.replay)
      for (const [run, spans] of runs) {
        if (signal.aborted) return
        // A token's distribution depends only on the tokens before it, so the replay can stop at the last one needed.
        const ids = run.ids.slice(0, Math.max(...spans.map((p) => p.k!)) + 1)
        const first = col.pieces.find((p) => p.run === run)!
        const prefix = run.prefix ?? col.text.slice(0, first.start)
        const others = await engine.replay(theirs, ids, signal, prefix).catch(() => [])
        const own = run.more && others.length ? await engine.replay(mine, ids, signal, prefix).catch(() => []) : []
        if (signal.aborted) return
        for (const p of spans) {
          const o = others[p.k!]
          if (!o) continue
          if (run.more) {
            const m = own[p.k!]
            if (!m) continue
            p.p = m.p
            p.alts = m.top
          }
          p.other = otherOf(o, { id: p.tokenId, text: col.text.slice(p.start, p.end) })
        }
        onChange()
      }
    // Last resort, one model call per word, for pieces never tied to a token. Without ids, a word outside the other side's
    // top five only gets a bound.
    for (const piece of col.pieces) {
      if (signal.aborted) return
      if (!needs(piece) || piece.run || !/[A-Za-z0-9]/.test(col.text.slice(piece.start, piece.end))) continue
      const alts = await engine.probe(theirs, col.text.slice(0, piece.start), signal)
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
    const checked = col.pieces.filter((p) => p.other && !p.unscorable && p.p !== undefined)
    const changed = checked.filter((p) => isShifted(influenceOf(p, col.text))).length
    const tally = checked.length ? ` · ${changed} of ${checked.length} tokens shifted` : ""
    state.textContent = col.streaming ? `writing${tally}` : col.probing ? `comparing with the other side${tally}` : col.error ? "" : `${countWords(col.text)} words${tally}`
    state.title = "A token is a word or part of a word, the unit the model writes in. Shifted: the other side's values would have made a different token the first choice, or made this one at least twice as likely."
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

    // Every token is restyled from its current comparison, so purple disappears when a comparison is cleared or replaced.
    text.querySelectorAll<HTMLElement>(".tok").forEach((el) => {
      const piece = col.pieces[Number(el.dataset.k)]
      const inf = influenceOf(piece, col.text)
      const key = inf ? `${inf.kind}:${inf.lift.toFixed(2)}:${inf.approximate}` : piece?.unscorable ? "x" : ""
      if (el.dataset.key === key) return
      el.dataset.key = key
      el.classList.toggle("flip", inf?.kind === "changed")
      el.classList.toggle("boost", inf?.kind === "boosted")
      el.classList.toggle("approx", !!inf?.approximate)
      el.classList.toggle("unscorable", !!piece?.unscorable)
      if (inf) el.style.setProperty("--lift", inf.approximate ? "2" : Math.max(0, Math.min(4, inf.lift)).toFixed(2))
      else el.style.removeProperty("--lift")
      if (piece?.p !== undefined) el.style.setProperty("--u", (1 - piece.p).toFixed(3))
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
