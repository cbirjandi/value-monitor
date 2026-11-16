import type { Alt, Engine, Piece, WriteRequest } from "./engine.ts"
import { colorOf, V, type Scores, type ValueId } from "./values.ts"

/** Where a sentence sits in the column's text. */
export type Sentence = { text: string; start: number; end: number }
/** Where a streamed piece sits in the column's text; `forced` marks a word the user picked to rewrite from. */
export type Span = { start: number; end: number; p?: number; alts?: Alt[]; forced?: boolean }

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
  error?: string
}

export function newColumn(kind: Column["kind"], title: string, request: WriteRequest): Column {
  return { kind, title, request, text: "", pieces: [], sentences: [], scores: [], streaming: true, scoring: false }
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
      col.pieces.push({ start, end: col.text.length, p: piece.p, alts: piece.alts })
      col.sentences = splitSentences(col.text)
      onChange()
      kick()
    }
    col.streaming = false
    col.sentences = splitSentences(col.text)
    if (!col.sentences.length) throw new Error("The model sent back an empty reply. Run it again.")
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

/** Renders a column into `root`, appending new text instead of redrawing it, so streamed words fade in. Pieces with probabilities become clickable tokens. */
export function createColumnView(root: HTMLElement) {
  root.innerHTML = `
    <header class="col-head"><h2></h2><span class="col-state"></span></header>
    <p class="col-error" hidden></p>
    <div class="rows"></div>
    <footer class="col-sum"></footer>`
  const title = root.querySelector("h2")!
  const state = root.querySelector<HTMLElement>(".col-state")!
  const error = root.querySelector<HTMLElement>(".col-error")!
  const rows = root.querySelector<HTMLElement>(".rows")!
  const sum = root.querySelector<HTMLElement>(".col-sum")!
  let rowEls: { el: HTMLElement; start: number; done: number; piece: number }[] = []

  return function update(col: Column, chosen: ValueId[], scored: boolean, baseline?: Column) {
    title.textContent = col.title
    root.classList.toggle("live", col.streaming || col.scoring)
    root.classList.toggle("unscored", !scored)
    state.textContent = col.streaming ? "writing" : col.scoring ? "scoring" : col.error ? "" : `${col.sentences.length} sentences`
    error.hidden = !col.error
    error.textContent = col.error ?? ""

    const stale = col.sentences.length < rowEls.length || rowEls.some((r, i) => col.sentences[i]?.start !== r.start || col.sentences[i].end < r.done)
    if (stale) {
      rows.replaceChildren()
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
    if (!col.sentences.length && col.streaming && !rows.children.length) rows.innerHTML = `<p class="waiting">Waiting for the model</p>`

    paintSummary(sum, col, chosen, scored, baseline)
  }
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
    m.title = scores ? `${V[id].name}: ${label(v)}` : V[id].name
  })
  const given = scores ? chosen.filter((id) => scores[id] <= -1) : []
  row.classList.toggle("gives", given.length > 0)
  const html = given.map((id) => `<span class="pin" style="--c:${colorOf(chosen, id)}">gives up ${V[id].name.toLowerCase()}</span>`).join("")
  if (pins.innerHTML !== html) pins.innerHTML = html
}

function paintSummary(el: HTMLElement, col: Column, chosen: ValueId[], scored: boolean, baseline?: Column) {
  const rows = col.scores.filter((s): s is Scores => !!s)
  el.classList.toggle("empty", !rows.length)
  if (!scored) {
    el.innerHTML = `<p class="hint">Add a Gemini key in Settings to also score each sentence's values.</p>`
    return
  }
  if (!chosen.length) {
    el.innerHTML = `<p class="hint">Choose values above to see how this reply serves them.</p>`
    return
  }
  const mean = (list: Scores[], id: ValueId) => (list.length ? list.reduce((t, r) => t + r[id], 0) / list.length : 0)
  const base = baseline?.scores.filter((s): s is Scores => !!s) ?? []
  el.innerHTML = chosen
    .map((id) => {
      const m = mean(rows, id)
      const up = rows.filter((s) => s[id] >= 1).length
      const down = rows.filter((s) => s[id] <= -1).length
      const diff = base.length && rows.length && !col.streaming ? m - mean(base, id) : null
      const delta = diff === null || Math.abs(diff) < 0.05 ? "" : `<span class="delta ${diff > 0 ? "up" : "down"}">${diff > 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)}</span>`
      const w = (Math.min(Math.abs(m), 2) / 2) * 50
      return `<div class="sum-row" style="--c:${colorOf(chosen, id)}">
        <span class="sum-name">${V[id].name}</span>
        <span class="sum-bar"><i style="left:${m >= 0 ? 50 : 50 - w}%;width:${w}%"></i></span>
        <span class="sum-note">${rows.length ? `upheld ${up}${down ? ` · gave up ${down}` : ""}` : ""}${delta}</span>
      </div>`
    })
    .join("")
}

function label(v: number) {
  return ["clearly gives it up", "gives a little up", "not involved", "leans toward it", "clearly upholds it"][v + 2]
}

function messageOf(e: unknown) {
  return e instanceof Error ? e.message : String(e)
}
