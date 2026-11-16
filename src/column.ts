import type { Engine } from "./engine.ts"
import { colorOf, V, type Scores, type ValueId } from "./values.ts"

export type Sentence = { text: string }

export type Column = {
  title: string
  prompt: string
  text: string
  sentences: Sentence[]
  scores: (Scores | undefined)[]
  streaming: boolean
  scoring: boolean
  error?: string
}

export function newColumn(title: string, prompt: string): Column {
  return { title, prompt, text: "", sentences: [], scores: [], streaming: true, scoring: false }
}

export function splitSentences(text: string): Sentence[] {
  return [...text.matchAll(/[^.!?\n]+(?:[.!?]+["”’')\]]*|$)/g)].map((m) => ({ text: m[0].trim() })).filter((s) => s.text)
}

/**
 * Streams `source` into the column and scores sentences as soon as they are complete, a few at a time, so the rail fills in while the reply is still being written.
 * While streaming, the last sentence may still grow, so it waits until the next one starts or the stream ends.
 */
export async function runColumn(col: Column, engine: Engine, source: AsyncIterable<string>, signal: AbortSignal, onChange: () => void) {
  const requested = new Set<number>()
  let inflight: Promise<void> | null = null

  const ready = () =>
    col.sentences.map((_, i) => i).filter((i) => !col.scores[i] && !requested.has(i) && (!col.streaming || i < col.sentences.length - 1))

  const kick = () => {
    if (inflight || col.error) return
    const batch = ready()
    if (batch.length < (col.streaming ? 2 : 1)) return
    batch.forEach((i) => requested.add(i))
    col.scoring = true
    onChange()
    inflight = engine
      .score(col.prompt, col.sentences.map((s) => s.text), batch, signal)
      .then((scores) => scores.forEach((s, i) => (col.scores[i] = s)))
      .catch((e) => void (signal.aborted || (col.error = messageOf(e))))
      .finally(() => {
        inflight = null
        col.scoring = false
        if (!signal.aborted) (onChange(), kick())
      })
  }

  try {
    for await (const chunk of source) {
      col.text += chunk
      col.sentences = splitSentences(col.text)
      onChange()
      kick()
    }
    col.streaming = false
    col.sentences = splitSentences(col.text)
    if (!col.sentences.length) throw new Error("The model sent back an empty reply. Run it again.")
    onChange()
    kick()
    while (inflight || (ready().length && !col.error)) {
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

/** Renders a column into `root`, appending new text instead of redrawing it, so streamed words can fade in. */
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
  let shown: string[] = []
  let rowEls: HTMLElement[] = []

  return function update(col: Column, chosen: ValueId[], baseline?: Column) {
    title.textContent = col.title
    root.classList.toggle("live", col.streaming || col.scoring)
    state.textContent = col.streaming ? "writing" : col.scoring ? "scoring" : col.error ? "" : `${col.sentences.length} sentences`
    error.hidden = !col.error
    error.textContent = col.error ?? ""

    if (col.sentences.length < shown.length || col.sentences.some((s, i) => shown[i] && !s.text.startsWith(shown[i]))) {
      rows.replaceChildren()
      shown = []
      rowEls = []
    }
    if (col.sentences.length) rows.querySelector(".waiting")?.remove()
    col.sentences.forEach((s, i) => {
      let row = rowEls[i]
      if (!row) {
        row = document.createElement("div")
        row.className = "row"
        row.innerHTML = `<div class="rail"></div><p class="text"></p><div class="pins"></div>`
        rows.append(row)
        rowEls[i] = row
        shown[i] = ""
      }
      const delta = s.text.slice(shown[i].length)
      if (delta) {
        const span = document.createElement("span")
        span.className = "fresh"
        span.textContent = delta
        row.querySelector(".text")!.append(span)
        shown[i] = s.text
      }
      paintRow(row, col.scores[i], chosen, col.scoring && !col.scores[i])
    })
    if (!col.sentences.length && col.streaming && !rows.children.length) rows.innerHTML = `<p class="waiting">Waiting for the first words</p>`

    paintSummary(sum, col, chosen, baseline)
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

function paintSummary(el: HTMLElement, col: Column, chosen: ValueId[], baseline?: Column) {
  const scored = col.scores.filter((s): s is Scores => !!s)
  el.classList.toggle("empty", !scored.length)
  if (!chosen.length) {
    el.innerHTML = `<p class="hint">Choose values above to see how this reply serves them.</p>`
    return
  }
  const mean = (rows: Scores[], id: ValueId) => (rows.length ? rows.reduce((t, r) => t + r[id], 0) / rows.length : 0)
  const base = baseline?.scores.filter((s): s is Scores => !!s) ?? []
  el.innerHTML = chosen
    .map((id) => {
      const m = mean(scored, id)
      const up = scored.filter((s) => s[id] >= 1).length
      const down = scored.filter((s) => s[id] <= -1).length
      const diff = base.length && scored.length && !col.streaming ? m - mean(base, id) : null
      const delta = diff === null || Math.abs(diff) < 0.05 ? "" : `<span class="delta ${diff > 0 ? "up" : "down"}">${diff > 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)}</span>`
      const w = (Math.min(Math.abs(m), 2) / 2) * 50
      return `<div class="sum-row" style="--c:${colorOf(chosen, id)}">
        <span class="sum-name">${V[id].name}</span>
        <span class="sum-bar"><i style="left:${m >= 0 ? 50 : 50 - w}%;width:${w}%"></i></span>
        <span class="sum-note">${scored.length ? `upheld ${up}${down ? ` · gave up ${down}` : ""}` : ""}${delta}</span>
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
