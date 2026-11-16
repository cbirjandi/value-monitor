import { countWords, createColumnView, cutAfter, influenceOf, newColumn, probeColumn, runColumn, truncateColumn, type Column, type Influence } from "./column.ts"
import { DEFAULT_TEMPERATURE, EXAMPLES, geminiEngine, simEngine, tidy, type Alt, type Driver, type Engine, type WriteRequest } from "./engine.ts"
import { defaultModel, listModels } from "./gemini.ts"
import { hasWebGPU, LOCAL_MODELS, localEngine, type Progress } from "./local.ts"
import { addCustomValue, allValues, colorOf, MAX_CHOSEN, nameOf, removeCustomValue, topValuesFromLab, type Strength, type ValueId } from "./values.ts"

type EngineKind = Engine["kind"]
const STORE = { key: "vm-gemini-key", model: "vm-gemini-model", engine: "vm-engine", local: "vm-local-model", ui: "vm-ui-3" }
const RERUN_MS = 650

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const promptView = $("prompt-view")
const promptEdit = $<HTMLTextAreaElement>("prompt")
const chipsEl = $("chips")
const strengthEl = $("strength")
const compare = $<HTMLInputElement>("compare")
const output = $("output")
const statusEl = $("status")
const focusNote = $("focus-note")
const popover = $("popover")
const loadBar = $("load-bar")

type Highlight = "values" | "doubt" | "off"
type Ui = { prompt: string; example: number; chosen: ValueId[]; strength: Strength; compare: boolean; highlight: Highlight; temperature: number; amplify: number }
const TUNE_DEFAULTS = { highlight: "values" as Highlight, temperature: DEFAULT_TEMPERATURE, amplify: 0 }
const ui: Ui = { prompt: EXAMPLES[0].prompt, example: 0, chosen: ["candor", "kindness"], strength: "firm", compare: true, ...TUNE_DEFAULTS, ...loadUi() }

let progress: Progress | null = null
let engine: Engine = engineFromStorage()
let controller: AbortController | undefined
let rerunTimer: number | undefined
type Slot = { col: Column; view: ReturnType<typeof createColumnView>; el: HTMLElement }
let slots: Slot[] = []
/** How many words of each reply are revealed; `null` follows the reply as it is written. */
let reveal: number | null = null
/** The plain reply doesn't depend on the chosen values, so it is kept per model and prompt. */
const plainCache = new Map<string, Column>()
const driverCache = new Map<string, Driver[]>()

ui.chosen = ui.chosen.filter((id) => allValues().some((v) => v.id === id))
renderChips()
bindAddForm()
renderStrength()
renderPrompt()
compare.checked = ui.compare
renderEmpty()
updateEngineUi()
bindSettings()
document.body.dataset.ready = "1"

promptView.addEventListener("click", startEditing)
promptView.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), startEditing()))
promptEdit.addEventListener("input", autosize)
promptEdit.addEventListener("blur", stopEditing)
promptEdit.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    stopEditing()
    run()
  }
  if (e.key === "Escape") stopEditing()
})
$("shuffle").addEventListener("click", () => {
  ui.example = (ui.example + 1) % EXAMPLES.length
  ui.prompt = EXAMPLES[ui.example].prompt
  saveUi()
  renderPrompt()
  run()
})
$("run").addEventListener("click", run)
compare.addEventListener("change", () => ((ui.compare = compare.checked), saveUi(), scheduleRerun()))
bindTune()
output.addEventListener("click", onTokenClick)
document.addEventListener("click", (e) => {
  const t = e.target as HTMLElement
  if (!popover.hidden && !popover.contains(t) && !t.closest(".tok")) closePopover()
})
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePopover()
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) (e.preventDefault(), run())
})
window.addEventListener("resize", closePopover)

function run() {
  window.clearTimeout(rerunTimer)
  closePopover()
  reveal = null
  const prompt = ui.prompt.trim()
  if (!prompt) return setStatus("Write a question first.")
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  setStatus("")

  const plainId = `${engine.label}\n${ui.temperature}\n${prompt}`
  const jobs: Promise<void>[] = []
  const cols: Column[] = []
  if (ui.compare || !ui.chosen.length) {
    const cached = plainCache.get(plainId)
    const covers = (c: Column) => ui.chosen.every((id) => c.scores.every((sc) => !sc || id in sc))
    if (cached && !cached.error && covers(cached)) cols.push(cached)
    else {
      const col = newColumn("plain", "No values", requestOf(prompt, []))
      cols.push(col)
      jobs.push(runColumn(col, engine, tidy(engine.write(col.request, signal)), signal, paint).then(() => void (!col.error && !signal.aborted && plainCache.set(plainId, col))))
    }
  }
  if (ui.chosen.length) {
    const col = newColumn("guided", guidedTitle(), requestOf(prompt, ui.chosen))
    cols.push(col)
    jobs.push(runColumn(col, engine, tidy(engine.write(col.request, signal)), signal, paint))
  }

  output.replaceChildren()
  output.classList.toggle("two", cols.length === 2)
  slots = cols.map((col) => mountSlot(col))
  paint()
  loadDrivers(prompt, signal)
  settle(jobs, signal).then(() => compareWords(signal))
}

/** Once the replies are written, check which words of the guided reply the values changed (in-browser model only). */
async function compareWords(signal: AbortSignal) {
  for (const { col } of slots) if (!signal.aborted && !col.error) await probeColumn(col, engine, signal, paint)
}

function mountSlot(col: Column, replace?: HTMLElement): Slot {
  const el = document.createElement("article")
  el.className = "col"
  if (replace) replace.replaceWith(el)
  else output.append(el)
  const view = createColumnView(el)
  bindKnob(view)
  return { col, el, view }
}

/**
 * Dragging either reply's knob moves one shared cursor, so both replies are cut at the same word, as in the Spaceship IDE:
 * letting go further back throws away what came after and writes a new ending from there; reaching the end keeps the model writing.
 */
function bindKnob(view: ReturnType<typeof createColumnView>) {
  const { knob } = view
  const longest = () => Math.max(0, ...slots.map((s) => countWords(s.col.text)))
  const set = (words: number) => {
    reveal = Math.max(0, Math.min(words, longest()))
    paint()
  }
  /** Judged on the reply being dragged: reaching the end of that one keeps both writing. */
  const own = () => countWords(slots.find((s) => s.view === view)?.col.text ?? "")
  const mine = () => slots.find((s) => s.view === view)
  const settleAt = (from: number) => {
    if (reveal === null) return paint()
    if (reveal >= own()) grow("extend", mine())
    else if (reveal < from) grow("regrow")
    else paint()
  }
  knob.addEventListener("pointerdown", (e) => {
    e.preventDefault()
    knob.setPointerCapture(e.pointerId)
    closePopover()
    window.clearTimeout(keyTimer)
    const from = reveal ?? longest()
    const start = { x: e.clientX, y: e.clientY }
    let moved = false
    document.body.classList.add("scrubbing")
    const move = (ev: PointerEvent) => {
      if (!moved && Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 4) return
      moved = true
      set(view.wordsAt(ev.clientX, ev.clientY))
    }
    const up = () => {
      knob.removeEventListener("pointermove", move)
      document.body.classList.remove("scrubbing")
      // A click on the knob at the end of a finished reply asks that reply to keep writing.
      if (!moved) return void ((reveal === null || reveal >= own()) && grow("extend", mine()))
      settleAt(from)
    }
    knob.addEventListener("pointermove", move)
    knob.addEventListener("pointerup", up, { once: true })
    knob.addEventListener("pointercancel", up, { once: true })
  })
  let keyFrom: number | null = null
  knob.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 5 : 1
    const now = reveal ?? longest()
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return
    e.preventDefault()
    keyFrom ??= now
    if (e.key === "ArrowLeft") set(now - step)
    else if (e.key === "ArrowRight") set(now + step)
    else if (e.key === "Home") set(0)
    else set(own())
    window.clearTimeout(keyTimer)
    keyTimer = window.setTimeout(() => {
      const from = keyFrom ?? now
      keyFrom = null
      settleAt(from)
    }, 700)
  })
}
let keyTimer: number | undefined

/**
 * Writes on from where each reply should continue, in place:
 * `extend` adds more after the end; `regrow` replaces everything after the cursor with a new ending;
 * `revalue` does the same for the guided reply only, with the values now chosen.
 */
function grow(mode: "extend" | "regrow" | "revalue", only?: Slot) {
  if (!slots.length) return
  if (mode === "extend" && slots.some((s) => s.col.streaming)) return setStatus("Let the replies finish, then drag to the end to keep writing.")
  window.clearTimeout(rerunTimer)
  closePopover()
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const jobs: Promise<void>[] = []
  for (const slot of slots) {
    const { col } = slot
    if (only && slot !== only) continue
    if (mode === "revalue" && col.kind !== "guided") continue
    // Text written in a separate "keep going" turn is re-read before comparing (see probeColumn).
    if (mode === "extend" && engine.kind === "local") col.reprobeFrom = Math.min(col.reprobeFrom ?? Infinity, col.pieces.length)
    if (mode !== "extend") truncateColumn(col, cutAfter(col.text, reveal ?? countWords(col.text)))
    col.request = requestOf(col.request.prompt, col.kind === "guided" ? ui.chosen : [])
    if (col.kind === "guided") col.title = guidedTitle()
    const request = { ...col.request, more: mode === "extend" }
    jobs.push(runColumn(col, engine, tidy(engine.write(request, signal, col.text), col.text), signal, paint))
  }
  if (mode === "extend") reveal = null
  setStatus("")
  paint()
  settle(jobs, signal).then(() => {
    if (signal.aborted) return
    if (mode === "extend" && !slots.some((s) => (!only || s === only) && s.col.grew))
      setStatus(engine.kind === "sim" ? "Sim mode can't write more. Open Settings to use a real model." : "The model had nothing more to add here.")
    compareWords(signal)
  })
}

/** A request with the current tuning. Amplification only applies where it can: in-browser models, with values. */
function requestOf(prompt: string, values: ValueId[]): WriteRequest {
  return { prompt, values, strength: ui.strength, temperature: ui.temperature, amplify: engine.kind === "local" && values.length ? ui.amplify : 0 }
}

function guidedTitle() {
  const amp = engine.kind === "local" && ui.amplify > 0 ? ` · amplified ×${ui.amplify}` : ""
  return `With ${listOf(ui.chosen.map((id) => nameOf(id).toLowerCase()))}${amp}`
}

function settle(jobs: Promise<void>[], signal: AbortSignal) {
  const diff = $("diff")
  if (slots.length < 2) diff.hidden = true
  else if (!diff.hidden) diff.classList.add("stale")
  return Promise.all(jobs).then(() => {
    if (signal.aborted) return
    const failed = slots.find((s) => s.col.error)
    setStatus(failed ? failed.col.error! : "")
    compareReplies(signal)
  })
}

const comparisons = new Map<string, string[]>()

/** A second model reads both finished replies and names the differences in substance, not wording. */
async function compareReplies(signal: AbortSignal) {
  const diff = $("diff")
  const list = $("diff-points")
  const note = $("diff-note")
  const plain = slots.find((s) => s.col.kind === "plain")?.col
  const guided = slots.find((s) => s.col.kind === "guided")?.col
  if (!plain || !guided || !plain.text || !guided.text) return void (diff.hidden = true)
  diff.hidden = false
  if (!engine.compare) {
    diff.classList.remove("stale", "loading")
    list.innerHTML = ""
    note.textContent = "Add a Gemini key in Settings to get a short written comparison of the two replies."
    return
  }
  const id = `${engine.label}\n${plain.text}\n${guided.text}`
  diff.classList.add("loading")
  note.textContent = "Comparing the two replies…"
  try {
    if (!comparisons.has(id)) comparisons.set(id, await engine.compare(guided.request.prompt, plain.text, guided.text, guided.request.values, signal))
    if (signal.aborted) return
    const points = comparisons.get(id)!
    list.innerHTML = points.map((p) => `<li>${esc(p)}</li>`).join("")
    note.textContent = points.length ? `Written by ${engine.kind === "local" ? "Gemini" : engine.label}, reading both full replies.` : "No substantive differences found."
  } catch (e) {
    if (!signal.aborted) note.textContent = e instanceof Error ? e.message : String(e)
  } finally {
    if (!signal.aborted) diff.classList.remove("loading", "stale")
  }
}

function paint() {
  const plain = slots.find((s) => s.col.kind === "plain")?.col
  for (const { col, view } of slots) view.update(col, ui.chosen, !!engine.score, col === plain ? undefined : plain, reveal)
  output.classList.toggle("scrubbed", reveal !== null)
  paintHint()
}

/** Changing values or strength rewrites the reply: from the cursor onward if the cursor was pulled back, otherwise from the start. */
function scheduleRerun() {
  if (!slots.length) return
  window.clearTimeout(rerunTimer)
  rerunTimer = window.setTimeout(() => (canRewriteFuture() ? grow("revalue") : run()), RERUN_MS)
}

function canRewriteFuture() {
  return reveal !== null && reveal > 0 && ui.chosen.length > 0 && slots.some((s) => s.col.kind === "guided")
}

async function loadDrivers(prompt: string, signal: AbortSignal) {
  if (!engine.drivers) return renderPrompt()
  const id = `${engine.label}\n${prompt}`
  if (!driverCache.has(id)) {
    try {
      driverCache.set(id, await engine.drivers(prompt, signal))
    } catch {
      return
    }
  }
  if (!signal.aborted) renderPrompt()
}

/* Token probabilities: what the model almost said, and rewriting from a different word. */

function onTokenClick(e: MouseEvent) {
  const tok = (e.target as HTMLElement).closest<HTMLElement>(".tok")
  if (!tok) return
  const slot = slots.find((s) => s.el.contains(tok))
  const k = Number(tok.dataset.k)
  const piece = slot?.col.pieces[k]
  if (!slot || !piece || piece.p === undefined) return
  output.querySelectorAll(".tok.open").forEach((t) => t.classList.remove("open"))
  tok.classList.add("open")

  const chosenText = slot.col.text.slice(piece.start, piece.end)
  const busy = slots.some((s) => s.col.streaming || s.col.scoring || s.col.probing)
  const canBranch = engine.kind === "local" && !busy
  const inf = influenceOf(piece, slot.col.text)
  // The rows: every word either distribution put in its top five, ordered by the distribution this reply was written with.
  const withValues = inf?.withValues ?? [...(piece.alts ?? []), { text: chosenText, p: piece.p }].filter((a, i, l) => l.findIndex((b) => b.text === a.text) === i).sort((a, b) => b.p - a.p)
  const rows: Alt[] = [...withValues]
  for (const a of inf?.withoutValues ?? []) if (!rows.some((r) => r.text === a.text)) rows.push({ text: a.text, p: 0 })
  const pOf = (list: Alt[] | undefined, t: string) => list?.find((a) => a.text === t)?.p
  const two = !!inf
  const cell = (p: number | undefined, cls: string) =>
    p === undefined ? `<span class="pop-cell none">–</span>` : `<span class="pop-cell ${cls}"><span class="pop-bar"><i style="width:${Math.max(2, p * 100)}%"></i></span><span class="pop-p">${pct(p)}</span></span>`
  popover.classList.toggle("two", two)
  popover.innerHTML = `
    <p class="pop-head">${piece.forced ? "You picked" : "The model wrote"} <b>${shown(chosenText)}</b></p>
    ${inf ? `<p class="pop-plain ${inf.kind}">${explain(inf)}</p>` : ""}
    <div class="pop-grid${two ? " two" : ""}">
      <span class="pop-colhead"></span>${two ? `<span class="pop-colhead">with values${piece.amplified ? ` ×${piece.amplified}` : ""}</span><span class="pop-colhead">no values</span>` : `<span class="pop-colhead">probability</span>`}
      ${rows
        .map(
          (a, i) => `<button type="button" class="pop-row${a.text === chosenText ? " picked" : ""}" data-i="${i}" ${a.text === chosenText || !canBranch ? "disabled" : ""} title="${a.text === chosenText ? "The word in the reply" : "Rewrite the rest of the reply starting from this word"}">
            <span class="pop-word">${shown(a.text)}</span>${cell(pOf(withValues, a.text), "with")}${two ? cell(pOf(inf!.withoutValues, a.text), "without") : ""}
          </button>`,
        )
        .join("")}
    </div>
    <p class="pop-foot">${two ? "Top five words for each, from the same model reading the same reply up to this word. – means not in that top five. " : ""}${canBranch ? "Click a word to rewrite from here." : busy ? "Rewriting is available once the replies finish." : ""}</p>`
  popover.hidden = false
  positionPopover(tok)
  popover.querySelectorAll<HTMLButtonElement>("button[data-i]").forEach((b) =>
    b.addEventListener("click", () => {
      closePopover()
      const a = rows[Number(b.dataset.i)]
      branch(slot, k, { text: a.text, p: pOf(withValues, a.text) ?? 0 })
    }),
  )
}

/** One plain sentence on what the two distributions say about this word. */
function explain(inf: Influence) {
  const w = `<b>${shown(inf.word)}</b>`
  const times = inf.lift >= 1 ? ` (${Math.round(2 ** inf.lift * 10) / 10}× likelier)` : ""
  const topWithout = inf.withoutValues[0]
  const without = inf.pWithout === undefined ? "under " + pct(inf.withoutValues.at(-1)?.p ?? 0) : pct(inf.pWithout)
  if (inf.kind === "changed")
    return `Your values changed the model's first choice here. With no values it would most likely write <b>${shown(topWithout.text)}</b> (${pct(topWithout.p)}); with your values ${w} went from ${without} to ${pct(inf.pWith)}${times}.`
  if (inf.kind === "boosted") return `Same first choice either way, but your values made ${w} more likely: ${without} → ${pct(inf.pWith)}${times}.`
  if (inf.kind === "chance")
    return `This word was the model's #${inf.rank} option, picked by chance (the model samples, it doesn't always take the top word). Its first choice with your values was <b>${shown(inf.withValues[0].text)}</b>.`
  return `Your values made little difference here: ${without} without them, ${pct(inf.pWith)} with them.`
}

function positionPopover(anchor: HTMLElement) {
  const r = anchor.getBoundingClientRect()
  const w = popover.offsetWidth
  const h = popover.offsetHeight
  const left = Math.min(Math.max(12, r.left + r.width / 2 - w / 2), window.innerWidth - w - 12)
  const below = r.bottom + 8 + h < window.innerHeight
  popover.style.left = `${left + window.scrollX}px`
  popover.style.top = `${(below ? r.bottom + 8 : r.top - h - 8) + window.scrollY}px`
}

function closePopover() {
  popover.hidden = true
  output.querySelectorAll(".tok.open").forEach((t) => t.classList.remove("open"))
}

/** Keeps the reply up to the chosen word, swaps in the alternative, and lets the model continue from there. */
function branch(slot: Slot, k: number, alt: Alt) {
  const old = slot.col
  const piece = old.pieces[k]
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const col = newColumn(old.kind, `${old.title.replace(/ · rewritten.*$/, "")} · rewritten from “${alt.text.trim()}”`, old.request)
  col.pieces = old.pieces.slice(0, k).map((p) => ({ ...p }))
  col.text = old.text.slice(0, piece.start) + alt.text
  col.pieces.push({ start: piece.start, end: col.text.length, p: alt.p, alts: piece.alts, forced: true })
  const keep = old.sentences.findIndex((s) => s.end > piece.start)
  col.scores = old.scores.slice(0, keep < 0 ? old.scores.length : keep)
  const next = mountSlot(col, slot.el)
  slots = slots.map((s) => (s === slot ? next : s))
  reveal = null
  paint()
  settle([runColumn(col, engine, tidy(engine.write(col.request, signal, col.text), col.text), signal, paint)], signal).then(() => compareWords(signal))
}

const pct = (p: number) => `${p >= 0.995 ? ">99" : p < 0.01 ? "<1" : Math.round(p * 100)}%`
const shown = (t: string) => (t.trim() ? esc(t.trim()) : t.includes("\n") ? "↵" : "␣")

/* Values, strength, prompt */

function renderChips() {
  chipsEl.innerHTML =
    allValues()
      .map(
        (v) =>
          `<button type="button" class="chip${v.custom ? " custom" : ""}" data-id="${esc(v.id)}" title="${esc(v.principle)}"><i></i>${esc(v.name)}${
            v.custom ? `<span class="chip-x" role="button" aria-label="Remove ${esc(v.name)}" data-remove="${esc(v.id)}">×</span>` : ""
          }</button>`,
      )
      .join("") + `<button type="button" class="chip add" id="add-value">+ Add your own</button>`
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip[data-id]").forEach((b) =>
    b.addEventListener("click", (e) => {
      const id = b.dataset.id as ValueId
      if ((e.target as HTMLElement).dataset.remove) return removeValue(id)
      if (ui.chosen.includes(id)) ui.chosen = ui.chosen.filter((x) => x !== id)
      else if (ui.chosen.length < MAX_CHOSEN) ui.chosen = [...ui.chosen, id]
      else return pulse(b)
      chosenChanged()
    }),
  )
  $("add-value").addEventListener("click", openAddForm)
  paintChips()
}

function chosenChanged() {
  saveUi()
  paintChips()
  renderPrompt()
  paint()
  scheduleRerun()
}

function removeValue(id: ValueId) {
  removeCustomValue(id)
  const was = ui.chosen.includes(id)
  ui.chosen = ui.chosen.filter((x) => x !== id)
  renderChips()
  if (was) chosenChanged()
}

function openAddForm() {
  const form = $<HTMLFormElement>("add-form")
  form.hidden = false
  $<HTMLInputElement>("new-name").focus()
}

function bindAddForm() {
  const form = $<HTMLFormElement>("add-form")
  const name = $<HTMLInputElement>("new-name")
  const meaning = $<HTMLInputElement>("new-meaning")
  const close = () => {
    form.hidden = true
    name.value = ""
    meaning.value = ""
  }
  $("add-cancel").addEventListener("click", close)
  form.addEventListener("keydown", (e) => e.key === "Escape" && close())
  form.addEventListener("submit", (e) => {
    e.preventDefault()
    if (!name.value.trim()) return name.focus()
    const id = addCustomValue(name.value, meaning.value)
    close()
    renderChips()
    if (!ui.chosen.includes(id) && ui.chosen.length < MAX_CHOSEN) {
      ui.chosen = [...ui.chosen, id]
      chosenChanged()
    } else if (!ui.chosen.includes(id)) setStatus(`Added. You already have ${MAX_CHOSEN} values chosen; remove one to use it.`)
  })
}

function paintChips() {
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip[data-id]").forEach((b) => {
    const id = b.dataset.id as ValueId
    const on = ui.chosen.includes(id)
    b.setAttribute("aria-pressed", String(on))
    b.style.setProperty("--c", on ? colorOf(ui.chosen, id) : "transparent")
    b.classList.toggle("full", !on && ui.chosen.length >= MAX_CHOSEN)
  })
}

function pulse(el: HTMLElement) {
  el.classList.remove("nope")
  void el.offsetWidth
  el.classList.add("nope")
  setStatus(`Pick up to ${MAX_CHOSEN} values. Tap one to remove it.`)
}

function renderStrength() {
  strengthEl.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.v === ui.strength))
    b.onclick = () => {
      ui.strength = b.dataset.v as Strength
      saveUi()
      renderStrength()
      scheduleRerun()
    }
  })
}

function renderPrompt() {
  const text = ui.prompt
  const drivers = (driverCache.get(`${engine.label}\n${text}`) ?? []).filter((d) => ui.chosen.includes(d.value))
  const marks: { start: number; end: number; value: ValueId }[] = []
  for (const d of drivers) {
    const start = text.toLowerCase().indexOf(d.phrase.toLowerCase())
    const end = start + d.phrase.length
    if (start >= 0 && !marks.some((m) => start < m.end && end > m.start)) marks.push({ start, end, value: d.value })
  }
  marks.sort((a, b) => a.start - b.start)
  let html = ""
  let at = 0
  for (const m of marks) {
    html += esc(text.slice(at, m.start)) + `<mark style="--c:${colorOf(ui.chosen, m.value)}" title="Pulls on ${nameOf(m.value).toLowerCase()}">${esc(text.slice(m.start, m.end))}</mark>`
    at = m.end
  }
  html += esc(text.slice(at))
  promptView.innerHTML = html || `<span class="placeholder">Ask something…</span>`
  focusNote.textContent = marks.length ? "Highlighted: phrases the scorer says pull on your values" : ""
}

function startEditing() {
  promptEdit.value = ui.prompt
  promptView.hidden = true
  promptEdit.hidden = false
  autosize()
  promptEdit.focus()
  promptEdit.setSelectionRange(promptEdit.value.length, promptEdit.value.length)
}

function stopEditing() {
  if (promptEdit.hidden) return
  ui.prompt = promptEdit.value.trim()
  saveUi()
  promptEdit.hidden = true
  promptView.hidden = false
  renderPrompt()
}

function autosize() {
  promptEdit.style.height = "auto"
  promptEdit.style.height = `${promptEdit.scrollHeight}px`
}

function renderEmpty() {
  $("diff").hidden = true
  output.classList.remove("two")
  output.innerHTML = `<div class="empty-state"><p>Pick the values the assistant should hold, then press <b>Run</b>.</p><p class="sub">Each sentence of the reply is checked against them as it's written.</p></div>`
}

function setStatus(text: string) {
  statusEl.textContent = text
}

function listOf(items: string[]) {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`
}

/* Tuning: what to highlight, temperature, and value amplification */

function bindTune() {
  const panel = $("tune")
  const btn = $("tune-btn")
  const toggle = (open: boolean) => {
    panel.hidden = !open
    btn.setAttribute("aria-expanded", String(open))
  }
  btn.addEventListener("click", (e) => (e.stopPropagation(), toggle(panel.hidden)))
  document.addEventListener("click", (e) => !panel.hidden && !panel.contains(e.target as Node) && toggle(false))
  document.addEventListener("keydown", (e) => e.key === "Escape" && toggle(false))
  $("highlight").querySelectorAll<HTMLButtonElement>("button").forEach((b) =>
    b.addEventListener("click", () => {
      ui.highlight = b.dataset.v as Highlight
      saveUi()
      updateEngineUi()
      paint()
    }),
  )
  const temp = $<HTMLInputElement>("temperature")
  const amp = $<HTMLInputElement>("amplify")
  temp.addEventListener("input", () => ((ui.temperature = Number(temp.value)), saveUi(), paintTune()))
  amp.addEventListener("input", () => ((ui.amplify = Number(amp.value)), saveUi(), paintTune()))
  temp.addEventListener("change", () => slots.length && run())
  amp.addEventListener("change", () => slots.length && scheduleRerun())
  $("tune-reset").addEventListener("click", () => {
    const rerun = ui.temperature !== TUNE_DEFAULTS.temperature || ui.amplify !== TUNE_DEFAULTS.amplify
    Object.assign(ui, TUNE_DEFAULTS)
    saveUi()
    updateEngineUi()
    paint()
    if (rerun && slots.length) run()
  })
}

function paintTune() {
  const local = engine.kind === "local"
  $("highlight").querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.v === ui.highlight))
    b.disabled = !local
  })
  $("highlight-note").textContent = local ? "" : "Word-level highlights need an in-browser model, which shares its probabilities."
  $<HTMLInputElement>("temperature").value = String(ui.temperature)
  $("temperature-out").textContent = ui.temperature.toFixed(1)
  const amp = $<HTMLInputElement>("amplify")
  amp.value = String(ui.amplify)
  amp.disabled = !local
  $("amplify-out").textContent = ui.amplify > 0 ? `×${ui.amplify}` : "off"
  $("amplify-note").textContent = local
    ? "Pushes each word further in the direction your values move the model. Each word takes two model calls, so it writes more slowly."
    : "Needs an in-browser model: it works from the model's word probabilities."
  $("tune-btn").classList.toggle("active", ui.temperature !== TUNE_DEFAULTS.temperature || (local && ui.amplify > 0))
}

/* Engines and settings */

function savedKind(): EngineKind {
  const k = read(STORE.engine)
  if (k === "sim" || k === "gemini" || k === "local") return k
  return read(STORE.key) && read(STORE.model) ? "gemini" : "sim"
}

function geminiFromStorage() {
  const key = read(STORE.key)
  const model = read(STORE.model)
  return key && model ? geminiEngine(key, model) : null
}

function engineFromStorage(): Engine {
  const kind = savedKind()
  if (kind === "local") {
    const id = read(STORE.local) || LOCAL_MODELS[0].id
    progress = { fraction: 0, text: "Starting", done: false }
    return localEngine(id, geminiFromStorage(), onProgress)
  }
  progress = null
  return (kind === "gemini" && geminiFromStorage()) || simEngine
}

/** One quiet line explaining what can be done with the replies on screen. */
function paintHint() {
  const hint = $("hint")
  const local = engine.kind === "local"
  hint.hidden = !slots.length
  hint.innerHTML =
    `Drag <span class="key-knob"></span> back for a new ending, or click it at the end for more` +
    (local && ui.highlight === "values"
      ? ` · <span class="key-lift">purple</span> = how much likelier your values made a word · <span class="key-flip">colored</span> = values changed its first choice`
      : local && ui.highlight === "doubt"
        ? ` · <span class="key-shade">shaded</span> = where the model hesitated`
        : "")
}

function onProgress(p: Progress) {
  progress = p
  updateEngineUi()
  if (p.error) setStatus(p.error)
}

function updateEngineUi() {
  if (!engine) return
  const local = engine.kind === "local"
  const loading = local && !!progress && !progress.done
  $("model-label").textContent = loading
    ? `Loading ${engine.label.replace(" · in browser", "")} · ${Math.round((progress?.fraction ?? 0) * 100)}%`
    : engine.kind === "sim"
      ? "Sim mode · open settings to use a real model"
      : engine.label
  $("model-btn").classList.toggle("sim", engine.kind === "sim")
  $("model-btn").classList.toggle("loading", loading)
  loadBar.hidden = !loading
  loadBar.style.setProperty("--p", String(progress?.fraction ?? 0))
  paintHint()
  output.classList.toggle("conf", local && ui.highlight === "doubt")
  output.classList.toggle("infl", local && ui.highlight === "values")
  output.classList.toggle("plainview", !local || ui.highlight === "off")
  paintTune()
}

function bindSettings() {
  const dialog = $<HTMLDialogElement>("settings")
  const seg = $("engine-seg")
  const keyInput = $<HTMLInputElement>("key")
  const modelSel = $<HTMLSelectElement>("model")
  const localSel = $<HTMLSelectElement>("local-model")
  const modelsStatus = $("models-status")
  let kind: EngineKind = savedKind()

  localSel.innerHTML = LOCAL_MODELS.map((m) => `<option value="${m.id}">${m.name} · ${m.note}</option>`).join("")
  $("webgpu-note").textContent = hasWebGPU()
    ? "Downloads once, then stays cached in this browser. It runs on this computer's graphics chip, and nothing is sent anywhere."
    : "This browser can't run local models (no WebGPU). Use a current Chrome or Edge."

  const notes: Record<EngineKind, string> = {
    sim: "Canned replies for the examples. No setup, and nothing is sent anywhere.",
    gemini: "Google's Gemini writes and scores the replies. Best writing.",
    local: "A small open model runs inside this browser and shows how sure it was of every word. With a Gemini key, Gemini also scores the values.",
  }
  const paintKind = () => {
    seg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === kind)))
    $("local-field").hidden = kind !== "local"
    $("key-field").hidden = kind === "sim"
    $("model-field").hidden = kind === "sim"
    $("key-label").textContent = kind === "local" ? "Gemini API key (optional, for scoring values)" : "Gemini API key"
    $("engine-note").textContent = notes[kind]
  }
  seg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => ((kind = b.dataset.v as EngineKind), paintKind())))

  const open = () => {
    kind = savedKind()
    keyInput.value = read(STORE.key)
    localSel.value = read(STORE.local) || LOCAL_MODELS[0].id
    const saved = read(STORE.model)
    modelSel.innerHTML = saved ? `<option>${esc(saved)}</option>` : `<option value="">Paste a key, then Find my models</option>`
    modelsStatus.textContent = ""
    paintKind()
    dialog.showModal()
  }
  const findModels = async () => {
    const key = keyInput.value.trim()
    if (!key) return void (modelsStatus.textContent = "Paste a key first.")
    modelsStatus.textContent = "Looking…"
    try {
      const models = await listModels(key)
      if (!models.length) throw new Error("No text models are available for this key.")
      const pick = models.includes(read(STORE.model)) ? read(STORE.model) : defaultModel(models)
      modelSel.innerHTML = models.map((m) => `<option ${m === pick ? "selected" : ""}>${esc(m)}</option>`).join("")
      modelsStatus.textContent = `Selected ${pick}, the fastest one available.`
    } catch (e) {
      modelsStatus.textContent = e instanceof Error ? e.message : String(e)
    }
  }
  $("settings-btn").addEventListener("click", open)
  $("model-btn").addEventListener("click", open)
  $("load-models").addEventListener("click", findModels)
  keyInput.addEventListener("change", () => keyInput.value.trim() && !modelSel.value && findModels())
  $("lab-import").addEventListener("click", () => {
    const top = topValuesFromLab($<HTMLTextAreaElement>("lab-md").value)
    $("lab-status").textContent = top.length ? `Using ${listOf(top.map((id) => nameOf(id)))}.` : "No priorities found. Paste the whole Markdown."
    if (!top.length) return
    ui.chosen = top
    saveUi()
    paintChips()
    renderPrompt()
  })
  dialog.addEventListener("close", () => {
    if (dialog.returnValue !== "save") return
    write(STORE.engine, kind)
    write(STORE.key, keyInput.value.trim())
    if (modelSel.value) write(STORE.model, modelSel.value)
    write(STORE.local, localSel.value)
    engine = engineFromStorage()
    updateEngineUi()
    setStatus(kind === "gemini" && engine.kind !== "gemini" ? "Paste a key and press Find my models to use Gemini." : "")
    renderPrompt()
    paint()
  })
}

function loadUi(): Partial<Ui> {
  try {
    return JSON.parse(read(STORE.ui) || "{}")
  } catch {
    return {}
  }
}

function saveUi() {
  write(STORE.ui, JSON.stringify(ui))
}

function read(key: string) {
  try {
    return localStorage.getItem(key)?.trim() ?? ""
  } catch {
    return ""
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {}
}

function esc(t: string) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}
