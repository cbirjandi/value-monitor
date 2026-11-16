import { countWords, createColumnView, cutAfter, isFlip, newColumn, probeColumn, runColumn, truncateColumn, type Column } from "./column.ts"
import { EXAMPLES, geminiEngine, simEngine, tidy, type Alt, type Driver, type Engine } from "./engine.ts"
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
const confidence = $<HTMLInputElement>("confidence")
const output = $("output")
const statusEl = $("status")
const focusNote = $("focus-note")
const popover = $("popover")
const loadBar = $("load-bar")

type Ui = { prompt: string; example: number; chosen: ValueId[]; strength: Strength; compare: boolean; confidence: boolean }
const ui: Ui = { prompt: EXAMPLES[0].prompt, example: 0, chosen: ["candor", "kindness"], strength: "firm", compare: true, confidence: true, ...loadUi() }

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
confidence.checked = ui.confidence
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
confidence.addEventListener("change", () => ((ui.confidence = confidence.checked), saveUi(), updateEngineUi()))
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

  const plainId = `${engine.label}\n${prompt}`
  const jobs: Promise<void>[] = []
  const cols: Column[] = []
  if (ui.compare || !ui.chosen.length) {
    const cached = plainCache.get(plainId)
    const covers = (c: Column) => ui.chosen.every((id) => c.scores.every((sc) => !sc || id in sc))
    if (cached && !cached.error && covers(cached)) cols.push(cached)
    else {
      const col = newColumn("plain", "No values", { prompt, values: [], strength: ui.strength })
      cols.push(col)
      jobs.push(runColumn(col, engine, tidy(engine.write(col.request, signal)), signal, paint).then(() => void (!col.error && !signal.aborted && plainCache.set(plainId, col))))
    }
  }
  if (ui.chosen.length) {
    const col = newColumn("guided", `With ${listOf(ui.chosen.map((id) => nameOf(id).toLowerCase()))}`, { prompt, values: ui.chosen, strength: ui.strength })
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
  const settleAt = (from: number) => {
    if (reveal === null) return paint()
    if (reveal >= own()) grow("extend")
    else if (reveal < from) grow("regrow")
    else paint()
  }
  knob.addEventListener("pointerdown", (e) => {
    e.preventDefault()
    knob.setPointerCapture(e.pointerId)
    closePopover()
    window.clearTimeout(keyTimer)
    const from = reveal ?? longest()
    document.body.classList.add("scrubbing")
    const move = (ev: PointerEvent) => set(view.wordsAt(ev.clientX, ev.clientY))
    const up = () => {
      knob.removeEventListener("pointermove", move)
      document.body.classList.remove("scrubbing")
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
    else set(longest())
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
function grow(mode: "extend" | "regrow" | "revalue") {
  if (!slots.length) return
  window.clearTimeout(rerunTimer)
  closePopover()
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const jobs: Promise<void>[] = []
  for (const { col } of slots) {
    if (mode === "revalue" && col.kind !== "guided") continue
    if (mode !== "extend") truncateColumn(col, cutAfter(col.text, reveal ?? countWords(col.text)))
    if (col.kind === "guided") {
      col.request = { prompt: col.request.prompt, values: ui.chosen, strength: ui.strength }
      col.title = `With ${listOf(ui.chosen.map((id) => nameOf(id).toLowerCase()))}`
    }
    const request = { ...col.request, more: mode === "extend" }
    jobs.push(runColumn(col, engine, tidy(engine.write(request, signal, col.text), col.text), signal, paint))
  }
  if (mode === "extend") reveal = null
  setStatus("")
  paint()
  settle(jobs, signal).then(() => {
    if (signal.aborted) return
    if (mode === "extend" && !slots.some((s) => s.col.grew))
      setStatus(engine.kind === "sim" ? "Sim mode can't write more. Open Settings to use a real model." : "The model had nothing more to add here.")
    compareWords(signal)
  })
}

function settle(jobs: Promise<void>[], signal: AbortSignal) {
  return Promise.all(jobs).then(() => {
    if (signal.aborted) return
    const failed = slots.find((s) => s.col.error)
    setStatus(failed ? failed.col.error! : "")
  })
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
  const alts: Alt[] = [...(piece.alts ?? [])]
  if (!alts.some((a) => a.text === chosenText)) alts.push({ text: chosenText, p: piece.p })
  alts.sort((a, b) => b.p - a.p)
  const busy = slots.some((s) => s.col.streaming || s.col.scoring || s.col.probing)
  const canBranch = engine.kind === "local" && !busy
  const flip = isFlip(piece, slot.col.text)
  const plainLine = piece.plain
    ? `<p class="pop-plain${flip ? " flip" : ""}">${
        flip
          ? `Your values changed this word. With no values, the model's first choice was <b>${shown(piece.plain.top.text)}</b> ${pct(piece.plain.top.p)}, and this word was ${piece.plain.p === undefined ? "not in its top five" : pct(piece.plain.p)}.`
          : `With no values the model would also pick this word${piece.plain.p !== undefined ? ` (${pct(piece.plain.p)})` : ""}.`
      }</p>`
    : ""
  popover.innerHTML = `
    <p class="pop-head">${piece.forced ? "You picked" : slot.col.kind === "guided" ? "With your values, the model chose" : "The model chose"} <b>${shown(chosenText)}</b> <span>${pct(piece.p)}</span></p>
    ${plainLine}
    <ol class="pop-list">${alts
      .map(
        (a, i) => `<li class="${a.text === chosenText ? "picked" : ""}">
          <button type="button" data-i="${i}" ${a.text === chosenText || !canBranch ? "disabled" : ""} title="${a.text === chosenText ? "Already chosen" : "Rewrite the rest of the reply starting from this word"}">
            <span class="pop-word">${shown(a.text)}</span><span class="pop-bar"><i style="width:${Math.max(2, a.p * 100)}%"></i></span><span class="pop-p">${pct(a.p)}</span>
          </button></li>`,
      )
      .join("")}</ol>
    <p class="pop-foot">${canBranch ? "Click another word to rewrite from here." : busy ? "Wait for the replies to finish to rewrite from here." : ""}</p>`
  popover.hidden = false
  positionPopover(tok)
  popover.querySelectorAll<HTMLButtonElement>("button[data-i]").forEach((b) =>
    b.addEventListener("click", () => {
      closePopover()
      branch(slot, k, alts[Number(b.dataset.i)])
    }),
  )
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
  output.classList.remove("two")
  output.innerHTML = `<div class="empty-state"><p>Pick the values the assistant should hold, then press <b>Run</b>.</p><p class="sub">Each sentence of the reply is checked against them as it's written.</p></div>`
}

function setStatus(text: string) {
  statusEl.textContent = text
}

function listOf(items: string[]) {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`
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
    `Drag <span class="key-knob"></span> back to write a new ending from there, or to the end to keep it writing · change values to rewrite what comes after it` +
    (local && ui.confidence ? ` · <span class="key-shade">shaded</span> where the model hesitated · <span class="key-flip">colored</span> where your values changed its choice` : "")
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
  $("conf-toggle").hidden = !local
  paintHint()
  output.classList.toggle("conf", local && ui.confidence)
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
