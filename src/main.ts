import { createColumnView, newColumn, runColumn, type Column } from "./column.ts"
import { EXAMPLES, geminiEngine, simEngine, type Driver, type Engine } from "./engine.ts"
import { defaultModel, listModels } from "./gemini.ts"
import { colorOf, MAX_CHOSEN, topValuesFromLab, V, VALUES, type Strength, type ValueId } from "./values.ts"

const STORE = { key: "vm-gemini-key", model: "vm-gemini-model", ui: "vm-ui-2" }
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

type Ui = { prompt: string; example: number; chosen: ValueId[]; strength: Strength; compare: boolean }
const ui: Ui = { prompt: EXAMPLES[0].prompt, example: 0, chosen: ["candor", "kindness"], strength: "firm", compare: true, ...loadUi() }

let engine: Engine = engineFromStorage()
let controller: AbortController | undefined
let rerunTimer: number | undefined
let columns: { col: Column; update: ReturnType<typeof createColumnView> }[] = []
/** The plain reply doesn't depend on the chosen values, so it is kept per model and prompt. */
const plainCache = new Map<string, Column>()
const driverCache = new Map<string, Driver[]>()

renderChips()
renderStrength()
renderPrompt()
compare.checked = ui.compare
renderEmpty()
updateModel()
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
  const ex = EXAMPLES[ui.example]
  ui.prompt = ex.prompt
  saveUi()
  renderPrompt()
  run()
})
$("run").addEventListener("click", run)
compare.addEventListener("change", () => ((ui.compare = compare.checked), saveUi(), scheduleRerun()))
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) (e.preventDefault(), run())
})

function run() {
  window.clearTimeout(rerunTimer)
  const prompt = ui.prompt.trim()
  if (!prompt) return setStatus("Write a question first.")
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  setStatus("")

  const plainId = `${engine.label}\n${prompt}`
  const wanted: { col: Column; job?: Promise<void> }[] = []
  if (ui.compare || !ui.chosen.length) {
    const cached = plainCache.get(plainId)
    if (cached && !cached.error) wanted.push({ col: cached })
    else {
      const col = newColumn("No values", prompt)
      wanted.push({ col, job: runColumn(col, engine, engine.write({ prompt, values: [], strength: ui.strength }, signal), signal, paint).then(() => void (!col.error && !signal.aborted && plainCache.set(plainId, col))) })
    }
  }
  if (ui.chosen.length) {
    const col = newColumn(`With ${listOf(ui.chosen.map((id) => V[id].name.toLowerCase()))}`, prompt)
    wanted.push({ col, job: runColumn(col, engine, engine.write({ prompt, values: ui.chosen, strength: ui.strength }, signal), signal, paint) })
  }

  output.replaceChildren()
  output.classList.toggle("two", wanted.length === 2)
  columns = wanted.map(({ col }) => {
    const el = document.createElement("article")
    el.className = "col"
    output.append(el)
    return { col, update: createColumnView(el) }
  })
  paint()
  loadDrivers(prompt, signal)
  Promise.all(wanted.map((w) => w.job)).then(() => {
    if (signal.aborted) return
    const failed = columns.find((c) => c.col.error)
    setStatus(failed ? failed.col.error! : "")
  })
}

function paint() {
  const plain = columns.find((c) => c.col.title === "No values")?.col
  for (const { col, update } of columns) update(col, ui.chosen, col === plain ? undefined : plain)
}

function scheduleRerun() {
  if (!columns.length) return
  window.clearTimeout(rerunTimer)
  rerunTimer = window.setTimeout(run, RERUN_MS)
}

async function loadDrivers(prompt: string, signal: AbortSignal) {
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

function renderChips() {
  chipsEl.innerHTML = VALUES.map((v) => `<button type="button" class="chip" data-id="${v.id}" title="${esc(v.principle)}"><i></i>${v.name}</button>`).join("")
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip").forEach((b) =>
    b.addEventListener("click", () => {
      const id = b.dataset.id as ValueId
      if (ui.chosen.includes(id)) ui.chosen = ui.chosen.filter((x) => x !== id)
      else if (ui.chosen.length < MAX_CHOSEN) ui.chosen = [...ui.chosen, id]
      else return pulse(b)
      saveUi()
      paintChips()
      renderPrompt()
      paint()
      scheduleRerun()
    }),
  )
  paintChips()
}

function paintChips() {
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip").forEach((b) => {
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
    html += esc(text.slice(at, m.start)) + `<mark style="--c:${colorOf(ui.chosen, m.value)}" title="Pulls on ${V[m.value].name.toLowerCase()}">${esc(text.slice(m.start, m.end))}</mark>`
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

function engineFromStorage(): Engine {
  const key = read(STORE.key)
  const model = read(STORE.model)
  return key && model ? geminiEngine(key, model) : simEngine
}

function updateModel() {
  $("model-label").textContent = engine.isSim ? "Sim mode · add a key" : engine.label
  $("model-btn").classList.toggle("sim", engine.isSim)
}

function bindSettings() {
  const dialog = $<HTMLDialogElement>("settings")
  const keyInput = $<HTMLInputElement>("key")
  const modelSel = $<HTMLSelectElement>("model")
  const modelsStatus = $("models-status")
  const open = () => {
    keyInput.value = read(STORE.key)
    const saved = read(STORE.model)
    modelSel.innerHTML = saved ? `<option>${esc(saved)}</option>` : `<option value="">Paste a key, then Find my models</option>`
    modelsStatus.textContent = ""
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
    $("lab-status").textContent = top.length ? `Using ${listOf(top.map((id) => V[id].name))}.` : "No priorities found. Paste the whole Markdown."
    if (!top.length) return
    ui.chosen = top
    saveUi()
    paintChips()
    renderPrompt()
  })
  dialog.addEventListener("close", () => {
    if (dialog.returnValue !== "save") return
    write(STORE.key, keyInput.value.trim())
    if (modelSel.value) write(STORE.model, modelSel.value)
    engine = engineFromStorage()
    updateModel()
    if (keyInput.value.trim() && !modelSel.value) setStatus("Press Find my models and save again to switch from sim mode.")
    renderPrompt()
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
