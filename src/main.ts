import { allConflicts, oriented, parseConstitution, SAMPLE_CONSTITUTION, V, type ValueId } from "./constitution.ts"
import { DILEMMAS, geminiEngine, simEngine, steerWords, type Engine, type Steer } from "./engine.ts"
import { defaultModel, listModels } from "./gemini.ts"
import { emptyPanel, fillPanel, renderPanel, type PanelState } from "./panel.ts"

const STORE = { key: "vm-gemini-key", model: "vm-gemini-model", constitution: "vm-constitution", ui: "vm-ui" }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const exampleSel = $<HTMLSelectElement>("example")
const promptBox = $<HTMLTextAreaElement>("prompt")
const conflictSel = $<HTMLSelectElement>("conflict")
const steerInput = $<HTMLInputElement>("steer")
const follow = $<HTMLInputElement>("follow")
const compare = $<HTMLInputElement>("compare")
const auto = $<HTMLInputElement>("auto")
const statusEl = $("status")
const panelsEl = $("panels")
const badge = $<HTMLButtonElement>("engine-badge")

let constitutionMd = read(STORE.constitution) || SAMPLE_CONSTITUTION
let constitution = parseConstitution(constitutionMd)
let engine: Engine = engineFromStorage()
let panels: PanelState[] = []
let controller: AbortController | undefined
/** The unsteered reply is the same for every slider position, so it is kept per prompt and model. */
const baselines = new Map<string, PanelState>()

setupExamples()
setupConflicts()
restoreUi()
updateSteerLabels()
updateBadge()
renderAll()
bindSettings()
bindConstitution()
document.body.dataset.ready = "1"

exampleSel.addEventListener("change", () => {
  const d = DILEMMAS[Number(exampleSel.value)]
  if (!d) return
  promptBox.value = d.prompt
  setConflict(oriented(constitution, d.a, d.b))
  saveUi()
})
conflictSel.addEventListener("change", () => (updateSteerLabels(), renderAll(), saveUi()))
steerInput.addEventListener("input", updateSteerLabels)
steerInput.addEventListener("change", () => {
  saveUi()
  if (auto.checked) generate()
})
for (const el of [follow, compare, auto, promptBox]) el.addEventListener("change", saveUi)
$("generate").addEventListener("click", generate)
$("score-pasted").addEventListener("click", scorePasted)
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault()
    generate()
  }
})

function currentPair(): [ValueId, ValueId] {
  return conflictSel.value.split("|") as [ValueId, ValueId]
}

function currentSteer(): Steer | null {
  const [x, y] = currentPair()
  const t = Number(steerInput.value) / 100
  return Math.abs(t) < 0.15 ? null : { x, y, t }
}

function generate() {
  const prompt = promptBox.value.trim()
  if (!prompt) return setStatus("Write a prompt first.", true)
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const steer = currentSteer()
  const constitutionUsed = follow.checked ? constitution : null
  const changed = !!steer || !!constitutionUsed
  const main = emptyPanel(titleOf(steer, !!constitutionUsed), prompt)
  panels = [main]
  const jobs = [fillPanel(main, engine, engine.write({ prompt, steer, constitution: constitutionUsed }, signal), signal, renderAll)]

  if (compare.checked && changed) {
    const id = `${engine.label}\n${prompt}`
    const cached = baselines.get(id)
    if (cached && !cached.error) panels = [cached, main]
    else {
      const base = emptyPanel("Unsteered", prompt)
      panels = [base, main]
      jobs.push(fillPanel(base, engine, engine.write({ prompt, steer: null, constitution: null }, signal), signal, renderAll).then(() => void (!base.error && baselines.set(id, base))))
    }
  }
  setStatus("")
  renderAll()
  Promise.all(jobs).then(() => {
    if (!signal.aborted && !panels.some((p) => p.error)) setStatus(`Done with ${engine.label}.`)
  })
}

function scorePasted() {
  const text = $<HTMLTextAreaElement>("pasted").value.trim()
  const prompt = promptBox.value.trim()
  if (!text) return setStatus("Paste a reply first.", true)
  controller?.abort()
  controller = new AbortController()
  const panel = emptyPanel("Pasted reply", prompt || "(no prompt)")
  panels = [panel]
  fillPanel(panel, engine, text, controller.signal, renderAll)
}

function titleOf(steer: Steer | null, followed: boolean) {
  const s = steerWords(steer)
  const parts = [s ? `Steered ${s.level} toward ${V[s.to].name}` : "", followed ? "following constitution" : ""].filter(Boolean)
  return parts.length ? parts.join(", ") : "Unsteered"
}

function renderAll() {
  const [x, y] = currentPair()
  if (!panels.length) {
    panelsEl.innerHTML = `<article class="panel"><header class="phead"><h2>Ready</h2></header><p class="waiting">Pick an example or write a prompt, drag the slider toward a value, and press Generate. ${
      engine === simEngine ? "You're in sim mode: replies are canned. Add a free Gemini key in Settings for real ones." : ""
    }</p></article>`
    return
  }
  while (panelsEl.children.length > panels.length) panelsEl.lastElementChild!.remove()
  panels.forEach((p, i) => {
    let el = panelsEl.children[i] as HTMLElement | undefined
    if (!el || el.tagName !== "ARTICLE" || el.dataset.live !== "1") {
      el = document.createElement("article")
      el.className = "panel"
      el.dataset.live = "1"
      if (panelsEl.children[i]) panelsEl.children[i].replaceWith(el)
      else panelsEl.append(el)
    }
    renderPanel(el, p, { x, y, constitution })
  })
  const failed = panels.find((p) => p.error)
  if (failed) setStatus(failed.error!, true)
}

function updateSteerLabels() {
  const [x, y] = currentPair()
  $("end-x").textContent = `← ${V[x].name}`
  $("end-y").textContent = `${V[y].name} →`
  const s = steerWords(currentSteer())
  $("steer-readout").textContent = s ? `Lean ${s.level} toward ${V[s.to].name.toLowerCase()}` : "No steering: the model's default balance"
}

function setupExamples() {
  exampleSel.innerHTML = `<option value="">(write your own)</option>` + DILEMMAS.map((d, i) => `<option value="${i}">${d.title}</option>`).join("")
}

function setupConflicts() {
  const keep = conflictSel.value
  conflictSel.innerHTML = allConflicts(constitution)
    .map(([x, y]) => `<option value="${x}|${y}">${V[x].name} vs ${V[y].name}</option>`)
    .join("")
  if (keep) setConflict(keep.split("|") as [ValueId, ValueId])
}

function setConflict([x, y]: [ValueId, ValueId]) {
  const options = [...conflictSel.options].map((o) => o.value)
  conflictSel.value = options.includes(`${x}|${y}`) ? `${x}|${y}` : options.includes(`${y}|${x}`) ? `${y}|${x}` : options[0]
  updateSteerLabels()
  renderAll()
}

function setStatus(text: string, isError = false) {
  statusEl.textContent = text
  statusEl.classList.toggle("err", isError)
}

function engineFromStorage(): Engine {
  const key = read(STORE.key)
  const model = read(STORE.model)
  return key && model ? geminiEngine(key, model) : simEngine
}

function updateBadge() {
  badge.textContent = engine.label
  badge.classList.toggle("sim", engine === simEngine)
}

function bindSettings() {
  const dialog = $<HTMLDialogElement>("settings")
  const keyInput = $<HTMLInputElement>("key")
  const modelSel = $<HTMLSelectElement>("model")
  const modelsStatus = $("models-status")
  const open = () => {
    keyInput.value = read(STORE.key)
    const saved = read(STORE.model)
    modelSel.innerHTML = saved ? `<option>${saved}</option>` : `<option value="">Load your models first</option>`
    modelsStatus.textContent = ""
    dialog.showModal()
  }
  $("open-settings").addEventListener("click", open)
  badge.addEventListener("click", open)
  $("load-models").addEventListener("click", async () => {
    const key = keyInput.value.trim()
    if (!key) return void (modelsStatus.textContent = "Paste a key first.")
    modelsStatus.textContent = "Loading…"
    try {
      const models = await listModels(key)
      if (!models.length) throw new Error("No text models are available for this key.")
      const pick = models.includes(read(STORE.model)) ? read(STORE.model) : defaultModel(models)
      modelSel.innerHTML = models.map((m) => `<option ${m === pick ? "selected" : ""}>${m}</option>`).join("")
      modelsStatus.textContent = `${models.length} models. ${pick} is the fastest pick.`
    } catch (e) {
      modelsStatus.textContent = e instanceof Error ? e.message : String(e)
    }
  })
  dialog.addEventListener("close", () => {
    if (dialog.returnValue !== "save") return
    const key = keyInput.value.trim()
    write(STORE.key, key)
    if (modelSel.value) write(STORE.model, modelSel.value)
    if (key && !modelSel.value) setStatus("Press Load my models and pick a model, then save again.", true)
    engine = engineFromStorage()
    baselines.clear()
    updateBadge()
    renderAll()
  })
}

function bindConstitution() {
  const dialog = $<HTMLDialogElement>("constitution-dialog")
  const box = $<HTMLTextAreaElement>("constitution-md")
  const summary = $("constitution-summary")
  const describe = () => {
    const c = parseConstitution(box.value)
    summary.textContent = c.order.length ? `Read ${c.order.length} priorities (${c.order.map((id) => V[id].name).join(" > ")}) and ${c.rules.size} rules.` : "No priorities found yet. Paste Constitution Lab's Markdown."
  }
  $("open-constitution").addEventListener("click", () => {
    box.value = constitutionMd
    describe()
    dialog.showModal()
  })
  box.addEventListener("input", describe)
  $("reset-constitution").addEventListener("click", () => ((box.value = SAMPLE_CONSTITUTION), describe()))
  dialog.addEventListener("close", () => {
    if (dialog.returnValue !== "save") return
    constitutionMd = box.value
    constitution = parseConstitution(constitutionMd)
    write(STORE.constitution, constitutionMd)
    setupConflicts()
    renderAll()
  })
}

function saveUi() {
  write(STORE.ui, JSON.stringify({ prompt: promptBox.value, conflict: conflictSel.value, steer: steerInput.value, follow: follow.checked, compare: compare.checked, auto: auto.checked, example: exampleSel.value }))
}

function restoreUi() {
  let ui: Record<string, unknown> = {}
  try {
    ui = JSON.parse(read(STORE.ui) || "{}")
  } catch {}
  exampleSel.value = typeof ui.example === "string" ? ui.example : "0"
  const d = DILEMMAS[Number(exampleSel.value)]
  promptBox.value = typeof ui.prompt === "string" && ui.prompt ? ui.prompt : (d?.prompt ?? DILEMMAS[0].prompt)
  if (typeof ui.conflict === "string") setConflict(ui.conflict.split("|") as [ValueId, ValueId])
  else setConflict(oriented(constitution, d?.a ?? "candor", d?.b ?? "kindness"))
  if (typeof ui.steer === "string") steerInput.value = ui.steer
  if (typeof ui.follow === "boolean") follow.checked = ui.follow
  if (typeof ui.compare === "boolean") compare.checked = ui.compare
  if (typeof ui.auto === "boolean") auto.checked = ui.auto
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
