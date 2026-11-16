import { countWords, createColumnView, cutAfter, influenceOf, newColumn, probeColumn, runColumn, truncateColumn, type Column, type Influence } from "./column.ts"
import { DEFAULT_TEMPERATURE, EXAMPLES, geminiEngine, simEngine, tidy, type Alt, type Driver, type Engine, type WriteRequest } from "./engine.ts"
import { defaultModel, listModels } from "./gemini.ts"
import { hasWebGPU, LOCAL_MODELS, localEngine, type Progress } from "./local.ts"
import { addCustomValue, allValues, type Value, MAX_PER_SIDE, nameOf, removeCustomValue, SIDE_COLORS, topValuesFromLab, type Strength, type ValueId } from "./values.ts"

type EngineKind = Engine["kind"]
type Side = "a" | "b"
type View = "compare" | "steer"
type Highlight = "values" | "doubt" | "off"
const STORE = { key: "vm-gemini-key", model: "vm-gemini-model", engine: "vm-engine", local: "vm-local-model", judge: "vm-judge", ui: "vm-ui-5" }
const RERUN_MS = 650

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T
const promptView = $("prompt-view")
const promptEdit = $<HTMLTextAreaElement>("prompt")
const chipsEl = $("chips")
const strengthEl = $("strength")
const output = $("output")
const statusEl = $("status")
const focusNote = $("focus-note")
const popover = $("popover")
const loadBar = $("load-bar")
const hud = $("steer-hud")
const steerInput = $<HTMLInputElement>("steer")

type Ui = { prompt: string; example: number; a: ValueId[]; b: ValueId[]; strength: Strength; view: View; steer: number; highlight: Highlight; temperature: number; amplify: number }
const TUNE_DEFAULTS = { highlight: "values" as Highlight, temperature: DEFAULT_TEMPERATURE, amplify: 0 }
const first = EXAMPLES[0]
const ui: Ui = { prompt: first.prompt, example: 0, a: [first.b], b: [first.a], strength: "firm", view: "compare", steer: 0.5, ...TUNE_DEFAULTS, ...loadUi() }

let progress: Progress | null = null
let engine: Engine = engineFromStorage()
let controller: AbortController | undefined
let rerunTimer: number | undefined
type Slot = { col: Column; view: ReturnType<typeof createColumnView>; el: HTMLElement }
let slots: Slot[] = []
/** How many words of the steered reply are revealed; `null` follows it as it is written. */
let reveal: number | null = null
const driverCache = new Map<string, Driver[]>()
const comparisons = new Map<string, string[]>()

let showPersonas = false
const known = (id: ValueId) => allValues().some((v) => v.id === id)
ui.a = ui.a.filter(known)
ui.b = ui.b.filter(known)
renderChips()
bindAddForm()
renderStrength()
renderView()
renderPrompt()
renderEmpty()
updateEngineUi()
bindSettings()
bindTune()
bindSteer()
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
  // Each example pits one value against another: put one on each side.
  ui.a = [ex.b]
  ui.b = [ex.a]
  saveUi()
  paintChips()
  renderPrompt()
  run()
})
$("run").addEventListener("click", run)
output.addEventListener("click", (e) => {
  const more = (e.target as HTMLElement).closest<HTMLElement>("[data-more]")
  if (more) return grow("extend", slots.find((s) => s.el.contains(more)))
  onTokenClick(e)
})
document.addEventListener("click", (e) => {
  const t = e.target as HTMLElement
  if (!popover.hidden && !popover.contains(t) && !t.closest(".tok")) closePopover()
})
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePopover()
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) (e.preventDefault(), run())
})
window.addEventListener("resize", () => (closePopover(), paint()))

/* Requests */

function blendOf(w = ui.steer) {
  return { a: ui.a, b: ui.b, w }
}

/** A request with the current tuning. In side-by-side view, amplification pushes each side away from the other's values. */
function requestOf(prompt: string, side: Side | "steer", w = ui.steer): WriteRequest {
  const base = { prompt, strength: ui.strength, temperature: ui.temperature }
  if (side === "steer") return { ...base, values: [], blend: blendOf(w), amplify: 0 }
  const values = side === "a" ? ui.a : ui.b
  const against = side === "a" ? ui.b : ui.a
  const amplify = engine.kind === "local" && (values.length || against.length) ? ui.amplify : 0
  return { ...base, values, against, amplify }
}

function names(ids: ValueId[]) {
  return ids.length ? listOf(ids.map((id) => nameOf(id))) : "No values"
}
function titleOf(side: Side | "steer") {
  if (side === "steer") return "Steered reply"
  const amp = engine.kind === "local" && ui.amplify > 0 ? ` · amplified ×${ui.amplify}` : ""
  return `${side.toUpperCase()} · ${names(side === "a" ? ui.a : ui.b)}${amp}`
}

/* Running */

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
  output.replaceChildren()
  const sides: (Side | "steer")[] = ui.view === "steer" ? ["steer"] : ["a", "b"]
  output.classList.toggle("two", sides.length === 2)
  slots = sides.map((side) => mountSlot(newColumn(side, titleOf(side), requestOf(prompt, side))))
  const jobs = slots.map(({ col }) => runColumn(col, tidy(engine.write(col.request, signal)), signal, paint, col.side === "steer" ? ui.steer : undefined))
  paint()
  loadDrivers(prompt, signal)
  settle(jobs, signal)
}

/** Rewrites one side with its new values, keeping the other; both are then re-checked against each other. */
function runSide(side: Side) {
  const slot = slots.find((s) => s.col.side === side)
  if (!slot) return run()
  window.clearTimeout(rerunTimer)
  closePopover()
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const col = newColumn(side, titleOf(side), requestOf(slot.col.request.prompt, side))
  const next = mountSlot(col, slot.el)
  slots = slots.map((s) => (s === slot ? next : s))
  for (const { col: other } of slots)
    if (other !== col) {
      other.request = requestOf(other.request.prompt, other.side as Side)
      other.title = titleOf(other.side as Side)
      other.pieces.forEach((p) => (p.other = undefined))
      other.reprobeFrom = 0
    }
  paint()
  settle([runColumn(col, tidy(engine.write(col.request, signal)), signal, paint)], signal)
}

function mountSlot(col: Column, replace?: HTMLElement): Slot {
  const el = document.createElement("article")
  el.className = "col"
  if (replace) replace.replaceWith(el)
  else output.append(el)
  const view = createColumnView(el, col.side === "steer")
  if (view.knob) bindKnob(view)
  return { col, el, view }
}

function settle(jobs: Promise<void>[], signal: AbortSignal) {
  const diff = $("diff")
  if (ui.view !== "compare") diff.hidden = true
  else if (!diff.hidden) diff.classList.add("stale")
  return Promise.all(jobs).then(async () => {
    if (signal.aborted) return
    const failed = slots.find((s) => s.col.error)
    setStatus(failed ? failed.col.error! : "")
    await compareReplies(signal)
    await compareWords(signal)
  })
}

/** In side-by-side view with an in-browser model, checks each reply word by word against the other side's values. */
async function compareWords(signal: AbortSignal) {
  if (ui.highlight !== "values" || ui.view !== "compare" || !engine.probe) return
  for (const { col } of slots) {
    if (signal.aborted || col.error) continue
    const against = requestOf(col.request.prompt, col.side === "a" ? "b" : "a")
    await probeColumn(col, engine, { ...against, values: col.side === "a" ? ui.b : ui.a }, signal, paint)
  }
}

function paint() {
  for (const { col, view } of slots) {
    const more = col.side !== "steer" && !col.streaming && col.text ? `<button type="button" class="link more" data-more>Keep writing</button>` : ""
    view.update(col, col.side === "steer" ? reveal : null, more)
  }
  output.classList.toggle("scrubbed", reveal !== null)
  paintHint()
}

function scheduleRerun(side?: Side) {
  if (!slots.length) return
  window.clearTimeout(rerunTimer)
  rerunTimer = window.setTimeout(() => {
    if (ui.view === "steer") return grow("regrow")
    if (side) return runSide(side)
    run()
  }, RERUN_MS)
}

/* The steered reply: Spaceship-style cursor and future steering */

/**
 * Dragging the knob moves the cursor through the steered reply, as in the Spaceship IDE: letting go further back throws away
 * what came after and writes a new future from there with the current steering; reaching the end keeps the model writing.
 */
function bindKnob(view: ReturnType<typeof createColumnView>) {
  const knob = view.knob!
  const words = () => countWords(slots[0]?.col.text ?? "")
  const set = (n: number) => {
    reveal = Math.max(0, Math.min(n, words()))
    paint()
  }
  const settleAt = (from: number) => {
    if (reveal === null) return paint()
    if (reveal >= words()) grow("extend")
    else if (reveal < from) grow("regrow")
    else paint()
  }
  knob.addEventListener("pointerdown", (e) => {
    e.preventDefault()
    knob.setPointerCapture(e.pointerId)
    closePopover()
    const from = reveal ?? words()
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
      if (!moved) return void ((reveal === null || reveal >= words()) && grow("extend"))
      settleAt(from)
    }
    knob.addEventListener("pointermove", move)
    knob.addEventListener("pointerup", up, { once: true })
    knob.addEventListener("pointercancel", up, { once: true })
  })
  let keyFrom: number | null = null
  let keyTimer: number | undefined
  knob.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return
    e.preventDefault()
    const now = reveal ?? words()
    keyFrom ??= now
    const step = e.shiftKey ? 5 : 1
    set(e.key === "ArrowLeft" ? now - step : e.key === "ArrowRight" ? now + step : e.key === "Home" ? 0 : words())
    window.clearTimeout(keyTimer)
    keyTimer = window.setTimeout(() => {
      const from = keyFrom ?? now
      keyFrom = null
      settleAt(from)
    }, 700)
  })
}

/**
 * Writes on in place. `extend` adds more after the end. `regrow` replaces everything after the cursor with a new future,
 * written with the current steering (or, in side-by-side view, the current values).
 */
function grow(mode: "extend" | "regrow", only?: Slot) {
  if (!slots.length) return
  if (mode === "extend" && slots.some((s) => s.col.streaming)) return setStatus("Let the reply finish, then keep writing.")
  window.clearTimeout(rerunTimer)
  closePopover()
  controller?.abort()
  controller = new AbortController()
  const { signal } = controller
  const jobs: Promise<void>[] = []
  for (const slot of slots) {
    const { col } = slot
    if (only && slot !== only) continue
    if (mode === "regrow") truncateColumn(col, cutAfter(col.text, reveal ?? countWords(col.text)))
    if (mode === "extend" && engine.kind === "local") col.reprobeFrom = Math.min(col.reprobeFrom ?? Infinity, col.pieces.length)
    col.request = col.side === "steer" ? requestOf(col.request.prompt, "steer") : requestOf(col.request.prompt, col.side)
    const request = { ...col.request, more: mode === "extend" }
    jobs.push(runColumn(col, tidy(engine.write(request, signal, col.text), col.text), signal, paint, col.side === "steer" ? ui.steer : undefined))
  }
  // Writing resumes at the cursor, and the cursor rides along with the new text; the dot in the text marks where steering changed.
  reveal = null
  setStatus("")
  paint()
  settle(jobs, signal).then(() => {
    if (signal.aborted) return
    if (mode === "extend" && !slots.some((s) => (!only || s === only) && s.col.grew))
      setStatus(engine.kind === "sim" ? "Sim mode can't write more. Open Settings to use a real model." : "The model had nothing more to add here.")
  })
}

/**
 * The steering slider is a future knob: it sets the blend of A's and B's values for whatever is written after the cursor.
 * Letting go rewrites that future at once; if the reply is still being written, it is cut where it is and continues with the new blend.
 */
function bindSteer() {
  const show = () => {
    const pb = Math.round(ui.steer * 100)
    $("steer-note").innerHTML = `${100 - pb}% <span class="side-a">A</span> · ${pb}% <span class="side-b">B</span> for everything after <span class="key-knob"></span>. Let go to rewrite it; drag <span class="key-knob"></span> back to steer from earlier.`
    hud.style.setProperty("--mix", String(ui.steer))
  }
  steerInput.addEventListener("input", () => ((ui.steer = Number(steerInput.value)), saveUi(), show()))
  steerInput.addEventListener("change", () => {
    const slot = slots[0]
    if (!slot || slot.col.side !== "steer") return
    if (slot.col.streaming && reveal === null) reveal = countWords(slot.col.text)
    if (reveal === null) grow("extend")
    else grow("regrow")
  })
  steerInput.value = String(ui.steer)
  show()
}

function paintSteerEnds() {
  $("steer-a").innerHTML = `<b>A</b> ${esc(names(ui.a))}`
  $("steer-b").innerHTML = `<b>B</b> ${esc(names(ui.b))}`
}

/* What changed: a second reading of both replies */

async function compareReplies(signal: AbortSignal) {
  const diff = $("diff")
  const list = $("diff-points")
  const note = $("diff-note")
  const a = slots.find((s) => s.col.side === "a")?.col
  const b = slots.find((s) => s.col.side === "b")?.col
  if (ui.view !== "compare" || !a?.text || !b?.text) return void (diff.hidden = true)
  diff.hidden = false
  if (!engine.compare) {
    diff.classList.remove("stale", "loading")
    list.innerHTML = ""
    note.textContent = "Choose who writes the comparison in Settings to see how the replies differ."
    return
  }
  const id = `${engine.label}\n${a.text}\n${b.text}`
  diff.classList.add("loading")
  note.textContent = "Comparing the two replies…"
  try {
    if (!comparisons.has(id)) comparisons.set(id, await engine.compare(a.request.prompt, { text: a.text, values: ui.a }, { text: b.text, values: ui.b }, signal))
    if (signal.aborted) return
    const points = comparisons.get(id)!
    list.innerHTML = points.map((p) => `<li>${esc(p)}</li>`).join("")
    note.textContent = points.length ? `Written by ${engine.judgeLabel ?? engine.label}, reading both full replies.` : "No substantive differences found."
  } catch (e) {
    if (!signal.aborted) note.textContent = e instanceof Error ? e.message : String(e)
  } finally {
    if (!signal.aborted) diff.classList.remove("loading", "stale")
  }
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
  const busy = slots.some((s) => s.col.streaming || s.col.probing)
  const canBranch = engine.kind === "local" && !busy
  const inf = influenceOf(piece, slot.col.text)
  const side = slot.col.side
  const mineLabel = side === "steer" ? "probability" : `${side.toUpperCase()}'s values${piece.amplified ? ` ×${piece.amplified}` : ""}`
  const otherLabel = side === "a" ? "B's values" : "A's values"
  const mine = inf?.mine ?? [...(piece.alts ?? []), { text: chosenText, p: piece.p }].filter((a, i, l) => l.findIndex((b) => b.text === a.text) === i).sort((a, b) => b.p - a.p)
  const rows: Alt[] = [...mine]
  for (const a of inf?.other ?? []) if (!rows.some((r) => r.text === a.text)) rows.push({ text: a.text, p: 0 })
  const pOf = (list: Alt[] | undefined, t: string) => list?.find((a) => a.text === t)?.p
  const two = !!inf
  const cell = (p: number | undefined, cls: string) =>
    p === undefined ? `<span class="pop-cell none">–</span>` : `<span class="pop-cell ${cls}"><span class="pop-bar"><i style="width:${Math.max(2, p * 100)}%"></i></span><span class="pop-p">${pct(p)}</span></span>`
  popover.classList.toggle("two", two)
  popover.innerHTML = `
    <p class="pop-head">${piece.forced ? "You picked" : "The model wrote"} <b>${shown(chosenText)}</b></p>
    ${inf ? `<p class="pop-plain ${inf.kind}">${explain(inf, side as Side)}</p>` : ""}
    <div class="pop-grid${two ? " two" : ""}">
      <span class="pop-colhead"></span><span class="pop-colhead">${mineLabel}</span>${two ? `<span class="pop-colhead">${otherLabel}</span>` : ""}
      ${rows
        .map(
          (a, i) => `<button type="button" class="pop-row${a.text === chosenText ? " picked" : ""}" data-i="${i}" ${a.text === chosenText || !canBranch ? "disabled" : ""} title="${a.text === chosenText ? "The word in the reply" : "Rewrite the rest of the reply starting from this word"}">
            <span class="pop-word">${shown(a.text)}</span>${cell(pOf(mine, a.text), "with")}${two ? cell(pOf(inf!.other, a.text), "without") : ""}
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
      branch(slot, k, { text: a.text, p: pOf(mine, a.text) ?? 0 })
    }),
  )
}

/** One plain sentence on what the two distributions say about this word. */
function explain(inf: Influence, side: Side) {
  const me = side.toUpperCase()
  const them = side === "a" ? "B" : "A"
  const w = `<b>${shown(inf.word)}</b>`
  const times = inf.lift >= 1 ? ` (${Math.round(2 ** inf.lift * 10) / 10}× likelier)` : ""
  const theirs = inf.pOther === undefined ? "under " + pct(inf.other.at(-1)?.p ?? 0) : pct(inf.pOther)
  if (inf.kind === "changed")
    return `${me}'s values changed the model's first choice here. With ${them}'s values it would most likely write <b>${shown(inf.other[0].text)}</b> (${pct(inf.other[0].p)}); with ${me}'s, ${w} went from ${theirs} to ${pct(inf.pMine)}${times}.`
  if (inf.kind === "boosted") return `Same first choice with either side's values, but ${me}'s made ${w} more likely: ${theirs} → ${pct(inf.pMine)}${times}.`
  if (inf.kind === "chance")
    return `This word was the model's #${inf.rank} option, picked by chance (the model samples, it doesn't always take the top word). Its first choice with ${me}'s values was <b>${shown(inf.mine[0].text)}</b>.`
  return `The values made little difference here: ${theirs} with ${them}'s, ${pct(inf.pMine)} with ${me}'s.`
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
  const col = newColumn(old.side, `${old.title.replace(/ · rewritten.*$/, "")} · rewritten from “${alt.text.trim()}”`, old.request)
  col.pieces = old.pieces.slice(0, k).map((p) => ({ ...p }))
  col.segments = old.segments.filter((s) => s.start <= piece.start)
  col.text = old.text.slice(0, piece.start) + alt.text
  col.pieces.push({ start: piece.start, end: col.text.length, p: alt.p, alts: piece.alts, forced: true })
  const next = mountSlot(col, slot.el)
  slots = slots.map((s) => (s === slot ? next : s))
  reveal = null
  paint()
  settle([runColumn(col, tidy(engine.write(col.request, signal, col.text), col.text), signal, paint, col.side === "steer" ? ui.steer : undefined)], signal)
}

function pct(p: number) {
  return `${p >= 0.995 ? ">99" : p < 0.01 ? "<1" : Math.round(p * 100)}%`
}

function shown(t: string) {
  return t.trim() ? esc(t.trim()) : t.includes("\n") ? "↵" : "␣"
}

/* Values: one set per side */

function sideOf(id: ValueId): Side | null {
  return ui.a.includes(id) ? "a" : ui.b.includes(id) ? "b" : null
}

function renderChips() {
  const chip = (v: Value) =>
    `<button type="button" class="chip${v.custom ? " custom" : ""}${v.persona ? " persona" : ""}" data-id="${esc(v.id)}" title="${esc(v.principle)}"><i></i>${esc(v.name)}${
      v.custom ? `<span class="chip-x" role="button" aria-label="Remove ${esc(v.name)}" data-remove="${esc(v.id)}">×</span>` : ""
    }</button>`
  const values = allValues().filter((v) => !v.persona)
  const personas = allValues().filter((v) => v.persona)
  const open = showPersonas || personas.some((v) => sideOf(v.id))
  chipsEl.innerHTML =
    values.map(chip).join("") +
    `<button type="button" class="chip add" id="add-value">+ Add your own</button>` +
    `<button type="button" class="persona-toggle" id="persona-toggle" aria-expanded="${open}">Personalities <span aria-hidden="true">${open ? "−" : "+"}</span></button>` +
    (open ? `<div class="persona-row">${personas.map(chip).join("")}</div>` : "")
  $("persona-toggle").addEventListener("click", () => {
    showPersonas = !open
    if (!showPersonas && personas.some((v) => sideOf(v.id))) showPersonas = true
    renderChips()
  })
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip[data-id]").forEach((b) =>
    b.addEventListener("click", (e) => {
      const id = b.dataset.id as ValueId
      if ((e.target as HTMLElement).dataset.remove) return removeValue(id)
      // Off → A → B → off.
      const now = sideOf(id)
      if (now === "a") {
        if (ui.b.length >= MAX_PER_SIDE) return pulse(b, "b")
        ui.a = ui.a.filter((x) => x !== id)
        ui.b = [...ui.b, id]
        return valuesChanged(["a", "b"])
      }
      if (now === "b") {
        ui.b = ui.b.filter((x) => x !== id)
        return valuesChanged(["b"])
      }
      if (ui.a.length < MAX_PER_SIDE) {
        ui.a = [...ui.a, id]
        return valuesChanged(["a"])
      }
      if (ui.b.length < MAX_PER_SIDE) {
        ui.b = [...ui.b, id]
        return valuesChanged(["b"])
      }
      pulse(b, "a")
    }),
  )
  $("add-value").addEventListener("click", openAddForm)
  paintChips()
}

function valuesChanged(sides: Side[]) {
  saveUi()
  paintChips()
  renderPrompt()
  paintSteerEnds()
  for (const s of slots) if (s.col.side !== "steer") s.col.title = titleOf(s.col.side)
  paint()
  scheduleRerun(sides.length === 1 ? sides[0] : undefined)
}

function paintChips() {
  chipsEl.querySelectorAll<HTMLButtonElement>(".chip[data-id]").forEach((b) => {
    const side = sideOf(b.dataset.id as ValueId)
    b.setAttribute("aria-pressed", String(!!side))
    b.dataset.side = side ?? ""
    b.style.setProperty("--c", side ? SIDE_COLORS[side] : "transparent")
  })
  paintSteerEnds()
}

function pulse(el: HTMLElement, side: Side) {
  el.classList.remove("nope")
  void el.offsetWidth
  el.classList.add("nope")
  setStatus(`Each side holds up to ${MAX_PER_SIDE} values. Clear one from side ${side.toUpperCase()} first.`)
}

function removeValue(id: ValueId) {
  removeCustomValue(id)
  const side = sideOf(id)
  ui.a = ui.a.filter((x) => x !== id)
  ui.b = ui.b.filter((x) => x !== id)
  renderChips()
  if (side) valuesChanged([side])
}

function openAddForm() {
  $<HTMLFormElement>("add-form").hidden = false
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
    if (sideOf(id)) return
    if (ui.a.length < MAX_PER_SIDE) {
      ui.a = [...ui.a, id]
      valuesChanged(["a"])
    } else if (ui.b.length < MAX_PER_SIDE) {
      ui.b = [...ui.b, id]
      valuesChanged(["b"])
    } else setStatus(`Added. Both sides are full; clear a value to use it.`)
  })
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

function renderView() {
  $("view-seg")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.v === ui.view))
      b.onclick = () => {
        if (ui.view === b.dataset.v) return
        ui.view = b.dataset.v as View
        saveUi()
        renderView()
        if (slots.length) run()
      }
    })
  hud.hidden = ui.view !== "steer"
  paintSteerEnds()
}

function renderPrompt() {
  const text = ui.prompt
  const drivers = (driverCache.get(`${engine.label}\n${text}`) ?? []).filter((d) => sideOf(d.value))
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
    html += esc(text.slice(at, m.start)) + `<mark style="--c:${SIDE_COLORS[sideOf(m.value)!]}" title="Pulls on ${nameOf(m.value).toLowerCase()}">${esc(text.slice(m.start, m.end))}</mark>`
    at = m.end
  }
  html += esc(text.slice(at))
  promptView.innerHTML = html || `<span class="placeholder">Ask something…</span>`
  focusNote.textContent = marks.length ? "Highlighted: phrases that pull on each side's values" : ""
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
  output.innerHTML = `<div class="empty-state"><p>Give each side its values, then press <b>Run</b>.</p><p class="sub">Side by side writes one reply per side. Steer writes one reply you can push between them as it's written.</p></div>`
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
  $("highlight")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((b) =>
      b.addEventListener("click", () => {
        ui.highlight = b.dataset.v as Highlight
        saveUi()
        updateEngineUi()
        paint()
        if (ui.highlight === "values" && controller && !slots.some((s) => s.col.streaming || s.col.probing)) compareWords(controller.signal)
      }),
    )
  const temp = $<HTMLInputElement>("temperature")
  const amp = $<HTMLInputElement>("amplify")
  temp.addEventListener("input", () => ((ui.temperature = Number(temp.value)), saveUi(), paintTune()))
  amp.addEventListener("input", () => ((ui.amplify = Number(amp.value)), saveUi(), paintTune()))
  temp.addEventListener("change", () => slots.length && run())
  amp.addEventListener("change", () => slots.length && ui.view === "compare" && run())
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
  $("highlight")
    .querySelectorAll<HTMLButtonElement>("button")
    .forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.v === ui.highlight))
      b.disabled = !local
    })
  $("highlight-note").textContent = local ? "Value influence compares each side's words with the other side's values (side-by-side view)." : "Word-level highlights need an in-browser model, which shares its probabilities."
  $<HTMLInputElement>("temperature").value = String(ui.temperature)
  $("temperature-out").textContent = ui.temperature.toFixed(1)
  const amp = $<HTMLInputElement>("amplify")
  amp.value = String(ui.amplify)
  amp.disabled = !local
  $("amplify-out").textContent = ui.amplify > 0 ? `×${ui.amplify}` : "off"
  $("amplify-note").textContent = local
    ? "Side by side: pushes each reply's words further from what the other side's values would say. Each word takes two model calls, so it writes more slowly."
    : "Needs an in-browser model: it works from the model's word probabilities."
  $("tune-btn").classList.toggle("active", ui.temperature !== TUNE_DEFAULTS.temperature || (local && ui.amplify > 0))
}

/** One quiet line explaining what can be done with the replies on screen. */
function paintHint() {
  const hint = $("hint")
  const local = engine.kind === "local"
  hint.hidden = !slots.length
  const parts: string[] = []
  if (ui.view === "steer") parts.push(`Drag <span class="key-knob"></span> back to rewrite from there, or click it at the end for more · underline color shows the A–B blend that wrote each stretch`)
  else if (local && ui.highlight === "values") parts.push(`<span class="key-lift">purple</span> = how much likelier this side's values made a word than the other side's · <span class="key-flip">colored</span> = they changed its first choice`)
  if (local && ui.highlight === "doubt") parts.push(`<span class="key-shade">shaded</span> = where the model hesitated`)
  if (local) parts.push("click any word for its probabilities")
  hint.innerHTML = parts.join(" · ")
}

/* Engines and settings */

function savedKind(): EngineKind {
  const k = read(STORE.engine)
  if (k === "sim" || k === "gemini" || k === "local") return k
  return read(STORE.key) && read(STORE.model) ? "gemini" : "sim"
}

/** Who writes the comparison in in-browser mode; Gemini only if a key is saved and the user hasn't picked the local model. */
function savedJudge(): "self" | "gemini" {
  const j = read(STORE.judge)
  if (j === "self" || j === "gemini") return j
  return geminiFromStorage() ? "gemini" : "self"
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
    const gemini = geminiFromStorage()
    return localEngine(id, savedJudge() === "gemini" && gemini ? gemini : "self", onProgress)
  }
  progress = null
  return (kind === "gemini" && geminiFromStorage()) || simEngine
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
  output.classList.toggle("conf", local && ui.highlight === "doubt")
  output.classList.toggle("infl", local && ui.highlight === "values")
  output.classList.toggle("plainview", !local || ui.highlight === "off")
  paintTune()
  paintHint()
}

function bindSettings() {
  const dialog = $<HTMLDialogElement>("settings")
  const seg = $("engine-seg")
  const keyInput = $<HTMLInputElement>("key")
  const modelSel = $<HTMLSelectElement>("model")
  const localSel = $<HTMLSelectElement>("local-model")
  const modelsStatus = $("models-status")
  let kind: EngineKind = savedKind()

  const option = (m: (typeof LOCAL_MODELS)[number]) => `<option value="${m.id}">${m.name} · ${m.note}</option>`
  localSel.innerHTML =
    `<optgroup label="Safety-tuned (the makers' own chat models)">${LOCAL_MODELS.filter((m) => !m.steerable).map(option).join("")}</optgroup>` +
    `<optgroup label="Steerable (follow the system prompt, rarely refuse)">${LOCAL_MODELS.filter((m) => m.steerable).map(option).join("")}</optgroup>`
  $("webgpu-note").textContent = hasWebGPU()
    ? "Downloads once, then stays cached in this browser. It runs on this computer's graphics chip, and nothing is sent anywhere."
    : "This browser can't run local models (no WebGPU). Use a current Chrome or Edge."

  const notes: Record<EngineKind, string> = {
    sim: "Canned replies for the examples. No setup, and nothing is sent anywhere.",
    gemini: "Google's Gemini writes the replies and the comparison. Best writing.",
    local: "A small open model runs inside this browser and shows how sure it was of every word. Nothing you type leaves this computer unless you choose Gemini for the comparison.",
  }
  let judge: "self" | "gemini" = savedJudge()
  const judgeSeg = $("judge-seg")
  const paintJudge = () => {
    judgeSeg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === judge)))
    $("judge-note").textContent =
      judge === "self" ? "The same model reads both replies and writes the comparison. Free and private, but rougher." : "Gemini writes the comparison and highlights your prompt. Needs the key below."
  }
  judgeSeg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => ((judge = b.dataset.v as "self" | "gemini"), paintJudge(), paintKind())))
  const paintKind = () => {
    seg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.v === kind)))
    $("local-field").hidden = kind !== "local"
    $("judge-field").hidden = kind !== "local"
    const needsGemini = kind === "gemini" || (kind === "local" && judge === "gemini")
    $("key-field").hidden = !needsGemini
    $("model-field").hidden = !needsGemini
    $("engine-note").textContent = notes[kind]
  }
  seg.querySelectorAll<HTMLButtonElement>("button").forEach((b) => (b.onclick = () => ((kind = b.dataset.v as EngineKind), paintKind())))

  const open = () => {
    kind = savedKind()
    judge = savedJudge()
    paintJudge()
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
    $("lab-status").textContent = top.length ? `Side B now holds ${listOf(top.map((id) => nameOf(id)))}.` : "No priorities found. Paste the whole Markdown."
    if (!top.length) return
    ui.a = ui.a.filter((id) => !top.includes(id))
    ui.b = top
    saveUi()
    paintChips()
    renderPrompt()
  })
  dialog.addEventListener("close", () => {
    if (dialog.returnValue !== "save") return
    write(STORE.engine, kind)
    write(STORE.judge, judge)
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
