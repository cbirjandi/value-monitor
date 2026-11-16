import { flagOf, meanScores, replyPosition, splitSentences, type Sentence } from "./analysis.ts"
import { ruleFor, V, VALUE_IDS, type Constitution, type Scores, type ValueId } from "./constitution.ts"
import type { Engine } from "./engine.ts"

export type PanelState = {
  title: string
  prompt: string
  text: string
  sentences: Sentence[]
  scores: Scores[] | null
  streaming: boolean
  scoring: boolean
  error?: string
}

export type View = { x: ValueId; y: ValueId; constitution: Constitution }

export function emptyPanel(title: string, prompt: string): PanelState {
  return { title, prompt, text: "", sentences: [], scores: null, streaming: true, scoring: false }
}

/** Streams `source` into `state`, then scores its sentences. `onChange` runs after every update. */
export async function fillPanel(state: PanelState, engine: Engine, source: AsyncIterable<string> | string, signal: AbortSignal, onChange: () => void) {
  try {
    if (typeof source === "string") state.text = source
    else
      for await (const chunk of source) {
        state.text += chunk
        state.sentences = splitSentences(state.text)
        onChange()
      }
    state.streaming = false
    state.sentences = splitSentences(state.text)
    if (!state.sentences.length) throw new Error("The model returned an empty reply. Try again.")
    state.scoring = true
    onChange()
    state.scores = await engine.score(state.prompt, state.sentences.map((s) => s.text), signal)
  } catch (error) {
    if (signal.aborted) return
    state.error = error instanceof Error ? error.message : String(error)
  } finally {
    if (!signal.aborted) {
      state.streaming = false
      state.scoring = false
      onChange()
    }
  }
}

export function renderPanel(el: HTMLElement, s: PanelState, view: View) {
  const { x, y, constitution } = view
  const rows = s.sentences.map((sentence, i) => {
    const sc = s.scores?.[i]
    const flag = sc ? flagOf(constitution, sc) : undefined
    const lean = sc ? sc[x] - sc[y] : 0
    const rail = sc
      ? `${meter(V[x].name, sc[x], "x")}${meter(V[y].name, sc[y], "y")}`
      : `<div class="pending" aria-label="not scored yet">${s.scoring ? "scoring" : s.streaming ? "writing" : ""}</div>`
    const note = flag
      ? `<p class="flag">Breaks a rule: ${V[flag.loser!].name} over ${V[flag.winner!].name}. Your constitution ${
          flag.fromRanking ? `ranks ${V[flag.winner!].name} above ${V[flag.loser!].name}` : `says to ${flag.words} ${V[flag.winner!].name.toLowerCase()}`
        }.</p>`
      : ""
    const tip = sc ? VALUE_IDS.filter((id) => sc[id]).map((id) => `${V[id].name} ${sc[id] > 0 ? "+" : ""}${sc[id]}`).join(", ") || "no clear values" : ""
    return `<div class="srow${flag ? " bad" : ""}"><div class="rail">${rail}</div><div><p class="sent" style="--lean:${lean}" data-lean="${Math.sign(lean)}" title="${esc(tip)}">${esc(sentence.text)}</p>${note}</div></div>`
  })

  const all = s.scores ?? []
  const rule = ruleFor(constitution, x, y)
  const target = !rule ? null : !rule.winner ? 0.5 : rule.winner === x ? rule.target : 1 - rule.target
  const pos = replyPosition(all, x, y)
  const gap = target === null || !all.length ? "" : `<span>gap: ${Math.round(Math.abs(pos - target) * 100)} points toward ${V[pos > target ? x : y].name}</span>`
  const mean = meanScores(all)
  const flags = all.filter((sc) => flagOf(constitution, sc)).length

  el.innerHTML = `
    <header class="phead"><h2>${esc(s.title)}</h2><span class="pstate">${
      s.error ? "" : s.streaming ? "writing…" : s.scoring ? "scoring…" : `${s.sentences.length} sentences · ${flags} rule break${flags === 1 ? "" : "s"}`
    }</span></header>
    ${s.error ? `<p class="err">${esc(s.error)}</p>` : ""}
    <div class="sentences">${rows.join("") || (s.streaming ? `<p class="waiting">Waiting for the first words…</p>` : "")}</div>
    <div class="tug${all.length ? "" : " dim"}">
      <div class="ends"><span>${V[x].name}</span><span class="y">${V[y].name}</span></div>
      <div class="track"><span class="line"></span>
        ${target === null ? "" : `<span class="mk target" style="left:${(1 - target) * 100}%" title="constitution target"></span>`}
        ${all.length ? `<span class="mk reply" style="left:${(1 - pos) * 100}%" title="this reply"></span>` : ""}
      </div>
      <div class="legend">${target === null ? "" : `<span><i class="dot target"></i>constitution target</span>`}<span><i class="dot reply"></i>this reply</span>${gap}</div>
    </div>
    <div class="profile${all.length ? "" : " dim"}">${VALUE_IDS.map((id) => `<div>${meter(V[id].name, mean[id], "x", true)}</div>`).join("")}</div>`
}

/** A bar diverging from center: right of center upholds the value, left gives it up. */
function meter(name: string, score: number, side: "x" | "y", full = false) {
  const w = (Math.min(Math.abs(score), 2) / 2) * 50
  const left = score >= 0 ? 50 : 50 - w
  const label = full ? name : name.slice(0, 4)
  return `<span class="meter ${side}"><span class="mlabel" title="${esc(name)}">${esc(label)}</span><span class="bar"><i style="left:${left}%;width:${w}%"></i></span></span>`
}

function esc(t: string) {
  return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}
