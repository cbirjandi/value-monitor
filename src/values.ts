import data from "./data.json"

export type ValueId = string
export type Value = { id: ValueId; name: string; principle: string; custom?: boolean; persona?: boolean }
/** A score per value, from -2 (gives it up) to 2 (upholds it). */
export type Scores = Record<ValueId, number>
export type Strength = "gentle" | "firm" | "strict"

export const MAX_CHOSEN = 3
const CUSTOM_STORE = "vm-custom-values"

const BUILT_IN = data.values as Value[]
/**
 * Personalities: dispositions rather than values, steered the same way. Switched off for now; set to true to bring back
 * their chips and their two examples (definitions are kept in data.json).
 */
export const SHOW_PERSONAS = false
export const PERSONAS = SHOW_PERSONAS ? (data.personas as Value[]) : []
const PERSONA_IDS = new Set((data.personas as Value[]).map((v) => v.id))
/** Whether an example uses only values that are switched on. */
export const usable = (ids: ValueId[]) => SHOW_PERSONAS || !ids.some((id) => PERSONA_IDS.has(id))
let custom: Value[] = loadCustom()

export const allValues = () => [...BUILT_IN, ...custom, ...PERSONAS]
const fallback = (id: ValueId): Value => ({ id, name: id, principle: "" })
export const valueOf = (id: ValueId) => allValues().find((v) => v.id === id) ?? fallback(id)
export const nameOf = (id: ValueId) => valueOf(id).name

/** Adds a value of the user's own; it is kept in this browser. Returns its id. */
export function addCustomValue(name: string, meaning: string): ValueId {
  const clean = name.trim().replace(/\s+/g, " ").slice(0, 32)
  const existing = allValues().find((v) => v.name.toLowerCase() === clean.toLowerCase())
  if (existing) return existing.id
  const id = `custom-${clean.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${Date.now().toString(36).slice(-4)}`
  const principle = meaning.trim() ? meaning.trim().replace(/([^.!?])$/, "$1.") : `Uphold ${clean.toLowerCase()} in every answer.`
  custom = [...custom, { id, name: clean, principle, custom: true }]
  saveCustom()
  return id
}

export function removeCustomValue(id: ValueId) {
  custom = custom.filter((v) => v.id !== id)
  saveCustom()
}

/** Reply A's values are orange and reply B's blue, everywhere they appear. */
export const SIDE_COLORS = { a: "#e5532d", b: "#2f6fdb" }
export const MAX_PER_SIDE = 3

/** The top priorities of a constitution copied from Constitution Lab, matched by name. */
export function topValuesFromLab(markdown: string): ValueId[] {
  const byName = (name: string) => allValues().find((v) => v.name.toLowerCase() === name.toLowerCase())?.id
  return [...markdown.matchAll(/^\s*\d+\.\s+\*\*(.+?)\.\*\*/gm)]
    .map((m) => byName(m[1]))
    .filter((id): id is ValueId => !!id)
    .slice(0, MAX_CHOSEN)
}

function loadCustom(): Value[] {
  try {
    const list = JSON.parse(localStorage.getItem(CUSTOM_STORE) || "[]")
    return Array.isArray(list) ? list.filter((v) => v?.id && v?.name) : []
  } catch {
    return []
  }
}

function saveCustom() {
  try {
    localStorage.setItem(CUSTOM_STORE, JSON.stringify(custom))
  } catch {}
}
