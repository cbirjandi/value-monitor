import data from "./data.json"

export type ValueId = "helpful" | "caution" | "candor" | "kindness" | "autonomy" | "privacy" | "evenhanded" | "humility"
export type Value = { id: ValueId; name: string; principle: string }
export type Scores = Record<ValueId, number>
export type Strength = "gentle" | "firm" | "strict"

export const VALUES = data.values as Value[]
export const V = Object.fromEntries(VALUES.map((v) => [v.id, v])) as Record<ValueId, Value>
export const VALUE_IDS = VALUES.map((v) => v.id)
export const MAX_CHOSEN = 4

/** One color per chosen slot, so whatever four values are picked always read apart. */
export const SLOT_COLORS = ["#e5532d", "#2f6fdb", "#12937a", "#8b55d6"]

export const colorOf = (chosen: ValueId[], id: ValueId) => SLOT_COLORS[chosen.indexOf(id)] ?? "currentColor"

/** The top priorities of a constitution copied from Constitution Lab. */
export function topValuesFromLab(markdown: string): ValueId[] {
  const byName = (name: string) => VALUES.find((v) => v.name.toLowerCase() === name.toLowerCase())?.id
  return [...markdown.matchAll(/^\s*\d+\.\s+\*\*(.+?)\.\*\*/gm)]
    .map((m) => byName(m[1]))
    .filter((id): id is ValueId => !!id)
    .slice(0, MAX_CHOSEN)
}
