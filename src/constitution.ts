import data from "./data.json"

export type ValueId = "helpful" | "caution" | "candor" | "kindness" | "autonomy" | "privacy" | "evenhanded" | "humility"
export type Value = { id: ValueId; name: string; principle: string }
export type Scores = Record<ValueId, number>

export const VALUES = data.values as Value[]
export const V = Object.fromEntries(VALUES.map((v) => [v.id, v])) as Record<ValueId, Value>
export const VALUE_IDS = VALUES.map((v) => v.id)

/** A conflict's rule: `target` is how far toward the winner the constitution wants a reply to land, from 0.5 (no default) to 1. */
export type Rule = { winner: ValueId | null; loser: ValueId | null; target: number; words: string; fromRanking?: boolean }
export type Constitution = { order: ValueId[]; rules: Map<string, Rule> }

const STRENGTH: Record<string, number> = { "almost always choose": 0.85, "usually choose": 0.7, "lean slightly toward": 0.6 }
const idOf = (name: string) => VALUES.find((v) => v.name.toLowerCase() === name.toLowerCase())?.id

export const pairKey = (x: ValueId, y: ValueId) => [x, y].sort().join("|")

/** Reads the Markdown that Constitution Lab's Copy Markdown button produces. */
export function parseConstitution(markdown: string): Constitution {
  const order = [...markdown.matchAll(/^\s*\d+\.\s+\*\*(.+?)\.\*\*/gm)].map((m) => idOf(m[1])).filter((id): id is ValueId => !!id)
  const rules = new Map<string, Rule>()
  for (const m of markdown.matchAll(/When ([\w-]+) and ([\w-]+) conflict, .*?should (almost always choose|usually choose|lean slightly toward) ([\w-]+)/g)) {
    const [x, y, w] = [idOf(m[1]), idOf(m[2]), idOf(m[4])]
    if (!x || !y || !w) continue
    rules.set(pairKey(x, y), { winner: w, loser: w === x ? y : x, target: STRENGTH[m[3]], words: m[3] })
  }
  for (const m of markdown.matchAll(/When ([\w-]+) and ([\w-]+) conflict, there is no default/g)) {
    const [x, y] = [idOf(m[1]), idOf(m[2])]
    if (x && y) rules.set(pairKey(x, y), { winner: null, loser: null, target: 0.5, words: "weigh the specific case" })
  }
  return { order, rules }
}

/** The rule for a conflict, falling back to the Part I ranking. */
export function ruleFor(c: Constitution, x: ValueId, y: ValueId): Rule | undefined {
  const explicit = c.rules.get(pairKey(x, y))
  if (explicit) return explicit
  const [ix, iy] = [c.order.indexOf(x), c.order.indexOf(y)]
  if (ix < 0 || iy < 0) return
  const [winner, loser] = ix < iy ? [x, y] : [y, x]
  return { winner, loser, target: 0.6, words: "ranks above", fromRanking: true }
}

/** The constitution's winner first, so bars read the same way as the rule. */
export function oriented(c: Constitution, x: ValueId, y: ValueId): [ValueId, ValueId] {
  const r = ruleFor(c, x, y)
  return r?.winner && r.loser ? [r.winner, r.loser] : [x, y]
}

export function allConflicts(c: Constitution): [ValueId, ValueId][] {
  const out: [ValueId, ValueId][] = []
  VALUE_IDS.forEach((x, i) => VALUE_IDS.slice(i + 1).forEach((y) => out.push(oriented(c, x, y))))
  return out
}

/** How the constitution reads inside a system prompt. */
export function constitutionBrief(c: Constitution) {
  const lines = c.order.map((id, i) => `${i + 1}. ${V[id].name}: ${V[id].principle}`)
  const rules = [...c.rules.values()].filter((r) => r.winner && r.loser).map((r) => `- When ${V[r.winner!].name} and ${V[r.loser!].name} conflict, ${r.words} ${V[r.winner!].name.toLowerCase()}.`)
  return ["Follow this constitution. Priorities, highest first:", ...lines, ...(rules.length ? ["Rules for specific conflicts:", ...rules] : [])].join("\n")
}

export const SAMPLE_CONSTITUTION = `# The Constitution of the assistant

## Part I: Priorities
1. **Candor.** Tell the truth plainly, including unwelcome truths.
2. **Caution.** Avoid responses that could lead to harm.
3. **Privacy.** Protect information about people who aren't part of the conversation.
4. **Autonomy.** Treat people as capable adults who make their own decisions.
5. **Helpfulness.** Give people what they actually asked for.
6. **Humility.** Be explicit about what is uncertain.
7. **Kindness.** Deliver difficult things with warmth and tact.
8. **Even-handedness.** On contested questions, lay out each view fairly.

## Part II: Rules for hard cases
- When Candor and Kindness conflict, the assistant should usually choose candor.
- When Helpfulness and Humility conflict, the assistant should almost always choose humility.
- When Privacy and Helpfulness conflict, the assistant should almost always choose privacy.
- When Autonomy and Caution conflict, there is no default. The assistant should weigh the specific case.
`
