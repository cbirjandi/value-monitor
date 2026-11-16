import { ruleFor, VALUE_IDS, type Constitution, type Rule, type Scores, type ValueId } from "./constitution.ts"

export type Sentence = { text: string; start: number; end: number }
export type Flag = Rule & { margin: number }

/** Sentences with their positions in `text`; a trailing fragment still being streamed counts as one. */
export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = []
  for (const m of text.matchAll(/[^.!?\n]+(?:[.!?]+["”’')\]]*|$)/g)) {
    if (!m[0].trim()) continue
    const lead = m[0].length - m[0].trimStart().length
    out.push({ text: m[0].trim(), start: m.index! + lead, end: m.index! + m[0].trimEnd().length })
  }
  return out
}

/**
 * A sentence breaks a rule when it favors the rule's loser over its winner by at least 2 points on the -2..2 scale,
 * and actually leans toward the loser. Rules that only come from the Part I ranking need 3 points.
 */
export function flagOf(c: Constitution, s: Scores): Flag | undefined {
  let worst: Flag | undefined
  VALUE_IDS.forEach((x, i) =>
    VALUE_IDS.slice(i + 1).forEach((y) => {
      const r = ruleFor(c, x, y)
      if (!r?.winner || !r.loser) return
      const margin = s[r.loser] - s[r.winner]
      if (s[r.loser] > 0 && margin >= (r.fromRanking ? 3 : 2) && (!worst || margin > worst.margin)) worst = { ...r, margin }
    }),
  )
  return worst
}

export function meanScores(rows: Scores[]): Scores {
  return Object.fromEntries(VALUE_IDS.map((id) => [id, rows.length ? rows.reduce((t, r) => t + r[id], 0) / rows.length : 0])) as Scores
}

/** Where a reply sits between x (1) and y (0), from its sentences' average lean. */
export function replyPosition(rows: Scores[], x: ValueId, y: ValueId) {
  if (!rows.length) return 0.5
  const lean = rows.reduce((t, r) => t + (r[x] - r[y]), 0) / rows.length
  return Math.max(0.02, Math.min(0.98, 0.5 + lean / 6))
}
