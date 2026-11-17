import { WebWorkerMLCEngineHandler, type LogitProcessor } from "@mlc-ai/web-llm"

/**
 * Runs the in-browser model off the main thread, so the page stays smooth while it writes.
 *
 * It also installs a logit processor, which sees the model's raw scores at every step. Both measurements the page compares
 * come from here, computed the same way (softmax of the raw scores at the request's temperature, before top-p), so the
 * two sides' probabilities are directly comparable:
 * - Report mode, while writing: for each token the model picks, its id, exact probability and the top five.
 * - Force mode, replaying: given token ids and another prompt, the same numbers for each id, then that id is forced.
 *
 * Both modes belong to one request, named by a nonce. Every request ends its mode with "end" (also when it fails or is
 * interrupted), and an "end" for an older request can't touch a newer one, so a failed replay never leaks into the next reply.
 */

type Top = [id: number, p: number][]
type Step = { id: number; p: number; top: Top }
/** The distribution from the last processLogits call, kept until the sampled token is known. */
type Dist = { logits: Float32Array; max: number; sum: number; t: number; top: Top }
type Mode =
  | { kind: "report"; nonce: string; temperature: number; steps: Step[]; pending?: Dist }
  | { kind: "force"; nonce: string; temperature: number; steps: Step[]; ids: number[]; i: number }

const handler = new WebWorkerMLCEngineHandler()
let mode: Mode | null = null

/** One pass over the vocabulary: the max, the softmax denominator and the top five, at temperature `t`. */
function distribution(logits: Float32Array, t: number): Dist {
  let max = -Infinity
  for (let j = 0; j < logits.length; j++) if (logits[j] > max) max = logits[j]
  let sum = 0
  const top: Top = []
  for (let j = 0; j < logits.length; j++) {
    const e = Math.exp((logits[j] - max) / t)
    sum += e
    if (top.length < 5 || e > top[top.length - 1][1]) {
      top.push([j, e])
      top.sort((a, b) => b[1] - a[1])
      if (top.length > 5) top.pop()
    }
  }
  return { logits, max, sum, t, top: top.map(([id, e]) => [id, e / sum]) }
}
const probability = (d: Dist, id: number) => Math.exp((d.logits[id] - d.max) / d.t) / d.sum

const processor: LogitProcessor = {
  processLogits(logits) {
    if (!mode) return logits
    if (mode.kind === "report") {
      mode.pending = distribution(logits, mode.temperature || 1)
      return logits
    }
    if (mode.i >= mode.ids.length) return logits
    const target = mode.ids[mode.i]
    const d = distribution(logits, mode.temperature || 1)
    mode.steps.push({ id: target, p: probability(d, target), top: d.top })
    logits.fill(-Infinity)
    logits[target] = 0
    return logits
  },
  processSampledToken(token) {
    if (!mode) return
    if (mode.kind === "force") return void mode.i++
    const d = mode.pending
    mode.pending = undefined
    if (d) mode.steps.push({ id: token, p: probability(d, token), top: d.top })
  },
  resetState() {},
}

// The registry is looked up by model id at load time; every model gets the same processor.
handler.setLogitProcessorRegistry(
  new (class extends Map<string, LogitProcessor> {
    get() {
      return processor
    }
    has() {
      return true
    }
  })(),
)

function decodeIds(ids: number[]): string {
  const pipelines = (handler.engine as unknown as { loadedModelIdToPipeline?: Map<string, { tokenizer: { decode(ids: Int32Array): string } }> }).loadedModelIdToPipeline
  const pipeline = pipelines ? [...pipelines.values()][0] : undefined
  return pipeline ? pipeline.tokenizer.decode(Int32Array.from(ids)) : ""
}
const decode = (id: number) => decodeIds([id])

/** The text each token adds, decoded in context so multi-byte characters and leading spaces come out right. */
function textsOf(ids: number[]): string[] {
  const out: string[] = []
  let before = ""
  for (let k = 0; k < ids.length; k++) {
    const now = decodeIds(ids.slice(0, k + 1))
    out.push(now.startsWith(before) ? now.slice(before.length) : decode(ids[k]))
    before = now
  }
  return out
}

self.onmessage = (event: MessageEvent) => {
  const msg = event.data
  if (!(msg && typeof msg === "object" && "vm" in msg)) return handler.onmessage(event)
  if (msg.vm === "report") mode = { kind: "report", nonce: msg.nonce, temperature: msg.temperature, steps: [] }
  else if (msg.vm === "force") mode = { kind: "force", nonce: msg.nonce, temperature: msg.temperature, steps: [], ids: msg.ids, i: 0 }
  else if (msg.vm === "end") {
    // Hands back what this request measured and ends its mode; an older request's "end" gets nothing and changes nothing.
    const mine = mode?.nonce === msg.nonce ? mode : null
    if (mine) mode = null
    const steps = mine?.steps ?? []
    const texts = textsOf(steps.map((s) => s.id))
    self.postMessage({
      vm: "ended",
      nonce: msg.nonce,
      steps: steps.map((s, k) => ({ id: s.id, text: texts[k], p: s.p, top: s.top.map(([id, p]) => ({ id, text: decode(id), p })) })),
    })
  }
}
