import { WebWorkerMLCEngineHandler, type LogitProcessor } from "@mlc-ai/web-llm"

/**
 * Runs the in-browser model off the main thread, so the page stays smooth while it writes.
 *
 * It also installs a logit processor, which sees the model's raw scores at every step, for one job: checking a whole reply
 * against the other side's values in a single pass instead of one model call per word.
 * - While writing, it reports each token the model picks ("tok"), so the page knows the reply's token ids.
 * - In force mode, it replays given token ids under another prompt: at each step it records the probabilities of the top
 *   five tokens and of the forced one (softmax at the given temperature, as WebLLM computes its logprobs), then forces that
 *   token. One prefill plus one cheap step per token replaces a full prefill per word.
 */

type Force = { ids: number[]; i: number; temperature: number; out: { p: number; top: [number, number][] }[] }

const handler = new WebWorkerMLCEngineHandler()
let force: Force | null = null
let reporting = false

const processor: LogitProcessor = {
  processLogits(logits) {
    if (!force || force.i >= force.ids.length) return logits
    const target = force.ids[force.i]
    const t = force.temperature || 1
    let max = -Infinity
    for (let j = 0; j < logits.length; j++) if (logits[j] > max) max = logits[j]
    let sum = 0
    const top: [number, number][] = []
    for (let j = 0; j < logits.length; j++) {
      const e = Math.exp((logits[j] - max) / t)
      sum += e
      if (top.length < 5 || e > top[top.length - 1][1]) {
        top.push([j, e])
        top.sort((a, b) => b[1] - a[1])
        if (top.length > 5) top.pop()
      }
    }
    const pTarget = Math.exp((logits[target] - max) / t) / sum
    force.out.push({ p: pTarget, top: top.map(([id, e]) => [id, e / sum]) })
    logits.fill(-Infinity)
    logits[target] = 0
    return logits
  },
  processSampledToken(token) {
    if (force) force.i++
    else if (reporting) self.postMessage({ vm: "tok", id: token })
  },
  resetState() {},
}

// The registry is looked up by model id at load time; every model gets the same processor.
handler.setLogitProcessorRegistry(new (class extends Map<string, LogitProcessor> {
  get() {
    return processor
  }
  has() {
    return true
  }
})())

function decode(id: number): string {
  const pipelines = (handler.engine as unknown as { loadedModelIdToPipeline?: Map<string, { tokenizer: { decode(ids: Int32Array): string } }> }).loadedModelIdToPipeline
  const pipeline = pipelines ? [...pipelines.values()][0] : undefined
  return pipeline ? pipeline.tokenizer.decode(Int32Array.of(id)) : ""
}

self.onmessage = (event: MessageEvent) => {
  const msg = event.data
  if (msg && typeof msg === "object" && "vm" in msg) {
    if (msg.vm === "report") reporting = !!msg.on
    else if (msg.vm === "force") force = { ids: msg.ids, i: 0, temperature: msg.temperature, out: [] }
    else if (msg.vm === "collect") {
      const out = (force?.out ?? []).map((s, k) => ({ id: force!.ids[k], p: s.p, top: s.top.map(([id, p]) => ({ text: decode(id), p })) }))
      force = null
      self.postMessage({ vm: "collected", nonce: msg.nonce, out })
    }
    return
  }
  handler.onmessage(event)
}
