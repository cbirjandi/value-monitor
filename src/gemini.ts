const BASE = "https://generativelanguage.googleapis.com/v1beta"

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly retryAfterS?: number,
  ) {
    super(message)
  }
}

export type GeminiRequest = {
  key: string
  model: string
  system?: string
  user: string
  temperature?: number
  schema?: object
  signal: AbortSignal
}

/** Models that refused `thinkingBudget: 0`, so later calls skip it. */
const thinkingLocked = new Set<string>()

/** Models this key can call for text, without image, audio, embedding or live variants. */
export async function listModels(key: string, signal?: AbortSignal) {
  const res = await fetch(`${BASE}/models?pageSize=200`, { headers: { "x-goog-api-key": key }, signal })
  if (!res.ok) throw await errorOf(res)
  const body = (await res.json()) as { models?: { name: string; supportedGenerationMethods?: string[] }[] }
  return (body.models ?? [])
    .filter((m) => m.supportedGenerationMethods?.includes("generateContent"))
    .map((m) => m.name.replace(/^models\//, ""))
    .filter((name) => name.startsWith("gemini") && !/image|tts|audio|embed|live|vision|computer|robotics|native/.test(name))
}

/** The fastest stable text model available: flash-lite, then flash, then anything. */
export function defaultModel(models: string[]) {
  const stable = models.filter((m) => !/preview|exp/.test(m))
  const rank = (m: string) => (/flash-lite/.test(m) ? 0 : /flash/.test(m) ? 1 : 2)
  return [...(stable.length ? stable : models)].sort((a, b) => rank(a) - rank(b) || b.localeCompare(a, undefined, { numeric: true }))[0]
}

export async function* streamText(req: GeminiRequest): AsyncIterable<string> {
  const res = await post(req, `${req.model}:streamGenerateContent?alt=sse`)
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += value
    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""
    for (const line of lines) {
      if (!line.startsWith("data:")) continue
      const text = textOf(JSON.parse(line.slice(5)))
      if (text) yield text
    }
  }
}

export async function generateJson<T>(req: GeminiRequest): Promise<T> {
  const res = await post(req, `${req.model}:generateContent`)
  const text = textOf(await res.json())
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError(0, "The scorer's answer wasn't valid JSON. Try again.")
  }
}

async function post(req: GeminiRequest, path: string): Promise<Response> {
  const quiet = !thinkingLocked.has(req.model)
  const res = await fetch(`${BASE}/models/${path}`, {
    method: "POST",
    signal: req.signal,
    headers: { "Content-Type": "application/json", "x-goog-api-key": req.key },
    body: JSON.stringify(bodyOf(req, quiet)),
  })
  if (res.ok) return res
  const error = await errorOf(res)
  if (quiet && res.status === 400 && /thinking/i.test(error.message)) {
    thinkingLocked.add(req.model)
    return post(req, path)
  }
  throw error
}

/** Thinking is turned off where the model allows it: it adds seconds of delay, and these tasks don't need it. */
function bodyOf(req: GeminiRequest, quiet: boolean) {
  return {
    ...(req.system ? { systemInstruction: { parts: [{ text: req.system }] } } : {}),
    contents: [{ role: "user", parts: [{ text: req.user }] }],
    generationConfig: {
      temperature: req.temperature ?? 0.7,
      ...(req.schema ? { responseMimeType: "application/json", responseSchema: req.schema } : {}),
      ...(quiet ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
    },
  }
}

type Part = { text?: string; thought?: boolean }
function textOf(body: { candidates?: { content?: { parts?: Part[] } }[] }) {
  return (body.candidates?.[0]?.content?.parts ?? [])
    .filter((p) => !p.thought)
    .map((p) => p.text ?? "")
    .join("")
}

async function errorOf(res: Response) {
  let message = `${res.status} ${res.statusText}`
  let retry: number | undefined
  try {
    const body = await res.json()
    message = body.error?.message ?? message
    const delay = body.error?.details?.find((d: { retryDelay?: string }) => d.retryDelay)?.retryDelay
    if (delay) retry = parseFloat(delay)
  } catch {}
  if (res.status === 429) message = `Gemini's free-tier rate limit was hit${retry ? `; try again in ${Math.ceil(retry)} s` : "; wait a minute and try again"}.`
  if (res.status === 400 && /api key/i.test(message)) message = "Gemini rejected the API key. Check it in Settings."
  if (res.status === 403) message = "This key can't use that model. Pick another in Settings."
  return new ApiError(res.status, message, retry)
}

/** A rubric question for Gemini Decision 1: `criteria[i]` describes level i. */
export type ScoreQuestion = { type: "score"; instructions: string; criteria: string[] }
export type ScoreAnswer = { type: "score"; score: number; probabilities?: Record<string, number>; legend?: Record<string, string>; confidence?: number }

/**
 * Asks Gemini Decision 1 many closed questions in one pass. It doesn't write text; it returns a calibrated probability for
 * every level of every question. https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/
 */
export async function decide(key: string, input: string, questions: Record<string, ScoreQuestion>, signal: AbortSignal) {
  const res = await fetch(`${BASE}/decisions`, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({ model: "gemini-decision-1", input, questions }),
  })
  if (!res.ok) throw await errorOf(res)
  const body = (await res.json()) as { answers?: Record<string, ScoreAnswer> }
  return body.answers ?? {}
}
