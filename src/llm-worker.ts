import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm"

// Runs the in-browser model off the main thread, so the page stays smooth while it writes.
const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (event: MessageEvent) => handler.onmessage(event)
