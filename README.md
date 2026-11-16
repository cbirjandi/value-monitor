# Value Monitor

Pick the values an AI should hold, then watch each sentence of its reply honor or give them up, next to a reply written with no values at all.

Everything runs in the browser. Like the Spaceship IDE, there is no server: the page calls the Gemini API directly with your own key, which stays in your browser's local storage.

## Run it on your Mac

```bash
npm install
npm run dev
```

Open the address it prints (usually http://localhost:5173). With no key you're in **sim mode**, which replays canned replies for the example prompts. For real replies:

1. Get a free key at https://aistudio.google.com/apikey
2. Click **Settings**, paste the key, press **Load my models**, keep the suggested Flash model, and save.

## Put it on GitHub Pages

1. Create a GitHub repo and push this folder to its `main` branch.
2. In the repo, open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. The workflow in `.github/workflows/deploy.yml` builds and publishes on every push. Your site appears at `https://<you>.github.io/<repo>/`.

Visitors bring their own key. Never put your key in the code: anyone could read it from the page. If you want visitors to use your key, put a small proxy in front of Gemini (a Cloudflare Worker, for example) that holds the key and rate-limits requests.

## Three ways to run it

Open **Settings** (the gear) and choose who writes the replies:

* **Sim**: canned replies for the examples. No setup.
* **Gemini**: Google's model writes and scores. Needs a free API key.
* **In this browser**: a small open-weight model (Qwen2.5 0.5B or 1.5B) runs on your computer's graphics chip through WebGPU, with no key and no server. Because the model runs locally, the app sees how likely every word was: shaded words are where it hesitated, and clicking a word shows the alternatives it almost chose. Picking one rewrites the reply from that word. After a reply with your values is written, the app asks the same model, word by word, what it would have said with no values; words where its first choice was different are colored, so you can see exactly which words your values changed. Add a Gemini key too and Gemini scores the values. Works in current Chrome and Edge; the model downloads once (400 MB to 1 GB) and stays cached.

## How it works

| Step | What happens | File |
| --- | --- | --- |
| Choose | Tap up to four of twelve values (Candor, Helpfulness, Caution, Compassion, Autonomy, Privacy, Fairness, Rule of law, Transparency, Neutrality, Humility, Public interest), or add your own. Their order is their priority. Gentle, Firm or Strict sets how hard the instruction pushes. | `src/values.ts`, `src/main.ts` |
| Tidy | Replies are cleaned as they stream (no list numbers, markdown or line breaks). | `src/engine.ts` |
| Scrub | Like the Spaceship IDE, an orange cursor rides along as the reply is written. Drag it back and the text after it fades; the summary follows the cursor. Change values while it's pulled back and the reply is kept up to the cursor and rewritten from there. | `src/column.ts`, `src/main.ts` |
| Write | Two replies stream side by side: one with no values, one told to hold yours. The no-values reply is kept per question, so changing values only rewrites one side. | `src/engine.ts` |
| Score | As soon as a couple of sentences are complete, a second call rates them on every value from -2 to 2, so the rail fills in while the reply is still being written. | `src/column.ts` |
| Highlight | One more call picks the phrases in your question that pull on each value, like the Spaceship IDE's focus view. | `src/engine.ts` |
| Show | Rail bars per sentence, "gives up" pins, and a summary with the change versus the no-values reply. | `src/column.ts`, `src/style.css` |

Settings can import the top four priorities from a Constitution Lab constitution.

Sim, Gemini and the in-browser model (`src/local.ts`, running in a Web Worker via WebLLM) share one `Engine` type, so adding another provider means writing one more engine.

## Limits

* **Free-tier rate limits.** A first run makes about six requests; changing values makes about three. Short rate limits are waited out automatically.
* **This is behavioral, not internal.** Guidance is by instruction, and scores and highlights are another model call's judgment, not the model's attention or weights. Hosted APIs like Gemini don't expose those. The Colab notebook (`value_monitor.ipynb`) reads and steers an open-weight model's activations directly.
* **Model names change.** The app asks Google which models your key can use, so it keeps working as new ones ship.
