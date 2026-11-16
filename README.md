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

## How it works

| Step | What happens | File |
| --- | --- | --- |
| Choose | Tap up to four values. Their order is their priority. Gentle, Firm or Strict sets how hard the instruction pushes. | `src/main.ts` |
| Write | Two replies stream side by side: one with no values, one told to hold yours. The no-values reply is kept per question, so changing values only rewrites one side. | `src/engine.ts` |
| Score | As soon as a couple of sentences are complete, a second call rates them on every value from -2 to 2, so the rail fills in while the reply is still being written. | `src/column.ts` |
| Highlight | One more call picks the phrases in your question that pull on each value, like the Spaceship IDE's focus view. | `src/engine.ts` |
| Show | Rail bars per sentence, "gives up" pins, and a summary with the change versus the no-values reply. | `src/column.ts`, `src/style.css` |

Settings can import the top four priorities from a Constitution Lab constitution.

Sim and Gemini share one `Engine` type (`write`, `score`, `drivers`), so adding another provider means writing one more engine.

## Limits

* **Free-tier rate limits.** A first run makes about six requests; changing values makes about three. Short rate limits are waited out automatically.
* **This is behavioral, not internal.** Guidance is by instruction, and scores and highlights are another model call's judgment, not the model's attention or weights. Hosted APIs like Gemini don't expose those. The Colab notebook (`value_monitor.ipynb`) reads and steers an open-weight model's activations directly.
* **Model names change.** The app asks Google which models your key can use, so it keeps working as new ones ship.
