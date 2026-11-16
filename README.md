# Value Monitor

Steer a language model between two values and watch, sentence by sentence, which values its reply actually serves, checked against a constitution made in Constitution Lab.

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
| Steer | The slider becomes an instruction in the system prompt ("lean clearly toward candor over kindness"). "Follow my constitution" adds your ranked priorities and rules. | `src/engine.ts` |
| Write | The reply streams in from `streamGenerateContent`. With "Show the unsteered reply too", a plain reply is written alongside and kept for that prompt. | `src/gemini.ts`, `src/panel.ts` |
| Score | One more call rates every sentence on all eight values from -2 to 2, using a JSON schema so the answer is always structured. | `src/engine.ts` |
| Check | A sentence breaks a rule when it favors the rule's loser over its winner by 2 or more (3 for rules that only come from the ranking). | `src/analysis.ts` |
| Show | Value rail, sentence tint, rule-break flags, a tug-of-war bar (constitution target vs. this reply) and an eight-value profile. | `src/panel.ts` |
| Constitution | Paste Constitution Lab's **Copy Markdown** output in **Constitution**. | `src/constitution.ts` |

Sim and Gemini share one `Engine` type (`write` and `score`), so adding another provider means writing one more engine. Groq, Cerebras and OpenRouter use an OpenAI-style API and also accept requests straight from the browser.

## Limits

* **Free-tier rate limits.** Each Generate uses two to four requests (one or two replies, each scored). If you see a rate-limit message, wait for the time it gives.
* **This is behavioral, not internal.** Steering is by instruction and scoring is another model's judgment, so scores can be wrong and steering can be ignored. The Colab notebook (`value_monitor.ipynb`) reads and steers an open-weight model's activations directly.
* **Model names change.** The app asks Google which models your key can use, so it keeps working as new ones ship.
