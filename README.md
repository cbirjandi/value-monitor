# Value Monitor

Give two AI replies different values (say, Privacy against Security) and see how far the values move what gets written, side by side or steered between them as the reply is written.

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
* **In this browser**: a small open-weight model runs on your computer's graphics chip through WebGPU, with no key and no server. Because the model runs locally, the app sees how likely every word was: shaded words are where it hesitated, and clicking a word shows the alternatives it almost chose. Picking one rewrites the reply from that word. After a reply with your values is written, the app asks the same model, word by word, what it would have said with no values; each word is classed from the two distributions at that point: colored if the values changed the model's first choice, underlined if they made the word at least twice as likely, and left plain if the model just sampled a lower-ranked word by chance. Clicking a word shows both distributions side by side. Add a Gemini key too and Gemini scores the values. Works in current Chrome and Edge; the model downloads once (400 MB to 1 GB) and stays cached.

## Where the models come from

* **In this browser**: WebLLM's prebuilt builds, downloaded from the `mlc-ai` organization on Hugging Face (weights) and the `binary-mlc-llm-libs` GitHub repo (compiled GPU code), then cached by the browser. Two groups are offered:
  * *Safety-tuned*: the makers' own chat models (Qwen, Llama, Gemma, Phi, Mistral). Their trained-in values often win over the values you pick, which is itself worth showing.
  * *Steerable*: Nous Research's Hermes models, fine-tuned to follow the system prompt rather than refuse on moral grounds, so the values you pick steer them much further.
* **Gemini**: Google's API, called directly from the browser with your key.

In in-browser mode, Settings chooses who scores the sentences and writes the comparison: **this in-browser model** (free and private; it rates each sentence on your chosen values by reading the probabilities of a one-letter answer, A to E, the way a decision model does) or **Gemini** (Gemini Decision 1 for calibrated scores on every value, plus prompt highlights).

## Tune

The **Tune** button opens three settings:

* **Highlight words by**: *Value influence* shades every word by how many times likelier your values made it (log scale, deepest at 16×), with words whose first choice changed in purple text; *Uncertainty* shades where the model hesitated; or nothing.
* **Temperature** (0.1 to 1.5): how freely the model samples. Lower is more predictable and repeatable.
* **Value amplification** (in-browser models): classifier-free guidance. At each word the model's probabilities with values (pv) and without (p0) are combined as log pv + γ·(log pv − log p0), so whatever the values change gets exaggerated. Each word takes two model calls, so it writes more slowly.

## How it works

* **Two sides.** Tap a value once for side A, twice for side B, again to clear. A side with no values is the plain baseline. Changing one side rewrites only that side.
* **Personalities.** Under *Personalities* are dispositions you can give either side instead of (or as well as) values: Yes-man, Devil's advocate, Staunch denier, Wishy-washy, Alarmist, Cheerleader, Cynic and Bureaucrat.
* **Side by side.** Each side writes its own reply. With an in-browser model, every token is compared with the other side's values: the reply's own tokens are fed back through the model under the other side's prompt, and a hook in the model's worker records, at each step, the exact probability of the written token and the top five. Both sides' numbers come from that same computation. Tokens are shaded by how many times likelier this side's values made them. Under *Tune → Measure influence*, choose **While writing** (purple fills in every 24 tokens; writing runs at roughly half speed) or **After writing** (full-speed writing, then one pass per side). A second model reads both replies and lists how B differs from A.

### What the purple does and doesn't show

Purple is a *conditional* comparison: given exactly the text written so far, how likely was this token under each side's system prompt. It does not by itself show that the values caused the reply's overall direction, and a single run mixes value effects with sampling chance. For study use:

* Set a **seed** (Tune) to make a run repeatable on the same model and settings, and run several seeds per condition.
* **Export this run** (below the replies) saves the build, model, settings, both system prompts, and every token with its id and both sides' probabilities, as JSON.
* Compare response-level outcomes across runs (for the examples, the first-word verdict) alongside the token view.
* Pieces that aren't exactly one token (or that the display cleanup changed) aren't scored. Value amplification is experimental: it blends estimates from each side's top five, so it shows rankings, not probabilities.
* **Steer** (after the Spaceship IDE's future steering and real-time generation). One reply, with a cursor that rides along as it's written. The slider blends A's and B's values for everything after the cursor: let go and the future is rewritten at once, even mid-sentence. Drag the cursor back to steer from earlier, or click it at the end to keep writing. Each stretch is underlined in the blend color that wrote it, with a dot where the steering changed.
* **Tidy.** Replies are cleaned as they stream (no list numbers, markdown or line breaks).

## Limits

* **Values are given by instruction.** Word probabilities are the in-browser model's own; nothing here reads attention or weights. The Colab notebook (`value_monitor.ipynb`) reads and steers an open-weight model's activations directly.
* **Model names change.** The app asks Google which models your key can use, so it keeps working as new ones ship.
