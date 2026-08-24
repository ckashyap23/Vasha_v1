# BN ↔ EN Translation & Scoring UI

A small web UI that drives your real Kaggle notebooks:
- Upload an Excel file → picks a model → runs `bn_en_bulk_translate.ipynb` on Kaggle → download translated result.
- Upload a translated Excel file → runs `bn_en_score_translations.ipynb` on Kaggle → download scored result.

**Important: this is not instant.** Every request triggers a real Kaggle
kernel run — GPU cold start, model download/load, then inference. Expect
several minutes per job, same as running the notebook by hand. The UI polls
and shows progress; it doesn't make Kaggle itself faster.

## How it works

1. You upload a file in the browser.
2. The backend saves it, then:
   - Creates a new private Kaggle **Dataset** from your file (via `kaggle datasets create`).
   - Pushes a new version of the target **Kaggle notebook** (`kaggle kernels push`), attached to that dataset, along with a small `run_config.json` telling the notebook which file/run-label to use.
   - Polls `kaggle kernels status` until the run finishes.
   - Downloads the notebook's Output via `kaggle kernels output`.
3. The backend serves the resulting `.xlsx` back to you for download.

The two notebooks in `backend/notebooks/` are your originals with one
addition: at the top, they check for a `run_config.json` file (written by
the backend) and use it to auto-set `RUN_LABEL`/`INPUT_FILE` if present —
otherwise they fall back to the manual values you'd edit by hand. This means
**they still work standalone** in Kaggle's UI exactly as before.

## Prerequisites

- A Kaggle account with API access: go to kaggle.com → Account → **Create New API Token** → downloads `kaggle.json` containing your username and key.
- The two notebooks already existing on Kaggle under your account, with the **exact slugs** you configure below (a slug is the last part of the notebook's URL).
- Docker (for deployment) or Python 3.11 (to run locally without Docker).

## Configuration (environment variables)

| Variable | Required | Description |
|---|---|---|
| `KAGGLE_USERNAME` | yes | Your Kaggle username |
| `KAGGLE_KEY` | yes | Your Kaggle API key (from kaggle.json) |
| `TRANSLATE_KERNEL_SLUG` | no | Slug of your bulk-translate notebook (default: `bn-en-bulk-translate`) |
| `SCORE_KERNEL_SLUG` | no | Slug of your scoring notebook (default: `bn-en-score-translations`) |
| `TRANSLATE_NOTEBOOK_PATH` | no | Local path to the translate `.ipynb` (default: `./notebooks/bn_en_bulk_translate.ipynb`) |
| `SCORE_NOTEBOOK_PATH` | no | Local path to the scoring `.ipynb` (default: `./notebooks/bn_en_score_translations.ipynb`) |

The `kaggle` CLI reads `KAGGLE_USERNAME`/`KAGGLE_KEY` from the environment
automatically — no need to mount a `kaggle.json` file if you set these.

**Your notebook's own secrets (HF_TOKEN, OPENAI_API_KEY, etc.) stay exactly
where they are** — attached via Kaggle's Secrets UI on your Kaggle account,
same as before. This backend does not touch those; it only triggers runs.

## Run locally (no Docker)

```bash
cd backend
pip install -r requirements.txt
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
export TRANSLATE_KERNEL_SLUG=your-actual-translate-notebook-slug
export SCORE_KERNEL_SLUG=your-actual-score-notebook-slug
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Run with Docker (local test before deploying)

```bash
docker build -t bn-translate-ui .
docker run -p 8000:8000 \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_key \
  -e TRANSLATE_KERNEL_SLUG=your-actual-translate-notebook-slug \
  -e SCORE_KERNEL_SLUG=your-actual-score-notebook-slug \
  bn-translate-ui
```

Open `http://localhost:8000`.

## Deploying so others can access it

Any Docker-friendly host works. Two straightforward, low-effort options:

### Option A: Render.com
1. Push this whole `bn_translate_ui/` folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo.
3. Render auto-detects the `Dockerfile`.
4. Under **Environment**, add `KAGGLE_USERNAME`, `KAGGLE_KEY`, `TRANSLATE_KERNEL_SLUG`, `SCORE_KERNEL_SLUG` as secret env vars (never commit `KAGGLE_KEY` to the repo).
5. Deploy. Render gives you a public URL.

### Option B: Railway.app
Same idea — connect the GitHub repo, Railway detects the Dockerfile, add the same env vars under Variables, deploy.

Both have free tiers sufficient for light personal use. Note: since jobs take
several minutes and this backend uses a simple in-memory job store, avoid
scaling to multiple replicas (job status would split across instances) —
stick to a single instance unless you migrate `JOBS` to Redis/a database.

## Adding the fine-tuned model later

In `backend/main.py`, add the new model id to `SUPPORTED_MODELS`. In your
Kaggle translate notebook, extend the model-loading cell to branch on
`MODEL_CHOICE` (already passed through via `run_config.json` as
`_cfg.get("MODEL_CHOICE")`) — e.g. load the base model + your LoRA adapter
when `MODEL_CHOICE == "qwen_finetuned_v1"`.

## Known limitations

- **No queueing**: two simultaneous jobs will both try to run — fine for
  personal use, but add a simple queue (e.g. a lock or a task queue) if
  multiple people use this at once.
- **In-memory job store**: restarting the backend loses in-flight job status
  (though the underlying Kaggle run itself keeps going — you'd just lose the
  UI's ability to track it).
- **Kaggle GPU quota**: this backend doesn't manage your weekly 30-hour GPU
  quota — if you run out, jobs will fail with a Kaggle-side error, surfaced
  as the job's `error` field.
