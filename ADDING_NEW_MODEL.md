# Adding a New Translation Model

## Overview

Models are wired in three places: `main.py` (backend validation), the `MODEL_MAP` in the notebook (model loading), and the `translate_batch` function (inference). Each model family has its own inference pattern.

---

## Step 1 — Add the model key to `backend/main.py`

Open [backend/main.py](backend/main.py) and add your key to `SUPPORTED_MODELS`:

```python
SUPPORTED_MODELS = [
    "Qwen2.5_3B",
    "Indictrans2_1B",
    "FacebookNLLB-200_600M",
    "YourModelKey",          # ← add here
]
```

The key is what appears in the UI dropdown and is sent as `MODEL_CHOICE` in `run_config.json`.

---

## Step 2 — Add the model to `MODEL_MAP` in the notebook

Open `backend/notebooks/bn_en_bulk_translate.ipynb`, cell 4 (the config cell). Add an entry to `MODEL_MAP`:

```python
MODEL_MAP = {
    "Qwen2.5_3B":            ("qwen",  "Qwen/Qwen2.5-3B-Instruct"),
    "Indictrans2_1B":        ("it2",   "ai4bharat/indictrans2-indic-en-1B"),
    "FacebookNLLB-200_600M": ("nllb",  "facebook/nllb-200-distilled-600M"),
    "YourModelKey":          ("myfam", "org/your-hf-model-id"),  # ← add here
}
```

Pick a short `model_family` string (e.g. `"myfam"`). This is the discriminator used in the loading and inference cells.

---

## Step 3 — Add model loading in cell 5 (Model loading)

In the `if/elif` block in the model loading cell, add a branch for your family:

```python
elif MODEL_FAMILY == "myfam":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
```

Common patterns:

| Model type | Class to load |
|---|---|
| Causal LM (GPT-style) | `AutoModelForCausalLM` |
| Seq2seq (T5/BART-style) | `AutoModelForSeq2SeqLM` |
| Pipeline wrapper | `pipeline("translation", model=...)` |

---

## Step 4 — Add inference in `translate_batch` (cell 6)

Add an `elif` branch in `translate_batch`:

```python
elif MODEL_FAMILY == "myfam":
    for i in range(0, len(bengali_texts), batch_size):
        batch = bengali_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        outputs_ids = model.generate(**inputs, max_new_tokens=256)
        outputs.extend(
            tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        )
        print(f"  translated {i + len(batch)}/{len(bengali_texts)}")
```

---

## Step 5 — Install any extra dependencies

If the model needs extra packages (e.g. `sentencepiece`, `IndicTransToolkit`), add them to the pip install cell (cell 3):

```python
get_ipython().system(
    "pip install -q transformers accelerate sentencepiece your-extra-pkg"
)
```

---

## Step 6 — Handle gated models

If the model requires a HuggingFace account agreement (e.g. Llama):

1. Accept the license at `huggingface.co/your-org/your-model`
2. Go to the Kaggle notebook → **Add-ons → Secrets**, check `HF_TOKEN`, click **Save Version**
3. Make the auth cell strict for this model (remove the `except` fallback, or check `HF_TOKEN is not None` before loading)

---

## Step 7 — Commit and push

```bash
git add backend/main.py backend/notebooks/bn_en_bulk_translate.ipynb
git commit -m "feat: add YourModelKey model"
git push
```

Render will redeploy automatically. The new model appears in the UI dropdown immediately after deploy.

---

## GPU memory reference

| Model size | Quantisation | Min VRAM |
|---|---|---|
| ≤3B | 4-bit NF4 | 4 GB (T4) |
| 7B | 4-bit NF4 | 6 GB (T4) |
| 1B seq2seq | fp32 | 4 GB |
| 600M seq2seq | fp32 | 2 GB |

Kaggle T4 has 16 GB VRAM — any model ≤7B in 4-bit fits comfortably.
