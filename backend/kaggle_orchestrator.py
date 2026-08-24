"""
Orchestrates a full Kaggle notebook run driven from this backend:
  1. Push the uploaded file as a new version of a private Kaggle Dataset.
  2. Push a new version of the target notebook (kernel), configured to read
     from that dataset, with any run-specific parameters (e.g. RUN_LABEL,
     which model) baked into kernel-metadata via environment variables.
  3. Poll kaggle kernels status until the run finishes.
  4. Download the notebook's Output files and return the path to the
     result file the caller asked for.

Requires the `kaggle` CLI to be installed and authenticated (KAGGLE_USERNAME
+ KAGGLE_KEY as env vars, or a kaggle.json in ~/.kaggle/), same credentials
you'd use locally to run `kaggle kernels ...` by hand.
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

WORKDIR = Path(os.environ.get("ORCHESTRATOR_WORKDIR", "/tmp/bn_translate_ui"))
WORKDIR.mkdir(parents=True, exist_ok=True)

def _get_kaggle_username() -> str:
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise RuntimeError("KAGGLE_USERNAME environment variable is not set.")
    return username

# Configure these to match your actual Kaggle notebook slugs.
# A "slug" is the last part of the notebook URL, e.g.
# kaggle.com/code/ckashyap/bn-en-bulk-translate -> "bn-en-bulk-translate"
TRANSLATE_KERNEL_SLUG = os.environ.get("TRANSLATE_KERNEL_SLUG", "bn-en-bulk-translate")
SCORE_KERNEL_SLUG = os.environ.get("SCORE_KERNEL_SLUG", "bn-en-score-translations")

POLL_INTERVAL_SECONDS = 5  # check every 5s so we don't miss the brief queued/running window
MAX_POLL_MINUTES = 60  # translation/scoring runs can be slow; 1000-row runs take ~35 min


class KaggleRunError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Optional[Path] = None) -> str:
    # Force UTF-8 in the Kaggle CLI subprocess so it can read notebooks with
    # non-ASCII characters (arrows, em dashes, etc.) on Windows.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", env=env
    )
    if result.returncode != 0:
        raise KaggleRunError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _dataset_slug_for_job(job_id: str) -> str:
    # Kaggle dataset slugs must be lowercase, alnum + dashes
    return f"bn-ui-upload-{job_id}"[:50]


def upload_input_dataset(local_file_path: Path, job_id: str, extra_files: Optional[dict] = None) -> str:
    """
    Creates a new private Kaggle Dataset from the uploaded file.
    extra_files: optional dict of {filename: text_content} to bundle into the dataset.
    Returns the dataset's full ref, e.g. "ckashyap/bn-ui-upload-abc123".
    """
    slug = _dataset_slug_for_job(job_id)
    dataset_dir = WORKDIR / f"dataset_{job_id}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy the uploaded file into the dataset staging folder
    dest_file = dataset_dir / local_file_path.name
    dest_file.write_bytes(local_file_path.read_bytes())

    if extra_files:
        for filename, content in extra_files.items():
            (dataset_dir / filename).write_text(content)

    metadata = {
        "title": slug,
        "id": f"{_get_kaggle_username()}/{slug}",
        "licenses": [{"name": "unknown"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    _run(["kaggle", "datasets", "create", "-p", str(dataset_dir), "--dir-mode", "zip"])

    return f"{_get_kaggle_username()}/{slug}"


def _kernel_metadata_path(kernel_slug: str, job_id: str) -> Path:
    return WORKDIR / f"kernel_{kernel_slug}_{job_id}"


def _cleared_notebook(notebook_local_path: Path, dest: Path) -> None:
    """Write notebook to dest with all cell outputs stripped.

    Kaggle skips re-execution if the pushed notebook already has outputs
    embedded. Clearing them forces a fresh run.
    """
    nb = json.loads(notebook_local_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    dest.write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")


def push_and_run_kernel(
    kernel_slug: str,
    notebook_local_path: Path,
    dataset_ref: str,
    env_overrides: dict,
) -> str:
    """
    Pushes a new version of the given kernel, attached to dataset_ref as an
    input, with env_overrides injected. Returns the full kernel ref, e.g.
    "ckashyap/bn-en-bulk-translate".
    """
    job_dir = _kernel_metadata_path(kernel_slug, str(uuid.uuid4())[:8])
    job_dir.mkdir(parents=True, exist_ok=True)

    notebook_dest = job_dir / notebook_local_path.name
    _cleared_notebook(notebook_local_path, notebook_dest)

    # run_config.json is bundled into the dataset (not here) so the notebook
    # can find it at /kaggle/input/<dataset-slug>/run_config.json.

    metadata = {
        "id": f"{_get_kaggle_username()}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": notebook_dest.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

    _run(["kaggle", "kernels", "push", "-p", str(job_dir)])

    return f"{_get_kaggle_username()}/{kernel_slug}"


def wait_for_kernel(kernel_ref: str) -> None:
    """Polls kernel status until it finishes. Raises on error or timeout."""
    deadline = time.time() + MAX_POLL_MINUTES * 60

    # Phase 1: wait until Kaggle transitions OUT of "complete" (the old run's
    # status) into "queued" or "running" for our new push. Without this,
    # the very first poll sees "complete" from the previous run and returns
    # immediately, downloading stale output.
    # Give Kaggle a few seconds to register the new version before first poll.
    time.sleep(10)
    start_deadline = time.time() + 5 * 60  # up to 5 min for queuing
    seen_active = False
    while time.time() < start_deadline:
        status_output = _run(["kaggle", "kernels", "status", kernel_ref])
        status_lower = status_output.lower()
        if "queued" in status_lower or "running" in status_lower:
            seen_active = True
            break
        if "error" in status_lower or "failed" in status_lower:
            raise KaggleRunError(f"Kernel failed before starting: {status_output}")
        time.sleep(POLL_INTERVAL_SECONDS)

    if not seen_active:
        raise KaggleRunError(
            f"Kernel {kernel_ref!r} never entered queued/running state within 5 minutes. "
            "Check that the kernel push succeeded and Kaggle is scheduling it."
        )

    # Phase 2: now wait for completion of the active run.
    while time.time() < deadline:
        status_output = _run(["kaggle", "kernels", "status", kernel_ref])
        status_lower = status_output.lower()
        if "complete" in status_lower:
            return
        if "error" in status_lower or "failed" in status_lower:
            raise KaggleRunError(f"Kernel run failed: {status_output}")
        time.sleep(POLL_INTERVAL_SECONDS)

    raise KaggleRunError(f"Kernel run timed out after {MAX_POLL_MINUTES} minutes: {kernel_ref}")


def download_kernel_output(kernel_ref: str, job_id: str) -> Path:
    """Downloads the kernel's Output files, returns the local output directory."""
    output_dir = WORKDIR / f"output_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)])
    return output_dir


def find_output_file(output_dir: Path, suffix: str) -> Optional[Path]:
    """Finds the most recently modified file in output_dir matching a suffix."""
    candidates = sorted(
        output_dir.glob(f"*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _count_rows(path: Path) -> int:
    """Count rows in an xlsx or parquet file to pass as MAX_ROWS safeguard."""
    try:
        import pandas as _pd
        if path.suffix.lower() in (".xlsx", ".xls"):
            # Try openpyxl first (.xlsx); fall back to xlrd for old binary .xls
            try:
                return len(_pd.read_excel(path, engine="openpyxl"))
            except Exception:
                return len(_pd.read_excel(path, engine="xlrd"))
        return len(_pd.read_parquet(path))
    except Exception:
        return 0  # if counting fails, let the notebook decide


def run_translate_job(local_input_path: Path, run_label: str, model_choice: str, job_id: str) -> Path:
    """
    Full pipeline: upload input -> push translate notebook -> wait -> pull xlsx output.
    model_choice is passed through as an env override the notebook can read
    to decide which model to load (only "qwen_base" supported for now).
    """
    slug = _dataset_slug_for_job(job_id)
    row_count = _count_rows(local_input_path)
    run_config = json.dumps({
        "INPUT_FILE_DATASET": slug,
        "RUN_LABEL": run_label,
        "MODEL_CHOICE": model_choice,
        "MAX_ROWS": row_count,
    }, indent=2)
    dataset_ref = upload_input_dataset(local_input_path, job_id, extra_files={"run_config.json": run_config})

    notebook_path = Path(os.environ.get(
        "TRANSLATE_NOTEBOOK_PATH", "./notebooks/bn_en_bulk_translate.ipynb"
    ))
    kernel_ref = push_and_run_kernel(
        TRANSLATE_KERNEL_SLUG,
        notebook_path,
        dataset_ref,
        env_overrides={},
    )

    wait_for_kernel(kernel_ref)
    output_dir = download_kernel_output(kernel_ref, job_id)

    result_file = find_output_file(output_dir, ".xlsx")
    if result_file is None:
        raise KaggleRunError(f"No .xlsx output found in {output_dir}")
    return result_file


def run_score_job(local_input_path: Path, run_label: str, job_id: str) -> Path:
    """Full pipeline for the scoring notebook."""
    slug = _dataset_slug_for_job(job_id)
    row_count = _count_rows(local_input_path)
    run_config = json.dumps({
        "INPUT_FILE_DATASET": slug,
        "RUN_LABEL": run_label,
        "MAX_ROWS": row_count,
    }, indent=2)
    dataset_ref = upload_input_dataset(local_input_path, job_id, extra_files={"run_config.json": run_config})

    notebook_path = Path(os.environ.get(
        "SCORE_NOTEBOOK_PATH", "./notebooks/bn_en_score_translations.ipynb"
    ))
    kernel_ref = push_and_run_kernel(
        SCORE_KERNEL_SLUG,
        notebook_path,
        dataset_ref,
        env_overrides={},
    )

    wait_for_kernel(kernel_ref)
    output_dir = download_kernel_output(kernel_ref, job_id)

    result_file = find_output_file(output_dir, "scored" + f"__{run_label}.xlsx")
    if result_file is None:
        result_file = find_output_file(output_dir, ".xlsx")
    if result_file is None:
        raise KaggleRunError(f"No scored .xlsx output found in {output_dir}")
    return result_file
